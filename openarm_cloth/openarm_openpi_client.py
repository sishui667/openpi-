# -*- coding: utf-8 -*-
"""Run the local OpenArm MuJoCo scene with a remote OpenPI policy.

The DGX server runs the official ``pi0_aloha_towel`` policy and exposes the
OpenPI websocket protocol.  This script stays on the local machine:

* captures the four ALOHA-style camera observations from the MuJoCo scene;
* converts the 7-DoF + 2-finger OpenArm state to the 14-D ALOHA state;
* sends observations to the DGX policy server in a background thread;
* applies the returned 14-D action through the existing OpenArm IK/PD loop.

The adapter is intentionally conservative.  The ALOHA policy has six arm
joints plus one gripper per side, while OpenArm has seven arm joints plus two
finger slides.  The OpenArm seventh joint is held at its current target, and
the two finger slides receive the same gripper target.  ``--apply`` is
required before remote actions can move the simulated arms; without it the
script is a connectivity/observation preview only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any

import glfw
import msgpack
import mujoco
import mujoco.viewer
import numpy as np
from websockets.sync.client import connect

from teleop_towel_dataset import (
    ARM_Q_NAMES,
    EE_BODY_NAMES,
    FINGER_Q_NAMES,
    ArmSpec,
    apply_controller,
    build_arm_specs,
    build_finger_actuators,
    enable_robot_cloth_contact,
    id_for,
    latest_scene,
    reset_to_ready,
    safety_reason,
)


LOGGER = logging.getLogger("openarm_openpi_client")
ROOT = Path(__file__).resolve().parent


def pack_array(obj: Any) -> Any:
    """Match openpi-client's safe NumPy/msgpack representation."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class WebsocketPolicy:
    """Small local copy of the OpenPI websocket client dependency."""

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._ws = connect(uri, compression=None, max_size=None, open_timeout=timeout_s)
        self._packer = msgpack.Packer(default=pack_array)
        self.metadata = msgpack.unpackb(self._ws.recv(), object_hook=unpack_array)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"DGX OpenPI server error: {response}")
        return msgpack.unpackb(response, object_hook=unpack_array)

    def close(self) -> None:
        self._ws.close()


class PolicyWorker:
    """Keep the MuJoCo loop responsive while DGX inference is running."""

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._requests: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_actions: np.ndarray | None = None
        self._generation = 0
        self._busy = False
        self._connected = threading.Event()
        self.last_error: str | None = None
        self.metadata: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name="openpi-policy", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        policy: WebsocketPolicy | None = None
        try:
            policy = WebsocketPolicy(self._host, self._port, self._timeout_s)
            with self._lock:
                self.metadata = dict(policy.metadata)
            self._connected.set()
            LOGGER.info("Connected to DGX OpenPI: %s", self.metadata)
            while not self._stop.is_set():
                try:
                    observation = self._requests.get(timeout=0.2)
                except queue.Empty:
                    continue
                if observation is None:
                    break
                with self._lock:
                    self._busy = True
                try:
                    result = policy.infer(observation)
                    actions = np.asarray(result["actions"], dtype=np.float64)
                    if actions.ndim != 2 or actions.shape[1] < 14:
                        raise ValueError(f"DGX returned unexpected action shape: {actions.shape}")
                    with self._lock:
                        self._latest_actions = actions[:, :14].copy()
                        self._generation += 1
                        self.last_error = None
                except Exception as exc:  # keep the viewer alive for diagnosis
                    with self._lock:
                        self.last_error = repr(exc)
                    LOGGER.exception("Remote inference failed")
                finally:
                    with self._lock:
                        self._busy = False
        except Exception as exc:
            with self._lock:
                self.last_error = repr(exc)
            LOGGER.exception("Could not connect to DGX OpenPI")
        finally:
            self._connected.set()
            if policy is not None:
                policy.close()

    def wait_connected(self, timeout_s: float) -> bool:
        return self._connected.wait(timeout_s)

    def submit(self, observation: dict[str, Any]) -> bool:
        with self._lock:
            if self.last_error and not self._thread.is_alive():
                return False
        try:
            self._requests.put_nowait(observation)
            return True
        except queue.Full:
            return False

    def latest(self) -> tuple[np.ndarray | None, int, bool, str | None]:
        with self._lock:
            actions = None if self._latest_actions is None else self._latest_actions.copy()
            return actions, self._generation, self._busy, self.last_error

    def close(self) -> None:
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)


@dataclass
class FingerSpec:
    qpos: np.ndarray


class OpenArmAlohaAdapter:
    """Convert between the local 18-D OpenArm layout and ALOHA's 14-D layout."""

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_specs: dict[str, ArmSpec],
        joint_signs: dict[str, np.ndarray],
        gripper_scale: float,
    ) -> None:
        self.arm_specs = arm_specs
        self.joint_signs = joint_signs
        self.gripper_scale = float(gripper_scale)
        self.fingers: dict[str, FingerSpec] = {}
        for side in ("left", "right"):
            qpos = []
            for name in FINGER_Q_NAMES[side]:
                jid = id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos.append(int(model.jnt_qposadr[jid]))
            self.fingers[side] = FingerSpec(np.asarray(qpos, dtype=np.int32))

    def _gripper_to_aloha(self, value_m: float) -> float:
        return float(np.clip(value_m / 0.044, 0.0, 1.0))

    def _gripper_from_aloha(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0) * 0.044 * self.gripper_scale)

    def observation_state(self, data: mujoco.MjData) -> np.ndarray:
        values: list[float] = []
        for side in ("left", "right"):
            spec = self.arm_specs[side]
            q = data.qpos[spec.qpos][:6] * self.joint_signs[side]
            finger = float(np.mean(data.qpos[self.fingers[side].qpos]))
            values.extend(q.tolist())
            values.append(self._gripper_to_aloha(finger))
        return np.asarray(values, dtype=np.float32)

    def action_to_targets(
        self,
        action: np.ndarray,
        data: mujoco.MjData,
        q_target: dict[str, np.ndarray],
        max_step_rad: float,
        gain: float,
    ) -> dict[str, float]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size < 14:
            raise ValueError(f"Expected at least 14 action values, got {action.shape}")
        grippers: dict[str, float] = {}
        for offset, side in ((0, "left"), (7, "right")):
            spec = self.arm_specs[side]
            desired = action[offset : offset + 6] * self.joint_signs[side]
            desired = np.clip(desired, spec.q_range[:6, 0], spec.q_range[:6, 1])
            current = q_target[side][:6]
            delta = gain * (desired - current)
            delta = np.clip(delta, -max_step_rad, max_step_rad)
            q_target[side][:6] = np.clip(
                current + delta, spec.q_range[:6, 0], spec.q_range[:6, 1]
            )
            # OpenPI/ALOHA has six arm joints.  Keep OpenArm's seventh wrist
            # joint at the safe target generated by the local scene.
            q_target[side][6] = np.clip(
                q_target[side][6], spec.q_range[6, 0], spec.q_range[6, 1]
            )
            grippers[side] = self._gripper_from_aloha(action[offset + 6])
        return grippers


@dataclass
class ViewerControls:
    """Small safety state controlled by MuJoCo viewer keyboard events."""

    allow_actions: bool
    actions_enabled: bool
    physics_paused: bool = False
    reset_requested: bool = False

    def on_key(self, keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            if not self.allow_actions:
                LOGGER.warning("Remote actions are disabled; restart with --apply before pressing Space")
                return
            self.actions_enabled = not self.actions_enabled
            LOGGER.warning("Remote actions %s", "ENABLED" if self.actions_enabled else "PAUSED")
        elif keycode == glfw.KEY_P:
            self.physics_paused = not self.physics_paused
            LOGGER.warning("MuJoCo physics %s", "PAUSED" if self.physics_paused else "RUNNING")
        elif keycode == glfw.KEY_R:
            self.reset_requested = True
            LOGGER.warning("Reset requested: towel_ready")


def parse_signs(text: str) -> np.ndarray:
    values = np.asarray([float(value.strip()) for value in text.split(",")], dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isin(values, (-1.0, 1.0))):
        raise argparse.ArgumentTypeError("joint signs must be six comma-separated values, each +1 or -1")
    return values


def camera_observation(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    del width, height  # renderer owns these dimensions
    rendered: dict[str, np.ndarray] = {}
    camera_map = {
        "cam_high": "chest_d435i_preview",
        "cam_low": "chest_d435i_preview",
        "cam_left_wrist": "v22_left_d405_rgb",
        "cam_right_wrist": "v22_right_d405_rgb",
    }
    for name, camera in camera_map.items():
        renderer.update_scene(data, camera=camera)
        rgb = np.asarray(renderer.render(), dtype=np.uint8)
        rendered[name] = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
    return rendered


def make_observation(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    adapter: OpenArmAlohaAdapter,
    width: int,
    height: int,
    prompt: str,
) -> dict[str, Any]:
    return {
        "state": adapter.observation_state(data),
        "images": camera_observation(renderer, data, width, height),
        "prompt": prompt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="100.122.120.76", help="DGX OpenPI host or ws:// URI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--prompt", default="fold the towel")
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--contact", action="store_true", help="enable runtime robot-towel contacts")
    parser.add_argument("--apply", action="store_true", help="allow remote actions to move the arms")
    parser.add_argument("--action-gain", type=float, default=0.25)
    parser.add_argument("--max-step-deg", type=float, default=3.0)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--left-signs", type=parse_signs, default=np.ones(6))
    parser.add_argument("--right-signs", type=parse_signs, default=np.ones(6))
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--steps", type=int, default=200, help="headless physics steps")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.control_hz <= 0 or args.policy_hz <= 0 or args.steps <= 0:
        raise ValueError("control/policy frequencies and steps must be positive")
    if not 0.0 < args.action_gain <= 1.0:
        raise ValueError("--action-gain must be in (0, 1]")

    scene = args.scene or latest_scene()
    scene = scene if scene.is_absolute() else (ROOT / scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready")
    reset_to_ready(model, data, key_id)
    if args.contact:
        enable_robot_cloth_contact(model)
        mujoco.mj_forward(model, data)

    arm_specs = build_arm_specs(model)
    finger_actuators = build_finger_actuators(model)
    adapter = OpenArmAlohaAdapter(
        model,
        arm_specs,
        {"left": args.left_signs, "right": args.right_signs},
        args.gripper_scale,
    )
    q_target = {side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()}
    gripper = {"left": 0.020, "right": 0.020}
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    worker = PolicyWorker(args.host, args.port, args.connect_timeout)
    if not worker.wait_connected(args.connect_timeout):
        worker.close()
        raise TimeoutError(f"Timed out waiting for DGX OpenPI at {args.host}:{args.port}")
    if worker.last_error and not worker.metadata:
        worker.close()
        raise RuntimeError(worker.last_error)

    LOGGER.info("Scene: %s", scene)
    LOGGER.info("DGX metadata: %s", worker.metadata)
    LOGGER.info("Mode: %s", "APPLY remote actions" if args.apply else "PREVIEW only")
    LOGGER.info("Adapter: OpenArm 7+2 per arm -> ALOHA 6+1 per arm; joint7 held")
    controls = ViewerControls(allow_actions=bool(args.apply), actions_enabled=bool(args.apply))
    LOGGER.info("Viewer keys: Space=remote action pause/resume, P=physics pause, R=ready reset")

    period = 1.0 / args.control_hz
    policy_period = 1.0 / args.policy_hz
    next_policy_request = 0.0
    last_generation = -1
    action_cursor = 0
    latest_action: np.ndarray | None = None
    latest_error_printed: str | None = None
    steps_per_tick = max(1, int(round(0.01 / float(model.opt.timestep))))
    wall_tick = time.perf_counter()

    def tick(now: float) -> None:
        nonlocal next_policy_request, last_generation, action_cursor, latest_action, gripper
        nonlocal latest_error_printed
        if controls.reset_requested:
            reset_to_ready(model, data, key_id)
            q_target.clear()
            q_target.update({side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()})
            gripper = {"left": 0.020, "right": 0.020}
            controls.reset_requested = False
        if now >= next_policy_request:
            observation = make_observation(
                renderer, data, adapter, args.image_width, args.image_height, args.prompt
            )
            if worker.submit(observation):
                next_policy_request = now + policy_period
            else:
                next_policy_request = now + min(0.1, policy_period)

        actions, generation, _busy, error = worker.latest()
        if error and error != latest_error_printed:
            LOGGER.error("DGX policy error: %s", error)
            latest_error_printed = error
        if actions is not None and generation != last_generation:
            latest_action = actions
            action_cursor = 0
            last_generation = generation

        if latest_action is not None and controls.actions_enabled and not controls.physics_paused:
            frame = latest_action[min(action_cursor, len(latest_action) - 1)]
            action_cursor += 1
            gripper = adapter.action_to_targets(
                frame,
                data,
                q_target,
                max_step_rad=np.deg2rad(args.max_step_deg),
                gain=args.action_gain,
            )

        if not controls.physics_paused:
            for _ in range(steps_per_tick):
                apply_controller(model, data, arm_specs, finger_actuators, q_target, gripper)
                mujoco.mj_step(model, data)
        reason = safety_reason(data, arm_specs)
        if reason is not None:
            LOGGER.error("Safety pause: %s", reason)
            for side, spec in arm_specs.items():
                q_target[side] = data.qpos[spec.qpos].copy()

    try:
        if args.headless:
            LOGGER.info("Headless test: %d control steps", args.steps)
            initial = data.qpos.copy()
            for _ in range(args.steps):
                tick(time.perf_counter())
                time.sleep(period)
            actions, generation, _busy, error = worker.latest()
            LOGGER.info(
                "Headless result: generation=%d action_shape=%s qpos_max_change=%.6f error=%s",
                generation,
                None if actions is None else tuple(actions.shape),
                float(np.max(np.abs(data.qpos - initial))),
                error,
            )
        else:
            with mujoco.viewer.launch_passive(model, data, key_callback=controls.on_key) as viewer:
                viewer.cam.lookat[:] = [0.45, 0.0, 0.38]
                viewer.cam.distance = 1.05
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -25
                viewer.sync()
                while viewer.is_running():
                    tick(time.perf_counter())
                    viewer.sync()
                    wall_tick += period
                    time.sleep(max(0.0, wall_tick - time.perf_counter()))
    finally:
        worker.close()
        renderer.close()
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
