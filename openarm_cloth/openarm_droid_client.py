# -*- coding: utf-8 -*-
"""Run one OpenArm in MuJoCo with the remote OpenPI DROID policy.

This client deliberately uses the official DROID observation/action contract:

* ``observation/exterior_image_1_left``: one external RGB camera;
* ``observation/wrist_image_left``: one wrist RGB camera;
* ``observation/joint_position``: seven joint positions;
* ``observation/gripper_position``: one normalized gripper position;
* returned actions: seven joint velocities plus one gripper position.

The local OpenArm scene still contains two arms.  Only the selected arm is
controlled by DROID; the other arm remains at its safe ready target.  The
velocity action is integrated into a joint-position target before the existing
gravity-compensated MuJoCo controller applies it.  Remote actions are disabled
unless ``--apply`` is supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import glfw
import msgpack
import mujoco
import mujoco.viewer
import numpy as np
from websockets.sync.client import connect

from build_droid_object_scene import ensure_scene
from teleop_towel_dataset import (
    ARM_Q_NAMES,
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


LOGGER = logging.getLogger("openarm_droid_client")
ROOT = Path(__file__).resolve().parent


class LoopState(str, Enum):
    INIT = "初始化"
    OBSERVE = "获取胸部/腕部观测"
    WAIT_POLICY = "等待DGX策略"
    EXECUTE = "执行动作"
    CHECK_GRASP = "检查抓取结果"
    RECOVER_CAMERA = "调整腕部相机"
    REPLAN = "重新规划"
    SUCCESS = "抓取成功"
    PAUSED = "暂停"
    SAFETY = "安全暂停"
    ERROR = "策略错误"


class LoopStateReporter:
    """Print one Chinese status line whenever the closed-loop state changes."""

    def __init__(self) -> None:
        self.current: LoopState | None = None

    def set(self, state: LoopState, detail: str = "") -> None:
        if state == self.current:
            return
        self.current = state
        suffix = f"：{detail}" if detail else ""
        LOGGER.info("闭环状态机 | %s%s", state.value, suffix)


def pack_array(obj: Any) -> Any:
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
    """Run blocking DGX inference away from the MuJoCo viewer thread."""

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
        self._thread = threading.Thread(target=self._run, name="openpi-droid-policy", daemon=True)
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
                    if actions.ndim != 2 or actions.shape[1] < 8:
                        raise ValueError(f"DGX returned unexpected DROID action shape: {actions.shape}")
                    with self._lock:
                        self._latest_actions = actions[:, :8].copy()
                        self._generation += 1
                        self.last_error = None
                except Exception as exc:  # keep the viewer alive for diagnostics
                    with self._lock:
                        self.last_error = repr(exc)
                    LOGGER.exception("Remote DROID inference failed")
                finally:
                    with self._lock:
                        self._busy = False
        except Exception as exc:
            with self._lock:
                self.last_error = repr(exc)
            LOGGER.exception("Could not connect to DGX DROID OpenPI server")
        finally:
            self._connected.set()
            if policy is not None:
                policy.close()

    def wait_connected(self, timeout_s: float) -> bool:
        return self._connected.wait(timeout_s)

    def submit(self, observation: dict[str, Any]) -> bool:
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


class OpenArmDroidAdapter:
    """Adapt one OpenArm (7+2) to the DROID (7+1) interface."""

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_specs: dict[str, ArmSpec],
        side: str,
        joint_signs: np.ndarray,
        velocity_scale: float,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.side = side
        self.spec = arm_specs[side]
        self.joint_signs = np.asarray(joint_signs, dtype=np.float64)
        self.velocity_scale = float(velocity_scale)
        if self.joint_signs.shape != (7,) or not np.all(np.isin(self.joint_signs, (-1.0, 1.0))):
            raise ValueError("DROID joint signs must contain seven values, each +1 or -1")
        qpos = []
        for name in FINGER_Q_NAMES[side]:
            jid = id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos.append(int(model.jnt_qposadr[jid]))
        self.finger_qpos = np.asarray(qpos, dtype=np.int32)

    def _gripper_to_droid(self, value_m: float) -> float:
        return float(np.clip(value_m / 0.044, 0.0, 1.0))

    def state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        q = data.qpos[self.spec.qpos].copy() * self.joint_signs
        finger = float(np.mean(data.qpos[self.finger_qpos]))
        gripper = np.asarray([self._gripper_to_droid(finger)], dtype=np.float32)
        return q.astype(np.float32), gripper

    def integrate_action(
        self,
        action: np.ndarray,
        q_target: np.ndarray,
        gripper_target: float,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size < 8:
            raise ValueError(f"Expected eight DROID action values, got {action.shape}")
        # Official DROID actions are normalized joint velocities.  Convert them
        # to local joint-position increments before the MuJoCo PD loop.
        velocity = np.clip(action[:7], -1.0, 1.0) * self.velocity_scale
        desired = q_target + self.joint_signs * velocity * dt
        desired = np.clip(desired, self.spec.q_range[:, 0], self.spec.q_range[:, 1])
        # DROID uses 0=closed and 1=open for the position gripper action.
        gripper = 0.044 if action[7] > 0.5 else 0.0
        return desired, gripper


@dataclass
class ViewerControls:
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
            LOGGER.warning("DROID actions %s", "ENABLED" if self.actions_enabled else "PAUSED")
        elif keycode == glfw.KEY_P:
            self.physics_paused = not self.physics_paused
            LOGGER.warning("MuJoCo physics %s", "PAUSED" if self.physics_paused else "RUNNING")
        elif keycode == glfw.KEY_R:
            self.reset_requested = True
            LOGGER.warning("Reset requested: towel_ready")


class CameraPreview:
    """Show the exact RGB observations in two independent Tk windows."""

    def __init__(self, enabled: bool, width: int, height: int) -> None:
        self.enabled = bool(enabled)
        self.closed = False
        self._tk = None
        self._image = None
        self._root = None
        self._windows: list[Any] = []
        self._labels: list[Any] = []
        self._photos: list[Any] = []
        if not self.enabled:
            return
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("--show-cameras requires tkinter and Pillow in the openpi environment") from exc
        self._tk = tk
        self._image = Image
        self._image_tk = ImageTk
        display_width = max(int(width), 320)
        display_height = max(int(height), 320)
        self._root = tk.Tk()
        self._root.withdraw()
        window_names = (
            "Chest camera | observation/exterior_image_1_left",
            "Wrist camera | observation/wrist_image_left",
        )
        for name in window_names:
            window = tk.Toplevel(self._root)
            window.title(name)
            window.protocol("WM_DELETE_WINDOW", self._on_close)
            window.bind("<Escape>", lambda _event: self._on_close())
            window.bind("q", lambda _event: self._on_close())
            window.geometry(f"{display_width}x{display_height}")
            label = tk.Label(window, bg="black")
            label.pack(fill=tk.BOTH, expand=True)
            self._windows.append(window)
            self._labels.append(label)
        self._root.update_idletasks()
        self._root.update()

    def _on_close(self) -> None:
        self.closed = True
        LOGGER.info("Camera preview closed by user")

    def show(self, exterior: np.ndarray, wrist: np.ndarray) -> None:
        if not self.enabled or self.closed or self._root is None:
            return
        try:
            display_size = (self._windows[0].winfo_width(), self._windows[0].winfo_height())
            display_size = (
                max(1, int(display_size[0])),
                max(1, int(display_size[1])),
            )
            self._photos = []
            # Show both views in the same orientation as the RGB observations
            # sent to DGX.
            preview_images = (exterior, wrist)
            for label, rgb in zip(self._labels, preview_images):
                image = self._image.fromarray(rgb)
                image = image.resize(display_size, self._image.Resampling.NEAREST)
                photo = self._image_tk.PhotoImage(image=image)
                label.configure(image=photo)
                label.image = photo
                self._photos.append(photo)
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self.closed = True

    def close(self) -> None:
        if self.enabled and self._root is not None:
            try:
                self._root.destroy()
            except self._tk.TclError:
                pass


@dataclass
class VisualEvidence:
    """Cheap RGB evidence used by the local closed-loop monitor.

    This is intentionally a safety/attention signal, not a replacement for
    the DROID policy.  The known-color test objects make it possible to detect
    whether the object is visible in the chest and wrist views without adding
    another neural-network dependency to the local simulator.
    """

    exterior_pixels: int
    wrist_pixels: int
    exterior_centroid: np.ndarray | None
    wrist_centroid: np.ndarray | None

    @property
    def wrist_visible(self) -> bool:
        return self.wrist_pixels > 0


def _target_mask(image: np.ndarray, object_name: str) -> np.ndarray:
    rgb = image.astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if object_name == "cube":
        # MuJoCo's shaded red material is often below 105/255.  Difference
        # thresholds are more stable than an absolute bright-red threshold.
        return (red > 35.0) & (red - green > 22.0) & (red - blue > 22.0)
    if object_name == "bottle":
        return (blue > 30.0) & (blue - red > 18.0) & (blue - green > 8.0)
    if object_name == "ball":
        return (red > 55.0) & (green > 35.0) & (red - blue > 18.0) & (green - blue > 5.0)
    # The original towel is blue.  Keep this heuristic deliberately broad so
    # it still provides a useful attention signal under lighting changes.
    return (blue > 35.0) & (green > 25.0) & (blue - red > 12.0)


def _mask_evidence(image: np.ndarray, object_name: str) -> tuple[int, np.ndarray | None]:
    mask = _target_mask(image, object_name)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, None
    height, width = image.shape[:2]
    centroid = np.asarray([float(xs.mean() / width), float(ys.mean() / height)], dtype=np.float32)
    return int(len(xs)), centroid


def visual_evidence(exterior: np.ndarray, wrist: np.ndarray, object_name: str) -> VisualEvidence:
    exterior_pixels, exterior_centroid = _mask_evidence(exterior, object_name)
    wrist_pixels, wrist_centroid = _mask_evidence(wrist, object_name)
    return VisualEvidence(exterior_pixels, wrist_pixels, exterior_centroid, wrist_centroid)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    return quaternion / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quaternion_slerp(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    """Small local slerp because this MuJoCo Python build lacks mju_slerpQuat."""

    q0 = _normalize_quaternion(first)
    q1 = _normalize_quaternion(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    amount = float(np.clip(amount, 0.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + amount * (q1 - q0))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    return _normalize_quaternion(
        (np.sin((1.0 - amount) * theta) / sin_theta) * q0
        + (np.sin(amount * theta) / sin_theta) * q1
    )


class WristCameraController:
    """Adjust only the selected wrist camera's orientation in the sim.

    The camera remains attached to its existing wrist body.  On a failed
    grasp, it is smoothly re-aimed at the object so the next DROID request has
    a clearer close-up, while the chest camera continues to provide global
    context.
    """

    def __init__(self, model: mujoco.MjModel, camera_name: str) -> None:
        self.model = model
        self.camera_id = id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.body_id = int(model.cam_bodyid[self.camera_id])
        self.nominal_quat = model.cam_quat[self.camera_id].copy()

    def aim_at(self, data: mujoco.MjData, target_position: np.ndarray, blend: float) -> bool:
        if self.body_id < 0:
            return False
        camera_position = np.asarray(data.cam_xpos[self.camera_id], dtype=np.float64)
        forward = np.asarray(target_position, dtype=np.float64) - camera_position
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-6:
            return False
        forward /= forward_norm
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, world_up))) > 0.95:
            world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up = np.cross(right, forward)
        # MuJoCo cameras look along their local -Z axis.
        world_rotation = np.column_stack((right, up, -forward))
        body_rotation = np.asarray(data.xmat[self.body_id], dtype=np.float64).reshape(3, 3)
        local_rotation = body_rotation.T @ world_rotation
        desired_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(desired_quat, local_rotation.reshape(-1))
        model_quat = self.model.cam_quat[self.camera_id].copy()
        self.model.cam_quat[self.camera_id] = _quaternion_slerp(model_quat, desired_quat, blend)
        mujoco.mj_forward(self.model, data)
        return True

    def restore_nominal(self, data: mujoco.MjData) -> None:
        self.model.cam_quat[self.camera_id] = self.nominal_quat
        mujoco.mj_forward(self.model, data)


def target_world_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    """Return the rigid test object's position or the towel mesh centroid."""

    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "droid_object")
    if object_body >= 0:
        return data.xpos[object_body].copy()
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "folding_towel_v25_4")
    if flex_id >= 0 and model.flex_vertadr[flex_id] >= 0 and model.flex_vertnum[flex_id] > 0:
        start = int(model.flex_vertadr[flex_id])
        count = int(model.flex_vertnum[flex_id])
        return np.mean(data.flexvert_xpos[start : start + count], axis=0)
    return None


class GraspMonitor:
    """Simulation-side confirmation used to decide whether recovery is needed."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, adapter: OpenArmDroidAdapter) -> None:
        self.model = model
        self.object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "droid_object")
        self.object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "droid_object_geom")
        self.initial_z = float(data.xpos[self.object_body][2]) if self.object_body >= 0 else None
        self.initial_flex_position = (
            target_world_position(model, data) if self.object_body < 0 else None
        )
        self.hand_body = int(adapter.spec.body_id)
        self.finger_geoms = {
            geom_id
            for geom_id in range(model.ngeom)
            if "finger_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
            and model.geom_bodyid[geom_id] == self.hand_body
        }

    def _object_finger_contact(self, data: mujoco.MjData) -> bool:
        if self.object_geom < 0 or not self.finger_geoms:
            return False
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.object_geom in pair and pair.intersection(self.finger_geoms):
                return True
        return False

    def is_grasped(self, data: mujoco.MjData, gripper_closed: bool) -> bool | None:
        # For a deformable towel, use its mesh centroid as a conservative
        # local-sim proxy: it must rise and remain near the selected hand.
        if self.object_body < 0:
            if self.initial_flex_position is None:
                return None
            current_position = target_world_position(self.model, data)
            if current_position is None:
                return None
            lifted = float(current_position[2]) > float(self.initial_flex_position[2]) + 0.012
            near_hand = float(np.linalg.norm(current_position - data.xpos[self.hand_body])) < 0.24
            return bool(lifted and near_hand and gripper_closed)
        if self.initial_z is None:
            return None
        object_position = data.xpos[self.object_body]
        hand_position = data.xpos[self.hand_body]
        lifted = float(object_position[2]) > self.initial_z + 0.025
        near_hand = float(np.linalg.norm(object_position - hand_position)) < 0.18
        return bool(lifted and near_hand and (gripper_closed or self._object_finger_contact(data)))


def parse_signs(text: str) -> np.ndarray:
    values = np.asarray([float(value.strip()) for value in text.split(",")], dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isin(values, (-1.0, 1.0))):
        raise argparse.ArgumentTypeError("joint signs must be seven comma-separated values, each +1 or -1")
    return values


def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return np.ascontiguousarray(renderer.render(), dtype=np.uint8)


def format_exterior_image(image: np.ndarray, horizontal_flip: bool) -> np.ndarray:
    """Match the chest camera's image-left/right convention to the robot view."""

    if not horizontal_flip:
        return image
    return np.ascontiguousarray(np.fliplr(image), dtype=np.uint8)


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 MuJoCo world rotation to XYZ roll-pitch-yaw radians.

    The returned convention is the usual fixed-axis XYZ RPY representation,
    equivalent to ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if sy > 1e-8:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        # Gimbal-lock fallback: keep yaw at zero and retain a continuous roll.
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def end_effector_xyzrpy(
    data: mujoco.MjData,
    spec: ArmSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the selected TCP pose from the current MuJoCo joint state.

    ``spec.body_id`` is the OpenArm TCP body selected in ``build_arm_specs``.
    Position is returned in metres and RPY is returned in degrees for readable
    terminal diagnostics.
    """

    xyz = np.asarray(data.xpos[spec.body_id], dtype=np.float64).copy()
    rotation = np.asarray(data.xmat[spec.body_id], dtype=np.float64).reshape(3, 3)
    rpy_deg = np.degrees(rotation_matrix_to_rpy(rotation))
    return xyz, rpy_deg


def format_vector(values: np.ndarray, precision: int = 3) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(value):+.{precision}f}" for value in values) + "]"


def add_world_axes(viewer: Any, origin: np.ndarray | None = None, length: float = 0.25) -> None:
    """Overlay a colored XYZ frame in the MuJoCo main viewer only."""

    scene = viewer.user_scn
    scene.ngeom = 0
    origin = np.asarray(origin if origin is not None else [0.0, 0.0, 0.02], dtype=np.float64)
    directions = (
        ("X", np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.10, 0.10, 1.0], dtype=np.float32)),
        ("Y", np.asarray([0.0, 1.0, 0.0]), np.asarray([0.10, 1.0, 0.10, 1.0], dtype=np.float32)),
        ("Z", np.asarray([0.0, 0.0, 1.0]), np.asarray([0.10, 0.35, 1.0, 1.0], dtype=np.float32)),
    )
    for label, direction, rgba in directions:
        if scene.ngeom + 2 > scene.maxgeom:
            raise RuntimeError("MuJoCo user scene does not have enough geometry slots for XYZ axes")
        endpoint = origin + direction * float(length)
        arrow = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            arrow,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.asarray([0.008, 0.008, 0.008], dtype=np.float64),
            origin,
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, origin, endpoint)
        scene.ngeom += 1

        text = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            text,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3, dtype=np.float64),
            endpoint + direction * 0.018,
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        text.label = label
        scene.ngeom += 1


def make_observation(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    adapter: OpenArmDroidAdapter,
    exterior_camera: str,
    wrist_camera: str,
    prompt: str,
    object_name: str,
    exterior_horizontal_flip: bool = True,
    exterior_image: np.ndarray | None = None,
    wrist_image: np.ndarray | None = None,
) -> tuple[dict[str, Any], VisualEvidence]:
    joint_position, gripper_position = adapter.state(data)
    exterior = (
        exterior_image
        if exterior_image is not None
        else format_exterior_image(render_camera(renderer, data, exterior_camera), exterior_horizontal_flip)
    )
    wrist = wrist_image if wrist_image is not None else render_camera(renderer, data, wrist_camera)
    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }, visual_evidence(exterior, wrist, object_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="100.122.120.76", help="DGX OpenPI host or ws:// URI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument(
        "--object",
        choices=("cube", "bottle", "ball", "towel"),
        default="cube",
        help="rigid test object; use towel to keep the original towel scene",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--prompt", default="pick up the red cube")
    parser.add_argument("--control-hz", type=int, default=15)
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=4,
        help="maximum action frames before requesting a fresh chest+wrist observation",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.2,
        help="rad/s for a normalized DROID velocity of 1; 1.2 moderate, 1.5 fast",
    )
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument(
        "--show-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the chest and wrist RGB observations in two separate GUI windows",
    )
    parser.add_argument(
        "--exterior-horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flip chest RGB observations left/right so physical left/right matches the main view",
    )
    parser.add_argument("--exterior-camera", default="chest_d435i_preview")
    parser.add_argument("--wrist-camera", default=None)
    parser.add_argument(
        "--closed-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="check camera evidence and grasp state before continuing an action chunk",
    )
    parser.add_argument(
        "--camera-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-aim the wrist camera at the object after weak evidence or failed grasp",
    )
    parser.add_argument("--camera-recovery-blend", type=float, default=0.35)
    parser.add_argument("--visual-min-pixels", type=int, default=24)
    parser.add_argument("--grasp-check-steps", type=int, default=6)
    parser.add_argument("--contact", action="store_true", help="enable runtime robot-towel contacts")
    parser.add_argument("--apply", action="store_true", help="allow remote actions to move the selected arm")
    parser.add_argument("--left-signs", type=parse_signs, default=np.ones(7))
    parser.add_argument("--right-signs", type=parse_signs, default=np.ones(7))
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--steps", type=int, default=120, help="headless control steps")
    parser.add_argument(
        "--action-log-every",
        type=int,
        default=1,
        help="print action parameters every N executed frames; default=1",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.control_hz <= 0 or args.open_loop_horizon <= 0 or args.steps <= 0 or args.action_log_every <= 0:
        raise ValueError("control frequency, open-loop horizon, steps, and action-log-every must be positive")
    if args.velocity_scale <= 0:
        raise ValueError("--velocity-scale must be positive")
    if not 0.0 < args.camera_recovery_blend <= 1.0:
        raise ValueError("--camera-recovery-blend must be in (0, 1]")
    if args.visual_min_pixels < 1 or args.grasp_check_steps < 1:
        raise ValueError("visual pixel threshold and grasp check steps must be positive")

    if args.scene is not None:
        scene = args.scene
    elif args.object == "towel":
        scene = latest_scene()
    else:
        scene = ensure_scene(args.object)
    scene = scene if scene.is_absolute() else (ROOT / scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready")
    reset_to_ready(model, data, key_id)
    if args.contact and model.nflex:
        enable_robot_cloth_contact(model)
        mujoco.mj_forward(model, data)
    elif args.contact:
        LOGGER.info("Rigid object scene: normal MuJoCo robot-object contacts are already enabled")

    arm_specs = build_arm_specs(model)
    finger_actuators = build_finger_actuators(model)
    wrist_camera = args.wrist_camera or f"v22_{args.arm}_d405_rgb"
    id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, args.exterior_camera)
    id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, wrist_camera)
    signs = args.left_signs if args.arm == "left" else args.right_signs
    adapter = OpenArmDroidAdapter(model, arm_specs, args.arm, signs, args.velocity_scale)
    wrist_camera_controller = WristCameraController(model, wrist_camera)
    grasp_monitor = GraspMonitor(model, data, adapter)

    q_target = {side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()}
    gripper = {"left": 0.020, "right": 0.020}
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    if args.headless and args.show_cameras:
        LOGGER.warning("--show-cameras is ignored in --headless mode")
    camera_preview = CameraPreview(not args.headless and args.show_cameras, args.image_width, args.image_height)
    worker = PolicyWorker(args.host, args.port, args.connect_timeout)
    if not worker.wait_connected(args.connect_timeout):
        worker.close()
        raise TimeoutError(f"Timed out waiting for DGX DROID OpenPI at {args.host}:{args.port}")
    if worker.last_error and not worker.metadata:
        worker.close()
        raise RuntimeError(worker.last_error)

    LOGGER.info("Scene: %s", scene)
    LOGGER.info("Test object: %s", args.object if args.scene is None else "custom scene")
    LOGGER.info("DGX metadata: %s", worker.metadata)
    LOGGER.info("DROID arm: %s; wrist camera: %s", args.arm, wrist_camera)
    LOGGER.info("Chest observation left/right flip: %s", args.exterior_horizontal_flip)
    LOGGER.info(
        "Closed loop: %s; camera recovery: %s; horizon=%d; visual_min_pixels=%d",
        args.closed_loop,
        args.camera_recovery,
        args.open_loop_horizon,
        args.visual_min_pixels,
    )
    LOGGER.info("Mode: %s", "APPLY remote actions" if args.apply else "PREVIEW only")
    LOGGER.info("DROID adapter: 7 joint velocities -> OpenArm joint targets; gripper position -> two fingers")
    controls = ViewerControls(allow_actions=bool(args.apply), actions_enabled=bool(args.apply))
    state_reporter = LoopStateReporter()
    state_reporter.set(LoopState.INIT, "客户端已连接，等待首帧观测")
    LOGGER.info("Viewer keys: Space=remote action pause/resume, P=physics pause, R=ready reset")

    period = 1.0 / args.control_hz
    next_request = True
    waiting_for_chunk = False
    latest_action: np.ndarray | None = None
    last_generation = -1
    action_cursor = 0
    latest_error_printed: str | None = None
    last_evidence: VisualEvidence | None = None
    last_recovery_time = -float("inf")
    recovery_count = 0
    closed_ticks = 0
    grasp_recovery_done = False
    grasp_success_logged = False
    steps_per_tick = max(1, int(round(0.01 / float(model.opt.timestep))))
    wall_tick = time.perf_counter()

    def tick() -> None:
        nonlocal next_request, waiting_for_chunk, latest_action, last_generation, action_cursor
        nonlocal latest_error_printed, gripper
        nonlocal last_evidence, last_recovery_time, recovery_count
        nonlocal closed_ticks, grasp_recovery_done, grasp_success_logged
        preview_exterior: np.ndarray | None = None
        preview_wrist: np.ndarray | None = None
        if camera_preview.enabled:
            preview_exterior = format_exterior_image(
                render_camera(renderer, data, args.exterior_camera),
                args.exterior_horizontal_flip,
            )
            preview_wrist = render_camera(renderer, data, wrist_camera)
            camera_preview.show(preview_exterior, preview_wrist)
            if camera_preview.closed:
                state_reporter.set(LoopState.PAUSED, "相机预览窗口已关闭")
                return
        if controls.reset_requested:
            reset_to_ready(model, data, key_id)
            q_target.clear()
            q_target.update({side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()})
            gripper = {"left": 0.020, "right": 0.020}
            latest_action = None
            action_cursor = 0
            next_request = True
            waiting_for_chunk = False
            last_evidence = None
            last_recovery_time = -float("inf")
            recovery_count = 0
            closed_ticks = 0
            grasp_recovery_done = False
            grasp_success_logged = False
            wrist_camera_controller.restore_nominal(data)
            state_reporter.set(LoopState.INIT, "已复位到 ready 姿态")
            controls.reset_requested = False

        actions, generation, busy, error = worker.latest()
        if error and error != latest_error_printed:
            LOGGER.error("DGX policy error: %s", error)
            latest_error_printed = error
            state_reporter.set(LoopState.ERROR, "DGX 策略推理失败")
        if actions is not None and generation != last_generation:
            latest_action = actions
            action_cursor = 0
            last_generation = generation
            waiting_for_chunk = False
            state_reporter.set(LoopState.EXECUTE, f"收到动作块 #{generation}，共 {len(actions)} 帧")

        if next_request and not busy and not waiting_for_chunk:
            state_reporter.set(LoopState.OBSERVE, "采集胸部相机和腕部相机")
            observation, evidence = make_observation(
                renderer,
                data,
                adapter,
                args.exterior_camera,
                wrist_camera,
                args.prompt,
                args.object,
                exterior_horizontal_flip=args.exterior_horizontal_flip,
                exterior_image=preview_exterior,
                wrist_image=preview_wrist,
            )
            if args.closed_loop and args.camera_recovery and recovery_count < 3:
                target_position = target_world_position(model, data)
                wrist_is_weak = evidence.wrist_pixels < args.visual_min_pixels
                cooldown_elapsed = time.perf_counter() - last_recovery_time > 0.75
                if target_position is not None and wrist_is_weak and cooldown_elapsed:
                    if wrist_camera_controller.aim_at(data, target_position, args.camera_recovery_blend):
                        recovery_count += 1
                        last_recovery_time = time.perf_counter()
                        state_reporter.set(
                            LoopState.RECOVER_CAMERA,
                            f"腕部目标像素不足，调整第 {recovery_count} 次",
                        )
                        LOGGER.warning(
                            "Camera recovery #%d: wrist evidence weak (wrist_pixels=%d, chest_pixels=%d); "
                            "re-aim wrist camera at target=%s and re-observe",
                            recovery_count,
                            evidence.wrist_pixels,
                            evidence.exterior_pixels,
                            np.round(target_position, 3).tolist(),
                        )
                        observation, evidence = make_observation(
                            renderer,
                            data,
                            adapter,
                            args.exterior_camera,
                            wrist_camera,
                            args.prompt,
                            args.object,
                            exterior_horizontal_flip=args.exterior_horizontal_flip,
                        )
                        state_reporter.set(LoopState.REPLAN, "使用新视角重新请求动作")
                        camera_preview.show(
                            observation["observation/exterior_image_1_left"],
                            observation["observation/wrist_image_left"],
                        )
            last_evidence = evidence
            if worker.submit(observation):
                next_request = False
                waiting_for_chunk = True
                state_reporter.set(LoopState.WAIT_POLICY, "已发送观测，等待 DGX 返回")

        if latest_action is not None and controls.actions_enabled and not controls.physics_paused:
            if action_cursor < min(args.open_loop_horizon, len(latest_action)):
                if state_reporter.current in (
                    LoopState.INIT,
                    LoopState.OBSERVE,
                    LoopState.WAIT_POLICY,
                    LoopState.REPLAN,
                ):
                    state_reporter.set(LoopState.EXECUTE, "执行当前动作片段")
                frame = latest_action[action_cursor]
                action_cursor += 1
                action_limit = min(args.open_loop_horizon, len(latest_action))
                frame_number = action_cursor
                current_joint_position = data.qpos[adapter.spec.qpos].copy()
                ee_xyz, ee_rpy_deg = end_effector_xyzrpy(data, adapter.spec)
                normalized_velocity = np.asarray(frame[:7], dtype=np.float64)
                effective_velocity = (
                    np.clip(normalized_velocity, -1.0, 1.0)
                    * adapter.velocity_scale
                    * adapter.joint_signs
                )
                gripper_command = float(frame[7])
                q_target[args.arm], gripper[args.arm] = adapter.integrate_action(
                    frame,
                    q_target[args.arm],
                    gripper[args.arm],
                    dt=1.0 / args.control_hz,
                )
                if frame_number == 1 or frame_number % args.action_log_every == 0 or frame_number == action_limit:
                    LOGGER.info(
                        "执行动作参数 | %s臂 | 动作块=%d | 帧=%d/%d | "
                        "qdot归一化=%s | qdot实际(rad/s)=%s | 速度倍率=%.3f | "
                        "夹爪指令=%.3f(%s) | 当前关节角(rad)=%s | "
                        "末端XYZ(m)=%s | RPY(deg)=%s | 目标夹爪(m)=%.4f",
                        "左" if args.arm == "left" else "右",
                        last_generation,
                        frame_number,
                        action_limit,
                        format_vector(normalized_velocity),
                        format_vector(effective_velocity),
                        adapter.velocity_scale,
                        gripper_command,
                        "张开" if gripper_command > 0.5 else "闭合",
                        format_vector(current_joint_position),
                        format_vector(ee_xyz),
                        format_vector(ee_rpy_deg),
                        gripper[args.arm],
                    )
            else:
                # Do not hold the last velocity while a new chunk is pending.
                q_target[args.arm] = np.clip(
                    q_target[args.arm], adapter.spec.q_range[:, 0], adapter.spec.q_range[:, 1]
                )
                next_request = True
                waiting_for_chunk = False

        # The remote policy can close the gripper before it has actually
        # captured the object.  Hold that event briefly, then use both camera
        # evidence and the local simulator label to decide whether to recover.
        current_gripper = adapter.state(data)[1]
        gripper_closed = bool(float(current_gripper[0]) < 0.12 or gripper[args.arm] < 0.006)
        if not gripper_closed:
            closed_ticks = 0
            grasp_recovery_done = False
            grasp_success_logged = False
        elif args.closed_loop:
            if not grasp_success_logged:
                state_reporter.set(LoopState.CHECK_GRASP, "夹爪已闭合，检查是否抓取成功")
            closed_ticks += 1
            grasped = grasp_monitor.is_grasped(data, gripper_closed=True)
            camera_visible = bool(
                last_evidence is not None
                and (
                    last_evidence.wrist_pixels >= args.visual_min_pixels
                    or last_evidence.exterior_pixels >= args.visual_min_pixels
                )
            )
            if grasped is True and camera_visible and not grasp_success_logged:
                state_reporter.set(LoopState.SUCCESS, "目标已抬起且相机仍可观测")
                LOGGER.info(
                    "Closed-loop grasp confirmed: wrist_pixels=%d chest_pixels=%d",
                    last_evidence.wrist_pixels if last_evidence else 0,
                    last_evidence.exterior_pixels if last_evidence else 0,
                )
                grasp_success_logged = True
            elif (
                closed_ticks >= args.grasp_check_steps
                and not grasp_recovery_done
                and args.camera_recovery
                and recovery_count < 3
            ):
                target_position = target_world_position(model, data)
                cooldown_elapsed = time.perf_counter() - last_recovery_time > 0.75
                if target_position is not None and cooldown_elapsed:
                    if wrist_camera_controller.aim_at(data, target_position, args.camera_recovery_blend):
                        recovery_count += 1
                        last_recovery_time = time.perf_counter()
                        grasp_recovery_done = True
                        latest_action = None
                        action_cursor = 0
                        next_request = True
                        waiting_for_chunk = False
                        state_reporter.set(
                            LoopState.RECOVER_CAMERA,
                            f"抓取未确认，调整第 {recovery_count} 次",
                        )
                        LOGGER.warning(
                            "Grasp not confirmed after %d closed frames (sim_grasp=%s, "
                            "camera_visible=%s); adjusted wrist camera and requested a new action",
                            closed_ticks,
                            grasped,
                            camera_visible,
                        )
                        state_reporter.set(LoopState.REPLAN, "放弃旧动作片段，重新请求策略")

        if not controls.physics_paused:
            for _ in range(steps_per_tick):
                apply_controller(model, data, arm_specs, finger_actuators, q_target, gripper)
                mujoco.mj_step(model, data)
        reason = safety_reason(data, arm_specs)
        if reason is not None:
            LOGGER.error("Safety pause: %s", reason)
            state_reporter.set(LoopState.SAFETY, reason)
            controls.physics_paused = True
            for side, spec in arm_specs.items():
                q_target[side] = data.qpos[spec.qpos].copy()

    try:
        if args.headless:
            LOGGER.info("Headless test: %d control steps", args.steps)
            initial = data.qpos.copy()
            initial_arm_qpos = {
                side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()
            }
            for _ in range(args.steps):
                tick()
                time.sleep(period)
            actions, generation, _busy, error = worker.latest()
            arm_changes = {
                side: float(np.max(np.abs(data.qpos[spec.qpos] - initial_arm_qpos[side])))
                for side, spec in arm_specs.items()
            }
            LOGGER.info(
                "Headless result: generation=%d action_shape=%s arm_qpos_change=%s qpos_max_change=%.6f error=%s",
                generation,
                None if actions is None else tuple(actions.shape),
                arm_changes,
                float(np.max(np.abs(data.qpos - initial))),
                error,
            )
        else:
            with mujoco.viewer.launch_passive(model, data, key_callback=controls.on_key) as viewer:
                # Main MuJoCo view: overhead view from the support side toward
                # the tabletop, keeping both arms and the object in frame.
                viewer.cam.lookat[:] = [0.38, 0.0, 0.34]
                viewer.cam.distance = 1.15
                viewer.cam.azimuth = 0
                viewer.cam.elevation = -55
                add_world_axes(viewer)
                viewer.sync()
                while viewer.is_running() and not camera_preview.closed:
                    tick()
                    viewer.sync()
                    wall_tick += period
                    time.sleep(max(0.0, wall_tick - time.perf_counter()))
    finally:
        worker.close()
        renderer.close()
        camera_preview.close()
    return 0


def main() -> None:
    # Keep Chinese state-machine messages readable in PowerShell/conda pipes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
