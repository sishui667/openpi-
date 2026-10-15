# -*- coding: utf-8 -*-
"""Keyboard teleoperation and dataset recording for the V25.4 towel scene.

Run inside the ``openpi`` conda environment::

    python teleop_towel_dataset.py --robot-cloth-contact --images
    python teleop_towel_dataset.py --robot-cloth-contact --gui

The arm actuators in the MJCF are torque motors, so keyboard input changes
Cartesian end-effector targets.  A damped-least-squares IK layer converts the
targets to joint targets and a gravity-compensated PD controller produces the
motor torques.  Finger actuators remain position controlled.

The output is one episode directory containing ``episode.npz`` and optional
PNG camera frames.  The NPZ contains robot state, desired action, end-effector
poses and timing, which is a useful neutral format for a later OpenPI export.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "simulation" / "v1"
SCENE_PATTERN = "folding_scene_V25_4_TOWEL_SUPPORT_*.xml"

ARM_Q_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
FINGER_Q_NAMES = {
    "left": ["openarm_left_finger_joint1", "openarm_left_finger_joint2"],
    "right": ["openarm_right_finger_joint1", "openarm_right_finger_joint2"],
}
EE_BODY_NAMES = {
    "left": "openarm_left_hand_tcp",
    "right": "openarm_right_hand_tcp",
}
ARM_ACTUATOR_NAMES = {
    "left": [f"left_joint{i}_ctrl" for i in range(1, 8)],
    "right": [f"right_joint{i}_ctrl" for i in range(1, 8)],
}
FINGER_ACTUATOR_NAMES = {
    "left": ["left_finger1_ctrl", "left_finger2_ctrl"],
    "right": ["right_finger1_ctrl", "right_finger2_ctrl"],
}

# Keyboard increments.  X is the folding direction in the V25.4 scene.
TRANSLATION_KEYS = {
    glfw.KEY_I: (np.array([1.0, 0.0, 0.0]), "x+"),
    glfw.KEY_K: (np.array([-1.0, 0.0, 0.0]), "x-"),
    glfw.KEY_L: (np.array([0.0, 1.0, 0.0]), "y+"),
    glfw.KEY_J: (np.array([0.0, -1.0, 0.0]), "y-"),
    glfw.KEY_U: (np.array([0.0, 0.0, 1.0]), "z+"),
    glfw.KEY_O: (np.array([0.0, 0.0, -1.0]), "z-"),
}
ROTATION_KEYS = {
    glfw.KEY_Q: (np.array([0.0, 0.0, 1.0]), 1.0, "yaw+"),
    glfw.KEY_E: (np.array([0.0, 0.0, 1.0]), -1.0, "yaw-"),
    glfw.KEY_Z: (np.array([0.0, 1.0, 0.0]), 1.0, "pitch+"),
    glfw.KEY_C: (np.array([0.0, 1.0, 0.0]), -1.0, "pitch-"),
    glfw.KEY_X: (np.array([1.0, 0.0, 0.0]), 1.0, "roll+"),
    glfw.KEY_V: (np.array([1.0, 0.0, 0.0]), -1.0, "roll-"),
}
POSE_ACTIONS = {
    "x+", "x-", "y+", "y-", "z+", "z-",
    "yaw+", "yaw-", "pitch+", "pitch-", "roll+", "roll-",
}


def latest_scene() -> Path:
    candidates = list(SCENE_DIR.glob(SCENE_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            "No V25.4 scene found. Run build_towel_scene_v25_4.py first."
        )

    def version(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=version)


def id_for(model, obj_type, name: str) -> int:
    value = mujoco.mj_name2id(model, obj_type, name)
    if value < 0:
        raise RuntimeError(f"Required MuJoCo object is missing: {name}")
    return int(value)


def quat_from_body(model, data, body_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.xmat[body_id])
    return quat


def body_pose(model, data, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.xpos[body_id].copy(), quat_from_body(model, data, body_id)


def reset_to_ready(model, data, key_id: int) -> None:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def enable_robot_cloth_contact(model) -> None:
    """Enable a separate collision bit between robot geoms and the flex cloth.

    V25.4 intentionally isolates the cloth from the robot for its stability
    validation.  This runtime opt-in keeps the verified XML unchanged while
    enabling contact for teleoperation.  The hidden support keeps bit 4;
    robot-cloth uses bit 8, so robot geoms do not collide with that support.
    """
    flex_name = "folding_towel_v25_4"
    flex_id = id_for(model, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
    model.flex_contype[flex_id] = int(model.flex_contype[flex_id]) | 4
    model.flex_conaffinity[flex_id] = int(model.flex_conaffinity[flex_id]) | 8

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        is_robot = False
        while body_id > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith("openarm_"):
                is_robot = True
                break
            body_id = int(model.body_parentid[body_id])
        if is_robot:
            model.geom_contype[geom_id] = int(model.geom_contype[geom_id]) | 8
            model.geom_conaffinity[geom_id] = int(model.geom_conaffinity[geom_id]) | 4


@dataclass
class ArmSpec:
    qpos: np.ndarray
    dof: np.ndarray
    actuators: np.ndarray
    body_id: int
    q_range: np.ndarray


def build_arm_specs(model) -> dict[str, ArmSpec]:
    specs: dict[str, ArmSpec] = {}
    for side in ("left", "right"):
        qpos = []
        dof = []
        ranges = []
        for name in ARM_Q_NAMES[side]:
            jid = id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos.append(int(model.jnt_qposadr[jid]))
            dof.append(int(model.jnt_dofadr[jid]))
            ranges.append(model.jnt_range[jid].copy())
        actuators = [
            id_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_ACTUATOR_NAMES[side]
        ]
        body_id = id_for(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAMES[side])
        specs[side] = ArmSpec(
            qpos=np.asarray(qpos, dtype=np.int32),
            dof=np.asarray(dof, dtype=np.int32),
            actuators=np.asarray(actuators, dtype=np.int32),
            body_id=body_id,
            q_range=np.asarray(ranges, dtype=np.float64),
        )
    return specs


def build_finger_actuators(model) -> dict[str, np.ndarray]:
    return {
        side: np.asarray(
            [
                id_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in FINGER_ACTUATOR_NAMES[side]
            ],
            dtype=np.int32,
        )
        for side in ("left", "right")
    }


class TeleopState:
    """Small thread-safe command store shared by the viewer callback."""

    def __init__(self, model, data, arm_specs: dict[str, ArmSpec]):
        self._lock = threading.Lock()
        self.active_arm = "left"
        self.mirror_y = True
        self.step_scale = 1.0
        self.linear_step_m = 0.015
        self.angular_step_deg = 8.0
        self.repeat_ms = 50
        self.paused = False
        self.gripper = {"left": 0.020, "right": 0.020}
        self.target_pos: dict[str, np.ndarray] = {}
        self.target_quat: dict[str, np.ndarray] = {}
        self.events: list[dict[str, Any]] = []
        self.commands: list[str] = []
        self._set_targets_from_data(model, data, arm_specs)

    def _set_targets_from_data(self, model, data, arm_specs: dict[str, ArmSpec]) -> None:
        for side, spec in arm_specs.items():
            pos, quat = body_pose(model, data, spec.body_id)
            self.target_pos[side] = pos
            self.target_quat[side] = quat

    def reset_targets(self, model, data, arm_specs: dict[str, ArmSpec]) -> None:
        with self._lock:
            self._set_targets_from_data(model, data, arm_specs)
            self.gripper = {"left": 0.020, "right": 0.020}

    def _selected_sides(self) -> tuple[str, ...]:
        if self.active_arm == "both":
            return ("left", "right")
        return (self.active_arm,)

    def set_motion_params(
        self,
        linear_mm: float | None = None,
        angular_deg: float | None = None,
        repeat_ms: int | None = None,
    ) -> None:
        with self._lock:
            if linear_mm is not None:
                self.linear_step_m = float(np.clip(linear_mm, 1.0, 50.0)) / 1000.0
            if angular_deg is not None:
                self.angular_step_deg = float(np.clip(angular_deg, 1.0, 30.0))
            if repeat_ms is not None:
                self.repeat_ms = int(np.clip(repeat_ms, 20, 250))
            self._event(
                "PARAM",
                f"step_{self.linear_step_m * 1000:.1f}mm_{self.angular_step_deg:.1f}deg_{self.repeat_ms}ms",
            )

    def set_preset(self, name: str) -> None:
        presets = {
            "slow": (5.0, 3.0, 80),
            "standard": (12.0, 6.0, 55),
            "fast": (25.0, 12.0, 35),
        }
        if name not in presets:
            raise ValueError(f"Unknown motion preset: {name}")
        self.set_motion_params(*presets[name])

    def _event(self, key: str, action: str) -> None:
        self.events.append(
            {
                "wall_time": time.time(),
                "key": key,
                "action": action,
                "active_arm": self.active_arm,
            }
        )

    def on_key(self, keycode: int) -> None:
        with self._lock:
            if keycode == glfw.KEY_1:
                self.active_arm = "left"
                self._event("1", "select_left")
                return
            if keycode == glfw.KEY_2:
                self.active_arm = "right"
                self._event("2", "select_right")
                return
            if keycode == glfw.KEY_3:
                self.active_arm = "both"
                self._event("3", "select_both")
                return
            if keycode == glfw.KEY_M:
                self.mirror_y = not self.mirror_y
                self._event("M", "mirror_y_on" if self.mirror_y else "mirror_y_off")
                return
            if keycode == glfw.KEY_LEFT_BRACKET:
                self.step_scale = max(0.25, self.step_scale / 1.5)
                self._event("[", f"speed_{self.step_scale:.3f}")
                return
            if keycode == glfw.KEY_RIGHT_BRACKET:
                self.step_scale = min(4.0, self.step_scale * 1.5)
                self._event("]", f"speed_{self.step_scale:.3f}")
                return
            if keycode == glfw.KEY_SPACE:
                self.commands.append("toggle_record")
                self._event("SPACE", "toggle_record")
                return
            if keycode == glfw.KEY_R:
                self.commands.append("reset")
                self._event("R", "reset_ready")
                return
            if keycode == glfw.KEY_P:
                self.paused = not self.paused
                self._event("P", "pause" if self.paused else "resume")
                return
            if keycode == glfw.KEY_F1:
                self.commands.append("help")
                self._event("F1", "help")
                return
            if keycode == glfw.KEY_G:
                for side in self._selected_sides():
                    self.gripper[side] = 0.000
                self._event("G", "close_gripper")
                return
            if keycode == glfw.KEY_H:
                for side in self._selected_sides():
                    self.gripper[side] = 0.044
                self._event("H", "open_gripper")
                return
            if keycode in TRANSLATION_KEYS:
                direction, name = TRANSLATION_KEYS[keycode]
                distance = self.linear_step_m * self.step_scale
                for side in self._selected_sides():
                    local_direction = direction.copy()
                    if (
                        self.active_arm == "both"
                        and self.mirror_y
                        and abs(float(direction[1])) > 0.0
                        and side == "right"
                    ):
                        local_direction[1] *= -1.0
                    self.target_pos[side] += local_direction * distance
                self._event(chr(keycode), name)
                return
            if keycode in ROTATION_KEYS:
                axis, sign, name = ROTATION_KEYS[keycode]
                angle = np.deg2rad(self.angular_step_deg * self.step_scale) * sign
                delta = np.zeros(4, dtype=np.float64)
                mujoco.mju_axisAngle2Quat(delta, axis, angle)
                for side in self._selected_sides():
                    updated = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mulQuat(updated, delta, self.target_quat[side])
                    self.target_quat[side] = updated / np.linalg.norm(updated)
                self._event(chr(keycode), name)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_arm": self.active_arm,
                "mirror_y": self.mirror_y,
                "step_scale": self.step_scale,
                "linear_step_mm": self.linear_step_m * 1000.0 * self.step_scale,
                "angular_step_deg": self.angular_step_deg * self.step_scale,
                "repeat_ms": self.repeat_ms,
                "paused": self.paused,
                "gripper": dict(self.gripper),
                "target_pos": {k: v.copy() for k, v in self.target_pos.items()},
                "target_quat": {k: v.copy() for k, v in self.target_quat.items()},
            }

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = self.events
            self.events = []
            return events

    def drain_commands(self) -> list[str]:
        with self._lock:
            commands = self.commands
            self.commands = []
            return commands

    def emergency_pause(self, reason: str) -> None:
        with self._lock:
            if not self.paused:
                self.paused = True
                self._event("SAFETY", reason)


def safety_reason(data, specs: dict[str, ArmSpec]) -> str | None:
    """Return a reason to pause before an arm disappears from the workspace."""
    limits = {
        "x": (0.15, 0.80),
        "y": (-0.50, 0.50),
        "z": (0.08, 0.85),
    }
    for side, spec in specs.items():
        pos = data.xpos[spec.body_id]
        if not np.isfinite(pos).all():
            return f"{side}_ee_non_finite"
        if not (limits["x"][0] <= pos[0] <= limits["x"][1]):
            return f"{side}_ee_x_out_of_workspace"
        if not (limits["y"][0] <= pos[1] <= limits["y"][1]):
            return f"{side}_ee_y_out_of_workspace"
        if not (limits["z"][0] <= pos[2] <= limits["z"][1]):
            return f"{side}_ee_z_out_of_workspace"
    if not np.isfinite(data.qpos).all():
        return "qpos_non_finite"
    return None


def ik_step(model, data, spec: ArmSpec, target_pos, target_quat, q_target) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, spec.body_id)

    current_quat = quat_from_body(model, data, spec.body_id)
    rotation_error = np.zeros(3, dtype=np.float64)
    mujoco.mju_subQuat(rotation_error, target_quat, current_quat)
    position_error = target_pos - data.xpos[spec.body_id]
    task_error = np.concatenate(
        [np.clip(position_error, -0.025, 0.025), np.clip(rotation_error, -0.20, 0.20)]
    )

    jacobian = np.vstack([jacp[:, spec.dof], jacr[:, spec.dof]])
    regularizer = 2e-4 * np.eye(6, dtype=np.float64)
    try:
        delta_q = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer, task_error
        )
    except np.linalg.LinAlgError:
        return q_target
    delta_norm = float(np.linalg.norm(delta_q))
    if delta_norm > 0.08:
        delta_q *= 0.08 / delta_norm
    return np.clip(q_target + delta_q, spec.q_range[:, 0], spec.q_range[:, 1])


def apply_controller(
    model,
    data,
    specs: dict[str, ArmSpec],
    finger_actuators: dict[str, np.ndarray],
    q_target: dict[str, np.ndarray],
    gripper: dict[str, float],
) -> np.ndarray:
    action = np.zeros(18, dtype=np.float64)
    for side, spec in specs.items():
        q = data.qpos[spec.qpos]
        qvel = data.qvel[spec.dof]
        # Gains are deliberately moderate because the V25.4 scene uses direct
        # torque motors and a small timestep.
        kp = np.asarray([80, 80, 55, 55, 25, 20, 20], dtype=np.float64)
        kd = np.asarray([8, 8, 5, 5, 3, 2, 2], dtype=np.float64)
        torque = data.qfrc_bias[spec.dof] + kp * (q_target[side] - q) - kd * qvel
        actuator = spec.actuators
        limits = model.actuator_forcerange[actuator]
        data.ctrl[actuator] = np.clip(torque, limits[:, 0], limits[:, 1])

        base = 0 if side == "left" else 9
        action[base : base + 7] = q_target[side]
        finger_value = float(np.clip(gripper[side], 0.0, 0.044))
        data.ctrl[finger_actuators[side]] = finger_value
        action[base + 7 : base + 9] = finger_value
    return action


def current_ee_pose(model, data, specs: dict[str, ArmSpec]) -> np.ndarray:
    pose = []
    for side in ("left", "right"):
        pos, quat = body_pose(model, data, specs[side].body_id)
        pose.append(np.concatenate([pos, quat]))
    return np.asarray(pose, dtype=np.float64)


def towel_vertices(model, data) -> np.ndarray:
    flex_id = id_for(model, mujoco.mjtObj.mjOBJ_FLEX, "folding_towel_v25_4")
    start = int(model.flex_vertadr[flex_id])
    count = int(model.flex_vertnum[flex_id])
    return data.flexvert_xpos[start : start + count].copy()


class EpisodeRecorder:
    def __init__(self, output_root: Path, save_images: bool, cameras: list[str], image_stride: int, save_cloth: bool):
        self.output_root = output_root
        self.save_images = save_images
        self.cameras = cameras
        self.image_stride = image_stride
        self.save_cloth = save_cloth
        self.active = False
        self.episode_dir: Path | None = None
        self.events: list[dict[str, Any]] = []
        self.time: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.ee_pose: list[np.ndarray] = []
        self.ee_target: list[np.ndarray] = []
        self.towel_center: list[np.ndarray] = []
        self.cloth: list[np.ndarray] = []
        self.frame_count = 0

    def start(self, metadata: dict[str, Any]) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.episode_dir = self.output_root / f"episode_{stamp}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        if self.save_images:
            for camera in self.cameras:
                (self.episode_dir / "images" / camera).mkdir(parents=True, exist_ok=True)
        self.active = True
        self.events = []
        self.time = []
        self.qpos = []
        self.qvel = []
        self.ctrl = []
        self.action = []
        self.ee_pose = []
        self.ee_target = []
        self.towel_center = []
        self.cloth = []
        self.frame_count = 0
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[RECORD] started: {self.episode_dir}")

    def append(self, model, data, action: np.ndarray, ee_pose: np.ndarray, ee_target: np.ndarray, renderer=None) -> None:
        if not self.active or self.episode_dir is None:
            return
        self.time.append(float(data.time))
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.ctrl.append(data.ctrl.copy())
        self.action.append(action.copy())
        self.ee_pose.append(ee_pose.copy())
        self.ee_target.append(ee_target.copy())
        vertices = towel_vertices(model, data)
        self.towel_center.append(vertices.mean(axis=0))
        if self.save_cloth:
            self.cloth.append(vertices)

        if renderer is not None and self.frame_count % self.image_stride == 0:
            for camera in self.cameras:
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                path = self.episode_dir / "images" / camera / f"{self.frame_count:06d}.png"
                Image.fromarray(rgb).save(path)
        self.frame_count += 1

    def stop(self, metadata: dict[str, Any]) -> Path | None:
        if not self.active or self.episode_dir is None:
            return None
        if not self.time:
            print("[RECORD] discarded empty episode")
            self.active = False
            return None
        arrays = {
            "time": np.asarray(self.time, dtype=np.float64),
            "qpos": np.asarray(self.qpos, dtype=np.float64),
            "qvel": np.asarray(self.qvel, dtype=np.float64),
            "ctrl": np.asarray(self.ctrl, dtype=np.float64),
            "action_target": np.asarray(self.action, dtype=np.float64),
            "ee_pose": np.asarray(self.ee_pose, dtype=np.float64),
            "ee_target": np.asarray(self.ee_target, dtype=np.float64),
            "towel_center": np.asarray(self.towel_center, dtype=np.float64),
        }
        if self.save_cloth:
            arrays["towel_vertices"] = np.asarray(self.cloth, dtype=np.float64)
        np.savez_compressed(self.episode_dir / "episode.npz", **arrays)
        (self.episode_dir / "events.json").write_text(
            json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata = dict(metadata)
        metadata.update({"frames": len(self.time), "duration_s": self.time[-1] - self.time[0]})
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[RECORD] saved: {self.episode_dir} ({len(self.time)} frames)")
        path = self.episode_dir
        self.active = False
        return path


class ControlPanel:
    """A small Tk control panel synchronized with the MuJoCo viewer window."""

    def __init__(self, state: TeleopState):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.state = state
        self.closed = False
        self.root = tk.Tk()
        self.root.title("OpenArm 毛巾遥操作控制面板")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="控制对象：左臂 | 仿真运行中")
        self.linear_var = tk.DoubleVar(value=15.0)
        self.angular_var = tk.DoubleVar(value=8.0)
        self.repeat_var = tk.IntVar(value=50)
        self._held: dict[int, bool] = {}
        self._after_ids: dict[int, Any] = {}
        self.root.bind_all("<ButtonRelease-1>", lambda _event: self._stop_all())

        ttk.Label(self.root, text="OpenArm 双臂毛巾遥操作", font=("Microsoft YaHei UI", 12, "bold")).pack(
            padx=10, pady=(10, 4)
        )
        ttk.Label(self.root, textvariable=self.status, foreground="#075985").pack(
            padx=10, pady=(0, 8)
        )

        mode = ttk.LabelFrame(self.root, text="控制对象")
        mode.pack(fill="x", padx=10, pady=4)
        self._button(mode, "左臂", glfw.KEY_1, 0, 0)
        self._button(mode, "右臂", glfw.KEY_2, 0, 1)
        self._button(mode, "双臂", glfw.KEY_3, 0, 2)
        self._button(mode, "双臂Y镜像", glfw.KEY_M, 0, 3)

        params = ttk.LabelFrame(self.root, text="速度参数")
        params.pack(fill="x", padx=10, pady=4)
        ttk.Label(params, text="平移步长(mm)").grid(row=0, column=0, padx=3, pady=3)
        tk.Spinbox(params, from_=1, to=50, increment=1, textvariable=self.linear_var, width=7).grid(
            row=0, column=1, padx=3, pady=3
        )
        ttk.Label(params, text="旋转步长(度)").grid(row=0, column=2, padx=3, pady=3)
        tk.Spinbox(params, from_=1, to=30, increment=1, textvariable=self.angular_var, width=7).grid(
            row=0, column=3, padx=3, pady=3
        )
        ttk.Label(params, text="连续间隔(ms)").grid(row=1, column=0, padx=3, pady=3)
        tk.Spinbox(params, from_=20, to=250, increment=5, textvariable=self.repeat_var, width=7).grid(
            row=1, column=1, padx=3, pady=3
        )
        self._callback_button(params, "应用参数", self._apply_params, 1, 2)
        self._callback_button(params, "慢速", lambda: self._preset("slow"), 2, 0)
        self._callback_button(params, "标准", lambda: self._preset("standard"), 2, 1)
        self._callback_button(params, "快速", lambda: self._preset("fast"), 2, 2)

        move = ttk.LabelFrame(self.root, text="末端移动（按住按钮连续运动）")
        move.pack(fill="x", padx=10, pady=4)
        self._hold_button(move, "X+ 前进", glfw.KEY_I, 0, 0)
        self._hold_button(move, "X- 后退", glfw.KEY_K, 0, 1)
        self._hold_button(move, "Y+", glfw.KEY_L, 1, 0)
        self._hold_button(move, "Y-", glfw.KEY_J, 1, 1)
        self._hold_button(move, "Z+ 上升", glfw.KEY_U, 2, 0)
        self._hold_button(move, "Z- 下降", glfw.KEY_O, 2, 1)

        rotate = ttk.LabelFrame(self.root, text="末端姿态（按住按钮连续旋转）")
        rotate.pack(fill="x", padx=10, pady=4)
        self._hold_button(rotate, "Yaw+", glfw.KEY_Q, 0, 0)
        self._hold_button(rotate, "Yaw-", glfw.KEY_E, 0, 1)
        self._hold_button(rotate, "Pitch+", glfw.KEY_Z, 1, 0)
        self._hold_button(rotate, "Pitch-", glfw.KEY_C, 1, 1)
        self._hold_button(rotate, "Roll+", glfw.KEY_X, 2, 0)
        self._hold_button(rotate, "Roll-", glfw.KEY_V, 2, 1)

        gripper = ttk.LabelFrame(self.root, text="当前选中夹爪")
        gripper.pack(fill="x", padx=10, pady=4)
        self._button(gripper, "闭合 G", glfw.KEY_G, 0, 0)
        self._button(gripper, "打开 H", glfw.KEY_H, 0, 1)

        actions = ttk.Frame(self.root)
        actions.pack(fill="x", padx=10, pady=(4, 10))
        self._button(actions, "复位 R", glfw.KEY_R, 0, 0)
        self._button(actions, "暂停/继续 P", glfw.KEY_P, 0, 1)
        self._button(actions, "录制 SPACE", glfw.KEY_SPACE, 0, 2)
        self._button(actions, "帮助 F1", glfw.KEY_F1, 0, 3)

    def _button(self, parent, text: str, keycode: int, row: int, column: int) -> None:
        from tkinter import ttk

        button = ttk.Button(parent, text=text, command=lambda: self.state.on_key(keycode), width=12)
        button.grid(row=row, column=column, padx=3, pady=3, sticky="ew")

    def _callback_button(self, parent, text: str, callback, row: int, column: int) -> None:
        from tkinter import ttk

        button = ttk.Button(parent, text=text, command=callback, width=12)
        button.grid(row=row, column=column, padx=3, pady=3, sticky="ew")

    def _hold_button(self, parent, text: str, keycode: int, row: int, column: int) -> None:
        from tkinter import ttk

        button = ttk.Button(parent, text=text, width=12)
        button.bind("<ButtonPress-1>", lambda _event: self._start_hold(keycode))
        button.bind("<ButtonRelease-1>", lambda _event: self._stop_hold(keycode))
        button.grid(row=row, column=column, padx=3, pady=3, sticky="ew")

    def _start_hold(self, keycode: int) -> None:
        if self.closed or self._held.get(keycode, False):
            return
        self._held[keycode] = True
        self.state.on_key(keycode)
        self._repeat_hold(keycode)

    def _repeat_hold(self, keycode: int) -> None:
        if self.closed or not self._held.get(keycode, False):
            return
        self.state.on_key(keycode)
        delay = max(20, int(self.state.snapshot()["repeat_ms"]))
        self._after_ids[keycode] = self.root.after(delay, lambda: self._repeat_hold(keycode))

    def _stop_hold(self, keycode: int) -> None:
        self._held[keycode] = False
        after_id = self._after_ids.pop(keycode, None)
        if after_id is not None and not self.closed:
            try:
                self.root.after_cancel(after_id)
            except self.tk.TclError:
                pass

    def _stop_all(self) -> None:
        for keycode in list(self._held):
            self._stop_hold(keycode)

    def _apply_params(self) -> None:
        try:
            self.state.set_motion_params(
                linear_mm=float(self.linear_var.get()),
                angular_deg=float(self.angular_var.get()),
                repeat_ms=int(self.repeat_var.get()),
            )
        except (TypeError, ValueError, self.tk.TclError):
            return

    def _preset(self, name: str) -> None:
        presets = {
            "slow": (5.0, 3.0, 80),
            "standard": (12.0, 6.0, 55),
            "fast": (25.0, 12.0, 35),
        }
        linear, angular, repeat = presets[name]
        self.linear_var.set(linear)
        self.angular_var.set(angular)
        self.repeat_var.set(repeat)
        self._apply_params()

    def update_status(self, recording: bool) -> None:
        if self.closed:
            return
        snapshot = self.state.snapshot()
        names = {"left": "左臂", "right": "右臂", "both": "双臂"}
        state_name = "暂停" if snapshot["paused"] else "运行中"
        record_name = "录制中" if recording else "未录制"
        mirror_name = "Y镜像" if snapshot["mirror_y"] else "Y不镜像"
        self.status.set(
            f"控制对象：{names[snapshot['active_arm']]} | {state_name} | {record_name} | "
            f"{snapshot['linear_step_mm']:.0f}mm/{snapshot['angular_step_deg']:.0f}°/{snapshot['repeat_ms']}ms | {mirror_name}"
        )

    def poll(self) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def close(self) -> None:
        if not self.closed:
            self._stop_all()
            self.closed = True
            self.root.destroy()


def key_name(keycode: int) -> str:
    if 32 <= keycode <= 126:
        return chr(keycode)
    return str(keycode)


def make_metadata(scene: Path, model, args) -> dict[str, Any]:
    return {
        "format": "openarm_towel_teleop_v2",
        "scene": str(scene.resolve()),
        "mujoco_version": mujoco.__version__,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "timestep_s": float(model.opt.timestep),
        "control_hz": int(args.control_hz),
        "robot_cloth_contact": bool(args.robot_cloth_contact),
        "cameras": args.cameras if args.images else [],
        "image_size": [int(args.image_width), int(args.image_height)] if args.images else None,
        "image_stride": int(args.image_stride),
        "save_cloth_state": bool(args.save_cloth_state),
        "motion_defaults": {
            "linear_step_mm": 15.0,
            "angular_step_deg": 8.0,
            "repeat_ms": 50,
            "hold_buttons": True,
        },
        "arm_joint_order": ARM_Q_NAMES,
        "action_layout": "left 7 arm joints + 2 finger targets, then right 7 + 2",
        "keyboard": {
            "1/2/3": "select left/right/both arms",
            "M": "toggle mirrored Y movement in both-arm mode",
            "I/K": "x +/-; folding direction",
            "J/L": "y -/+",
            "U/O": "z +/-",
            "Q/E": "yaw +/-",
            "Z/C": "pitch +/-",
            "X/V": "roll +/-",
            "G/H": "close/open selected gripper",
            "SPACE": "start/stop recording",
            "R": "reset towel_ready pose",
            "P": "pause/resume physics",
            "F1": "print controls",
            "[/]": "change Cartesian step size",
        },
    }


def print_controls() -> None:
    print("[HELP] 1=select LEFT only; 2=select RIGHT only; 3=select BOTH")
    print("[HELP] After selecting: I/K=X +/-; J/L=Y -/+; U/O=Z +/-; GUI movement buttons can be held")
    print("[HELP] Q/E=yaw +/-; Z/C=pitch +/-; X/V=roll +/-")
    print("[HELP] G=close selected gripper; H=open selected gripper")
    print("[HELP] SPACE=start/stop recording; R=reset; P=pause/resume; F1=help")
    print("[HELP] M=mirror Y in both-arm mode; [/]=change step scale; GUI has step/preset controls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=None, help="V25.4 XML scene")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "towel_teleop")
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--record", action="store_true", help="start recording immediately")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open a button control panel beside the MuJoCo visualization",
    )
    parser.add_argument("--images", action="store_true", help="save camera PNG frames")
    parser.add_argument("--cameras", nargs="+", default=["v22_left_d405_rgb", "v22_right_d405_rgb", "chest_d435i_preview"])
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--save-cloth-state", action="store_true", help="also save all flex vertices")
    parser.add_argument(
        "--robot-cloth-contact",
        action="store_true",
        help="enable runtime contact between OpenArm geoms and the towel",
    )
    args = parser.parse_args()
    if args.control_hz <= 0 or args.image_stride <= 0:
        parser.error("--control-hz and --image-stride must be positive")

    scene = args.scene or latest_scene()
    if not scene.is_absolute():
        scene = (ROOT / scene).resolve()
    if not scene.is_file():
        parser.error(f"Scene XML not found: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready")
    reset_to_ready(model, data, key_id)
    if args.robot_cloth_contact:
        enable_robot_cloth_contact(model)
        mujoco.mj_forward(model, data)
    if args.images:
        for camera in args.cameras:
            id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    specs = build_arm_specs(model)
    finger_actuators = build_finger_actuators(model)
    state = TeleopState(model, data, specs)
    q_target = {side: data.qpos[spec.qpos].copy() for side, spec in specs.items()}
    dirty_sides: set[str] = set()
    renderer = None
    if args.images:
        renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)

    metadata = make_metadata(scene, model, args)
    recorder = EpisodeRecorder(
        args.output_dir,
        save_images=args.images,
        cameras=args.cameras,
        image_stride=args.image_stride,
        save_cloth=args.save_cloth_state,
    )
    if args.record:
        recorder.start(metadata)
    panel = ControlPanel(state) if args.gui else None

    print(f"[SCENE] {scene}")
    print("[MODE] keyboard Cartesian IK + gravity-compensated PD")
    print_controls()
    print("[INFO] Pressing 1/2/3 only selects an arm; it does not move the robot.")
    print("[INFO] Close this window and restart after changing the script.")
    if args.robot_cloth_contact:
        print("[CONTACT] robot-cloth contact enabled for this session")
    else:
        print("[CONTACT] disabled; use --robot-cloth-contact for physical towel contact")
    if panel is not None:
        print("[GUI] button control panel opened beside the MuJoCo window")

    sample_period = 1.0 / args.control_hz
    sample_time = 0.0
    wall_tick = time.perf_counter()
    tick_period = 0.01
    steps_per_tick = max(1, int(round(tick_period / float(model.opt.timestep))))
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=state.on_key) as viewer:
            viewer.cam.lookat[:] = [0.45, 0.0, 0.38]
            viewer.cam.distance = 1.05
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25
            viewer.sync()
            while viewer.is_running():
                if panel is not None:
                    panel.poll()
                    if panel.closed:
                        panel = None
                    else:
                        panel.update_status(recorder.active)
                for event in state.drain_events():
                    if recorder.active:
                        recorder.events.append(event)
                    if event["action"] == "select_left":
                        print("[SELECT] LEFT ONLY. Use I/K/J/L/U/O to move the left arm.")
                    elif event["action"] == "select_right":
                        print("[SELECT] RIGHT ONLY. Use I/K/J/L/U/O to move the right arm.")
                    elif event["action"] == "select_both":
                        print("[SELECT] BOTH ARMS. X/Z move together; Y is mirrored by default.")
                    if event["action"] in POSE_ACTIONS:
                        if event["active_arm"] == "both":
                            dirty_sides.update(("left", "right"))
                        else:
                            dirty_sides.add(event["active_arm"])
                for command in state.drain_commands():
                    if command == "toggle_record":
                        if recorder.active:
                            recorder.stop(metadata)
                        else:
                            recorder.start(metadata)
                    elif command == "reset":
                        reset_to_ready(model, data, key_id)
                        q_target = {side: data.qpos[spec.qpos].copy() for side, spec in specs.items()}
                        state.reset_targets(model, data, specs)
                        dirty_sides.clear()
                        sample_time = data.time
                    elif command == "help":
                        print_controls()

                snapshot = state.snapshot()
                # IK is event-driven.  Re-solving against the current body
                # pose on every tick causes redundant 7-DOF arms to drift
                # even when the operator is only selecting an arm.
                for side in dirty_sides:
                    spec = specs[side]
                    q_target[side] = ik_step(
                        model,
                        data,
                        spec,
                        snapshot["target_pos"][side],
                        snapshot["target_quat"][side],
                        data.qpos[spec.qpos].copy(),
                    )
                dirty_sides.clear()
                if not snapshot["paused"]:
                    for _ in range(steps_per_tick):
                        action = apply_controller(
                            model, data, specs, finger_actuators, q_target, snapshot["gripper"]
                        )
                        mujoco.mj_step(model, data)
                else:
                    action = apply_controller(
                        model, data, specs, finger_actuators, q_target, snapshot["gripper"]
                    )

                reason = safety_reason(data, specs)
                if reason is not None and not snapshot["paused"]:
                    state.emergency_pause(reason)
                    print(f"[SAFETY] physics paused: {reason}; press R then P to recover")

                if recorder.active and data.time + 1e-9 >= sample_time:
                    target_pose = np.asarray(
                        [
                            np.concatenate([snapshot["target_pos"][side], snapshot["target_quat"][side]])
                            for side in ("left", "right")
                        ],
                        dtype=np.float64,
                    )
                    recorder.append(
                        model,
                        data,
                        action,
                        current_ee_pose(model, data, specs),
                        target_pose,
                        renderer,
                    )
                    sample_time += sample_period

                viewer.sync()
                wall_tick += tick_period
                time.sleep(max(0.0, wall_tick - time.perf_counter()))
    finally:
        if recorder.active:
            recorder.stop(metadata)
        if renderer is not None:
            renderer.close()
        if panel is not None:
            panel.close()


if __name__ == "__main__":
    main()
