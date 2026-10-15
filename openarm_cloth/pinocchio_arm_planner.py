# -*- coding: utf-8 -*-
"""Pinocchio kinematics and smooth joint-path helpers for OpenArm.

Pinocchio is used here for robot kinematics and joint-manifold interpolation.
MuJoCo remains the source of truth for contact, dynamics, rendering, and the
actual simulation state.  The project uses a user-writable Conda environment
because the system ``openpi`` environment is not writable on this machine.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_PINOCCHIO_PREFIX = Path.home() / ".conda" / "envs" / "openpi-pinocchio-py312"


def _import_pinocchio():
    """Import Pinocchio from the current interpreter or the user-level env."""

    try:
        return importlib.import_module("pinocchio")
    except ImportError as first_error:
        prefix = Path(
            os.environ.get("OPENARM_PINOCCHIO_PREFIX", str(DEFAULT_PINOCCHIO_PREFIX))
        ).expanduser()
        site_packages = prefix / "Lib" / "site-packages"
        if not site_packages.is_dir():
            raise ImportError(
                "Pinocchio 未安装。请设置 OPENARM_PINOCCHIO_PREFIX，或创建 "
                f"{DEFAULT_PINOCCHIO_PREFIX}"
            ) from first_error

        # Windows does not search a foreign Conda environment's DLL folder
        # after sys.path is changed. Keep the handles alive on the module.
        dll_handles = []
        for dll_dir in (prefix / "Library" / "bin", prefix / "DLLs"):
            if dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
                dll_handles.append(os.add_dll_directory(str(dll_dir)))
        _import_pinocchio._dll_handles = dll_handles
        sys.path.insert(0, str(site_packages))
        return importlib.import_module("pinocchio")


def _build_model_from_mjcf(pin, scene: Path):
    """Build a Pinocchio model from MJCF, ignoring MuJoCo-only keyframes."""

    # Pinocchio's MJCF parser intentionally builds the articulated robot, while
    # MuJoCo keyframes may also contain the free object's 7-DoF state. Removing
    # the keyframe avoids a misleading "keyframe size" parser error.
    root = ET.parse(scene).getroot()
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)
    with tempfile.TemporaryDirectory(prefix="openarm_pinocchio_") as temp_dir:
        stripped_scene = Path(temp_dir) / scene.name
        ET.ElementTree(root).write(
            stripped_scene,
            encoding="utf-8",
            xml_declaration=True,
        )
        return pin.buildModelFromMJCF(str(stripped_scene))


@dataclass(frozen=True)
class PinocchioSolveResult:
    q: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    iterations: int


class PinocchioArmPlanner:
    """Pinocchio Jacobian IK and smooth paths for one OpenArm chain."""

    def __init__(
        self,
        scene: Path,
        arm: str,
        q_range: np.ndarray,
        frame_name: str | None = None,
    ) -> None:
        self.pin = _import_pinocchio()
        self.scene = Path(scene).resolve()
        self.arm = arm
        self.model = _build_model_from_mjcf(self.pin, self.scene)
        self.data = self.pin.Data(self.model)
        self.frame_name = frame_name or f"openarm_{arm}_hand"
        if not self.model.existFrame(self.frame_name):
            raise ValueError(f"Pinocchio 中不存在末端坐标系: {self.frame_name}")
        self.frame_id = self.model.getFrameId(self.frame_name)
        self.joint_names = tuple(f"openarm_{arm}_joint{i}" for i in range(1, 8))
        if any(not self.model.existJointName(name) for name in self.joint_names):
            raise ValueError(f"Pinocchio 中缺少 {arm} 臂的 7 个关节")
        self.joint_ids = tuple(self.model.getJointId(name) for name in self.joint_names)
        self.q_indices = np.asarray(
            [self.model.joints[joint_id].idx_q for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.v_indices = np.asarray(
            [self.model.joints[joint_id].idx_v for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.q_lower = np.asarray(q_range[:, 0], dtype=np.float64)
        self.q_upper = np.asarray(q_range[:, 1], dtype=np.float64)
        if self.q_lower.shape != (7,) or self.q_upper.shape != (7,):
            raise ValueError("OpenArm 关节范围必须为 7x2")

    def _full_q(self, arm_q: np.ndarray) -> np.ndarray:
        arm_q = np.asarray(arm_q, dtype=np.float64)
        if arm_q.shape != (7,):
            raise ValueError(f"期望 7 个关节角，实际为 {arm_q.shape}")
        q = self.pin.neutral(self.model)
        q[self.q_indices] = np.clip(arm_q, self.q_lower, self.q_upper)
        return q

    def _update(self, q: np.ndarray):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id]

    def frame_pose(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the selected frame's world position and rotation."""

        transform = self._update(self._full_q(arm_q))
        return transform.translation.copy(), transform.rotation.copy()

    def solve_pose(
        self,
        seed: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        max_iterations: int = 180,
    ) -> PinocchioSolveResult:
        """Solve a bounded 6D frame pose with damped Pinocchio Jacobian IK."""

        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)
        if target_position.shape != (3,) or target_rotation.shape != (3, 3):
            raise ValueError("目标位姿必须是 position=(3,) 和 rotation=(3,3)")
        q = self._full_q(seed)
        target = self.pin.SE3(target_rotation, target_position)
        damping = 2e-4
        last_position_error = float("inf")
        last_rotation_error = float("inf")
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            current = self._update(q)
            local_error = self.pin.log6(current.inverse() * target).vector
            position_error = float(np.linalg.norm(target_position - current.translation))
            rotation_error = float(np.linalg.norm(local_error[3:]))
            last_position_error = position_error
            last_rotation_error = rotation_error
            if position_error < 1e-5 and rotation_error < 1e-4:
                break

            self.pin.computeJointJacobians(self.model, self.data, q)
            jacobian = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                self.pin.ReferenceFrame.LOCAL,
            )
            jacobian = jacobian[:, self.v_indices]
            # Pinocchio's LOCAL Jacobian and log6 error use the same frame.
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), local_error
            )
            step = np.clip(step, -0.12, 0.12)
            velocity = np.zeros(self.model.nv, dtype=np.float64)
            velocity[self.v_indices] = 0.65 * step
            q = self.pin.integrate(self.model, q, velocity)
            q[self.q_indices] = np.clip(q[self.q_indices], self.q_lower, self.q_upper)

        solved_q = q[self.q_indices].copy()
        current = self._update(q)
        final_error = self.pin.log6(current.inverse() * target).vector
        return PinocchioSolveResult(
            q=solved_q,
            position_error_m=float(np.linalg.norm(target_position - current.translation)),
            rotation_error_rad=float(np.linalg.norm(final_error[3:])),
            iterations=iterations,
        )

    def plan_joint_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        waypoint_count: int,
    ) -> list[np.ndarray]:
        """Interpolate one arm on Pinocchio's joint manifold with limit checks."""

        if waypoint_count < 2:
            return [np.asarray(goal, dtype=np.float64).copy()]
        start_q = self._full_q(start)
        goal_q = self._full_q(goal)
        delta = self.pin.difference(self.model, start_q, goal_q)
        path: list[np.ndarray] = []
        for alpha in np.linspace(0.0, 1.0, waypoint_count):
            q = self.pin.integrate(self.model, start_q, alpha * delta)
            arm_q = np.clip(q[self.q_indices], self.q_lower, self.q_upper)
            path.append(np.asarray(arm_q, dtype=np.float64).copy())
        path[-1] = np.asarray(goal, dtype=np.float64).copy()
        return path

