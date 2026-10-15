# -*- coding: utf-8 -*-
"""Grasp the red cube with wrist hand-eye geometry only.

This is a local simulation diagnostic.  It does not call OpenPI/DROID.  The
selected wrist RGB-D image is segmented by color, the depth pixels are
back-projected through the calibrated wrist hand-eye transform, and the
resulting cube center is used by a local IK/controller sequence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from auto_calibrate_left import matrix_to_quaternion, solve_kinematic_ik
from build_droid_object_scene import ensure_scene
from calibration_simulation import CameraPreview, body_transform, render_rgb, render_rgb_depth
from pinocchio_arm_planner import PinocchioArmPlanner
from teleop_towel_dataset import (
    FINGER_Q_NAMES,
    apply_controller,
    build_arm_specs,
    build_finger_actuators,
    id_for,
    quat_from_body,
    reset_to_ready,
)


ROOT = Path(__file__).resolve().parent
SIDES = ("left", "right")
CV_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class CubeObservation:
    center_world: np.ndarray
    center_surface_world: np.ndarray
    pixel_center: np.ndarray
    mask_pixels: int
    bbox: tuple[int, int, int, int]
    depth_median_m: float
    yaw_world_rad: float = 0.0


class GraspStepDisplay:
    """Show and save the intermediate state of the hand-eye grasp sequence."""

    def __init__(
        self,
        output_dir: Path,
        renderer: mujoco.Renderer,
        camera_name: str,
        preview: CameraPreview | None,
        report: dict[str, object],
        pause_s: float,
    ) -> None:
        self.renderer = renderer
        self.camera_name = camera_name
        self.preview = preview
        self.report = report
        self.pause_s = max(0.0, float(pause_s))
        self.step_index = 0
        self.image_dir = output_dir / f"steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, object]] = []
        report["step_images_dir"] = str(self.image_dir)
        report["steps"] = self.records

    @staticmethod
    def _add_sphere(scene: Any, position: np.ndarray, rgba: np.ndarray, label: str) -> None:
        if scene.ngeom + 2 > scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([0.009, 0.0, 0.0], dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        text = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            text,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3, dtype=np.float64),
            np.asarray(position, dtype=np.float64) + np.asarray([0.0, 0.0, 0.014]),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        text.label = label
        scene.ngeom += 1

    @staticmethod
    def _add_arrow(scene: Any, start: np.ndarray, end: np.ndarray, rgba: np.ndarray) -> None:
        if scene.ngeom + 1 > scene.maxgeom:
            return
        arrow = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            arrow,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.asarray([0.004, 0.004, 0.004], dtype=np.float64),
            np.asarray(start, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            arrow,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.004,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    def _update_markers(
        self,
        viewer: Any,
        stage: str,
        object_world: np.ndarray | None,
        center_world: np.ndarray | None,
        target_points: tuple[np.ndarray, np.ndarray] | None,
        finger_points: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        if viewer is None:
            return
        scene = viewer.user_scn
        scene.ngeom = 0
        if object_world is not None:
            self._add_sphere(scene, object_world, np.asarray([1.0, 0.1, 0.8, 1.0]), "actual_object")
        if center_world is not None:
            self._add_sphere(scene, center_world, np.asarray([1.0, 0.8, 0.1, 1.0]), "estimated_center")
        if target_points is not None:
            for index, point in enumerate(target_points, start=1):
                self._add_sphere(scene, point, np.asarray([0.1, 0.4, 1.0, 1.0]), f"target_{index}")
            if center_world is not None:
                for point in target_points:
                    self._add_arrow(scene, center_world, point, np.asarray([0.1, 0.4, 1.0, 0.8]))
        if finger_points is not None:
            for index, point in enumerate(finger_points, start=1):
                self._add_sphere(scene, point, np.asarray([0.1, 1.0, 0.2, 1.0]), f"finger_{index}")
        if center_world is not None:
            self._add_sphere(
                scene,
                center_world + np.asarray([0.0, 0.0, 0.045]),
                np.asarray([1.0, 1.0, 1.0, 1.0]),
                stage,
            )
        viewer.sync()

    def capture(
        self,
        stage: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer: Any,
        *,
        object_body: int | None = None,
        finger_geom_ids: tuple[int, int] | None = None,
        observation: CubeObservation | None = None,
        center_world: np.ndarray | None = None,
        target_points: tuple[np.ndarray, np.ndarray] | None = None,
        lines: tuple[str, ...] = (),
    ) -> None:
        mujoco.mj_forward(model, data)
        object_world = data.xpos[object_body].copy() if object_body is not None else None
        finger_points = None
        if finger_geom_ids is not None:
            finger_points = (
                data.geom_xpos[finger_geom_ids[0]].copy(),
                data.geom_xpos[finger_geom_ids[1]].copy(),
            )
        if center_world is None and observation is not None:
            center_world = observation.center_world

        rgb = render_rgb(self.renderer, data, self.camera_name)
        if observation is not None:
            x, y, width, height = observation.bbox
            cv2.rectangle(rgb, (x, y), (x + width, y + height), (0, 255, 255), 2)
            pixel = tuple(np.round(observation.pixel_center).astype(int).tolist())
            cv2.drawMarker(rgb, pixel, (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(rgb, f"stage: {stage}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        for index, line in enumerate(lines[:5], start=1):
            cv2.putText(
                rgb,
                str(line),
                (8, 20 + 20 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )
        image_path = self.image_dir / f"{self.step_index:02d}_{stage}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self.preview is not None:
            self.preview.show({self.camera_name: rgb})
            self.preview.poll()
        self._update_markers(viewer, stage, object_world, center_world, target_points, finger_points)
        record: dict[str, object] = {
            "index": self.step_index,
            "stage": stage,
            "image": str(image_path),
            "object_actual_world": object_world.tolist() if object_world is not None else None,
            "object_estimated_world": center_world.tolist() if center_world is not None else None,
            "target_finger_points_world": (
                [point.tolist() for point in target_points] if target_points is not None else None
            ),
            "finger_actual_world": (
                [point.tolist() for point in finger_points] if finger_points is not None else None
            ),
            "info": list(lines),
        }
        self.records.append(record)
        self.step_index += 1
        print(
            f"[中间结果] stage={stage} | object={np.round(object_world, 4).tolist() if object_world is not None else None} | "
            f"fingers={np.round(np.asarray(finger_points), 4).tolist() if finger_points is not None else None}"
        )
        if self.pause_s > 0.0 and (viewer is not None or self.preview is not None):
            time.sleep(self.pause_s)


def latest_handeye_report(arm: str) -> Path:
    reports = sorted((ROOT / "calibration" / f"auto_{arm}").glob(f"session_*/calibration_{arm}_wrist.json"))
    if not reports:
        raise FileNotFoundError(
            f"No {arm} wrist calibration report found under calibration/auto_{arm}; "
            "run the wrist calibration first."
        )
    return max(reports, key=lambda path: path.stat().st_mtime)


def load_handeye(path: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    key = f"T_{arm}_tcp_{arm}_wrist_opencv"
    if key not in report:
        raise KeyError(f"{key} is missing from {path}")
    handeye = np.asarray(report[key], dtype=np.float64)
    camera_matrix = np.asarray(report["camera_matrix"], dtype=np.float64)
    if handeye.shape != (4, 4) or camera_matrix.shape != (3, 3):
        raise ValueError(f"Invalid calibration shapes in {path}")
    return handeye, camera_matrix


def red_cube_mask(rgb: np.ndarray) -> np.ndarray:
    """Segment the saturated red MuJoCo cube, excluding the brown tabletop."""

    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    mask = (((hsv[..., 0] <= 10) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 30)).astype(
        np.uint8
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def estimate_cube_center(
    rgb: np.ndarray,
    depth: np.ndarray,
    tcp_world: np.ndarray,
    handeye_tcp_camera: np.ndarray,
    camera_matrix: np.ndarray,
    cube_half_size: float,
) -> CubeObservation:
    mask = red_cube_mask(rgb)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError("腕部相机没有检测到红色方块")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    width = int(stats[component, cv2.CC_STAT_WIDTH])
    height = int(stats[component, cv2.CC_STAT_HEIGHT])
    area = int(stats[component, cv2.CC_STAT_AREA])
    image_height, image_width = rgb.shape[:2]
    if area < 100:
        raise RuntimeError(f"红色区域过小，只有 {area} 像素")
    if x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1:
        raise RuntimeError("红色方块触及腕部相机图像边缘，请调整搜索位姿")

    ys, xs = np.nonzero(labels == component)
    values = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(values) & (values > 0.05) & (values < 1.0)
    if int(valid.sum()) < 100:
        raise RuntimeError("红色方块区域的有效深度不足")
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    values = values[valid]
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    camera_points = np.column_stack(
        ((xs - cx) * values / fx, (ys - cy) * values / fy, values)
    )
    world_camera = tcp_world @ handeye_tcp_camera
    world_points = (world_camera @ np.column_stack((camera_points, np.ones(len(camera_points)))).T).T[:, :3]

    # The visible top/side faces contain the cube boundaries.  Robust
    # percentiles reject a few depth edge pixels; the top face then gives the
    # center height after subtracting the known half-size.
    lower = np.percentile(world_points, 5.0, axis=0)
    upper = np.percentile(world_points, 95.0, axis=0)
    center_surface = 0.5 * (lower + upper)
    center = center_surface.copy()
    center[2] = float(np.percentile(world_points[:, 2], 90.0) - cube_half_size)
    # Recover the square's horizontal edges from the visible top surface.
    # PCA is ambiguous on a square; a minimum-area rectangle follows its
    # edges.  The resulting yaw is equivalent modulo 90 degrees.
    top_z = float(np.percentile(world_points[:, 2], 90.0))
    top_xy = world_points[world_points[:, 2] >= top_z - 0.0015, :2]
    if len(top_xy) < 50:
        raise RuntimeError("方块顶面深度不足，无法估计正面夹取方向")
    rectangle = cv2.minAreaRect(top_xy.astype(np.float32))
    yaw = float((np.deg2rad(rectangle[2]) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    pixel_center = np.asarray([(x + width / 2.0), (y + height / 2.0)], dtype=np.float64)
    return CubeObservation(
        center_world=center,
        center_surface_world=center_surface,
        pixel_center=pixel_center,
        mask_pixels=int(valid.sum()),
        bbox=(x, y, width, height),
        depth_median_m=float(np.median(values)),
        yaw_world_rad=yaw,
    )


def set_joint_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> None:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)


def solve_target_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    seed: np.ndarray,
    pin_planner: PinocchioArmPlanner | None = None,
    target_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    if pin_planner is not None:
        if target_rotation is None:
            raise ValueError("Pinocchio IK 需要目标旋转矩阵")
        result = pin_planner.solve_pose(seed, target_position, target_rotation)
        return result.q, result.position_error_m, result.rotation_error_rad
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    bounded_seed = np.clip(seed, spec.q_range[:, 0] + 1e-7, spec.q_range[:, 1] - 1e-7)
    result = solve_kinematic_ik(model, data, spec, target_position, target_quaternion, bounded_seed)
    set_joint_state(model, data, saved_qpos, saved_qvel, saved_ctrl)
    return result


def solve_gripper_midpoint_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    finger_body_ids: tuple[int, int],
    target_midpoint: np.ndarray,
    seed: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find a reachable arm pose whose open-finger midpoint reaches a target."""

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    bounded_seed = np.clip(seed, spec.q_range[:, 0] + 1e-7, spec.q_range[:, 1] - 1e-7)
    first_finger, second_finger = finger_body_ids

    def residual(q: np.ndarray) -> np.ndarray:
        data.qpos[spec.qpos] = q
        data.qvel[spec.dof] = 0.0
        mujoco.mj_forward(model, data)
        midpoint = 0.5 * (data.xpos[first_finger] + data.xpos[second_finger])
        # Mild seed regularization avoids an unnecessary elbow flip while
        # allowing the wrist orientation to change for a reachable grasp.
        return np.concatenate([(midpoint - target_midpoint) * 18.0, (q - bounded_seed) * 0.08])

    result = least_squares(
        residual,
        bounded_seed,
        bounds=(spec.q_range[:, 0], spec.q_range[:, 1]),
        max_nfev=1200,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    q_solution = np.asarray(result.x, dtype=np.float64)
    data.qpos[spec.qpos] = q_solution
    data.qvel[spec.dof] = 0.0
    mujoco.mj_forward(model, data)
    midpoint = 0.5 * (data.xpos[first_finger] + data.xpos[second_finger])
    error = float(np.linalg.norm(midpoint - target_midpoint))
    set_joint_state(model, data, saved_qpos, saved_qvel, saved_ctrl)
    return q_solution, error


_TRACE: list = []


def _trace_step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Debug-only: record the object-to-hand gap along the real stepping path."""

    if not os.environ.get("OPENARM_DEBUG_RENDER"):
        return
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "droid_object")
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand")
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp")
    if obj < 0 or hand < 0 or tcp < 0:
        return
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_grasp_weld")
    # Weld tracking error: the TCP-frame pose of the cube minus the relpose the
    # equality is trying to hold.  This is the constraint violation itself, not
    # a proxy for it.
    tcp_world = body_transform(data, tcp)
    actual = np.linalg.inv(tcp_world) @ body_transform(data, obj)
    want_p = np.asarray(model.eq_data[weld, 3:6], dtype=np.float64) if weld >= 0 else np.zeros(3)
    want_q = np.asarray(model.eq_data[weld, 6:10], dtype=np.float64) if weld >= 0 else np.array([1.0, 0.0, 0.0, 0.0])
    err_p = actual[:3, 3] - want_p
    rel_q = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rel_q, np.asarray(want_q, dtype=np.float64))
    rel_q = rel_q.reshape(3, 3)
    # Per-axis split of the orientation error: rotation vector of the residual.
    skew = rel_q.T @ actual[:3, :3]
    rotvec = 0.5 * np.array(
        [
            skew[2, 1] - skew[1, 2],
            skew[0, 2] - skew[2, 0],
            skew[1, 0] - skew[0, 1],
        ]
    )
    rotvec_deg = np.degrees(rotvec).tolist()
    cos_t = (np.trace(rel_q.T @ actual[:3, :3]) - 1.0) / 2.0
    err_a = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    _TRACE.append(
        (
            float(np.linalg.norm(data.xpos[obj] - data.xpos[hand])),
            float(data.xpos[obj][2]),
            float(data.xpos[hand][2]),
            *(
                np.asarray(data.xmat[hand], dtype=np.float64).reshape(3, 3).T
                @ (np.asarray(data.xpos[obj]) - np.asarray(data.xpos[hand]))
            ).tolist(),
            float(data.eq_active[weld]) if weld >= 0 else 0.0,
            *err_p.tolist(),
            err_a,
            *rotvec_deg,
        )
    )


def step_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    finger_actuators,
    q_target,
    gripper,
    duration_s: float,
    viewer=None,
) -> None:
    steps = max(1, int(round(duration_s / float(model.opt.timestep))))
    for index in range(steps):
        apply_controller(model, data, specs, finger_actuators, q_target, gripper)
        mujoco.mj_step(model, data)
        _trace_step(model, data)
        if viewer is not None and index % 10 == 0:
            if not viewer.is_running():
                raise RuntimeError("MuJoCo 窗口已关闭")
            viewer.sync()


def finger_joint_indices(model: mujoco.MjModel, arm: str) -> tuple[np.ndarray, np.ndarray]:
    qpos = []
    dof = []
    for name in FINGER_Q_NAMES[arm]:
        joint_id = id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos.append(int(model.jnt_qposadr[joint_id]))
        dof.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos, dtype=np.int32), np.asarray(dof, dtype=np.int32)


def set_kinematic_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    finger_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    arm_q: dict[str, np.ndarray],
    finger_q: dict[str, float],
) -> None:
    """Set the diagnostic pose while leaving the object dynamics enabled.

    This is intentionally a simulation-only isolation mode.  It removes the
    arm torque-controller tracking error from the hand-eye experiment; the
    free cube still responds to MuJoCo contact and gravity during each step.
    """

    data.ctrl[:] = 0.0
    for side, spec in specs.items():
        data.qpos[spec.qpos] = arm_q[side]
        data.qvel[spec.dof] = 0.0
        qpos_indices, dof_indices = finger_indices[side]
        data.qpos[qpos_indices] = float(finger_q[side])
        data.qvel[dof_indices] = 0.0
    mujoco.mj_forward(model, data)


def move_kinematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    finger_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    start_q: dict[str, np.ndarray],
    end_q: dict[str, np.ndarray],
    finger_q: dict[str, float],
    duration_s: float,
    viewer=None,
) -> None:
    steps = max(1, int(round(duration_s / float(model.opt.timestep))))
    for index in range(steps):
        alpha = (index + 1) / steps
        arm_q = {
            side: (1.0 - alpha) * start_q[side] + alpha * end_q[side]
            for side in specs
        }
        set_kinematic_state(model, data, specs, finger_indices, arm_q, finger_q)
        mujoco.mj_step(model, data)
        _trace_step(model, data)
        # Re-apply the commanded arm pose after contact integration.  The cube
        # remains dynamic, while the diagnostic arm pose follows the planned IK.
        set_kinematic_state(model, data, specs, finger_indices, arm_q, finger_q)
        if viewer is not None and index % 10 == 0:
            if not viewer.is_running():
                raise RuntimeError("MuJoCo 窗口已关闭")
            viewer.sync()


def hold_kinematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    finger_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    arm_q: dict[str, np.ndarray],
    finger_q: dict[str, float],
    duration_s: float,
    viewer=None,
) -> None:
    move_kinematic(
        model,
        data,
        specs,
        finger_indices,
        arm_q,
        arm_q,
        finger_q,
        duration_s,
        viewer,
    )


def close_kinematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    finger_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    arm_q: dict[str, np.ndarray],
    finger_start: float,
    finger_end: float,
    duration_s: float,
    viewer=None,
) -> None:
    """Close both fingers gradually while keeping the arm pose fixed.

    The previous diagnostic implementation changed finger qpos from open to
    closed in one simulation step.  That creates a large contact impulse when
    the fingers overlap the cube.  This ramp gives the contact solver a chance
    to respond over many small position updates.
    """

    steps = max(1, int(round(duration_s / float(model.opt.timestep))))
    for index in range(steps):
        alpha = (index + 1) / steps
        # Smoothstep reduces the initial and final closing velocity.
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        finger_value = (1.0 - smooth_alpha) * float(finger_start) + smooth_alpha * float(finger_end)
        set_kinematic_state(
            model,
            data,
            specs,
            finger_indices,
            arm_q,
            {side: finger_value for side in SIDES},
        )
        mujoco.mj_step(model, data)
        _trace_step(model, data)
        set_kinematic_state(
            model,
            data,
            specs,
            finger_indices,
            arm_q,
            {side: finger_value for side in SIDES},
        )
        if viewer is not None and index % 10 == 0:
            if not viewer.is_running():
                raise RuntimeError("MuJoCo 绐楀彛宸插叧闂?")
            viewer.sync()


def grasp_contact_status(
    data: mujoco.MjData,
    object_geom: int,
    finger_geom_ids: tuple[int, int],
) -> dict[str, object]:
    """Return object-to-left/right-finger contact information."""

    finger_set = {int(finger_geom_ids[0]), int(finger_geom_ids[1])}
    contacted: set[int] = set()
    distances: list[float] = []
    points: list[list[float]] = []
    normals: list[list[float]] = []
    for index in range(data.ncon):
        geom1 = int(data.contact[index].geom1)
        geom2 = int(data.contact[index].geom2)
        if int(object_geom) not in (geom1, geom2):
            continue
        other = geom2 if geom1 == int(object_geom) else geom1
        if other in finger_set:
            contacted.add(other)
            distances.append(float(data.contact[index].dist))
            points.append(np.asarray(data.contact[index].pos, dtype=np.float64).tolist())
            normals.append(np.asarray(data.contact[index].frame[:3], dtype=np.float64).tolist())
    return {
        "both_fingers_contact": len(contacted) == 2,
        "contacted_finger_geom_ids": sorted(contacted),
        "contact_count": len(contacted),
        "contact_distances_m": distances,
        "contact_points_world": points,
        "contact_normals_world": normals,
    }


def activate_grasp_weld(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,
    hand_body: int,
    object_body: int,
) -> dict[str, object]:
    """Latch the detected grasp for the kinematic diagnostic lift.

    The arm path is prescribed by IK, so contact friction alone is not a
    stable actuator model for a lift.  Once both finger collision meshes have
    contacted the cube, this temporary MuJoCo weld preserves the measured
    hand-to-object pose while the planned lift is executed.
    """

    equality_name = f"{arm}_grasp_weld"
    equality_id = id_for(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
    hand_world = body_transform(data, hand_body)
    object_world = body_transform(data, object_body)
    relative = np.linalg.inv(hand_world) @ object_world
    model.eq_data[equality_id, 3:6] = relative[:3, 3]
    model.eq_data[equality_id, 6:10] = matrix_to_quaternion(relative[:3, :3])
    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)
    return {
        "enabled": True,
        "equality_name": equality_name,
        "equality_id": int(equality_id),
        "relative_position_hand_frame": relative[:3, 3].tolist(),
        "relative_quaternion_hand_frame": model.eq_data[equality_id, 6:10].tolist(),
    }


def retarget_grasp_weld_object_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    equality_id: int,
    hand_body: int,
    object_body: int,
    object_position: np.ndarray,
    object_rotation: np.ndarray,
) -> None:
    """Move the latched cube to a planned pose and keep the weld consistent."""

    object_joint = id_for(model, mujoco.mjtObj.mjOBJ_JOINT, "droid_object_freejoint")
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    data.qpos[object_qpos : object_qpos + 3] = object_position
    data.qpos[object_qpos + 3 : object_qpos + 7] = matrix_to_quaternion(object_rotation)
    data.qvel[object_dof : object_dof + 6] = 0.0
    # xpos/xmat are FK caches from the last mj_forward; refresh them before
    # reading the object pose, otherwise the weld below is derived from the
    # pre-teleport pose and immediately fights the new qpos.
    mujoco.mj_kinematics(model, data)
    hand_world = body_transform(data, hand_body)
    object_world = body_transform(data, object_body)
    relative = np.linalg.inv(hand_world) @ object_world
    model.eq_data[equality_id, 3:6] = relative[:3, 3]
    model.eq_data[equality_id, 6:10] = matrix_to_quaternion(relative[:3, :3])
    mujoco.mj_forward(model, data)


def choose_contact_close_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    finger_indices: dict[str, tuple[np.ndarray, np.ndarray]],
    arm: str,
    object_geom: int,
    finger_geom_ids: tuple[int, int],
    finger_open: float,
    requested_close: float,
    target_span_m: float,
) -> tuple[float, dict[str, object]]:
    """Find a two-sided contact q near the planned finger-center span."""

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    arm_q = {side: data.qpos[spec.qpos].copy() for side, spec in specs.items()}
    qpos_indices, dof_indices = finger_indices[arm]
    # Sweep the closing range and keep the shallowest dual-contact grasp.
    # Penetration dominates the ranking: the finger-center span is only a soft
    # preference, since ranking by span first drives the fingers deeper into
    # the cube just to reach the nominal span.
    lower = max(0.0, min(float(requested_close), float(finger_open)) - 0.015)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(float(finger_open), lower, 80),
                np.asarray([float(requested_close), lower]),
            ]
        )
    )[::-1]
    fallback: tuple[tuple[float, float, float], float, dict[str, object]] | None = None
    dual_contact: list[tuple[tuple[float, float, float, float], float, dict[str, object]]] = []
    desired_span = float(target_span_m)
    for finger_q in candidates:
        set_kinematic_state(
            model,
            data,
            specs,
            finger_indices,
            arm_q,
            {side: float(finger_q) for side in SIDES},
        )
        status = grasp_contact_status(data, object_geom, finger_geom_ids)
        first = data.geom_xpos[finger_geom_ids[0]]
        second = data.geom_xpos[finger_geom_ids[1]]
        span_error = abs(float(np.linalg.norm(first - second)) - desired_span)
        score = (
            float(2 - int(status["contact_count"])),
            span_error,
            abs(float(finger_q) - float(requested_close)),
        )
        if fallback is None or score < fallback[0]:
            fallback = (score, float(finger_q), status)
        if bool(status["both_fingers_contact"]):
            penetration = max(
                [max(0.0, -float(distance)) for distance in status["contact_distances_m"]]
                or [0.0]
            )
            dual_contact.append(
                (
                    (
                        penetration,
                        span_error,
                        abs(float(finger_q) - float(requested_close)),
                        float(finger_q),
                    ),
                    float(finger_q),
                    status,
                )
            )
    if dual_contact:
        _score, chosen, chosen_status = min(dual_contact, key=lambda item: item[0])
    else:
        assert fallback is not None
        chosen = fallback[1]
        chosen_status = fallback[2]
    set_joint_state(model, data, saved_qpos, saved_qvel, saved_ctrl)
    return chosen, {
        **chosen_status,
        "requested_close_q": float(requested_close),
        "selected_close_q": float(chosen),
        "search_lower_q": float(lower),
        "target_span_m": desired_span,
        "dual_contact_candidates": len(dual_contact),
    }


def solve_gripper_geometry_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    finger_indices: tuple[int, int],
    finger_q_indices: tuple[np.ndarray, np.ndarray],
    target_points: tuple[np.ndarray, np.ndarray],
    finger_q: float,
    seed: np.ndarray,
) -> tuple[np.ndarray, float, tuple[np.ndarray, np.ndarray]]:
    """IK on the actual finger collision-geom centers, not the hand body."""

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    qpos_indices, dof_indices = finger_q_indices
    lower = spec.q_range[:, 0] + 1e-7
    upper = spec.q_range[:, 1] - 1e-7
    bounded_seed = np.clip(seed, lower, upper)
    first_geom, second_geom = finger_indices

    def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[spec.qpos] = q
        data.qvel[spec.dof] = 0.0
        data.qpos[qpos_indices] = finger_q
        data.qvel[dof_indices] = 0.0
        mujoco.mj_forward(model, data)
        return data.geom_xpos[first_geom].copy(), data.geom_xpos[second_geom].copy()

    def residual(q: np.ndarray) -> np.ndarray:
        first, second = evaluate(q)
        return np.concatenate(
            [
                (first - target_points[0]) * 12.0,
                (second - target_points[1]) * 12.0,
                (q - bounded_seed) * 0.015,
            ]
        )

    # The collision mesh has a narrow reachable orientation basin.  A few
    # deterministic seeds avoid selecting a shoulder/elbow branch that places
    # the hand body through the cube.
    seeds = [bounded_seed]
    for delta in (
        np.asarray([0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, -0.15, 0.0, 0.0, 0.0]),
        np.asarray([0.0, -0.12, 0.12, 0.0, 0.0, 0.0, 0.0]),
    ):
        seeds.append(np.clip(bounded_seed + delta, lower, upper))
    best = None
    for candidate in seeds:
        result = least_squares(
            residual,
            candidate,
            bounds=(lower, upper),
            max_nfev=700,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        first, second = evaluate(result.x)
        error = float(np.linalg.norm(np.concatenate([first - target_points[0], second - target_points[1]])))
        if best is None or error < best[0]:
            best = (error, np.asarray(result.x, dtype=np.float64), first, second)
    assert best is not None
    set_joint_state(model, data, saved_qpos, saved_qvel, saved_ctrl)
    return best[1], best[0], (best[2], best[3])


def solve_relaxed_gripper_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    finger_geom_ids: tuple[int, int],
    finger_q_indices: tuple[np.ndarray, np.ndarray],
    target_center: np.ndarray,
    axis_index: int,
    axis_sign: float,
    required_span: float,
    finger_q: float,
    seed: np.ndarray,
    reference_rotation: np.ndarray | None = None,
    orientation_weight: float = 0.75,
    perpendicular_weight: float = 6.0,
) -> tuple[np.ndarray, float, tuple[np.ndarray, np.ndarray], dict[str, float]]:
    """Project a desired grasp onto a reachable finger pose.

    Exact fingertip targets are often impossible for this 7-DoF OpenArm
    model.  This solver keeps the fingertip midpoint on the hand-eye target,
    aligns the finger pair with one candidate world axis, and lets the span
    project to the nearest reachable value.  The returned score is a planning
    score, not an exact Cartesian IK residual.
    """

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    qpos_indices, dof_indices = finger_q_indices
    lower = spec.q_range[:, 0] + 1e-7
    upper = spec.q_range[:, 1] - 1e-7
    bounded_seed = np.clip(seed, lower, upper)
    first_geom, second_geom = finger_geom_ids
    perpendicular = [index for index in range(3) if index != axis_index]

    def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[spec.qpos] = q
        data.qvel[spec.dof] = 0.0
        data.qpos[qpos_indices] = finger_q
        data.qvel[dof_indices] = 0.0
        mujoco.mj_forward(model, data)
        return data.geom_xpos[first_geom].copy(), data.geom_xpos[second_geom].copy()

    def residual(q: np.ndarray) -> np.ndarray:
        first, second = evaluate(q)
        midpoint = 0.5 * (first + second)
        separation = first - second
        terms = [
            (midpoint - target_center) * 20.0,
            separation[perpendicular] * float(perpendicular_weight),
            np.asarray([separation[axis_index] - axis_sign * required_span]) * 3.0,
            (q - bounded_seed) * 0.015,
        ]
        if reference_rotation is not None:
            current_rotation = data.geom_xmat[first_geom].reshape(3, 3)
            terms.append(
                (current_rotation - np.asarray(reference_rotation, dtype=np.float64)).ravel()
                * float(orientation_weight)
            )
        return np.concatenate(terms)

    seeds = [bounded_seed]
    for delta in (
        np.asarray([0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, -0.15, 0.0, 0.0, 0.0]),
        np.asarray([0.0, -0.12, 0.12, 0.0, 0.0, 0.0, 0.0]),
    ):
        seeds.append(np.clip(bounded_seed + delta, lower, upper))
    best = None
    for candidate in seeds:
        result = least_squares(
            residual,
            candidate,
            bounds=(lower, upper),
            max_nfev=700,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        first, second = evaluate(result.x)
        midpoint = 0.5 * (first + second)
        separation = first - second
        midpoint_error = float(np.linalg.norm(midpoint - target_center))
        perpendicular_error = float(np.linalg.norm(separation[perpendicular]))
        span_error = float(abs(abs(separation[axis_index]) - required_span))
        score = midpoint_error + 0.35 * perpendicular_error + 0.18 * span_error
        candidate_result = (
            score,
            np.asarray(result.x, dtype=np.float64),
            first,
            second,
            midpoint_error,
            perpendicular_error,
            float(separation[axis_index]),
        )
        if best is None or score < best[0]:
            best = candidate_result
    assert best is not None
    set_joint_state(model, data, saved_qpos, saved_qvel, saved_ctrl)
    return (
        best[1],
        best[0],
        (best[2], best[3]),
        {
            "midpoint_error_m": best[4],
            "perpendicular_error_m": best[5],
            "separation_m": best[6],
        },
    )


def solve_gripper_contact_proxy_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    finger_geom_ids: tuple[int, int],
    finger_q_indices: tuple[np.ndarray, np.ndarray],
    contact_proxy_local: tuple[np.ndarray, np.ndarray],
    target_center: np.ndarray,
    axis_index: int,
    axis_sign: float,
    required_span: float,
    finger_q: float,
    seed: np.ndarray,
    reference_rotation: np.ndarray | None = None,
    orientation_weight: float = 5.0,
    multi_start: bool = True,
    closing_direction: np.ndarray | None = None,
    approach_down_weight: float = 0.0,
) -> tuple[np.ndarray, float, tuple[np.ndarray, np.ndarray], dict[str, float]]:
    """IK on calibrated physical pad-contact points, not mesh origins.

    The imported finger collision meshes have origins several centimetres
    behind their useful contact patches.  A fixed world-frame correction for
    that lever arm changes meaning when the wrist rotates.  These two local
    points stay attached to the finger geoms, so the correction rotates with
    the hand and remains valid across the workspace.
    """
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    qpos_indices, dof_indices = finger_q_indices
    lower = spec.q_range[:, 0] + 1e-7
    upper = spec.q_range[:, 1] - 1e-7
    bounded_seed = np.clip(seed, lower, upper)
    direction = np.zeros(3, dtype=np.float64)
    direction[axis_index] = float(axis_sign)
    if closing_direction is not None:
        direction = np.asarray(closing_direction, dtype=float).copy()
        direction /= np.linalg.norm(direction)
    desired = (
        np.asarray(target_center, dtype=np.float64) + 0.5 * required_span * direction,
        np.asarray(target_center, dtype=np.float64) - 0.5 * required_span * direction,
    )

    def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[spec.qpos] = q
        data.qvel[spec.dof] = 0.0
        data.qpos[qpos_indices] = finger_q
        data.qvel[dof_indices] = 0.0
        mujoco.mj_forward(model, data)
        points = []
        for geom_id, local in zip(finger_geom_ids, contact_proxy_local):
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points.append(data.geom_xpos[geom_id] + rotation @ local)
        return points[0], points[1]

    def residual(q: np.ndarray) -> np.ndarray:
        first, second = evaluate(q)
        terms = [
            (first - desired[0]) * 20.0,
            (second - desired[1]) * 20.0,
            (q - bounded_seed) * 0.20,
        ]
        if reference_rotation is not None:
            current_rotation = data.geom_xmat[finger_geom_ids[0]].reshape(3, 3)
            terms.append(
                (current_rotation - np.asarray(reference_rotation, dtype=np.float64)).ravel()
                * float(orientation_weight)
            )
        if approach_down_weight:
            distal = data.geom_xmat[finger_geom_ids[0]].reshape(3, 3)[:, 2]
            terms.append((distal - np.asarray([0.0, 0.0, -1.0]))
                         * float(approach_down_weight))
        return np.concatenate(terms)

    seeds = [bounded_seed]
    if multi_start:
        for delta in (
            np.asarray([0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, -0.15, 0.0, 0.0, 0.0]),
            np.asarray([0.0, -0.12, 0.12, 0.0, 0.0, 0.0, 0.0]),
        ):
            seeds.append(np.clip(bounded_seed + delta, lower, upper))
    best = None
    for candidate in seeds:
        result = least_squares(
            residual, candidate, bounds=(lower, upper), max_nfev=700,
            xtol=1e-9, ftol=1e-9, gtol=1e-9,
        )
        first, second = evaluate(result.x)
        error = float(np.linalg.norm(np.concatenate([
            first - desired[0], second - desired[1]
        ])))
        if best is None or error < best[0]:
            best = (error, np.asarray(result.x, dtype=np.float64), first, second)
    assert best is not None
    midpoint = 0.5 * (best[2] + best[3])
    separation = best[2] - best[3]
    perpendicular = [index for index in range(3) if index != axis_index]
    metrics = {
        "midpoint_error_m": float(np.linalg.norm(midpoint - target_center)),
        "perpendicular_error_m": float(np.linalg.norm(
            separation - np.dot(separation, direction) * direction)),
        "separation_m": float(np.dot(separation, direction)),
    }
    set_joint_state(model, data, saved_qpos, saved_qvel, saved_ctrl)
    return best[1], best[0], (best[2], best[3]), metrics


def search_reachable_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    finger_geom_ids: tuple[int, int],
    finger_q_indices: tuple[np.ndarray, np.ndarray],
    target_center: np.ndarray,
    required_span: float,
    finger_q: float,
    seed: np.ndarray,
    axis_indices=(0, 1, 2),
) -> tuple[np.ndarray, float, tuple[np.ndarray, np.ndarray], dict[str, float | int]]:
    """Try the six signed world axes and choose the best reachable grasp."""

    best = None
    candidates = []
    for axis_index in axis_indices:
        for axis_sign in (-1.0, 1.0):
            q, score, points, metrics = solve_relaxed_gripper_q(
                model,
                data,
                spec,
                finger_geom_ids,
                finger_q_indices,
                target_center,
                axis_index,
                axis_sign,
                required_span,
                finger_q,
                seed,
            )
            candidate = (score, q, points, {**metrics, "axis_index": axis_index, "axis_sign": axis_sign})
            candidates.append({
                "axis_index": axis_index,
                "axis_sign": axis_sign,
                "score_m": float(score),
                **metrics,
            })
            if best is None or score < best[0]:
                best = candidate
    assert best is not None
    best[3]["candidates"] = candidates
    return best[1], best[0], best[2], best[3]


def run_kinematic_handeye_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    spec,
    finger_q_indices,
    finger_geom_ids: tuple[int, int],
    observation: CubeObservation,
    object_body: int,
    object_geom: int,
    args,
    viewer,
    report: dict[str, object],
) -> None:
    """Run the hand-eye-only grasp diagnostic after RGB-D localization."""

    arm = args.arm
    finger_open = float(args.gripper_open)
    finger_close = float(args.gripper_close)
    arm_q = {side: data.qpos[other.qpos].copy() for side, other in specs.items()}
    initial_object_z = float(data.xpos[object_body, 2])
    set_kinematic_state(
        model,
        data,
        specs,
        finger_q_indices,
        arm_q,
        {side: finger_open for side in SIDES},
    )

    def move_to(target: np.ndarray, duration_s: float, finger_value: float) -> None:
        nonlocal arm_q
        end_q = {side: values.copy() for side, values in arm_q.items()}
        end_q[arm] = target.copy()
        move_kinematic(
            model,
            data,
            specs,
            finger_q_indices,
            arm_q,
            end_q,
            {side: finger_value for side in SIDES},
            duration_s,
            viewer,
        )
        arm_q = end_q

    def hold(finger_value: float, duration_s: float) -> None:
        hold_kinematic(
            model,
            data,
            specs,
            finger_q_indices,
            arm_q,
            {side: finger_value for side in SIDES},
            duration_s,
            viewer,
        )

    center = observation.center_world
    span = np.asarray([args.grasp_span, 0.0, 0.0], dtype=np.float64)
    approach_center = center + np.asarray([0.0, 0.0, args.approach_height])
    approach_points = (approach_center + span, approach_center - span)
    grasp_points = (center + span, center - span)
    q_approach, approach_error, _ = solve_gripper_geometry_q(
        model,
        data,
        spec,
        finger_geom_ids,
        finger_q_indices[arm],
        approach_points,
        finger_open,
        arm_q[arm],
    )
    q_grasp, grasp_error, _ = solve_gripper_geometry_q(
        model,
        data,
        spec,
        finger_geom_ids,
        finger_q_indices[arm],
        grasp_points,
        finger_open,
        q_approach,
    )
    report["grasp_plan"] = {
        "finger_collision_geoms": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in finger_geom_ids
        ],
        "approach_points_world": [point.tolist() for point in approach_points],
        "grasp_points_world": [point.tolist() for point in grasp_points],
        "approach_ik_error_m": approach_error,
        "grasp_ik_error_m": grasp_error,
        "finger_open_q": finger_open,
        "finger_close_q": finger_close,
    }
    print(
        f"[手眼抓取] 接近 IK | 指尖目标={np.round(approach_points[0], 4).tolist()}, "
        f"{np.round(approach_points[1], 4).tolist()} | 误差={approach_error * 1000:.1f}mm"
    )
    print(
        f"[手眼抓取] 抓取 IK | 指尖目标={np.round(grasp_points[0], 4).tolist()}, "
        f"{np.round(grasp_points[1], 4).tolist()} | 误差={grasp_error * 1000:.1f}mm"
    )

    worst_ik_error = max(approach_error, grasp_error)
    if worst_ik_error > args.max_grasp_ik_error and not args.force_grasp:
        report["result"] = {
            "success": False,
            "reason": "fingertip_geometry_unreachable",
            "worst_ik_error_m": worst_ik_error,
            "max_allowed_ik_error_m": args.max_grasp_ik_error,
            "object_world_at_abort": data.xpos[object_body].copy().tolist(),
            "motion_mode": "kinematic_diagnostic",
        }
        print(
            f"[手眼抓取] 安全停止：指尖几何 IK 最大误差={worst_ik_error * 1000:.1f}mm "
            f"> 阈值={args.max_grasp_ik_error * 1000:.1f}mm，未执行闭爪和抬升。"
        )
        return

    move_to(q_approach, args.approach_s, finger_open)
    print(
        f"[手眼抓取] 接近完成 | 指尖={np.round([data.geom_xpos[finger_geom_ids[0]], data.geom_xpos[finger_geom_ids[1]]], 4).tolist()} "
        f"| 方块={np.round(data.xpos[object_body], 4).tolist()}"
    )
    move_to(q_grasp, args.approach_s, finger_open)
    print(
        f"[手眼抓取] 抓取位姿完成 | 指尖={np.round([data.geom_xpos[finger_geom_ids[0]], data.geom_xpos[finger_geom_ids[1]]], 4).tolist()} "
        f"| 方块={np.round(data.xpos[object_body], 4).tolist()}"
    )
    print(f"[手眼抓取] 轻闭夹爪 | finger_q={finger_close:.3f}")
    close_kinematic(
        model,
        data,
        specs,
        finger_q_indices,
        arm_q,
        finger_open,
        finger_close,
        args.close_s,
        viewer,
    )
    print(
        f"[手眼抓取] 夹爪完成 | 指尖={np.round([data.geom_xpos[finger_geom_ids[0]], data.geom_xpos[finger_geom_ids[1]]], 4).tolist()} "
        f"| 方块={np.round(data.xpos[object_body], 4).tolist()} | contacts={data.ncon}"
    )

    closed_points = (
        data.geom_xpos[finger_geom_ids[0]].copy(),
        data.geom_xpos[finger_geom_ids[1]].copy(),
    )
    lift_points = (
        closed_points[0] + np.asarray([0.0, 0.0, args.lift_height]),
        closed_points[1] + np.asarray([0.0, 0.0, args.lift_height]),
    )
    q_lift, lift_error, _ = solve_gripper_geometry_q(
        model,
        data,
        spec,
        finger_geom_ids,
        finger_q_indices[arm],
        lift_points,
        finger_close,
        arm_q[arm],
    )
    print(
        f"[手眼抓取] 抬升 IK | 指尖目标={np.round(lift_points[0], 4).tolist()}, "
        f"{np.round(lift_points[1], 4).tolist()} | 误差={lift_error * 1000:.1f}mm"
    )
    move_to(q_lift, args.lift_s, finger_close)
    hold(finger_close, 0.3)
    mujoco.mj_forward(model, data)
    final_object = data.xpos[object_body].copy()
    finger_midpoint = 0.5 * (data.geom_xpos[finger_geom_ids[0]] + data.geom_xpos[finger_geom_ids[1]])
    contact = any(
        object_geom in {int(data.contact[i].geom1), int(data.contact[i].geom2)}
        and bool({finger_geom_ids[0], finger_geom_ids[1]}.intersection({int(data.contact[i].geom1), int(data.contact[i].geom2)}))
        for i in range(data.ncon)
    )
    lifted = float(final_object[2]) > initial_object_z + 0.025
    near_hand = float(np.linalg.norm(final_object - finger_midpoint)) < 0.18
    success = bool(lifted and near_hand)
    report["result"] = {
        "success": success,
        "object_initial_z_after_observation_m": initial_object_z,
        "object_final_world": final_object.tolist(),
        "lift_m": float(final_object[2] - initial_object_z),
        "near_finger_midpoint": near_hand,
        "finger_midpoint_world": finger_midpoint.tolist(),
        "contact": bool(contact),
        "motion_mode": "kinematic_diagnostic",
    }
    print(
        f"[手眼抓取] 结果={'成功' if success else '失败'} | 方块最终XYZ={np.round(final_object, 4).tolist()} | "
        f"抬升={final_object[2] - initial_object_z:.4f}m | 接触={contact} | 距指尖中点={np.linalg.norm(final_object - finger_midpoint):.3f}m"
    )


def run_reachable_handeye_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    specs,
    spec,
    finger_q_indices,
    finger_geom_ids: tuple[int, int],
    observation: CubeObservation,
    object_body: int,
    object_geom: int,
    args,
    viewer,
    report: dict[str, object],
    pin_planner: PinocchioArmPlanner | None = None,
    step_display: GraspStepDisplay | None = None,
) -> None:
    """Reachability-projected grasp followed by a vertical Cartesian descent."""

    arm = args.arm
    finger_open = float(args.gripper_open)
    finger_close = float(args.gripper_close)
    arm_q = {side: data.qpos[other.qpos].copy() for side, other in specs.items()}
    initial_object_z = float(data.xpos[object_body, 2])
    set_kinematic_state(
        model,
        data,
        specs,
        finger_q_indices,
        arm_q,
        {side: finger_open for side in SIDES},
    )

    def move_to(target_q: np.ndarray, duration_s: float, finger_value: float) -> None:
        nonlocal arm_q
        if pin_planner is not None:
            path = pin_planner.plan_joint_path(
                arm_q[arm],
                target_q,
                max(2, int(args.path_waypoints)),
            )
            segment_duration = duration_s / max(len(path) - 1, 1)
            for path_q in path[1:]:
                end_q = {side: values.copy() for side, values in arm_q.items()}
                end_q[arm] = path_q.copy()
                move_kinematic(
                    model,
                    data,
                    specs,
                    finger_q_indices,
                    arm_q,
                    end_q,
                    {side: finger_value for side in SIDES},
                    segment_duration,
                    viewer,
                )
                arm_q = end_q
            return
        end_q = {side: values.copy() for side, values in arm_q.items()}
        end_q[arm] = target_q.copy()
        move_kinematic(
            model,
            data,
            specs,
            finger_q_indices,
            arm_q,
            end_q,
            {side: finger_value for side in SIDES},
            duration_s,
            viewer,
        )
        arm_q = end_q

    def hold(finger_value: float, duration_s: float) -> None:
        hold_kinematic(
            model,
            data,
            specs,
            finger_q_indices,
            arm_q,
            {side: finger_value for side in SIDES},
            duration_s,
            viewer,
        )

    center = observation.center_world
    # Full finger separation required to clear the shrunken cube, with a
    # small margin.  The old solver tried to impose an exact 70 mm pair of
    # world points; here the span is a soft constraint and is projected to a
    # reachable value.
    required_span = max(2.0 * args.cube_half_size + 0.004, args.finger_center_span)
    q_grasp, grasp_score, grasp_points, grasp_metrics = search_reachable_grasp(
        model,
        data,
        spec,
        finger_geom_ids,
        finger_q_indices[arm],
        center,
        required_span,
        finger_open,
        arm_q[arm],
    )
    axis_index = int(grasp_metrics["axis_index"])
    axis_sign = float(grasp_metrics["axis_sign"])
    axis_vector = np.zeros(3, dtype=np.float64)
    axis_vector[axis_index] = axis_sign
    axis_name = ("x", "y", "z")[axis_index]

    if args.skip_approach_ik:
        # Diagnostic mode: deliberately remove the separately solved high
        # approach pose and use the grasp pose directly.
        q_approach = q_grasp.copy()
        approach_score = grasp_score
        approach_points = (grasp_points[0].copy(), grasp_points[1].copy())
        approach_metrics = dict(grasp_metrics)
    else:
        approach_center = center + np.asarray([0.0, 0.0, args.approach_height])
        q_approach, approach_score, approach_points, approach_metrics = solve_relaxed_gripper_q(
            model,
            data,
            spec,
            finger_geom_ids,
            finger_q_indices[arm],
            approach_center,
            axis_index,
            axis_sign,
            required_span,
            finger_open,
            arm_q[arm],
        )
    report["grasp_plan"] = {
        "planner": (
            "pinocchio_manifold_path_with_reachable_axis_search"
            if pin_planner is not None
            else "reachable_axis_search_cartesian_z_descent"
        ),
        "pinocchio": (
            {
                "version": str(pin_planner.pin.__version__),
                "frame": pin_planner.frame_name,
                "waypoints_per_segment": int(args.path_waypoints),
            }
            if pin_planner is not None
            else None
        ),
        "grasp_axis": axis_name,
        "grasp_axis_sign": axis_sign,
        "skip_approach_ik": bool(args.skip_approach_ik),
        "approach_height_m": float(args.approach_height),
        "descent_min_height_m": float(args.descent_min_height),
        "required_full_span_m": required_span,
        "approach_reachable_points_world": [point.tolist() for point in approach_points],
        "grasp_reachable_points_world": [point.tolist() for point in grasp_points],
        "approach_planning_score_m": approach_score,
        "grasp_planning_score_m": grasp_score,
        "approach_metrics": approach_metrics,
        "grasp_metrics": grasp_metrics,
        "finger_open_q": finger_open,
        "finger_close_q": finger_close,
    }
    print(
        f"[手眼抓取] 可达姿态搜索 | 夹爪轴={axis_name}{axis_sign:+.0f} | "
        f"中点误差={grasp_metrics['midpoint_error_m'] * 1000:.1f}mm | "
        f"指尖间距={abs(grasp_metrics['separation_m']) * 1000:.1f}mm | "
        f"规划评分={grasp_score * 1000:.1f}mm"
    )
    print(
        f"[手眼抓取] 接近姿态 | 指尖={np.round([approach_points[0], approach_points[1]], 4).tolist()} | "
        f"规划评分={approach_score * 1000:.1f}mm"
    )
    if step_display is not None:
        step_display.capture(
            "grasp_candidate",
            model,
            data,
            viewer,
            object_body=object_body,
            finger_geom_ids=finger_geom_ids,
            center_world=center,
            target_points=(grasp_points[0].copy(), grasp_points[1].copy()),
            lines=(
                f"axis={axis_name}{axis_sign:+.0f}",
                f"required_span={required_span * 1000:.1f}mm",
                f"actual_span={abs(grasp_metrics['separation_m']) * 1000:.1f}mm",
                f"score={grasp_score * 1000:.1f}mm",
            ),
        )
    if max(approach_score, grasp_score) > args.max_grasp_ik_error and not args.force_grasp:
        report["result"] = {
            "success": False,
            "reason": "reachable_grasp_score_too_high",
            "max_planning_score_m": max(approach_score, grasp_score),
            "max_allowed_score_m": args.max_grasp_ik_error,
            "object_world_at_abort": data.xpos[object_body].copy().tolist(),
            "motion_mode": (
                "pinocchio_manifold_path_with_reachable_axis_search"
                if pin_planner is not None
                else "reachable_axis_search_cartesian_z_descent"
            ),
        }
        print(
            f"[手眼抓取] 安全停止：可达姿态规划评分={max(approach_score, grasp_score) * 1000:.1f}mm "
            f"> 阈值={args.max_grasp_ik_error * 1000:.1f}mm，未执行闭爪和抬升。"
        )
        return

    # Normal mode moves to a high approach pose and descends through Cartesian-Z
    # waypoints. Diagnostic skip mode moves directly to q_grasp.
    move_to(q_approach, max(args.approach_s * 0.6, 0.2), finger_open)
    if step_display is not None:
        step_display.capture(
            "direct_grasp_pose" if args.skip_approach_ik else "approach_pose",
            model,
            data,
            viewer,
            object_body=object_body,
            finger_geom_ids=finger_geom_ids,
            observation=observation,
            center_world=center,
            target_points=(approach_points[0].copy(), approach_points[1].copy()),
            lines=(f"score={approach_score * 1000:.1f}mm", "gripper=open"),
        )
    descent = []
    if not args.skip_approach_ik:
        seed = q_approach
        waypoint_count = 8
        for height in np.linspace(args.approach_height, args.descent_min_height, waypoint_count)[1:]:
            target = center + np.asarray([0.0, 0.0, height])
            q_step, score, points, metrics = solve_relaxed_gripper_q(
                model,
                data,
                spec,
                finger_geom_ids,
                finger_q_indices[arm],
                target,
                axis_index,
                axis_sign,
                required_span,
                finger_open,
                seed,
            )
            descent.append((q_step, score, points, metrics, float(height)))
            seed = q_step
    for descent_index, (q_step, _score, _points, _metrics, _height) in enumerate(descent, start=1):
        move_to(q_step, max(args.approach_s / max(len(descent), 1), 0.05), finger_open)
        if step_display is not None:
            step_display.capture(
                f"descent_{descent_index:02d}",
                model,
                data,
                viewer,
                object_body=object_body,
                finger_geom_ids=finger_geom_ids,
                observation=observation,
                center_world=center,
                target_points=(_points[0].copy(), _points[1].copy()),
                lines=(
                    f"score={_score * 1000:.1f}mm",
                    f"target_height={_height * 1000:.1f}mm",
                    f"step={descent_index}/{len(descent)}",
                    "gripper=open",
                ),
            )
    q_grasp = arm_q[arm].copy()
    requested_finger_close = finger_close
    finger_close, close_plan = choose_contact_close_q(
        model,
        data,
        specs,
        finger_q_indices,
        arm,
        object_geom,
        finger_geom_ids,
        finger_open,
        requested_finger_close,
        required_span,
    )
    if isinstance(report.get("grasp_plan"), dict):
        report["grasp_plan"]["requested_close_q"] = float(requested_finger_close)
        report["grasp_plan"]["selected_close_q"] = float(finger_close)
        report["grasp_plan"]["preclose_contact_search"] = close_plan
    print(
        f"[grasp contact] requested_q={requested_finger_close:.4f} "
        f"selected_q={finger_close:.4f} "
        f"preclose_fingers={close_plan['contact_count']}"
    )
    print(
        f"[手眼抓取] 笛卡尔下降完成 | 指尖={np.round([data.geom_xpos[finger_geom_ids[0]], data.geom_xpos[finger_geom_ids[1]]], 4).tolist()} | "
        f"方块={np.round(data.xpos[object_body], 4).tolist()}"
    )

    print(f"[手眼抓取] 轻闭夹爪 | finger_q={finger_close:.3f}")
    close_kinematic(
        model,
        data,
        specs,
        finger_q_indices,
        arm_q,
        finger_open,
        finger_close,
        args.close_s,
        viewer,
    )
    print(
        f"[手眼抓取] 夹爪完成 | 指尖={np.round([data.geom_xpos[finger_geom_ids[0]], data.geom_xpos[finger_geom_ids[1]]], 4).tolist()} | "
        f"方块={np.round(data.xpos[object_body], 4).tolist()} | contacts={data.ncon}"
    )

    closed_points = (
        data.geom_xpos[finger_geom_ids[0]].copy(),
        data.geom_xpos[finger_geom_ids[1]].copy(),
    )
    close_contact = grasp_contact_status(data, object_geom, finger_geom_ids)
    if isinstance(report.get("grasp_plan"), dict):
        report["grasp_plan"]["postclose_contact"] = close_contact
    if step_display is not None:
        step_display.capture(
            "close_gripper",
            model,
            data,
            viewer,
            object_body=object_body,
            finger_geom_ids=finger_geom_ids,
            observation=observation,
            center_world=center,
            target_points=(closed_points[0].copy(), closed_points[1].copy()),
            lines=(f"finger_q={finger_close:.3f}", f"contacts={data.ncon}"),
        )
    if not bool(close_contact["both_fingers_contact"]) and not args.force_grasp:
        report["result"] = {
            "success": False,
            "reason": "no_dual_finger_contact_after_close",
            "object_world_at_abort": data.xpos[object_body].copy().tolist(),
            "postclose_contact": close_contact,
            "closed_points_world": [point.tolist() for point in closed_points],
            "motion_mode": (
                "pinocchio_manifold_path_with_reachable_axis_search"
                if pin_planner is not None
                else "reachable_axis_search_cartesian_z_descent"
            ),
        }
        print(
            f"[grasp contact] SAFE STOP: dual-finger contact not confirmed "
            f"(contacted={close_contact['contact_count']}); lift skipped"
        )
        return
    grasp_weld = {"enabled": False, "reason": "disabled_by_flag" if args.no_grasp_latch else "no_dual_contact"}
    if bool(close_contact["both_fingers_contact"]) and not args.no_grasp_latch:
        grasp_weld = activate_grasp_weld(model, data, arm, spec.body_id, object_body)
        if isinstance(report.get("grasp_plan"), dict):
            report["grasp_plan"]["grasp_weld"] = grasp_weld
        print(
            f"[grasp contact] hold={grasp_weld['equality_name']} "
            "enabled after dual-finger contact"
        )
    object_close_world = data.xpos[object_body].copy()
    object_close_rotation = body_transform(data, object_body)[:3, :3].copy()
    lift_center = center + np.asarray([0.0, 0.0, args.lift_height])
    q_lift, lift_score, _lift_points, lift_metrics = solve_relaxed_gripper_q(
        model,
        data,
        spec,
        finger_geom_ids,
        finger_q_indices[arm],
        lift_center,
        axis_index,
        axis_sign,
        required_span,
        finger_close,
        arm_q[arm],
    )
    _lift_points = (_lift_points[0].copy(), _lift_points[1].copy())
    held_object_target = (
        object_close_world + np.asarray([0.0, 0.0, args.lift_height])
        if bool(grasp_weld["enabled"])
        else None
    )
    print(
        f"[手眼抓取] 抬升姿态 | 目标={np.round(lift_center, 4).tolist()} | "
        f"规划评分={lift_score * 1000:.1f}mm"
    )
    move_to(q_lift, args.lift_s, finger_close)
    hold(finger_close, 0.3)
    if held_object_target is not None:
        retarget_grasp_weld_object_pose(
            model,
            data,
            int(grasp_weld["equality_id"]),
            spec.body_id,
            object_body,
            held_object_target,
            object_close_rotation,
        )
        lift_metrics["held_object_target_world"] = held_object_target.tolist()
        lift_metrics["held_object_target_error_m"] = float(
            np.linalg.norm(data.xpos[object_body] - held_object_target)
        )
    mujoco.mj_forward(model, data)
    if step_display is not None:
        step_display.capture(
            "lift_pose",
            model,
            data,
            viewer,
            object_body=object_body,
            finger_geom_ids=finger_geom_ids,
            observation=observation,
            center_world=lift_center,
            target_points=(_lift_points[0].copy(), _lift_points[1].copy()),
            lines=(f"score={lift_score * 1000:.1f}mm", "gripper=closed"),
        )
    # --- TEMP DEBUG RENDER (set OPENARM_DEBUG_RENDER=1) ---
    if os.environ.get("OPENARM_DEBUG_RENDER"):
        def _pen(geom_id: int) -> tuple[float, float]:
            """Max penetration of this geom's mesh inside the cube box (m, mm-AABB)."""
            mid = int(model.geom_dataid[geom_id])
            a = int(model.mesh_vertadr[mid])
            n = int(model.mesh_vertnum[mid])
            v = np.asarray(model.mesh_vert[a : a + n], dtype=np.float64)
            span_mm = float(np.ptp(v, axis=0).max() * 1000.0)
            R = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            t = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            w = v @ R.T + t
            oR = np.asarray(data.geom_xmat[object_geom], dtype=np.float64).reshape(3, 3)
            ot = np.asarray(data.geom_xpos[object_geom], dtype=np.float64)
            hs = np.asarray(model.geom_size[object_geom], dtype=np.float64)[:3]
            ql = (w - ot) @ oR
            d = hs - np.abs(ql)
            ins = np.all(d > 0.0, axis=1)
            pen = np.where(ins, d.min(axis=1), 0.0)
            return (float(pen.max()) if pen.size else 0.0), span_mm

        _gid = lambda nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        _nm = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
        print("[debug] object_geom=%d ..." % object_geom)
        print("[debug] 抬升瞬间方块接触:")
        for i in range(data.ncon):
            g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
            if object_geom in (g1, g2):
                print(f"   {_nm(g1)} <-> {_nm(g2)}  dist={data.contact[i].dist*1000:.2f}mm")
        for _b in ("openarm_left_hand", "openarm_left_link7", "openarm_left_link6", "openarm_left_link5"):
            _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _b)
            if _bid >= 0:
                print(f"[debug] {_b:22s} world={np.round(data.xpos[_bid],4).tolist()}"
                      f"  ->方块距离={np.linalg.norm(data.xpos[_bid]-data.xpos[object_body])*1000:.1f}mm")
        for _nm in (
            "openarm_left_hand_collision",
            "openarm_left_hand_visual_0",
            "openarm_left_right_finger_collision",
            "openarm_left_left_finger_collision",
            "openarm_left_right_finger_visual_1",
            "openarm_left_left_finger_visual_1",
        ):
            _g = _gid(_nm)
            _p, _s = _pen(_g) if _g >= 0 else (0.0, 0.0)
            print(f"[debug] {_nm:38s} meshspan={_s:7.2f}mm  cube-penetration={_p*1000:7.2f}mm")

        model.geom_rgba[object_geom] = [1.0, 0.15, 0.15, 0.30]
        _dbg = mujoco.Renderer(model, height=480, width=640)
        _visopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(_visopt)
        _visopt.geomgroup[3] = 0
        for _tag, _opt in (("all", None), ("vis", _visopt)):
            for _nm, _az, _el in (("front", 0.0, -5.0), ("side", 90.0, -5.0), ("iso", 135.0, -20.0)):
                _cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, _cam)
                _cam.lookat[:] = data.xpos[object_body]
                _cam.distance = 0.42
                _cam.azimuth = _az
                _cam.elevation = _el
                if _opt is None:
                    _dbg.update_scene(data, camera=_cam)
                else:
                    _dbg.update_scene(data, camera=_cam, scene_option=_opt)
                cv2.imwrite(
                    str(args.output_dir / f"dbg_lift_{_tag}_{_nm}.png"),
                    cv2.cvtColor(_dbg.render(), cv2.COLOR_RGB2BGR),
                )
        _dbg.close()
        print("[debug] 已输出 dbg_lift_{all,vis}_{front,side,iso}.png")
        _t = np.asarray(_TRACE, dtype=np.float64)
        if _t.size:
            _loc = _t[:, 3:6] * 1000.0
            _tail = _loc[-1500:]
            print(f"[debug] 真实步进路径 {len(_t)} 步 | timestep={model.opt.timestep}")
            print("[debug] 方块在手部坐标系中的偏移 — 全程范围 (mm):")
            print("[debug]   ", np.round((_loc.max(0) - _loc.min(0)), 2).tolist())
            print("[debug] 末段 1500 步（抬升+保持）范围 (mm):")
            print("[debug]   ", np.round((_tail.max(0) - _tail.min(0)), 2).tolist())
            _weld_on = np.nonzero(_t[:, 6] > 0.5)[0]
            _on = int(_weld_on[0]) if _weld_on.size else -1
            _name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_id)
            _rel = np.asarray(grasp_weld.get("relative_position_hand_frame", [0.0] * 3))
            print(
                f"[debug] 焊接激活于第 {_on} 步 / 共 {len(_t)} 步 | "
                f"焊接锚点 body={_name} | |relpose|={np.linalg.norm(_rel) * 1000:.2f}mm"
            )
            if _on >= 0:
                _err = np.linalg.norm(_t[_on:, 7:10], axis=1) * 1000.0
                _ang = _t[_on:, 10]
                print(
                    f"[debug] 焊接跟踪误差（约束违反）— 抬升+保持段 {len(_err)} 步:"
                    f" 位置 max={_err.max():.2f}mm 均值={_err.mean():.2f}mm"
                    f" | 姿态 max={_ang.max():.2f}deg 均值={_ang.mean():.2f}deg"
                )
                print("[debug] 位置误差时序 (mm), 每 400 步:")
                print("[debug]   ", np.round(_err[::400], 2).tolist())
                print("[debug] 姿态误差时序 (deg), 每 400 步:")
                print("[debug]   ", np.round(_ang[::400], 2).tolist())
                _ep = _t[_on:, 7:10] * 1000.0
                _rv = _t[_on:, 11:14]
                print(
                    "[debug] 位置误差分量 最大绝对值 (mm) [TCP系 x,y,z]: "
                    + str(np.round(np.abs(_ep).max(0), 3).tolist())
                    + " | 末值: "
                    + str(np.round(_ep[-1], 3).tolist())
                )
                print(
                    "[debug] 姿态误差分量 最大绝对值 (deg) [TCP系 rx,ry,rz]: "
                    + str(np.round(np.abs(_rv).max(0), 3).tolist())
                    + " | 末值: "
                    + str(np.round(_rv[-1], 3).tolist())
                )
    # --- /TEMP DEBUG RENDER ---
    final_object = data.xpos[object_body].copy()
    finger_midpoint = 0.5 * (data.geom_xpos[finger_geom_ids[0]] + data.geom_xpos[finger_geom_ids[1]])
    contact = any(
        object_geom in {int(data.contact[i].geom1), int(data.contact[i].geom2)}
        and bool({finger_geom_ids[0], finger_geom_ids[1]}.intersection({int(data.contact[i].geom1), int(data.contact[i].geom2)}))
        for i in range(data.ncon)
    )
    lifted = float(final_object[2]) > initial_object_z + 0.025
    near_hand = float(np.linalg.norm(final_object - finger_midpoint)) < 0.18
    success = bool(lifted and near_hand)
    if step_display is not None:
        step_display.capture(
            "result_success" if success else "result_failure",
            model,
            data,
            viewer,
            object_body=object_body,
            finger_geom_ids=finger_geom_ids,
            observation=observation,
            center_world=center,
            target_points=(closed_points[0].copy(), closed_points[1].copy()),
            lines=(
                f"lift={final_object[2] - initial_object_z:.4f}m",
                f"contact={contact}",
                f"near_hand={near_hand}",
            ),
        )
    report["result"] = {
        "success": success,
        "object_initial_z_after_observation_m": initial_object_z,
        "object_final_world": final_object.tolist(),
        "lift_m": float(final_object[2] - initial_object_z),
        "near_finger_midpoint": near_hand,
        "finger_midpoint_world": finger_midpoint.tolist(),
        "contact": bool(contact),
        "motion_mode": (
            "pinocchio_manifold_path_with_reachable_axis_search"
            if pin_planner is not None
            else "reachable_axis_search_cartesian_z_descent"
        ),
        "closed_points_world": [point.tolist() for point in closed_points],
        "lift_metrics": lift_metrics,
        "postclose_contact": close_contact,
        "grasp_weld": grasp_weld,
    }
    print(
        f"[手眼抓取] 结果={'成功' if success else '失败'} | 方块最终XYZ={np.round(final_object, 4).tolist()} | "
        f"抬升={final_object[2] - initial_object_z:.4f}m | 接触={contact} | "
        f"距指尖中点={np.linalg.norm(final_object - finger_midpoint):.3f}m"
    )


def make_tcp_target_for_hand_position(
    tcp_world: np.ndarray,
    hand_world: np.ndarray,
    target_hand_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tcp_from_hand = np.linalg.inv(tcp_world) @ hand_world
    target_tcp = np.eye(4, dtype=np.float64)
    target_tcp[:3, :3] = tcp_world[:3, :3]
    target_tcp[:3, 3] = target_hand_position - tcp_world[:3, :3] @ tcp_from_hand[:3, 3]
    return target_tcp[:3, 3], matrix_to_quaternion(target_tcp[:3, :3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SIDES, default="left")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument(
        "--cube-test-pose", type=float, nargs=3,
        metavar=("X", "Y", "YAW_DEG"),
        help=("simulation fixture only: relocate the cube before wrist-camera "
              "observation; these coordinates are never supplied to perception or IK"),
    )
    parser.add_argument("--hand-eye", type=Path, default=None)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--search-x", type=float, default=0.28)
    parser.add_argument("--search-y", type=float, default=0.0)
    parser.add_argument("--search-z", type=float, default=0.60)
    parser.add_argument("--cube-half-size", type=float, default=0.0288)
    parser.add_argument("--approach-height", type=float, default=0.07)
    parser.add_argument(
        "--descent-min-height",
        type=float,
        default=0.02,
        help="分段下降的最低高度，默认距方块中心 20 mm，不下降到 10/0 mm",
    )
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--search-s", type=float, default=1.5)
    parser.add_argument("--approach-s", type=float, default=1.5)
    parser.add_argument("--close-s", type=float, default=0.8)
    parser.add_argument("--lift-s", type=float, default=1.2)
    parser.add_argument(
        "--planner",
        choices=("pinocchio", "scipy"),
        default="pinocchio",
        help="关节 IK/路径规划器；Pinocchio 使用雅可比 IK 和关节流形插值",
    )
    parser.add_argument(
        "--pinocchio-prefix",
        type=Path,
        default=None,
        help="外部 Pinocchio Conda 环境路径；默认自动使用用户级 Python 3.12 环境",
    )
    parser.add_argument(
        "--path-waypoints",
        type=int,
        default=12,
        help="每段关节流形路径的插值点数，越大越平滑但仿真时间越长",
    )
    parser.add_argument(
        "--grasp-span",
        type=float,
        default=0.035,
        help="target half-span between the two finger collision geometries (m)",
    )
    parser.add_argument(
        "--finger-center-span",
        type=float,
        default=0.070,
        help="target full span between finger collision-geometry centers after closing (m)",
    )
    parser.add_argument(
        "--skip-approach-ik",
        action="store_true",
        help="屏蔽单独的接近位姿 IK，直接从搜索位姿移动到抓取位姿并闭爪",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.044,
        help="simulation finger joint position for open (m)",
    )
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=0.020,
        help="simulation finger joint position for the gentle grasp (m)",
    )
    parser.add_argument(
        "--dynamic-controller",
        action="store_true",
        help="use the original torque controller instead of kinematic diagnostic motion",
    )
    parser.add_argument(
        "--max-grasp-ik-error",
        type=float,
        default=0.020,
        help="abort the diagnostic when either fingertip IK error exceeds this value (m)",
    )
    parser.add_argument(
        "--force-grasp",
        action="store_true",
        help="execute even when fingertip IK is outside the safety threshold",
    )
    parser.add_argument(
        "--no-grasp-latch",
        action="store_true",
        help="disable the post-contact weld used to stabilize the kinematic diagnostic lift",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="显示并保存检测、候选姿态、下降、闭爪、抬升和结果等中间状态",
    )
    parser.add_argument(
        "--step-pause",
        type=float,
        default=0.5,
        help="中间状态显示停留时间（秒）；仅对可视化窗口生效",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "calibration" / "handeye_grasp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.cube_half_size <= 0.0
        or args.finger_center_span <= 0.0
        or args.approach_height <= 0.0
        or args.lift_height <= 0.0
        or args.descent_min_height < 0.0
        or args.descent_min_height >= args.approach_height
    ):
        raise ValueError(
            "cube-half-size, approach-height, and lift-height must be positive; "
            "descent-min-height must be in [0, approach-height)"
        )
    if args.path_waypoints < 2:
        raise ValueError("path-waypoints must be at least 2")
    if args.pinocchio_prefix is not None:
        os.environ["OPENARM_PINOCCHIO_PREFIX"] = str(args.pinocchio_prefix.resolve())
    scene = args.scene or ensure_scene("cube")
    scene = scene if scene.is_absolute() else (ROOT / scene).resolve()
    calibration_path = args.hand_eye.resolve() if args.hand_eye else latest_handeye_report(args.arm)
    handeye, camera_matrix = load_handeye(calibration_path, args.arm)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    reset_to_ready(model, data, id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready"))
    if args.cube_test_pose is not None:
        x, y, yaw_deg = map(float, args.cube_test_pose)
        cube_joint = id_for(model, mujoco.mjtObj.mjOBJ_JOINT,
                            "droid_object_freejoint")
        qadr = int(model.jnt_qposadr[cube_joint])
        dadr = int(model.jnt_dofadr[cube_joint])
        yaw = np.deg2rad(yaw_deg)
        data.qpos[qadr:qadr + 2] = (x, y)
        data.qpos[qadr + 3:qadr + 7] = (
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
    specs = build_arm_specs(model)
    spec = specs[args.arm]
    pin_planner = None
    if args.planner == "pinocchio" and not args.dynamic_controller:
        pin_planner = PinocchioArmPlanner(
            scene,
            args.arm,
            spec.q_range,
            frame_name=f"openarm_{args.arm}_hand",
        )
        print(
            f"[路径规划] Pinocchio {pin_planner.pin.__version__} | "
            f"末端坐标系={pin_planner.frame_name} | "
            f"关节流形插值点数={args.path_waypoints}"
        )
    finger_actuators = build_finger_actuators(model)
    finger_q_indices = {side: finger_joint_indices(model, side) for side in SIDES}
    wrist_camera = f"v22_{args.arm}_d405_rgb"
    camera_id = id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, wrist_camera)
    object_body = id_for(model, mujoco.mjtObj.mjOBJ_BODY, "droid_object")
    object_geom = id_for(model, mujoco.mjtObj.mjOBJ_GEOM, "droid_object_geom")
    hand_body = id_for(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{args.arm}_hand")
    finger_body_ids = (
        id_for(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{args.arm}_right_finger"),
        id_for(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{args.arm}_left_finger"),
    )
    finger_geom_ids = (
        id_for(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarm_{args.arm}_right_finger_collision"),
        id_for(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarm_{args.arm}_left_finger_collision"),
    )
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    preview = CameraPreview((wrist_camera,), args.image_width, args.image_height) if args.preview else None
    q_target = {side: data.qpos[other.qpos].copy() for side, other in specs.items()}
    gripper = {"left": args.gripper_open, "right": args.gripper_open}
    # In this scene the direct joint coordinate 0.044 is open and 0.020 is a
    # gentle grasp.  The left XML uses a force motor, so the diagnostic path
    # writes qpos directly; --dynamic-controller remains available for testing
    # the original torque path separately.
    if not args.dynamic_controller:
        set_kinematic_state(
            model,
            data,
            specs,
            finger_q_indices,
            q_target,
            gripper,
        )
    initial_object_z = float(data.xpos[object_body, 2])
    report: dict[str, object] = {
        "format": "openarm_handeye_red_cube_grasp_v2",
        "arm": args.arm,
        "scene": str(scene),
        "hand_eye_report": str(calibration_path),
        "wrist_camera": wrist_camera,
        "object_initial_world": data.xpos[object_body].copy().tolist(),
    }
    step_display = (
        GraspStepDisplay(
            args.output_dir,
            renderer,
            wrist_camera,
            preview,
            report,
            args.step_pause,
        )
        if args.show_steps
        else None
    )

    viewer_context = None if args.headless else mujoco.viewer.launch_passive(model, data)
    if viewer_context is None:
        class HeadlessContext:
            def __enter__(self): return None
            def __exit__(self, *_args): return False
        viewer_context = HeadlessContext()
    try:
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = [0.38, 0.0, 0.34]
                viewer.cam.distance = 1.15
                viewer.cam.azimuth = 0
                viewer.cam.elevation = -55
                viewer.sync()

            q_ready = data.qpos[spec.qpos].copy()
            quat_ready = quat_from_body(model, data, spec.body_id)
            hand_rotation_ready = body_transform(data, spec.body_id)[:3, :3].copy()
            if step_display is not None:
                step_display.capture(
                    "ready",
                    model,
                    data,
                    viewer,
                    object_body=object_body,
                    finger_geom_ids=finger_geom_ids,
                    lines=("gripper=open", "camera=wrist"),
                )

            def move_arm_to(target_q: np.ndarray, duration_s: float, finger_value: float) -> None:
                finger_values = {"left": finger_value, "right": finger_value}
                if args.dynamic_controller:
                    q_target[args.arm] = target_q
                    gripper.update(finger_values)
                    step_controller(
                        model,
                        data,
                        specs,
                        finger_actuators,
                        q_target,
                        gripper,
                        duration_s,
                        viewer,
                    )
                else:
                    start_q = {side: data.qpos[other.qpos].copy() for side, other in specs.items()}
                    end_q = {side: values.copy() for side, values in start_q.items()}
                    end_q[args.arm] = target_q.copy()
                    move_kinematic(
                        model,
                        data,
                        specs,
                        finger_q_indices,
                        start_q,
                        end_q,
                        finger_values,
                        duration_s,
                        viewer,
                    )

            def hold_arm(target_q: np.ndarray, duration_s: float, finger_value: float) -> None:
                finger_values = {"left": finger_value, "right": finger_value}
                if args.dynamic_controller:
                    q_target[args.arm] = target_q
                    gripper.update(finger_values)
                    step_controller(
                        model,
                        data,
                        specs,
                        finger_actuators,
                        q_target,
                        gripper,
                        duration_s,
                        viewer,
                    )
                else:
                    arm_q = {side: data.qpos[other.qpos].copy() for side, other in specs.items()}
                    arm_q[args.arm] = target_q.copy()
                    hold_kinematic(
                        model,
                        data,
                        specs,
                        finger_q_indices,
                        arm_q,
                        finger_values,
                        duration_s,
                        viewer,
                    )

            search_position = np.asarray([args.search_x, args.search_y, args.search_z], dtype=np.float64)
            # Clear the tabletop first.  The arm starts beside the cube, and a
            # direct diagonal move can otherwise brush it before the wrist
            # camera has made its observation.
            clear_y = 0.16 if args.arm == "left" else -0.16
            clear_position = np.asarray(
                [args.search_x, clear_y, max(args.search_z + 0.12, 0.70)], dtype=np.float64
            )
            q_clear, clear_pos_err, clear_rot_err = solve_target_q(
                model,
                data,
                spec,
                clear_position,
                quat_ready,
                q_ready,
                pin_planner=pin_planner,
                target_rotation=hand_rotation_ready,
            )
            print(
                f"[手眼抓取] 阶段=避障位姿 | 位置误差={clear_pos_err * 1000:.1f}mm "
                f"| 旋转误差={np.degrees(clear_rot_err):.2f}°"
            )
            move_arm_to(q_clear, args.search_s, args.gripper_open)
            if step_display is not None:
                step_display.capture(
                    "clear_pose",
                    model,
                    data,
                    viewer,
                    object_body=object_body,
                    finger_geom_ids=finger_geom_ids,
                    lines=(
                        f"pos_error={clear_pos_err * 1000:.1f}mm",
                        f"rot_error={np.degrees(clear_rot_err):.2f}deg",
                    ),
                )
            q_search, search_pos_err, search_rot_err = solve_target_q(
                model,
                data,
                spec,
                search_position,
                quat_ready,
                q_clear,
                pin_planner=pin_planner,
                target_rotation=hand_rotation_ready,
            )
            print(
                f"[手眼抓取] 阶段=搜索位姿 | 位置误差={search_pos_err * 1000:.1f}mm "
                f"| 旋转误差={np.degrees(search_rot_err):.2f}°"
            )
            move_arm_to(q_search, args.search_s, args.gripper_open)
            if step_display is not None:
                step_display.capture(
                    "search_pose",
                    model,
                    data,
                    viewer,
                    object_body=object_body,
                    finger_geom_ids=finger_geom_ids,
                    lines=(
                        f"pos_error={search_pos_err * 1000:.1f}mm",
                        f"rot_error={np.degrees(search_rot_err):.2f}deg",
                    ),
                )
            mujoco.mj_forward(model, data)
            tcp_world = body_transform(data, spec.body_id)
            rgb, depth = render_rgb_depth(renderer, data, wrist_camera)
            observation = estimate_cube_center(
                rgb, depth, tcp_world, handeye, camera_matrix, args.cube_half_size
            )
            report["observation"] = {
                "center_world": observation.center_world.tolist(),
                "surface_center_world": observation.center_surface_world.tolist(),
                "pixel_center": observation.pixel_center.tolist(),
                "mask_pixels": observation.mask_pixels,
                "bbox": list(observation.bbox),
                "depth_median_m": observation.depth_median_m,
                "tcp_world": tcp_world.tolist(),
                # Simulation-only validation; this value is never used to
                # plan the grasp.
                "object_actual_world_at_observation": data.xpos[object_body].copy().tolist(),
            }
            print(
                f"[手眼抓取] 红色方块 | 像素中心={np.round(observation.pixel_center, 1).tolist()} "
                f"| 基座XYZ={np.round(observation.center_world, 4).tolist()} "
                f"| mask={observation.mask_pixels}"
            )
            if preview is not None:
                preview.show({wrist_camera: rgb})
                preview.poll()
            if step_display is not None:
                step_display.capture(
                    "detect_cube",
                    model,
                    data,
                    viewer,
                    object_body=object_body,
                    finger_geom_ids=finger_geom_ids,
                    observation=observation,
                    center_world=observation.center_world,
                    lines=(
                        f"pixel={np.round(observation.pixel_center, 1).tolist()}",
                        f"xyz={np.round(observation.center_world, 4).tolist()}",
                        f"mask={observation.mask_pixels}",
                    ),
                )
            if args.detect_only:
                report["result"] = "detect_only"
                return

            if not args.dynamic_controller:
                run_reachable_handeye_grasp(
                    model,
                    data,
                    specs,
                    spec,
                    finger_q_indices,
                    finger_geom_ids,
                    observation,
                    object_body,
                    object_geom,
                    args,
                    viewer,
                    report,
                    pin_planner,
                    step_display,
                )
                return

            approach_midpoint = observation.center_world + np.asarray([0.0, 0.0, args.approach_height])
            q_approach, approach_midpoint_err = solve_gripper_midpoint_q(
                model,
                data,
                spec,
                finger_body_ids,
                approach_midpoint,
                data.qpos[spec.qpos].copy(),
            )
            q_grasp, grasp_midpoint_err = solve_gripper_midpoint_q(
                model,
                data,
                spec,
                finger_body_ids,
                observation.center_world,
                q_approach,
            )
            print(
                f"[手眼抓取] 接近参数 | 两指中点目标={np.round(approach_midpoint, 4).tolist()} "
                f"| IK中点误差={approach_midpoint_err * 1000:.1f}mm"
            )
            print(
                f"[手眼抓取] 抓取参数 | 两指中点目标={np.round(observation.center_world, 4).tolist()} "
                f"| IK中点误差={grasp_midpoint_err * 1000:.1f}mm"
            )
            q_target[args.arm] = q_approach
            step_controller(model, data, specs, finger_actuators, q_target, gripper, args.approach_s, viewer)
            print(
                f"[手眼抓取] 接近完成 | 手={np.round(data.xpos[hand_body], 4).tolist()} "
                f"| 两指中点={np.round(0.5 * (data.xpos[finger_body_ids[0]] + data.xpos[finger_body_ids[1]]), 4).tolist()} "
                f"| 方块={np.round(data.xpos[object_body], 4).tolist()}"
            )
            q_target[args.arm] = q_grasp
            step_controller(model, data, specs, finger_actuators, q_target, gripper, args.approach_s, viewer)
            print(
                f"[手眼抓取] 抓取位姿完成 | 手={np.round(data.xpos[hand_body], 4).tolist()} "
                f"| 两指中点={np.round(0.5 * (data.xpos[finger_body_ids[0]] + data.xpos[finger_body_ids[1]]), 4).tolist()} "
                f"| 方块={np.round(data.xpos[object_body], 4).tolist()}"
            )
            print("[手眼抓取] 阶段=闭合夹爪 | gripper=0.000")
            gripper[args.arm] = 0.0
            step_controller(model, data, specs, finger_actuators, q_target, gripper, args.close_s, viewer)
            print(
                f"[手眼抓取] 闭爪完成 | 手={np.round(data.xpos[hand_body], 4).tolist()} "
                f"| 两指中点={np.round(0.5 * (data.xpos[finger_body_ids[0]] + data.xpos[finger_body_ids[1]]), 4).tolist()} "
                f"| 方块={np.round(data.xpos[object_body], 4).tolist()} | contacts={data.ncon}"
            )

            lift_midpoint = observation.center_world + np.asarray([0.0, 0.0, args.lift_height])
            q_lift, lift_midpoint_err = solve_gripper_midpoint_q(
                model,
                data,
                spec,
                finger_body_ids,
                lift_midpoint,
                data.qpos[spec.qpos].copy(),
            )
            q_target[args.arm] = q_lift
            print(
                f"[手眼抓取] 阶段=抬升 | 两指中点目标={np.round(lift_midpoint, 4).tolist()} "
                f"| IK中点误差={lift_midpoint_err * 1000:.1f}mm"
            )
            step_controller(model, data, specs, finger_actuators, q_target, gripper, args.lift_s, viewer)
            mujoco.mj_forward(model, data)
            final_object = data.xpos[object_body].copy()
            finger_geoms = {
                geom_id
                for geom_id in range(model.ngeom)
                if "finger_collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                and int(model.geom_bodyid[geom_id]) == hand_body
            }
            contact = any(
                object_geom in {int(data.contact[i].geom1), int(data.contact[i].geom2)}
                and bool(finger_geoms.intersection({int(data.contact[i].geom1), int(data.contact[i].geom2)}))
                for i in range(data.ncon)
            )
            lifted = float(final_object[2]) > initial_object_z + 0.025
            near_hand = float(np.linalg.norm(final_object - data.xpos[hand_body])) < 0.18
            success = bool(lifted and near_hand)
            report["result"] = {
                "success": success,
                "object_final_world": final_object.tolist(),
                "lift_m": float(final_object[2] - initial_object_z),
                "near_hand": near_hand,
                "contact": bool(contact),
                "gripper": gripper[args.arm],
            }
            print(
                f"[手眼抓取] 结果={'成功' if success else '失败'} | "
                f"方块最终XYZ={np.round(final_object, 4).tolist()} | "
                f"抬升={final_object[2] - initial_object_z:.4f}m | 接触={contact}"
            )
    finally:
        if preview is not None:
            preview.close()
        renderer.close()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[手眼抓取] 报告: {output}")


if __name__ == "__main__":
    main()
