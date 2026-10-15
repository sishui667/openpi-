# -*- coding: utf-8 -*-
"""Automatically collect and solve OpenArm wrist-camera calibration.

The collector moves one TCP through camera poses that look at the fixed
simulation checkerboard.  Once each pose is settled, it records chest/wrist
RGB-D images and robot/camera ground truth using ``calibration_simulation``.
The solver then detects checkerboard corners, estimates the simulated
wrist RGB intrinsics, solves OpenCV hand-eye calibration, and writes a report.

Run::

    conda activate openpi
    cd D:\\openpi\\openarm_cloth
    python .\\auto_calibrate_left.py --arm left --preview
    python .\\auto_calibrate_right.py --preview
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from build_calibration_scene import ensure_scene
from calibration_simulation import (
    CalibrationRecorder,
    CameraPreview,
    body_transform,
    camera_transform,
    camera_intrinsics,
    render_rgb,
    transform_from_pose,
)
from teleop_towel_dataset import (
    ArmSpec,
    POSE_ACTIONS,
    apply_controller,
    build_arm_specs,
    build_finger_actuators,
    body_pose,
    id_for,
    ik_step,
    quat_from_body,
    reset_to_ready,
    safety_reason,
    TeleopState,
)


ROOT = Path(__file__).resolve().parent
SIDES = ("left", "right")
DEFAULT_CAMERAS_BY_ARM = {
    "left": ("chest_d435i_preview", "v22_left_d405_rgb"),
    "right": ("chest_d435i_preview", "v22_right_d405_rgb"),
}


def wrist_camera_name(arm: str) -> str:
    return f"v22_{arm}_d405_rgb"


def arm_label(arm: str) -> str:
    return "左臂" if arm == "left" else "右臂"


def camera_local_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return a small rotation applied in the nominal camera frame."""

    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    rx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    ry = np.asarray(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.asarray(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=np.float64).reshape(-1))
    return quaternion


def make_wrist_camera_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_spec: ArmSpec,
    board_body_id: int,
    camera_id: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reachable TCP targets while keeping the wrist camera on-board."""

    board_center = data.xpos[board_body_id].copy()
    world_tcp = body_transform(data, arm_spec.body_id)
    world_camera = camera_transform(data, camera_id)
    tcp_to_camera = np.linalg.inv(world_tcp) @ world_camera
    nominal_camera_rotation = world_camera[:3, :3]
    patterns = [
        (0.0, 0.0, 0.0),
        (-10.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, -10.0, 0.0),
        (0.0, 10.0, 0.0),
    ]
    offsets = [(distance, *pattern) for distance in (0.22, 0.27, 0.32, 0.38, 0.44) for pattern in patterns]
    targets: list[tuple[np.ndarray, np.ndarray]] = []
    for distance, roll_deg, pitch_deg, yaw_deg in offsets:
        camera_rotation = nominal_camera_rotation @ camera_local_rotation(roll_deg, pitch_deg, yaw_deg)
        forward = camera_rotation @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        camera_position = board_center - forward * distance
        world_camera_target = transform_from_pose(camera_position, camera_rotation)
        world_tcp_target = world_camera_target @ np.linalg.inv(tcp_to_camera)
        tcp_position = world_tcp_target[:3, 3]
        if not (0.18 <= tcp_position[0] <= 0.75 and -0.45 <= tcp_position[1] <= 0.45 and 0.12 <= tcp_position[2] <= 0.80):
            continue
        targets.append((tcp_position.copy(), matrix_to_quaternion(world_tcp_target[:3, :3])))
    return targets


def make_left_camera_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_spec: ArmSpec,
    board_body_id: int,
    camera_id: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Backward-compatible left-arm target generator."""

    return make_wrist_camera_targets(model, data, left_spec, board_body_id, camera_id)


def solve_kinematic_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: ArmSpec,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_seed: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Solve a static TCP target for the simulation collector.

    Calibration only needs accurate, repeatable poses.  Solving in joint space
    and interpolating the resulting qpos avoids mixing controller dynamics
    into the calibration data.
    """

    q_initial = np.asarray(q_seed, dtype=np.float64).copy()

    def residual(q: np.ndarray) -> np.ndarray:
        data.qpos[spec.qpos] = q
        data.qvel[spec.dof] = 0.0
        mujoco.mj_forward(model, data)
        rotation_error = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(rotation_error, target_quat, quat_from_body(model, data, spec.body_id))
        position_error = np.asarray(target_pos, dtype=np.float64) - data.xpos[spec.body_id]
        # Scale metres so position and orientation contribute comparably.
        return np.concatenate([position_error * 10.0, rotation_error])

    result = least_squares(
        residual,
        q_initial,
        bounds=(spec.q_range[:, 0], spec.q_range[:, 1]),
        max_nfev=500,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    q_solution = np.asarray(result.x, dtype=np.float64)
    data.qpos[spec.qpos] = q_solution
    data.qvel[spec.dof] = 0.0
    mujoco.mj_forward(model, data)
    final_position_error = float(np.linalg.norm(np.asarray(target_pos) - data.xpos[spec.body_id]))
    final_rotation_error = np.zeros(3, dtype=np.float64)
    mujoco.mju_subQuat(final_rotation_error, target_quat, quat_from_body(model, data, spec.body_id))
    return q_solution, final_position_error, float(np.linalg.norm(final_rotation_error))


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def detect_corners_image(image: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    color_image = image if image.ndim == 3 else None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(image, pattern_size, flags)
    if not found or corners is None:
        return None
    corners = np.asarray(corners, dtype=np.float32).reshape(pattern_size[1], pattern_size[0], 2)
    if color_image is not None:
        red, green, blue = color_image[..., 0], color_image[..., 1], color_image[..., 2]
        marker_mask = (red > 80) & (red - green > 45) & (red - blue > 45)
        ys, xs = np.nonzero(marker_mask)
        if len(xs) >= 8:
            marker_center = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)
            orientations = (
                corners,
                corners[:, ::-1, :],
                corners[::-1, :, :],
                corners[::-1, ::-1, :],
            )
            corners = min(orientations, key=lambda candidate: float(np.linalg.norm(candidate[0, 0] - marker_center)))
    return corners.reshape(-1, 1, 2)


def detect_corners(image_path: Path, pattern_size: tuple[int, int]) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return detect_corners_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), pattern_size)


def make_left_joint_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_spec: ArmSpec,
    camera_id: int,
    renderer: mujoco.Renderer,
    pose_count: int,
) -> list[np.ndarray]:
    """Search safe joint perturbations and retain views with full board detection."""

    base_q = data.qpos[left_spec.qpos].copy()
    pattern_size = (8, 5)
    scales = np.asarray([0.10, 0.08, 0.10, 0.10, 0.14, 0.16, 0.18], dtype=np.float64)
    rng = np.random.default_rng(20260924)
    candidates = [base_q]
    for _ in range(80):
        candidates.append(base_q + rng.uniform(-1.0, 1.0, size=7) * scales)
    selected: list[np.ndarray] = []
    selected_camera_poses: list[np.ndarray] = []
    for candidate in candidates:
        candidate = np.clip(candidate, left_spec.q_range[:, 0] + 1e-4, left_spec.q_range[:, 1] - 1e-4)
        data.qpos[left_spec.qpos] = candidate
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        image = render_rgb(renderer, data, "v22_left_d405_rgb")
        if detect_corners_image(image, pattern_size) is None:
            continue
        camera_pose = camera_transform(data, camera_id)
        too_close = any(
            np.linalg.norm(camera_pose[:3, 3] - previous[:3, 3]) < 0.003
            and rotation_error_deg(camera_pose[:3, :3], previous[:3, :3]) < 2.0
            for previous in selected_camera_poses
        )
        if selected and too_close:
            continue
        selected.append(candidate.copy())
        selected_camera_poses.append(camera_pose.copy())
        if len(selected) >= pose_count:
            break
    data.qpos[left_spec.qpos] = base_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return selected


def solve_wrist_session(session_dir: Path, arm: str, free_distortion: bool = False) -> Path:
    if arm not in SIDES:
        raise ValueError(f"Unsupported arm: {arm}")
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    board = metadata["board"]
    pattern_size = (int(board["inner_corners_columns"]), int(board["inner_corners_rows"]))
    object_points = np.asarray(board["inner_corners_board_xyz_m"], dtype=np.float32)
    board_surface_z = float(object_points[0, 2])
    calibration_points = object_points.copy()
    calibration_points[:, 2] = 0.0
    camera_names = metadata["camera_order"]
    wrist_name = wrist_camera_name(arm)
    wrist_index = camera_names.index(wrist_name)
    arm_index = SIDES.index(arm)
    samples = np.load(session_dir / "samples.npz")
    frame_count = len(samples["time_s"])
    detected_indices: list[int] = []
    image_points: list[np.ndarray] = []
    object_point_sets: list[np.ndarray] = []
    target_to_camera: list[np.ndarray] = []
    robot_gripper_to_base: list[np.ndarray] = []
    depth_rmse_values: list[float] = []
    depth_valid_ratios: list[float] = []
    camera_intrinsic_seed = metadata["camera_intrinsics"][wrist_name]
    image_size = (int(camera_intrinsic_seed["width"]), int(camera_intrinsic_seed["height"]))
    for index in range(frame_count):
        image_path = session_dir / "images" / wrist_name / "rgb" / f"{index:06d}.png"
        corners = detect_corners(image_path, pattern_size)
        if corners is None:
            continue
        detected_indices.append(index)
        image_points.append(corners)
        object_point_sets.append(calibration_points.copy())

    if len(detected_indices) < 8:
        raise RuntimeError(
            f"Only {len(detected_indices)}/{frame_count} {arm}-wrist checkerboards detected; "
            "need at least 8 valid views. Move the board or increase the automatic pose count."
        )

    camera_matrix_seed = np.asarray(
        [
            [camera_intrinsic_seed["fx"], 0.0, camera_intrinsic_seed["cx"]],
            [0.0, camera_intrinsic_seed["fy"], camera_intrinsic_seed["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if free_distortion:
        rms, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
            object_point_sets,
            image_points,
            image_size,
            None,
            None,
        )
        intrinsics_source = "opencv_free_calibration"
    else:
        # MuJoCo's renderer is an ideal pinhole camera.  Fitting radial
        # distortion to this synthetic data is ill-conditioned and can make
        # the hand-eye result worse, so use the renderer model as fixed K.
        camera_matrix = camera_matrix_seed
        distortion = np.zeros((5, 1), dtype=np.float64)
        intrinsics_source = "simulation_metadata_fixed_pinhole"
        reprojection_errors = []
        for corners in image_points:
            ok, rvec, tvec = cv2.solvePnP(
                calibration_points,
                corners,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            projected, _ = cv2.projectPoints(
                calibration_points,
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )
            reprojection_errors.extend(
                np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).tolist()
            )
        rms = float(np.sqrt(np.mean(np.square(reprojection_errors))))
    valid_robot_gripper_to_base: list[np.ndarray] = []
    for index, corners in zip(detected_indices, image_points):
        ok, rvec, tvec = cv2.solvePnP(
            calibration_points,
            corners,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        target_to_camera.append(make_transform(rotation, tvec.reshape(3)))
        valid_robot_gripper_to_base.append(samples["tcp_pose_world"][index, arm_index])

        depth_path = session_dir / "images" / wrist_name / "depth" / f"{index:06d}.npy"
        depth = np.load(depth_path)
        pixels = corners.reshape(-1, 2)
        valid = []
        measured = []
        predicted = []
        fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
        cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
        for point, pixel in zip(calibration_points, pixels):
            u, v = int(round(float(pixel[0]))), int(round(float(pixel[1])))
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue
            value = float(depth[v, u])
            if not np.isfinite(value) or value <= 1e-6:
                continue
            depth_point = np.asarray([(u - cx) * value / fx, (v - cy) * value / fy, value])
            predicted_point = (target_to_camera[-1] @ np.r_[point, 1.0])[:3]
            measured.append(depth_point)
            predicted.append(predicted_point)
            valid.append(value)
        if measured:
            depth_rmse_values.append(float(np.sqrt(np.mean(np.sum((np.asarray(measured) - np.asarray(predicted)) ** 2, axis=1)))) * 1000.0)
            depth_valid_ratios.append(float(len(valid) / len(calibration_points)))

    valid_count = len(target_to_camera)
    if valid_count < 8:
        raise RuntimeError(f"Only {valid_count} valid solvePnP poses; cannot solve hand-eye calibration")
    # Keep robot poses aligned with the exact image poses that passed solvePnP.
    robot_gripper_to_base = valid_robot_gripper_to_base
    R_gripper2base = [pose[:3, :3] for pose in robot_gripper_to_base]
    t_gripper2base = [pose[:3, 3] for pose in robot_gripper_to_base]
    R_target2cam = [pose[:3, :3] for pose in target_to_camera]
    t_target2cam = [pose[:3, 3] for pose in target_to_camera]

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    }
    board_surface_offset = np.eye(4, dtype=np.float64)
    board_surface_offset[2, 3] = board_surface_z
    board_world = samples["board_pose_world"][detected_indices[0]] @ board_surface_offset
    candidates: list[tuple[float, str, np.ndarray]] = []
    for name, method in methods.items():
        rotation, translation = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=method,
        )
        hand_eye = make_transform(rotation, translation.reshape(3))
        residuals = []
        for gripper, target_camera in zip(robot_gripper_to_base, target_to_camera):
            predicted_board = gripper @ hand_eye @ target_camera
            residuals.append(float(np.linalg.norm(predicted_board[:3, 3] - board_world[:3, 3])))
        candidates.append((float(np.median(residuals)), name, hand_eye))
    _, selected_method, hand_eye = min(candidates, key=lambda item: item[0])

    base_board_errors = []
    for gripper, target_camera in zip(robot_gripper_to_base, target_to_camera):
        predicted_board = gripper @ hand_eye @ target_camera
        base_board_errors.append(float(np.linalg.norm(predicted_board[:3, 3] - board_world[:3, 3])))

    ground_truth_tcp = samples["tcp_pose_world"][detected_indices[0], arm_index]
    ground_truth_camera = samples["camera_pose_world_opencv"][detected_indices[0], wrist_index]
    ground_truth_hand_eye = np.linalg.inv(ground_truth_tcp) @ ground_truth_camera
    translation_error_mm = float(np.linalg.norm(hand_eye[:3, 3] - ground_truth_hand_eye[:3, 3]) * 1000.0)
    rotation_error = rotation_error_deg(hand_eye[:3, :3], ground_truth_hand_eye[:3, :3])
    report = {
        "format": f"openarm_{arm}_wrist_calibration_v1",
        "session": str(session_dir.resolve()),
        "camera": wrist_name,
        "detector": "cv2.findChessboardCornersSB",
        "pattern_size_inner_corners": list(pattern_size),
        "total_frames": frame_count,
        "detected_frames": len(detected_indices),
        "detected_indices": detected_indices,
        "intrinsics_rms_reprojection_px": float(rms),
        "intrinsics_source": intrinsics_source,
        "camera_matrix": camera_matrix.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "hand_eye_method": selected_method,
        f"T_{arm}_tcp_{arm}_wrist_opencv": hand_eye.tolist(),
        "board_world_translation_rmse_mm": float(np.sqrt(np.mean(np.square(base_board_errors))) * 1000.0),
        "board_world_translation_median_mm": float(np.median(base_board_errors) * 1000.0),
        "depth_corner_rmse_median_mm": float(np.median(depth_rmse_values)) if depth_rmse_values else None,
        "depth_valid_ratio_median": float(np.median(depth_valid_ratios)) if depth_valid_ratios else None,
        "simulation_ground_truth_check": {
            "translation_error_mm": translation_error_mm,
            "rotation_error_deg": rotation_error,
            f"T_{arm}_tcp_{arm}_wrist_opencv_ground_truth": ground_truth_hand_eye.tolist(),
        },
    }
    output = session_dir / f"calibration_{arm}_wrist.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{arm_label(arm)}腕标定] 检测 {len(detected_indices)}/{frame_count} 帧 | "
        f"内参 RMS={rms:.4f}px | 手眼方法={selected_method} | "
        f"相对仿真真值: 平移={translation_error_mm:.2f}mm, 旋转={rotation_error:.3f}°"
    )
    print(f"[{arm_label(arm)}腕标定] 结果: {output}")
    return output


def solve_left_session(session_dir: Path, free_distortion: bool = False) -> Path:
    """Backward-compatible left-arm solver."""

    return solve_wrist_session(session_dir, "left", free_distortion=free_distortion)


def build_parser(default_arm: str = "left") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SIDES, default=default_arm)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--move-s", type=float, default=0.50, help="kinematic interpolation time per pose")
    parser.add_argument("--settle-s", type=float, default=0.65)
    parser.add_argument("--pose-count", type=int, default=15)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--session", type=Path, default=None, help="solve an existing session; skip collection")
    parser.add_argument("--no-solve", action="store_true", help="only collect data")
    parser.add_argument(
        "--free-distortion",
        action="store_true",
        help="fit camera distortion with OpenCV; not recommended for the ideal MuJoCo camera",
    )
    return parser


def collect_wrist(args: argparse.Namespace, arm: str) -> Path:
    if arm not in SIDES:
        raise ValueError(f"Unsupported arm: {arm}")
    if args.move_s <= 0.0 or args.settle_s <= 0.0 or args.pose_count < 8:
        raise ValueError("move-s and settle-s must be positive; pose-count must be at least 8")
    scene = ensure_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready")
    reset_to_ready(model, data, key_id)
    specs = build_arm_specs(model)
    finger_actuators = build_finger_actuators(model)
    camera_names = tuple(args.cameras or DEFAULT_CAMERAS_BY_ARM[arm])
    camera_ids = {name: id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in camera_names}
    wrist_name = wrist_camera_name(arm)
    wrist_camera_id = id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, wrist_name)
    board_body_id = id_for(model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board")
    targets = make_wrist_camera_targets(model, data, specs[arm], board_body_id, wrist_camera_id)
    if len(targets) < args.pose_count:
        raise RuntimeError(f"Only {len(targets)} safe automatic poses were generated")
    q_seed = data.qpos[specs[arm].qpos].copy()
    candidate_solutions: list[np.ndarray] = []
    for index, (target_pos, target_quat) in enumerate(targets):
        q_solution, position_error, rotation_error = solve_kinematic_ik(
            model, data, specs[arm], target_pos, target_quat, q_seed
        )
        if position_error > 0.006 or rotation_error > 0.05:
            print(
                f"[auto] skip pose {index}: IK residual position={position_error:.4f}m "
                f"rotation={rotation_error:.4f}rad"
            )
            continue
        candidate_solutions.append(q_solution.copy())
        q_seed = q_solution.copy()
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    solutions: list[np.ndarray] = []
    for candidate in candidate_solutions:
        data.qpos[specs[arm].qpos] = candidate
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        image = render_rgb(renderer, data, wrist_name)
        if detect_corners_image(image, (8, 5)) is not None:
            solutions.append(candidate.copy())
        if len(solutions) >= args.pose_count:
            break
    if len(solutions) < 8:
        renderer.close()
        raise RuntimeError(f"Only {len(solutions)} automatic poses with visible checkerboards were found")
    reset_to_ready(model, data, key_id)
    output_dir = args.output_dir or (ROOT / "calibration" / f"auto_{arm}")
    recorder = CalibrationRecorder(
        output_dir,
        model,
        camera_names,
        camera_ids,
        specs,
        board_body_id,
        renderer,
        args.image_width,
        args.image_height,
    )
    preview = CameraPreview(camera_names, args.image_width, args.image_height) if args.preview else None
    q_start = data.qpos[specs[arm].qpos].copy()
    pose_index = 0
    preview_time = -float("inf")
    tick_period = 0.01
    sim_time = 0.0
    phase_start = 0.0
    wall_tick = time.perf_counter()
    print(f"[自动采集] {arm_label(arm)}通过 IK 验证的目标姿态: {len(solutions)} 个")

    def auto_key(keycode: int) -> None:
        if keycode == glfw.KEY_ESCAPE:
            return

    try:
        if args.headless:
            viewer_context = None
        else:
            viewer_context = mujoco.viewer.launch_passive(model, data, key_callback=auto_key)
        if viewer_context is None:
            class HeadlessContext:
                def __enter__(self): return None
                def __exit__(self, *_args): return False
            viewer_context = HeadlessContext()
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = [0.38, 0.0, 0.34]
                viewer.cam.distance = 1.15
                viewer.cam.azimuth = 0
                viewer.cam.elevation = -55
                viewer.sync()
            while pose_index < len(solutions):
                if viewer is not None and not viewer.is_running():
                    break
                if preview is not None:
                    preview.poll()
                    if preview.closed:
                        break
                elapsed = sim_time - phase_start
                target_q = solutions[pose_index]
                if elapsed < args.move_s:
                    alpha = float(np.clip(elapsed / args.move_s, 0.0, 1.0))
                    current_q = q_start + alpha * (target_q - q_start)
                else:
                    current_q = target_q
                data.qpos[specs[arm].qpos] = current_q
                data.qvel[:] = 0.0
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                if elapsed >= args.move_s + args.settle_s:
                    recorder.capture(data, timestamp=sim_time)
                    pose_index += 1
                    q_start = target_q.copy()
                    phase_start = sim_time
                    print(f"[auto] captured {pose_index}/{len(solutions)}")
                if preview is not None and sim_time >= preview_time:
                    preview.show({name: render_rgb(renderer, data, name) for name in camera_names})
                    preview_time = sim_time + 0.10
                if viewer is not None:
                    viewer.sync()
                sim_time += tick_period
                wall_tick += tick_period
                time.sleep(max(0.0, wall_tick - time.perf_counter()))
    finally:
        recorder.save()
        if preview is not None:
            preview.close()
        renderer.close()
    if not recorder.time:
        raise RuntimeError("No automatic calibration samples were captured")
    return recorder.episode_dir


def collect_left(args: argparse.Namespace) -> Path:
    """Backward-compatible left-arm collector."""

    return collect_wrist(args, "left")


def main() -> None:
    args = build_parser().parse_args()
    if args.session is not None:
        session = args.session.resolve()
    else:
        session = collect_wrist(args, args.arm)
    if not args.no_solve:
        solve_wrist_session(session, args.arm, free_distortion=args.free_distortion)
    else:
        print(f"[自动采集] 未执行求解，数据目录: {session}")


if __name__ == "__main__":
    main()
