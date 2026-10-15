# -*- coding: utf-8 -*-
"""Automatically calibrate the fixed chest RGB-D camera in simulation.

Unlike a wrist camera, the chest camera is fixed to the robot body.  The
calibration target is therefore moved through the chest-camera field of view;
the result is the fixed transform ``T_base_chest_camera_opencv`` rather than a
TCP-to-camera hand-eye transform.

Run::

    conda activate openpi
    cd D:\\openpi\\openarm_cloth
    python .\\auto_calibrate_chest.py --preview
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from auto_calibrate_left import (
    camera_local_rotation,
    detect_corners,
    detect_corners_image,
    make_transform,
    matrix_to_quaternion,
    rotation_error_deg,
)
from build_calibration_scene import ensure_scene
from calibration_simulation import (
    CalibrationRecorder,
    CameraPreview,
    camera_transform,
    render_rgb,
)
from teleop_towel_dataset import build_arm_specs, id_for, reset_to_ready


ROOT = Path(__file__).resolve().parent
CAMERA_NAME = "chest_d435i_preview"
PATTERN_SIZE = (8, 5)


def set_board_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    board_body_id: int,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    """Move the static visual board through model body pose at runtime."""

    model.body_pos[board_body_id] = np.asarray(position, dtype=np.float64)
    model.body_quat[board_body_id] = np.asarray(quaternion, dtype=np.float64)
    mujoco.mj_forward(model, data)


def board_rotation_towards_camera(
    board_position: np.ndarray,
    camera_position: np.ndarray,
    camera_right: np.ndarray,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Build a board frame whose front (+Z) faces the chest camera."""

    normal = np.asarray(camera_position, dtype=np.float64) - np.asarray(board_position, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    x_axis = np.asarray(camera_right, dtype=np.float64) - normal * float(np.dot(camera_right, normal))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.asarray([1.0, 0.0, 0.0]) - normal * float(normal[0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    base_rotation = np.column_stack([x_axis, y_axis, normal])
    return base_rotation @ camera_local_rotation(roll_deg, pitch_deg, yaw_deg)


def make_chest_board_targets(
    data: mujoco.MjData,
    board_body_id: int,
    camera_id: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create board poses with translation, distance, and orientation diversity."""

    camera_pose = camera_transform(data, camera_id)
    camera_position = camera_pose[:3, 3]
    camera_right = camera_pose[:3, 0]
    camera_up = camera_pose[:3, 1]
    camera_forward = camera_pose[:3, :3] @ np.asarray([0.0, 0.0, -1.0])
    surface_z = 0.005
    layouts = [
        (0.00, 0.00, 0.0, 0.0, 0.0),
        (-0.07, 0.00, 0.0, 0.0, 0.0),
        (0.07, 0.00, 0.0, 0.0, 0.0),
        (0.00, -0.045, 0.0, 0.0, 0.0),
        (0.00, 0.045, 0.0, 0.0, 0.0),
        (0.00, 0.00, -8.0, 0.0, 0.0),
        (0.00, 0.00, 8.0, 0.0, 0.0),
        (0.00, 0.00, 0.0, -8.0, 0.0),
        (0.00, 0.00, 0.0, 8.0, 0.0),
    ]
    targets: list[tuple[np.ndarray, np.ndarray]] = []
    for distance in (0.42, 0.50, 0.58):
        for right_offset, up_offset, roll, pitch, yaw in layouts:
            surface_center = (
                camera_position
                + camera_forward * distance
                + camera_right * right_offset
                + camera_up * up_offset
            )
            # Keep the board above the tabletop after rotating it to face the
            # chest camera; otherwise the lower half can be occluded by the
            # table in the rendered view.
            surface_center[2] = 0.45 + float(up_offset)
            rotation = board_rotation_towards_camera(
                surface_center, camera_position, camera_right, roll, pitch, yaw
            )
            body_position = surface_center - rotation @ np.asarray([0.0, 0.0, surface_z])
            targets.append((body_position, matrix_to_quaternion(rotation)))
    return targets


def intrinsic_matrix(intrinsic: dict) -> np.ndarray:
    return np.asarray(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def solve_chest_session(session_dir: Path, free_distortion: bool = False) -> Path:
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    board = metadata["board"]
    pattern_size = (int(board["inner_corners_columns"]), int(board["inner_corners_rows"]))
    object_points = np.asarray(board["inner_corners_board_xyz_m"], dtype=np.float32)
    board_surface_z = float(object_points[0, 2])
    calibration_points = object_points.copy()
    calibration_points[:, 2] = 0.0
    camera_names = metadata["camera_order"]
    if CAMERA_NAME not in camera_names:
        raise RuntimeError(f"Session does not contain {CAMERA_NAME}: {camera_names}")
    camera_index = camera_names.index(CAMERA_NAME)
    samples = np.load(session_dir / "samples.npz")
    frame_count = len(samples["time_s"])
    detected_indices: list[int] = []
    image_points: list[np.ndarray] = []
    object_point_sets: list[np.ndarray] = []
    intrinsic_seed = metadata["camera_intrinsics"][CAMERA_NAME]
    image_size = (int(intrinsic_seed["width"]), int(intrinsic_seed["height"]))
    for index in range(frame_count):
        image_path = session_dir / "images" / CAMERA_NAME / "rgb" / f"{index:06d}.png"
        corners = detect_corners(image_path, pattern_size)
        if corners is None:
            continue
        detected_indices.append(index)
        image_points.append(corners)
        object_point_sets.append(calibration_points.copy())

    if len(detected_indices) < 12:
        raise RuntimeError(
            f"Only {len(detected_indices)}/{frame_count} chest-camera checkerboards detected; "
            "need at least 12 valid views."
        )

    K_seed = intrinsic_matrix(intrinsic_seed)
    if free_distortion:
        rms, K, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
            object_point_sets, image_points, image_size, None, None
        )
        intrinsics_source = "opencv_free_calibration"
    else:
        # MuJoCo uses an ideal pinhole renderer.  Keep its known K and zero
        # distortion instead of fitting synthetic radial distortion.
        K = K_seed
        distortion = np.zeros((5, 1), dtype=np.float64)
        intrinsics_source = "simulation_metadata_fixed_pinhole"
        reprojection_errors: list[float] = []
        for corners in image_points:
            ok, rvec, tvec = cv2.solvePnP(
                calibration_points, corners, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue
            projected, _ = cv2.projectPoints(calibration_points, rvec, tvec, K, distortion)
            reprojection_errors.extend(
                np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).tolist()
            )
        rms = float(np.sqrt(np.mean(np.square(reprojection_errors))))

    board_surface_offset = np.eye(4, dtype=np.float64)
    board_surface_offset[2, 3] = board_surface_z
    world_camera_estimates: list[np.ndarray] = []
    depth_rmse_values: list[float] = []
    depth_valid_ratios: list[float] = []
    for index, corners in zip(detected_indices, image_points):
        ok, rvec, tvec = cv2.solvePnP(
            calibration_points, corners, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_from_board_surface = make_transform(rotation, tvec.reshape(3))
        world_board_surface = samples["board_pose_world"][index] @ board_surface_offset
        world_camera_estimates.append(world_board_surface @ np.linalg.inv(camera_from_board_surface))

        depth_path = session_dir / "images" / CAMERA_NAME / "depth" / f"{index:06d}.npy"
        depth = np.load(depth_path)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        measured: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        for point, pixel in zip(calibration_points, corners.reshape(-1, 2)):
            u, v = int(round(float(pixel[0]))), int(round(float(pixel[1])))
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue
            value = float(depth[v, u])
            if not np.isfinite(value) or value <= 1e-6:
                continue
            measured.append(np.asarray([(u - cx) * value / fx, (v - cy) * value / fy, value]))
            predicted.append((camera_from_board_surface @ np.r_[point, 1.0])[:3])
        if measured:
            depth_rmse_values.append(
                float(
                    np.sqrt(
                        np.mean(np.sum((np.asarray(measured) - np.asarray(predicted)) ** 2, axis=1))
                    )
                    * 1000.0
                )
            )
            depth_valid_ratios.append(float(len(measured) / len(calibration_points)))

    if len(world_camera_estimates) < 12:
        raise RuntimeError("Not enough valid chest-camera PnP poses for extrinsic calibration")
    translations = np.asarray([pose[:3, 3] for pose in world_camera_estimates])
    mean_rotation = Rotation.from_matrix(
        np.asarray([pose[:3, :3] for pose in world_camera_estimates])
    ).mean().as_matrix()
    world_camera = make_transform(mean_rotation, np.median(translations, axis=0))
    translation_consistency = [
        float(np.linalg.norm(pose[:3, 3] - world_camera[:3, 3]))
        for pose in world_camera_estimates
    ]
    rotation_consistency = [
        rotation_error_deg(pose[:3, :3], world_camera[:3, :3]) for pose in world_camera_estimates
    ]
    ground_truth_camera = samples["camera_pose_world_opencv"][detected_indices[0], camera_index]
    translation_error_mm = float(np.linalg.norm(world_camera[:3, 3] - ground_truth_camera[:3, 3]) * 1000.0)
    rotation_error = rotation_error_deg(world_camera[:3, :3], ground_truth_camera[:3, :3])
    report = {
        "format": "openarm_chest_camera_calibration_v1",
        "session": str(session_dir.resolve()),
        "camera": CAMERA_NAME,
        "detector": "cv2.findChessboardCornersSB",
        "pattern_size_inner_corners": list(pattern_size),
        "total_frames": frame_count,
        "detected_frames": len(detected_indices),
        "detected_indices": detected_indices,
        "intrinsics_rms_reprojection_px": float(rms),
        "intrinsics_source": intrinsics_source,
        "camera_matrix": K.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "T_base_chest_camera_opencv": world_camera.tolist(),
        "world_camera_translation_consistency_rmse_mm": float(
            np.sqrt(np.mean(np.square(translation_consistency))) * 1000.0
        ),
        "world_camera_translation_consistency_median_mm": float(np.median(translation_consistency) * 1000.0),
        "world_camera_rotation_consistency_median_deg": float(np.median(rotation_consistency)),
        "depth_corner_rmse_median_mm": float(np.median(depth_rmse_values)) if depth_rmse_values else None,
        "depth_valid_ratio_median": float(np.median(depth_valid_ratios)) if depth_valid_ratios else None,
        "simulation_ground_truth_check": {
            "translation_error_mm": translation_error_mm,
            "rotation_error_deg": rotation_error,
            "T_base_chest_camera_opencv_ground_truth": ground_truth_camera.tolist(),
        },
    }
    output = session_dir / "calibration_chest.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[胸部相机标定] 检测 {len(detected_indices)}/{frame_count} 帧 | "
        f"内参 RMS={rms:.4f}px | 外参真值误差: "
        f"平移={translation_error_mm:.2f}mm, 旋转={rotation_error:.3f}°"
    )
    print(f"[胸部相机标定] 结果: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "calibration" / "auto_chest")
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--move-s", type=float, default=0.30)
    parser.add_argument("--settle-s", type=float, default=0.50)
    parser.add_argument("--pose-count", type=int, default=20)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--no-solve", action="store_true")
    parser.add_argument("--free-distortion", action="store_true")
    return parser


def collect_chest(args: argparse.Namespace) -> Path:
    if args.move_s <= 0.0 or args.settle_s <= 0.0 or args.pose_count < 12:
        raise ValueError("move-s and settle-s must be positive; pose-count must be at least 12")
    scene = ensure_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    reset_to_ready(model, data, id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready"))
    specs = build_arm_specs(model)
    camera_id = id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    board_body_id = id_for(model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board")
    initial_position = model.body_pos[board_body_id].copy()
    initial_quaternion = model.body_quat[board_body_id].copy()
    targets = make_chest_board_targets(data, board_body_id, camera_id)
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    valid_targets: list[tuple[np.ndarray, np.ndarray]] = []
    for position, quaternion in targets:
        set_board_pose(model, data, board_body_id, position, quaternion)
        image = render_rgb(renderer, data, CAMERA_NAME)
        if detect_corners_image(image, PATTERN_SIZE) is not None:
            valid_targets.append((position.copy(), quaternion.copy()))
        if len(valid_targets) >= args.pose_count:
            break
    if len(valid_targets) < 12:
        renderer.close()
        raise RuntimeError(f"Only {len(valid_targets)} chest-camera board views were found; need at least 12")
    set_board_pose(model, data, board_body_id, initial_position, initial_quaternion)
    recorder = CalibrationRecorder(
        args.output_dir,
        model,
        (CAMERA_NAME,),
        {CAMERA_NAME: camera_id},
        specs,
        board_body_id,
        renderer,
        args.image_width,
        args.image_height,
    )
    preview = CameraPreview((CAMERA_NAME,), args.image_width, args.image_height) if args.preview else None
    start_position = initial_position.copy()
    start_quaternion = initial_quaternion.copy()
    pose_index = 0
    phase_start = 0.0
    sim_time = 0.0
    preview_time = -float("inf")
    tick_period = 0.01
    wall_tick = time.perf_counter()
    print(f"[自动采集] 胸部相机通过检测的标定板姿态: {len(valid_targets)} 个")

    def auto_key(_keycode: int) -> None:
        return

    try:
        viewer_context = None if args.headless else mujoco.viewer.launch_passive(
            model, data, key_callback=auto_key
        )
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
            while pose_index < len(valid_targets):
                if viewer is not None and not viewer.is_running():
                    break
                if preview is not None:
                    preview.poll()
                    if preview.closed:
                        break
                elapsed = sim_time - phase_start
                target_position, target_quaternion = valid_targets[pose_index]
                if elapsed < args.move_s:
                    alpha = float(np.clip(elapsed / args.move_s, 0.0, 1.0))
                    position = start_position + alpha * (target_position - start_position)
                    quaternion = (1.0 - alpha) * start_quaternion + alpha * target_quaternion
                    quaternion /= np.linalg.norm(quaternion)
                else:
                    position = target_position
                    quaternion = target_quaternion
                set_board_pose(model, data, board_body_id, position, quaternion)
                data.qvel[:] = 0.0
                data.ctrl[:] = 0.0
                if elapsed >= args.move_s + args.settle_s:
                    recorder.capture(data, timestamp=sim_time)
                    pose_index += 1
                    start_position = target_position.copy()
                    start_quaternion = target_quaternion.copy()
                    phase_start = sim_time
                    print(f"[auto] captured {pose_index}/{len(valid_targets)}")
                if preview is not None and sim_time >= preview_time:
                    preview.show({CAMERA_NAME: render_rgb(renderer, data, CAMERA_NAME)})
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
        raise RuntimeError("No automatic chest-camera samples were captured")
    return recorder.episode_dir


def main() -> None:
    args = build_parser().parse_args()
    session = args.session.resolve() if args.session is not None else collect_chest(args)
    if not args.no_solve:
        solve_chest_session(session, free_distortion=args.free_distortion)
    else:
        print(f"[自动采集] 未执行求解，数据目录: {session}")


if __name__ == "__main__":
    main()
