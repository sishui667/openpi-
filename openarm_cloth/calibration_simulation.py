# -*- coding: utf-8 -*-
"""Interactive MuJoCo calibration-board simulation and synchronized recorder.

This tool is the first stage of the camera/hand-eye calibration workflow.  It
adds a fixed checkerboard to the OpenArm scene, lets the operator move either
arm with the existing Cartesian keyboard controls, and records synchronized
RGB, depth, robot joint state, TCP pose, board pose, and camera ground truth.

Run in the openpi environment::

    python calibration_simulation.py --preview

Keyboard controls:

    1 / 2       select left / right arm
    I/K J/L     move TCP in X/Y
    U/O         move TCP in Z
    Q/E Z/C     yaw / pitch
    X/V         roll
    N           capture one synchronized calibration sample
    SPACE       start/stop continuous recording
    R           reset robot to ready pose
    P           pause/resume physics
    F1          print controls
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from build_calibration_scene import board_metadata, ensure_scene
from teleop_towel_dataset import (
    ARM_Q_NAMES,
    ArmSpec,
    ControlPanel,
    POSE_ACTIONS,
    apply_controller,
    body_pose,
    build_arm_specs,
    build_finger_actuators,
    id_for,
    ik_step,
    reset_to_ready,
    safety_reason,
    TeleopState,
)


ROOT = Path(__file__).resolve().parent
SIDES = ("left", "right")
CAMERA_DEFAULTS = (
    "chest_d435i_preview",
    "v22_left_d405_rgb",
    "v22_right_d405_rgb",
)
CV_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])


def transform_from_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def body_transform(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return transform_from_pose(data.xpos[body_id], data.xmat[body_id])


def camera_transform(data: mujoco.MjData, camera_id: int) -> np.ndarray:
    return transform_from_pose(data.cam_xpos[camera_id], data.cam_xmat[camera_id])


def pose7(transform: np.ndarray) -> np.ndarray:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(transform[:3, :3], dtype=np.float64).reshape(-1))
    return np.concatenate([transform[:3, 3], quaternion])


def camera_intrinsics(model: mujoco.MjModel, camera_id: int, width: int, height: int) -> dict[str, Any]:
    fovy_deg = float(model.cam_fovy[camera_id])
    fy = 0.5 * float(height) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": (float(width) - 1.0) / 2.0,
        "cy": (float(height) - 1.0) / 2.0,
        "distortion_model": "none",
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "fovy_deg": fovy_deg,
        "depth_units": "meter",
        "source": "MuJoCo pinhole renderer",
    }


def render_rgb_depth(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = np.ascontiguousarray(renderer.render(), dtype=np.uint8).copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = np.ascontiguousarray(renderer.render(), dtype=np.float32).copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def render_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    return np.ascontiguousarray(renderer.render(), dtype=np.uint8).copy()


class CameraPreview:
    def __init__(self, camera_names: tuple[str, ...], width: int, height: int) -> None:
        import tkinter as tk
        from PIL import ImageTk

        self.tk = tk
        self.image_tk = ImageTk
        self.closed = False
        self.root = tk.Tk()
        self.root.withdraw()
        self.camera_names = tuple(camera_names)
        self.windows: list[Any] = []
        self.labels: list[Any] = []
        self.photos: list[Any] = []
        display_width = max(320, int(width))
        display_height = max(240, int(height))
        for camera_name in camera_names:
            window = tk.Toplevel(self.root)
            window.title(f"Calibration | {camera_name}")
            window.geometry(f"{display_width}x{display_height}")
            window.protocol("WM_DELETE_WINDOW", self.close)
            label = tk.Label(window, bg="black")
            label.pack(fill=tk.BOTH, expand=True)
            self.windows.append(window)
            self.labels.append(label)
        self.root.update_idletasks()
        self.root.update()

    def show(self, images: dict[str, np.ndarray]) -> None:
        if self.closed:
            return
        try:
            self.photos = []
            for camera_name, label, window in zip(self.camera_names, self.labels, self.windows):
                image = Image.fromarray(images[camera_name])
                size = (max(1, window.winfo_width()), max(1, window.winfo_height()))
                image = image.resize(size, Image.Resampling.NEAREST)
                photo = self.image_tk.PhotoImage(image=image)
                label.configure(image=photo)
                label.image = photo
                self.photos.append(photo)
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def poll(self) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass


class CalibrationRecorder:
    def __init__(
        self,
        output_root: Path,
        model: mujoco.MjModel,
        camera_names: tuple[str, ...],
        camera_ids: dict[str, int],
        arm_specs: dict[str, ArmSpec],
        board_body_id: int,
        renderer: mujoco.Renderer,
        image_width: int,
        image_height: int,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.episode_dir = output_root / f"session_{stamp}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.model = model
        self.camera_names = camera_names
        self.camera_ids = camera_ids
        self.arm_specs = arm_specs
        self.board_body_id = board_body_id
        self.renderer = renderer
        self.image_width = image_width
        self.image_height = image_height
        self.frame_index = 0
        self.time: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.tcp_pose: list[np.ndarray] = []
        self.board_pose: list[np.ndarray] = []
        self.board_corners_world: list[np.ndarray] = []
        self.camera_pose_mujoco: list[np.ndarray] = []
        self.camera_pose_opencv: list[np.ndarray] = []
        self.rgb_paths: list[list[str]] = []
        self.depth_paths: list[list[str]] = []
        self.metadata = {
            "format": "openarm_sim_calibration_v1",
            "scene": "calibration_board_scene.xml",
            "mujoco_version": mujoco.__version__,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "timestep_s": float(model.opt.timestep),
            "camera_order": list(camera_names),
            "camera_coordinate_convention": {
                "mujoco": "local +X right, +Y up, camera looks along -Z",
                "opencv": "+X right, +Y down, +Z forward",
                "T_world_camera_opencv": "T_world_camera_mujoco @ diag(1,-1,-1,1)",
            },
            "camera_intrinsics": {
                name: camera_intrinsics(model, camera_ids[name], image_width, image_height)
                for name in camera_names
            },
            "board": board_metadata(),
            "arm_joint_order": ARM_Q_NAMES,
            "tcp_body": {
                side: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, arm_specs[side].body_id)
                for side in SIDES
            },
            "rgb_depth": {
                "same_simulated_pinhole_camera": True,
                "depth_scale": 1.0,
                "depth_file_format": "NumPy float32 meters",
            },
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for camera_name in camera_names:
            (self.episode_dir / "images" / camera_name / "rgb").mkdir(parents=True, exist_ok=True)
            (self.episode_dir / "images" / camera_name / "depth").mkdir(parents=True, exist_ok=True)
        print(f"[标定记录] 输出目录: {self.episode_dir}")

    def capture(self, data: mujoco.MjData, timestamp: float | None = None) -> None:
        index = self.frame_index
        camera_rgb_paths: list[str] = []
        camera_depth_paths: list[str] = []
        rgb_for_preview: dict[str, np.ndarray] = {}
        camera_mj: list[np.ndarray] = []
        camera_cv: list[np.ndarray] = []
        for camera_name in self.camera_names:
            rgb, depth = render_rgb_depth(self.renderer, data, camera_name)
            rgb_for_preview[camera_name] = rgb
            rgb_path = self.episode_dir / "images" / camera_name / "rgb" / f"{index:06d}.png"
            depth_path = self.episode_dir / "images" / camera_name / "depth" / f"{index:06d}.npy"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)
            camera_rgb_paths.append(str(rgb_path.relative_to(self.episode_dir)))
            camera_depth_paths.append(str(depth_path.relative_to(self.episode_dir)))
            transform_mj = camera_transform(data, self.camera_ids[camera_name])
            camera_mj.append(transform_mj)
            camera_cv.append(transform_mj @ CV_FROM_MUJOCO)

        board_transform = body_transform(data, self.board_body_id)
        board_points = np.asarray(self.metadata["board"]["inner_corners_board_xyz_m"], dtype=np.float64)
        board_points_h = np.concatenate(
            [board_points, np.ones((len(board_points), 1), dtype=np.float64)], axis=1
        )
        board_world_points = (board_transform @ board_points_h.T).T[:, :3]
        sample_time = float(data.time if timestamp is None else timestamp)
        self.time.append(sample_time)
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.ctrl.append(data.ctrl.copy())
        self.tcp_pose.append(np.asarray([body_transform(data, self.arm_specs[side].body_id) for side in SIDES]))
        self.board_pose.append(board_transform)
        self.board_corners_world.append(board_world_points)
        self.camera_pose_mujoco.append(np.asarray(camera_mj))
        self.camera_pose_opencv.append(np.asarray(camera_cv))
        self.rgb_paths.append(camera_rgb_paths)
        self.depth_paths.append(camera_depth_paths)
        self.frame_index += 1
        print(
            f"[标定记录] 样本 {index:04d} | t={sample_time:.3f}s | "
            f"TCP左={np.round(data.xpos[self.arm_specs['left'].body_id], 4).tolist()} | "
            f"TCP右={np.round(data.xpos[self.arm_specs['right'].body_id], 4).tolist()}"
        )
        self.last_rgb = rgb_for_preview

    def save(self) -> None:
        if not self.time:
            print("[标定记录] 没有样本，未生成 samples.npz")
            return
        arrays = {
            "time_s": np.asarray(self.time, dtype=np.float64),
            "qpos": np.asarray(self.qpos, dtype=np.float64),
            "qvel": np.asarray(self.qvel, dtype=np.float64),
            "ctrl": np.asarray(self.ctrl, dtype=np.float64),
            "tcp_pose_world": np.asarray(self.tcp_pose, dtype=np.float64),
            "board_pose_world": np.asarray(self.board_pose, dtype=np.float64),
            "board_inner_corners_world": np.asarray(self.board_corners_world, dtype=np.float64),
            "camera_pose_world_mujoco": np.asarray(self.camera_pose_mujoco, dtype=np.float64),
            "camera_pose_world_opencv": np.asarray(self.camera_pose_opencv, dtype=np.float64),
        }
        np.savez_compressed(self.episode_dir / "samples.npz", **arrays)
        records = [
            {
                "sample_index": index,
                "time_s": self.time[index],
                "rgb": self.rgb_paths[index],
                "depth": self.depth_paths[index],
            }
            for index in range(len(self.time))
        ]
        (self.episode_dir / "frames.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.metadata = dict(self.metadata)
        self.metadata.update({"frames": len(self.time), "duration_s": self.time[-1] - self.time[0]})
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[标定记录] 已保存 {len(self.time)} 个样本: {self.episode_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=None, help="calibration XML; default is generated scene")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "calibration" / "sim_data")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERA_DEFAULTS))
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--record-hz", type=float, default=5.0)
    parser.add_argument("--record", action="store_true", help="start continuous recording immediately")
    parser.add_argument("--preview", action="store_true", help="show RGB previews in separate windows")
    parser.add_argument("--gui", action="store_true", help="show the existing Cartesian button panel")
    parser.add_argument("--headless", action="store_true", help="capture static samples without opening a viewer")
    parser.add_argument("--samples", type=int, default=1, help="headless sample count")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.image_width <= 0 or args.image_height <= 0 or args.record_hz <= 0 or args.samples <= 0:
        raise ValueError("image dimensions, record frequency, and samples must be positive")
    scene = args.scene or ensure_scene()
    scene = scene if scene.is_absolute() else (ROOT / scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = id_for(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready")
    reset_to_ready(model, data, key_id)
    arm_specs = build_arm_specs(model)
    finger_actuators = build_finger_actuators(model)
    camera_names = tuple(args.cameras)
    camera_ids = {
        name: id_for(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in camera_names
    }
    board_body_id = id_for(model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board")
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    recorder = CalibrationRecorder(
        args.output_dir,
        model,
        camera_names,
        camera_ids,
        arm_specs,
        board_body_id,
        renderer,
        args.image_width,
        args.image_height,
    )
    if args.headless:
        for _ in range(args.samples):
            recorder.capture(data)
        recorder.save()
        renderer.close()
        return 0

    state = TeleopState(model, data, arm_specs)
    q_target = {side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()}
    dirty_sides: set[str] = set()
    preview = CameraPreview(camera_names, args.image_width, args.image_height) if args.preview else None
    panel = ControlPanel(state) if args.gui else None
    capture_requested = False
    recording = bool(args.record)
    sample_period = 1.0 / args.record_hz
    next_sample_time = float(data.time)
    next_preview_time = float(data.time)
    print(f"[场景] {scene}")
    print("[标定板] 8×6 棋盘格，内角点 7×5，方格边长 35 mm")
    print("[控制] 1/2 选左右臂；N 手动采样；SPACE 连续记录；R 复位；P 暂停；F1 帮助")
    wall_tick = time.perf_counter()
    tick_period = 0.01
    steps_per_tick = max(1, int(round(tick_period / float(model.opt.timestep))))

    def on_key(keycode: int) -> None:
        nonlocal capture_requested, recording, next_sample_time
        if keycode == glfw.KEY_N:
            capture_requested = True
            return
        if keycode == glfw.KEY_SPACE:
            recording = not recording
            next_sample_time = float(data.time)
            print(f"[标定记录] {'开始' if recording else '暂停'}连续记录")
            return
        state.on_key(keycode)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            viewer.cam.lookat[:] = [0.38, 0.0, 0.34]
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -55
            viewer.sync()
            while viewer.is_running() and not (preview and preview.closed):
                if panel is not None:
                    panel.poll()
                if preview is not None:
                    preview.poll()
                for event in state.drain_events():
                    if event["action"] in POSE_ACTIONS:
                        if event["active_arm"] == "both":
                            dirty_sides.update(SIDES)
                        else:
                            dirty_sides.add(event["active_arm"])
                for command in state.drain_commands():
                    if command == "toggle_record":
                        recording = not recording
                        next_sample_time = float(data.time)
                        print(f"[标定记录] {'开始' if recording else '暂停'}连续记录")
                    elif command == "reset":
                        reset_to_ready(model, data, key_id)
                        q_target = {side: data.qpos[spec.qpos].copy() for side, spec in arm_specs.items()}
                        state.reset_targets(model, data, arm_specs)
                        dirty_sides.clear()
                    elif command == "help":
                        print("[帮助] N=单帧采样；SPACE=连续记录；移动和姿态控制沿用键盘遥操作")

                snapshot = state.snapshot()
                for side in dirty_sides:
                    spec = arm_specs[side]
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
                        apply_controller(model, data, arm_specs, finger_actuators, q_target, snapshot["gripper"])
                        mujoco.mj_step(model, data)

                if capture_requested or (recording and data.time + 1e-9 >= next_sample_time):
                    recorder.capture(data)
                    capture_requested = False
                    next_sample_time = float(data.time) + sample_period
                    if preview is not None:
                        preview.show(recorder.last_rgb)
                if preview is not None and data.time + 1e-9 >= next_preview_time:
                    preview_images = {name: render_rgb(renderer, data, name) for name in camera_names}
                    preview.show(preview_images)
                    next_preview_time = float(data.time) + 0.10

                reason = safety_reason(data, arm_specs)
                if reason is not None and not snapshot["paused"]:
                    state.emergency_pause(reason)
                    print(f"[安全暂停] {reason}")
                viewer.sync()
                wall_tick += tick_period
                time.sleep(max(0.0, wall_tick - time.perf_counter()))
    finally:
        recorder.save()
        if preview is not None:
            preview.close()
        if panel is not None:
            panel.close()
        renderer.close()
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
