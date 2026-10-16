"""Bounded camera search when the initial chest RGB-D target is missing."""
from __future__ import annotations

import mujoco
import numpy as np

from calibration_simulation import body_transform, render_rgb_depth
from handeye_grasp_red_cube import (
    estimate_cube_center, latest_handeye_report, load_handeye,
    quat_from_body, solve_target_q,
)
from teleop_towel_dataset import apply_controller


def acquire_visual_target(model, data, specs, actuators, holds, cube_half,
                          finger_open, chest_transform, chest_matrix,
                          workspace_calibration, progress=None):
    """Return a camera observation, or a structured failure after finite search.

    Search waypoints are robot observation poses, not object coordinates.
    This function never reads the simulator object's pose or segmentation ID.
    """
    attempts = []

    def announce(message):
        print(message, flush=True)
        if progress:
            progress(message)

    def capture(side=None):
        camera = "chest_d435i_preview" if side is None else f"v22_{side}_d405_rgb"
        if side is None:
            transform, matrix = chest_transform, chest_matrix
            correction = workspace_calibration["chest"]
            height, width = 480, 640
        else:
            handeye, matrix = load_handeye(latest_handeye_report(side), side)
            transform = body_transform(data, specs[side].body_id) @ handeye
            correction = workspace_calibration["wrist"][side]
            height, width = 240, 320
        renderer = mujoco.Renderer(model, height, width)
        try:
            rgb, depth = render_rgb_depth(renderer, data, camera)
            observation = estimate_cube_center(
                rgb, depth, np.eye(4), transform, matrix, cube_half)
            raw = np.asarray(observation.center_world, dtype=float)
            center = raw.copy()
            center[:2] = np.asarray([raw[0], raw[1], 1.0]) @ np.asarray(
                correction["xy_coeff"])
            center[2] += float(correction.get("z_bias_m", 0.0))
            attempts.append({"camera": camera, "status": "detected",
                             "mask_pixels": observation.mask_pixels})
            return {"observation": observation, "raw_center": raw,
                    "center": center, "source": camera, "attempts": attempts}
        except RuntimeError as exc:
            attempts.append({"camera": camera, "status": "unavailable",
                             "reason": str(exc)})
            announce(f"Camera unavailable ({camera}): {exc}")
            return None
        finally:
            renderer.close()

    def move(side, goal):
        spec = specs[side]
        start = data.qpos[spec.qpos].copy()
        dt = float(model.opt.timestep)
        steps = int(2.0 / dt)
        for i in range(steps + int(0.8 / dt)):
            fraction = min(1.0, (i + 1) / steps)
            fraction = fraction * fraction * (3 - 2 * fraction)
            commands = dict(holds)
            commands[side] = start + fraction * (goal - start)
            apply_controller(model, data, specs, actuators, commands,
                             {"left": finger_open, "right": finger_open})
            mujoco.mj_step(model, data)
            if progress and i % max(1, int(0.04 / dt)) == 0:
                progress(f"CAMERA SEARCH: moving {side} arm")
        holds[side] = data.qpos[spec.qpos].copy()

    observation = capture()
    if observation is not None:
        return observation, attempts
    announce("Initial chest detection missing; searching with robot cameras")
    for side in ("right", "left"):
        spec = specs[side]
        quaternion = quat_from_body(model, data, spec.body_id)
        transform = body_transform(data, spec.body_id)
        rotation = transform[:3, :3].copy()
        raised = transform[:3, 3].copy()
        raised[2] = np.clip(raised[2] + 0.18, 0.68, 0.78)
        q_raise, *_ = solve_target_q(
            model, data, spec, raised, quaternion, holds[side],
            target_rotation=rotation)
        announce(f"CAMERA SEARCH: lift {side} arm to remove occlusion")
        move(side, q_raise)
        observation = capture()
        if observation is not None:
            return observation, attempts
        sign = 1.0 if side == "left" else -1.0
        q_clear, *_ = solve_target_q(
            model, data, spec, np.array([0.28, sign * 0.16, 0.72]),
            quaternion, holds[side], target_rotation=rotation)
        move(side, q_clear)
        q_view, *_ = solve_target_q(
            model, data, spec, np.array([0.25, sign * 0.08, 0.60]),
            quaternion, holds[side], target_rotation=rotation)
        announce(f"CAMERA SEARCH: observe with {side} wrist RGB-D")
        move(side, q_view)
        observation = capture(side)
        # Return to a high pose before another camera search or grasp planning.
        move(side, q_clear)
        if observation is not None:
            return observation, attempts
        observation = capture()
        if observation is not None:
            return observation, attempts
    announce("No valid camera target after chest and both wrist searches")
    return None, attempts
