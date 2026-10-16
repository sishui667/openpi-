"""Check that hidden simulator cube coordinates cannot drive the IK planner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from calibration_simulation import render_rgb_depth
from demo_grasp_viewer import (
    CUBE_HALF, FINGER_OPEN, PAD_CONTACT_PROXY_LOCAL, SCENE,
    apply_workspace_rgbd_calibration, load_chest_calibration,
    load_workspace_rgbd_calibration, place_cube_for_blind_test,
)
from handeye_grasp_red_cube import estimate_cube_center, finger_joint_indices
from teleop_towel_dataset import build_arm_specs
from visual_grasp_planner import plan_face_grasp
from task_workspace import load_task_model, TABLE_TOP


def verify():
    model = load_task_model(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    spec = build_arm_specs(model)["right"]
    fingers = tuple(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM,
        f"openarm_right_{side}_finger_collision") for side in ("right", "left"))
    seed = data.qpos[spec.qpos].copy()
    args = dict(
        model=model, data=data, spec=spec, finger_geoms=fingers,
        finger_indices=finger_joint_indices(model, "right"),
        proxies=PAD_CONTACT_PROXY_LOCAL,
        center=np.array([0.365, -0.060, TABLE_TOP + CUBE_HALF]),
        yaw=np.pi / 4, seed=seed, clear_q=seed, finger_open=FINGER_OPEN,
        approach_height=0.07, transit_height=0.16,
    )
    first = plan_face_grasp(**args)
    place_cube_for_blind_test(model, data, (0.8, 0.7, -20.0))
    hidden_changed = plan_face_grasp(**args)
    delta_hidden = max(float(np.max(np.abs(first[key] - hidden_changed[key])))
                       for key in ("q_grasp", "q_approach", "q_transit"))
    assert delta_hidden < 1e-12, "Hidden cube pose changed planned actions"
    assert first["candidates"] == hidden_changed["candidates"]

    args["center"] = np.array([0.405, -0.060, TABLE_TOP + CUBE_HALF])
    visual_changed = plan_face_grasp(**args)
    delta_visual = float(np.max(np.abs(
        first["q_grasp"] - visual_changed["q_grasp"])))
    assert delta_visual > 1e-3, "Planner did not respond to changed visual target"

    # Also check the actual image path: different fixture placements must
    # produce different RGB-D measurements.  Fixture coordinates are used
    # only to set the world, never supplied to estimate_cube_center().
    transform, intrinsics, _ = load_chest_calibration()
    correction = load_workspace_rgbd_calibration()["chest"]
    observed = []
    renderer = mujoco.Renderer(model, 480, 640)
    try:
        for fixture in ((0.330, -0.080, 0.0), (0.480, 0.140, 30.0)):
            place_cube_for_blind_test(model, data, fixture)
            rgb, depth = render_rgb_depth(renderer, data, "chest_d435i_preview")
            measurement = estimate_cube_center(
                rgb, depth, np.eye(4), transform, intrinsics, CUBE_HALF)
            observed.append(apply_workspace_rgbd_calibration(
                measurement.center_world, correction).tolist())
    finally:
        renderer.close()
    measured_motion = float(np.linalg.norm(
        np.asarray(observed[1])[:2] - np.asarray(observed[0])[:2]))
    assert measured_motion > 0.15, "Image perception did not follow the moved cube"
    return {
        "verdict": "PASS",
        "hidden_pose_changed_max_joint_delta_rad": delta_hidden,
        "visual_target_changed_max_joint_delta_rad": delta_visual,
        "chest_rgbd_observed_centers_m": observed,
        "camera_measured_position_change_m": measured_motion,
        "scope": "current visual planner and chest RGB-D detector; not historical Pose2Arm tools",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    result = verify()
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
