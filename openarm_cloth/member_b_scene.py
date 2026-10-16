"""Shared current Member B scene; one active arm, no object oracle in control."""
from pathlib import Path

import mujoco
import numpy as np

from task_workspace import load_task_model
from teleop_towel_dataset import build_arm_specs, build_finger_actuators

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "simulation/v1/droid_cube_scene_perception.xml"
TARGET_ZONE = {"center_xy_m": [0.43, -0.10], "radius_m": 0.03}


def repair_finger_servos(model, actuators):
    """Match the imported left motor to the right position servo; no XML edit."""
    left, right = actuators["left"], actuators["right"]
    for attr in ("actuator_gaintype", "actuator_gainprm", "actuator_biastype",
                 "actuator_biasprm", "actuator_forcerange"):
        getattr(model, attr)[left] = getattr(model, attr)[right]


def build_scene():
    from calibration_simulation import body_transform
    from handeye_grasp_red_cube import (solve_target_q, quat_from_body,
                                       finger_joint_indices)
    model = load_task_model(SCENE, target_zone=TARGET_ZONE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    specs = build_arm_specs(model)
    repair_finger_servos(model, build_finger_actuators(model))
    mujoco.mj_forward(model, data)
    # Identical fixed high standby used by the current visual grasp demo.
    # Robot reset conditions only: no object position is read.
    for side in ("left", "right"):
        spec = specs[side]
        rotation = body_transform(data, spec.body_id)[:3, :3].copy()
        quaternion = quat_from_body(model, data, spec.body_id)
        sign = 1.0 if side == "left" else -1.0
        q, *_ = solve_target_q(
            model, data, spec, np.array([0.28, sign * 0.16, 0.68]),
            quaternion, data.qpos[spec.qpos].copy(), target_rotation=rotation)
        data.qpos[spec.qpos] = q
        data.qpos[finger_joint_indices(model, side)[0]] = 0.044
        mujoco.mj_forward(model, data)
    data.qvel[:] = 0
    return model, data
