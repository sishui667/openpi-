"""Robot-only scene for live human motion. Frozen grasp XMLs remain untouched."""
import mujoco
import numpy as np
from member_b_scene import SCENE, repair_finger_servos
from teleop_towel_dataset import build_arm_specs, build_finger_actuators
from handeye_grasp_red_cube import finger_joint_indices


def build_motion_scene():
    spec=mujoco.MjSpec.from_file(str(SCENE))
    # Remove dependent contacts/welds before deleting the object. Finger mimic
    # constraints, robot collision meshes and all actuator physics are retained.
    for pair in list(spec.pairs):
        if 'droid_object_geom' in (pair.geomname1,pair.geomname2): spec.delete(pair)
    for equality in list(spec.equalities):
        if 'droid_object' in (equality.name1,equality.name2): spec.delete(equality)
    for key in list(spec.keys): spec.delete(key)
    spec.delete(spec.body('droid_object'))
    spec.delete(spec.geom('folding_table'))
    spec.delete(spec.site('cube_spawn_reference'))
    spec.delete(spec.camera('workspace_camera'))
    spec.modelname='openarm_live_motion_only'
    model=spec.compile()
    repair_finger_servos(model,build_finger_actuators(model))
    data=mujoco.MjData(model)
    specs=build_arm_specs(model)
    # Bent-elbow, neutral-wrist initial posture; no table/object or grasp target.
    for side in ('left','right'):
        data.qpos[specs[side].qpos]=[0,-.6 if side=='left' else .6,0,.9,0,0,0]
        data.qpos[finger_joint_indices(model,side)[0]]=.044
    mujoco.mj_forward(model,data)
    return model,data
