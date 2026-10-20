"""Camera-observed geometry adapter for the existing physical-pad IK planner.

The query target's type, size and pose come exclusively from observation.
Robot encoders are copied by robot joint name. No execution object pose,
geometry, mass, colour or segmentation identifier is consumed here.
"""
from __future__ import annotations

import mujoco
import numpy as np

from generic_object_scene import build_object_model, validate_geometry
from handeye_grasp_red_cube import finger_joint_indices
from teleop_towel_dataset import build_arm_specs
from visual_grasp_planner import plan_face_grasp


PAD_CONTACT_PROXY_LOCAL = (
    np.asarray([-.00018, -.00798, .045]),
    np.asarray([.00018, -.00798, .045]),
)


def validate_query_motion(plan,start,end,observation=None,finger_position=None,ignore_target_contacts=False):
    """Sample a revised motion in the private measured-geometry model only.

    This does not guarantee continuous collision-free motion. Existing robot
    seam contacts are baseline-subtracted; target penetration is not excused
    during open-finger descent. Closed-grip lift may ignore target contacts,
    but still checks robot/table/self contacts; physical grip is evaluated later.
    """
    model,data,spec=plan['query_model'],plan['query_data'],plan['query_spec']
    start=np.asarray(start,dtype=float);end=np.asarray(end,dtype=float)
    if start.shape!=(7,) or end.shape!=(7,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError('Motion endpoints must be finite seven-joint vectors')
    if np.any(end<spec.q_range[:,0]) or np.any(end>spec.q_range[:,1]):
        raise RuntimeError('Revised IK exceeds actual robot joint limits')
    if observation is not None:
        address=int(model.joint('droid_object_freejoint').qposadr[0]);yaw=observation.yaw_world_rad
        data.qpos[address:address+3]=observation.center_world
        data.qpos[address+3:address+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        plan['query_geometry']['center_world']=observation.center_world.tolist()
    saved=data.qpos.copy();object_geom=model.geom('droid_object_geom').id
    if finger_position is not None: data.qpos[plan['query_finger_indices'][0]]=finger_position
    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,plan['query_finger_geoms'][0])
    prefix='_'.join(name.split('_')[:2])+'_'
    active={g for g in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith(prefix)}
    def contacts(q):
        data.qpos[spec.qpos]=q;mujoco.mj_forward(model,data);result={}
        for contact in data.contact:
            pair=tuple(sorted((int(contact.geom1),int(contact.geom2))))
            if not active.intersection(pair) or contact.dist>=-.0005: continue
            if ignore_target_contacts and object_geom in pair: continue
            result[pair]=max(result.get(pair,0.),float(-contact.dist))
        return result
    try:
        baseline={pair:depth for pair,depth in contacts(start).items() if object_geom not in pair}
        samples=max(21,int(np.ceil(np.max(np.abs(end-start))/.025))+1);maximum=0.
        for blend in np.linspace(0,1,samples):
            for pair,depth in contacts(start+(end-start)*blend).items():
                maximum=max(maximum,max(0.,depth-baseline.get(pair,0.)))
        record=dict(max_new_penetration_mm=maximum*1000,samples=samples,
                    target_contacts_ignored=bool(ignore_target_contacts))
        if maximum>.001: raise RuntimeError(f'Revised motion has {maximum*1000:.2f} mm of new penetration')
        return record
    finally:
        data.qpos[:]=saved;mujoco.mj_forward(model,data)


def plan_observed_grasp(robot_model, robot_data, arm, observation, seed,
                        clear_q, finger_open=.044, approach_height=.07,
                        transit_height=.16, progress=None):
    """Plan using an isolated, vision-sized query model.

    Boxes use the visible face normals that physically fit the gripper. For a
    cylinder four horizontal radial axes are searched, not every radial angle.
    The returned route remains a candidate, not a grasp-success guarantee.
    """
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    shape, size = validate_geometry(observation.shape, observation.size_xyz)
    center = np.asarray(observation.center_world, dtype=float)
    yaw = float(observation.yaw_world_rad)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
        raise ValueError("Visual center and yaw must be finite")
    seed = np.asarray(seed, dtype=float)
    clear_q = np.asarray(clear_q, dtype=float)
    if seed.shape != (7,) or clear_q.shape != (7,) or not np.isfinite(seed).all() or not np.isfinite(clear_q).all():
        raise ValueError("seed and clear_q must be finite seven-joint vectors")
    finger_open = float(finger_open)
    if not np.isfinite(finger_open) or not 0 <= finger_open <= .044:
        raise ValueError("finger_open must respect the physical [0, .044] m joint range")

    # Do not clone execution target geometry. Compiling from the frozen robot
    # definition plus the observation closes the hidden-size information path.
    query_model = build_object_model(shape, size, rgba=(.5, .5, .5, 1))
    query_data = mujoco.MjData(query_model)
    for destination in range(query_model.njnt):
        name = mujoco.mj_id2name(query_model, mujoco.mjtObj.mjOBJ_JOINT, destination) or ""
        if not name.startswith("openarm_"):
            continue
        source = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if source < 0:
            raise ValueError(f"Execution robot has no joint {name}")
        # The OpenArm contract uses only scalar hinges/slides; don't copy free
        # joints or arbitrary contiguous qpos blocks containing hidden objects.
        scalar = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        if query_model.jnt_type[destination] not in scalar or robot_model.jnt_type[source] not in scalar:
            raise ValueError(f"Unsupported non-scalar robot joint {name}")
        query_data.qpos[int(query_model.jnt_qposadr[destination])] = robot_data.qpos[int(robot_model.jnt_qposadr[source])]

    specs = build_arm_specs(query_model)
    spec = specs[arm]
    finger_geoms = tuple(mujoco.mj_name2id(query_model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"openarm_{arm}_{side}_finger_collision")
                         for side in ("right", "left"))
    fingers = finger_joint_indices(query_model, arm)
    query_data.qpos[fingers[0]] = finger_open
    obj_address = int(query_model.joint("droid_object_freejoint").qposadr[0])
    query_data.qpos[obj_address:obj_address + 3] = center
    query_data.qpos[obj_address + 3:obj_address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(query_model, query_data)
    proxies = [query_data.geom_xpos[g] + query_data.geom_xmat[g].reshape(3, 3) @ p
               for g, p in zip(finger_geoms, PAD_CONTACT_PROXY_LOCAL)]
    open_span = float(np.linalg.norm(proxies[0] - proxies[1]))
    if min(size[:2]) >= open_span:
        raise RuntimeError(f"Observed minimum width {min(size[:2]):.4f} m exceeds physical open-pad span {open_span:.4f} m")

    if shape=='cylinder':
        axes=[np.array([np.cos(theta),np.sin(theta),0.]) for theta in np.arange(4)*np.pi/4]
    else:
        c,s=np.cos(yaw),np.sin(yaw)
        axes=[axis for axis,width in ((np.array([c,s,0.]),size[0]),(np.array([-s,c,0.]),size[1]))
              if width+.004<open_span]
    if not axes: raise RuntimeError('Observed object leaves no physical open-finger clearance')

    result = plan_face_grasp(query_model, query_data, spec, finger_geoms, fingers,
                             PAD_CONTACT_PROXY_LOCAL, center, yaw, seed, clear_q,
                             finger_open, approach_height, transit_height,
                             progress=progress,closing_axes=axes)
    direction = np.asarray(result["direction"], dtype=float)
    if shape == "cylinder":
        selected_width = float(.5 * (size[0] + size[1]))
    else:
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ direction
        selected_width = float(np.dot(np.abs(local), size))
    # Verify the selected width as well as filtering candidates before IK.
    if selected_width >= open_span:
        raise RuntimeError('Selected visual width exceeds the physical opening')
    result["query_geometry"] = dict(shape=shape, size_xyz=size.tolist(),
                                    center_world=center.tolist(), yaw_world_rad=yaw,
                                    source="camera_observation_only", independent_model=True,
                                    selected_width_m=selected_width,
                                    closing_axes=[axis.tolist() for axis in axes])
    # Private kinematic context for subsequent visually commanded lift IK.
    # Never replace this data with execution MjData after the initial plan.
    result["query_model"] = query_model
    result["query_data"] = query_data
    result["query_spec"] = spec
    result["query_finger_geoms"] = finger_geoms
    result["query_finger_indices"] = fingers
    result["query_proxies"] = tuple(p.copy() for p in PAD_CONTACT_PROXY_LOCAL)
    result["scope"] = ("Observed box/cylinder geometry and sampled horizontal closing axes; "
                       "sampled collision checks; no complete reachability or grasp guarantee")
    return result
