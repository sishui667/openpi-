"""Face-grasp candidate planning using a camera-derived virtual cube only."""
from __future__ import annotations

import mujoco
import numpy as np

from handeye_grasp_red_cube import solve_gripper_contact_proxy_q


def plan_face_grasp(model, data, spec, finger_geoms, finger_indices,
                    proxies, center, yaw, seed, clear_q, finger_open,
                    approach_height, transit_height, progress=None,
                    closing_axes=None):
    # Replace the object in a separate planning state with the visual estimate.
    # No object pose, contact position or segmentation ID from execution enters
    # the search.  Robot joints and static model geometry are available inputs.
    planning = mujoco.MjData(model)
    # Copy robot encoders explicitly, never the execution object's qpos.
    # OpenArm joints here are scalar hinges/slides; object free joints are
    # deliberately excluded from this interface.
    for robot_joint in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, robot_joint) or ""
        if joint_name.startswith("openarm_"):
            robot_address = int(model.jnt_qposadr[robot_joint])
            planning.qpos[robot_address] = data.qpos[robot_address]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            "droid_object_freejoint")
    address = int(model.jnt_qposadr[joint])
    planning.qpos[address:address + 3] = center
    planning.qpos[address + 3:address + 7] = (
        np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    planning.qpos[finger_indices[0]] = finger_open
    mujoco.mj_forward(model, planning)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  "droid_object_geom")
    # Infer the active arm from its known finger geom name.
    finger_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  finger_geoms[0])
    prefix = "_".join(finger_name.split("_")[:2]) + "_"
    active = {g for g in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
              .startswith(prefix)}

    def collisions(q):
        planning.qpos[spec.qpos] = q
        mujoco.mj_forward(model, planning)
        pairs = []
        for i in range(planning.ncon):
            c = planning.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if not ({g1, g2} & active) or c.dist >= -0.0005:
                continue
            pairs.append((tuple(sorted((g1, g2))), float(-c.dist)))
        return pairs

    baseline = {pair: depth for pair, depth in collisions(seed)
                if object_geom not in pair}

    def collision_cost(q):
        return sum(max(0.0, depth - baseline.get(pair, 0.0))
                   for pair, depth in collisions(q))

    def rotation(q):
        planning.qpos[spec.qpos] = q
        mujoco.mj_forward(model, planning)
        return planning.geom_xmat[finger_geoms[0]].reshape(3, 3).copy()

    points = [planning.geom_xpos[g] + planning.geom_xmat[g].reshape(3, 3) @ p
              for g, p in zip(finger_geoms, proxies)]
    physical_span = float(np.linalg.norm(points[0] - points[1]))
    candidates = []
    c, s = np.cos(yaw), np.sin(yaw)
    axes=(np.array([c,s,0.0]),np.array([-s,c,0.0])) if closing_axes is None else closing_axes
    axes=[np.asarray(axis,dtype=float) for axis in axes]
    if not axes or any(axis.shape!=(3,) or not np.isfinite(axis).all() or
                       np.linalg.norm(axis)<1e-8 or abs(axis[2])>1e-8 for axis in axes):
        raise ValueError('closing_axes must contain finite nonzero horizontal directions')
    if closing_axes is not None: axes=[axis/np.linalg.norm(axis) for axis in axes]
    for down_weight in (0.8, 0.0):
        if down_weight == 0.0:
            current_best = min(candidates, key=lambda item: item[0])[6]
            if (current_best["midpoint_error_m"] < 0.005
                    and current_best["perpendicular_error_m"] < 0.008
                    and current_best["max_new_penetration_mm"] < 0.5):
                break
            print("Trying alternate wrist tilts for reach and face alignment", flush=True)
        for face_index, normal in enumerate(axes):
            for sign in (1.0, -1.0):
                direction = sign * normal
                q, residual, pads, metrics = solve_gripper_contact_proxy_q(
                    model, planning, spec, finger_geoms, finger_indices, proxies,
                    center, 1, sign, physical_span, finger_open, seed,
                    closing_direction=direction, approach_down_weight=down_weight,
                    multi_start=False)
                frame = rotation(q)
                q_approach, *_ = solve_gripper_contact_proxy_q(
                    model, planning, spec, finger_geoms, finger_indices, proxies,
                    center + [0, 0, approach_height], 1, sign, physical_span,
                    finger_open, q, reference_rotation=frame,
                    closing_direction=direction, multi_start=False)
                q_transit, *_ = solve_gripper_contact_proxy_q(
                    model, planning, spec, finger_geoms, finger_indices, proxies,
                    center + [0, 0, transit_height], 1, sign, physical_span,
                    finger_open, q_approach, reference_rotation=frame,
                    closing_direction=direction, multi_start=False)
                collision_penalty = 0.0
                for start, end in ((clear_q, q_transit),
                                   (q_transit, q_approach), (q_approach, q)):
                    for blend in np.linspace(0, 1, 13):
                        collision_penalty = max(
                            collision_penalty,
                            collision_cost(start + blend * (end - start)))
                downward = float(frame[2, 2])
                score = (metrics["midpoint_error_m"] * 10
                         + metrics["perpendicular_error_m"] * 5
                         + collision_penalty * 100
                         + max(0.0, downward + 0.8) * 0.03)
                record = {
                    "face_index": face_index, "sign": sign,
                    "downward_weight": down_weight,
                    "direction": direction.tolist(), "score": score,
                    "max_new_penetration_mm": collision_penalty * 1000,
                    "distal_down_cosine": -downward, **metrics,
                }
                if progress:
                    progress(record)
                candidates.append((score, q, q_approach, q_transit, frame,
                                   pads, record))
    winner = min(candidates, key=lambda item: item[0])
    return {
        "q_grasp": winner[1], "q_approach": winner[2],
        "q_transit": winner[3], "rotation": winner[4],
        "pads": winner[5], "metrics": winner[6],
        "direction": np.asarray(winner[6]["direction"]),
        "span": physical_span,
        "candidates": [item[6] for item in candidates],
    }
