"""Headless inter-arm collision diagnostic for Pose2Arm NPZ trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ik_solver import OpenArmIK  # noqa: E402
from openarm_common import OPENARM_BASE_OFFSETS  # noqa: E402
from openarm_live import _segment_distance  # noqa: E402
from collision_safety import (  # noqa: E402
    GRIPPER_CLOSE_QPOS, GRIPPER_COL, GRIPPER_OPEN_QPOS,
    InterArmCollisionProjector)

DEFAULT_XML = Path(
    r"D:\openpi\openarm_cloth_with_dependencies_20260929\openarm_cloth"
    r"\official_openarm_mujoco\v2\openarm_bimanual.xml"
)
JOINT_NAMES = [
    f"openarm_{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)
]


def _joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing joint {name}")
        addresses.append(model.jnt_qposadr[joint_id])
    return np.asarray(addresses, dtype=int)


def _write_gripper(model, data, grip: np.ndarray) -> None:
    for side in ("left", "right"):
        closure = float(np.clip(grip[GRIPPER_COL[side]], 0.0, 1.0))
        target = (GRIPPER_OPEN_QPOS[side]
                  + closure * (GRIPPER_CLOSE_QPOS[side]
                               - GRIPPER_OPEN_QPOS[side]))
        for finger in (1, 2):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT,
                f"openarm_{side}_finger_joint{finger}")
            data.qpos[model.jnt_qposadr[jid]] = target


def _side(body_name: str | None) -> str | None:
    if body_name and "_left_" in body_name:
        return "left"
    if body_name and "_right_" in body_name:
        return "right"
    return None


def _collision_geoms(model: mujoco.MjModel, side: str) -> list[int]:
    result = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if _side(body_name) == side and geom_name and "collision" in geom_name:
            result.append(geom_id)
    return result


def _fk_metrics(left_ik: OpenArmIK, right_ik: OpenArmIK, frame: np.ndarray) -> tuple[float, float]:
    left_points = left_ik.forward_points(frame[:7])
    right_points = right_ik.forward_points(frame[7:])
    # Reproduce generate_ik_dataset.py exactly: its old metric omits the base offsets.
    left_joints = np.asarray(left_points["joint_positions"])
    right_joints = np.asarray(right_points["joint_positions"])
    left_key = np.asarray([left_joints[0], left_joints[2], left_joints[3], left_joints[5],
                           left_points["tool"]])
    right_key = np.asarray([right_joints[0], right_joints[2], right_joints[3], right_joints[5],
                            right_points["tool"]])
    key_distance = min(float(np.linalg.norm(a - b)) for a in left_key for b in right_key)

    left_upper_mid = 0.35 * left_points["elbow"] + OPENARM_BASE_OFFSETS["left"]
    right_upper_mid = 0.35 * right_points["elbow"] + OPENARM_BASE_OFFSETS["right"]
    left_segments = [(left_upper_mid,
                      left_points["elbow"] + OPENARM_BASE_OFFSETS["left"]),
                     (left_points["elbow"] + OPENARM_BASE_OFFSETS["left"],
                      left_points["tool"] + OPENARM_BASE_OFFSETS["left"])]
    right_segments = [(right_upper_mid,
                       right_points["elbow"] + OPENARM_BASE_OFFSETS["right"]),
                      (right_points["elbow"] + OPENARM_BASE_OFFSETS["right"],
                       right_points["tool"] + OPENARM_BASE_OFFSETS["right"])]
    segment_distance = min(_segment_distance(*a, *b) for a in left_segments for b in right_segments)
    return key_distance, segment_distance


def _interarm_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, float, list[str]]:
    contacts = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
        if {_side(name1), _side(name2)} == {"left", "right"}:
            contacts.append((float(contact.dist), f"{name1}<->{name2}"))
    if not contacts:
        return 0, float("inf"), []
    return len(contacts), min(item[0] for item in contacts), sorted({item[1] for item in contacts})


def _all_contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[float, str]]:
    result = []
    for contact in data.contact:
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
        result.append((float(contact.dist), "<->".join(sorted((name1, name2)))))
    return result


def _visual_collision_model(xml_path: Path) -> mujoco.MjModel:
    """Use render meshes as collision meshes to detect visible interpenetration."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        is_collision_mesh = "collision" in geom_name
        model.geom_contype[geom_id] = 0 if is_collision_mesh else 1
        model.geom_conaffinity[geom_id] = 0 if is_collision_mesh else 1
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=REPO_ROOT / "data" / "rl_poses_output.npz")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--margin", type=float, default=0.005,
                        help="硬安全层检查余量，米（默认 0.005）")
    args = parser.parse_args()

    loaded = np.load(args.npz)
    trajectory = np.asarray(loaded["joint_angles"], dtype=float)
    gripper = (np.asarray(loaded["gripper_position"], dtype=float)
               if "gripper_position" in loaded.files else None)
    projector = InterArmCollisionProjector(args.xml, margin=args.margin)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    visual_model = _visual_collision_model(args.xml)
    visual_data = mujoco.MjData(visual_model)
    qpos_addresses = _joint_qpos_addresses(model)
    left_geoms = _collision_geoms(model, "left")
    right_geoms = _collision_geoms(model, "right")
    left_ik, right_ik = OpenArmIK("left"), OpenArmIK("right")
    rows = []
    visual_all_contacts = []

    for frame_id, frame in enumerate(trajectory):
        data.qpos[qpos_addresses] = frame
        if gripper is not None:
            _write_gripper(model, data, gripper[frame_id])
        mujoco.mj_forward(model, data)
        visual_data.qpos[qpos_addresses] = frame
        if gripper is not None:
            _write_gripper(visual_model, visual_data, gripper[frame_id])
        mujoco.mj_forward(visual_model, visual_data)
        key_distance, segment_distance = _fk_metrics(left_ik, right_ik, frame)
        collision_count, collision_depth, collision_pairs = _interarm_contacts(model, data)
        visual_count, visual_depth, visual_pairs = _interarm_contacts(visual_model, visual_data)
        visual_all_contacts.extend((depth, frame_id, pair)
                                   for depth, pair in _all_contact_pairs(visual_model, visual_data))
        rows.append((visual_depth, frame_id, key_distance, segment_distance,
                     collision_count, visual_count, collision_pairs, visual_pairs))

    rows.sort(key=lambda row: row[0])
    split = len(trajectory) // 2
    visual_rows = [row for row in rows if row[5] > 0]
    margin_unsafe = sum(
        not projector.is_safe(frame, None if gripper is None else gripper[i])
        for i, frame in enumerate(trajectory))
    false_safe = [row for row in visual_rows if row[2] >= 0.08]
    print(f"frames={len(rows)} split={split} collision_geoms={len(left_geoms)}x{len(right_geoms)}")
    print(f"collision_mesh_contact_frames={sum(row[4] > 0 for row in rows)} visual_mesh_contact_frames={len(visual_rows)} old_keypoint_false_safe={len(false_safe)}")
    print(f"hard_safety_margin_m={args.margin:.4f} margin_unsafe_frames={margin_unsafe}")
    print(f"visual_orig={sum(row[5] > 0 and row[1] < split for row in rows)} visual_mirror={sum(row[5] > 0 and row[1] >= split for row in rows)} segment_below_5.5cm={sum(row[3] < 0.055 for row in rows)}")
    print("worst: visual_depth_cm frame clip old_key_cm segment_cm collision_n visual_n visual_pairs")
    for visual_depth, frame_id, key_distance, segment_distance, collision_count, visual_count, _, visual_pairs in rows[:args.top]:
        clip = "orig" if frame_id < split else "mirror"
        depth = 100 * visual_depth if np.isfinite(visual_depth) else float("inf")
        print(f"{depth:15.3f} {frame_id:5d} {clip:6s} {100*key_distance:10.3f} {100*segment_distance:10.3f} {collision_count:11d} {visual_count:8d} {','.join(visual_pairs)}")
    pair_counts = {}
    for _, _, pair in visual_all_contacts:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    contact_frames = len({frame_id for _, frame_id, _ in visual_all_contacts})
    print(f"any_visual_self_or_interarm_contact_frames={contact_frames}")
    for pair, count in sorted(pair_counts.items(), key=lambda item: -item[1])[:20]:
        worst_depth, worst_frame, _ = min(item for item in visual_all_contacts if item[2] == pair)
        print(f"pair count={count:5d} worst_cm={100*worst_depth:8.3f} frame={worst_frame:4d} {pair}")


if __name__ == "__main__":
    main()
