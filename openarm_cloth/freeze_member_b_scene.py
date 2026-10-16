"""Freeze/verify current B task geometry, physics, cameras and runtime hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from member_b_scene import ROOT, SCENE, TARGET_ZONE, build_scene
from task_workspace import CONFIG, CONFIG_PATH
from teleop_towel_dataset import build_arm_specs, build_finger_actuators

DEFAULT_MANIFEST = ROOT / "deliverables/member_AB_20261016/scene_manifest.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_scene():
    model, data = build_scene()
    specs, fingers = build_arm_specs(model), build_finger_actuators(model)
    base = SCENE.parent / "droid_cube_scene.xml"
    tracked = [base, SCENE, CONFIG_PATH, ROOT / "task_workspace.py",
               ROOT / "member_b_scene.py", ROOT / "pose2arm_driver.py",
               ROOT / "workspace_camera.py", ROOT / "member_b_realtime.py",
               ROOT / "member_b_actuators.py", ROOT / "freeze_member_b_scene.py", ROOT / "demo_member_ab.py",
               ROOT / "realtime_capture.py", ROOT / "teleop_towel_dataset.py"]
    a_root = ROOT.parent / "pose2arm-main"
    for name in ("member_b_export.py", "gripper_map.py", "smoothing.py",
                 "scripts/generate_ik_dataset.py", "collision_safety.py", "hand_association.py", "extract_hand.py"):
        tracked.append(a_root / name)
    tree = ET.parse(base).getroot()
    compiler = tree.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    for node in tree.iter():
        if node.tag in ("mesh", "texture") and "file" in node.attrib:
            directory = meshdir if node.tag == "mesh" else ""
            tracked.append(base.parent / directory / node.attrib["file"])
    hashes = {Path(os.path.relpath(p.resolve(), ROOT)).as_posix(): digest(p) for p in tracked}
    active_welds = [i for i in range(model.neq)
                    if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD and data.eq_active[i]]
    if active_welds:
        raise ValueError(f"Forbidden active object weld(s): {active_welds}")
    cameras = []
    for i in range(model.ncam):
        f = float(240 / np.tan(np.deg2rad(model.cam_fovy[i]) / 2))
        cameras.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i),
                        "position_local_m": model.cam_pos[i].tolist(),
                        "quaternion_wxyz": model.cam_quat[i].tolist(),
                        "fovy_deg": float(model.cam_fovy[i]),
                        "intrinsics_640x480": [[f, 0, 320], [0, f, 240], [0, 0, 1]]})
    obj, table = model.geom("droid_object_geom"), model.geom("folding_table")
    body = model.body("droid_object")
    physics = {
        "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
        "noslip_iterations": int(model.opt.noslip_iterations),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "cube_mass_kg": float(body.mass[0]), "cube_half_size_m": obj.size.tolist(),
        "cube_inertia_kg_m2": body.inertia.tolist(), "cube_friction": obj.friction.tolist(),
        "cube_solref": obj.solref.tolist(), "cube_solimp": obj.solimp.tolist(),
        "table_center_m": table.pos.tolist(), "table_half_size_m": table.size.tolist(),
        "table_friction": table.friction.tolist(),
        "explicit_contact_pairs": [{"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[i])),
                                     "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[i])),
                                     "condim": int(model.pair_dim[i]),
                                     "friction": model.pair_friction[i].tolist(),
                                     "solref": model.pair_solref[i].tolist()}
                                    for i in range(model.npair)],
        "active_welds": active_welds,
    }
    return {
        "schema_version": 2, "role": "Merged Member A/B offline demonstration interface",
        "source_scene": SCENE.relative_to(ROOT).as_posix(), "mujoco_version": mujoco.__version__,
        "active_arm": "right", "inactive_arm": "left", "inactive_policy": "fixed initial high standby",
        "workspace": CONFIG, "physics": physics, "cameras": cameras,
        "monocular_control_camera": "workspace_camera",
        "monocular_camera_note": "simulated RGB only; fixed OVERHEAD (90deg), not human-facing USB camera",
        "human_capture_camera": "OpenCV USB device 0, DSHOW; separate from simulated scene cameras",
        "task_definition": {
            "target_zone": TARGET_ZONE, "maximum_duration_s": 15.0,
            "grasp_height_above_table_m": 0.03, "jaw_object_distance_m": 0.02,
            "grasp_hold_s": 0.5, "release_settle_s": 0.5,
            "release_linear_speed_m_s": 0.02, "release_angular_speed_rad_s": 0.2,
            "scope": "environment/evaluation contract; supplied NPZ is not a validated pick-and-place demonstration",
        },
        "actuators": {side: {
            "joints": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in spec.actuators],
            "q_range_rad": spec.q_range.tolist(),
            "torque_limits_nm": model.actuator_forcerange[spec.actuators].tolist(),
            "finger_gainprm": model.actuator_gainprm[fingers[side]].tolist(),
            "finger_biasprm": model.actuator_biasprm[fingers[side]].tolist(),
        } for side, spec in specs.items()},
        "controller": {"kp": [80,80,55,55,25,20,20], "kd": [8,8,5,5,3,2,2],
                       "feedforward": "qfrc_bias", "frequency_hz": 20.0,
                       "physics_steps_per_tick": 100},
        "files_sha256": hashes,
    }


def verify(manifest):
    saved = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for name, expected in saved["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen file drift: {name}")
    current = inspect_scene()
    for key in ("physics", "workspace", "actuators", "cameras", "task_definition", "controller"):
        if saved[key] != current[key]:
            raise ValueError(f"Compiled scene drift: {key}")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="explicitly create/update the freeze manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(inspect_scene(), indent=2, ensure_ascii=False), encoding="utf-8")
    result = verify(args.manifest)
    print(json.dumps({"verdict": "PASS", "manifest": str(args.manifest),
                      "hashed_files": len(result["files_sha256"]),
                      "table_top_m": sum([result["physics"]["table_center_m"][2], result["physics"]["table_half_size_m"][2]]),
                      "active_arm": result["active_arm"], "active_welds": result["physics"]["active_welds"]}))


if __name__ == "__main__":
    main()
