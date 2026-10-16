"""RGB-D visibility screening only; never claims a successful grasp."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OPENARM_TASK_WORKSPACE_CONFIG"] = str(args.config.resolve())
    import mujoco
    import numpy as np
    from demo_grasp_viewer import (SCENE, CUBE_HALF, initialize_observation_posture,
                                  build_arm_specs, place_cube_for_blind_test,
                                  load_chest_calibration, apply_workspace_rgbd_calibration,
                                  load_workspace_rgbd_calibration)
    from task_workspace import load_task_model, TABLE_TOP
    from calibration_simulation import render_rgb_depth
    from handeye_grasp_red_cube import estimate_cube_center
    m = load_task_model(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    initialize_observation_posture(m, d, build_arm_specs(m))
    ready = d.qpos.copy()
    address = int(m.joint("droid_object_freejoint").qposadr[0])
    transform, intrinsics, _ = load_chest_calibration()
    correction = load_workspace_rgbd_calibration()["chest"]
    renderer = mujoco.Renderer(m, 480, 640)
    rows = []
    try:
        for x in (.33, .35, .36, .4, .45, .5, .52, .54, .55):
            for y in (-.22, -.20, -.18, -.16, -.14, 0, .14, .16, .18, .20, .22):
                for yaw in (0, 45):
                    d.qpos[:] = ready
                    place_cube_for_blind_test(m, d, (x, y, yaw))
                    # Known fixture support height; independent detector gets
                    # only camera pixels and hand-eye calibration below.
                    d.qpos[address + 2] = TABLE_TOP + CUBE_HALF
                    mujoco.mj_forward(m, d)
                    rgb, depth = render_rgb_depth(renderer, d, "chest_d435i_preview")
                    row = {"fixture": [x, y, yaw], "visible": False}
                    try:
                        obs = estimate_cube_center(rgb, depth, np.eye(4), transform, intrinsics, CUBE_HALF)
                        measured = apply_workspace_rgbd_calibration(obs.center_world, correction)
                        row.update(visible=True, pixels=int(obs.mask_pixels),
                                   evaluation_xy_error_mm=float(np.linalg.norm(measured[:2] - [x, y]) * 1000))
                    except RuntimeError as exc:
                        row["reason"] = str(exc)
                    rows.append(row)
    finally:
        renderer.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for x in (.33, .35, .36, .4, .45, .5, .52, .54, .55):
        valid = [r["fixture"][1] for r in rows if r["fixture"][0] == x and r["visible"]]
        print(x, min(valid) if valid else None, max(valid) if valid else None)


if __name__ == "__main__":
    main()
