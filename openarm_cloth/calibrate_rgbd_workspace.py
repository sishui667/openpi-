# -*- coding: utf-8 -*-
"""Offline planar calibration for chest + wrist RGB-D cube localisation.

Fixture coordinates are used only to build this calibration file.  The grasp
controller later consumes camera estimates and the frozen affine maps; it does
not receive MuJoCo object coordinates during an episode.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from calibration_simulation import body_transform, render_rgb_depth
from demo_grasp_viewer import (CUBE_HALF, FINGER_OPEN, ROOT, SCENE,
                               load_chest_calibration,
                               place_cube_for_blind_test,
                               repair_left_finger_position_servos)
from handeye_grasp_red_cube import (estimate_cube_center,
                                    latest_handeye_report, load_handeye,
                                    quat_from_body, solve_target_q)
from teleop_towel_dataset import (apply_controller, build_arm_specs,
                                  build_finger_actuators)

OUTPUT = ROOT / "calibration" / "rgbd_workspace_affine.json"
XS = (0.31, 0.39, 0.47, 0.55)
YS = (-0.18, -0.09, 0.0, 0.09, 0.18)


def new_scene(x: float, y: float):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    specs = build_arm_specs(model)
    finger_actuators = build_finger_actuators(model)
    repair_left_finger_position_servos(model, finger_actuators)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    place_cube_for_blind_test(model, data, (x, y, 0.0))
    holds = {side: data.qpos[specs[side].qpos].copy()
             for side in ("left", "right")}
    for _ in range(int(1.2 / model.opt.timestep)):
        apply_controller(model, data, specs, finger_actuators, holds,
                         {"left": FINGER_OPEN, "right": FINGER_OPEN})
        mujoco.mj_step(model, data)
    object_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "droid_object")
    return model, data, specs, holds, object_id


def affine_fit(rows):
    observed = np.asarray([[*row["observed_xy"], 1.0] for row in rows])
    truth = np.asarray([row["truth_xy"] for row in rows])
    coeff, *_ = np.linalg.lstsq(observed, truth, rcond=None)
    predicted = observed @ coeff
    errors = np.linalg.norm(predicted - truth, axis=1)
    return coeff, errors


def correct_xy(point, coeff):
    return np.asarray([point[0], point[1], 1.0]) @ coeff


def main():
    chest_transform, chest_matrix, chest_path = load_chest_calibration()
    chest_rows = []
    for x in XS:
        for y in YS:
            model, data, _specs, _holds, object_id = new_scene(x, y)
            renderer = mujoco.Renderer(model, 480, 640)
            rgb, depth = render_rgb_depth(
                renderer, data, "chest_d435i_preview")
            renderer.close()
            try:
                obs = estimate_cube_center(
                    rgb, depth, np.eye(4), chest_transform, chest_matrix,
                    CUBE_HALF)
            except RuntimeError as exc:
                print(f"chest {x:.2f},{y:+.2f} missed: {exc}")
                continue
            truth = data.xpos[object_id].copy()
            chest_rows.append({
                "fixture_xy": [x, y],
                "observed_xy": list(map(float, obs.center_world[:2])),
                "truth_xy": list(map(float, truth[:2])),
                "z_error_m": float(truth[2] - obs.center_world[2]),
            })
            print(f"chest {x:.2f},{y:+.2f} -> "
                  f"{obs.center_world[0]:.3f},{obs.center_world[1]:+.3f}")
    chest_coeff, chest_errors = affine_fit(chest_rows)
    print(f"chest affine RMS/max: "
          f"{np.sqrt(np.mean(chest_errors**2))*1000:.2f}/"
          f"{chest_errors.max()*1000:.2f} mm")

    wrist_rows = {"left": [], "right": []}
    for x in XS:
        for y in YS:
            model, data, specs, holds, object_id = new_scene(x, y)
            # Re-render chest because each sample has an independent scene.
            renderer = mujoco.Renderer(model, 480, 640)
            rgb, depth = render_rgb_depth(
                renderer, data, "chest_d435i_preview")
            renderer.close()
            try:
                chest_obs = estimate_cube_center(
                    rgb, depth, np.eye(4), chest_transform, chest_matrix,
                    CUBE_HALF)
            except RuntimeError as exc:
                print(f"chest/wrist {x:.2f},{y:+.2f} skipped: {exc}")
                continue
            coarse = np.asarray(chest_obs.center_world, dtype=float)
            coarse[:2] = correct_xy(coarse, chest_coeff)
            arm = "left" if coarse[1] > 0.02 else "right"
            spec = specs[arm]
            q_ready = holds[arm].copy()
            quat_ready = quat_from_body(model, data, spec.body_id)
            rotation_ready = body_transform(data, spec.body_id)[:3, :3].copy()
            side_sign = 1.0 if arm == "left" else -1.0
            q_clear, _, _ = solve_target_q(
                model, data, spec, np.array([0.28, side_sign * 0.16, 0.72]),
                quat_ready, q_ready, target_rotation=rotation_ready)
            search_position = np.array([
                np.clip(coarse[0] - 0.19, 0.18, 0.38),
                np.clip(coarse[1] * 0.65, -0.13, 0.13), 0.60,
            ])
            q_search, _, _ = solve_target_q(
                model, data, spec, search_position, quat_ready, q_clear,
                target_rotation=rotation_ready)
            data.qpos[spec.qpos] = q_search
            data.qvel[spec.dof] = 0.0
            mujoco.mj_forward(model, data)
            handeye, wrist_matrix = load_handeye(
                latest_handeye_report(arm), arm)
            renderer = mujoco.Renderer(model, 240, 320)
            rgb, depth = render_rgb_depth(
                renderer, data, f"v22_{arm}_d405_rgb")
            renderer.close()
            try:
                obs = estimate_cube_center(
                    rgb, depth, body_transform(data, spec.body_id),
                    handeye, wrist_matrix, CUBE_HALF)
            except RuntimeError as exc:
                print(f"wrist {arm} {x:.2f},{y:+.2f} missed: {exc}")
                continue
            truth = data.xpos[object_id].copy()
            wrist_rows[arm].append({
                "fixture_xy": [x, y],
                "observed_xy": list(map(float, obs.center_world[:2])),
                "truth_xy": list(map(float, truth[:2])),
                "z_error_m": float(truth[2] - obs.center_world[2]),
            })
            print(f"wrist {arm} {x:.2f},{y:+.2f} -> "
                  f"{obs.center_world[0]:.3f},{obs.center_world[1]:+.3f}")

    output = {
        "method": "offline_known_fixture_affine_camera_calibration",
        "controller_runtime_inputs": "camera estimates only",
        "chest_source": str(chest_path),
        "grid_x": list(XS),
        "grid_y": list(YS),
        "chest": {
            "xy_coeff": chest_coeff.tolist(),
            "z_bias_m": float(np.mean([r["z_error_m"] for r in chest_rows])),
            "rms_mm": float(np.sqrt(np.mean(chest_errors**2)) * 1000),
            "max_mm": float(chest_errors.max() * 1000),
            "samples": chest_rows,
        },
        "wrist": {},
    }
    for arm, rows in wrist_rows.items():
        coeff, errors = affine_fit(rows)
        output["wrist"][arm] = {
            "xy_coeff": coeff.tolist(),
            "z_bias_m": float(np.mean([r["z_error_m"] for r in rows])),
            "rms_mm": float(np.sqrt(np.mean(errors**2)) * 1000),
            "max_mm": float(errors.max() * 1000),
            "sample_count": len(rows),
            "samples": rows,
        }
        print(f"wrist {arm} affine RMS/max: "
              f"{output['wrist'][arm]['rms_mm']:.2f}/"
              f"{output['wrist'][arm]['max_mm']:.2f} mm")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
