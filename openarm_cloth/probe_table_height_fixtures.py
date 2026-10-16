"""Evaluation-only: did a robot's initial standby knock the fixture away?"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--yaws", type=float, nargs="+", default=[0])
    args = parser.parse_args()
    os.environ["OPENARM_TASK_WORKSPACE_CONFIG"] = str(args.config.resolve())
    import mujoco
    import numpy as np
    from benchmark_table_heights import POSITIONS
    from demo_grasp_viewer import (
        SCENE, FINGER_OPEN, build_arm_specs, build_finger_actuators,
        repair_left_finger_position_servos, initialize_observation_posture,
        place_cube_for_blind_test, apply_controller)
    from task_workspace import TABLE_TOP, load_task_model
    model = load_task_model(SCENE)
    specs = build_arm_specs(model)
    actuators = build_finger_actuators(model)
    repair_left_finger_position_servos(model, actuators)
    rows = []
    for x, y in POSITIONS:
        for yaw in args.yaws:
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            initialize_observation_posture(model, data, specs)
            place_cube_for_blind_test(model, data, (x, y, yaw))
            holds = {side: data.qpos[spec.qpos].copy() for side, spec in specs.items()}
            for _ in range(int(1.2 / model.opt.timestep)):
                apply_controller(model, data, specs, actuators, holds,
                                 {"left": FINGER_OPEN, "right": FINGER_OPEN})
                mujoco.mj_step(model, data)
            truth = data.body("droid_object").xpos.copy()
            drift = float(np.linalg.norm(truth[:2] - [x, y]) * 1000)
            rows.append({"table_top_m": TABLE_TOP, "fixture_xy_yaw": [x, y, yaw],
                         "settled_truth_m_evaluation_only": truth.tolist(),
                         "xy_drift_mm_evaluation_only": drift,
                         "fixture_valid": bool(drift < 1 and truth[2] > TABLE_TOP)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"height": TABLE_TOP, "cases": len(rows),
                      "valid": sum(r["fixture_valid"] for r in rows),
                      "max_xy_drift_mm": max(r["xy_drift_mm_evaluation_only"] for r in rows)}))


if __name__ == "__main__":
    main()
