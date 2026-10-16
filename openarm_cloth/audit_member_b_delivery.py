"""Read-only audit of B logs, physical timestep progression and bounded commands."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from freeze_member_b_scene import DEFAULT_MANIFEST, verify
from member_b_scene import build_scene
from pose2arm_driver import POSE2ARM_VEL_LIMITS
from teleop_towel_dataset import build_arm_specs


def audit(directory, specs):
    report = json.loads((directory / "latency_log.json").read_text(encoding="utf-8"))
    with (directory / "latency_frames.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["frames"] and rows, "frame count mismatch"
    work = np.asarray([float(r["tick_work_ms"]) for r in rows])
    assert int(np.count_nonzero(work > report["budget_ms"])) == report["ticks_over_budget"]
    for i, row in enumerate(rows):
        assert int(row["tick"]) == i
        start, delivered, finished = [int(row[k]) for k in
                                     ("host_request_ns", "host_delivered_ns", "physics_finished_ns")]
        assert start <= delivered <= finished
        stages = sum(float(row[k+"_ms"]) for k in ("capture", "mask", "fit", "mapping", "control_physics"))
        assert abs(stages-float(row["host_e2e_ms"])) < 1e-6
        assert abs((finished-start)/1e6-float(row["host_e2e_ms"])) < 1e-6
        if report["source"] == "simulation":
            assert delivered <= int(row["actuator_written_ns"]) <= int(row["first_physics_consumed_ns"]) <= finished
        if i:
            assert int(row["source_sequence"]) > int(rows[i-1]["source_sequence"]), "duplicate source frame"
    result = {"source": report["source"], "frames": len(rows),
              "budget_verdict": report["budget_verdict"],
              "mean_ms": report["end_to_end_ms"]["mean"],
              "p95_ms": report["end_to_end_ms"]["p95"],
              "max_ms": report["end_to_end_ms"]["max"],
              "over_budget_frames": report["ticks_over_budget"],
              "actual_hz": report["actual_stream_hz"], "evidence_audit": "PASS"}
    if report["source"] == "simulation":
        with np.load(directory / "actuator_telemetry.npz", allow_pickle=False) as z:
            action, qpos, sim_time = z["action"], z["qpos"], z["sim_time"]
        assert action.shape == (len(rows), 18) and np.isfinite(action).all()
        assert qpos.shape[0] == len(rows) and np.isfinite(qpos).all()
        np.testing.assert_allclose(np.diff(sim_time), 1/report["control_hz"], atol=1e-9)
        np.testing.assert_allclose(sim_time, [float(r["sim_time_s"]) for r in rows], atol=1e-12)
        np.testing.assert_allclose(action[:, :7], np.broadcast_to(action[0,:7], (len(rows),7)), atol=1e-12)
        for side, offset in (("left", 0), ("right", 9)):
            q = action[:,offset:offset+7]
            limits = specs[side].q_range
            assert np.all(q >= limits[:,0]-1e-8) and np.all(q <= limits[:,1]+1e-8)
            assert np.all(action[:,offset+7:offset+9] >= 0)
            assert np.all(action[:,offset+7:offset+9] <= .044)
        steps = np.abs(np.diff(action[:,9:16], axis=0))
        assert np.all(steps <= POSE2ARM_VEL_LIMITS/report["control_hz"]+1e-8)
        error = np.abs(qpos[:,specs["right"].qpos]-action[:,9:16])
        result["measured_active_tracking_error_rad"] = {
            "rms": float(np.sqrt(np.mean(error**2))), "max": float(error.max())}
        result["source_to_bounded_command_max_lag_rad"] = report["driver"]["max_lag_rad"]
        result["perception_valid_frames"] = report["perception_valid_frames"]
        result["gripper_counts"] = report["driver"]["gripper"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_MANIFEST.parent)
    args = parser.parse_args()
    verify(args.manifest)
    model, _ = build_scene()
    specs = build_arm_specs(model)
    result = {"scene_verification": "PASS", "runs": {},
              "note": "Audit PASS means evidence consistency, not necessarily latency PASS or grasp success."}
    for path in sorted((args.outdir / "latency").iterdir()):
        if path.is_dir() and (path / "latency_log.json").exists():
            result["runs"][path.name] = audit(path, specs)
    full = json.loads((args.outdir / "latency/sim_full_npz/latency_log.json").read_text(encoding="utf-8"))
    source = Path(full["npz"])
    result["input_npz"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "copied_into_delivery": False}
    (args.outdir / "acceptance_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
