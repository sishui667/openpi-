"""Audit merged A human gripper signals -> B real actuators, not grasp success."""
from pathlib import Path
import csv
import json

import numpy as np

from freeze_member_b_scene import verify, DEFAULT_MANIFEST
from member_b_scene import build_scene
from pose2arm_driver import POSE2ARM_VEL_LIMITS
from teleop_towel_dataset import build_arm_specs, build_finger_actuators


def main():
    root = DEFAULT_MANIFEST.parent
    verify(DEFAULT_MANIFEST)
    model,_ = build_scene()
    specs,fingers = build_arm_specs(model),build_finger_actuators(model)
    current=json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    old=json.loads((DEFAULT_MANIFEST.parent.parent/"member_B_20261016/scene_manifest.json").read_text(encoding="utf-8"))
    for key in ("physics","workspace","cameras","actuators","controller"):
        assert current[key]==old[key],f"Optimized scene unexpectedly changed: {key}"
    summary = {"scope":"A offline video signals -> B physics; NOT autonomous grasp or live-human inference",
               "optimized_B_scene_unchanged":"PASS",
               "scene_hash_verification":"PASS", "runs":{}}
    for name in ("hybrid","reference"):
        source_path = root / f"a_{name}_orig_20hz.npz"
        logdir = root / "latency_final" / name
        report = json.loads((logdir/"latency_log.json").read_text(encoding="utf-8"))
        with np.load(source_path,allow_pickle=False) as source:
            closure = source["gripper_position"][:,0].copy()
            source_duration = float(source["source_duration_s"])
        with np.load(logdir/"actuator_telemetry.npz",allow_pickle=False) as recorded:
            action,qpos,ctrl,times = [recorded[k].copy() for k in ("action","qpos","ctrl","sim_time")]
        with (logdir/"latency_frames.csv").open(encoding="utf-8",newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(action)==len(closure)==len(rows)==report["frames"]
        expected = .044*(1-closure)
        np.testing.assert_allclose(action[:,16:18],np.repeat(expected[:,None],2,axis=1),atol=1e-12)
        np.testing.assert_allclose(ctrl[:,fingers["right"]],np.repeat(expected[:,None],2,axis=1),atol=1e-12)
        np.testing.assert_allclose(action[:,7:9],.044,atol=1e-12)
        np.testing.assert_allclose(action[:,:7],np.broadcast_to(action[0,:7],(len(action),7)),atol=1e-12)
        np.testing.assert_allclose(np.diff(times),.05,atol=1e-9)
        assert np.all(np.abs(np.diff(action[:,9:16],axis=0))<=POSE2ARM_VEL_LIMITS/20+1e-8)
        for row in rows:
            clock = [int(row[k]) for k in ("host_request_ns","host_delivered_ns","actuator_written_ns",
                                          "first_physics_consumed_ns","physics_finished_ns")]
            assert clock==sorted(clock)
        measured_finger = qpos[:,[int(model.joint(f"openarm_right_finger_joint{i}").qposadr[0]) for i in (1,2)]]
        finger_error = measured_finger-expected[:,None]
        summary["runs"][name] = {
            "audit":"PASS", "frames":len(action), "human_closure_over_half_frames":int((closure>.5).sum()),
            "source_duration_s":source_duration,"simulated_playback_duration_s":len(action)*.05,
            "finger_target_range_m":[float(expected.min()),float(expected.max())],
            "measured_finger_range_m":[float(measured_finger.min()),float(measured_finger.max())],
            "finger_tracking_rms_m":float(np.sqrt(np.mean(finger_error**2))),
            "driver":report["driver"], "actual_hz":report["actual_stream_hz"],
            "end_to_end_ms":report["end_to_end_ms"], "deadline_misses":report["ticks_over_budget"],
            "latency_verdict":report["budget_verdict"], "grasp_success":"not_evaluated",
        }
    (root/"acceptance_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
