"""End-to-end quality and safety regression for the p2a_test1 output."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from scripts.diagnose_motion_regression import _tracking  # noqa: E402


legacy = np.load(ROOT / "data" / "p2a_test1_exchange_rrt.npz")
current = np.load(ROOT / "data" / "p2a_test1_reference_locked.npz")


def quality(data):
    tracking = _tracking(
        data["joint_angles"], data["target_obs"], data["wrist_targets"])
    wrist = tracking["wrist"].max(axis=1)
    direction = tracking["direction"].max(axis=(1, 2))
    bad = (wrist > 0.10) | (direction > 30.0)
    q = np.asarray(data["joint_angles"], dtype=float)
    split = len(q) // 2
    fps = float(data["fps"])
    jerk = [
        np.linalg.norm(np.diff(q[offset:offset + split], n=3, axis=0)
                       * fps ** 3, axis=1)
        for offset in (0, split)
    ]
    holds = [
        int(np.sum(np.linalg.norm(
            np.diff(q[offset:offset + split], axis=0), axis=1) < 1e-7))
        for offset in (0, split)
    ]
    return int(bad.sum()), float(tracking["wrist"].mean()), [
        float(np.percentile(values, 95)) for values in jerk], holds


legacy_bad, legacy_wrist, legacy_jerk, _ = quality(legacy)
bad, wrist, jerk, holds = quality(current)
assert bad <= int(0.80 * legacy_bad), (legacy_bad, bad)
assert wrist < legacy_wrist, (legacy_wrist, wrist)
assert all(new < 0.55 * old for new, old in zip(jerk, legacy_jerk)), (
    legacy_jerk, jerk)
assert max(holds) <= 1, holds
assert "source_arm_trusted" in current.files
assert "offline_joint_smoothing_params" in current.files
assert str(current["trajectory_optimizer"]) == \
    "reference_locked_task_space_l_bfgs_b"

projector = InterArmCollisionProjector(margin=0.005)
unsafe = sum(not projector.is_safe(q, grip) for q, grip in zip(
    current["joint_angles"], current["gripper_position"]))
assert unsafe == 0

print(f"bad_frames={legacy_bad}->{bad} wrist={legacy_wrist:.4f}->{wrist:.4f}m")
print(f"jerk_p95={legacy_jerk}->{jerk} holds={holds} unsafe={unsafe}")
print("Reference-locked output regression passed.")
