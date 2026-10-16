"""Check the produced hybrid replay against the validated simulation baseline."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402


def main() -> None:
    baseline = np.load(ROOT / "data" / "p2a_test1_reference_locked.npz")
    hybrid = np.load(ROOT / "data" / "p2a_test1_rtmw_hybrid_sim.npz")
    q = hybrid["joint_angles"]
    grip = hybrid["gripper_position"]
    split = len(q) // 2
    assert np.array_equal(q, baseline["joint_angles"]), "arm trajectory changed"
    assert np.isfinite(grip).all() and ((grip >= 0) & (grip <= 1)).all()
    assert np.count_nonzero(np.abs(grip - baseline["gripper_position"]) > 1e-3) > 0
    assert np.all(hybrid["hybrid_collision_stats"] == 0)
    assert len(hybrid["hybrid_reverted_clips"]) == 0
    old_step = np.abs(np.diff(baseline["gripper_position"][:split], axis=0))
    new_step = np.abs(np.diff(grip[:split], axis=0))
    assert np.percentile(new_step, 95) < np.percentile(old_step, 95)

    projector = InterArmCollisionProjector(margin=0.005)
    initial = projector.data.qpos[projector.qpos_addresses].copy()
    for sl in (slice(0, split), slice(split, None)):
        check_q, check_g, stats = projector.project_trajectory(
            q[sl], gripper_positions=grip[sl], policy="error",
            initial_q=initial)
        assert np.array_equal(check_q, q[sl])
        assert np.array_equal(check_g, grip[sl])
        assert stats.unsafe_candidates == 0
    print("RTMW hybrid simulation replay passed: arm unchanged, grip smoother,"
          " swept collision checks clear.")


if __name__ == "__main__":
    main()
