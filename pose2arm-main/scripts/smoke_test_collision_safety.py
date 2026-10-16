"""Regression checks for the hard inter-arm collision projector."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402


def main() -> None:
    source = ROOT / "data" / "p2a_test1_stage1.npz"
    trajectory = np.asarray(np.load(source)["joint_angles"], dtype=float)
    gripper = np.asarray(np.load(source)["gripper_position"], dtype=float)
    projector = InterArmCollisionProjector(margin=0.005)
    unsafe_before = sum(not projector.is_safe(q, g)
                        for q, g in zip(trajectory, gripper))
    assert unsafe_before > 0, "regression fixture should contain the reported collision"

    safe, safe_gripper, stats = projector.project_trajectory(
        trajectory, gripper_positions=gripper)
    unsafe_after = sum(not projector.is_safe(q, g)
                       for q, g in zip(safe, safe_gripper))
    assert unsafe_after == 0
    assert stats.unsafe_candidates > 0
    print(f"unsafe_before={unsafe_before} unsafe_after={unsafe_after} "
          f"projected={stats.projected_frames} held={stats.held_frames}")
    print("Collision safety smoke test passed.")


if __name__ == "__main__":
    main()
