"""Regression: the mirrored close-hand window needs a non-direct path."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from rrt_connect_planner import BimanualRRTConnect  # noqa: E402


data = np.load(ROOT / "data" / "p2a_test1_coupled.npz")
start, stop = 612 + 138, 612 + 155
q = np.asarray(data["joint_angles"][start:stop + 1], dtype=float)
grip = np.asarray(data["gripper_position"][start:stop + 1], dtype=float)

projector = InterArmCollisionProjector(margin=0.005)
planner = BimanualRRTConnect(projector, max_iterations=1800, seed=1019)
probe = np.asarray([np.median(grip, axis=0)])
assert not planner._edge_safe(q[0], q[-1], probe), (
    "fixture no longer represents a blocked direct connection")

path, stats = planner.plan(q[0], q[-1], q, grip)
assert path is not None and stats.solved
assert len(path) > 2, "expected RRT detour, not a direct interpolation"
resampled = planner.resample(path, len(q), grip)
assert resampled is not None
assert all(projector.is_safe(x, g) for x, g in zip(resampled, grip))

print(f"iterations={stats.iterations} nodes={stats.nodes} "
      f"waypoints={stats.raw_waypoints}->{stats.shortcut_waypoints}")
print("RRT-Connect exchange regression passed.")
