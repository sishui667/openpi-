"""Regression: refine a real exchange window without losing safety/fidelity."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from task_space_trajectory_optimizer import (  # noqa: E402
    ReferenceLockedTrajectoryOptimizer,
)


data = np.load(ROOT / "data" / "p2a_test1_exchange_rrt.npz")
start, stop = 158, 194
seed = np.asarray(data["joint_angles"][start:stop], dtype=float)
reference = np.asarray(
    data["locally_optimized_joint_angles"][start:stop], dtype=float)
grip = np.asarray(data["gripper_position"][start:stop], dtype=float)
obs = np.asarray(data["target_obs"][start:stop], dtype=float)
wrists = np.asarray(data["wrist_targets"][start:stop], dtype=float)

projector = InterArmCollisionProjector(margin=0.005)
optimizer = ReferenceLockedTrajectoryOptimizer(
    projector, fps=float(data["fps"]), control_points=4, max_iterations=2)
before = optimizer._objective(seed, reference, obs, wrists)
refined = optimizer.refine(seed, reference, grip, obs, wrists)
after = optimizer._objective(refined, reference, obs, wrists)
metrics = optimizer.metrics(refined, reference, obs, wrists)

assert optimizer._edges_safe(refined, grip)
assert after < before, (before, after)
assert metrics.direction_mean_deg < 20.0
assert metrics.jerk_p95 < 4000.0

print(f"objective={before:.3f}->{after:.3f} "
      f"direction={metrics.direction_mean_deg:.1f}deg "
      f"jerk_p95={metrics.jerk_p95:.0f}")
print("Reference-locked task-space trajectory regression passed.")
