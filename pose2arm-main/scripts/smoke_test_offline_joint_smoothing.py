"""Regression checks for zero-phase offline joint smoothing."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smoothing import offline_joint_trajectory  # noqa: E402


rng = np.random.default_rng(4)
t = np.linspace(0.0, 1.0, 101)
clean = np.column_stack([t, t ** 2, np.sin(2 * np.pi * t)])
noisy = clean + rng.normal(0.0, 0.015, clean.shape)
smoothed = offline_joint_trajectory(noisy, window_length=9, polyorder=2)

raw_jerk = np.percentile(np.linalg.norm(np.diff(noisy, n=3, axis=0), axis=1), 95)
smooth_jerk = np.percentile(np.linalg.norm(np.diff(smoothed, n=3, axis=0), axis=1), 95)
assert smoothed.shape == noisy.shape
assert smooth_jerk < 0.45 * raw_jerk, (raw_jerk, smooth_jerk)
assert np.mean(np.linalg.norm(smoothed - clean, axis=1)) < np.mean(
    np.linalg.norm(noisy - clean, axis=1))

print(f"jerk_p95={raw_jerk:.5f}->{smooth_jerk:.5f}")
print("Offline zero-phase joint smoothing regression passed.")
