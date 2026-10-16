"""Regression checks for per-arm landmark confidence gating."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pose_constants import LANDMARK  # noqa: E402
from pose_quality import gate_arm_landmarks  # noqa: E402


pose = np.zeros((6, 33, 4), dtype=float)
pose[:, :, 3] = 1.0
present = np.ones(6, dtype=bool)

left_wrist = LANDMARK["LEFT_WRIST"]
right_wrist = LANDMARK["RIGHT_WRIST"]
pose[:, left_wrist, 0] = np.arange(6, dtype=float)
pose[:, right_wrist, 0] = 10.0 + np.arange(6, dtype=float)

# The left wrist glitches while the independently trustworthy right arm moves.
pose[2:4, left_wrist, 0] = (100.0, 200.0)
pose[2:4, left_wrist, 3] = 0.1

gated, confidence, trusted, stats = gate_arm_landmarks(
    pose, present, min_visibility=0.55)

assert np.allclose(gated[2:4, left_wrist, 0], 1.0)
assert np.allclose(gated[2:4, right_wrist, 0], (12.0, 13.0))
assert not trusted[2:4, 0].any() and trusted[2:4, 1].all()
assert np.allclose(confidence[2:4, 0], 0.1)
assert stats.trusted_left == 4 and stats.trusted_right == 6

print("Per-arm pose quality gating regression passed.")
