"""Regression checks for coupled dual-arm position retargeting."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openarm_common import OPENARM_ARM_LENGTH, OPENARM_BASE_OFFSETS  # noqa: E402
from retarget import coupled_wrist_targets, shared_arm_scale  # noqa: E402


# Deliberately asymmetric landmark estimates: per-side calibration would use
# different gains and distort the relationship between the hands.
frames = {
    "left": [
        (np.array([-0.15, 0.0, 0.0]), np.array([-0.20, 0.18, -0.05]),
         np.array([-0.30, 0.10, -0.15])),
        (np.array([-0.15, 0.0, 0.0]), np.array([-0.20, 0.17, -0.05]),
         np.array([-0.30, 0.10, -0.15])),
    ],
    "right": [
        (np.array([0.15, 0.0, 0.0]), np.array([0.21, 0.20, -0.06]),
         np.array([0.31, 0.10, -0.16])),
        (np.array([0.15, 0.0, 0.0]), np.array([0.21, 0.19, -0.06]),
         np.array([0.31, 0.10, -0.16])),
    ],
}

scale = shared_arm_scale(frames, OPENARM_ARM_LENGTH)
assert np.isfinite(scale) and scale > 0

left_wrist = frames["left"][0][2]
right_wrist = frames["right"][0][2]
targets = coupled_wrist_targets(left_wrist, right_wrist, scale)

left_world = targets["left"] + OPENARM_BASE_OFFSETS["left"]
right_world = targets["right"] + OPENARM_BASE_OFFSETS["right"]
target_separation = np.linalg.norm(left_world - right_world)
human_separation = scale * np.linalg.norm(left_wrist - right_wrist)

# Away from reach saturation, a shared transform must preserve inter-wrist
# separation. Independent shoulder-local direction mapping cannot guarantee it.
assert np.isclose(target_separation, human_separation, rtol=0.02), (
    target_separation, human_separation)
for side in ("left", "right"):
    assert np.linalg.norm(targets[side]) <= 0.95 * OPENARM_ARM_LENGTH + 1e-9

print(f"shared_scale={scale:.3f} wrist_separation={target_separation:.3f}m")
print("Coupled retarget smoke test passed.")
