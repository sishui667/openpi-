"""Confidence gating for MediaPipe arm landmarks.

MediaPipe can report a pose for the whole person while an individual elbow or
wrist is poorly observed. Feeding those inferred landmarks into IK creates
large joint jumps that downstream collision planning cannot repair reliably.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_constants import LANDMARK


@dataclass(frozen=True)
class ArmQualityStats:
    trusted_left: int
    trusted_right: int
    total: int


def _nearest_previous_indices(mask: np.ndarray) -> np.ndarray:
    """Return hold-last-good indices, using the first good frame as prefix."""
    mask = np.asarray(mask, dtype=bool)
    good = np.flatnonzero(mask)
    if not len(good):
        raise ValueError("arm has no trustworthy shoulder/elbow/wrist frame")
    result = np.empty(len(mask), dtype=int)
    last = int(good[0])
    for frame in range(len(mask)):
        if mask[frame]:
            last = frame
        result[frame] = last
    result[:int(good[0])] = int(good[0])
    return result


def gate_arm_landmarks(pose_world: np.ndarray, pose_present: np.ndarray,
                       min_visibility: float = 0.55
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  ArmQualityStats]:
    """Replace untrusted arm landmarks independently for each arm.

    Returns ``(gated_pose, confidence, trusted, stats)``. ``confidence`` and
    ``trusted`` have shape ``(T, 2)`` with columns ``[left, right]``.
    """
    pose = np.asarray(pose_world, dtype=float)
    present = np.asarray(pose_present, dtype=bool)
    if pose.ndim != 3 or pose.shape[1:] != (33, 4):
        raise ValueError(f"pose_world must have shape (T,33,4), got {pose.shape}")
    if present.shape != (len(pose),):
        raise ValueError("pose_present must have shape (T,)")
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be between 0 and 1")

    output = pose.copy()
    confidence = np.empty((len(pose), 2), dtype=np.float32)
    trusted = np.empty((len(pose), 2), dtype=bool)
    for column, side in enumerate(("left", "right")):
        indices = np.asarray([
            LANDMARK[f"{side.upper()}_SHOULDER"],
            LANDMARK[f"{side.upper()}_ELBOW"],
            LANDMARK[f"{side.upper()}_WRIST"],
        ], dtype=int)
        values = pose[:, indices]
        confidence[:, column] = np.min(values[:, :, 3], axis=1)
        finite = np.isfinite(values[:, :, :3]).all(axis=(1, 2))
        trusted[:, column] = (
            present & finite & (confidence[:, column] >= min_visibility))
        source = _nearest_previous_indices(trusted[:, column])
        bad = ~trusted[:, column]
        for landmark in indices:
            output[bad, landmark] = pose[source[bad], landmark]

    stats = ArmQualityStats(
        trusted_left=int(trusted[:, 0].sum()),
        trusted_right=int(trusted[:, 1].sum()),
        total=len(pose),
    )
    return output, confidence, trusted, stats
