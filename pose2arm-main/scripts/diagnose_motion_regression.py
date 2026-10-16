#!/usr/bin/env python3
"""Compare pose tracking and temporal smoothness before/after collision planning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ik_solver import OpenArmIK  # noqa: E402
from openarm_common import (  # noqa: E402
    T_FORE_L,
    T_FORE_R,
    T_PLANE_L,
    T_PLANE_R,
    T_UPPER_L,
    T_UPPER_R,
)


def _angle_deg(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    denom = np.linalg.norm(actual) * np.linalg.norm(target)
    if denom < 1e-12:
        return 0.0
    cosine = np.clip(np.dot(actual, target) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _tracking(q: np.ndarray, obs: np.ndarray,
              wrists: np.ndarray) -> dict[str, np.ndarray]:
    iks = (OpenArmIK("left"), OpenArmIK("right"))
    slices = (
        ((T_UPPER_L, T_FORE_L, T_PLANE_L), (T_UPPER_R, T_FORE_R, T_PLANE_R)),
    )[0]
    wrist_error = np.empty((len(q), 2), dtype=float)
    direction_error = np.empty((len(q), 2, 3), dtype=float)
    for frame, row in enumerate(q):
        for side, (ik, arm_q, arm_slices) in enumerate(zip(
                iks, (row[:7], row[7:]), slices)):
            points = ik.forward_points(arm_q)
            wrist_error[frame, side] = np.linalg.norm(
                points["tool"] - wrists[frame, side])
            for metric, (key, target_slice) in enumerate(zip(
                    ("upper_dir", "forearm_dir", "arm_plane"), arm_slices)):
                direction_error[frame, side, metric] = _angle_deg(
                    points[key], obs[frame, target_slice])
    return {"wrist": wrist_error, "direction": direction_error}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    output = []
    start = previous = int(ids[0])
    for value in ids[1:]:
        value = int(value)
        if value != previous + 1:
            output.append((start, previous))
            start = value
        previous = value
    output.append((start, previous))
    return output


def _summary(name: str, q: np.ndarray, obs: np.ndarray, wrists: np.ndarray,
             fps: float, split: int) -> None:
    track = _tracking(q, obs, wrists)
    wrist_flat = track["wrist"].ravel()
    direction_flat = track["direction"].ravel()
    print(f"\n[{name}]")
    print("  wrist error m: "
          f"mean={np.mean(wrist_flat):.4f} "
          f"p95={np.percentile(wrist_flat, 95):.4f} "
          f"max={np.max(wrist_flat):.4f}")
    print("  direction error deg: "
          f"mean={np.mean(direction_flat):.2f} "
          f"p95={np.percentile(direction_flat, 95):.2f} "
          f"max={np.max(direction_flat):.2f}")
    for clip_name, clip in (("orig", q[:split]), ("mirror", q[split:])):
        velocity = np.diff(clip, axis=0) * fps
        acceleration = np.diff(clip, n=2, axis=0) * fps ** 2
        jerk = np.diff(clip, n=3, axis=0) * fps ** 3
        accel_norm = np.linalg.norm(acceleration, axis=1)
        jerk_norm = np.linalg.norm(jerk, axis=1)
        half = len(jerk_norm) // 2
        print(
            f"  {clip_name} dynamics: "
            f"vel_p95={np.percentile(np.linalg.norm(velocity, axis=1), 95):.1f} "
            f"acc_p95/max={np.percentile(accel_norm, 95):.1f}/{np.max(accel_norm):.1f} "
            f"jerk_p95/max={np.percentile(jerk_norm, 95):.1f}/{np.max(jerk_norm):.1f} "
            f"jerk_second/first={np.percentile(jerk_norm[half:], 95) / max(np.percentile(jerk_norm[:half], 95), 1e-9):.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="NPZ before exchange RRT")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="NPZ after exchange RRT")
    args = parser.parse_args()

    baseline = np.load(args.baseline)
    candidate = np.load(args.candidate)
    fps = float(candidate["fps"])
    final = np.asarray(candidate["joint_angles"], dtype=float)
    split = len(final) // 2
    obs = np.asarray(candidate["target_obs"], dtype=float)
    wrists = np.asarray(candidate["wrist_targets"], dtype=float)
    stages = {
        "nominal IK": np.asarray(candidate["nominal_joint_angles"], dtype=float),
        "local optimizer": np.asarray(
            candidate["locally_optimized_joint_angles"], dtype=float),
        "safety projection": np.asarray(baseline["joint_angles"], dtype=float),
        "exchange RRT final": final,
    }
    print(f"frames={len(final)} split={split} fps={fps:.3f}")
    for name, q in stages.items():
        _summary(name, q, obs, wrists, fps, split)

    projected = stages["safety projection"]
    local = stages["local optimizer"]
    delta_projected = np.linalg.norm(final - projected, axis=1)
    delta_local = np.linalg.norm(final - local, axis=1)
    changed = delta_projected > 1e-6
    print("\n[RRT intervention]")
    print(f"  changed frames={np.sum(changed)}/{len(final)} "
          f"({100 * np.mean(changed):.1f}%)")
    print(f"  deviation from projection rad: "
          f"p95={np.percentile(delta_projected, 95):.3f} "
          f"max={np.max(delta_projected):.3f}")
    print(f"  deviation from human-tracking local rad: "
          f"p95={np.percentile(delta_local, 95):.3f} "
          f"max={np.max(delta_local):.3f}")
    for offset, clip_name in ((0, "orig"), (split, "mirror")):
        mask = changed[offset:offset + split]
        print(f"  {clip_name} changed segments={_segments(mask)}")

    # Attribute the largest jerk peaks to RRT-modified frames and their seams.
    for offset, clip_name in ((0, "orig"), (split, "mirror")):
        clip = final[offset:offset + split]
        jerk = np.linalg.norm(np.diff(clip, n=3, axis=0) * fps ** 3, axis=1)
        top = np.argsort(jerk)[-10:][::-1]
        rows = []
        for index in top:
            frame = int(index + 2)
            lo, hi = max(0, frame - 2), min(split, frame + 3)
            touched = bool(np.any(changed[offset + lo:offset + hi]))
            source_frame = int(candidate["frame_indices"][offset + frame])
            rows.append((frame, source_frame, round(float(jerk[index]), 1), touched))
        print(f"  {clip_name} top jerk (clip_frame, video_frame, value, near_RRT)={rows}")


if __name__ == "__main__":
    main()
