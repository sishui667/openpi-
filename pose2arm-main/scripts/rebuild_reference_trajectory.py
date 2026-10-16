#!/usr/bin/env python3
"""Rebuild smoothing/local-avoidance/safety stages from saved nominal IK."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from local_trajectory_optimizer import SlidingWindowTrajectoryOptimizer  # noqa: E402
from openarm_common import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS  # noqa: E402
from smoothing import offline_joint_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = dict(np.load(args.input))
    nominal = np.asarray(data["nominal_joint_angles"], dtype=float)
    grip = np.asarray(data["gripper_position"], dtype=float).copy()
    if len(nominal) % 2:
        raise SystemExit("expected original + mirror clips of equal length")
    split = len(nominal) // 2
    low = np.concatenate([LEFT_JOINT_LIMITS[:, 0], RIGHT_JOINT_LIMITS[:, 0]])
    high = np.concatenate([LEFT_JOINT_LIMITS[:, 1], RIGHT_JOINT_LIMITS[:, 1]])
    smoothed = np.concatenate([
        offline_joint_trajectory(
            nominal[:split], args.window, args.polyorder),
        offline_joint_trajectory(
            nominal[split:], args.window, args.polyorder),
    ], axis=0)
    smoothed = np.clip(smoothed, low, high).astype(np.float32)

    projector = InterArmCollisionProjector(margin=args.margin)
    optimizer = SlidingWindowTrajectoryOptimizer(
        projector, padding=args.padding, population=args.population,
        iterations=args.iterations, seed=args.seed)
    local_parts = []
    safe_parts = []
    grip_parts = []
    local_stats = np.zeros((2, 5), dtype=np.int32)
    safety_stats = np.zeros((2, 3), dtype=np.int32)
    for clip_id, (name, sl) in enumerate((
            ("orig", slice(0, split)), ("mirror", slice(split, None)))):
        optimizer.rng = np.random.default_rng(args.seed + clip_id)
        local, stats = optimizer.optimize(smoothed[sl], grip[sl])
        safe, safe_grip, safety = projector.project_trajectory(
            local, gripper_positions=grip[sl], policy="project")
        local_parts.append(local)
        safe_parts.append(safe)
        grip_parts.append(safe_grip)
        local_stats[clip_id] = [
            stats.unsafe_before, stats.unsafe_after,
            stats.improved_windows, len(stats.windows), stats.evaluations]
        safety_stats[clip_id] = [
            safety.unsafe_candidates, safety.projected_frames,
            safety.held_frames]
        print(f"[{name}] local unsafe={stats.unsafe_before}->{stats.unsafe_after} "
              f"safety projected={safety.projected_frames} "
              f"held={safety.held_frames}")

    data["nominal_joint_angles"] = smoothed
    data["locally_optimized_joint_angles"] = np.concatenate(local_parts)
    data["joint_angles"] = np.concatenate(safe_parts)
    data["gripper_position"] = np.concatenate(grip_parts)
    data["offline_joint_smoothing_params"] = np.asarray(
        [args.window, args.polyorder], dtype=np.int32)
    data["local_optimization_stats"] = local_stats
    data["collision_safety_stats"] = safety_stats
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print(f"[OK] rebuilt reference trajectory -> {args.output}")


if __name__ == "__main__":
    main()
