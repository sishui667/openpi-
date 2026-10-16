"""Optimize collision windows, then enforce safety with hard projection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from local_trajectory_optimizer import SlidingWindowTrajectoryOptimizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = dict(np.load(args.input))
    joints = np.asarray(data["joint_angles"], dtype=float)
    gripper = np.asarray(data["gripper_position"], dtype=float)
    if len(joints) != len(gripper) or len(joints) % 2:
        raise SystemExit("joint_angles / gripper_position 长度不匹配")

    split = len(joints) // 2
    projector = InterArmCollisionProjector(margin=args.margin)
    optimizer = SlidingWindowTrajectoryOptimizer(
        projector, padding=args.padding, population=args.population,
        iterations=args.iterations, seed=args.seed)

    safe_parts, grip_parts, opt_rows, safety_rows = [], [], [], []
    for clip_id, (name, sl) in enumerate(
            (("orig", slice(0, split)), ("mirror", slice(split, None)))):
        # Different deterministic stream for the mirrored clip.
        optimizer.rng = np.random.default_rng(args.seed + clip_id)
        optimized, opt = optimizer.optimize(joints[sl], gripper[sl])
        safe, safe_grip, safety = projector.project_trajectory(
            optimized, gripper_positions=gripper[sl], policy="project")
        safe_parts.append(safe)
        grip_parts.append(safe_grip)
        opt_rows.append([opt.unsafe_before, opt.unsafe_after,
                         opt.improved_windows, len(opt.windows), opt.evaluations])
        safety_rows.append([safety.unsafe_candidates, safety.projected_frames,
                            safety.held_frames])
        print(f"[{name}] optimizer unsafe {opt.unsafe_before} → {opt.unsafe_after}; "
              f"windows {opt.improved_windows}/{len(opt.windows)}; "
              f"fallback projected={safety.projected_frames} held={safety.held_frames}")

    data["joint_angles"] = np.concatenate(safe_parts, axis=0)
    data["gripper_position"] = np.concatenate(grip_parts, axis=0)
    data["local_optimization_params"] = np.array([
        args.margin, args.padding, args.population, args.iterations, args.seed],
        dtype=np.float32)
    data["local_optimization_stats"] = np.asarray(opt_rows, dtype=np.int32)
    data["collision_safety_params"] = np.array([
        args.margin, projector.sweep_samples, projector.bisection_steps],
        dtype=np.float32)
    data["collision_safety_stats"] = np.asarray(safety_rows, dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)

    remaining = sum(not projector.is_safe(q, g)
                    for q, g in zip(data["joint_angles"], data["gripper_position"]))
    if remaining:
        raise RuntimeError(f"hard safety postcondition failed: {remaining} frames")
    print(f"[OK] collision-free output → {args.output}")


if __name__ == "__main__":
    main()
