"""Repair close-hand hold intervals with lightweight 14-DOF RRT-Connect."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from openarm_common import OPENARM_BASE_OFFSETS  # noqa: E402
from rrt_connect_planner import BimanualRRTConnect  # noqa: E402
from task_space_trajectory_optimizer import (  # noqa: E402
    ReferenceLockedTrajectoryOptimizer,
    TrajectoryMetrics,
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    result = []
    start = previous = int(ids[0])
    for value in ids[1:]:
        value = int(value)
        if value > previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _hold_count(q: np.ndarray, threshold: float = 1e-7) -> int:
    return int(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1) < threshold))


def _both_slow_count(q: np.ndarray, threshold: float = 0.012) -> int:
    delta = np.diff(q, axis=0)
    return int(np.sum((np.linalg.norm(delta[:, :7], axis=1) < threshold)
                      & (np.linalg.norm(delta[:, 7:], axis=1) < threshold)))


def _within_quality_budget(candidate: TrajectoryMetrics,
                           reference: TrajectoryMetrics,
                           max_wrist_regression: float,
                           max_direction_regression: float,
                           max_jerk_ratio: float,
                           max_joint_deviation: float,
                           baseline: TrajectoryMetrics | None = None) -> bool:
    jerk_limit = max(2500.0, reference.jerk_p95 * max_jerk_ratio)
    if baseline is not None:
        jerk_limit = max(jerk_limit, 0.90 * baseline.jerk_p95)
    return (
        candidate.wrist_mean_m
        <= reference.wrist_mean_m + max_wrist_regression
        and candidate.direction_mean_deg
        <= reference.direction_mean_deg + max_direction_regression
        and candidate.direction_p95_deg
        <= reference.direction_p95_deg + 2.0 * max_direction_regression
        and candidate.jerk_p95 <= jerk_limit
        and candidate.max_joint_deviation <= max_joint_deviation
    )


def _quality_key(metrics: TrajectoryMetrics, within_budget: bool,
                 slow_frames: int) -> tuple:
    return (
        not within_budget,
        slow_frames,
        metrics.wrist_mean_m,
        metrics.direction_mean_deg,
        metrics.jerk_p95,
        metrics.max_joint_deviation,
    )


def _prioritized_paths(path: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Turn a coupled detour into left-yields/right-yields coordination."""
    if len(path) < 3:
        return []
    start = np.asarray(path[0], dtype=float)
    via = np.asarray(path[len(path) // 2], dtype=float)
    goal = np.asarray(path[-1], dtype=float)
    left_yields = [
        start,
        np.r_[via[:7], start[7:]],
        np.r_[via[:7], goal[7:]],
        goal,
    ]
    right_yields = [
        start,
        np.r_[start[:7], via[7:]],
        np.r_[goal[:7], via[7:]],
        goal,
    ]
    return [left_yields, right_yields]


def _candidate_windows(q: np.ndarray, wrist_targets: np.ndarray,
                       padding: int, proximity: float,
                       min_hold: int) -> list[tuple[int, int]]:
    hold_edges = np.linalg.norm(np.diff(q, axis=0), axis=1) < 1e-7
    # Edge i means frame i+1 repeats frame i.
    frame_mask = np.zeros(len(q), dtype=bool)
    frame_mask[1:] = hold_edges
    left_world = wrist_targets[:, 0] + OPENARM_BASE_OFFSETS["left"]
    right_world = wrist_targets[:, 1] + OPENARM_BASE_OFFSETS["right"]
    wrist_distance = np.linalg.norm(left_world - right_world, axis=1)
    windows = []
    for first, last in _runs(frame_mask):
        if last - first + 1 < min_hold:
            continue
        start = max(0, first - padding)
        stop = min(len(q) - 1, last + padding)
        if float(np.min(wrist_distance[start:stop + 1])) <= proximity:
            windows.append((start, stop))
    # Padding can make adjacent hold intervals overlap.
    merged = []
    for start, stop in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--proximity", type=float, default=0.40)
    parser.add_argument("--min-hold", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=1800)
    parser.add_argument("--goal-lookahead", type=int, default=24)
    parser.add_argument("--min-goal-distance", type=float, default=0.12)
    parser.add_argument("--max-tracking-regression", type=float, default=0.025,
                        help="为避免双臂冻结，允许绕行窗口平均腕误差最多增加的米数")
    parser.add_argument("--max-direction-regression", type=float, default=11.0,
                        help="绕行窗口相对人体参考允许增加的平均方向误差（度）")
    parser.add_argument("--max-jerk-ratio", type=float, default=2.1,
                        help="绕行窗口 jerk p95 相对人体参考的最大倍率")
    parser.add_argument("--max-joint-deviation", type=float, default=1.60,
                        help="单帧相对人体参考的最大 14D 关节偏移范数（rad）")
    parser.add_argument("--trajopt-iterations", type=int, default=8,
                        help="参考锁定轨迹优化的最大迭代数")
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    data = dict(np.load(args.input))
    if "wrist_targets" not in data:
        raise SystemExit("input has no wrist_targets; regenerate with coupled retargeting")
    joints = np.asarray(data["joint_angles"], dtype=float)
    if "locally_optimized_joint_angles" not in data:
        raise SystemExit(
            "input has no pre-projection reference; regenerate with the updated generator")
    reference = np.asarray(data["locally_optimized_joint_angles"], dtype=float)
    grips = np.asarray(data["gripper_position"], dtype=float)
    wrists = np.asarray(data["wrist_targets"], dtype=float)
    observations = np.asarray(data["target_obs"], dtype=float)
    if len(joints) % 2:
        raise SystemExit("expected original + mirror clips of equal length")

    split = len(joints) // 2
    projector = InterArmCollisionProjector(margin=args.margin)
    planner = BimanualRRTConnect(
        projector, max_iterations=args.max_iterations, seed=args.seed)
    trajectory_optimizer = ReferenceLockedTrajectoryOptimizer(
        projector, fps=float(data["fps"]),
        max_iterations=args.trajopt_iterations)
    output_parts, grip_parts, rows = [], [], []
    for clip_id, (name, sl) in enumerate(
            (("orig", slice(0, split)), ("mirror", slice(split, None)))):
        q = joints[sl].copy()
        ref = reference[sl]
        grip = grips[sl].copy()
        wrist = wrists[sl]
        observation = observations[sl]
        before = _hold_count(q)
        slow_before = _both_slow_count(q)
        windows = _candidate_windows(
            q, wrist, args.padding, args.proximity, args.min_hold)
        solved = direct = rrt = failed = 0
        for window_id, (start, stop) in enumerate(windows):
            # The projected trajectory may still be stuck at ``stop``. Search
            # forward for the first collision-free *nominal* target that has
            # genuinely progressed, and plan to that instead.
            goal_index = None
            for candidate in range(stop, min(len(q), stop + args.goal_lookahead + 1)):
                if (np.linalg.norm(ref[candidate] - q[start])
                        >= args.min_goal_distance
                        and projector.is_safe(ref[candidate], grip[candidate])):
                    goal_index = candidate
                    break
            if goal_index is None:
                failed += 1
                print(f"[{name}] window {start}:{stop} has no safe nominal goal")
                continue
            stop = goal_index
            planner.rng = np.random.default_rng(
                args.seed + clip_id * 1000 + window_id)
            # Preserve a meaningful out-and-back gesture even when the window
            # endpoints are nearly identical. The frame of maximum wrist
            # excursion becomes a semantic via goal if its nominal IK pose is
            # already collision-free.
            excursion = np.linalg.norm(
                wrist[start:stop + 1] - wrist[start], axis=2).max(axis=1)
            peak = start + int(np.argmax(excursion))
            semantic_via = (
                float(np.max(excursion)) >= 0.08
                and peak >= start + 3 and peak <= stop - 3
                and projector.is_safe(ref[peak], grip[peak]))
            path = None
            stats = None
            replacement = None
            if semantic_via:
                first_path, first_stats = planner.plan(
                    q[start], ref[peak], ref[start:peak + 1],
                    grip[start:peak + 1])
                planner.rng = np.random.default_rng(
                    args.seed + 500 + clip_id * 1000 + window_id)
                second_path, second_stats = planner.plan(
                    ref[peak], ref[stop], ref[peak:stop + 1],
                    grip[peak:stop + 1])
                if first_path is not None and second_path is not None:
                    first = planner.resample(
                        first_path, peak - start + 1, grip[start:peak + 1])
                    second = planner.resample(
                        second_path, stop - peak + 1, grip[peak:stop + 1])
                    if first is not None and second is not None:
                        replacement = np.concatenate([first[:-1], second], axis=0)
                        path = first_path[:-1] + second_path
                        stats = second_stats
            if path is None:
                semantic_via = False
                path, stats = planner.plan(
                    q[start], ref[stop], ref[start:stop + 1],
                    grip[start:stop + 1])
            if path is None:
                failed += 1
                print(f"[{name}] window {start}:{stop} RRT failed "
                      f"({stats.nodes} nodes)")
                continue
            if replacement is None:
                replacement = planner.resample(
                    path, stop - start + 1, grip[start:stop + 1])
            if replacement is None:
                failed += 1
                print(f"[{name}] window {start}:{stop} resample rejected")
                continue

            ref_window = ref[start:stop + 1]
            grip_window = grip[start:stop + 1]
            obs_window = observation[start:stop + 1]
            wrist_window = wrist[start:stop + 1]
            reference_metrics = trajectory_optimizer.metrics(
                ref_window, ref_window, obs_window, wrist_window)
            baseline_metrics = trajectory_optimizer.metrics(
                q[start:stop + 1], ref_window, obs_window, wrist_window)
            seeds = [(replacement, path, stats, semantic_via)]

            # A direct path can be safe yet semantically wrong for a hand
            # exchange. Search alternative homotopies, but rank them by full
            # task-space fidelity instead of wrist position alone.
            primary_metrics = trajectory_optimizer.metrics(
                replacement, ref_window, obs_window, wrist_window)
            primary_within = _within_quality_budget(
                primary_metrics, reference_metrics,
                args.max_tracking_regression,
                args.max_direction_regression,
                args.max_jerk_ratio,
                args.max_joint_deviation,
                baseline=baseline_metrics)
            if len(path) == 2 or not primary_within:
                ordinary_bias = (planner.goal_bias, planner.corridor_bias,
                                 planner.corridor_sigma)
                planner.goal_bias = 0.10
                planner.corridor_bias = 0.45
                planner.corridor_sigma = 0.45
                for retry in range(4):
                    planner.rng = np.random.default_rng(
                        args.seed + 10000 + clip_id * 1000
                        + window_id * 10 + retry)
                    detour, detour_stats = planner.plan(
                        q[start], ref[stop], ref_window, grip_window,
                        allow_direct=False, preserve_detour=True)
                    if detour is None:
                        continue
                    path_variants = [detour, *_prioritized_paths(detour)]
                    for path_variant in path_variants:
                        candidate = planner.resample(
                            path_variant, stop - start + 1, grip_window)
                        if candidate is not None:
                            seeds.append((candidate, detour, detour_stats, False))
                (planner.goal_bias, planner.corridor_bias,
                 planner.corridor_sigma) = ordinary_bias

            ranked = []
            for seed_trajectory, seed_path, seed_stats, seed_semantic in seeds:
                metrics = trajectory_optimizer.metrics(
                    seed_trajectory, ref_window, obs_window, wrist_window)
                within = _within_quality_budget(
                    metrics, reference_metrics,
                    args.max_tracking_regression,
                    args.max_direction_regression,
                    args.max_jerk_ratio,
                    args.max_joint_deviation,
                    baseline=baseline_metrics)
                ranked.append((
                    _quality_key(metrics, within,
                                 _both_slow_count(seed_trajectory)),
                    seed_trajectory, seed_path, seed_stats, seed_semantic,
                ))
            ranked.sort(key=lambda item: item[0])

            # Optimize only the strongest topology seed. This keeps runtime
            # bounded and prevents random RRT alternatives from being executed
            # merely because they are collision-free.
            _, seed_trajectory, path, stats, semantic_via = ranked[0]
            replacement = trajectory_optimizer.refine(
                seed_trajectory, ref_window, grip_window,
                obs_window, wrist_window)
            old_holds = _hold_count(q[start:stop + 1])
            new_holds = _hold_count(replacement)
            old_metrics = trajectory_optimizer.metrics(
                q[start:stop + 1], ref_window, obs_window, wrist_window)
            new_metrics = trajectory_optimizer.metrics(
                replacement, ref_window, obs_window, wrist_window)
            within_budget = _within_quality_budget(
                new_metrics, reference_metrics,
                args.max_tracking_regression,
                args.max_direction_regression,
                args.max_jerk_ratio,
                args.max_joint_deviation,
                baseline=old_metrics)
            if new_holds >= old_holds or not within_budget:
                failed += 1
                print(f"[{name}] window {start}:{stop} rejected: "
                      f"holds={old_holds}->{new_holds} "
                      f"wrist={old_metrics.wrist_mean_m:.3f}->"
                      f"{new_metrics.wrist_mean_m:.3f}m "
                      f"direction={old_metrics.direction_mean_deg:.1f}->"
                      f"{new_metrics.direction_mean_deg:.1f}deg "
                      f"direction-p95={new_metrics.direction_p95_deg:.1f}/"
                      f"{reference_metrics.direction_p95_deg + 2.0 * args.max_direction_regression:.1f}deg "
                      f"jerk-p95={old_metrics.jerk_p95:.0f}->"
                      f"{new_metrics.jerk_p95:.0f}/"
                      f"{max(2500.0, reference_metrics.jerk_p95 * args.max_jerk_ratio, 0.90 * old_metrics.jerk_p95):.0f} "
                      f"joint-dev={new_metrics.max_joint_deviation:.2f}/"
                      f"{args.max_joint_deviation:.2f}")
                continue
            q[start:stop + 1] = replacement
            solved += 1
            if semantic_via:
                rrt += 1
                mode = "via/RRT"
            elif len(path) == 2:
                direct += 1
                mode = "direct"
            else:
                rrt += 1
                mode = "RRT"
            print(f"[{name}] window {start}:{stop} {mode} solved; "
                  f"waypoints={len(path)} holds={old_holds}->{new_holds} "
                  f"wrist={old_metrics.wrist_mean_m:.3f}->"
                  f"{new_metrics.wrist_mean_m:.3f}m "
                  f"direction={old_metrics.direction_mean_deg:.1f}->"
                  f"{new_metrics.direction_mean_deg:.1f}deg "
                  f"jerk-p95={old_metrics.jerk_p95:.0f}->"
                  f"{new_metrics.jerk_p95:.0f}")

        # Re-enforce the exact same safety postcondition after all replacements.
        q, grip, safety = projector.project_trajectory(
            q, gripper_positions=grip, policy="project")
        after = _hold_count(q)
        slow_after = _both_slow_count(q)
        rows.append([len(windows), solved, direct, rrt, failed,
                     before, after, safety.held_frames,
                     slow_before, slow_after])
        output_parts.append(q)
        grip_parts.append(grip)
        print(f"[{name}] exchange windows={len(windows)} solved={solved} "
              f"(direct={direct}, RRT={rrt}) holds={before}->{after} "
              f"both-slow={slow_before}->{slow_after} "
              f"fallback-held={safety.held_frames}")

    data["joint_angles"] = np.concatenate(output_parts, axis=0)
    data["gripper_position"] = np.concatenate(grip_parts, axis=0)
    data["exchange_rrt_params"] = np.array([
        args.margin, args.padding, args.proximity, args.min_hold,
        args.max_iterations, args.seed, args.goal_lookahead,
        args.min_goal_distance, args.max_tracking_regression,
        args.max_direction_regression, args.max_jerk_ratio,
        args.max_joint_deviation, args.trajopt_iterations], dtype=np.float32)
    data["trajectory_optimizer"] = np.asarray(
        "reference_locked_task_space_l_bfgs_b")
    data["exchange_rrt_stats"] = np.asarray(rows, dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)

    remaining = sum(not projector.is_safe(q, g) for q, g in zip(
        data["joint_angles"], data["gripper_position"]))
    if remaining:
        raise RuntimeError(f"RRT output violates safety margin in {remaining} frames")
    print(f"[OK] exchange-aware collision-free output -> {args.output}")


if __name__ == "__main__":
    main()
