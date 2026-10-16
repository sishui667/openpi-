"""Build a safety-checked simulation replay with RTMW3D hand tracking.

The arm trajectory stays on the validated metric MediaPipe/reference-locked
retargeting path. RTMW3D crop-space 3D coordinates are deliberately not fed
to IK. This adapter only changes the continuous gripper signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402
from gripper_map import map_positions  # noqa: E402


def guarded_pinch_ratio(xy: np.ndarray, score: np.ndarray, start: int,
                        width: int, height: int,
                        threshold: float = 0.3) -> np.ndarray:
    """Return palm-normalized thumb/index distance; invalid frames are NaN."""
    xy = np.asarray(xy, dtype=float)
    score = np.asarray(score, dtype=float)
    ids = np.arange(start, start + 21)
    hand = xy[:, ids]
    inside = (np.isfinite(hand).all(axis=(1, 2))
              & (hand >= 0).all(axis=(1, 2))
              & (hand[..., 0] < width).all(axis=1)
              & (hand[..., 1] < height).all(axis=1)
              & (score[:, ids] >= threshold).all(axis=1))
    palm = np.linalg.norm(hand[:, 5] - hand[:, 17], axis=1)
    pinch = np.linalg.norm(hand[:, 4] - hand[:, 8], axis=1)
    ratio = pinch / np.maximum(palm, 1.0)
    valid = inside & (palm >= 12.0) & (ratio >= 0) & (ratio <= 3.5)

    wrist = hand[:, 0]
    max_jump = 0.12 * np.hypot(width, height)
    for frame in range(1, len(xy) - 1):
        if not (valid[frame - 1] and valid[frame] and valid[frame + 1]):
            continue
        before = np.linalg.norm(wrist[frame] - wrist[frame - 1])
        after = np.linalg.norm(wrist[frame] - wrist[frame + 1])
        bridge = np.linalg.norm(wrist[frame - 1] - wrist[frame + 1])
        if before > max_jump and after > max_jump and bridge < max_jump:
            valid[frame] = False
    ratio[~valid] = np.nan
    return ratio


def centered_median_valid(ratio: np.ndarray) -> np.ndarray:
    """Reject one-frame hand-ratio spikes without filling missing detections."""
    out = np.asarray(ratio, dtype=float).copy()
    for frame in range(1, len(out) - 1):
        window = ratio[frame - 1:frame + 2]
        if np.isfinite(window).all():
            out[frame] = np.median(window)
    return out


def slew_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values.copy()
    out = np.empty_like(values)
    out[0] = float(np.clip(values[0], 0, 1))
    for frame in range(1, len(values)):
        out[frame] = np.clip(values[frame], out[frame - 1] - max_step,
                             out[frame - 1] + max_step)
    return np.clip(out, 0, 1).astype(np.float32)


def safe_grip_with_fixed_arms(projector: InterArmCollisionProjector,
                              q: np.ndarray, baseline: np.ndarray,
                              candidate: np.ndarray,
                              initial_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accept as much new grip as possible without changing any arm joint.

    Each trial checks the incoming swept transition and a one-frame lookahead
    to the already-validated baseline, so a local choice cannot trap the next
    frame at the collision boundary.
    """
    out = np.empty_like(baseline)
    factors = np.zeros(len(q), dtype=np.float32)
    previous_q = np.asarray(initial_q, dtype=float)
    previous_g = np.zeros(2, dtype=float)
    choices = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0)
    for i in range(len(q)):
        accepted = False
        for factor in choices:
            trial = baseline[i] + factor * (candidate[i] - baseline[i])
            if factor > 0:
                trial = np.clip(trial, previous_g - 0.10,
                                previous_g + 0.10)
            if not projector.is_safe(q[i], trial):
                continue
            _, _, changed, _ = projector.project_step(
                previous_q, q[i], previous_g, trial)
            if changed:
                continue
            if i + 1 < len(q):
                _, _, future_changed, _ = projector.project_step(
                    q[i], q[i + 1], trial, baseline[i + 1])
                if future_changed:
                    continue
            out[i] = trial
            factors[i] = factor
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"no safe fixed-arm grip at frame {i}")
        previous_q, previous_g = q[i], out[i]
    return out, factors


def build(base_path: Path, rtmw_path: Path, output: Path,
          report_path: Path) -> None:
    base = dict(np.load(base_path, allow_pickle=False))
    rt = dict(np.load(rtmw_path, allow_pickle=False))
    q = np.asarray(base["joint_angles"], dtype=np.float32)
    old_grip = np.asarray(base["gripper_position"], dtype=np.float32)
    frame_indices = np.asarray(base["frame_indices"], dtype=int)
    split = len(q) // 2
    if len(q) % 2 or q.shape[1] != 14 or old_grip.shape != (len(q), 2):
        raise ValueError("baseline must contain matching original/mirror arm and grip frames")
    if rt["image_xy"].shape != (len(rt["score"]), 133, 2):
        raise ValueError("RTMW3D NPZ must have 133 image-space keypoints")
    if np.min(frame_indices) < 0 or np.max(frame_indices) >= len(rt["image_xy"]):
        raise ValueError("RTMW3D extraction does not cover all baseline frames")
    fps = float(base["fps"])
    width, height = int(rt["frame_width"]), int(rt["frame_height"])
    ratios = {}
    closures = {}
    for side, start in (("left", 91), ("right", 112)):
        ratio = guarded_pinch_ratio(rt["image_xy"], rt["score"], start,
                                    width, height)
        ratios[side] = centered_median_valid(ratio)
        closures[side] = map_positions(ratios[side], fps,
                                       min_cutoff=2.5, beta=1.0)

    candidate = old_grip.copy()
    source_valid = np.array([np.isfinite(ratios["right"]),
                             np.isfinite(ratios["left"])], dtype=bool).T
    blend = 0.70
    for clip_id, sl in enumerate((slice(0, split), slice(split, None))):
        indices = frame_indices[sl]
        for column, side in enumerate(("right", "left") if clip_id == 0
                                      else ("left", "right")):
            valid = np.isfinite(ratios[side][indices])
            target = candidate[sl, column].copy()
            target[valid] = (blend * closures[side][indices[valid]]
                             + (1 - blend) * old_grip[sl, column][valid])
            candidate[sl, column] = slew_limit(target, max_step=0.10)

    projector = InterArmCollisionProjector(margin=0.005)
    initial_q = np.asarray(projector.data.qpos[projector.qpos_addresses],
                           dtype=float).copy()
    safe_grip = np.empty_like(candidate)
    safety = []
    reverted_clips = []
    accepted_factors = []
    for clip_id, sl in enumerate((slice(0, split), slice(split, None))):
        try:
            safe_grip[sl], factors = safe_grip_with_fixed_arms(
                projector, q[sl], old_grip[sl], candidate[sl], initial_q)
        except RuntimeError:
            safe_grip[sl] = old_grip[sl]
            factors = np.zeros(split, dtype=np.float32)
            reverted_clips.append(clip_id)
        accepted_factors.append(factors)
        projected_q, projected_g, stats = projector.project_trajectory(
            q[sl], gripper_positions=safe_grip[sl], policy="project",
            initial_q=initial_q)
        if np.max(np.abs(projected_q - q[sl])) > 1e-6:
            safe_grip[sl] = old_grip[sl]
            factors[:] = 0
            reverted_clips.append(clip_id)
            _, _, stats = projector.project_trajectory(
                q[sl], gripper_positions=old_grip[sl], policy="error",
                initial_q=initial_q)
        else:
            safe_grip[sl] = projected_g
        safety.append([stats.unsafe_candidates, stats.projected_frames,
                       stats.held_frames])
    for joint, grip in zip(q, safe_grip):
        if not projector.is_safe(joint, grip):
            raise RuntimeError("unsafe output after gripper projection")

    base["gripper_position"] = safe_grip.astype(np.float32)
    base["gripper_cmd"] = (safe_grip > 0.5).astype(np.float32)
    base["source_rtmw_hand_valid"] = source_valid
    base["source_rtmw_pinch_ratio"] = np.stack(
        [ratios["right"], ratios["left"]], axis=1).astype(np.float32)
    base["hybrid_grip_params"] = np.array([blend, 0.10, 0.3, 0.005],
                                           dtype=np.float32)
    base["hybrid_collision_stats"] = np.asarray(safety, dtype=np.int32)
    base["hybrid_reverted_clips"] = np.asarray(reverted_clips, dtype=np.int32)
    base["hybrid_accepted_blend"] = np.concatenate(accepted_factors)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **base)

    def metrics(array: np.ndarray) -> tuple[float, float]:
        changes = np.abs(np.diff(array[:split], axis=0))
        return float(np.percentile(changes, 95)), float(np.max(changes))

    old_p95, old_max = metrics(old_grip)
    new_p95, new_max = metrics(safe_grip)
    changed = int(np.count_nonzero(np.abs(safe_grip - old_grip) > 1e-3))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# RTMW3D 仿真接入验证\n\n"
        f"基线：`{base_path.resolve()}`\n\n"
        f"候选回放：`{output.resolve()}`\n\n"
        "本版本仅接入 RTMW3D 手指关键点到夹爪连续开合；双臂关节轨迹与基线逐值相同。"
        "RTMW3D 的非米制三维坐标未进入 IK。\n\n"
        f"- 有效 RTMW3D 手部帧：右 {int(source_valid[:, 0].sum())}/{len(source_valid)}，"
        f"左 {int(source_valid[:, 1].sum())}/{len(source_valid)}。\n"
        f"- 夹爪输出改变的采样数：{changed}/{safe_grip.size}。\n"
        f"- 原片夹爪相邻帧变化量 p95/max：基线 {old_p95:.4f}/{old_max:.4f}，"
        f"候选 {new_p95:.4f}/{new_max:.4f}。\n"
        f"- 新手势目标的每帧变化限制为 0.10；遇到安全回退时可能沿用基线较大的步长。"
        f"碰撞统计 [unsafe, projected, held]：{safety}。\n"
        f"- 因会改变安全手臂轨迹而回退的片段：{reverted_clips or '无'}。\n\n"
        f"- 接受新手势的帧数（原片/镜像）："
        f"{[int((a > 0).sum()) for a in accepted_factors]}。\n\n"
        "局限：以上是仿真几何安全与时序平滑指标，不是手势真值精度。"
        "RTMW3D 遮挡/快速近镜头动作仍可能误检；不可直接用于实体机械臂。\n",
        encoding="utf-8")
    print(f"saved {output}; changed={changed}; safety={safety}; "
          f"accepted_frames={[int((a > 0).sum()) for a in accepted_factors]}; "
          f"reverted={reverted_clips}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "data" / "p2a_test1_reference_locked.npz")
    parser.add_argument("--rtmw", type=Path,
                        default=ROOT / "out" / "pose_ab" / "rtmw3d.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "p2a_test1_rtmw_hybrid_sim.npz")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "out" / "pose_ab" / "rtmw_hybrid_sim_report.md")
    args = parser.parse_args()
    build(args.baseline, args.rtmw, args.output, args.report)


if __name__ == "__main__":
    main()
