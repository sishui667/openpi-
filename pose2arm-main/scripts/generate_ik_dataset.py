#!/usr/bin/env python3
"""Build the full RL training dataset from a video in one unified pipeline.

Pipeline:
    1. MediaPipe pose extraction + One Euro smoothing
    2. Retarget to OpenArm robot frame → PoseTarget per arm
    3. Build target_obs (N, 18) from original frames
    4. Mirror target_obs across the sagittal plane → (N, 18) mirrored
    5. Phase-1 IK (20 restarts) on all 2N poses
    6. Phase-2 14-DOF joint refinement with inter-arm collision penalty on all 2N poses
    7. Save data/rl_poses.npz — (2N, 18) target_obs + (2N, 14) joint_angles
       Layout: [0:N] original clip, [N:2N] mirror clip (both temporally ordered)

Usage:
    python3 scripts/generate_ik_dataset.py
    python3 scripts/generate_ik_dataset.py --input videos/custom.mp4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extract_pose import extract_from_video
from extract_hand import extract_hands_from_video, normalized_pinch_distance, pinch_distance
from gripper_map import (
    DEFAULT_CLOSE_M, DEFAULT_OPEN_M, DEFAULT_HOLD_FRAMES, DEFAULT_LOST_FRAMES,
    DEFAULT_CLOSE_RATIO, DEFAULT_OPEN_RATIO,
    DEFAULT_GRIPPER_MIN_CUTOFF, DEFAULT_GRIPPER_BETA,
    map_positions, map_states,
)
from ik_solver import OpenArmIK, POSE_LIKENESS_WEIGHTS
from pose_constants import LANDMARK
from retarget import (
    OPENARM_ANCHOR, OPENARM_ARM_LENGTH, ARM_BETA, ARM_MIN_CUTOFF,
    Retargeter, PoseTarget, compute_frame_sequence, coupled_wrist_targets,
    openarm_body_to_robot_for_side, to_body_frame,
    shared_arm_scale,
)
from smoothing import offline_joint_trajectory, one_euro_sequence
from pose_quality import gate_arm_landmarks
from collision_safety import InterArmCollisionProjector
from local_trajectory_optimizer import SlidingWindowTrajectoryOptimizer
from openarm_common import (
    JOINT_VELOCITY_LIMITS,
    OPENARM_BASE_OFFSETS,
    T_UPPER_L, T_FORE_L, T_PLANE_L,
    T_UPPER_R, T_FORE_R, T_PLANE_R,
)

# ── IK hyper-parameters ───────────────────────────────────────────────────────
N_RESTARTS     = 20    # per-arm Phase-1 restarts
N_RESTARTS_14D = 5     # Phase-2 14D refinement restarts
MAX_NFEV       = 2000
MAX_NFEV_14D   = 3000
_MIN_ARM_DIST  = 0.08  # 8 cm minimum separation between arm key-points
_COLLISION_W   = 15.0  # collision penalty weight

_OFFLINE_WEIGHTS = {
    **POSE_LIKENESS_WEIGHTS,
    "smooth":         0.20,
    "step_margin":    1.0,
    "shoulder_home":  0.0,
    "home":           0.01,
    # Preserve the torso-centred bilateral wrist geometry strongly enough that
    # direction matching cannot pull both robot hands into the same corridor.
    "wrist_position": 150.0,
    "wrist_dir":      0.0,   # zeroed: target_obs stores no wrist segment lengths, normalize(u+f) approximation is unreliable
}

# Approximate sign pattern for mirroring joint angles (warm-start only)
_APPROX_SIGNS = np.array([1, 1, 1, -1, 1, 1, 1], dtype=float)

# Sagittal-plane mirror transforms
_FLIP_Y      = np.array([1, -1, 1], dtype=np.float32)   # regular vectors
_FLIP_PSEUDO = np.array([-1, 1, -1], dtype=np.float32)  # pseudovectors (cross products)


# ── Target obs helpers ────────────────────────────────────────────────────────

def _build_target_obs(pt_l, pt_r) -> np.ndarray:
    out = np.zeros(18, dtype=np.float32)
    out[T_UPPER_L] = pt_l.upper_dir
    out[T_FORE_L]  = pt_l.forearm_dir
    out[T_PLANE_L] = pt_l.arm_plane
    out[T_UPPER_R] = pt_r.upper_dir
    out[T_FORE_R]  = pt_r.forearm_dir
    out[T_PLANE_R] = pt_r.arm_plane
    return out


def _mirror_target_obs(obs: np.ndarray) -> np.ndarray:
    """Mirror (N,18) target_obs across sagittal plane, swapping L/R arms."""
    return np.concatenate([
        obs[:, T_UPPER_R] * _FLIP_Y,
        obs[:, T_FORE_R]  * _FLIP_Y,
        obs[:, T_PLANE_R] * _FLIP_PSEUDO,
        obs[:, T_UPPER_L] * _FLIP_Y,
        obs[:, T_FORE_L]  * _FLIP_Y,
        obs[:, T_PLANE_L] * _FLIP_PSEUDO,
    ], axis=1).astype(np.float32)


def _mirror_wrist_targets(targets: np.ndarray) -> np.ndarray:
    """Swap arms and mirror shoulder-local wrist targets through world space."""
    targets = np.asarray(targets, dtype=float)
    result = np.empty_like(targets)
    right_world = targets[:, 1] + OPENARM_BASE_OFFSETS["right"]
    left_world = targets[:, 0] + OPENARM_BASE_OFFSETS["left"]
    result[:, 0] = right_world * _FLIP_Y - OPENARM_BASE_OFFSETS["left"]
    result[:, 1] = left_world * _FLIP_Y - OPENARM_BASE_OFFSETS["right"]
    return result.astype(np.float32)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _reconstruct_pose_target(upper_dir, forearm_dir, arm_plane) -> PoseTarget:
    u = np.asarray(upper_dir, float)
    f = np.asarray(forearm_dir, float)
    return PoseTarget(
        upper_dir=u,
        forearm_dir=f,
        wrist_dir=_normalize(u + f),
        arm_plane=np.asarray(arm_plane, float),
        elbow_angle=float(np.arccos(np.clip(np.dot(-u, f), -1.0, 1.0))),
        wrist_position=np.zeros(3),
        elbow_position=np.zeros(3),
        orientation=np.eye(3),
    )


# ── IK helpers ────────────────────────────────────────────────────────────────

def _arm_key_points(ik: OpenArmIK, q: np.ndarray) -> np.ndarray:
    pts = ik.forward_points(q)
    jp  = np.asarray(pts["joint_positions"])
    return np.array([jp[0], jp[2], jp[3], jp[5], pts["tool"]], dtype=float)


def _collision_residual(ik_l, ik_r, q_l, q_r) -> np.ndarray:
    kp_l = _arm_key_points(ik_l, q_l)
    kp_r = _arm_key_points(ik_r, q_r)
    res = []
    for pl in kp_l:
        for pr in kp_r:
            dist = float(np.linalg.norm(pl - pr))
            res.append(np.sqrt(_COLLISION_W) * max(0.0, _MIN_ARM_DIST - dist) / _MIN_ARM_DIST)
    return np.array(res, dtype=float)


def _min_arm_dist(ik_l, ik_r, q_l, q_r) -> float:
    kp_l = _arm_key_points(ik_l, q_l)
    kp_r = _arm_key_points(ik_r, q_r)
    return float(min(np.linalg.norm(pl - pr) for pl in kp_l for pr in kp_r))


def solve_offline(ik: OpenArmIK, pose_target, first_seed=None,
                  n_restarts=N_RESTARTS, max_nfev=MAX_NFEV):
    """Multi-restart 7-DOF IK. Returns (q, cost); q=None on total failure."""
    rng = np.random.default_rng()
    if first_seed is not None:
        # slot 0: caller's warm-start, slot 1: home_seed, rest: random
        slot0, slot1 = first_seed.copy(), ik.home_seed.copy()
    else:
        # slot 0: home_seed, slot 1: random (avoids duplicate home_seed restart)
        slot0 = ik.home_seed.copy()
        slot1 = rng.uniform(ik.joint_limits[:, 0], ik.joint_limits[:, 1])
    seeds = [slot0, slot1,
             *[rng.uniform(ik.joint_limits[:, 0], ik.joint_limits[:, 1])
               for _ in range(n_restarts - 2)]]
    best_q, best_cost = None, float("inf")
    for seed in seeds:
        try:
            res = least_squares(
                lambda q, s=seed: ik.pose_likeness_residual(
                    q, pose_target, seed=s, weights=_OFFLINE_WEIGHTS),
                seed,
                bounds=(ik.joint_limits[:, 0], ik.joint_limits[:, 1]),
                max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7,
            )
        except Exception:
            continue
        q = res.x
        if not np.isfinite(q).all():
            continue
        if (np.any(q < ik.joint_limits[:, 0] - 1e-4)
                or np.any(q > ik.joint_limits[:, 1] + 1e-4)):
            continue
        cost = float(np.dot(res.fun, res.fun))
        if cost < best_cost:
            best_cost, best_q = cost, q.copy()
    return best_q, best_cost


def solve_14d(ik_l, ik_r, pt_l, pt_r, q0_l, q0_r,
              reference_l=None, reference_r=None,
              n_restarts=N_RESTARTS_14D, max_nfev=MAX_NFEV_14D):
    """Simultaneous 14-DOF refinement with inter-arm collision penalty."""
    lo = np.concatenate([ik_l.joint_limits[:, 0], ik_r.joint_limits[:, 0]])
    hi = np.concatenate([ik_l.joint_limits[:, 1], ik_r.joint_limits[:, 1]])
    q0 = np.concatenate([q0_l, q0_r])
    reference_l = q0_l if reference_l is None else np.asarray(reference_l, dtype=float)
    reference_r = q0_r if reference_r is None else np.asarray(reference_r, dtype=float)

    def residual(q):
        ql, qr = q[:7], q[7:]
        return np.concatenate([
            ik_l.pose_likeness_residual(ql, pt_l, seed=reference_l, weights=_OFFLINE_WEIGHTS),
            ik_r.pose_likeness_residual(qr, pt_r, seed=reference_r, weights=_OFFLINE_WEIGHTS),
            _collision_residual(ik_l, ik_r, ql, qr),
        ])

    rng   = np.random.default_rng()
    seeds = [q0.copy(),
             *[np.clip(q0 + rng.normal(0.0, 0.15, 14), lo, hi)
               for _ in range(n_restarts - 1)]]
    best_q, best_cost = None, float("inf")
    for seed in seeds:
        try:
            res = least_squares(residual, seed, bounds=(lo, hi),
                                max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        except Exception:
            continue
        q = res.x
        if not np.isfinite(q).all():
            continue
        cost = float(np.dot(res.fun, res.fun))
        if cost < best_cost:
            best_cost, best_q = cost, q.copy()

    if best_q is None:
        return q0_l.copy(), q0_r.copy(), float("inf")
    return best_q[:7].copy(), best_q[7:].copy(), best_cost


# ── Pose extraction ───────────────────────────────────────────────────────────

def _extract_body_data(video_path, delegate="cpu", min_arm_visibility=0.55):
    data       = extract_from_video(video_path, delegate=delegate)
    pose_world = data["pose_world"]
    present    = data["pose_present"]
    fps        = float(data["fps"])
    T          = len(present)
    print(f"  {T} frames @ {fps:.1f} fps, detected {int(present.sum())}")

    pose_world, arm_confidence, arm_trusted, quality = gate_arm_landmarks(
        pose_world, present, min_visibility=min_arm_visibility)
    print("  Arm landmark quality: "
          f"left={quality.trusted_left}/{quality.total} "
          f"right={quality.trusted_right}/{quality.total} "
          f"(visibility >= {min_arm_visibility:.2f})")

    hold = np.full(T, -1, dtype=int)
    last = -1
    for t in range(T):
        if present[t]:
            last = t
        hold[t] = last
    valid = hold >= 0
    if not valid.any():
        raise RuntimeError("no pose detected")
    hold[~valid] = hold[valid][0]

    root_idx = [LANDMARK[k] for k in
                ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")]
    frames = compute_frame_sequence(pose_world[hold][:, root_idx, :3], fps)

    sides  = ("left", "right")
    lm_idx = {s: {k: LANDMARK[f"{s.upper()}_{k}"]
                  for k in ("SHOULDER", "ELBOW", "WRIST")} for s in sides}

    body = {}
    for side in sides:
        raw = [[], [], []]
        for t in range(T):
            f = pose_world[hold[t]]
            o, R = frames[t]
            raw[0].append(to_body_frame(f[lm_idx[side]["SHOULDER"], :3], o, R))
            raw[1].append(to_body_frame(f[lm_idx[side]["ELBOW"],    :3], o, R))
            raw[2].append(to_body_frame(f[lm_idx[side]["WRIST"],    :3], o, R))
        stacked = np.concatenate([np.array(raw[0]), np.array(raw[1]), np.array(raw[2])], axis=1)
        stacked = one_euro_sequence(stacked, fps, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA)
        body[side] = dict(S=stacked[:, :3], E=stacked[:, 3:6], W=stacked[:, 6:])

    calibration = {}
    for side in sides:
        calibration[side] = [
            (body[side]["S"][t], body[side]["E"][t], body[side]["W"][t])
            for t in range(T) if valid[t]][:10]
    common_scale = shared_arm_scale(calibration, OPENARM_ARM_LENGTH)

    retargeters = {}
    for side in sides:
        rt = Retargeter(robot_arm_length=OPENARM_ARM_LENGTH,
                        robot_anchor=OPENARM_ANCHOR,
                        body_to_robot=openarm_body_to_robot_for_side(side))
        rt.scale = common_scale
        retargeters[side] = rt
    print(f"  Shared bilateral arm scale: {common_scale:.3f}")

    # 手指捏合距离 D(t): Hand Landmarker 拇指尖-食指尖 world 距离
    # （BlazePose 自带手部点实测噪声过大不可用，见 extract_hand.py 文档）
    hands = extract_hands_from_video(video_path, pose_norm=data["pose"],
                                     delegate=delegate)
    hand_D = {s: pinch_distance(hands["hand_world"], s) for s in sides}
    hand_ratio = {
        s: normalized_pinch_distance(hands["hand_world"], s)[0] for s in sides
    }

    return (body, retargeters, valid, fps, T, hand_D, hand_ratio,
            arm_confidence, arm_trusted)


# ── IK pass ───────────────────────────────────────────────────────────────────

def _run_ik_pass(target_obs: np.ndarray, iks: dict, wrist_targets=None,
                 warm_starts_l=None, warm_starts_r=None,
                 label="", n_restarts=N_RESTARTS, n_restarts_14d=N_RESTARTS_14D,
                 max_nfev=MAX_NFEV, max_nfev_14d=MAX_NFEV_14D):
    """Run Phase-1 + Phase-2 IK on all rows of target_obs.

    warm_starts_l/r: (N,7) arrays of initial seeds (optional).
    Returns joint_angles (N,14) and costs (N,).
    """
    N = len(target_obs)
    joints_l, joints_r, costs = [], [], []
    d_before, d_after = [], []
    n_fail_l = n_fail_r = 0
    t0 = time.monotonic()

    prev_l = prev_r = None
    for i in range(N):
        row  = target_obs[i]
        pt_l = _reconstruct_pose_target(row[T_UPPER_L], row[T_FORE_L], row[T_PLANE_L])
        pt_r = _reconstruct_pose_target(row[T_UPPER_R], row[T_FORE_R], row[T_PLANE_R])
        if wrist_targets is not None:
            pt_l.wrist_position = np.asarray(wrist_targets[i, 0], dtype=float)
            pt_r.wrist_position = np.asarray(wrist_targets[i, 1], dtype=float)

        seed_l = prev_l if prev_l is not None else (
            warm_starts_l[i] if warm_starts_l is not None else None)
        seed_r = prev_r if prev_r is not None else (
            warm_starts_r[i] if warm_starts_r is not None else None)

        q_l, _ = solve_offline(iks["left"],  pt_l, first_seed=seed_l,
                                n_restarts=n_restarts, max_nfev=max_nfev)
        q_r, _ = solve_offline(iks["right"], pt_r, first_seed=seed_r,
                                n_restarts=n_restarts, max_nfev=max_nfev)

        if q_l is None:
            n_fail_l += 1
            q_l = iks["left"].home_seed.copy()
        if q_r is None:
            n_fail_r += 1
            q_r = iks["right"].home_seed.copy()

        d_before.append(_min_arm_dist(iks["left"], iks["right"], q_l, q_r))

        q_l, q_r, cost = solve_14d(iks["left"], iks["right"], pt_l, pt_r, q_l, q_r,
                                    reference_l=seed_l, reference_r=seed_r,
                                    n_restarts=n_restarts_14d, max_nfev=max_nfev_14d)
        prev_l, prev_r = q_l.copy(), q_r.copy()
        d_after.append(_min_arm_dist(iks["left"], iks["right"], q_l, q_r))

        joints_l.append(q_l)
        joints_r.append(q_r)
        costs.append(cost)

        if (i + 1) % 50 == 0 or i + 1 == N:
            elapsed = time.monotonic() - t0
            eta     = elapsed / (i + 1) * (N - i - 1)
            print(f"  {label}[{i+1}/{N}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                  flush=True)

    costs_arr  = np.array(costs)
    before_arr = np.array(d_before)
    after_arr  = np.array(d_after)
    if n_fail_l or n_fail_r:
        print(f"  WARNING: Phase-1 IK failed on {n_fail_l}L / {n_fail_r}R frames "
              f"— home_seed used as Phase-2 seed for those rows")
    print(f"  14D residual: mean={costs_arr.mean():.3f}  "
          f"p95={np.percentile(costs_arr,95):.3f}  max={costs_arr.max():.3f}")
    print(f"  Arm separation — before: {before_arr.mean()*100:.1f}cm  "
          f"after: {after_arr.mean()*100:.1f}cm  "
          f"violations: {(after_arr<_MIN_ARM_DIST).sum()}/{N}")

    jl = np.array(joints_l, dtype=np.float32)
    jr = np.array(joints_r, dtype=np.float32)
    return np.concatenate([jl, jr], axis=1), costs_arr


def _limit_joint_velocity(frames: np.ndarray, fps: float) -> np.ndarray:
    """Causal physical velocity limiter for a (T,14) trajectory."""
    out = np.asarray(frames, dtype=np.float32).copy()
    max_step = np.tile(JOINT_VELOCITY_LIMITS / float(fps), 2)
    clipped = 0
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        limited = np.clip(delta, -max_step, max_step)
        clipped += int(np.count_nonzero(np.abs(delta - limited) > 1e-7))
        out[i] = out[i - 1] + limited
    if clipped:
        print(f"  Velocity limiter adjusted {clipped}/{out.size} joint samples")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_dataset(video_path, out_path, delegate="cpu",
                     n_restarts=N_RESTARTS, n_restarts_14d=N_RESTARTS_14D,
                     max_nfev=MAX_NFEV, max_nfev_14d=MAX_NFEV_14D,
                     gripper_close=DEFAULT_CLOSE_M, gripper_open=DEFAULT_OPEN_M,
                     gripper_hold=DEFAULT_HOLD_FRAMES, gripper_lost=DEFAULT_LOST_FRAMES,
                     collision_margin=0.005, collision_policy="project",
                     local_avoidance=True, local_padding=10,
                     local_population=24, local_iterations=10,
                     local_seed=7, min_arm_visibility=0.55):
    print(f"[1/4] Extracting pose from {video_path} ...")
    (body, retargeters, valid, fps, T, hand_D, hand_ratio,
     arm_confidence, arm_trusted) = _extract_body_data(
         video_path, delegate, min_arm_visibility=min_arm_visibility)

    # ── Step 1: build target_obs for original frames ──────────────────────────
    print("[2/4] Retargeting + building target_obs ...")
    target_obs_list = []
    wrist_target_list = []
    good_frames     = []
    sides           = ("left", "right")

    for t in range(T):
        if not valid[t]:
            continue
        pts, ok = {}, True
        for side in sides:
            try:
                pts[side] = retargeters[side].frame_to_pose_target(
                    body[side]["S"][t], body[side]["E"][t], body[side]["W"][t])
            except ValueError:
                ok = False; break
        if not ok:
            continue
        wrists = coupled_wrist_targets(
            body["left"]["W"][t], body["right"]["W"][t],
            retargeters["left"].scale)
        pts["left"].wrist_position = wrists["left"]
        pts["right"].wrist_position = wrists["right"]
        target_obs_list.append(_build_target_obs(pts["left"], pts["right"]))
        wrist_target_list.append([wrists["left"], wrists["right"]])
        good_frames.append(t)

    N_orig = len(good_frames)
    print(f"  {N_orig} usable frames from {int(valid.sum())} detected")

    target_obs_orig   = np.array(target_obs_list, dtype=np.float32)  # (N, 18)
    target_obs_mirror = _mirror_target_obs(target_obs_orig)            # (N, 18)
    wrist_targets_orig = np.asarray(wrist_target_list, dtype=np.float32)
    wrist_targets_mirror = _mirror_wrist_targets(wrist_targets_orig)

    # Sanity: unit norms survive the mirror
    for name, sl in [("upper_L", T_UPPER_L), ("fore_L", T_FORE_L), ("plane_L", T_PLANE_L),
                     ("upper_R", T_UPPER_R), ("fore_R", T_FORE_R), ("plane_R", T_PLANE_R)]:
        norms = np.linalg.norm(target_obs_mirror[:, sl], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), \
            f"mirror broke unit norm for {name}: min={norms.min():.4f}"
    print("  Mirror unit-norm check passed")

    iks = {"left": OpenArmIK(side="left"), "right": OpenArmIK(side="right")}

    # ── Step 2: IK on original frames ─────────────────────────────────────────
    print(f"[3/4] IK on {N_orig} original frames ...")
    joint_angles_orig, _ = _run_ik_pass(
        target_obs_orig, iks, wrist_targets=wrist_targets_orig, label="orig ",
        n_restarts=n_restarts, n_restarts_14d=n_restarts_14d,
        max_nfev=max_nfev, max_nfev_14d=max_nfev_14d,
    )
    joint_angles_orig = _limit_joint_velocity(joint_angles_orig, fps)
    joint_angles_orig = offline_joint_trajectory(
        joint_angles_orig, window_length=9, polyorder=2)
    joint_angles_orig = _limit_joint_velocity(joint_angles_orig, fps)

    # ── Step 3: IK on mirrored frames (warm-start from mirrored orig solution) ─
    print(f"[4/4] IK on {N_orig} mirrored frames ...")
    # Approximate seed: swap left/right and flip sign of joint 4 (elbow axis)
    warm_l = joint_angles_orig[:, 7:] * _APPROX_SIGNS   # orig right → mirror left
    warm_r = joint_angles_orig[:, :7] * _APPROX_SIGNS   # orig left  → mirror right
    joint_angles_mirror, _ = _run_ik_pass(
        target_obs_mirror, iks, wrist_targets=wrist_targets_mirror,
        warm_starts_l=warm_l, warm_starts_r=warm_r,
        label="mirror ",
        n_restarts=n_restarts, n_restarts_14d=n_restarts_14d,
        max_nfev=max_nfev, max_nfev_14d=max_nfev_14d,
    )
    joint_angles_mirror = _limit_joint_velocity(joint_angles_mirror, fps)
    joint_angles_mirror = offline_joint_trajectory(
        joint_angles_mirror, window_length=9, polyorder=2)
    joint_angles_mirror = _limit_joint_velocity(joint_angles_mirror, fps)

    # Preserve the human-tracking reference before collision handling. Global
    # planners must not use a projected/held trajectory as their own goal or
    # they merely spread a stop over several slow frames.
    nominal_joint_angles = np.concatenate(
        [joint_angles_orig, joint_angles_mirror], axis=0).astype(np.float32)

    # ── Combine: [orig | mirror] — both temporally ordered ────────────────────
    target_obs_all   = np.concatenate([target_obs_orig,   target_obs_mirror],   axis=0)
    joint_angles_all = np.concatenate([joint_angles_orig, joint_angles_mirror], axis=0)
    frame_indices    = np.tile(np.array(good_frames), 2)

    # ── Gripper channel: 0/1 开合状态，列 [右臂, 左臂] ─────────────────────────
    # orig 片段: 机器人右臂 ← 人右手; mirror 片段左右手对调（与 target_obs 镜像一致）
    gripper_states = {s: map_states(hand_D[s], gripper_close, gripper_open,
                                    gripper_hold, gripper_lost)
                      for s in ("right", "left")}
    gripper_positions = {
        s: map_positions(hand_ratio[s], fps, DEFAULT_CLOSE_RATIO,
                         DEFAULT_OPEN_RATIO, DEFAULT_GRIPPER_MIN_CUTOFF,
                         DEFAULT_GRIPPER_BETA, gripper_lost)
        for s in ("right", "left")
    }
    gf = np.asarray(good_frames, dtype=int)
    gripper_cmd = np.zeros((2 * N_orig, 2), dtype=np.float32)
    gripper_cmd[:N_orig, 0] = gripper_states["right"][gf]
    gripper_cmd[:N_orig, 1] = gripper_states["left"][gf]
    gripper_cmd[N_orig:, 0] = gripper_states["left"][gf]
    gripper_cmd[N_orig:, 1] = gripper_states["right"][gf]
    gripper_position = np.zeros((2 * N_orig, 2), dtype=np.float32)
    gripper_position[:N_orig, 0] = gripper_positions["right"][gf]
    gripper_position[:N_orig, 1] = gripper_positions["left"][gf]
    gripper_position[N_orig:, 0] = gripper_positions["left"][gf]
    gripper_position[N_orig:, 1] = gripper_positions["right"][gf]

    # Hard safety projection must include the actual finger openings. Checking
    # only the 14 arm joints validates a different pose than replay displays.
    projector = InterArmCollisionProjector(margin=collision_margin)
    local_stats = np.zeros((2, 5), dtype=np.int32)
    if local_avoidance:
        optimizer = SlidingWindowTrajectoryOptimizer(
            projector, padding=local_padding, population=local_population,
            iterations=local_iterations, seed=local_seed)
        optimized_parts = []
        for clip_id, (name, joints, grip) in enumerate((
                ("orig", joint_angles_orig, gripper_position[:N_orig]),
                ("mirror", joint_angles_mirror, gripper_position[N_orig:]))):
            optimizer.rng = np.random.default_rng(local_seed + clip_id)
            optimized, stats = optimizer.optimize(joints, grip)
            optimized_parts.append(optimized)
            local_stats[clip_id] = [
                stats.unsafe_before, stats.unsafe_after,
                stats.improved_windows, len(stats.windows), stats.evaluations]
            print(f"  Local avoidance {name}: unsafe={stats.unsafe_before}"
                  f"->{stats.unsafe_after}, improved windows="
                  f"{stats.improved_windows}/{len(stats.windows)}")
        joint_angles_orig, joint_angles_mirror = optimized_parts
    locally_optimized_joint_angles = np.concatenate(
        [joint_angles_orig, joint_angles_mirror], axis=0).astype(np.float32)

    # The optimizer reduces intervention and creates a smooth detour. This
    # projector is still the non-negotiable postcondition for every frame and
    # every interpolated transition.
    joint_angles_orig, gripper_position[:N_orig], safety_orig = \
        projector.project_trajectory(
            joint_angles_orig, policy=collision_policy,
            gripper_positions=gripper_position[:N_orig])
    joint_angles_mirror, gripper_position[N_orig:], safety_mirror = \
        projector.project_trajectory(
            joint_angles_mirror, policy=collision_policy,
            gripper_positions=gripper_position[N_orig:])
    joint_angles_all = np.concatenate(
        [joint_angles_orig, joint_angles_mirror], axis=0)
    print(f"  Collision safety orig: unsafe={safety_orig.unsafe_candidates} "
          f"projected={safety_orig.projected_frames} held={safety_orig.held_frames}")
    print(f"  Collision safety mirror: unsafe={safety_mirror.unsafe_candidates} "
          f"projected={safety_mirror.projected_frames} held={safety_mirror.held_frames}")
    for side in ("right", "left"):
        s = gripper_states[side][gf]
        n_close  = int((s > 0.5).sum())
        n_switch = int((np.diff(s) != 0).sum())
        print(f"  gripper[{side}]: 闭合帧 {n_close}/{len(gf)}, 状态切换 {n_switch} 次")

    print(f"\nFinal dataset: {N_orig} → {len(target_obs_all)} frames "
          f"(original + mirror)")
    print(f"  target_obs:   {target_obs_all.shape}")
    print(f"  joint_angles: {joint_angles_all.shape}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             joint_order=np.array("left-first"),
             gripper_order=np.array("right-first"),
             gripper_semantics=np.array("normalized_closure_0_open_1_closed"),
             target_obs=target_obs_all,
             wrist_targets=np.concatenate(
                 [wrist_targets_orig, wrist_targets_mirror], axis=0),
             nominal_joint_angles=nominal_joint_angles,
             locally_optimized_joint_angles=locally_optimized_joint_angles,
             joint_angles=joint_angles_all,
             frame_indices=frame_indices,
             fps=fps,
             gripper_cmd=gripper_cmd,
             gripper_position=gripper_position,
             gripper_params=np.array([gripper_close, gripper_open,
                                      gripper_hold, gripper_lost], dtype=np.float32),
             gripper_position_params=np.array([
                 DEFAULT_CLOSE_RATIO, DEFAULT_OPEN_RATIO,
                 DEFAULT_GRIPPER_MIN_CUTOFF, DEFAULT_GRIPPER_BETA,
                 gripper_lost], dtype=np.float32),
             source_arm_confidence=arm_confidence,
             source_arm_trusted=arm_trusted,
             arm_visibility_threshold=np.float32(min_arm_visibility),
             offline_joint_smoothing_params=np.array([9, 2], dtype=np.int32),
             local_optimization_params=np.array([
                 collision_margin, local_padding, local_population,
                 local_iterations, local_seed, int(local_avoidance)],
                 dtype=np.float32),
             local_optimization_stats=local_stats,
             collision_safety_params=np.array([
                 collision_margin, projector.sweep_samples,
                 projector.bisection_steps], dtype=np.float32),
             collision_safety_stats=np.array([
                 [safety_orig.unsafe_candidates, safety_orig.projected_frames,
                  safety_orig.held_frames],
                 [safety_mirror.unsafe_candidates, safety_mirror.projected_frames,
                  safety_mirror.held_frames],
             ], dtype=np.int32))
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="videos/p2a_test.mp4")
    p.add_argument("--output",       default="test1.npz")
    p.add_argument("--delegate",     choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--restarts",     type=int, default=N_RESTARTS)
    p.add_argument("--restarts-14d", type=int, default=N_RESTARTS_14D)
    p.add_argument("--max-nfev",     type=int, default=MAX_NFEV)
    p.add_argument("--max-nfev-14d", type=int, default=MAX_NFEV_14D)
    p.add_argument("--gripper-close", type=float, default=DEFAULT_CLOSE_M,
                   help="夹爪闭合阈值: D 小于该值(米) → 闭合")
    p.add_argument("--gripper-open",  type=float, default=DEFAULT_OPEN_M,
                   help="夹爪张开阈值: D 大于该值(米) → 张开")
    p.add_argument("--gripper-hold",  type=int, default=DEFAULT_HOLD_FRAMES,
                   help="夹爪状态切换后最小保持帧数")
    p.add_argument("--gripper-lost",  type=int, default=DEFAULT_LOST_FRAMES,
                   help="手部连续不可测超过该帧数 → 回落张开")
    p.add_argument("--collision-margin", type=float, default=0.005,
                   help="左右臂碰撞几何硬安全余量，米（默认 0.005）")
    p.add_argument("--collision-policy", choices=["project", "hold", "error"],
                   default="project", help="碰撞候选处理方式（默认 project）")
    p.add_argument("--disable-local-avoidance", action="store_true",
                   help="关闭滑动窗口局部轨迹优化，仅使用硬安全投影")
    p.add_argument("--local-padding", type=int, default=10,
                   help="碰撞区间前后扩展帧数（默认 10）")
    p.add_argument("--local-population", type=int, default=24,
                   help="每轮局部优化的候选轨迹数（默认 24）")
    p.add_argument("--local-iterations", type=int, default=10,
                   help="每个碰撞窗口的优化轮数（默认 10）")
    p.add_argument("--local-seed", type=int, default=7,
                   help="局部优化随机种子，保证结果可复现")
    p.add_argument("--min-arm-visibility", type=float, default=0.55,
                   help="肩/肘/腕最低可信度；低于阈值时按单臂保持最近可信姿态")
    args = p.parse_args()
    generate_dataset(args.input, args.output, args.delegate,
                     args.restarts, args.restarts_14d,
                     args.max_nfev, args.max_nfev_14d,
                     args.gripper_close, args.gripper_open,
                     args.gripper_hold, args.gripper_lost,
                     args.collision_margin, args.collision_policy,
                     not args.disable_local_avoidance, args.local_padding,
                     args.local_population, args.local_iterations,
                     args.local_seed, args.min_arm_visibility)


if __name__ == "__main__":
    main()
