"""Geometric retargeting: human shoulder/elbow/wrist keypoints -> robot end-effector target.

Two stages, deliberately kept separate:

1. `to_body_frame`: re-express raw keypoints in a frame attached to the torso
   (origin at mid-shoulder, axes from the shoulder line + spine). Without
   this, a person walking sideways in frame is indistinguishable from the
   person swinging their arm. Built from `pose_world` (MediaPipe's metric,
   GHUM-trained world landmarks), not the normalized image-space `pose`
   field -- world landmarks carry a population-level anatomical-proportion
   prior from training, which empirically cuts frame-to-frame jitter by
   ~80% before any temporal filtering is even applied, with zero added
   latency. `stabilize_root` applies a light residual filter on top of that
   already-much-cleaner signal. See FAILURE_MODES.md for the full
   investigation, including two heavier approaches (aggressive temporal
   filtering, Kabsch/Procrustes rigid-template fitting) that were tried and
   superseded by this once the real fix -- the input data source -- was found.

2. `compute_arm_frame` / `Retargeter`: turn the body-frame shoulder/elbow/wrist
   triplet into a 6-DOF end-effector target (position + orientation) using
   vector geometry, not a raw coordinate copy. The forearm direction becomes
   the tool's pointing axis; the arm-plane normal (shoulder x forearm) becomes
   a second axis; Gram-Schmidt completes an orthonormal frame. Elbow flexion
   angle is computed too and reported as a diagnostic, but it does not directly
   drive a robot joint: a 7-DOF arm's own elbow configuration is resolved by
   the IK solver's nullspace, not pinned 1:1 from 3 human keypoints. Matching
   the human's elbow configuration exactly would need a full anatomical
   retargeting scheme (e.g. SEW-Mimic-style swing/twist decomposition) -- noted
   as a documented simplification, not implemented here.
"""
from dataclasses import dataclass

import numpy as np

from pose_constants import LANDMARK
from ik_solver import OpenArmIK
from openarm_common import (OPENARM_ARM_LENGTH, OPENARM_ANCHOR,
                            OPENARM_BASE_OFFSETS)  # noqa: F401 (re-exported)
from smoothing import one_euro_sequence

# Light residual filter, not the primary jitter fix (see module docstring).
# Both are fixed constants, deliberately NOT calibrated per-video: an earlier
# version calibrated min_cutoff from each video's own first ~30 frames
# (mirroring Retargeter.calibrate's calibrate-once pattern), but that
# assumes a stable "rest" baseline exists in a short prefix -- on two real
# clips with normal scene content (a person posing, then moving dynamically)
# the calibrated value swung by >10x depending on which frames happened to
# be included (see FAILURE_MODES.md), because how much smoothing a clip
# "needs" depends on what the person is doing in that window, not on a
# fixed noise property the way arm length or shoulder width is. MediaPipe's
# own legacy smoothing calculator (RelativeVelocityFilter) makes the same
# choice -- fixed window_size/velocity_scale constants, no per-video
# calibration -- and relies on beta's frame-by-frame adaptation (which
# doesn't need calibration; it reacts to the immediately preceding sample)
# to handle within-clip motion changes. 0.3/3.0 were tuned with the full
# two-step One Euro method (Casiez et al. 2012: pick min_cutoff for
# rest-jitter, then beta for fast-motion lag, not jitter alone which is
# blind to the lag side) and cross-checked on representative calm/fast
# segments from two different real clips -- not calibrated on either,
# just validated against both.
ROOT_MIN_CUTOFF = 0.3
ROOT_BETA = 3.0

# Same algorithm (One Euro Filter) as the root filter above, deliberately
# different constants: the root filter's two objectives (rest-jitter, fast-
# motion lag) pointed the same direction, but for the wrist/elbow signal
# jerk-reduction and lag-reduction pull in *opposite* directions -- jerk is
# a 3rd derivative, hypersensitive to any leftover high-frequency content
# (real fast motion included), so minimizing it wants heavy smoothing
# (beta=0, low cutoff) while minimizing lag wants the opposite. On a real
# clip's wrist trajectory, min_cutoff=0.3/beta=0.0 got 95.6% jerk reduction
# at the cost of the worst tracking lag in the grid tested; min_cutoff=8.0/
# beta=10.0 got the best lag at only 27% jerk reduction. Chose to sit at the
# jerk-favoring end deliberately: the deliverable is what the robot arm
# looks like, not a faithful reproduction of the performer's exact motion,
# The original 0.3/0.0 setting suppressed jitter but behaved like a very slow
# fixed low-pass filter. 1.0/4.0 retains useful rest smoothing while opening
# the cutoff during deliberate motion, which removes most of the visible
# trailing without bypassing the filter.
ARM_MIN_CUTOFF = 1.0
ARM_BETA = 4.0

# A straight arm makes the shoulder/forearm vectors near-parallel, so their
# cross product (used to build the arm-plane normal) vanishes -- a real
# geometric singularity, not a bug. Below this norm we reuse the last valid
# normal instead of dividing by ~0.
_DEGENERATE_CROSS_NORM = 1e-4

# Body frame columns from compute_body_frame:
#   +body_x: human left shoulder -> right shoulder
#   +body_y: hip midpoint -> shoulder midpoint
#   +body_z: derived chest-normal axis from cross(body_x, body_y)
#
# Gazebo/OpenArm display convention:
#   +robot_x: robot face/front
#   -robot_y: robot's right side, +robot_y: robot's left side
#   +robot_z: up
#
# MediaPipe pose_world chest depth sign makes human forward motion appear along
# -body_z in our torso frame. Therefore:
#   human forward  (-body_z) -> +robot_x
#   human right    (+body_x) -> -robot_y
#   human left     (-body_x) -> +robot_y
#   human up       (+body_y) -> +robot_z
OPENARM_BODY_TO_ROBOT = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def openarm_body_to_robot_for_side(side):
    body_to_robot = np.array(OPENARM_BODY_TO_ROBOT, dtype=float)
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError(f"unknown OpenArm side {side!r}")
    return body_to_robot


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def shared_arm_scale(frames_by_side, robot_arm_length=OPENARM_ARM_LENGTH):
    """One robust human-to-robot scale shared by both arms.

    Separate left/right gains turn landmark-estimation asymmetry into a false
    change in hand separation. Pooling both sides preserves bilateral geometry.
    """
    lengths = []
    for side in ("left", "right"):
        for shoulder, elbow, wrist in frames_by_side[side]:
            length = (np.linalg.norm(np.asarray(elbow) - shoulder)
                      + np.linalg.norm(np.asarray(wrist) - elbow))
            if np.isfinite(length) and length > 1e-6:
                lengths.append(float(length))
    if not lengths:
        raise ValueError("cannot calibrate shared scale from empty/invalid arms")
    return float(robot_arm_length) / float(np.median(lengths))


def _soft_reach_limit(target, max_reach, soft_ratio=0.88):
    """Radially compress the outer workspace without a hard target jump."""
    target = np.asarray(target, dtype=float)
    radius = float(np.linalg.norm(target))
    soft = float(max_reach) * float(soft_ratio)
    if radius <= soft or radius < 1e-12:
        return target
    span = max(float(max_reach) - soft, 1e-9)
    limited_radius = soft + span * np.tanh((radius - soft) / span)
    return target * (limited_radius / radius)


def coupled_wrist_targets(left_wrist_body, right_wrist_body, scale,
                           robot_arm_length=OPENARM_ARM_LENGTH,
                           max_reach_ratio=0.95):
    """Map both wrists through one torso-centred similarity transform.

    Returned positions are local to each robot shoulder, matching OpenArmIK's
    forward kinematics. Before reach compression, adding the two robot base
    offsets reconstructs a pair of world targets whose separation is exactly
    the scaled human inter-wrist separation.
    """
    max_reach = float(robot_arm_length) * float(max_reach_ratio)
    result = {}
    for side, wrist in (("left", left_wrist_body),
                        ("right", right_wrist_body)):
        world_target = float(scale) * (OPENARM_BODY_TO_ROBOT @ np.asarray(wrist, dtype=float))
        local_target = world_target - OPENARM_BASE_OFFSETS[side]
        result[side] = _soft_reach_limit(local_target, max_reach)
    return result


@dataclass
class PoseTarget:
    upper_dir: np.ndarray
    forearm_dir: np.ndarray
    wrist_dir: np.ndarray
    arm_plane: np.ndarray
    elbow_angle: float
    wrist_position: np.ndarray
    elbow_position: np.ndarray
    orientation: np.ndarray


def compute_body_frame(left_shoulder, right_shoulder, left_hip, right_hip):
    """Returns (origin, R) for the torso frame: R's columns are (x, y, z) in
    world/camera coordinates, x = shoulder line, y = spine (hip->shoulder),
    z = chest-forward. Re-orthogonalized via Gram-Schmidt so R is a valid
    rotation matrix even though the four landmarks are noisy and not
    perfectly planar."""
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0

    y_axis = _normalize(mid_shoulder - mid_hip)
    x_raw = _normalize(right_shoulder - left_shoulder)
    z_axis = _normalize(np.cross(x_raw, y_axis))
    x_axis = np.cross(y_axis, z_axis)  # re-orthogonalize against y, z

    R = np.column_stack([x_axis, y_axis, z_axis])
    return mid_shoulder, R


def to_body_frame(point, origin, R):
    """World/camera-space point -> torso-local coordinates."""
    return R.T @ (point - origin)


def frame_angular_jitter_deg(R_sequence):
    """RMS frame-to-frame rotation angle (degrees) of a sequence of body-frame
    rotation matrices -- quantifies how much the derived frame itself wobbles,
    independent of any actual arm motion. Diagnostic only, not used by the
    retargeting math itself; see FAILURE_MODES.md for context."""
    deltas = []
    for t in range(1, len(R_sequence)):
        relative = R_sequence[t - 1].T @ R_sequence[t]
        cos_angle = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
        deltas.append(np.degrees(np.arccos(cos_angle)))
    return float(np.sqrt(np.mean(np.square(deltas)))) if deltas else 0.0


def stabilize_root(root_points, freq, min_cutoff=ROOT_MIN_CUTOFF, beta=ROOT_BETA):
    """root_points: (T, 4, 3) array of (L_SH, R_SH, L_HIP, R_HIP) world-landmark
    positions per frame. Light residual One Euro Filter -- see the module
    docstring for why this doesn't need to be heavy, and why min_cutoff/beta
    are fixed constants rather than calibrated per-video."""
    root_points = np.asarray(root_points, dtype=float)
    T = len(root_points)
    flat = one_euro_sequence(root_points.reshape(T, 12), freq, min_cutoff=min_cutoff, beta=beta)
    return flat.reshape(T, 4, 3)


def compute_frame_sequence(raw_root, freq):
    """Returns a list of (origin, R) torso frames, one per input frame:
    stabilize_root (light residual filtering) then the ordinary 2-vector
    Gram-Schmidt compute_body_frame, now operating on already-clean
    world-landmark input."""
    stable_root = stabilize_root(raw_root, freq)
    return [compute_body_frame(*stable_root[t]) for t in range(len(stable_root))]


def compute_arm_frame(shoulder, elbow, wrist, prev_normal=None):
    """shoulder/elbow/wrist in body-local coordinates.

    Returns (R, elbow_angle_rad, wrist_relative_to_shoulder, normal_used).
    R's z-axis points along the forearm (elbow->wrist); pass this as the IK
    target orientation. elbow_angle is the interior angle at the elbow vertex
    (pi = fully extended, 0 = fully folded) -- a diagnostic, not an IK input.
    """
    upper_arm = elbow - shoulder
    forearm = wrist - elbow

    elbow_angle = np.arccos(
        np.clip(np.dot(-upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm)), -1.0, 1.0)
    )

    cross = np.cross(upper_arm, forearm)
    if np.linalg.norm(cross) < _DEGENERATE_CROSS_NORM:
        if prev_normal is None:
            raise ValueError(
                "arm is too straight to define a plane and no previous frame to fall back on "
                "(see FAILURE_MODES.md: straight-arm orientation singularity)"
            )
        normal = prev_normal
    else:
        normal = _normalize(cross)

    z_axis = _normalize(forearm)
    x_axis = _normalize(np.cross(normal, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    wrist_relative = wrist - shoulder
    return R, elbow_angle, wrist_relative, normal


class Retargeter:
    """Converts a body-frame (shoulder, elbow, wrist) sequence into robot
    end-effector targets, auto-calibrating the human-arm -> robot-arm scale
    from the first valid frames instead of hardcoding a magic number."""

    def __init__(self, robot_arm_length=0.455, robot_anchor=None, body_to_robot=None):
        self.robot_arm_length = robot_arm_length
        self.robot_anchor = np.zeros(3, dtype=float) if robot_anchor is None else np.asarray(robot_anchor, dtype=float)
        self.body_to_robot = OPENARM_BODY_TO_ROBOT if body_to_robot is None else np.asarray(body_to_robot, dtype=float)
        self.scale = None
        self._prev_normal = None

    def calibrate(self, shoulder_elbow_wrist_frames):
        """shoulder_elbow_wrist_frames: iterable of (S, E, W) body-frame triplets."""
        lengths = []
        for S, E, W in shoulder_elbow_wrist_frames:
            lengths.append(np.linalg.norm(E - S) + np.linalg.norm(W - E))
        human_arm_length = float(np.median(lengths))
        if human_arm_length < 1e-6:
            raise ValueError("calibration frames have ~zero arm length -- check keypoint extraction")
        self.scale = self.robot_arm_length / human_arm_length
        return self.scale

    def frame_to_target(self, shoulder, elbow, wrist):
        if self.scale is None:
            raise RuntimeError("call calibrate() with a few representative frames first")

        R, elbow_angle, wrist_relative, normal = compute_arm_frame(shoulder, elbow, wrist, self._prev_normal)
        self._prev_normal = normal

        target_position = self.robot_anchor + self.scale * (self.body_to_robot @ wrist_relative)
        target_R = self.body_to_robot @ R
        return target_position, target_R, elbow_angle

    def frame_to_pose_target(self, shoulder, elbow, wrist):
        if self.scale is None:
            raise RuntimeError("call calibrate() with a few representative frames first")

        R, elbow_angle, wrist_relative, normal = compute_arm_frame(shoulder, elbow, wrist, self._prev_normal)
        self._prev_normal = normal

        upper = elbow - shoulder
        forearm = wrist - elbow
        wrist_dir = wrist - shoulder
        target_position = self.robot_anchor + self.scale * (self.body_to_robot @ wrist_relative)
        elbow_position = self.robot_anchor + self.scale * (self.body_to_robot @ upper)
        return PoseTarget(
            upper_dir=self.body_to_robot @ _normalize(upper),
            forearm_dir=self.body_to_robot @ _normalize(forearm),
            wrist_dir=self.body_to_robot @ _normalize(wrist_dir),
            arm_plane=self.body_to_robot @ _normalize(normal),
            elbow_angle=elbow_angle,
            wrist_position=target_position,
            elbow_position=elbow_position,
            orientation=self.body_to_robot @ R,
        )


def _fill_holds(present):
    """Indices to use per-frame after a "hold last good value" fallback for
    low-confidence/missing-detection frames (documented simplification --
    see FAILURE_MODES.md). Frames before the first valid detection just
    reuse the first valid frame."""
    idx = np.full(len(present), -1, dtype=int)
    last_good = -1
    for t in range(len(present)):
        if present[t]:
            last_good = t
        idx[t] = last_good
    return idx


def run_pipeline(keypoints_npz, smoothing=True, side="right", smoothing_stage="keypoints",
                  robot_arm_length=OPENARM_ARM_LENGTH, robot_anchor=OPENARM_ANCHOR,
                  freq=None, verbose=True):
    """Runs body-frame conversion -> (optional keypoint smoothing) -> geometric
    retargeting -> IK -> (optional joint-angle smoothing) over an entire
    extract_pose.py keypoint sequence. Returns a dict ready to np.savez:
    joint_angles (T,7), ee_positions (T,3), target_positions (T,3), fps.

    Reads from the "pose_world" field (metric, GHUM-trained world landmarks),
    not "pose" (normalized image coordinates) -- see module docstring."""
    body_to_robot = openarm_body_to_robot_for_side(side)

    data = np.load(keypoints_npz)
    pose, present, fps = data["pose_world"], data["pose_present"], float(data["fps"])
    if freq is None:
        freq = fps
    hold_idx = _fill_holds(present)
    valid_mask = hold_idx >= 0
    if not valid_mask.any():
        raise ValueError("no frame in this video ever had a detected pose")
    hold_idx[~valid_mask] = hold_idx[valid_mask][0]  # frames before first detection borrow it

    sh_i = LANDMARK[f"{side.upper()}_SHOULDER"]
    el_i = LANDMARK[f"{side.upper()}_ELBOW"]
    wr_i = LANDMARK[f"{side.upper()}_WRIST"]

    root_indices = [LANDMARK["LEFT_SHOULDER"], LANDMARK["RIGHT_SHOULDER"], LANDMARK["LEFT_HIP"], LANDMARK["RIGHT_HIP"]]
    raw_root = pose[hold_idx][:, root_indices, :3]
    frame_sequence = compute_frame_sequence(raw_root, freq)

    body_S, body_E, body_W, R_sequence = [], [], [], []
    for t in range(len(pose)):
        frame = pose[hold_idx[t]]
        origin, R_body = frame_sequence[t]
        R_sequence.append(R_body)
        body_S.append(to_body_frame(frame[sh_i, :3], origin, R_body))
        body_E.append(to_body_frame(frame[el_i, :3], origin, R_body))
        body_W.append(to_body_frame(frame[wr_i, :3], origin, R_body))
    body_S, body_E, body_W = np.array(body_S), np.array(body_E), np.array(body_W)

    if verbose:
        jitter = frame_angular_jitter_deg(R_sequence)
        print(f"body-frame RMS frame-to-frame jitter = {jitter:.2f} deg/frame")

    if smoothing_stage == "keypoints" and smoothing:
        stacked = np.concatenate([body_S, body_E, body_W], axis=1)  # (T, 9)
        stacked = one_euro_sequence(stacked, freq, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA)
        body_S, body_E, body_W = stacked[:, 0:3], stacked[:, 3:6], stacked[:, 6:9]

    rt = Retargeter(
        robot_arm_length=robot_arm_length,
        robot_anchor=robot_anchor,
        body_to_robot=body_to_robot,
    )
    n_calib = max(1, min(10, int(valid_mask.sum())))
    calib_frames = [(body_S[t], body_E[t], body_W[t]) for t in range(len(pose)) if valid_mask[t]][:n_calib]
    rt.calibrate(calib_frames)

    ik = OpenArmIK(side=side)
    targets, joint_angles, ee_positions = [], [], []
    pose_metrics = []
    for t in range(len(pose)):
        pose_target = rt.frame_to_pose_target(body_S[t], body_E[t], body_W[t])
        target_pos = pose_target.wrist_position
        angles = ik.solve_pose_likeness(pose_target)
        pose_metrics.append(_pose_likeness_metrics(ik.forward_points(angles), pose_target))
        ee_pos, _ = ik.forward_kinematics(angles)
        targets.append(target_pos)
        joint_angles.append(angles)
        ee_positions.append(ee_pos)
    joint_angles, targets, ee_positions = np.array(joint_angles), np.array(targets), np.array(ee_positions)

    if smoothing_stage == "joint_angles" and smoothing:
        joint_angles = one_euro_sequence(joint_angles, freq, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA)
        ee_positions = np.array([ik.forward_kinematics(q)[0] for q in joint_angles])

    out = {
        "joint_angles": joint_angles,
        "ee_positions": ee_positions,
        "target_positions": targets,
        "fps": fps,
        "robot": "openarm",
        "ik_mode": "pose_likeness",
    }
    if pose_metrics:
        for key in pose_metrics[0]:
            out[key] = np.array([m[key] for m in pose_metrics], dtype=float)
    return out


def _cosine_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    return float(0.5 - 0.5 * np.dot(a, b) / (an * bn))


def _pose_likeness_metrics(points, pose_target):
    return {
        "upper_dir_error": _cosine_error(points["upper_dir"], pose_target.upper_dir),
        "forearm_dir_error": _cosine_error(points["forearm_dir"], pose_target.forearm_dir),
        "wrist_dir_error": _cosine_error(points["wrist_dir"], pose_target.wrist_dir),
        "arm_plane_error": _cosine_error(points["arm_plane"], pose_target.arm_plane),
        "elbow_angle_error": abs(points["elbow_angle"] - pose_target.elbow_angle),
    }
