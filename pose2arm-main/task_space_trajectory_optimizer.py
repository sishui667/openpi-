"""Reference-locked task-space trajectory refinement for bimanual motion.

The global planner is allowed to choose a collision-free homotopy, but its raw
joint-space path is never executed directly.  This module pulls that seed back
toward the demonstrated motion while penalizing task-space error and jerk.
Exact MuJoCo edge checks remain the hard postcondition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from collision_safety import InterArmCollisionProjector
from ik_solver import OpenArmIK
from openarm_common import (
    LEFT_JOINT_LIMITS,
    OPENARM_BASE_OFFSETS,
    RIGHT_JOINT_LIMITS,
    T_FORE_L,
    T_FORE_R,
    T_PLANE_L,
    T_PLANE_R,
    T_UPPER_L,
    T_UPPER_R,
)


@dataclass(frozen=True)
class TrajectoryMetrics:
    wrist_mean_m: float
    direction_mean_deg: float
    direction_p95_deg: float
    jerk_p95: float
    max_joint_deviation: float


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(
        np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def _segment_distance(a0: np.ndarray, a1: np.ndarray,
                      b0: np.ndarray, b1: np.ndarray) -> float:
    """Shortest distance between finite 3-D line segments."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    dd, ee = np.dot(u, w), np.dot(v, w)
    denom = aa * cc - bb * bb
    small = 1e-12
    if denom < small:
        s = 0.0
        t = np.clip(ee / max(cc, small), 0.0, 1.0)
    else:
        s = np.clip((bb * ee - cc * dd) / denom, 0.0, 1.0)
        t = np.clip((aa * ee - bb * dd) / denom, 0.0, 1.0)
        # Recompute after clamping one parameter so endpoints are handled.
        s = np.clip((bb * t - dd) / max(aa, small), 0.0, 1.0)
        t = np.clip((bb * s + ee) / max(cc, small), 0.0, 1.0)
    return float(np.linalg.norm(w + s * u - t * v))


class ReferenceLockedTrajectoryOptimizer:
    """Optimize a small set of temporal control points around a safe seed."""

    def __init__(self, projector: InterArmCollisionProjector,
                 fps: float, control_points: int = 6,
                 max_iterations: int = 18,
                 capsule_clearance: float = 0.065):
        self.projector = projector
        self.fps = float(fps)
        self.control_points = max(3, int(control_points))
        self.max_iterations = max(1, int(max_iterations))
        self.capsule_clearance = float(capsule_clearance)
        self.iks = (OpenArmIK("left"), OpenArmIK("right"))
        self.low = np.concatenate(
            [LEFT_JOINT_LIMITS[:, 0], RIGHT_JOINT_LIMITS[:, 0]])
        self.high = np.concatenate(
            [LEFT_JOINT_LIMITS[:, 1], RIGHT_JOINT_LIMITS[:, 1]])
        self.ranges = self.high - self.low

    def _basis(self, count: int) -> np.ndarray:
        knots = min(self.control_points, max(3, count - 2))
        x = np.linspace(0.0, 1.0, count)
        centers = np.linspace(0.0, 1.0, knots + 2)[1:-1]
        spacing = centers[1] - centers[0] if len(centers) > 1 else 0.5
        basis = np.maximum(0.0, 1.0 - np.abs(
            x[:, None] - centers[None, :]) / spacing)
        basis[0] = 0.0
        basis[-1] = 0.0
        return basis

    def _task_residuals(self, q: np.ndarray, target_obs: np.ndarray,
                        wrist_targets: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        wrist_residuals = []
        direction_residuals = []
        slices = (
            (T_UPPER_L, T_FORE_L, T_PLANE_L),
            (T_UPPER_R, T_FORE_R, T_PLANE_R),
        )
        for frame, row in enumerate(q):
            for side, (ik, arm_q, arm_slices) in enumerate(zip(
                    self.iks, (row[:7], row[7:]), slices)):
                points = ik.forward_points(arm_q)
                wrist_residuals.extend(
                    (points["tool"] - wrist_targets[frame, side]) / 0.05)
                for key, target_slice in zip(
                        ("upper_dir", "forearm_dir", "arm_plane"), arm_slices):
                    direction_residuals.extend(
                        _unit(points[key]) - _unit(target_obs[frame, target_slice]))
        return (np.asarray(wrist_residuals, dtype=float),
                np.asarray(direction_residuals, dtype=float))

    def _clearance_residuals(self, q: np.ndarray) -> np.ndarray:
        residuals = []
        for row in q:
            points = []
            for side, ik, arm_q in zip(
                    ("left", "right"), self.iks, (row[:7], row[7:])):
                fk = ik.forward_points(arm_q)
                offset = OPENARM_BASE_OFFSETS[side]
                shoulder = offset
                elbow = fk["elbow"] + offset
                tool = fk["tool"] + offset
                # Ignore the proximal shoulder section where the two bases are
                # intentionally close; model the exposed upper/forearm links.
                upper_start = shoulder + 0.30 * (elbow - shoulder)
                points.append(((upper_start, elbow), (elbow, tool), tool))
            left, right = points
            distances = [
                _segment_distance(*left[0], *right[0]),
                _segment_distance(*left[0], *right[1]),
                _segment_distance(*left[1], *right[0]),
                _segment_distance(*left[1], *right[1]),
                float(np.linalg.norm(left[2] - right[2])),
            ]
            residuals.extend(max(0.0, self.capsule_clearance - distance)
                             / self.capsule_clearance for distance in distances)
        return np.asarray(residuals, dtype=float)

    def _objective(self, q: np.ndarray, reference: np.ndarray,
                   target_obs: np.ndarray, wrist_targets: np.ndarray) -> float:
        wrist, direction = self._task_residuals(q, target_obs, wrist_targets)
        joint = (q - reference) / self.ranges
        velocity = np.diff(q, axis=0) / self.ranges
        acceleration = np.diff(q, n=2, axis=0) / self.ranges
        jerk = np.diff(q, n=3, axis=0) / self.ranges
        clearance = self._clearance_residuals(q)
        boundary = 0.0
        if len(q) >= 3:
            boundary = (
                np.mean(((q[1] - q[0]) - (reference[1] - reference[0])) ** 2)
                + np.mean(((q[-1] - q[-2])
                           - (reference[-1] - reference[-2])) ** 2))
        return float(
            2.5 * np.mean(wrist ** 2)
            + 3.0 * np.mean(direction ** 2)
            + 1.5 * np.mean(joint ** 2)
            + 0.15 * np.mean(velocity ** 2)
            + (3.0 * np.mean(acceleration ** 2) if len(acceleration) else 0.0)
            + (20.0 * np.mean(jerk ** 2) if len(jerk) else 0.0)
            + 2.0 * boundary
            + (120.0 * np.mean(clearance ** 2) if len(clearance) else 0.0)
        )

    def _edges_safe(self, q: np.ndarray, grip: np.ndarray) -> bool:
        return all(self.projector.is_safe(row, opening)
                   for row, opening in zip(q, grip)) and all(
            self._edge_safe(q[i - 1], q[i], grip[i - 1:i + 1])
            for i in range(1, len(q)))

    def _edge_safe(self, start: np.ndarray, stop: np.ndarray,
                   grips: np.ndarray) -> bool:
        delta = np.max(np.abs(stop - start))
        steps = max(1, int(np.ceil(delta / 0.03)))
        for step in range(1, steps + 1):
            alpha = step / steps
            q = start + alpha * (stop - start)
            grip = grips[0] + alpha * (grips[-1] - grips[0])
            if not self.projector.is_safe(q, grip):
                return False
        return True

    def refine(self, seed: np.ndarray, reference: np.ndarray,
               grip: np.ndarray, target_obs: np.ndarray,
               wrist_targets: np.ndarray) -> np.ndarray:
        seed = np.asarray(seed, dtype=float)
        reference = np.asarray(reference, dtype=float)
        grip = np.asarray(grip, dtype=float)
        target_obs = np.asarray(target_obs, dtype=float)
        wrist_targets = np.asarray(wrist_targets, dtype=float)
        if seed.shape != reference.shape or seed.ndim != 2 or seed.shape[1] != 14:
            raise ValueError("seed and reference must have shape (T,14)")
        if not self._edges_safe(seed, grip):
            raise ValueError("trajectory optimizer requires a collision-free seed")

        basis = self._basis(len(seed))
        knots = basis.shape[1]

        def decode(flat: np.ndarray) -> np.ndarray:
            offsets = basis @ flat.reshape(knots, 14)
            return np.clip(seed + offsets, self.low, self.high)

        def objective(flat: np.ndarray) -> float:
            return self._objective(
                decode(flat), reference, target_obs, wrist_targets)

        # Keep the trust region small. RRT selects topology; optimization is
        # only allowed to improve fidelity within that corridor.
        bounds = [(-0.45, 0.45)] * (knots * 14)
        result = minimize(
            objective, np.zeros(knots * 14, dtype=float), method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iterations, "ftol": 1e-7,
                     "maxls": 12},
        )
        optimized = decode(result.x)

        # Exact-mesh feasibility is non-negotiable. If the approximate
        # clearance model crossed a mesh boundary, backtrack from the safe
        # seed and keep the best exact-safe improvement.
        candidates = [seed]
        for alpha in (1.0, 0.75, 0.50, 0.25):
            candidate = seed + alpha * (optimized - seed)
            if self._edges_safe(candidate, grip):
                candidates.append(candidate)
        best = min(candidates, key=lambda q: self._objective(
            q, reference, target_obs, wrist_targets))
        return np.asarray(best, dtype=np.float32)

    def metrics(self, q: np.ndarray, reference: np.ndarray,
                target_obs: np.ndarray,
                wrist_targets: np.ndarray) -> TrajectoryMetrics:
        q = np.asarray(q, dtype=float)
        wrist_errors = []
        direction_errors = []
        slices = (
            (T_UPPER_L, T_FORE_L, T_PLANE_L),
            (T_UPPER_R, T_FORE_R, T_PLANE_R),
        )
        for frame, row in enumerate(q):
            for side, (ik, arm_q, arm_slices) in enumerate(zip(
                    self.iks, (row[:7], row[7:]), slices)):
                points = ik.forward_points(arm_q)
                wrist_errors.append(float(np.linalg.norm(
                    points["tool"] - wrist_targets[frame, side])))
                for key, target_slice in zip(
                        ("upper_dir", "forearm_dir", "arm_plane"), arm_slices):
                    direction_errors.append(_angle_deg(
                        points[key], target_obs[frame, target_slice]))
        jerk = np.linalg.norm(np.diff(q, n=3, axis=0) * self.fps ** 3, axis=1)
        return TrajectoryMetrics(
            wrist_mean_m=float(np.mean(wrist_errors)),
            direction_mean_deg=float(np.mean(direction_errors)),
            direction_p95_deg=float(np.percentile(direction_errors, 95)),
            jerk_p95=float(np.percentile(jerk, 95)) if len(jerk) else 0.0,
            max_joint_deviation=float(np.max(np.linalg.norm(
                q - np.asarray(reference, dtype=float), axis=1))),
        )
