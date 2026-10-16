"""Collision-aware sliding-window trajectory optimization for Pose2Arm.

This is a small STOMP/CEM-style derivative-free optimizer. It perturbs only
windows around detected inter-arm collisions, scores complete short
trajectories, and keeps the original motion everywhere else. The hard
``InterArmCollisionProjector`` remains the postcondition-enforcing fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collision_safety import InterArmCollisionProjector
from openarm_common import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS


@dataclass
class LocalOptimizationStats:
    unsafe_before: int = 0
    unsafe_after: int = 0
    windows: list[tuple[int, int]] = field(default_factory=list)
    improved_windows: int = 0
    evaluations: int = 0


class SlidingWindowTrajectoryOptimizer:
    """Stochastic local optimizer over rare collision intervals."""

    def __init__(self, projector: InterArmCollisionProjector,
                 padding: int = 10, population: int = 24,
                 iterations: int = 10, elite_count: int = 6,
                 seed: int = 7):
        self.projector = projector
        self.padding = max(2, int(padding))
        self.population = max(4, int(population))
        self.iterations = max(1, int(iterations))
        self.elite_count = min(max(2, int(elite_count)), self.population)
        self.rng = np.random.default_rng(seed)
        self.low = np.concatenate([LEFT_JOINT_LIMITS[:, 0], RIGHT_JOINT_LIMITS[:, 0]])
        self.high = np.concatenate([LEFT_JOINT_LIMITS[:, 1], RIGHT_JOINT_LIMITS[:, 1]])
        self.ranges = self.high - self.low

    @staticmethod
    def _windows(mask: np.ndarray, padding: int) -> list[tuple[int, int]]:
        ids = np.flatnonzero(mask)
        if not len(ids):
            return []
        raw = []
        start = prev = int(ids[0])
        for idx in ids[1:]:
            idx = int(idx)
            if idx > prev + 1:
                raw.append((start, prev + 1))
                start = idx
            prev = idx
        raw.append((start, prev + 1))

        length = len(mask)
        expanded = [(max(0, a - padding), min(length, b + padding)) for a, b in raw]
        merged = []
        for a, b in expanded:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    @staticmethod
    def _smooth_noise(noise: np.ndarray) -> np.ndarray:
        # Repeated low-pass filtering keeps samples trajectory-like instead of
        # producing independent frame jitter. Reflection avoids edge spikes.
        kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
        out = noise.copy()
        for _ in range(3):
            filtered = np.empty_like(out)
            for j in range(out.shape[1]):
                padded = np.pad(out[:, j], (2, 2), mode="reflect")
                filtered[:, j] = np.convolve(padded, kernel, mode="valid")
            out = filtered
        return out

    def _score(self, candidate: np.ndarray, target: np.ndarray,
               grip: np.ndarray) -> float:
        tracking = np.mean(((candidate - target) / self.ranges) ** 2)
        velocity = np.diff(candidate, axis=0) / self.ranges
        accel = np.diff(candidate, n=2, axis=0) / self.ranges
        smoothness = (0.50 * np.mean(velocity ** 2)
                      + (4.00 * np.mean(accel ** 2) if len(accel) else 0.0))
        collision = 0.0
        for i, q in enumerate(candidate):
            collision += self.projector.collision_penalty(q, grip[i])
            if i:
                collision += self.projector.collision_penalty(
                    0.5 * (candidate[i - 1] + q),
                    0.5 * (grip[i - 1] + grip[i]))
        collision /= max(1, 2 * len(candidate) - 1)
        return float(tracking + smoothness + 80.0 * collision)

    def _optimize_window(self, target: np.ndarray, grip: np.ndarray,
                         stats: LocalOptimizationStats,
                         initial: np.ndarray | None = None) -> np.ndarray:
        mean = target.copy() if initial is None else np.asarray(initial, dtype=float).copy()
        fixed_start = mean[0].copy()
        fixed_end = mean[-1].copy()
        best = mean.copy()
        best_score = self._score(best, target, grip)
        stats.evaluations += 1
        sigma = np.tile(np.array([
            0.18, 0.18, 0.13, 0.13, 0.10, 0.10, 0.10,
            0.18, 0.18, 0.13, 0.13, 0.10, 0.10, 0.10,
        ]), (len(target), 1))

        anchor_start = self.projector.is_safe(fixed_start, grip[0])
        anchor_end = self.projector.is_safe(fixed_end, grip[-1])
        for _ in range(self.iterations):
            candidates, scores = [], []
            for sample_id in range(self.population):
                if sample_id == 0:
                    candidate = mean.copy()
                else:
                    noise = self._smooth_noise(self.rng.normal(size=mean.shape) * sigma)
                    candidate = np.clip(mean + noise, self.low, self.high)
                    if anchor_start:
                        candidate[0] = fixed_start
                    if anchor_end:
                        candidate[-1] = fixed_end
                score = self._score(candidate, target, grip)
                stats.evaluations += 1
                candidates.append(candidate)
                scores.append(score)
                if score < best_score:
                    best_score, best = score, candidate.copy()
            elite_ids = np.argsort(scores)[:self.elite_count]
            elites = np.asarray([candidates[i] for i in elite_ids])
            mean = 0.35 * mean + 0.65 * elites.mean(axis=0)
            sigma = np.maximum(0.015, 0.85 * sigma + 0.15 * elites.std(axis=0))
        return best

    def refine(self, initial: np.ndarray, target: np.ndarray,
               gripper_positions: np.ndarray) -> np.ndarray:
        """Refine a feasible global-planner seed toward the nominal motion."""
        initial = np.asarray(initial, dtype=float)
        target = np.asarray(target, dtype=float)
        grip = np.asarray(gripper_positions, dtype=float)
        if initial.shape != target.shape or initial.ndim != 2 or initial.shape[1] != 14:
            raise ValueError("initial and target must both have shape (T,14)")
        if grip.shape != (len(initial), 2):
            raise ValueError("gripper_positions must have shape (T,2)")
        stats = LocalOptimizationStats()
        return self._optimize_window(target, grip, stats, initial=initial).astype(np.float32)

    def optimize(self, frames: np.ndarray, gripper_positions: np.ndarray
                 ) -> tuple[np.ndarray, LocalOptimizationStats]:
        frames = np.asarray(frames, dtype=float)
        grip = np.asarray(gripper_positions, dtype=float)
        if frames.ndim != 2 or frames.shape[1] != 14:
            raise ValueError(f"expected (T,14), got {frames.shape}")
        if grip.shape != (len(frames), 2):
            raise ValueError(f"expected gripper (T,2), got {grip.shape}")

        unsafe = np.array([not self.projector.is_safe(q, g)
                           for q, g in zip(frames, grip)], dtype=bool)
        stats = LocalOptimizationStats(unsafe_before=int(unsafe.sum()))
        stats.windows = self._windows(unsafe, self.padding)
        result = frames.copy()
        for start, stop in stats.windows:
            original = result[start:stop].copy()
            optimized = self._optimize_window(original, grip[start:stop], stats)
            old_unsafe = sum(not self.projector.is_safe(q, g)
                             for q, g in zip(original, grip[start:stop]))
            new_unsafe = sum(not self.projector.is_safe(q, g)
                             for q, g in zip(optimized, grip[start:stop]))
            if new_unsafe < old_unsafe:
                result[start:stop] = optimized
                stats.improved_windows += 1
        stats.unsafe_after = sum(not self.projector.is_safe(q, g)
                                 for q, g in zip(result, grip))
        return result.astype(np.float32), stats
