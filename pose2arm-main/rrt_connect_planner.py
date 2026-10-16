"""Lightweight joint-space RRT-Connect for bimanual exchange motions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collision_safety import InterArmCollisionProjector
from openarm_common import LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS


@dataclass
class RRTConnectStats:
    solved: bool = False
    iterations: int = 0
    nodes: int = 0
    raw_waypoints: int = 0
    shortcut_waypoints: int = 0


class _Tree:
    def __init__(self, root: np.ndarray, root_is_start: bool):
        self.nodes = [np.asarray(root, dtype=float).copy()]
        self.parents = [-1]
        self.root_is_start = root_is_start

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(q, dtype=float).copy())
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def trace_from_root(self, index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        return path[::-1]


class BimanualRRTConnect:
    """Bidirectional RRT with MuJoCo edge validation in 14-DOF joint space."""

    def __init__(self, projector: InterArmCollisionProjector,
                 step_size: float = 0.12, edge_resolution: float = 0.03,
                 max_iterations: int = 1800, goal_bias: float = 0.20,
                 corridor_bias: float = 0.70, corridor_sigma: float = 0.22,
                 seed: int = 19):
        self.projector = projector
        self.step_size = float(step_size)
        self.edge_resolution = float(edge_resolution)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.corridor_bias = float(corridor_bias)
        self.corridor_sigma = float(corridor_sigma)
        self.rng = np.random.default_rng(seed)
        self.low = np.concatenate([LEFT_JOINT_LIMITS[:, 0], RIGHT_JOINT_LIMITS[:, 0]])
        self.high = np.concatenate([LEFT_JOINT_LIMITS[:, 1], RIGHT_JOINT_LIMITS[:, 1]])

    def _state_safe(self, q: np.ndarray, grip_probes: np.ndarray) -> bool:
        return all(self.projector.is_safe(q, grip) for grip in grip_probes)

    def _edge_safe(self, a: np.ndarray, b: np.ndarray,
                   grip_probes: np.ndarray) -> bool:
        steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / self.edge_resolution)))
        return all(self._state_safe(a + (i / steps) * (b - a), grip_probes)
                   for i in range(1, steps + 1))

    def _nearest(self, tree: _Tree, target: np.ndarray) -> int:
        # Normalize dimensions so large-range joints do not dominate.
        scale = np.maximum(self.high - self.low, 1e-6)
        distances = [np.linalg.norm((q - target) / scale) for q in tree.nodes]
        return int(np.argmin(distances))

    def _steer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - source
        distance = float(np.linalg.norm(delta))
        if distance <= self.step_size:
            return np.clip(target, self.low, self.high)
        return np.clip(source + (self.step_size / distance) * delta,
                       self.low, self.high)

    def _extend(self, tree: _Tree, target: np.ndarray,
                grip_probes: np.ndarray) -> tuple[int | None, bool]:
        parent = self._nearest(tree, target)
        new_q = self._steer(tree.nodes[parent], target)
        if np.allclose(new_q, tree.nodes[parent], atol=1e-10):
            return None, False
        if not self._edge_safe(tree.nodes[parent], new_q, grip_probes):
            return None, False
        index = tree.add(new_q, parent)
        reached = np.linalg.norm(new_q - target) <= 1e-8
        return index, reached

    def _connect(self, tree: _Tree, target: np.ndarray,
                 grip_probes: np.ndarray) -> tuple[int | None, bool]:
        last = None
        while True:
            index, reached = self._extend(tree, target, grip_probes)
            if index is None:
                return last, False
            last = index
            if reached:
                return index, True

    @staticmethod
    def _join(tree_a: _Tree, index_a: int,
              tree_b: _Tree, index_b: int) -> list[np.ndarray]:
        a = tree_a.trace_from_root(index_a)
        b = tree_b.trace_from_root(index_b)
        if tree_a.root_is_start:
            return a + b[-2::-1]
        return b + a[-2::-1]

    def _shortcut(self, path: list[np.ndarray], grip_probes: np.ndarray,
                  attempts: int = 100,
                  preserve_detour: bool = False) -> list[np.ndarray]:
        result = [q.copy() for q in path]
        for _ in range(attempts):
            if len(result) <= (3 if preserve_detour else 2):
                break
            i, j = sorted(self.rng.integers(0, len(result), size=2))
            if j <= i + 1:
                continue
            if preserve_detour and i == 0 and j == len(result) - 1:
                continue
            if self._edge_safe(result[i], result[j], grip_probes):
                result = result[:i + 1] + result[j:]
        return result

    def plan(self, start: np.ndarray, goal: np.ndarray,
             corridor: np.ndarray, gripper_window: np.ndarray,
             allow_direct: bool = True, preserve_detour: bool = False,
             ) -> tuple[list[np.ndarray] | None, RRTConnectStats]:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        corridor = np.asarray(corridor, dtype=float)
        grips = np.asarray(gripper_window, dtype=float)
        # Tree nodes do not carry time. Plan with the representative window
        # opening, then validate the resampled path against the actual
        # time-varying gripper sequence. Requiring every tree node to be safe
        # under every opening rejects valid endpoints for no physical reason.
        grip_probes = np.asarray([np.median(grips, axis=0)])
        stats = RRTConnectStats()
        if (not self.projector.is_safe(start, grips[0])
                or not self.projector.is_safe(goal, grips[-1])):
            return None, stats
        if allow_direct and self._edge_safe(start, goal, grip_probes):
            stats.solved = True
            stats.raw_waypoints = stats.shortcut_waypoints = 2
            return [start.copy(), goal.copy()], stats

        tree_a = _Tree(start, True)
        tree_b = _Tree(goal, False)
        for iteration in range(1, self.max_iterations + 1):
            draw = self.rng.random()
            if draw < self.goal_bias:
                sample = tree_b.nodes[0]
            elif draw < self.goal_bias + self.corridor_bias and len(corridor):
                center = corridor[self.rng.integers(len(corridor))]
                sample = np.clip(center + self.rng.normal(
                    0.0, self.corridor_sigma, 14),
                                 self.low, self.high)
            else:
                sample = self.rng.uniform(self.low, self.high)
            index_a, _ = self._extend(tree_a, sample, grip_probes)
            if index_a is not None:
                index_b, connected = self._connect(
                    tree_b, tree_a.nodes[index_a], grip_probes)
                if connected and index_b is not None:
                    raw = self._join(tree_a, index_a, tree_b, index_b)
                    short = self._shortcut(
                        raw, grip_probes, preserve_detour=preserve_detour)
                    stats.solved = True
                    stats.iterations = iteration
                    stats.nodes = len(tree_a.nodes) + len(tree_b.nodes)
                    stats.raw_waypoints = len(raw)
                    stats.shortcut_waypoints = len(short)
                    return short, stats
            tree_a, tree_b = tree_b, tree_a
        stats.iterations = self.max_iterations
        stats.nodes = len(tree_a.nodes) + len(tree_b.nodes)
        return None, stats

    def resample(self, path: list[np.ndarray], count: int,
                 gripper_window: np.ndarray) -> np.ndarray | None:
        points = np.asarray(path, dtype=float)
        if len(points) == 1:
            return np.repeat(points, count, axis=0)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        if cumulative[-1] < 1e-10:
            return np.repeat(points[:1], count, axis=0)
        query = np.linspace(0.0, cumulative[-1], count)
        result = np.column_stack([
            np.interp(query, cumulative, points[:, j]) for j in range(14)
        ])
        grips = np.asarray(gripper_window, dtype=float)
        for i in range(1, len(result)):
            if not self._edge_safe(result[i - 1], result[i], grips[[i - 1, i]]):
                return None
        return result.astype(np.float32)
