"""Hard inter-arm collision safety projection using MuJoCo collision geoms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_XML = (Path(__file__).resolve().parent.parent /
               "openarm_cloth_with_dependencies_20260929/openarm_cloth/"
               "official_openarm_mujoco/v2/openarm_bimanual.xml")
GRIPPER_OPEN_QPOS = {"left": 0.7854, "right": -0.7854}
GRIPPER_CLOSE_QPOS = {"left": 0.0, "right": 0.0}
GRIPPER_COL = {"right": 0, "left": 1}


@dataclass
class ProjectionStats:
    projected_frames: int = 0
    held_frames: int = 0
    unsafe_candidates: int = 0
    minimum_contact_distance: float = float("inf")


class InterArmCollisionProjector:
    """Projects joint targets onto the collision-free part of a trajectory.

    MuJoCo's collision geoms are expanded by ``margin``. A candidate is unsafe
    when a generated contact connects a left-arm body to a right-arm body.
    Same-arm contacts are intentionally ignored: adjacent links and the two
    mimic fingers naturally touch in this model.
    """

    def __init__(self, xml_path: Path = DEFAULT_XML, margin: float = 0.005,
                 sweep_samples: int = 24, bisection_steps: int = 14):
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("collision margin must be non-negative")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.margin = float(margin)
        self.sweep_samples = max(2, int(sweep_samples))
        self.bisection_steps = max(1, int(bisection_steps))
        names = [f"openarm_{side}_joint{i}"
                 for side in ("left", "right") for i in range(1, 8)]
        self.qpos_addresses = np.asarray([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in names
        ], dtype=int)
        self.finger_qpos_addresses = {
            side: np.asarray([
                self.model.jnt_qposadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"openarm_{side}_finger_joint{i}")]
                for i in (1, 2)
            ], dtype=int)
            for side in ("left", "right")
        }
        for geom_id in range(self.model.ngeom):
            body_name = self._body_name_for_geom(geom_id)
            if self._side(body_name) is not None:
                self.model.geom_margin[geom_id] = self.margin

    @staticmethod
    def _side(body_name: str | None) -> str | None:
        if body_name and "_left_" in body_name:
            return "left"
        if body_name and "_right_" in body_name:
            return "right"
        return None

    def _body_name_for_geom(self, geom_id: int) -> str | None:
        body_id = int(self.model.geom_bodyid[geom_id])
        return mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def _write_pose(self, q: np.ndarray, grip: np.ndarray | None = None) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("collision input must contain 14 finite joint angles")
        self.data.qpos[self.qpos_addresses] = q
        if grip is not None:
            grip = np.asarray(grip, dtype=float)
            if grip.shape != (2,) or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
                raise ValueError("gripper collision state must be finite normalized [right,left]")
            for side in ("left", "right"):
                closure = float(np.clip(grip[GRIPPER_COL[side]], 0.0, 1.0))
                target = (GRIPPER_OPEN_QPOS[side]
                          + closure * (GRIPPER_CLOSE_QPOS[side]
                                       - GRIPPER_OPEN_QPOS[side]))
                self.data.qpos[self.finger_qpos_addresses[side]] = target

    def contacts(self, q: np.ndarray,
                 grip: np.ndarray | None = None) -> list[tuple[float, str, str]]:
        self._write_pose(q, grip)
        mujoco.mj_forward(self.model, self.data)
        result = []
        for contact in self.data.contact:
            body1 = self._body_name_for_geom(int(contact.geom1))
            body2 = self._body_name_for_geom(int(contact.geom2))
            if {self._side(body1), self._side(body2)} == {"left", "right"}:
                result.append((float(contact.dist), body1 or "?", body2 or "?"))
        return result

    def is_safe(self, q: np.ndarray, grip: np.ndarray | None = None) -> bool:
        return not self.contacts(q, grip)

    def collision_penalty(self, q: np.ndarray,
                          grip: np.ndarray | None = None) -> float:
        """Dimensionless soft cost inside the configured safety margin."""
        contacts = self.contacts(q, grip)
        if not contacts:
            return 0.0
        scale = max(self.margin, 1e-4)
        # Deduplicate mesh triangles by body pair so detailed finger meshes do
        # not dominate merely because they emit more contact points.
        worst_by_pair: dict[tuple[str, str], float] = {}
        for distance, body1, body2 in contacts:
            pair = tuple(sorted((body1, body2)))
            worst_by_pair[pair] = min(distance, worst_by_pair.get(pair, float("inf")))
        return float(sum((max(0.0, self.margin - distance) / scale) ** 2
                         for distance in worst_by_pair.values()))

    def project_step(self, previous: np.ndarray, candidate: np.ndarray,
                     previous_grip: np.ndarray, candidate_grip: np.ndarray,
                     policy: str = "project"
                     ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        """Return (safe_q, safe_grip, changed, held) for one transition."""
        previous = np.asarray(previous, dtype=float)
        candidate = np.asarray(candidate, dtype=float)
        previous_grip = np.asarray(previous_grip, dtype=float)
        candidate_grip = np.asarray(candidate_grip, dtype=float)
        if not self.is_safe(previous, previous_grip):
            raise ValueError("previous pose passed to safety projector is unsafe")

        safe_alpha = 0.0
        unsafe_alpha = None
        for alpha in np.linspace(1.0 / self.sweep_samples, 1.0, self.sweep_samples):
            probe = previous + alpha * (candidate - previous)
            probe_grip = previous_grip + alpha * (candidate_grip - previous_grip)
            if self.is_safe(probe, probe_grip):
                safe_alpha = float(alpha)
            else:
                unsafe_alpha = float(alpha)
                break

        if unsafe_alpha is None:
            return candidate.copy(), candidate_grip.copy(), False, False
        if policy == "error":
            raise RuntimeError(f"collision detected at path alpha={unsafe_alpha:.6f}")
        if policy == "hold":
            return previous.copy(), previous_grip.copy(), True, True
        if policy != "project":
            raise ValueError(f"unknown collision policy {policy!r}")

        lo, hi = safe_alpha, unsafe_alpha
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            probe = previous + mid * (candidate - previous)
            probe_grip = previous_grip + mid * (candidate_grip - previous_grip)
            if self.is_safe(probe, probe_grip):
                lo = mid
            else:
                hi = mid

        # Stay one coarse sweep interval inside the safe side. A merely
        # epsilon-sized retreat leaves the next frame sitting on the contact
        # boundary and can make the trajectory stick there for many frames.
        alpha = max(0.0, lo - 1.0 / self.sweep_samples)
        projected = previous + alpha * (candidate - previous)
        projected_grip = previous_grip + alpha * (candidate_grip - previous_grip)
        if not self.is_safe(projected, projected_grip):
            return previous.copy(), previous_grip.copy(), True, True
        return projected, projected_grip, True, alpha <= 1e-9

    def project_trajectory(self, frames: np.ndarray, policy: str = "project",
                           gripper_positions: np.ndarray | None = None,
                           initial_q: np.ndarray | None = None,
                           initial_grip: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray, ProjectionStats]:
        frames = np.asarray(frames, dtype=float)
        if frames.ndim != 2 or frames.shape[1] != 14:
            raise ValueError(f"expected (T,14) trajectory, got {frames.shape}")
        if gripper_positions is None:
            gripper_positions = np.zeros((len(frames), 2), dtype=float)
        gripper_positions = np.asarray(gripper_positions, dtype=float)
        if gripper_positions.shape != (len(frames), 2):
            raise ValueError(
                f"expected (T,2) gripper positions, got {gripper_positions.shape}")
        if initial_q is None:
            initial_q = np.asarray(self.data.qpos[self.qpos_addresses], dtype=float).copy()
        if initial_grip is None:
            initial_grip = np.zeros(2, dtype=float)
        if not self.is_safe(initial_q, initial_grip):
            raise RuntimeError("MuJoCo initial pose is not inter-arm collision-free")

        stats = ProjectionStats()
        out = np.empty_like(frames)
        grip_out = np.empty_like(gripper_positions)
        previous = np.asarray(initial_q, dtype=float)
        previous_grip = np.asarray(initial_grip, dtype=float)
        for i, (candidate, candidate_grip) in enumerate(
                zip(frames, gripper_positions)):
            safe_q, safe_grip, changed, held = self.project_step(
                previous, candidate, previous_grip, candidate_grip, policy)
            out[i] = safe_q
            grip_out[i] = safe_grip
            if changed:
                stats.unsafe_candidates += 1
                stats.projected_frames += int(not held)
                stats.held_frames += int(held)
            previous = safe_q
            previous_grip = safe_grip

        # Required postcondition: never return an unsafe output frame.
        unsafe = [i for i, (q, grip) in enumerate(zip(out, grip_out))
                  if not self.is_safe(q, grip)]
        if unsafe:
            raise RuntimeError(f"collision safety postcondition failed at frames {unsafe[:10]}")
        return out.astype(np.float32), grip_out.astype(np.float32), stats

