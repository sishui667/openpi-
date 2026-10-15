# -*- coding: utf-8 -*-
"""Member B: Pose2Arm output -> OpenArm actuator driver interface.

Takes the npz that pose2arm-main produces and drives the MuJoCo OpenArm
actuators with it.

The npz contract (verified against data/p2a_test.npz):

    target_obs     (F, 18) float32 in [-1, 1]   normalised human arm geometry,
                                                L UPPER 0:3 FORE 3:6 PLANE 6:9,
                                                R UPPER 9:12 FORE 12:15 PLANE 15:18
    joint_angles   (F, 14) float32              openarm joints, LEFT arm first
    frame_indices  (F,)    int64                source frame each row came from
    fps            scalar

TWO TRAPS, both measured rather than assumed:

1. joint_angles is [left 7, right 7] even though openarm_common.py defines
   JOINT_NAMES = RIGHT_JOINT_NAMES with the right arm listed first.  Reading it
   the way the file is written swaps the two arms.  verify_joint_order() below
   re-derives the layout from the joint limits on every load, so this cannot
   drift silently: column 0 reaches -3.2960 rad and the right arm's joint1 stops
   at -1.3963 while the left reaches -3.4907, so only "left first" fits.

2. Pose2Arm emits NO hand data at all, so the task book's gripper hysteresis
   (D < 2.5 cm close, D > 4.5 cm open) has no human-side input.  Per the team
   decision, D is taken from the simulation side: the distance from the gripper's
   jaw midpoint to the monocularly perceived object centre.  This is a sim-side
   substitute, not hand input, and is labelled as such wherever it is reported.

Usage:
    python pose2arm_driver.py --npz <p2a.npz> --dry-run       # contract check only
    python pose2arm_driver.py --npz <p2a.npz> --arm right     # play it in the scene
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from teleop_towel_dataset import (apply_controller, build_arm_specs,  # noqa: E402
                                  build_finger_actuators)

SCENE = ROOT / "simulation" / "v1" / "droid_cube_scene_perception.xml"

# OpenArm v2.0 URDF velocity limits, rad/s, per joint 1..7.
# Copied from pose2arm-main/openarm_common.py:JOINT_VELOCITY_LIMITS so that this
# module does not import across into Member A's tree.
POSE2ARM_VEL_LIMITS = np.array([16.755, 16.755, 5.4454, 5.4454, 20.944, 20.944, 20.944])

# The right arm's declared limits, used only to disambiguate the column order.
RIGHT_JOINT_LIMITS = np.array([[-1.3963, 3.4907], [-0.17453, 3.3161], [-1.5708, 1.5708],
                               [0.0, 2.4435], [-1.5708, 1.5708], [-0.7854, 0.7854],
                               [-0.7854, 0.7854]])
# Mirrored on joints 1 and 2 only; joints 3-7 are identical on both arms.
LEFT_JOINT_LIMITS = np.array([[-3.4907, 1.3963], [-3.3161, 0.17453], [-1.5708, 1.5708],
                              [0.0, 2.4435], [-1.5708, 1.5708], [-0.7854, 0.7854],
                              [-0.7854, 0.7854]])

# task book: D < 2.5 cm -> close, D > 4.5 cm -> open, stay put in between
CLOSE_BELOW_M = 0.025
OPEN_ABOVE_M = 0.045
FINGER_OPEN_Q = 0.044
FINGER_CLOSED_Q = 0.0

# A step this many times the velocity limit is a data fault, not fast motion.
RATE_GLITCH_MULTIPLE = 20.0


def load_pose2arm(npz_path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    need = {"target_obs", "joint_angles", "frame_indices"}
    missing = need - set(z.files)
    if missing:
        raise KeyError(f"{npz_path} is missing {sorted(missing)}; has {sorted(z.files)}")
    return {"target_obs": z["target_obs"], "joint_angles": z["joint_angles"],
            "frame_indices": z["frame_indices"],
            "fps": float(z["fps"]) if "fps" in z.files else None}


def verify_joint_order(joint_angles) -> tuple[str, str]:
    """Return (order, evidence).  Refuses to guess if the data cannot tell."""
    q = np.asarray(joint_angles, dtype=float)
    if q.shape[1] != 14:
        raise ValueError(f"joint_angles must have 14 columns, got {q.shape[1]}")

    def fits(cols, limits) -> bool:
        return all(np.all(q[:, c] >= limits[i % 7, 0] - 1e-4)
                   and np.all(q[:, c] <= limits[i % 7, 1] + 1e-4)
                   for i, c in enumerate(cols))

    lr = fits(range(7), LEFT_JOINT_LIMITS) and fits(range(7, 14), RIGHT_JOINT_LIMITS)
    rl = fits(range(7), RIGHT_JOINT_LIMITS) and fits(range(7, 14), LEFT_JOINT_LIMITS)
    if lr and not rl:
        return "left-first", "cols 0-6 fit LEFT limits, 7-13 fit RIGHT"
    if rl and not lr:
        return "right-first", "cols 0-6 fit RIGHT limits, 7-13 fit LEFT"
    if lr and rl:
        return "ambiguous", "the data fits both orders; not safe to pick one"
    return "violates", "neither order fits the declared joint limits"


@dataclass
class GripperHysteresis:
    """Task book rule, driven here from the sim-side finger-to-object distance."""

    state: str = "open"

    def update(self, d_m: float | None) -> float:
        if d_m is None:
            return FINGER_OPEN_Q if self.state == "open" else FINGER_CLOSED_Q
        if d_m < CLOSE_BELOW_M:
            self.state = "closed"
        elif d_m > OPEN_ABOVE_M:
            self.state = "open"
        # in between: hold the previous state, exactly as the task book says
        return FINGER_CLOSED_Q if self.state == "closed" else FINGER_OPEN_Q


@dataclass
class DriverReport:
    frames: int
    order: str
    order_evidence: str
    arm: str
    clipped_joints: int = 0
    clipped_by_joint: dict = field(default_factory=dict)
    rate_limited_frames: int = 0
    max_clip_rad: float = 0.0
    max_lag_rad: float = 0.0
    drive_seconds: float = 0.0
    estop: str | None = None
    estop_frame: int | None = None
    gripper: dict = field(default_factory=lambda: {"closed": 0, "open": 0})
    source_max_step_rad: dict = field(default_factory=dict)
    ratio_to_20hz_limit: dict = field(default_factory=dict)
    ratio_to_source_fps_limit: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


class Pose2ArmDriver:
    """Feeds pose2arm joint angles into the OpenArm actuators.

    Only the arm named by `arm` is driven; the other holds its home pose, because
    the frozen task is single-arm.
    """

    def __init__(self, model, data, npz_path, arm="right", control_hz=20.0,
                 vel_limits=POSE2ARM_VEL_LIMITS, object_centre_fn=None):
        self.m = model
        self.d = data
        self.blob = load_pose2arm(npz_path)
        self.arm = arm
        self.other = "left" if arm == "right" else "right"
        self.control_hz = float(control_hz)
        self.vel_limits = np.asarray(vel_limits, dtype=float)
        self.object_centre_fn = object_centre_fn

        self.specs = build_arm_specs(model)
        self.finger_act = build_finger_actuators(model)
        geom_id = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.finger_geoms = tuple(geom_id(f"openarm_{arm}_{s}_finger_collision")
                                  for s in ("right", "left"))
        self.hysteresis = GripperHysteresis()

        self.order, self.order_evidence = verify_joint_order(self.blob["joint_angles"])
        if self.order == "left-first":
            self.cols = {"left": slice(0, 7), "right": slice(7, 14)}
        elif self.order == "right-first":
            self.cols = {"right": slice(0, 7), "left": slice(7, 14)}
        else:
            raise ValueError(
                f"cannot safely map pose2arm joint columns: {self.order_evidence}. "
                "Refusing to drive the arms on a guess.")

        self.report = DriverReport(frames=len(self.blob["frame_indices"]),
                                   order=self.order, order_evidence=self.order_evidence,
                                   arm=arm)
        source_q = np.asarray(self.blob["joint_angles"][:, self.cols[arm]], dtype=float)
        source_steps = np.abs(np.diff(source_q, axis=0))
        if len(source_steps):
            maxima = source_steps.max(axis=0)
            self.report.source_max_step_rad = {
                f"{arm}_joint{i + 1}": float(value) for i, value in enumerate(maxima)
            }
            self.report.ratio_to_20hz_limit = {
                f"{arm}_joint{i + 1}": float(value / (self.vel_limits[i] / self.control_hz))
                for i, value in enumerate(maxima)
            }
            if self.blob["fps"]:
                self.report.ratio_to_source_fps_limit = {
                    f"{arm}_joint{i + 1}": float(value / (self.vel_limits[i] / self.blob["fps"]))
                    for i, value in enumerate(maxima)
                }
        self.dt = float(model.opt.timestep)
        self.steps_per_frame = max(1, int(round(1.0 / (self.control_hz * self.dt))))
        self._prev_q = None

    # ---- joint-limit guard -------------------------------------------------
    def _clamp(self, side, q):
        lo, hi = self.specs[side].q_range[:, 0], self.specs[side].q_range[:, 1]
        out = np.clip(q, lo, hi)
        n = int(np.count_nonzero(np.abs(out - q) > 1e-9))
        if n:
            self.report.clipped_joints += n
            for j in np.flatnonzero(np.abs(out - q) > 1e-9):
                key = f"{side}_joint{int(j)+1}"
                self.report.clipped_by_joint[key] = self.report.clipped_by_joint.get(key, 0) + 1
        return out

    def _rate_limit(self, side, q, prev):
        """Cap each per-step joint move to the URDF velocity limit.

        Clamping is the NORMAL operating mode here, not an error.  Pose2Arm's
        retargeted angles are not velocity limited: over data/p2a_test.npz the
        largest single step is 3.279 rad on R1 and 2.643 rad on R3, which is
        3.9x and 9.7x what 20 Hz allows (and 5.9x / 14.5x the 30 fps limit), so
        the arm cannot do what the reference asks and has to lag behind it.
        Throwing the frame away instead would abort every run at frame 9.

        What IS worth stopping for is a step so large it can only be a data
        fault rather than fast motion -- RATE_GLITCH_MULTIPLE times the limit --
        or a non-finite number.

        Returns (q_clamped, is_glitch).
        """
        max_step = self.vel_limits * (1.0 / self.control_hz)
        delta = np.asarray(q) - prev
        if not np.all(np.isfinite(delta)):
            return prev, True
        over = np.abs(delta) > max_step
        if np.any(over):
            clip = np.abs(delta) - max_step
            self.report.rate_limited_frames += 1
            self.report.max_clip_rad = max(self.report.max_clip_rad, float(clip.max()))
            q = np.where(over, prev + np.sign(delta) * max_step, q)
        return q, bool(np.any(np.abs(delta) > RATE_GLITCH_MULTIPLE * max_step))

    def joint_targets(self, i):
        q = {self.arm: self.blob["joint_angles"][i, self.cols[self.arm]]}
        # the other arm holds whatever pose it is already in
        q[self.other] = self.d.qpos[self.specs[self.other].qpos].copy()
        return q

    def jaw_distance(self):
        """Sim-side D: jaw midpoint to the perceived object centre.

        The two finger COLLISION geoms are meshes whose origin sits far from the
        contacting surface, so their midpoint is the honest jaw centre; the EE
        body origin is not (measured 80 mm away on this model).
        """
        if self.object_centre_fn is None:
            return None
        centre = self.object_centre_fn()
        if centre is None:
            return None
        mid = 0.5 * (self.d.geom_xpos[self.finger_geoms[0]]
                     + self.d.geom_xpos[self.finger_geoms[1]])
        return float(np.linalg.norm(mid - np.asarray(centre)))

    def step_frame(self, i):
        if self.report.estop:
            return
        raw = self.joint_targets(i)
        q = {self.arm: self._clamp(self.arm, raw[self.arm]),
             self.other: raw[self.other]}
        if self._prev_q is not None:
            before = q[self.arm].copy()
            q[self.arm], glitch = self._rate_limit(self.arm, before, self._prev_q)
            # how far the arm has fallen behind the reference it was handed
            self.report.max_lag_rad = max(
                self.report.max_lag_rad, float(np.max(np.abs(q[self.arm] - before))))
            if glitch:
                self.report.estop = (
                    f"frame {i}: joint step exceeds {RATE_GLITCH_MULTIPLE:.0f}x the URDF "
                    f"velocity limit at {self.control_hz:.0f} Hz -- data fault, not motion")
                self.report.estop_frame = int(i)
                return
        self._prev_q = q[self.arm].copy()

        finger = self.hysteresis.update(self.jaw_distance())
        self.report.gripper["closed" if finger == FINGER_CLOSED_Q else "open"] += 1
        for _ in range(self.steps_per_frame):
            apply_controller(self.m, self.d, self.specs, self.finger_act,
                             q, {s: finger for s in ("left", "right")})
            mujoco.mj_step(self.m, self.d)


def build_scene():
    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(SCENE))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    return m, d


def dry_run(npz_path) -> int:
    blob = load_pose2arm(npz_path)
    order, evidence = verify_joint_order(blob["joint_angles"])
    print(f"npz            {npz_path}")
    print(f"  target_obs   {blob['target_obs'].shape} {blob['target_obs'].dtype}")
    print(f"  joint_angles {blob['joint_angles'].shape} {blob['joint_angles'].dtype}")
    print(f"  frames       {len(blob['frame_indices'])}  "
          f"source frames {int(blob['frame_indices'].min())}..{int(blob['frame_indices'].max())}")
    print(f"  fps          {blob['fps']}")
    print(f"  arm order    {order}  ({evidence})")
    print(f"  gripper      D<{CLOSE_BELOW_M*100:.1f}cm close / D>{OPEN_ABOVE_M*100:.1f}cm open "
          "from the sim-side jaw-to-object distance")
    return 0 if order in ("left-first", "right-first") else 1


def play(npz_path, arm, max_frames=None, report_path=None) -> int:
    from workspace_camera import CAM_NAME, H, W, perceive

    m, d = build_scene()

    # One persistent renderer for the whole run.  perceive() allocates and
    # closes its own when handed none, which means a fresh GL context per frame:
    # measured at ~70 ms/frame and the process grew past 1.9 GB before the run
    # was stopped.  The renderer is the same object for every frame, so the
    # perception result is unchanged -- only the allocation goes away.
    renderer = mujoco.Renderer(m, H, W)
    try:
        t_first = time.perf_counter()
        renderer.update_scene(d, camera=CAM_NAME)
        renderer.render()
        print(f"  renderer   first frame {(time.perf_counter()-t_first)*1e3:.0f} ms "
              "(GL context), reused for every frame after")

        def perceived_centre():
            pose = perceive(m, d, renderer=renderer, verbose=False)[0]
            return None if pose is None else pose["center"]

        driver = Pose2ArmDriver(m, d, npz_path, arm=arm, object_centre_fn=perceived_centre)
        print(f"npz        {npz_path}")
        print(f"  order    {driver.order} ({driver.order_evidence})")
        print(f"  driving  {arm} arm only; the other holds its home pose")
        print(f"  {driver.report.frames} frames at {driver.control_hz:.0f} Hz "
              f"({driver.steps_per_frame} sim steps each)")

        n = min(driver.report.frames, max_frames) if max_frames else driver.report.frames
        t_run = time.perf_counter()
        for i in range(n):
            driver.step_frame(i)
            if driver.report.estop:
                print(f"\nEMERGENCY STOP at frame {i}: {driver.report.estop}")
                break
        drive_s = time.perf_counter() - t_run
    finally:
        renderer.close()
    r = driver.report
    r.drive_seconds = drive_s
    if max_frames:
        r.frames = min(r.frames, max_frames)
    print()
    print("=" * 74)
    print(f"frames driven         {r.frames if not r.estop else r.estop_frame}")
    print(f"joints clamped        {r.clipped_joints}"
          + (f"  {r.clipped_by_joint}" if r.clipped_by_joint else ""))
    print(f"frames rate-limited   {r.rate_limited_frames}"
          + (f"  (largest single-frame clip {r.max_clip_rad:.3f} rad)" if r.rate_limited_frames else ""))
    print(f"max tracking lag      {r.max_lag_rad:.3f} rad"
          f"   <- how far the arm fell behind pose2arm's reference")
    print(f"gripper frames        closed {r.gripper['closed']} / open {r.gripper['open']}")
    print(f"emergency stop        {r.estop or 'not tripped'}")
    if r.drive_seconds:
        print(f"wall clock            {r.drive_seconds:.1f} s for "
              f"{(r.frames if not r.estop else r.estop_frame)} frames "
              f"({(r.frames if not r.estop else r.estop_frame)/r.drive_seconds:.1f} fps)")
    if r.clipped_joints:
        print("\nclamped joints mean pose2arm asks for angles outside this model's "
              "jnt_range. That is a source/model disagreement, not a rounding issue.")
    if report_path is not None:
        payload = {
            "role": "Member B -- Pose2Arm output to OpenArm actuator validation",
            "source_npz": str(Path(npz_path).resolve()),
            "scene": str(SCENE.resolve()),
            "control_hz": driver.control_hz,
            "steps_per_frame": driver.steps_per_frame,
            **asdict(r),
        }
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
            encoding="utf-8", newline="\n")
        print(f"wrote                  {report_path}")
    return 1 if r.estop else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--arm", choices=("left", "right"), default="right")
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the npz contract without loading the scene")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many frames (smoke test)")
    ap.add_argument("--report", type=Path, default=None,
                    help="write a JSON validation report (play mode only)")
    a = ap.parse_args()
    sys.exit(dry_run(a.npz) if a.dry_run
             else play(a.npz, a.arm, a.max_frames, a.report))
