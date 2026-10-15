# -*- coding: utf-8 -*-
"""Why does the blind-pose grasp miss?

Part 1 (probe)  -- kinematics.  Where does solve_relaxed_gripper_q actually put
                   the jaw, and which way do the fingers point?
Part 2 (close)  -- dynamics.  Teleport to the solved grasp pose, close the
                   fingers, and read the contacts back in the object frame.
                   This is the step where a correct kinematic solution can
                   still fail, so it is the one that has to be measured.

    python diag_blind_pose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import workspace_camera as wc                                              # noqa: E402
from demo_grasp_viewer import (ARM, CUBE_HALF, FINGER_OPEN, SCENE,          # noqa: E402
                               load_baked_grasp, place_cube_for_blind_test)
from handeye_grasp_red_cube import (finger_joint_indices,                   # noqa: E402
                                    search_reachable_grasp,
                                    solve_relaxed_gripper_q)
from teleop_towel_dataset import (apply_controller, build_arm_specs,        # noqa: E402
                                  build_finger_actuators)


def setup(cube_xy, yaw_deg):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    ctx = {"m": m, "d": d, "DT": float(m.opt.timestep)}
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)        # noqa: E731
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)        # noqa: E731
    ctx.update(specs=build_arm_specs(m), fing_act=build_finger_actuators(m),
               spec=build_arm_specs(m)[ARM], ob=bid("droid_object"),
               og=gid("droid_object_geom"),
               FGI=(gid(f"openarm_{ARM}_right_finger_collision"),
                    gid(f"openarm_{ARM}_left_finger_collision")),
               FQ=finger_joint_indices(m, ARM),
               SPAN=max(2 * CUBE_HALF + 0.004, 0.070))
    mujoco.mj_resetDataKeyframe(m, d, 0)
    place_cube_for_blind_test(m, d, [cube_xy[0], cube_xy[1], yaw_deg])
    ctx["q_hold"] = {s: d.qpos[ctx["specs"][s].qpos].copy() for s in ("left", "right")}
    for _ in range(int(1.2 / ctx["DT"])):
        apply_controller(m, d, ctx["specs"], ctx["fing_act"], ctx["q_hold"],
                         {"left": FINGER_OPEN, "right": FINGER_OPEN})
        mujoco.mj_step(m, d)
    renderer = mujoco.Renderer(m, wc.H, wc.W)
    pose, _rgb, mask, _cam = wc.perceive(m, d, renderer=renderer, verbose=False)
    renderer.close()
    ctx["pose"], ctx["mask_px"] = pose, int(np.count_nonzero(mask))
    return ctx


def yaw_of(ctx):
    """Cube yaw about z, read back from the simulator (it can spin on settle)."""
    R = ctx["d"].xmat[ctx["ob"]].reshape(3, 3)
    return float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))


def probe(ctx, offset, ai, asgn):
    """Kinematics only: does the IK put the jaw where it was asked to?"""
    m, d = ctx["m"], ctx["d"]
    spec, FGI, FQ, ob = ctx["spec"], ctx["FGI"], ctx["FQ"], ctx["ob"]
    if ctx["pose"] is None:
        print(f"  camera sees nothing ({ctx['mask_px']} px) -- skipped")
        return None
    cen = ctx["pose"]["center"]
    target = cen + offset

    def fk(q_arm, finger_q):
        saved_q, saved_v = d.qpos.copy(), d.qvel.copy()
        d.qpos[spec.qpos] = q_arm
        d.qvel[spec.dof] = 0.0
        qi, di = FQ
        d.qpos[qi] = finger_q
        d.qvel[di] = 0.0
        mujoco.mj_forward(m, d)
        out = (d.geom_xpos[FGI[0]].copy(), d.geom_xpos[FGI[1]].copy(),
               d.xpos[ob].copy())
        d.qpos[:], d.qvel[:] = saved_q, saved_v
        mujoco.mj_forward(m, d)
        return out

    q_grasp, score, _p, _me = solve_relaxed_gripper_q(
        m, d, spec, FGI, FQ, target, ai, asgn, ctx["SPAN"], FINGER_OPEN,
        ctx["q_hold"][ARM].copy())
    g0, g1, cu = fk(q_grasp, FINGER_OPEN)
    axis = g0 - g1
    axis /= np.linalg.norm(axis)
    print(f"  perceived ({cen[0]:.4f}, {cen[1]:.4f})  IoU {ctx['pose']['iou']:.3f}  "
          f"cube yaw {yaw_of(ctx):6.1f} deg")
    print(f"  IK score {score*1000:6.2f} mm   jaw-target "
          f"{np.linalg.norm((g0+g1)/2 - target)*1000:5.1f} mm   "
          f"finger axis {np.round(axis, 2)}  gap {np.linalg.norm(g0-g1)*1000:.0f} mm")
    return q_grasp, target


def close(ctx, q_grasp):
    """Dynamics: teleport to the grasp pose, close, and read the contacts."""
    m, d = ctx["m"], ctx["d"]
    spec, FGI, FQ, og = ctx["spec"], ctx["FGI"], ctx["FQ"], ctx["og"]
    other = "left" if ARM == "right" else "right"
    d.qpos[spec.qpos] = q_grasp
    d.qvel[spec.dof] = 0.0
    qi, di = FQ
    d.qpos[qi] = FINGER_OPEN
    d.qvel[di] = 0.0
    mujoco.mj_forward(m, d)

    q_cmd = {ARM: q_grasp, other: ctx["q_hold"][other]}
    for _ in range(int(1.5 / ctx["DT"])):
        apply_controller(m, d, ctx["specs"], ctx["fing_act"], q_cmd,
                         {"left": 0.0, "right": 0.0})
        mujoco.mj_step(m, d)

    R = d.xmat[ctx["ob"]].reshape(3, 3)
    cs = [d.contact[i] for i in range(d.ncon)
          if og in {d.contact[i].geom1, d.contact[i].geom2}
          and (FGI[0] in {d.contact[i].geom1, d.contact[i].geom2}
               or FGI[1] in {d.contact[i].geom1, d.contact[i].geom2})]
    force = float(np.abs(d.actuator_force[ctx["fing_act"][ARM]]).sum())
    mid = 0.5 * (d.geom_xpos[FGI[0]] + d.geom_xpos[FGI[1]])
    print(f"  after close: {len(cs)} contacts, actuator force {force:.3f} N, "
          f"jaw-cube {np.linalg.norm(mid - d.xpos[ctx['ob']])*1000:.1f} mm")
    for c in cs:
        local = R.T @ (c.pos - d.xpos[ctx["ob"]]) * 1000
        which = ("x" if abs(abs(local[0]) - CUBE_HALF * 1000) < 3 else
                 "y" if abs(abs(local[1]) - CUBE_HALF * 1000) < 3 else
                 "z" if abs(abs(local[2]) - CUBE_HALF * 1000) < 3 else "?")
        print(f"    object-local {np.round(local, 1)} mm  -> {which} face")


def full_episode(ctx, offset, ai, asgn, axis_check=False):
    """Reproduce demo_grasp_viewer's move sequence and watch the cube.

    The teleport test proved the kinematic solution closes correctly.  This runs
    the same thing the way the demo does -- approach from above, descend, then
    close -- and reports the cube's position after every stage, so a cube that
    gets knocked out of place shows up as a jump.
    """
    m, d = ctx["m"], ctx["d"]
    DT = ctx["DT"]
    spec, FGI, FQ, ob = ctx["spec"], ctx["FGI"], ctx["FQ"], ctx["ob"]
    other = "left" if ARM == "right" else "right"
    pose = ctx["pose"]
    if pose is None:
        return
    target = pose["center"] + offset

    if axis_check:
        # exactly what demo_grasp_viewer does before solving: six full solves
        # whose only purpose is to report which axis wins
        _, sc, _, met = search_reachable_grasp(
            m, d, spec, FGI, FQ, pose["center"], ctx["SPAN"], FINGER_OPEN,
            ctx["q_hold"][ARM].copy())
        print(f"  axis check: search wins "
              f"{'xyz'[int(met['axis_index'])]}{'+' if met['axis_sign'] > 0 else '-'}"
              f" (score {sc*1000:.2f} mm)")

    q_grasp, _, _p, _me = solve_relaxed_gripper_q(
        m, d, spec, FGI, FQ, target, ai, asgn, ctx["SPAN"], FINGER_OPEN,
        ctx["q_hold"][ARM].copy())
    q_appr, *_ = solve_relaxed_gripper_q(
        m, d, spec, FGI, FQ, target + np.array([0, 0, 0.07]), ai, asgn,
        ctx["SPAN"], FINGER_OPEN, q_grasp)

    start = d.xpos[ob].copy()

    def step(q_arm, finger):
        apply_controller(m, d, ctx["specs"], ctx["fing_act"],
                         {ARM: q_arm, other: ctx["q_hold"][other]},
                         {"left": finger, "right": finger})
        mujoco.mj_step(m, d)

    def report(tag):
        err = float(np.abs(d.qpos[spec.qpos] - q_cur).max())
        mid = 0.5 * (d.geom_xpos[FGI[0]] + d.geom_xpos[FGI[1]])
        drift = np.linalg.norm(d.xpos[ob][:2] - start[:2]) * 1000
        print(f"  {tag:<10} joint err {err:6.4f} rad   cube moved {drift:6.1f} mm   "
              f"cube z {d.xpos[ob,2]*1000:7.1f}   jaw-cube "
              f"{np.linalg.norm(mid - d.xpos[ob])*1000:6.1f} mm")

    def move(q_to, seconds, finger, tol=0.012, extra=3.0):
        n = max(1, int(seconds / DT))
        q_from = d.qpos[spec.qpos].copy()
        for i in range(n):
            step(q_from + (q_to - q_from) * ((i + 1) / n), finger)
        t = 0.0
        while t < extra and np.abs(d.qpos[spec.qpos] - q_to).max() > tol:
            step(q_to, finger)
            t += DT
        return float(np.abs(d.qpos[spec.qpos] - q_to).max())

    q_cur = q_appr
    move(q_appr, 2.0, FINGER_OPEN)
    report("approach")
    q_cur = q_grasp
    move(q_grasp, 2.0, FINGER_OPEN)
    report("descend")
    move(q_grasp, 1.5, 0.0)
    report("close")
    for _ in range(int(1.5 / DT)):
        step(q_grasp, 0.0)
    report("hold")
    force = float(np.abs(d.actuator_force[ctx["fing_act"][ARM]]).sum())
    print(f"  grip force {force:.3f} N   cube moved total "
          f"{np.linalg.norm(d.xpos[ob][:2] - start[:2])*1000:.1f} mm")


def main():
    offset, ai, asgn = load_baked_grasp()
    print(f"baked offset {np.round(offset*1000, 1)} mm   "
          f"axis {'xyz'[ai]}{'+' if asgn > 0 else '-'}")

    cases = [((0.43, 0.03), 0.0, "blind pose that failed"),
             ((0.45, 0.00), 0.0, "calibrated pose (control)")]
    for xy, yaw, label in cases:
        for axis_check in (False, True):
            print(f"\n=== {label}: cube at {xy}  "
                  f"axis_check={'ON (as demo does)' if axis_check else 'off'} ===")
            ctx = setup(xy, yaw)
            full_episode(ctx, offset, ai, asgn, axis_check=axis_check)


if __name__ == "__main__":
    main()
