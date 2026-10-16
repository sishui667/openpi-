# -*- coding: utf-8 -*-
"""HISTORICAL one-physics-step benchmark, NOT current Member B acceptance.

Retained for provenance of the 2026-10-14 logs. Use realtime_capture.py for
the current full 100-physics-step control interval and real USB capture.

This is the delivery behind two of Member B's three named artifacts --
「接入单目摄像头实时流，测试端到端系统延迟」 and 「实时采集脚本及延迟日志」.

It runs the actual control tick of the frozen single-arm cube task, in a loop,
and times every stage of it:

    render -> mask -> fit -> gripper -> actuators -> physics

Nothing here is a mock-up of the loop: `workspace_camera.perceive` does the
fitting, `pose2arm_driver.GripperHysteresis` makes the open/close decision, and
`teleop_towel_dataset.apply_controller` writes the actuator torques.  The only
thing this script forces is the ARM POSE, which it holds still -- the point is
to measure the loop's cost, not to reach anywhere.

What "end-to-end" means here, precisely: from the moment the camera frame is
requested to the moment the physics step that consumed the resulting gripper
command has completed. That is the closed-loop latency a 20 Hz controller has
to live inside, and it does NOT include the time the sim takes to boot or the
renderer's first-frame GL context creation. Both are reported separately,
because quoting the boot cost as if it were per-frame latency would be a lie in
the flattering direction.

    python realtime_capture.py                          # 200 frames, 20 Hz
    python realtime_capture.py --frames 500 --hz 30
    python realtime_capture.py --save-frames            # also dump the RGB stream
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import workspace_camera as wc                                              # noqa: E402
from pose2arm_driver import GripperHysteresis                              # noqa: E402
from teleop_towel_dataset import (apply_controller, build_arm_specs,       # noqa: E402
                                  build_finger_actuators)

STAGES = ("render", "mask", "fit", "control")


_LOAD_X = np.random.default_rng(0).random((1024, 512))
_LOAD_REPS = 40


def request_high_priority():
    """Raise only this benchmark/control process for its lifetime on Windows."""
    if sys.platform != "win32":
        return False, "unsupported platform"
    high_priority_class = 0x00000080
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ctypes.set_last_error(0)
    ok = bool(kernel32.SetPriorityClass(handle, high_priority_class))
    error = ctypes.get_last_error()
    return ok, ("HIGH_PRIORITY_CLASS" if ok else f"WinError {error}")


def machine_load_index():
    """Fixed CPU workload, timed, in milliseconds.

    Not a load average: a direct measurement of how much CPU this process can
    actually get, taken in the same units as the benchmark.  It is here because
    this machine's timings swing by 3-5x depending on what else is running --
    one run was timed against a Chrome process holding ~16 of 32 logical cores
    -- and a latency log without it cannot be compared to any other run.

    Reports a raw number, not a ratio to some assumed-idle baseline: there is no
    idle baseline on this machine to measure against.  Use it to compare runs.
    """
    np.fft.rfft(_LOAD_X, axis=1)          # warm the FFT plan / thread pool
    t = time.perf_counter()
    for _ in range(_LOAD_REPS):
        np.fft.rfft(_LOAD_X, axis=1)
    return (time.perf_counter() - t) * 1e3


def percentiles(xs):
    xs = sorted(xs)
    n = len(xs)

    def pct(q):
        if n == 1:
            return xs[0]
        k = q * (n - 1)
        lo, hi = int(np.floor(k)), int(np.ceil(k))
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"mean": statistics.fmean(xs), "p50": pct(0.50), "p95": pct(0.95),
            "p99": pct(0.99),
            "min": xs[0], "max": xs[-1], "stdev": statistics.pstdev(xs) if n > 1 else 0.0}


def build_scene():
    m = mujoco.MjModel.from_xml_path(str(wc.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    while d.time < 2.0:                # let the cube fall to its resting pose
        mujoco.mj_step(m, d)
    return m, d


def run(frames, hz, warmup, save_frames, outdir, width=wc.W, height=wc.H,
        fit_topk=40, priority_note="normal", max_miss_rate=0.0):
    m, d = build_scene()
    specs = build_arm_specs(m)
    finger_act = build_finger_actuators(m)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)       # noqa: E731
    finger_geoms = tuple(gid(f"openarm_right_{s}_finger_collision")
                         for s in ("right", "left"))
    hyst = GripperHysteresis()

    # The arm is held where it starts: this measures the loop, not a reach.
    hold = {side: d.qpos[specs[side].qpos].copy() for side in ("left", "right")}

    load_before = machine_load_index()
    boot_t = time.perf_counter()
    renderer = mujoco.Renderer(m, height, width)
    boot_ms = (time.perf_counter() - boot_t) * 1e3

    # first render pays for the GL context; time it and set it aside
    t = time.perf_counter()
    renderer.update_scene(d, camera=wc.CAM_NAME)
    renderer.render()
    first_frame_ms = (time.perf_counter() - t) * 1e3
    for _ in range(warmup):
        renderer.update_scene(d, camera=wc.CAM_NAME)
        renderer.render()

    step_us = int(round(1e6 / hz))
    budget_ms = 1000.0 / hz
    stage_t = {s: [] for s in STAGES}
    total_t = []
    poses, pose_errors_mm, grip_states, declared_gap = [], [], [], []
    previous_pose = None
    global_fallbacks = 0
    object_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "droid_object")

    if save_frames:
        outdir.mkdir(parents=True, exist_ok=True)

    for i in range(frames):
        t_frame = time.perf_counter()

        t0 = time.perf_counter()
        renderer.update_scene(d, camera=wc.CAM_NAME)
        rgb = renderer.render()
        t1 = time.perf_counter()
        mask = wc.red_mask(rgb)
        t2 = time.perf_counter()
        cam = wc.Pinhole(m, d, wc.CAM_NAME, w=width, h=height)
        pose, score = (None, 0.0)
        if int(np.count_nonzero(mask)):
            if previous_pose is None:
                best, score = wc.fit(cam, mask, topk=fit_topk)
            else:
                best, score, fell_back = wc.fit_tracking(cam, mask, previous_pose)
                global_fallbacks += int(fell_back)
            if best is not None:
                pose = dict(center=np.array([best[0], best[1], wc.TABLE_TOP + wc.CUBE_HALF]),
                            yaw=best[2], iou=score)
                previous_pose = best
        t3 = time.perf_counter()

        # gripper: sim-side jaw midpoint to the perceived centre (team decision --
        # pose2arm emits no hand data, so D cannot come from the human side)
        if pose is None:
            finger_q = hyst.update(None)
            dist = None
        else:
            mid = 0.5 * (d.geom_xpos[finger_geoms[0]] + d.geom_xpos[finger_geoms[1]])
            dist = float(np.linalg.norm(mid - pose["center"]))
            finger_q = hyst.update(dist)
        apply_controller(m, d, specs, finger_act, hold,
                         {s: finger_q for s in ("left", "right")})
        mujoco.mj_step(m, d)
        t4 = time.perf_counter()

        stage_t["render"].append((t1 - t0) * 1e3)
        stage_t["mask"].append((t2 - t1) * 1e3)
        stage_t["fit"].append((t3 - t2) * 1e3)
        stage_t["control"].append((t4 - t3) * 1e3)
        total_t.append((t4 - t_frame) * 1e3)
        poses.append(None if pose is None else [float(v) for v in pose["center"][:2]])
        if pose is not None:
            pose_errors_mm.append(float(
                np.linalg.norm(pose["center"][:2] - d.xpos[object_body, :2]) * 1000.0))
        grip_states.append(hyst.state)

        if save_frames:
            import cv2
            cv2.imwrite(str(outdir / f"frame_{i:06d}.png"),
                        cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))

        # real-time pacing: sleep off whatever is left of this tick
        slack = step_us - (time.perf_counter() - t_frame) * 1e6
        if slack > 0:
            time.sleep(slack / 1e6)
        else:
            declared_gap.append(i)

    load_after = machine_load_index()
    achieved_hz = frames / sum(total_t) * 1e3
    miss_rate = len(declared_gap) / frames
    strict_verdict = ("PASS" if len(declared_gap) == 0
                      else f"FAIL on {len(declared_gap)}/{frames} ticks")
    soft_pass = (miss_rate <= max_miss_rate
                 and percentiles(total_t)["p95"] <= budget_ms)
    log = {
        "machine_load_index_ms": {
            "before": load_before, "after": load_after,
            "drift_factor": load_after / load_before,
            "note": "identical fixed workload timed before and after. A quiet "
                    "machine measures ~6 ms here. The whole run is only "
                    "comparable to another run with a similar value.",
        },
        "role": "Member B -- monocular camera real-time stream, end-to-end latency",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scene": str(wc.SCENE_XML),
        "camera": wc.CAM_NAME,
        "resolution_wh": [width, height],
        "fit_topk": fit_topk,
        "tracking": {
            "mode": "first-frame global, then local with global fallback",
            "global_fallbacks": global_fallbacks,
        },
        "process_priority": priority_note,
        "control_hz": hz,
        "budget_ms": budget_ms,
        "frames": frames,
        "warmup_frames": warmup,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
        },
        "excluded_from_latency": {
            "sim_boot_ms": None,           # filled below
            "renderer_alloc_ms": boot_ms,
            "first_frame_render_ms": first_frame_ms,
            "note": "one-off costs. Quoting them as per-frame latency would "
                    "overstate the loop's cost by a factor of two or more.",
        },
        "stages_ms": {s: percentiles(stage_t[s]) for s in STAGES},
        "end_to_end_ms": percentiles(total_t),
        "achieved_hz": achieved_hz,
        "ticks_over_budget": len(declared_gap),
        "deadline_miss_rate": miss_rate,
        "strict_deadline_verdict": strict_verdict,
        "allowed_miss_rate": max_miss_rate,
        "budget_pass": soft_pass,
        "budget_verdict": ("PASS" if soft_pass and len(declared_gap) == 0
                           else "PASS_SOFT_REALTIME" if soft_pass
                           else strict_verdict),
        "gripper": {
            "input": "sim-side jaw-midpoint to perceived-centre distance",
            "rule": "D<2.5cm close / D>4.5cm open / hold in between",
            "final_state": hyst.state,
            "distinct_states": sorted(set(grip_states)),
            "state_changes": sum(1 for a, b in zip(grip_states, grip_states[1:]) if a != b),
        },
        "perception": {
            "valid_frames": sum(1 for p in poses if p is not None),
            "pose_xy_m": [p for p in poses[:5]],
            "ground_truth_xy_m": [float(v) for v in d.xpos[object_body, :2]],
            "error_mm": percentiles(pose_errors_mm) if pose_errors_mm else None,
        },
        "note": ("The arm pose is held fixed: this measures the cost of one "
                 "closed-loop control tick, not a reach. The latency log is only "
                 "meaningful for the end-to-end figure alongside the gripper "
                 "decision, which is included in the control stage."),
    }
    print(f"end-to-end   mean {log['end_to_end_ms']['mean']:.1f} ms   "
          f"p50 {log['end_to_end_ms']['p50']:.1f}   p95 {log['end_to_end_ms']['p95']:.1f}   "
          f"max {log['end_to_end_ms']['max']:.1f}")
    for s in STAGES:
        st = log["stages_ms"][s]
        print(f"  {s:<8} mean {st['mean']:6.2f}  p95 {st['p95']:6.2f}  max {st['max']:7.2f} ms")
    print(f"budget       {budget_ms:.1f} ms at {hz} Hz -> {log['budget_verdict']}   "
          f"({achieved_hz:.1f} Hz achieved)")
    print(f"machine load {load_before:.1f} -> {load_after:.1f} ms "
          f"({load_after/load_before:.2f}x)  [compare between runs, no absolute scale]")
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--save-frames", action="store_true",
                    help="also write every RGB frame as a PNG (slow; off by default)")
    ap.add_argument("--width", type=int, default=wc.W,
                    help=f"stream width (default: {wc.W})")
    ap.add_argument("--height", type=int, default=wc.H,
                    help=f"stream height (default: {wc.H})")
    ap.add_argument("--fit-topk", type=int, default=40,
                    help="analytic candidates rescored with real IoU (default: 40)")
    ap.add_argument("--high-priority", action="store_true",
                    help="request Windows HIGH_PRIORITY_CLASS for this process only")
    ap.add_argument("--max-miss-rate", type=float, default=0.0,
                    help="allowed fraction of ticks over budget; 0 is strict (default)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="defaults to outputs/YYYYMMDD/")
    a = ap.parse_args()
    outdir = a.outdir or (ROOT / "outputs" / datetime.now().strftime("%Y%m%d"))
    outdir.mkdir(parents=True, exist_ok=True)

    t_boot = time.perf_counter()
    if a.width <= 0 or a.height <= 0:
        ap.error("--width and --height must be positive")
    if a.fit_topk <= 0:
        ap.error("--fit-topk must be positive")
    if not 0.0 <= a.max_miss_rate <= 1.0:
        ap.error("--max-miss-rate must be between 0 and 1")
    priority_note = "normal"
    if a.high_priority:
        ok, priority_note = request_high_priority()
        if not ok:
            print(f"warning: could not raise process priority ({priority_note})",
                  file=sys.stderr)
    log = run(a.frames, a.hz, a.warmup, a.save_frames, outdir,
              width=a.width, height=a.height, fit_topk=a.fit_topk,
              priority_note=priority_note, max_miss_rate=a.max_miss_rate)
    log["excluded_from_latency"]["sim_boot_ms"] = (time.perf_counter() - t_boot) * 1e3

    path = outdir / "latency_log.json"
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=float) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"wrote        {path}")
    return 0 if log["budget_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
