"""Measured Member B streams: simulated monocular control, or real USB capture.

USB mode is capture-only until A provides a camera-to-joints online producer.
It must never feed webcam pixels into the simulator's camera calibration.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import threading
import time

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent


def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "min": float(a.min()), "max": float(a.max())}


def request_high_priority():
    if sys.platform != "win32":
        return "unsupported platform"
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    ok = bool(api.SetPriorityClass(api.GetCurrentProcess(), 0x80))
    return "HIGH_PRIORITY_CLASS" if ok else f"WinError {ctypes.get_last_error()}"


class USBLatestFrame:
    """Bounded latest-frame slot; never accumulate an old-frame FIFO."""
    def __init__(self, device, width, height, fps):
        self.cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"USB camera {device} could not be opened")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Set the codec LAST: DSHOW frame-rate setup may rebuild its device.
        codec_ok = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        auto_ok = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.metadata = {"device": device, "backend": self.cap.getBackendName(),
                         "mjpg_requested_ok": bool(codec_ok),
                         "auto_exposure_requested_ok": bool(auto_ok),
                         "reported_width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                         "reported_height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         "reported_fps": self.cap.get(cv2.CAP_PROP_FPS)}
        self.condition = threading.Condition()
        self.closed, self.sequence, self.latest, self.error = False, 0, None, None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.closed:
            start = time.perf_counter_ns()
            ok, bgr = self.cap.read()
            end = time.perf_counter_ns()
            with self.condition:
                if not ok:
                    self.error = "USB read failed"
                    self.condition.notify_all()
                    return
                self.sequence += 1
                self.latest = (self.sequence, start, end, bgr)
                self.condition.notify_all()

    def read_new(self, after, timeout=2):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > after or self.error is not None, timeout)
            if not ready or self.error:
                raise RuntimeError(self.error or "USB new-frame timeout")
            return self.latest

    def close(self):
        self.closed = True
        self.thread.join(timeout=2)
        self.cap.release()
        self.thread.join(timeout=1)


def simulation_setup(npz, hz, gripper_mode="auto"):
    import workspace_camera as wc
    from member_b_scene import build_scene
    from pose2arm_driver import Pose2ArmDriver
    from teleop_towel_dataset import apply_controller, build_arm_specs, build_finger_actuators
    model, data = build_scene()
    specs, actuators = build_arm_specs(model), build_finger_actuators(model)
    holds = {s: data.qpos[spec.qpos].copy() for s, spec in specs.items()}
    for _ in range(round(1.2 / model.opt.timestep)):
        apply_controller(model, data, specs, actuators, holds, {"left": 0.044, "right": 0.044})
        mujoco.mj_step(model, data)
    driver = Pose2ArmDriver(model, data, npz, arm="right", control_hz=hz, gripper_mode=gripper_mode) if npz else None
    return wc, model, data, specs, actuators, holds, driver


def run(args):
    from teleop_towel_dataset import apply_controller
    from pose2arm_driver import GripperHysteresis
    from task_workspace import CONFIG
    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        (output / "frames").mkdir(exist_ok=True)
    boot_start = time.perf_counter()
    usb = renderer = None
    wc = model = data = driver = None
    requested = args.frames
    period = 1 / args.hz
    previous = None
    skipped = last_sequence = failures = 0
    frame_rows, commands, measured, controls, sim_times, poses = [], [], [], [], [], []
    stage = {name: [] for name in ("capture", "mask", "fit", "mapping", "control_physics")}
    e2e, work, reads, ages, first_consumed, inactive_errors = [], [], [], [], [], []
    grip = GripperHysteresis()
    failure = None
    if args.source == "simulation":
        wc, model, data, specs, actuators, holds, driver = simulation_setup(args.npz, args.hz, args.gripper_mode)
        for key in wc.TRACKING_STATS:
            wc.TRACKING_STATS[key] = 0
        from member_b_actuators import ActuatorController
        controller = driver.controller if driver else ActuatorController(model, data)
        renderer = mujoco.Renderer(model, args.height, args.width)
        steps = round(period / float(model.opt.timestep))
        if abs(steps * model.opt.timestep - period) > 1e-8:
            raise ValueError("control period must be an integer number of physics timesteps")
        if driver:
            requested = min(requested, driver.report.frames)
        g = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             f"openarm_right_{s}_finger_collision") for s in ("right", "left")]
    else:
        usb = USBLatestFrame(args.device, args.width, args.height, max(30, args.hz))
        steps = 0
    boot_ms = (time.perf_counter() - boot_start) * 1000
    warm_start = time.perf_counter()
    try:
        for _ in range(args.warmup):
            if usb:
                last_sequence, *_ = usb.read_new(last_sequence)
            else:
                renderer.update_scene(data, camera=wc.CAM_NAME)
                rgb = renderer.render()
                mask = wc.red_mask(rgb)
                wc.fit(wc.Pinhole(model, data, wc.CAM_NAME, args.width, args.height), mask,
                       topk=args.fit_topk)
        warmup_ms = (time.perf_counter() - warm_start) * 1000
        run_start = time.perf_counter()
        fields = ["tick", "source_sequence", "host_request_ns", "host_delivered_ns",
                  "actuator_written_ns", "first_physics_consumed_ns", "physics_finished_ns",
                  "capture_ms", "mask_ms", "fit_ms", "mapping_ms", "control_physics_ms",
                  "host_e2e_ms", "tick_work_ms", "source_read_ms", "source_age_ms",
                  "perception_valid", "npz_row", "sim_time_s"]
        with (output / "latency_frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(requested):
                start = time.perf_counter_ns()
                actuator_stamp = consumed_stamp = 0
                source_row = i if driver else -1
                if usb:
                    sequence, read_start, read_end, bgr = usb.read_new(last_sequence)
                    skipped += max(0, sequence - last_sequence - 1)
                    last_sequence = sequence
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    sequence, read_start = i, start
                    renderer.update_scene(data, camera=wc.CAM_NAME)
                    rgb = renderer.render().copy()
                    read_end = time.perf_counter_ns()
                captured = time.perf_counter_ns()
                pose = None
                if not usb:
                    mask = wc.red_mask(rgb)
                    masked = time.perf_counter_ns()
                    camera = wc.Pinhole(model, data, wc.CAM_NAME, args.width, args.height)
                    if int(np.count_nonzero(mask)):
                        if previous is None:
                            best, score = wc.fit(camera, mask, topk=args.fit_topk)
                        else:
                            best, score, _ = wc.fit_tracking(camera, mask, previous)
                        if best is not None and score >= 0.6:
                            previous = best
                            pose = np.asarray([best[0], best[1], wc.TABLE_TOP + wc.CUBE_HALF])
                    fitted = time.perf_counter_ns()
                    if driver:
                        command = driver.prepare_frame(i, perceived_center=pose)
                        if command is None:
                            raise RuntimeError(driver.report.estop)
                        targets, fingers = command
                    else:
                        targets = holds
                        distance = None if pose is None else float(np.linalg.norm(
                            0.5 * (data.geom_xpos[g[0]] + data.geom_xpos[g[1]]) - pose))
                        fingers = {"right": grip.update(distance) if args.gripper_mode == "sim-proxy" else .044,
                                   "left": .044}
                    action = controller.set_targets(targets, fingers)
                    mapped = time.perf_counter_ns()
                    for j in range(steps):
                        controller.apply()
                        if j == 0:
                            actuator_stamp = time.perf_counter_ns()
                        mujoco.mj_step(model, data)
                        if j == 0:
                            consumed_stamp = time.perf_counter_ns()
                    finished = time.perf_counter_ns()
                    commands.append(action.copy())
                    measured.append(data.qpos.copy())
                    controls.append(data.ctrl.copy())
                    sim_times.append(float(data.time))
                    poses.append(np.full(3, np.nan) if pose is None else pose.copy())
                    inactive_errors.append(float(np.max(np.abs(data.qpos[specs["left"].qpos] - holds["left"]))))
                else:
                    masked = fitted = mapped = finished = captured
                row = {
                    "tick": i, "source_sequence": sequence, "host_request_ns": start,
                    "host_delivered_ns": captured, "actuator_written_ns": actuator_stamp,
                    "first_physics_consumed_ns": consumed_stamp, "physics_finished_ns": finished,
                    "capture_ms": (captured-start)/1e6, "mask_ms": (masked-captured)/1e6,
                    "fit_ms": (fitted-masked)/1e6, "mapping_ms": (mapped-fitted)/1e6,
                    "control_physics_ms": (finished-mapped)/1e6, "host_e2e_ms": (finished-start)/1e6,
                    "source_read_ms": (read_end-read_start)/1e6,
                    "source_age_ms": max(0, (captured-read_end)/1e6),
                    "perception_valid": int(pose is not None), "npz_row": source_row,
                    "sim_time_s": float(data.time) if data is not None else "",
                }
                if args.save_frames:
                    cv2.imwrite(str(output / "frames" / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if args.preview:
                    cv2.imshow("Member B real-time stream", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        failure = "user_stopped_preview"
                row["tick_work_ms"] = (time.perf_counter_ns() - start) / 1e6
                writer.writerow(row)
                frame_rows.append(row)
                for key in stage:
                    stage[key].append(row[key+"_ms"])
                e2e.append(row["host_e2e_ms"])
                work.append(row["tick_work_ms"])
                reads.append(row["source_read_ms"])
                ages.append(row["source_age_ms"])
                if consumed_stamp:
                    first_consumed.append((consumed_stamp-start)/1e6)
                if failure:
                    break
                # Absolute monotonic schedule avoids accumulating Windows
                # sleep overshoot once per tick; no physics steps are skipped.
                delay = run_start + (i + 1) * period - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        wall_seconds = time.perf_counter() - run_start
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        wall_seconds = time.perf_counter() - locals().get("run_start", warm_start)
    finally:
        if renderer is not None:
            renderer.close()
        if usb is not None:
            usb.close()
        if args.preview:
            cv2.destroyAllWindows()
    budget = period * 1000
    misses = sum(x > budget for x in work)
    completed = len(frame_rows)
    miss_rate = misses / completed if completed else 1.0
    soft_pass = bool(not failure and completed == requested and
                     miss_rate <= args.max_miss_rate and stats(work)["p95"] <= budget)
    verdict = ("FAIL" if not soft_pass else "PASS" if misses == 0 else "PASS_SOFT_REALTIME")
    if driver:
        driver.report.drive_seconds = wall_seconds
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": args.source, "camera": wc.CAM_NAME if wc else usb.metadata,
        "process_priority": args.process_priority,
        "resolution_wh": [args.width, args.height], "requested_frames": args.frames,
        "actual_resolution_wh": [int(rgb.shape[1]), int(rgb.shape[0])] if frame_rows else None,
        "frames": completed, "control_hz": args.hz, "budget_ms": budget,
        "steps_per_tick": steps, "physics_timestep_s": float(model.opt.timestep) if model else None,
        "simulated_s_per_tick": steps * float(model.opt.timestep) if model else None,
        "scene_workspace": CONFIG if model else None,
        "startup_ms_excluded": boot_ms, "warmup_ms_excluded": locals().get("warmup_ms"),
        "warmup_frames": args.warmup, "stages_ms": {k: stats(v) for k,v in stage.items()},
        "end_to_end_ms": stats(e2e), "tick_work_ms": stats(work),
        "first_command_consumed_ms": stats(first_consumed),
        "source_read_ms": stats(reads), "host_frame_age_ms": stats(ages),
        "wall_seconds_including_pacing": wall_seconds,
        "actual_stream_hz": completed/wall_seconds if wall_seconds else None,
        "tick_start_interval_ms": stats([(b["host_request_ns"]-a["host_request_ns"])/1e6
                                          for a,b in zip(frame_rows, frame_rows[1:])]),
        "processing_throughput_hz": 1000/float(np.mean(work)) if work else None,
        "ticks_over_budget": misses, "deadline_miss_rate": miss_rate,
        "allowed_miss_rate": args.max_miss_rate, "budget_verdict": verdict,
        "strict_zero_miss_pass": bool(soft_pass and misses == 0),
        "failure": failure, "perception_valid_frames": sum(r["perception_valid"] for r in frame_rows),
        "skipped_camera_frames_latest_only": skipped, "raw_frames_saved": args.save_frames,
        "inactive_left_max_error_rad": max(inactive_errors, default=None),
        "npz": str(args.npz.resolve()) if args.npz else None,
        "driver": asdict(driver.report) if driver else None,
        "processed_npz_rows": completed if driver else 0,
        "tracking": dict(wc.TRACKING_STATS) if wc else None,
        "scene_manifest": str(args.scene_manifest) if args.scene_manifest else None,
        "manifest_verification_ms_excluded": args.manifest_verification_ms,
        "latency_scope": ("simulated monocular frame request -> perception -> actuator mapping -> full physics control interval"
                          if model else "host request for latest USB frame -> delivery; capture only, no robot control"),
        "excluded": ["camera exposure-to-host latency (unavailable)",
                     "A human landmark extraction and Pose2Arm IK (offline NPZ, not measured)",
                     "monitor display latency", "startup and declared warmup",
                     "buffered CSV write excluded from tick_work_ms, included in actual_stream_hz"],
        "gripper_input": driver.gripper_source if driver else (args.gripper_mode if model else None),
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "mujoco": mujoco.__version__, "opencv": cv2.__version__},
    }
    (output / "latency_log.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if commands:
        np.savez_compressed(output / "actuator_telemetry.npz", action=np.asarray(commands),
                            qpos=np.asarray(measured), ctrl=np.asarray(controls),
                            sim_time=np.asarray(sim_times), perceived_center=np.asarray(poses))
    print(json.dumps({k: report[k] for k in ("source", "frames", "actual_stream_hz", "end_to_end_ms",
                                           "steps_per_tick", "budget_verdict", "failure")}, indent=2), flush=True)
    return 0 if soft_pass else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("simulation", "usb"), default="simulation")
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--gripper-mode", choices=("auto", "human-only", "sim-proxy"), default="auto")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fit-topk", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-miss-rate", type=float, default=0.01)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--high-priority", action="store_true",
                        help="Windows HIGH priority for this process only, never REALTIME class")
    parser.add_argument("--scene-manifest", type=Path,
                        help="verify an immutable B scene/runtime manifest before starting")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "outputs" / datetime.now().strftime("%Y%m%d") / "member_b_realtime")
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0 or args.width <= 0 or args.height <= 0 or args.fit_topk <= 0:
        parser.error("invalid frame count, warmup, size or fit-topk")
    if not np.isfinite(args.hz) or args.hz <= 0 or not 0 <= args.max_miss_rate <= 1:
        parser.error("invalid frequency or miss rate")
    if args.source == "usb" and args.npz:
        parser.error("USB mode captures only; NPZ replay is not inference from a live human image")
    args.manifest_verification_ms = 0
    if args.scene_manifest:
        from freeze_member_b_scene import verify
        verification_started = time.perf_counter()
        verify(args.scene_manifest)
        args.manifest_verification_ms = (time.perf_counter() - verification_started) * 1000
    args.process_priority = request_high_priority() if args.high_priority else "normal"
    try:
        return run(args)
    except Exception as exc:
        args.outdir.mkdir(parents=True, exist_ok=True)
        payload = {"source": args.source, "frames": 0, "budget_verdict": "FAIL",
                   "failure": f"initialization: {type(exc).__name__}: {exc}"}
        (args.outdir / "latency_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
