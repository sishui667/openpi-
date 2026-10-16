"""Compare physical table heights with identical blind RGB-D grasp fixtures.

Runs isolated processes. Only fixture setup receives X/Y/yaw; the unchanged
controller gets rendered cameras. Neither config nor truth is used as a target.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
POSITIONS = [(0.45, 0.0), (0.33, 0.0), (0.57, 0.0),
             (0.45, -0.22), (0.45, 0.22), (0.33, -0.22),
             (0.33, 0.22), (0.57, -0.22), (0.57, 0.22)]


def source_fingerprint():
    digest = hashlib.sha256()
    for name in ("demo_grasp_viewer.py", "task_workspace.py",
                 "visual_grasp_planner.py", "visual_observation_recovery.py",
                 "handeye_grasp_red_cube.py"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def run_case(folder, config_path, height, position, yaw, fingerprint, resume):
    x, y = position
    name = f"h{height*1000:.0f}_x{x*1000:.0f}_y{y*1000:+.0f}_r{yaw:.0f}"
    report = folder / f"{name}.json"
    row_path = folder / f"{name}.row.json"
    config_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if resume and row_path.exists():
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if (row.get("source_fingerprint") == fingerprint
                and row.get("config_fingerprint") == config_fingerprint):
            return row
    env = dict(os.environ, OPENARM_TASK_WORKSPACE_CONFIG=str(config_path),
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.perf_counter()
    previous_report_stamp = report.stat().st_mtime_ns if report.exists() else None
    result = {}
    try:
        process = subprocess.run([
            sys.executable, "-u", str(ROOT / "demo_grasp_viewer.py"),
            "--check", "--cube-test-pose", str(x), str(y), str(yaw),
            "--post-lift-s", "1.5", "--report", str(report),
        ], cwd=ROOT, env=env, capture_output=True, text=True,
           encoding="utf-8", errors="replace", timeout=210)
        output = process.stdout + process.stderr
        returncode = process.returncode
        if report.exists() and report.stat().st_mtime_ns != previous_report_stamp:
            result = json.loads(report.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        output = f"TIMEOUT 210s\n{exc.stdout!r}\n{exc.stderr!r}"
        returncode = -1
    report.with_suffix(".log").write_text(output, encoding="utf-8")
    quality = result.get("contact_quality_at_grip") or {}
    lifted = bool((result.get("lift_mm") or 0) > 30
                  and (result.get("success_continuous_s") or 0) >= 0.5
                  and (result.get("contacts_final") or 0) > 0
                  and quality.get("opposing_faces", False))
    verdict = result.get("verdict", "ERROR")
    if verdict == "PASS" and returncode != 0:
        verdict = "ERROR"
    failure = result.get("failure_reason")
    if verdict != "PASS" and not failure:
        if lifted and (result.get("episode_sim_s") or 0) > 15:
            failure = "lifted_but_action_timeout"
        elif returncode == -1:
            failure = "computation_timeout"
        elif result:
            failure = "grasp_or_contact_quality_failed"
        else:
            failure = "unhandled_error_see_log"
    row = {
        "case": name, "table_top_m": height, "fixture_xy_yaw": [x, y, yaw],
        "verdict": verdict, "lifted_with_opposing_faces": lifted,
        "failure_reason": failure, "returncode": returncode,
        "lift_mm": result.get("lift_mm"),
        "rotation_max_deg": result.get("orientation_max_deg"),
        "episode_sim_s": result.get("episode_sim_s"),
        "perception_error_mm_evaluation_only": result.get("perception_error_mm_evaluation_only"),
        "controller_source": result.get("controller_cube_pose_source"),
        "arm": result.get("selected_arm"), "wall_s": time.perf_counter() - started,
        "report": str(report), "source_fingerprint": fingerprint,
        "config_fingerprint": config_fingerprint,
    }
    row_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heights", type=float, nargs="+", default=[0.356, 0.4, 0.45, 0.5])
    parser.add_argument("--yaws", type=float, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="centre only, same physical table")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--x-values", type=float, nargs="+")
    parser.add_argument("--y-values", type=float, nargs="+")
    parser.add_argument("--workspace-config", type=Path,
                        help="use an existing physical config (one matching height only)")
    args = parser.parse_args()
    folder = (args.folder or ROOT / "outputs" / time.strftime("%Y%m%d") /
              ("table_height_smoke" if args.smoke else "table_height_grid")).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    positions = POSITIONS[:1] if args.smoke else POSITIONS
    if bool(args.x_values) != bool(args.y_values):
        parser.error("--x-values and --y-values must be provided together")
    if args.x_values:
        positions = [(x, y) for x in args.x_values for y in args.y_values]
    fingerprint = source_fingerprint()
    configs = {}
    for height in args.heights:
        config = {
            "table_center_m": [0.45, 0.0, height - 0.02],
            "table_half_size_m": [0.20, 0.30, 0.02],
            "placement_x_m": [0.33, 0.57], "placement_y_m": [-0.22, 0.22],
            "cube_half_size_m": 0.0288,
            "note": "Experimental 40x60cm table; common grid, not a reach guarantee. Default untouched.",
        }
        if args.workspace_config:
            config = json.loads(args.workspace_config.read_text(encoding="utf-8"))
            actual_height = config["table_center_m"][2] + config["table_half_size_m"][2]
            if len(args.heights) != 1 or abs(actual_height - height) > 1e-9:
                parser.error("--workspace-config requires a single matching height")
        path = folder / f"workspace_h{height*1000:.0f}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        configs[height] = path
    rows = []
    jobs = [(height, position, yaw) for position in positions
            for yaw in args.yaws for height in args.heights]
    manifest = {"source_fingerprint": fingerprint, "heights_m": args.heights,
                "positions_xy_m": positions, "yaws_deg": args.yaws,
                "jobs": len(jobs), "workers": args.workers,
                "default_config_modified": False}
    manifest["workspace_configs"] = {str(h): json.loads(p.read_text(encoding="utf-8"))
                                     for h, p in configs.items()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, folder, configs[h], h, pos, yaw,
                               fingerprint, args.resume) for h, pos, yaw in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: r["case"])
            (folder / "summary.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"done": len(rows), "total": len(jobs), **row}, ensure_ascii=False), flush=True)
    totals = [{"table_top_m": height,
               "cases": sum(r["table_top_m"] == height for r in rows),
               "pass": sum(r["table_top_m"] == height and r["verdict"] == "PASS" for r in rows),
               "lifted": sum(r["table_top_m"] == height and r["lifted_with_opposing_faces"] for r in rows)}
              for height in args.heights]
    (folder / "totals.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2), flush=True)
    # A comparison includes failed attempts; recording all cases is completion.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
