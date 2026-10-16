"""Blind fixture regressions: positions/yaws are not passed to the planner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CASES = [
    ("near_right45", 0.365, -0.060, 45),
    ("near_right0", 0.330, -0.080, 0),
    ("far_right30", 0.480, -0.160, 30),
    ("far_left45", 0.505, 0.140, 45),
    ("left_edge0", 0.430, 0.200, 0),
    ("front_right15", 0.550, -0.120, 15),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", choices=("compact-table", "legacy"),
                        default="compact-table")
    args = parser.parse_args()
    cases = CASES
    if args.case_set == "compact-table":
        from task_workspace import TABLE_X, TABLE_Y, TABLE_TOP
        cases = [(f"corner_{ix}_{iy}_{yaw}", x, y, yaw)
                 for ix, x in enumerate(TABLE_X)
                 for iy, y in enumerate(TABLE_Y) for yaw in (0, 45)]
        cases.append(("centre_30", sum(TABLE_X) / 2, 0, 30))
    folder = (ROOT / "outputs" / time.strftime("%Y%m%d") /
              (f"task_table_regression_h{TABLE_TOP*1000:.0f}" if args.case_set == "compact-table"
               else "legacy_large_table_positions"))
    folder.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, x, y, yaw in cases:
        report = folder / f"{name}.json"
        process = subprocess.run([
            sys.executable, "-u", str(ROOT / "demo_grasp_viewer.py"),
            "--check", "--cube-test-pose", str(x), str(y), str(yaw),
            "--post-lift-s", "1.5", "--report", str(report),
        ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
           errors="replace", timeout=240)
        report.with_suffix(".log").write_text(
            process.stdout + process.stderr, encoding="utf-8")
        result = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        row = {"case": name, "fixture": [x, y, yaw],
               "returncode": process.returncode,
               "verdict": result.get("verdict", "ERROR"),
               "lift_mm": result.get("lift_mm"),
               "rotation_deg": result.get("orientation_max_deg"),
               "episode_s": result.get("episode_sim_s"),
               "quality": result.get("contact_quality_at_grip")}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (folder / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(row["verdict"] == "PASS" for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
