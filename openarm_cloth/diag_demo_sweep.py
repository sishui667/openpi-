# -*- coding: utf-8 -*-
"""Sweep the REAL demo over cube positions and tabulate what fails.

My reimplementation of the demo's move sequence succeeds at the blind pose
while the demo itself fails there, so the reimplementation is not trustworthy
as a stand-in.  This drives ``demo_grasp_viewer.main`` directly -- the same
entry point ``--check`` uses -- and reads the report it writes.  No parallel
copy of the logic, so there is nothing left to diverge.

    python diag_demo_sweep.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import demo_grasp_viewer as dgv                                              # noqa: E402

OUT = ROOT / "outputs" / "20261014" / "sweep"


def face_of(point_mm):
    near = [i for i, v in enumerate(point_mm) if abs(abs(v) - dgv.CUBE_HALF * 1000) < 3.0]
    return "xyz"[near[0]] if len(near) == 1 else ("edge" if len(near) == 2 else "?")


def one(x, y, yaw):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"x{x:.2f}_y{y:+.2f}_a{yaw:.0f}"
    rp = OUT / f"{tag}.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = dgv.main(check_only=True, post_lift_s=1.5, report_path=str(rp),
                          test_cube_pose=(x, y, yaw))
        except Exception as exc:                                   # noqa: BLE001
            print(f"{tag}  CRASH {type(exc).__name__}: {exc}")
            return
    if not rp.exists():
        print(f"{tag}  no report (rc={rc}); tail: "
              f"{buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else ''}")
        return
    r = json.loads(rp.read_text(encoding="utf-8"))
    pts = r.get("contact_points_object_mm") or []
    faces = "/".join(face_of(p) for p in pts) or "-"
    print(f"{tag:<18} {r.get('verdict','?'):<5} contacts {r.get('contacts_at_grip','?')} "
          f"force {r.get('grip_force_n', 0):6.3f} N  lift {r.get('lift_mm', 0):+7.1f} mm  "
          f"jaw-obj {r.get('jaw_object_distance_final_mm', 0):6.1f} mm  faces {faces}")


def main():
    print("sweeping the real demo_grasp_viewer.main (headless)\n")
    for x in (0.36, 0.40, 0.44, 0.48, 0.52, 0.56):
        one(x, 0.0, 0.0)
    print()
    for y in (-0.08, 0.08):
        one(0.43, y, 0.0)
    print()
    for yaw in (30.0, 60.0):
        one(0.43, 0.03, yaw)
    print(f"\nreports in {OUT}")


if __name__ == "__main__":
    main()
