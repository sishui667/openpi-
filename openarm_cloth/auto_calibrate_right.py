# -*- coding: utf-8 -*-
"""Convenience entry point for automatic right-arm wrist-camera calibration."""
from __future__ import annotations

from auto_calibrate_left import build_parser, collect_wrist, solve_wrist_session


def main() -> None:
    args = build_parser(default_arm="right").parse_args()
    if args.session is not None:
        session = args.session.resolve()
    else:
        session = collect_wrist(args, "right")
    if not args.no_solve:
        solve_wrist_session(session, "right", free_distortion=args.free_distortion)
    else:
        print(f"[自动采集] 未执行求解，数据目录: {session}")


if __name__ == "__main__":
    main()
