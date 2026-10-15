"""Desktop launcher: open the interactive grasp demo without a console."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime
import traceback


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = root / "outputs" / datetime.now().strftime("%Y%m%d")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"desktop_demo_{stamp}.log"
    report_path = output / f"desktop_demo_{stamp}.json"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        sys.argv = [
            str(root / "demo_grasp_viewer.py"),
            "--interactive-place-s", "86400",
            "--preview-initial-s", "5",
            "--post-lift-s", "1.5",
            "--report", str(report_path),
        ]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(f"Simulation exited: {exc.code}")
                ctypes.windll.user32.MessageBoxW(
                    None, f"仿真运行失败，请查看日志：\n{log_path}", "抓取仿真", 0x10
                )
        except Exception:
            traceback.print_exc()
            log.flush()
            ctypes.windll.user32.MessageBoxW(
                None, f"仿真运行出现异常，请查看日志：\n{log_path}", "抓取仿真", 0x10
            )


if __name__ == "__main__":
    main()
