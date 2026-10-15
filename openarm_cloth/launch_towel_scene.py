# -*- coding: utf-8 -*-
"""Open the latest verified OpenArm towel scene in a ready-to-fold pose.

Examples (from D:\\openpi\\openarm_cloth, inside the openpi environment):
    python launch_towel_scene.py
    python launch_towel_scene.py --dynamic
    python launch_towel_scene.py --scene simulation/v1/folding_scene_V25_4_TOWEL_SUPPORT_02.xml
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "simulation" / "v1"
SCENE_PATTERN = "folding_scene_V25_4_TOWEL_SUPPORT_*.xml"


def latest_scene() -> Path:
    candidates = list(SCENE_DIR.glob(SCENE_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            "No verified towel scene found. Run: "
            "python build_towel_scene_v25_4.py"
        )

    def version(path: Path) -> int:
        match = re.search(r"_(\d+)\.xml$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Explicit towel XML; default is the newest V25.4 verified output.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Advance the simulation. Default is a static ready-pose preview.",
    )
    args = parser.parse_args()

    scene = args.scene or latest_scene()
    if not scene.is_absolute():
        scene = (ROOT / scene).resolve()
    if not scene.is_file():
        parser.error(f"Scene XML not found: {scene}")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "towel_ready")
    if key_id < 0:
        raise RuntimeError(
            f"{scene.name} has no towel_ready keyframe; rebuild it with "
            "build_towel_scene_v25_4.py."
        )
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Keep the robot at the saved ready pose when dynamic stepping is enabled.
    if args.dynamic:
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.ctrl[actuator_id] = data.qpos[qpos_address]

    print(f"[SCENE] {scene}")
    print(f"[READY] towel_ready keyframe loaded; nq={model.nq}, nflex={model.nflex}")
    print("[MODE] dynamic" if args.dynamic else "[MODE] static preview")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.38]
        viewer.cam.distance = 1.05
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        viewer.sync()
        while viewer.is_running():
            if args.dynamic:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01 if args.dynamic else 0.05)


if __name__ == "__main__":
    main()
