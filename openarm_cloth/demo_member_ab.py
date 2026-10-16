"""A human-motion NPZ -> B physics viewer. This is NOT a cube-grasp policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from member_b_scene import build_scene, ROOT
from pose2arm_driver import Pose2ArmDriver


def run(args):
    model,data=build_scene()
    driver=Pose2ArmDriver(model,data,args.npz,gripper_mode="human-only",control_hz=20)
    count=min(args.frames or driver.report.frames,driver.report.frames)
    print("OFFLINE A/B replay: human gripper signals, no object-coordinate control.",flush=True)
    print("SPACE pause/resume | R restart | Q or close window exits. Window stays open after replay.",flush=True)
    if args.check:
        for i in range(count):
            driver.step_frame(i)
            if driver.report.estop: raise RuntimeError(driver.report.estop)
    else:
        import mujoco.viewer
        flags={"paused":False,"restart":False,"quit":False}
        def key(code):
            if code==32: flags["paused"]=not flags["paused"]
            elif code in (82,114): flags["restart"]=True
            elif code in (81,113): flags["quit"]=True
        initial=data.qpos.copy()
        i=0
        targets={s:data.qpos[spec.qpos].copy() for s,spec in driver.specs.items()}
        fingers={"left":.044,"right":.044}
        with mujoco.viewer.launch_passive(model,data,key_callback=key) as view:
            view.cam.lookat[:]=[.43,0,.43]
            view.cam.distance=1.25
            view.cam.azimuth=180
            view.cam.elevation=-28
            while view.is_running() and not flags["quit"]:
                start=time.perf_counter()
                if flags["restart"]:
                    mujoco.mj_resetData(model,data)
                    data.qpos[:]=initial
                    mujoco.mj_forward(model,data)
                    driver=Pose2ArmDriver(model,data,args.npz,gripper_mode="human-only",control_hz=20)
                    i=0;flags["restart"]=False
                    targets={s:data.qpos[spec.qpos].copy() for s,spec in driver.specs.items()}
                    fingers={"left":.044,"right":.044}
                if i<count and not flags["paused"]:
                    command=driver.prepare_frame(i)
                    if command is None: raise RuntimeError(driver.report.estop)
                    targets,fingers=command
                    i+=1
                    if i%20==0 or i==count:
                        print(f"frame {i}/{count} | right finger target {fingers['right']:.4f} m",flush=True)
                driver.controller.set_targets(targets,fingers)
                for _ in range(driver.steps_per_frame):
                    driver.controller.apply();mujoco.mj_step(model,data)
                view.sync()
                remaining=.05-(time.perf_counter()-start)
                if remaining>0: time.sleep(remaining)
    result={"scope":"A NPZ actuator playback, NOT grasp-success evaluation",
            "frames_requested":count,"gripper_source":driver.gripper_source,"driver":asdict(driver.report)}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz",type=Path,default=ROOT/"deliverables/member_AB_20261016/a_hybrid_orig_20hz.npz")
    p.add_argument("--frames",type=int)
    p.add_argument("--check",action="store_true",help="headless physics check, no timing/grasp claim")
    p.add_argument("--report",type=Path)
    args=p.parse_args()
    if args.frames is not None and args.frames<=0: p.error("frames must be positive")
    run(args)


if __name__=="__main__": main()
