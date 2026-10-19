"""Live two-arm human following; independent fists, local video/test recording."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import ctypes
from datetime import datetime
import hashlib
import json
import time

import cv2
import mujoco
from mujoco import viewer as mjviewer
import numpy as np

from human_teleop_core import ROOT
from live_motion_scene import build_motion_scene
from live_bimanual_stream import BimanualHumanStream
from bimanual_teleop import BimanualController,SIDES
from human_gripper_gestures import gesture_diagnostics
from live_test_dataset import TestDatasetRecorder,array_or_nan
from live_test_cycle import finish_test_cycle
from handeye_grasp_red_cube import finger_joint_indices

WINDOW='LIVE BOTH ARMS | LOCAL RECORDING | C calibrate, T wrist mode, SPACE pause, Q exit'


def run(args):
    cv2.setNumThreads(1)
    timer=os.name=='nt' and ctypes.windll.winmm.timeBeginPeriod(1)==0
    out=ROOT/'outputs'/'live_bimanual'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    recorder=TestDatasetRecorder(out/'dataset',dict(settings=dict(vars(args)),created_local=datetime.now().isoformat(),
                              arms=list(SIDES),side_array_order=list(SIDES),aggregate_validity='any side',
                              identity='human anatomical left->robot left, right->right; display mirror only'),save_video=args.save_video)
    print(f'LIVE BIMANUAL | local video {args.save_video} | dataset {recorder.path}',flush=True)
    stream=BimanualHumanStream(args.camera,args.pose_model,(args.width,args.height))
    flags=dict(paused=False,quit=False,reset=False,mode=args.wrist_mode)
    def key(code):
        if code==32: flags['paused']=not flags['paused']
        elif code in (81,113,27): flags['quit']=True
        elif code in (67,99): flags['reset']=True
        elif code in (84,116): flags['mode']='full' if flags['mode']=='twist' else 'twist';flags['reset']=True
    samples={s:[] for s in SIDES};frames=0;last_seq=-1;failure=None
    start=time.perf_counter();last_print=start
    try:
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,960,540)
        splash=np.full((540,960,3),55,np.uint8)
        cv2.putText(splash,'Starting BOTH arms and local webcam...',(20,240),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        cv2.imshow(WINDOW,splash);cv2.waitKey(1)
        model,data=build_motion_scene();control=BimanualController(model,data,args.wrist_mode)
        model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.8,.8,.8]
        recorder.update_metadata(robot_joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)],
                                 actuator_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)],
                                 timestep_s=float(model.opt.timestep),clock='host perf_counter_ns; not optical exposure',
                                 action_layout='left joint7+finger2, right joint7+finger2; radians/metres',
                                 source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                                ('live_bimanual_human.py','live_bimanual_stream.py','bimanual_teleop.py','human_teleop_core.py','human_gripper_gestures.py','live_motion_scene.py')})
        steps=round(.05/model.opt.timestep)
        finger_indices={s:finger_joint_indices(model,s)[0] for s in SIDES}
        with (out/'latency.csv').open('w',newline='',encoding='utf-8') as f,mjviewer.launch_passive(model,data,key_callback=key) as view:
            writer=csv.writer(f)
            writer.writerow(['frame','source_seq','fresh_source','left_pose_valid','right_pose_valid','left_hand_valid','right_hand_valid',
                             'left_closed','right_closed','host_read_to_first_step_ms','inference_ms','prepare_ms','physics_ms','ui_ms',
                             'left_upper_deg','left_fore_deg','left_palm_deg','right_upper_deg','right_fore_deg','right_palm_deg'])
            view.cam.lookat[:]=[.15,0,.52];view.cam.distance=1.55;view.cam.azimuth=180;view.cam.elevation=-25
            print('BOTH WINDOWS OPEN | anatomical LEFT and RIGHT independently follow; FIST closes, OPEN PALM opens.',flush=True)
            while view.is_running() and not flags['quit']:
                if (out/'STOP').exists(): break
                tick=time.perf_counter();now=time.perf_counter_ns();obs=stream.snapshot()
                fresh=obs is not None and obs['seq']!=last_seq
                if flags['reset']: control.reset(flags['mode']);flags['reset']=False
                action=control.update(obs,now,flags['paused'])
                control.actuators.apply();first_step=time.perf_counter_ns();mujoco.mj_step(model,data)
                for _ in range(steps-1): control.actuators.apply();mujoco.mj_step(model,data)
                physics_end=time.perf_counter_ns()
                if not np.isfinite(data.qpos).all(): raise RuntimeError('non-finite robot state')
                errors=control.errors()
                frame=splash.copy() if obs is None else cv2.flip(cv2.resize(obs['image'],(960,540)),1)
                age=(now-obs['read_end'])/1e6 if obs else float('nan')
                inference=(obs['done']-obs['infer_start'])/1e6 if obs else float('nan')
                response=(first_step-obs['read_end'])/1e6 if obs else float('nan')
                lines=[('PAUSED' if flags['paused'] else 'LIVE BOTH ARMS')+' | WRIST '+flags['mode'].upper(),
                       'Your LEFT -> robot LEFT | your RIGHT -> robot RIGHT',
                       'C: calibrate both  T: twist/full  SPACE: pause  Q: exit',
                       'LOCAL '+('VIDEO + KEYPOINT RECORDING' if args.save_video else 'KEYPOINT RECORDING')]
                for i,s in enumerate(SIDES):
                    arm=obs['arms'][s] if obs else None
                    gesture=arm['fist']['gesture'] if control.hand_valid[s] and arm['fist'] else 'HAND MISSING'
                    lines.append(f'{s.upper()}: {"ARM OK" if control.valid[s] else "ARM HOLD"} | {gesture} | {"CLOSED" if control.closed[s] else "OPEN"}')
                    wrist=np.degrees(data.qpos[control.specs[s].qpos][4:])
                    lines.append(f'  upper/fore error {errors[i,0]:.0f}/{errors[i,1]:.0f} deg | wrist '+','.join(f'{x:.0f}' for x in wrist))
                lines.append(f'Input age {age:.0f} ms | inference {inference:.0f} ms | source {obs["seq"] if obs else -1}')
                if stream.error: lines=['RECOGNITION ERROR: see launch log','Q: exit']
                if recorder.error: lines.append('RECORDING ERROR - '+recorder.error[:70])
                for i,line in enumerate(lines):
                    cv2.putText(frame,line,(12,28+29*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3)
                    cv2.putText(frame,line,(12,28+29*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cv2.imshow(WINDOW,frame);code=cv2.waitKey(1)&255
                if code!=255: key(code)
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1: flags['quit']=True
                if hasattr(view,'set_texts'): view.set_texts((None,None,'LIVE BIMANUAL',f'LEFT {control.valid["left"]} | RIGHT {control.valid["right"]} | Q exit'))
                view.sync();ui_end=time.perf_counter_ns()
                writer.writerow([frames,obs['seq'] if obs else -1,int(fresh),*[int(control.valid[s]) for s in SIDES],
                                 *[int(control.hand_valid[s]) for s in SIDES],*[control.closed[s] for s in SIDES],response,inference,
                                 (first_step-now)/1e6,(physics_end-first_step)/1e6,(ui_end-physics_end)/1e6,*errors.reshape(-1)])
                row=dict(control_frame=np.asarray(frames),source_seq=np.asarray(obs['seq'] if obs else -1),fresh_source=np.asarray(fresh),
                         capture_read_end_ns=np.asarray(obs['read_end'] if obs else 0,dtype=np.int64),
                         inference_done_ns=np.asarray(obs['done'] if obs else 0,dtype=np.int64),
                         first_physics_step_ns=np.asarray(first_step,dtype=np.int64),physics_end_ns=np.asarray(physics_end,dtype=np.int64),
                         sim_time=np.asarray(data.time),startup=np.asarray(False),paused=np.asarray(flags['paused']),
                         pose_raw=array_or_nan(obs['pose_raw'] if obs else None,(33,4)),
                         pose_filtered=array_or_nan(obs['pose_filtered'] if obs else None,(33,4)),
                         pose_norm=array_or_nan(obs['pose_norm'] if obs else None,(33,4)),
                         action=action.copy(),robot_qpos=data.qpos.copy(),robot_qvel=data.qvel.copy(),robot_ctrl=data.ctrl.copy(),
                         pose_valid=np.asarray(any(control.valid.values())),hand_valid=np.asarray(any(control.hand_valid.values())),
                         side_pose_valid=np.asarray([control.valid[s] for s in SIDES]),side_hand_valid=np.asarray([control.hand_valid[s] for s in SIDES]),
                         side_closed=np.asarray([control.closed[s] for s in SIDES]),gripper_closed=np.asarray(any(control.closed.values())),
                         side_angle_errors_deg=errors.copy(),wrist_mode=np.asarray(flags['mode']),
                         read_to_step_ms=np.asarray(response),inference_ms=np.asarray(inference))
                for s in SIDES:
                    arm=obs['arms'][s] if obs else None
                    for field in ('raw','filtered','norm'): row[f'{s}_hand_{field}']=array_or_nan(arm[f'hand_{field}'] if arm else None,(21,3))
                    row[f'{s}_curl_scores']=np.asarray(arm['fist']['scores'] if arm and arm['fist'] else [float('nan')]*4)
                    row.update({f'{s}_{k}':v for k,v in gesture_diagnostics(arm['fist'] if arm else None).items()})
                    solver=control.solvers[s]
                    row[f'{s}_upper_target']=array_or_nan(solver.upper,(3,))
                    row[f'{s}_fore_target']=array_or_nan(solver.fore,(3,))
                    row[f'{s}_rotation_target']=array_or_nan(solver.rotation,(3,3))
                recorder.submit(row,obs['raw_image'] if fresh and args.save_video else None)
                if fresh:
                    last_seq=obs['seq']
                    if not flags['paused']:
                        for s in SIDES:
                            if control.valid[s]: samples[s].append(response)
                frames+=1
                if tick-last_print>2:
                    print(f'LIVE BOTH | seq {last_seq} | arms {control.valid} | hands {control.hand_valid} | CLOSED {control.closed} | source-to-step {response:.1f} ms',flush=True)
                    f.flush();last_print=tick
                if stream.error: raise RuntimeError(stream.error)
                if args.seconds and time.perf_counter()-start>=args.seconds: break
                remaining=.05-(time.perf_counter()-tick)
                if remaining>0: time.sleep(remaining)
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';raise
    finally:
        if 'control' in locals(): control.close()
        stream.close();cv2.destroyAllWindows();recorder.update_metadata(camera=stream.metadata,run_error=failure);recorder.close()
        if timer: ctypes.windll.winmm.timeEndPeriod(1)
        def stats(values):
            return None if not values else dict(n=len(values),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),max_ms=float(np.max(values)))
        report=dict(scope='LIVE bimanual following; not grasp success or perception ground truth',
                    left_latency=stats(samples['left']),right_latency=stats(samples['right']),
                    timing='host camera return to first physics step, not optical/settling latency',
                    dataset=str(recorder.path),rows=recorder.saved_rows,video_frames=recorder.video_frames,
                    recording_dropped_rows=recorder.dropped_rows,recording_error=recorder.error,run_error=failure)
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
        finish_test_cycle(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',type=int,default=0);p.add_argument('--pose-model',choices=('lite','full'),default='lite')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--wrist-mode',choices=('twist','full'),default='twist')
    p.add_argument('--save-video',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--runtime',choices=('split','legacy'),default='split')
    args=p.parse_args()
    if args.runtime=='split':
        from live_bimanual_runtime import run_split
        run_split(args)
    else: run(args)
