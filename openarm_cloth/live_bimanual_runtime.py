"""Split physics/GUI runtime. Only the control worker owns live MjData.

GUI uses a separate MjData copy, so slow OpenCV/GL rendering cannot turn each
50 ms physics interval into 130 ms of wall time. No physical steps are skipped.
"""
import csv
import ctypes
from datetime import datetime
import hashlib
import json
import os
import threading
import time
import traceback

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


class BimanualControlLoop:
    def __init__(self,model,data,stream,recorder,out,wrist_mode='twist'):
        self.model=model;self.data=data;self.stream=stream;self.recorder=recorder;self.out=out
        self.control=BimanualController(model,data,wrist_mode,direction_tau=0.,arm_speed=2.5)
        self.lock=threading.Lock();self.state=None;self.error=None
        self.stop=threading.Event();self.paused=False;self.reset_requested=False;self.wrist_mode=wrist_mode
        self.frames=0;self.periods=[];self.compute=[];self.latencies={s:[] for s in SIDES}
        self.thread=threading.Thread(target=self._run,name='robot-physics-20hz',daemon=True)

    def start(self): self.thread.start()

    def snapshot(self):
        with self.lock: return self.state

    def _row(self,obs,fresh,first,end,errors,response,inference):
        c=self.control;data=self.data
        row=dict(control_frame=np.asarray(self.frames),source_seq=np.asarray(obs['seq'] if obs else -1),fresh_source=np.asarray(fresh),
                 capture_read_end_ns=np.asarray(obs['read_end'] if obs else 0,dtype=np.int64),
                 inference_done_ns=np.asarray(obs['done'] if obs else 0,dtype=np.int64),
                 first_physics_step_ns=np.asarray(first,dtype=np.int64),physics_end_ns=np.asarray(end,dtype=np.int64),
                 sim_time=np.asarray(data.time),startup=np.asarray(False),paused=np.asarray(self.paused),
                 pose_raw=array_or_nan(obs['pose_raw'] if obs else None,(33,4)),
                 pose_filtered=array_or_nan(obs['pose_filtered'] if obs else None,(33,4)),
                 pose_norm=array_or_nan(obs['pose_norm'] if obs else None,(33,4)),
                 action=c.actuators.action.copy(),robot_qpos=data.qpos.copy(),robot_qvel=data.qvel.copy(),robot_ctrl=data.ctrl.copy(),
                 pose_valid=np.asarray(any(c.valid.values())),hand_valid=np.asarray(any(c.hand_valid.values())),
                 side_pose_valid=np.asarray([c.valid[s] for s in SIDES]),side_hand_valid=np.asarray([c.hand_valid[s] for s in SIDES]),
                 side_closed=np.asarray([c.closed[s] for s in SIDES]),gripper_closed=np.asarray(any(c.closed.values())),
                 side_angle_errors_deg=errors.copy(),wrist_mode=np.asarray(self.wrist_mode),
                 read_to_step_ms=np.asarray(response),inference_ms=np.asarray(inference))
        avoidance=c.coordinator.status
        row['avoidance_mode']=np.asarray(avoidance['mode'])
        row['avoidance_clearance_m']=np.asarray(avoidance['clearance_m'])
        row['avoidance_control_ms']=np.asarray(avoidance['planning_ms'])
        row['avoidance_route_id']=np.asarray(avoidance['route_id'])
        for s in SIDES:
            arm=obs['arms'][s] if obs else None
            for field in ('raw','filtered','norm'): row[f'{s}_hand_{field}']=array_or_nan(arm[f'hand_{field}'] if arm else None,(21,3))
            row[f'{s}_curl_scores']=np.asarray(arm['fist']['scores'] if arm and arm['fist'] else [float('nan')]*4)
            row.update({f'{s}_{k}':v for k,v in gesture_diagnostics(arm['fist'] if arm else None).items()})
            solver=c.solvers[s]
            row[f'{s}_upper_target']=array_or_nan(solver.upper,(3,));row[f'{s}_fore_target']=array_or_nan(solver.fore,(3,))
            row[f'{s}_rotation_target']=array_or_nan(solver.rotation,(3,3))
            row[f'{s}_human_palm_normal']=array_or_nan(arm['pose'].get('palm_normal') if arm and arm['pose'] else None,(3,))
            row[f'{s}_robot_palm_normal']=data.xmat[c.specs[s].body_id].reshape(3,3)[:,0].copy()
            # Status belongs to the new observed pose, not a repeated control
            # tick. Keep rejection and physical-limit saturation distinguishable.
            observed=fresh and not self.paused and c.valid[s] and c.hand_valid[s]
            row[f'{s}_wrist_rejected']=np.asarray(bool(observed and solver.absolute_rejected_this_frame))
            row[f'{s}_wrist_limited']=np.asarray(bool(observed and solver.twist_limited))
            row[f'{s}_wrist_rejected_total']=np.asarray(solver.absolute_rejected_total)
            row[f'{s}_wrist_reacquisitions']=np.asarray(solver.absolute_reacquisitions)
        return row

    def _run(self):
        steps=round(.05/self.model.opt.timestep);last_seq=-1;previous_tick=None;deadline=time.perf_counter();last_print=deadline
        try:
            with (self.out/'latency.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                writer.writerow(['frame','source_seq','fresh_source','left_pose_valid','right_pose_valid','left_hand_valid','right_hand_valid',
                                 'left_closed','right_closed','host_read_to_first_step_ms','inference_ms','prepare_ms','physics_ms',
                                 'control_compute_ms','control_period_ms','sim_time_s',
                                 'left_upper_deg','left_fore_deg','left_palm_deg','right_upper_deg','right_fore_deg','right_palm_deg'])
                while not self.stop.is_set():
                    if (self.out/'STOP').exists(): break
                    tick=time.perf_counter();now=time.perf_counter_ns();obs=self.stream.snapshot()
                    period=float('nan') if previous_tick is None else (tick-previous_tick)*1000;previous_tick=tick
                    if np.isfinite(period): self.periods.append(period)
                    fresh=obs is not None and obs['seq']!=last_seq
                    if self.reset_requested: self.control.reset(self.wrist_mode);self.reset_requested=False
                    self.control.update(obs,now,self.paused)
                    self.control.actuators.apply();first=time.perf_counter_ns();mujoco.mj_step(self.model,self.data)
                    for _ in range(steps-1): self.control.actuators.apply();mujoco.mj_step(self.model,self.data)
                    end=time.perf_counter_ns()
                    if not np.isfinite(self.data.qpos).all(): raise RuntimeError('non-finite robot state')
                    errors=self.control.errors();response=(first-obs['read_end'])/1e6 if obs else float('nan')
                    inference=(obs['done']-obs['infer_start'])/1e6 if obs else float('nan')
                    row=self._row(obs,fresh,first,end,errors,response,inference)
                    self.recorder.submit(row,obs['raw_image'] if fresh and self.recorder.save_video else None)
                    state=dict(frame=self.frames,obs=obs,qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),ctrl=self.data.ctrl.copy(),
                               sim_time=float(self.data.time),errors=errors,valid=dict(self.control.valid),hand_valid=dict(self.control.hand_valid),
                               closed=dict(self.control.closed),response=response,inference=inference,paused=self.paused,wrist_mode=self.wrist_mode,
                               avoidance=dict(self.control.coordinator.status),
                               wrist_rejected={s:bool(self.control.hand_valid[s] and self.control.valid[s] and self.control.solvers[s].absolute_rejected_this_frame) for s in SIDES},
                               wrist_limited={s:bool(self.control.hand_valid[s] and self.control.valid[s] and self.control.solvers[s].twist_limited) for s in SIDES})
                    with self.lock: self.state=state
                    if fresh:
                        last_seq=obs['seq']
                        if not self.paused:
                            for s in SIDES:
                                if self.control.valid[s]: self.latencies[s].append(response)
                    compute=(time.perf_counter()-tick)*1000;self.compute.append(compute)
                    writer.writerow([self.frames,last_seq,int(fresh),*[int(self.control.valid[s]) for s in SIDES],
                                     *[int(self.control.hand_valid[s]) for s in SIDES],*[self.control.closed[s] for s in SIDES],
                                     response,inference,(first-now)/1e6,(end-first)/1e6,compute,period,self.data.time,*errors.reshape(-1)])
                    self.frames+=1
                    if tick-last_print>2:
                        print(f'CONTROL 20Hz | frame {self.frames} | arms {self.control.valid} | CLOSED {self.control.closed} | compute {compute:.1f} ms | input {response:.1f} ms',flush=True)
                        f.flush();last_print=tick
                    if self.stream.error: raise RuntimeError(self.stream.error)
                    deadline+=.05
                    remaining=deadline-time.perf_counter()
                    if remaining>0: self.stop.wait(remaining)
                    else: deadline=time.perf_counter()
        except Exception:
            self.error=traceback.format_exc();print(self.error,flush=True)
        finally: self.stop.set()

    def close(self):
        self.stop.set();self.thread.join(timeout=5)
        self.control.close()


def run_split(args):
    cv2.setNumThreads(1);timer=os.name=='nt' and ctypes.windll.winmm.timeBeginPeriod(1)==0
    out=ROOT/'outputs'/'live_bimanual'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    recorder=TestDatasetRecorder(out/'dataset',dict(settings=dict(vars(args)),runtime='separate physics worker and GUI MjData',
                              side_array_order=list(SIDES),arms=list(SIDES),aggregate_validity='any side',created_local=datetime.now().isoformat()),save_video=args.save_video)
    stream=BimanualHumanStream(args.camera,args.pose_model,(args.width,args.height));loop=None;display_times=[];failure=None
    print(f'SPLIT BIMANUAL | dataset {recorder.path} | physics independent of GUI',flush=True)
    window='LIVE BOTH ARMS | local recording | C calibrate T wrist SPACE pause Q exit'
    flags=dict(quit=False)
    def key(code):
        if code in (81,113,27): flags['quit']=True
        elif loop is not None:
            if code==32: loop.paused=not loop.paused
            elif code in (67,99): loop.reset_requested=True
            elif code in (84,116): loop.wrist_mode='full' if loop.wrist_mode=='twist' else 'twist';loop.reset_requested=True
    start=time.perf_counter()
    try:
        cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,960,540)
        splash=np.full((540,960,3),55,np.uint8);cv2.putText(splash,'Starting independent 20 Hz robot control...',(20,240),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        cv2.imshow(window,splash);cv2.waitKey(1)
        model,data=build_motion_scene();render=mujoco.MjData(model)
        model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.8,.8,.8]
        loop=BimanualControlLoop(model,data,stream,recorder,out,args.wrist_mode)
        recorder.update_metadata(robot_joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)],
                                 actuator_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)],
                                 timestep_s=float(model.opt.timestep),physics_steps_per_control=100,
                                 clock='host perf_counter_ns; not optical exposure',action_layout='left joint7+finger2, right joint7+finger2',
                                 direction_filter='keypoint One Euro only; redundant direction EMA removed',arm_speed_rad_s=2.5,
                                 palm_alignment='absolute anatomical palm normal; robot palm broad face = TCP +X; preserve inter-hand facing',
                                 source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                                ('live_bimanual_runtime.py','live_bimanual_stream.py','bimanual_teleop.py','human_teleop_core.py','live_motion_scene.py','realtime_bimanual_planner.py')},
                                 avoidance='async joint-space RRT with latest-goal checking; sampled mesh clearance 8 mm; original physical collision unchanged')
        recorder.update_metadata(avoidance_rrt_sha256=hashlib.sha256((ROOT.parent/'pose2arm-main'/'rrt_connect_planner.py').read_bytes()).hexdigest())
        render.qpos[:]=data.qpos;mujoco.mj_forward(model,render)
        loop.start()
        with (out/'display.csv').open('w',newline='',encoding='utf-8') as display_file,mjviewer.launch_passive(model,render,key_callback=key) as view:
            display_writer=csv.writer(display_file);display_writer.writerow(['display_frame','control_frame','render_ms'])
            view.cam.lookat[:]=[.15,0,.52];view.cam.distance=1.55;view.cam.azimuth=180;view.cam.elevation=-25
            if view.user_scn is not None:
                view.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                view.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            print('BOTH WINDOWS OPEN | independent physics worker, latest-state display, no skipped physics',flush=True)
            while view.is_running() and not flags['quit'] and not loop.stop.is_set():
                begin=time.perf_counter();state=loop.snapshot()
                if state is None: time.sleep(.005);continue
                obs=state['obs'];render.qpos[:]=state['qpos'];render.qvel[:]=state['qvel'];render.ctrl[:]=state['ctrl'];render.time=state['sim_time']
                mujoco.mj_forward(model,render)
                frame=splash.copy() if obs is None else cv2.flip(cv2.resize(obs['image'],(960,540)),1)
                lines=[('PAUSED' if state['paused'] else 'LIVE BOTH ARMS')+' | PHYSICS INDEPENDENT OF DISPLAY',
                       'C calibrate both | T twist/full | SPACE pause | Q exit',
                       'LOCAL '+('VIDEO + KEYPOINT RECORDING' if args.save_video else 'KEYPOINT RECORDING')]
                for i,s in enumerate(SIDES):
                    arm=obs['arms'][s] if obs else None
                    gesture=arm['fist']['gesture'] if state['hand_valid'][s] and arm['fist'] else 'HAND MISSING'
                    if arm and arm['fist'] and arm['fist'].get('source')=='image_extension': gesture+=' [2D]'
                    lines.append(f'{s.upper()}: {"ARM OK" if state["valid"][s] else "ARM HOLD"} | {gesture} | {"CLOSED" if state["closed"][s] else "OPEN"}')
                    lines.append(f'  upper/fore error {state["errors"][i,0]:.0f}/{state["errors"][i,1]:.0f} deg')
                    if state['wrist_rejected'][s]: lines.append(f'  {s.upper()} WRIST: holding noisy palm estimate')
                    elif state['wrist_limited'][s]: lines.append(f'  {s.upper()} WRIST: physical joint limit')
                lines.append(f'Wrist {state["wrist_mode"]} | control frame {state["frame"]} | input-to-command {state["response"]:.0f} ms')
                lines.append(f'ARMS: {state["avoidance"]["mode"]} | route {state["avoidance"]["route_id"]} | check {state["avoidance"]["planning_ms"]:.1f} ms')
                if state['avoidance'].get('plan_failure'): lines.append('ARMS search: '+state['avoidance']['plan_failure'])
                if recorder.error: lines.append('RECORDING ERROR: '+recorder.error[:60])
                for i,line in enumerate(lines):
                    cv2.putText(frame,line,(12,28+29*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3)
                    cv2.putText(frame,line,(12,28+29*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cv2.imshow(window,frame);code=cv2.waitKey(1)&255
                if code!=255: key(code)
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: flags['quit']=True
                if hasattr(view,'set_texts'): view.set_texts((None,None,'LIVE BIMANUAL / INDEPENDENT CONTROL',f'frame {state["frame"]} | Q exit'))
                view.sync(state_only=True)
                elapsed=(time.perf_counter()-begin)*1000;display_times.append(elapsed)
                display_writer.writerow([len(display_times)-1,state['frame'],elapsed])
                if len(display_times)%20==0: display_file.flush()
                if args.seconds and time.perf_counter()-start>=args.seconds: break
                remaining=.05-(time.perf_counter()-begin)
                if remaining>0: time.sleep(remaining)
        if loop.error: raise RuntimeError(loop.error)
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';raise
    finally:
        if loop is not None: loop.close()
        stream.close();cv2.destroyAllWindows();recorder.update_metadata(camera=stream.metadata,run_error=failure);recorder.close()
        if timer: ctypes.windll.winmm.timeEndPeriod(1)
        def stats(values):
            return None if not values else dict(n=len(values),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),max_ms=float(np.max(values)))
        report=dict(runtime='split physics/gui; separate MjData',run_error=failure,recording_error=recorder.error,
                    dataset=str(recorder.path),rows=recorder.saved_rows,video_frames=recorder.video_frames,dropped_rows=recorder.dropped_rows,
                    control_period=stats(loop.periods if loop else []),control_compute=stats(loop.compute if loop else []),display=stats(display_times),
                    left_latency=stats(loop.latencies['left'] if loop else []),right_latency=stats(loop.latencies['right'] if loop else []),
                    scope='host timing only; not optical/physical tracking delay or task success')
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
        finish_test_cycle(out)
