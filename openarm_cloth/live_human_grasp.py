"""Local USB human wrist/pinch -> B physical simulator. No object-coordinate control.

Opt-in raw webcam segments and numerical control/latency telemetry are saved locally.
The fixed ready pose is a robot calibration, not an autonomous grasp trajectory.
"""
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
from datetime import datetime
import json
import hashlib
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

import cv2
import mujoco
from mujoco import viewer as mjviewer
import numpy as np

from human_teleop_core import (ROOT, PadCartesianIK, RelativeWristMapper,
                              ArmPoseIK, human_right_wrist, human_right_pose)
from gripper_map import GripperMapper
from human_gripper_gestures import FistGripperMapper, fist_features, gesture_diagnostics
from handeye_grasp_red_cube import finger_joint_indices
from hand_association import associate_hands
from smoothing import VectorOneEuroFilter
from member_b_realtime import USBLatestFrame
from member_b_scene import build_scene
from live_motion_scene import build_motion_scene
from member_b_actuators import ActuatorController
from live_test_dataset import TestDatasetRecorder,array_or_nan
from live_test_cycle import finish_test_cycle


def timed_pose_inference(detector,image,timestamp):
    result=detector.detect_for_video(image,timestamp)
    return result,time.perf_counter_ns()


class HumanStream:
    """Inference in its own thread, using only the newest available USB frame."""
    def __init__(self, device=0,pose_model='lite',resolution=(1280,720)):
        self.latest=None; self.error=None; self.closed=False; self.camera=None
        self.lock=threading.Lock()
        self.pose_model=pose_model;self.resolution=resolution;self.metadata={}
        self.thread=threading.Thread(target=self._run,args=(device,),daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock: return self.latest

    def _run(self,device):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            models=ROOT.parent/'pose2arm-main'/'models'
            options=dict(running_mode=vision.RunningMode.VIDEO)
            pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models/f'pose_landmarker_{self.pose_model}.task')),
                num_poses=1,**options))
            hands=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models/'hand_landmarker.task')),
                num_hands=2,**options))
            retry=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(models/'hand_landmarker.task')),
                running_mode=vision.RunningMode.VIDEO,num_hands=1))
            previous=np.full((2,2),np.nan)
            seen=np.zeros(2)
            self.camera=USBLatestFrame(device,*self.resolution,30)
            self.metadata=dict(self.camera.metadata,pose_model=self.pose_model)
            print('CAMERA CONFIG '+json.dumps(self.metadata),flush=True)
            sequence=0; timestamp=-1; previous_normal=None
            pose_filter=VectorOneEuroFilter(18,30,min_cutoff=1.5,beta=4)
            hand_filter=VectorOneEuroFilter(63,30,min_cutoff=2,beta=5)
            last_infer=None;last_hand=None;last_full_search=0;previous_pose_norm=None
            with pose,hands,retry,ThreadPoolExecutor(max_workers=1) as pose_pool:
                while not self.closed:
                    sequence,read_start,read_end,bgr=self.camera.read_new(sequence)
                    infer_start=time.perf_counter_ns()
                    timestamp=max(timestamp+1,infer_start//1000000)
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    # Same CURRENT frame feeds both models. Only ROI selection
                    # uses the previous pose; outputs retain this frame's clock.
                    future=pose_pool.submit(timed_pose_inference,pose,image,timestamp)
                    norm=previous_pose_norm
                    hand_start=time.perf_counter_ns()
                    previous[(infer_start/1e9-seen)>.4]=np.nan
                    wrists=[];sides=[];hand_norm=[];hand_world=[]
                    roi_used=False
                    # ROI tracking FIRST: do not run an expensive full-frame
                    # two-hand detector before each successful right-hand crop.
                    if norm is not None and norm[16,3]>=.6:
                        wh=np.array([bgr.shape[1],bgr.shape[0]])
                        wrist=norm[16,:2]*wh;elbow=norm[14,:2]*wh
                        center=wrist+.35*(wrist-elbow)
                        size=int(np.clip(np.linalg.norm(wrist-elbow)*1.3,180,min(wh)*.65))
                        lo=np.maximum(0,(center-size/2).astype(int));hi=np.minimum(wh,(center+size/2).astype(int))
                        if np.min(hi-lo)>=100:
                            crop=np.ascontiguousarray(rgb[lo[1]:hi[1],lo[0]:hi[0]])
                            hretry=retry.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop),timestamp)
                            if hretry.hand_landmarks:
                                mapped=[SimpleNamespace(x=(p.x*(hi[0]-lo[0])+lo[0])/wh[0],
                                                        y=(p.y*(hi[1]-lo[1])+lo[1])/wh[1],z=p.z*(hi[0]-lo[0])/wh[0])
                                        for p in hretry.hand_landmarks[0]]
                                if np.linalg.norm(np.array([mapped[0].x,mapped[0].y])-norm[16,:2])<.16:
                                    hand_norm.append(mapped);hand_world.append(hretry.hand_world_landmarks[0]);sides.append(1)
                                    wrists.append([mapped[0].x,mapped[0].y]);roi_used=True
                    if not roi_used and infer_start-last_full_search>200000000:
                        last_full_search=infer_start
                        hand=hands.detect_for_video(image,timestamp)
                        wrists=[[h[0].x,h[0].y] for h in hand.hand_landmarks]
                        labels=[h[0].category_name for h in hand.handedness]
                        sides=associate_hands(wrists,labels,norm,previous)
                        hand_norm=list(hand.hand_landmarks);hand_world=list(hand.hand_world_landmarks)
                    hand_done=time.perf_counter_ns()
                    body,pose_done=future.result()
                    world=norm=world_raw=None
                    if body.pose_landmarks:
                        norm=np.array([[p.x,p.y,p.z,p.visibility] for p in body.pose_landmarks[0]])
                        world=np.array([[p.x,p.y,p.z,p.visibility] for p in body.pose_world_landmarks[0]])
                        world_raw=world.copy()
                        ids=[11,12,14,16,23,24]
                        hz=30 if last_infer is None else 1/max(.01,min(.2,(infer_start-last_infer)/1e9))
                        for flt in pose_filter.filters: flt.freq=hz
                        if np.min(world[[11,12,14,16],3])>=.45:
                            world[ids,:3]=pose_filter.filter(world[ids,:3].reshape(-1)).reshape(6,3)
                    previous_pose_norm=norm
                    last_infer=infer_start
                    point=human_right_wrist(world)
                    if roi_used:
                        if norm is None or np.linalg.norm(np.asarray(wrists[0])-norm[16,:2])>.2:
                            sides=[None];roi_used=False
                    elif hand_norm:
                        sides=associate_hands(wrists,labels,norm,previous)
                    distance=None; right_hand=False; right_world=right_raw=right_norm=None;fist=None
                    for i,side in enumerate(sides):
                        if side is None: continue
                        previous[side]=wrists[i]
                        seen[side]=infer_start/1e9
                        if side==1:
                            hw=hand_world[i]
                            right_world=np.array([[p.x,p.y,p.z] for p in hw])
                            right_raw=right_world.copy()
                            right_norm=np.array([[p.x,p.y,p.z] for p in hand_norm[i]])
                            if last_hand is not None and (infer_start-last_hand)/1e9>.4:
                                hand_filter=VectorOneEuroFilter(63,30,min_cutoff=2,beta=5)
                            if last_hand is not None:
                                for flt in hand_filter.filters: flt.freq=1/max(.01,min(.2,(infer_start-last_hand)/1e9))
                            # Gesture uses unsmoothed anatomical geometry: filtering
                            # a whole closing hand can temporarily invent an open one.
                            fist=fist_features(right_world,right_norm,(bgr.shape[1],bgr.shape[0]))
                            right_world=hand_filter.filter(right_world.reshape(-1)).reshape(21,3)
                            last_hand=infer_start
                            distance=float(np.linalg.norm(
                                right_world[4]-right_world[8]))
                            right_hand=True
                    arm_pose=human_right_pose(world,right_world,previous_normal)
                    if arm_pose is not None: previous_normal=arm_pose['normal']
                    annotated=bgr.copy()
                    if norm is not None:
                        for a,b in ((11,12),(11,23),(12,24),(23,24),(12,14),(14,16)):
                            pts=[tuple((norm[i,:2]*[bgr.shape[1],bgr.shape[0]]).astype(int)) for i in (a,b)]
                            cv2.line(annotated,*pts,(0,220,0),2)
                    for h,side in zip(hand_norm,sides):
                        color=(0,255,255) if side==1 else (150,150,150)
                        for p in h: cv2.circle(annotated,(int(p.x*bgr.shape[1]),int(p.y*bgr.shape[0])),3,color,-1)
                    result=dict(seq=sequence,read_start=read_start,read_end=read_end,
                                done=time.perf_counter_ns(),infer_start=infer_start,
                                point=point,pose=arm_pose,distance=distance,right_hand=right_hand,image=annotated,
                                raw_image=bgr,pose_raw=world_raw,pose_filtered=world,pose_norm=norm,
                                hand_raw=right_raw,hand_filtered=right_world,hand_norm=right_norm,
                                pose_ms=(pose_done-infer_start)/1e6,hand_ms=(hand_done-hand_start)/1e6,
                                fist=fist,roi_used=roi_used,visibility=None if world is None else float(np.min(world[[11,12,14,16],3])))
                    with self.lock: self.latest=result
        except Exception:
            self.error=traceback.format_exc()
            print(self.error,flush=True)
        finally:
            if self.camera is not None: self.camera.close()

    def close(self):
        self.closed=True
        self.thread.join(timeout=4)


def run(args):
    cv2.setNumThreads(1)
    timer_enabled=False
    if os.name=='nt': timer_enabled=ctypes.windll.winmm.timeBeginPeriod(1)==0
    out=ROOT/'outputs'/'live_human'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    print(f'LOCAL LIVE USB input. Human video recording: {args.save_video}. Dataset: {out / "dataset"}',flush=True)
    recorder=TestDatasetRecorder(out/'dataset',dict(settings=dict(vars(args)),created_local=datetime.now().isoformat(),
                                    source='real USB webcam',joint_order='left7,right7; full qpos/qvel arrays use model joint names'),
                                 save_video=args.save_video)
    stream=HumanStream(args.camera,args.pose_model,(args.width,args.height))
    cv2.namedWindow('LIVE human camera - C recenter | SPACE pause | Q exit',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('LIVE human camera - C recenter | SPACE pause | Q exit',800,600)
    splash=np.full((480,640,3),70,np.uint8)
    cv2.putText(splash,'Starting local camera and robot IK...',(15,230),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    cv2.imshow('LIVE human camera - C recenter | SPACE pause | Q exit',splash);cv2.waitKey(1)
    flags=dict(paused=False,recenter=False,quit=False,mode=args.mode)
    def key(code):
        if code==32: flags['paused']=not flags['paused']
        elif code in (67,99): flags['recenter']=True
        elif code in (81,113,27): flags['quit']=True
        elif code in (71,103) and args.scene=='grasp':
            flags['mode']='grasp' if flags['mode']=='pose' else 'pose'
            flags['recenter']=True
        elif code in (84,116):
            args.wrist_mode='full' if args.wrist_mode=='twist' else 'twist'
            flags['wrist_changed']=True
    records=[]; infer_samples=[]; response_samples=[]
    try:
        model,data=build_motion_scene() if args.scene=='motion' else build_scene()
        print(f'SCENE {args.scene}: table/cube absent' if args.scene=='motion' else 'SCENE grasp: frozen physical table/cube',flush=True)
        model.vis.headlight.ambient[:]=[.5,.5,.5]
        model.vis.headlight.diffuse[:]=[.8,.8,.8]
        controller=ActuatorController(model,data)
        specs=controller.specs
        recorder.update_metadata(robot_joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)],
                                 actuator_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)],
                                 timestep_s=float(model.opt.timestep),clock='host perf_counter_ns; not camera optical exposure time',
                                 action_layout='left joint7 + left finger2, right joint7 + right finger2; radians/metres',
                                 source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                for name in ('live_human_grasp.py','human_teleop_core.py','human_gripper_gestures.py','live_motion_scene.py')})
        targets={s:data.qpos[spec.qpos].copy() for s,spec in specs.items()}
        start_right=targets['right'].copy()
        ik=PadCartesianIK(model,data,initialize_grasp=args.scene=='grasp')
        ready_joint_target=ik.ready_q.copy()
        if args.mode=='pose': ready_joint_target[4:]=0
        pose_ik=ArmPoseIK(ik,wrist_mode=args.wrist_mode)
        mapper=RelativeWristMapper(ik.ready_center)
        gripper=GripperMapper(hold_frames=3,lost_frames=10)
        fist_mapper=FistGripperMapper()
        right_finger_qpos=finger_joint_indices(model,'right')[0]
        steps=round(.05/model.opt.timestep)
        last_seq=-1; last_hand_time=time.perf_counter(); closure=0
        command_cache=None;last_pose_stamp=None
        ik_error=0.; frames=0; begun=time.perf_counter(); last_print=begun
        with (out/'latency.csv').open('w',newline='',encoding='utf-8') as f, mjviewer.launch_passive(model,data,key_callback=key) as view:
            writer=csv.writer(f)
            writer.writerow(['control_frame','source_seq','fresh_source','pose_valid','hand_valid','paused',
                             'capture_return_ns','inference_start_ns','inference_done_ns','first_step_ns',
                             'read_return_to_first_step_ms','inference_ms','control_cycle_ms','ik_error_m',
                             'gripper_closed','goal_x','goal_y','goal_z','mode',
                             'upper_angle_error_deg','forearm_angle_error_deg','palm_angle_error_deg',
                             'hand_roi_retry_used','minimum_arm_visibility',
                             'wrist_target_q5','wrist_target_q6','wrist_target_q7',
                             'wrist_actual_q5','wrist_actual_q6','wrist_actual_q7',
                             'gesture_mode','folded_fingers','extended_fingers','curl_index','curl_middle','curl_ring','curl_pinky',
                             'pinch_distance_m','right_finger_target_m','right_finger1_actual_m','right_finger2_actual_m',
                             'wrist_mapping_mode','twist_target_rad','twist_limit_hit',
                             'wrist_torque_q5','wrist_torque_q6','wrist_torque_q7','pose_inference_ms','hand_inference_ms',
                             'prepare_ms','physics_ms','ui_ms'])
            view.cam.lookat[:]=[.15,0,.52] if args.scene=='motion' else [.43,0,.46]
            view.cam.distance=1.45 if args.scene=='motion' else 1.3
            view.cam.azimuth=180;view.cam.elevation=-35
            print(f'WINDOWS OPEN | scene {args.scene} | wrist {args.wrist_mode} | gesture {args.gesture.upper()} | T wrist twist/full, C recenter, SPACE pause, Q exit.',flush=True)
            while view.is_running() and not flags['quit']:
                if (out/'STOP').exists(): break
                tick=time.perf_counter();now=time.perf_counter_ns()
                obs=stream.snapshot()
                fresh=obs is not None and obs['seq']!=last_seq
                valid=obs is not None and now-obs['read_end']<500000000
                point=obs['point'] if valid else None
                pose=obs['pose'] if valid else None
                if flags.pop('wrist_changed',False):
                    pose_ik=ArmPoseIK(ik,wrist_mode=args.wrist_mode)
                    command_cache=None;last_pose_stamp=None
                startup=frames<60
                if startup:
                    t=(frames+1)/60;t=t*t*(3-2*t)
                    targets['right']=start_right+t*(ready_joint_target-start_right)
                elif not flags['paused']:
                    if flags['recenter'] and point is not None:
                        ik.kin.qpos[ik.spec.qpos]=data.qpos[ik.spec.qpos]
                        mujoco.mj_forward(model,ik.kin)
                        mapper.recenter(point,ik.midpoint());flags['recenter']=False
                        pose_ik.reset()
                        command_cache=None;last_pose_stamp=None
                    goal=mapper.target.copy() if flags['mode']=='pose' else mapper.update(point)
                    if point is not None:
                        if flags['mode']=='pose':
                            command=command_cache
                            if fresh and pose is not None:
                                pose_dt=.05 if last_pose_stamp is None else np.clip((obs['read_end']-last_pose_stamp)/1e9,.01,.15)
                                pose_ik.update_target(pose,targets['right'],dt=float(pose_dt))
                                command_cache=command=pose_ik.solve(targets['right'],iterations=6)
                                last_pose_stamp=obs['read_end']
                        else:
                            command,ik_error=ik.solve(goal,targets['right'])
                        if command is not None:
                            max_joint_step=np.array([.09,.09,.09,.09,.22,.22,.22])
                            targets['right']+=np.clip(command-targets['right'],-max_joint_step,max_joint_step)
                    if args.gesture=='fist':
                        if fresh and valid:
                            closure=fist_mapper.update(obs['fist'],obs['seq'],tick)
                        elif not valid or (obs and not obs['right_hand']):
                            closure=fist_mapper.update(None,-1,tick)
                    else:
                        if fresh and valid:
                            closure=gripper.update(obs['distance']);last_hand_time=tick
                        elif tick-last_hand_time>.5:
                            closure=gripper.update(None)
                finger=.044 if startup else .044*(1-closure)
                controller.set_targets(targets,{'left':.044,'right':finger})
                controller.apply();first_step=time.perf_counter_ns();mujoco.mj_step(model,data)
                for _ in range(steps-1): controller.apply();mujoco.mj_step(model,data)
                physics_end=time.perf_counter_ns()
                if not np.isfinite(data.qpos).all(): raise RuntimeError('Non-finite physical state')
                state='READY MOTION' if startup else 'PAUSED' if flags['paused'] else 'LIVE' if point is not None else 'NO RIGHT ARM: HOLD'
                frame=splash.copy() if obs is None else cv2.flip(cv2.resize(obs['image'],(960,540)),1)
                lines=[state+' | '+flags['mode'].upper(),
                       'Shoulder/elbow/wrist + palm follow' if flags['mode']=='pose' else 'Tabletop wrist delta, fixed gripper direction',
                       'C: recenter  SPACE: pause  Q: exit' if args.scene=='motion' else 'C: recenter  G: pose/grasp  SPACE: pause  Q: exit',
                       f'Hand: {"RIGHT" if valid and obs["right_hand"] else "not detected"} | gripper {"CLOSED" if closure else "OPEN"}',
                       f'IK error {ik_error*1000:.1f} mm | input age {(now-obs["read_end"])/1e6:.0f} ms' if obs else 'Loading recognition models...']
                actual_angles=[float('nan')]*3
                if flags['mode']=='pose' and pose_ik.upper is not None:
                    ik.kin.qpos[ik.spec.qpos]=data.qpos[ik.spec.qpos];mujoco.mj_forward(model,ik.kin)
                    units=pose_ik.geometry()
                    actual_angles[:2]=[float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1))))
                                      for a,b in zip(units,(pose_ik.upper,pose_ik.fore))]
                    from scipy.spatial.transform import Rotation
                    actual_angles[2]=float(np.degrees(Rotation.from_matrix(pose_ik.rotation @ ik.kin.xmat[ik.spec.body_id].reshape(3,3).T).magnitude()))
                    lines[4]=f'Arm errors: upper {actual_angles[0]:.0f}, forearm {actual_angles[1]:.0f} deg'
                    lines.append(f'Palm {actual_angles[2]:.0f} deg | {"HAND TRACKED" if pose and pose["palm"] else "ARM FRAME ONLY"}')
                if obs and obs['distance'] is not None: lines.append(f'Estimated pinch {obs["distance"]*100:.1f} cm')
                if args.gesture=='fist':
                    features=obs['fist'] if valid else None
                    lines.append('FIST closes | OPEN PALM opens')
                    lines.append('Gesture: '+(features['gesture'] if features else 'HAND NOT DETECTED'))
                    if features:
                        lines.append('Finger curl: '+', '.join(f'{v:.2f}' for v in features['scores']))
                else: features=None
                if obs:
                    lines.append('Right-hand ROI tracking' if obs['roi_used'] else 'Full search / hand missing')
                    lines.append(f'Pose {obs["pose_ms"]:.0f} ms | hand {obs["hand_ms"]:.0f} ms | LOCAL REC'+(' VIDEO' if args.save_video else ' KEYPOINTS'))
                wrist_actual=data.qpos[specs['right'].qpos][4:]
                lines.append('Wrist degrees: '+', '.join(f'{v:.0f}' for v in np.degrees(wrist_actual)))
                lines.append(f'Wrist {args.wrist_mode.upper()} | T toggles twist/full'+(' | AXIAL LIMIT' if pose_ik.twist_limited else ''))
                lines.append(f'Finger target {finger*1000:.0f} mm | actual '+', '.join(f'{v*1000:.0f}' for v in data.qpos[right_finger_qpos]))
                if stream.error: lines=['CAMERA / RECOGNITION ERROR - see log','Q: exit']
                for i,line in enumerate(lines):
                    cv2.putText(frame,line,(10,25+25*i),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3)
                    cv2.putText(frame,line,(10,25+25*i),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                cv2.imshow('LIVE human camera - C recenter | SPACE pause | Q exit',frame)
                code=cv2.waitKey(1)&255
                if code!=255: key(code)
                if cv2.getWindowProperty('LIVE human camera - C recenter | SPACE pause | Q exit',cv2.WND_PROP_VISIBLE)<1: flags['quit']=True
                if hasattr(view,'set_texts'): view.set_texts((None,None,'LIVE HUMAN TELEOP',state+' | C recenter | SPACE pause | Q exit'))
                view.sync()
                ui_end=time.perf_counter_ns()
                cycle=(time.perf_counter()-tick)*1000
                inference=(obs['done']-obs['infer_start'])/1e6 if obs else float('nan')
                response=(first_step-obs['read_end'])/1e6 if obs else float('nan')
                writer.writerow([frames,obs['seq'] if obs else -1,int(fresh),int(point is not None),
                                 int(valid and obs['right_hand']),int(flags['paused']),
                                 obs['read_end'] if obs else 0,obs['infer_start'] if obs else 0,
                                 obs['done'] if obs else 0,first_step,response,inference,cycle,ik_error,closure,*mapper.target,
                                 flags['mode'],*actual_angles,int(obs['roi_used']) if obs else 0,
                                 obs['visibility'] if obs else float('nan'),*targets['right'][4:],*wrist_actual,
                                 args.gesture,features['folded'] if features else 0,features['extended'] if features else 0,
                                 *(features['scores'] if features else [float('nan')]*4),
                                 obs['distance'] if obs else float('nan'),finger,*data.qpos[right_finger_qpos],
                                 args.wrist_mode,pose_ik.twist_target,int(pose_ik.twist_limited),
                                 *data.ctrl[specs['right'].actuators][4:],
                                 obs['pose_ms'] if obs else float('nan'),obs['hand_ms'] if obs else float('nan'),
                                 (first_step-now)/1e6,(physics_end-first_step)/1e6,(ui_end-physics_end)/1e6])
                row=dict(control_frame=np.asarray(frames),source_seq=np.asarray(obs['seq'] if obs else -1),
                         fresh_source=np.asarray(fresh),capture_read_end_ns=np.asarray(obs['read_end'] if obs else 0,dtype=np.int64),
                         inference_done_ns=np.asarray(obs['done'] if obs else 0,dtype=np.int64),
                         first_physics_step_ns=np.asarray(first_step,dtype=np.int64),physics_end_ns=np.asarray(physics_end,dtype=np.int64),
                         sim_time=np.asarray(data.time),startup=np.asarray(startup),paused=np.asarray(flags['paused']),
                         pose_raw=array_or_nan(obs['pose_raw'] if obs else None,(33,4)),
                         pose_filtered=array_or_nan(obs['pose_filtered'] if obs else None,(33,4)),
                         pose_norm=array_or_nan(obs['pose_norm'] if obs else None,(33,4)),
                         right_hand_raw=array_or_nan(obs['hand_raw'] if obs else None,(21,3)),
                         right_hand_filtered=array_or_nan(obs['hand_filtered'] if obs else None,(21,3)),
                         right_hand_norm=array_or_nan(obs['hand_norm'] if obs else None,(21,3)),
                         action=controller.action.copy(),robot_qpos=data.qpos.copy(),robot_qvel=data.qvel.copy(),
                         robot_ctrl=data.ctrl.copy(),pose_valid=np.asarray(point is not None),hand_valid=np.asarray(bool(valid and obs['right_hand'])),
                         gripper_closed=np.asarray(closure),wrist_mode=np.asarray(args.wrist_mode),
                         curl_scores=np.asarray(features['scores'] if features else [float('nan')]*4),
                         twist_target_rad=np.asarray(pose_ik.twist_target),twist_limit_hit=np.asarray(pose_ik.twist_limited),
                         arm_angle_error_deg=np.asarray(actual_angles),
                         read_to_step_ms=np.asarray(response),inference_ms=np.asarray(inference))
                row.update({f'right_{k}':v for k,v in gesture_diagnostics(features).items()})
                recorder.submit(row,obs['raw_image'] if fresh and args.save_video else None)
                if fresh:
                    last_seq=obs['seq'];infer_samples.append(inference)
                    if valid and point is not None and not startup and not flags['paused']: response_samples.append(response)
                records.append(np.r_[data.time,controller.action,data.qpos[controller.qpos],
                                     obs['seq'] if obs else -1,closure])
                frames+=1
                if tick-last_print>2:
                    gesture=features['gesture'] if features else 'NO HAND' if args.gesture=='fist' else 'PINCH'
                    print(f'{state} {flags["mode"]} | seq {last_seq} | hand {bool(valid and obs["right_hand"])} | gesture {gesture} | close {closure} | actual fingers {data.qpos[right_finger_qpos]} | arm error deg {actual_angles}',flush=True)
                    f.flush();last_print=tick
                if stream.error: raise RuntimeError(stream.error)
                if args.seconds and time.perf_counter()-begun>=args.seconds: break
                remaining=.05-(time.perf_counter()-tick)
                if remaining>0: time.sleep(remaining)
    finally:
        pending_error=sys.exc_info()[1]
        run_error=f'{type(pending_error).__name__}: {pending_error}' if pending_error is not None else stream.error
        stream.close();cv2.destroyAllWindows();recorder.update_metadata(camera=stream.metadata,run_error=run_error);recorder.close()
        if timer_enabled: ctypes.windll.winmm.timeEndPeriod(1)
        if records: np.savez_compressed(out/'numeric_control.npz',rows=np.asarray(records),
                                       columns='sim_time,action18,left_right_joint14,source_seq,gripper_closed')
        def stats(samples):
            return None if not samples else dict(n=len(samples),mean_ms=float(np.mean(samples)),p95_ms=float(np.percentile(samples,95)),max_ms=float(np.max(samples)))
        report=dict(source='LIVE USB camera',raw_video_saved=args.save_video,object_coordinates_used_by_controller=False,
                    control='actual-model shoulder/elbow/wrist direction IK + calibrated palm + selected human gripper gesture; G toggles tabletop delta mode',
                    inference=stats(infer_samples),host_read_return_to_first_physics_step=stats(response_samples),
                    timing_scope='host capture return, not optical exposure; excludes physical settling; not proof of grasp',
                    error=stream.error,run_error=run_error)
        report['camera']=stream.metadata
        report['gesture']=args.gesture
        report['scene']=args.scene
        report['wrist_mapping_mode']=args.wrist_mode
        report['dataset']=dict(path=str(recorder.path),saved_rows=recorder.saved_rows,video_frames=recorder.video_frames,
                               dropped_rows=recorder.dropped_rows,error=recorder.error)
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
        finish_test_cycle(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--seconds',type=float,default=0,help='0: keep windows open until user exits')
    p.add_argument('--mode',choices=('pose','grasp'),default='pose')
    p.add_argument('--pose-model',choices=('full','lite'),default='lite')
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--gesture',choices=('fist','pinch'),default='fist')
    p.add_argument('--scene',choices=('motion','grasp'),default='motion')
    p.add_argument('--wrist-mode',choices=('twist','full'),default='twist',help='twist: axial turn, neutral wrist bend; full: all palm attitude changes')
    p.add_argument('--save-video',action=argparse.BooleanOptionalAction,default=True,help='user opted into local raw video recording; --no-save-video disables it')
    args=p.parse_args()
    if args.scene=='motion' and args.mode!='pose': p.error('Motion-only scene uses pose mode; choose --scene grasp for tabletop delta control')
    run(args)
