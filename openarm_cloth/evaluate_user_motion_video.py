"""Evaluate an existing live recognizer on a local video, without changing it.

Only the camera transport and clock are adapted, as in benchmark_bimanual_vision.
The production Lite/full model, ROI recovery, association, filtering and gesture
code run unchanged. Coverage and continuity are not ground-truth accuracy.
No webcam, simulator, retention/deletion, training or upload is used.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'pose2arm-main'))
import live_bimanual_stream as recognizer
from human_gripper_gestures import FistGripperMapper


def longest_missing(mask):
    best=current=0
    for present in mask:
        current=0 if present else current+1
        best=max(best,current)
    return best


def summary(rows,fps):
    result={'frames_processed':len(rows),'duration_s':len(rows)/fps,
            'coverage_is_not_accuracy':True,'manual_ground_truth_labels':False,
            'timing_scope':'offline inference wall time, not live end-to-end latency'}
    times=np.asarray([r['offline_wall_ms'] for r in rows])
    result['offline_inference_ms']={f'p{q}':float(np.percentile(times,q)) for q in (50,95,100)}
    for i,side in enumerate(('left','right')):
        body=np.array([r['arm_valid'][i] for r in rows],bool)
        hand=np.array([r['hand_valid'][i] for r in rows],bool)
        eligible=np.array([r['arm_visible_inframe'][i] for r in rows],bool)
        gestures=np.array([r['gesture'][i] for r in rows])
        result[side]={'arm_output_n':int(body.sum()),'arm_output_pct':100*float(body.mean()),
                      'hand_detection_n':int(hand.sum()),'hand_detection_pct':100*float(hand.mean()),
                      'body_visible_inframe_n':int(eligible.sum()),
                      'hand_coverage_when_body_visible_pct':100*float(hand[eligible].mean()) if eligible.any() else None,
                      'longest_hand_missing_s':longest_missing(hand)/fps,
                      'gesture_frame_counts':{str(g):int((gestures==g).sum()) for g in np.unique(gestures)}}
    both=np.array([all(r['hand_valid']) for r in rows],bool)
    result['both_hands_detected_pct']=100*float(both.mean())
    return result


class VideoEvaluation:
    def __init__(self,video,output):
        self.video=video;self.output=output;self.cap=cv2.VideoCapture(str(video))
        if not self.cap.isOpened(): raise FileNotFoundError(video)
        self.fps=float(self.cap.get(cv2.CAP_PROP_FPS));self.count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH));self.height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(self.fps) or self.fps<=0: raise ValueError('Invalid video FPS')
        self.writer=cv2.VideoWriter(str(output/'recognition_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                                    self.fps,(self.width,self.height))
        if not self.writer.isOpened(): raise RuntimeError('Cannot create overlay video')
        self.owner=None;self.ready=threading.Event();self.rows=[];self.index=0
        self.current=None;self.wall=None;self.capture_ns=1_000_000_000;self.eof=False
        self.mappers=[FistGripperMapper(),FistGripperMapper()]

    def clock_ns(self):
        return self.capture_ns if self.wall is None else self.capture_ns+time.perf_counter_ns()-self.wall

    def collect(self):
        if self.current is None: return
        item=self.owner.snapshot()
        if item is None or item['seq']!=self.current: raise RuntimeError('Source/result alignment lost')
        pose=item['pose_norm'];seconds=(self.current-1)/self.fps
        row=dict(frame_index=self.current-1,source_seconds=seconds,
                 offline_wall_ms=(time.perf_counter_ns()-self.wall)/1e6,
                 pose_norm=np.full((33,4),np.nan) if pose is None else pose.copy(),
                 pose_world_raw=np.full((33,4),np.nan) if item['pose_raw'] is None else item['pose_raw'].copy(),
                 arm_valid=[],hand_valid=[],arm_visible_inframe=[],gesture=[],gesture_source=[],
                 hand_norm=[],hand_world_raw=[],gripper_closed=[],curl_scores=[])
        image=item['image'].copy()
        cv2.rectangle(image,(0,0),(self.width,115),(20,20,20),-1)
        cv2.putText(image,f'CURRENT MODEL / frame {self.current-1} / {seconds:.2f}s',
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
        for i,side in enumerate(('left','right')):
            arm=item['arms'][side];f=arm['fist'];ids=[11+i,13+i,15+i]
            eligible=bool(pose is not None and np.isfinite(pose[ids]).all() and
                          np.all(pose[ids,3]>=.6) and np.all(pose[ids,:2]>=.03) and np.all(pose[ids,:2]<=.97))
            row['arm_valid'].append(arm['pose'] is not None);row['hand_valid'].append(bool(arm['hand_valid']))
            row['arm_visible_inframe'].append(eligible)
            gesture='NO HAND' if f is None else f['gesture']
            closed=self.mappers[i].update(f,self.current,seconds)
            row['gesture'].append(gesture);row['gesture_source'].append('missing' if f is None else f['source'])
            row['gripper_closed'].append(bool(closed))
            row['curl_scores'].append(np.full(4,np.nan) if f is None else f['scores'].copy())
            row['hand_norm'].append(np.full((21,3),np.nan) if arm['hand_norm'] is None else arm['hand_norm'].copy())
            row['hand_world_raw'].append(np.full((21,3),np.nan) if arm['hand_raw'] is None else arm['hand_raw'].copy())
            text=f'{side.upper()}: {gesture} | grip {"CLOSE" if closed else "OPEN"}'
            cv2.putText(image,text,(10,57+i*30),cv2.FONT_HERSHEY_SIMPLEX,.57,
                        (255,180,0) if i==0 else (0,255,255),2,cv2.LINE_AA)
        self.writer.write(image);self.rows.append(row)
        if self.current%60==0 or self.current==self.count:
            print(f'PROGRESS {self.current}/{self.count}',flush=True)
        self.current=None

    def read_new(self,sequence):
        if not self.ready.wait(60): raise RuntimeError('Recognizer owner not bound')
        self.collect()
        ok,image=self.cap.read()
        if not ok:
            self.eof=True;self.owner.closed=True
            return sequence,self.capture_ns,self.capture_ns,self.last_image
        self.index+=1;self.current=self.index
        self.capture_ns=1_000_000_000+round((self.index-1)*1e9/self.fps)
        self.wall=time.perf_counter_ns();self.last_image=image
        return self.index,self.capture_ns,self.capture_ns,image

    def close(self):
        self.cap.release();self.writer.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pose-model',choices=('lite','full'),default='lite')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/name).exists() for name in ('recognition_overlay.mp4','recognition.npz','report.json')):
        p.error('Choose a fresh output directory; existing results are never overwritten')
    evaluation=VideoEvaluation(args.video,args.output)
    class VideoCamera:
        def __init__(self,*a,**kw):
            self.metadata={'device':'local supplied MP4; no USB camera',
                           'reported_width':evaluation.width,'reported_height':evaluation.height,
                           'reported_fps':evaluation.fps,'timestamps':'source frame index / source FPS'}
        def read_new(self,sequence): return evaluation.read_new(sequence)
        def close(self): evaluation.close()
    original_camera,original_time=recognizer.USBLatestFrame,recognizer.time
    recognizer.USBLatestFrame=VideoCamera
    recognizer.time=SimpleNamespace(perf_counter_ns=evaluation.clock_ns)
    stream=None
    try:
        stream=recognizer.BimanualHumanStream(pose_model=args.pose_model,
                                            resolution=(evaluation.width,evaluation.height))
        evaluation.owner=stream;evaluation.ready.set()
        while stream.thread.is_alive(): stream.thread.join(.5)
        if stream.error: raise RuntimeError(stream.error)
        if not evaluation.eof or len(evaluation.rows)!=evaluation.count:
            raise RuntimeError(f'Incomplete decode: {len(evaluation.rows)}/{evaluation.count}')
        report=summary(evaluation.rows,evaluation.fps)
        report.update(input_video=str(args.video),source_fps=evaluation.fps,
                      pose_model=args.pose_model,side_order=['anatomical left','anatomical right'],
                      model_changed=False,training_performed=False,robot_motion_evaluated=False,
                      scope='Existing live recognizer replayed offline on supplied video; no accuracy ground truth')
        files=[ROOT/'live_bimanual_stream.py',ROOT/'human_gripper_gestures.py',ROOT/'human_teleop_core.py',
               ROOT.parent/'pose2arm-main'/'hand_association.py',
               ROOT.parent/'pose2arm-main'/'models'/f'pose_landmarker_{args.pose_model}.task',
               ROOT.parent/'pose2arm-main'/'models'/'hand_landmarker.task']
        report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        report['input_sha256']=hashlib.sha256(args.video.read_bytes()).hexdigest()
        np.savez_compressed(args.output/'recognition.npz',fps=evaluation.fps,
                            **{k:np.asarray([r[k] for r in evaluation.rows]) for k in evaluation.rows[0]})
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    finally:
        if stream is not None and stream.thread.is_alive(): stream.close()
        evaluation.close()
        recognizer.USBLatestFrame=original_camera;recognizer.time=original_time


if __name__=='__main__': main()
