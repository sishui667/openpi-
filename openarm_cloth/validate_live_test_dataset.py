"""Read-only validation/report of local live test chunks; no success labels invented."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2


def validate(path,check_video=True):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    rows=[];errors=[];completed_chunks=set()
    for name in sorted(path.glob('chunk_*.npz')):
        if '.pending.' in name.name: continue
        with np.load(name,allow_pickle=False) as data:
            n=len(data['control_frame'])
            if any(len(data[k])!=n for k in data): errors.append(f'{name.name}: inconsistent row counts')
            rows.append({k:data[k].copy() for k in data})
            completed_chunks.add(int(name.stem.rsplit('_',1)[1]))
    if not rows: return dict(valid=False,errors=['no completed chunks'],manifest=manifest)
    allrows={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    n=len(allrows['control_frame'])
    if np.any(np.diff(allrows['control_frame'])<=0): errors.append('non-increasing control frame')
    if np.any(allrows['physics_end_ns']<allrows['first_physics_step_ns']): errors.append('physics clock order')
    valid_source=allrows['fresh_source']&allrows['pose_valid']&~allrows['startup']&~allrows['paused']
    if np.any(allrows['first_physics_step_ns'][valid_source]<allrows['inference_done_ns'][valid_source]): errors.append('inference/consumption clock order')
    for field in ('robot_qpos','robot_qvel','robot_ctrl','action'):
        if not np.isfinite(allrows[field]).all(): errors.append(f'{field}: nonfinite robot state')
    if allrows['pose_raw'].shape[1:]!=(33,4) or allrows['right_hand_raw'].shape[1:]!=(21,3): errors.append('landmark schema shape')
    video_count=0
    if check_video:
        for file in sorted(path.glob('human_camera_*.mp4')):
            # A currently-open segment is not finalized yet; inspect completed chunks only.
            chunk=int(file.stem.rsplit('_',1)[1])
            if chunk not in completed_chunks: continue
            cap=cv2.VideoCapture(str(file))
            actual=0
            while True:
                ok,_=cap.read()
                if not ok: break
                actual+=1
            cap.release()
            expected=int(np.sum(allrows['video_chunk']==chunk))
            if actual!=expected: errors.append(f'{file.name}: {actual} decoded frames vs {expected} timestamps')
            video_count+=actual
    def stats(values):
        return None if len(values)==0 else dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=float(np.max(values)))
    fresh=allrows['fresh_source']&~allrows['startup']
    per_side=None
    if 'side_pose_valid' in allrows:
        per_side={}
        if allrows['side_pose_valid'].shape!=(n,2) or allrows['side_hand_valid'].shape!=(n,2): errors.append('bimanual validity schema shape')
        else:
            for index,side in enumerate(('left','right')):
                if allrows[f'{side}_hand_raw'].shape[1:]!=(21,3): errors.append(f'{side} hand schema shape')
                mask=fresh&allrows['side_pose_valid'][:,index]&~allrows['paused']
                per_side[side]=dict(pose_valid_fraction=float(np.mean(allrows['side_pose_valid'][fresh,index])) if fresh.any() else None,
                                    hand_valid_fraction=float(np.mean(allrows['side_hand_valid'][fresh,index])) if fresh.any() else None,
                                    host_read_to_first_step_ms=stats(allrows['read_to_step_ms'][mask]),
                                    closed_gripper_rows=int(np.sum(allrows['side_closed'][:,index])))
    return dict(valid=not errors,errors=errors,rows=n,decoded_video_frames=video_count,per_side=per_side,
                recording_finished=manifest['finished'],recording_dropped_rows=manifest['dropped_recording_rows'],
                recording_complete=bool(manifest['finished'] and manifest['dropped_recording_rows']==0 and not manifest['recording_error']),
                pose_valid_fraction=float(np.mean(allrows['pose_valid'][fresh])) if fresh.any() else None,
                hand_valid_fraction=float(np.mean(allrows['hand_valid'][fresh])) if fresh.any() else None,
                host_read_to_first_step_ms=stats(allrows['read_to_step_ms'][valid_source]),
                inference_ms=stats(allrows['inference_ms'][fresh]),
                closed_gripper_rows=int(np.sum(allrows['gripper_closed'])),
                scope='recording consistency and timing, not perception ground truth or task success')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
    p.add_argument('--no-video-check',action='store_true')
    args=p.parse_args();report=validate(args.path,not args.no_video_check)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if report['valid'] else 1)
