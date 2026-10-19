"""Replay identical local raw camera frames through an unmodified stream API.

No USB camera, simulator, recorder retention, model training, or upload is used.
Coverage is NOT accuracy: eligibility comes from the original recording's body
visibility, not manually labelled hand truth. Inference times are offline wall
times and must not be reported as end-to-end live latency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time as real_time
from types import SimpleNamespace

# Match the live entry point before numerical/OpenCV libraries initialize.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE


def selected_frames(dataset, start, end, stride):
    with (dataset / 'video_timestamps.csv').open(newline='', encoding='utf-8') as f:
        stamps = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
    origin = stamps[0]['capture_read_end_ns']
    selected = [r for r in stamps if start <= (r['capture_read_end_ns'] - origin) / 1e9 <= end][::stride]
    if not selected:
        raise ValueError('No raw source frames inside selected source-time range')
    wanted = {r['control_frame'] for r in selected}
    originals = {}
    for path in sorted(dataset.glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            for idx, frame in enumerate(z['control_frame']):
                if int(frame) in wanted:
                    originals[int(frame)] = {
                        'pose_norm': z['pose_norm'][idx].copy(),
                        'side_pose_valid': z['side_pose_valid'][idx].copy(),
                        'side_hand_valid': z['side_hand_valid'][idx].copy(),
                    }
    if len(originals) != len(selected):
        raise ValueError(f'Incomplete original row alignment: {len(originals)} for {len(selected)} source frames')
    for row in selected:
        row.update(originals[row['control_frame']])
        row['source_seconds'] = (row['capture_read_end_ns'] - origin) / 1e9
    return selected, len(stamps)


def raw_palm(hand, side):
    if hand is None:
        return np.full(3, np.nan)
    h = np.asarray(hand)
    # Camera-frame palm-plane normal, unaffected by body-frame estimation.
    n = np.cross(h[9] - h[0], h[5] - h[17])
    if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
        return np.full(3, np.nan)
    return n / np.linalg.norm(n) * (-1 if side == 'left' else 1)


def control_twist(arm_pose):
    """Actual filtered palm projected into the upper/forearm frame."""
    if arm_pose is None or arm_pose.get('palm_normal') is None:
        return float('nan')
    local = np.asarray(arm_pose['arm_rotation']).T @ np.asarray(arm_pose['palm_normal'])
    if not np.isfinite(local).all() or np.linalg.norm(local[:2]) <= .25:
        return float('nan')
    return float(np.arctan2(local[1], local[0]))


class ReplayState:
    def __init__(self, dataset, frames):
        self.dataset = dataset
        self.frames = frames
        self.owner = None
        self.ready = threading.Event()
        self.results = []
        self.index = 0
        self.current = None
        self.read_wall = None
        self.capture_ns = 0
        self.cap = None
        self.clip = None
        self.clip_position = 0
        self.eof = False

    def clock_ns(self):
        if self.read_wall is None:
            return self.frames[0]['capture_read_end_ns']
        return self.capture_ns + real_time.perf_counter_ns() - self.read_wall

    def collect(self):
        if self.current is None:
            return
        result = self.owner.snapshot()
        if result is None or result['seq'] != self.current['source_seq']:
            raise RuntimeError('Stream output does not match previous replay source frame')
        eligible = []
        for side_index, wrist in enumerate((15, 16)):
            pt = self.current['pose_norm'][wrist]
            eligible.append(bool(self.current['side_pose_valid'][side_index] and
                                 np.isfinite(pt).all() and pt[3] >= .6 and
                                 .06 < pt[0] < .94 and .06 < pt[1] < .94))
        self.results.append(dict(
            source_seq=self.current['source_seq'],
            control_frame=self.current['control_frame'],
            video_frame_index=self.current['video_frame_index'],
            source_seconds=self.current['source_seconds'],
            eligible=eligible,
            original_hand_valid=self.current['side_hand_valid'].astype(bool).tolist(),
            hand_valid=[bool(result['arms'][s]['hand_valid']) for s in ('left', 'right')],
            pose_valid=[result['arms'][s]['pose'] is not None for s in ('left', 'right')],
            palm_raw=[raw_palm(result['arms'][s]['hand_raw'], s) for s in ('left', 'right')],
            twist_angle=[control_twist(result['arms'][s]['pose']) for s in ('left', 'right')],
            hand_norm=[np.full((21, 3), np.nan) if result['arms'][s]['hand_norm'] is None
                       else result['arms'][s]['hand_norm'].copy() for s in ('left', 'right')],
            inference_wall_ms=(real_time.perf_counter_ns() - self.read_wall) / 1e6,
            pose_ms=result['pose_ms'],
            hand_ms=result['hand_ms'],
        ))
        if len(self.results) % 100 == 0:
            valid = np.array([r['hand_valid'] for r in self.results[-100:]])
            print(f"PROGRESS {len(self.results)}/{len(self.frames)} t={self.current['source_seconds']:.2f}s "
                  f"recent hand coverage L={valid[:, 0].mean():.3f} R={valid[:, 1].mean():.3f}", flush=True)
        self.current = None

    def read_new(self, sequence):
        if not self.ready.wait(60):
            raise RuntimeError('Replay owner not bound')
        self.collect()
        if self.index >= len(self.frames):
            # Normal stream loop exit, not a fake read failure. The repeated
            # final frame's result is deliberately not collected a second time.
            self.eof = True
            self.owner.closed = True
            return sequence, self.capture_ns, self.capture_ns, self.last_image
        row = self.frames[self.index]
        chunk, frame = row['video_chunk'], row['frame_in_chunk']
        if self.clip != chunk:
            if self.cap is not None:
                self.cap.release()
            path = self.dataset / f'human_camera_{chunk:05d}.mp4'
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open original video {path}')
            self.clip = chunk
            self.clip_position = 0
        if frame != self.clip_position:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode original chunk {chunk} frame {frame}')
        self.clip_position = frame + 1
        self.index += 1
        self.current = row
        self.capture_ns = row['capture_read_end_ns']
        self.read_wall = real_time.perf_counter_ns()
        self.last_image = image
        return row['source_seq'], self.capture_ns, self.capture_ns, image

    def close(self):
        if self.cap is not None:
            self.cap.release()


def percentiles(values):
    a = np.asarray(values)
    return {'n': len(a), **{f'p{q}': float(np.percentile(a, q)) if len(a) else None for q in (50, 95, 100)}}


def summarize(results):
    eligible = np.array([r['eligible'] for r in results], bool)
    valid = np.array([r['hand_valid'] for r in results], bool)
    original = np.array([r['original_hand_valid'] for r in results], bool)
    palms = np.array([r['palm_raw'] for r in results])
    twists = np.array([r['twist_angle'] for r in results])
    t = np.array([r['source_seconds'] for r in results])
    summary = {'source_frames_processed': len(results),
               'actual_source_time_range_s': [float(t[0]), float(t[-1])],
               'coverage_is_not_accuracy': True,
               'inference_timing_scope': 'offline wall time only, not live optical-to-robot latency',
               'inference_wall_ms': percentiles([r['inference_wall_ms'] for r in results])}
    for index, side in enumerate(('left', 'right')):
        mask = eligible[:, index]
        finite = np.isfinite(palms[:, index]).all(1)
        pairs = finite[1:] & finite[:-1] & (np.diff(t) <= .2)
        dot = np.sum(palms[1:, index] * palms[:-1, index], axis=1)
        jumps = np.degrees(np.arccos(np.clip(dot[pairs], -1, 1)))
        summary[side] = {
            'original_body_visible_inframe_eligible_n': int(mask.sum()),
            'hand_found_eligible_n': int((valid[:, index] & mask).sum()),
            'hand_coverage_when_original_body_visible': float(valid[mask, index].mean()) if mask.any() else None,
            'recorded_original_hand_coverage_same_rows': float(original[mask, index].mean()) if mask.any() else None,
            'raw_palm_interframe_jump_deg': percentiles(jumps),
            'raw_palm_jumps_over_60deg_n': int((jumps > 60).sum()),
            'raw_palm_jumps_over_90deg_n': int((jumps > 90).sum()),
        }
        twist_finite = np.isfinite(twists[:, index])
        twist_pairs = twist_finite[1:] & twist_finite[:-1] & (np.diff(t) <= .2)
        twist_delta = np.diff(twists[:, index])[twist_pairs]
        twist_jump = np.degrees(np.abs(np.arctan2(np.sin(twist_delta), np.cos(twist_delta))))
        summary[side]['filtered_control_twist_interframe_jump_deg'] = percentiles(twist_jump)
        summary[side]['filtered_control_twist_jumps_over_110deg_n'] = int((twist_jump > 110).sum())
    both = eligible.all(1)
    summary['both'] = {'original_body_visible_inframe_eligible_n': int(both.sum()),
                       'both_hands_found_n': int((valid.all(1) & both).sum()),
                       'hand_coverage_when_original_body_visible': float(valid[both].all(1).mean()) if both.any() else None,
                       'recorded_original_hand_coverage_same_rows': float(original[both].all(1).mean()) if both.any() else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, default=HERE / 'live_bimanual_stream.py')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--start', type=float, default=55)
    parser.add_argument('--end', type=float, default=115)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--pose-model', choices=('lite', 'full'), default='lite')
    parser.add_argument('--output', type=Path, required=True, help='Output prefix outside live recording directories')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be >= 1')
    if args.output.with_suffix('.json').exists() or args.output.with_suffix('.npz').exists():
        parser.error('Output prefix already exists; choose a new prefix')
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('Do not place benchmark output inside source recording')
    frames, total = selected_frames(args.dataset, args.start, args.end, args.stride)
    print(f'SOURCE selected={len(frames)} total_saved_raw_frames={total} '
          f"range={frames[0]['source_seconds']:.3f}..{frames[-1]['source_seconds']:.3f}s stride={args.stride}", flush=True)
    # Import the real robot core first so its ROOT stays authoritative. Staged
    # stream files are loaded directly, never shadowing the real core module.
    sys.path.insert(0, str(REPO.parent / 'pose2arm-main'))
    sys.path.insert(0, str(REPO))
    import human_teleop_core
    assert human_teleop_core.ROOT == REPO
    module_hash = hashlib.sha256(args.module.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('benchmark_stream_under_test', args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = ReplayState(args.dataset, frames)

    class ReplayCamera:
        def __init__(self, *args, **kwargs):
            self.metadata = {'device': 'offline original raw MP4 replay', 'reported_width': 1280,
                             'reported_height': 720, 'timestamps': 'original capture_read_end_ns'}

        def read_new(self, sequence):
            return state.read_new(sequence)

        def close(self):
            state.close()

    module.USBLatestFrame = ReplayCamera
    module.time = SimpleNamespace(perf_counter_ns=state.clock_ns)
    start_wall = real_time.perf_counter()
    stream = module.BimanualHumanStream(pose_model=args.pose_model)
    state.owner = stream
    state.ready.set()
    try:
        while stream.thread.is_alive():
            stream.thread.join(timeout=.5)
    except KeyboardInterrupt:
        stream.close()
        raise
    if stream.error:
        raise RuntimeError(stream.error)
    if not state.eof or len(state.results) != len(frames):
        raise RuntimeError(f'Incomplete replay: {len(state.results)}/{len(frames)}, eof={state.eof}')
    stats = summarize(state.results)
    stats.update(module=str(args.module), module_sha256=module_hash,
                 dataset=str(args.dataset), pose_model=args.pose_model,
                 requested_source_time_range_s=[args.start, args.end], stride=args.stride,
                 total_saved_raw_frames=total, replay_wall_seconds=real_time.perf_counter() - start_wall,
                 cv2_num_threads=cv2.getNumThreads(),
                 openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                 omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                 source_time_origin='first saved video capture timestamp; startup control rows excluded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.npz'),
                        **{key: np.array([r[key] for r in state.results]) for key in state.results[0]})
    print('RESULT ' + json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
