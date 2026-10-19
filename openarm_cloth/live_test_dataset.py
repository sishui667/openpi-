"""Bounded asynchronous local test recording. User explicitly opted into video.

Video contains consumed NEW webcam observations, not every sensor exposure.
Per-frame capture timestamps preserve variable timing; MP4 FPS is only playback.
"""
import csv
import json
from pathlib import Path
import queue
import threading

import cv2
import numpy as np


class TestDatasetRecorder:
    def __init__(self,path,metadata,save_video=True,chunk_rows=40):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.metadata=dict(metadata,schema='openarm_live_test_v1',save_video=save_video,
                           video_scope='consumed new USB observations; not guaranteed complete 30 Hz capture',
                           world_units='monocular model estimated metres, not calibrated physical measurement',
                           privacy='local only; human video and landmarks; no upload',task_success='not assessed')
        self.save_video=save_video;self.chunk_rows=chunk_rows
        self.queue=queue.Queue(maxsize=128);self.dropped_rows=0;self.saved_rows=0
        self.video_frames=0;self.chunks=0;self.error=None;self.closed=False;self.finished=False
        self.metadata_lock=threading.Lock()
        self._write_manifest()
        self.worker=threading.Thread(target=self._run,daemon=True);self.worker.start()

    def _write_manifest(self):
        with self.metadata_lock:
            payload=dict(self.metadata,saved_rows=self.saved_rows,chunks=self.chunks,
                         video_frames=self.video_frames,dropped_recording_rows=self.dropped_rows,
                         recording_error=self.error,finished=self.finished)
        tmp=self.path/'manifest.pending.json'
        tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
        tmp.replace(self.path/'manifest.json')

    def update_metadata(self,**fields):
        with self.metadata_lock: self.metadata.update(fields)

    def submit(self,row,frame=None):
        if self.closed or self.error: return False
        # All row arrays supplied by caller are snapshots, never live MjData views.
        try: self.queue.put_nowait((row,frame));return True
        except queue.Full: self.dropped_rows+=1;return False

    def _flush(self,batch):
        if not batch: return
        arrays={key:np.stack([row[key] for row in batch]) for key in batch[0]}
        tmp=self.path/f'chunk_{self.chunks:05d}.pending.npz'
        np.savez_compressed(tmp,**arrays)
        tmp.replace(self.path/f'chunk_{self.chunks:05d}.npz')
        self.saved_rows+=len(batch);self.chunks+=1;batch.clear();self._write_manifest()

    def _run(self):
        video=None;batch=[];clip_frame=0
        try:
            with (self.path/'video_timestamps.csv').open('w',newline='',encoding='utf-8') as f:
                timestamps=csv.writer(f)
                timestamps.writerow(['video_frame_index','video_chunk','frame_in_chunk','source_seq','capture_read_end_ns','control_frame'])
                while not self.closed or not self.queue.empty():
                    try: row,frame=self.queue.get(timeout=.2)
                    except queue.Empty:
                        if batch:
                            if video is not None: video.release();video=None;clip_frame=0
                            self._flush(batch);f.flush()
                        continue
                    video_index=-1
                    video_chunk=-1;frame_in_chunk=-1
                    if self.save_video and frame is not None:
                        if video is None:
                            height,width=frame.shape[:2]
                            video=cv2.VideoWriter(str(self.path/f'human_camera_{self.chunks:05d}.mp4'),
                                                  cv2.VideoWriter_fourcc(*'mp4v'),20,(width,height))
                            if not video.isOpened(): raise RuntimeError('MP4 video writer could not be opened')
                        video.write(frame);video_index=self.video_frames;self.video_frames+=1
                        video_chunk=self.chunks;frame_in_chunk=clip_frame;clip_frame+=1
                        timestamps.writerow([video_index,video_chunk,frame_in_chunk,int(row['source_seq']),int(row['capture_read_end_ns']),int(row['control_frame'])])
                    row=dict(row,video_frame_index=np.asarray(video_index,dtype=np.int64),
                             video_chunk=np.asarray(video_chunk),video_frame_in_chunk=np.asarray(frame_in_chunk));batch.append(row)
                    if len(batch)>=self.chunk_rows:
                        if video is not None: video.release();video=None;clip_frame=0
                        self._flush(batch);f.flush()
                if video is not None: video.release();video=None
                self._flush(batch)
        except Exception as error:
            self.error=f'{type(error).__name__}: {error}'
            print('DATASET RECORDING ERROR '+self.error,flush=True)
        finally:
            if video is not None: video.release()
            self.finished=self.error is None;self._write_manifest()

    def close(self):
        self.closed=True;self.worker.join(timeout=10)
        if self.worker.is_alive():
            print('Dataset writer still flushing; last completed NPZ chunks remain usable.',flush=True)


def array_or_nan(value,shape):
    return np.full(shape,np.nan) if value is None else np.asarray(value,dtype=float).copy()
