import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from live_test_dataset import TestDatasetRecorder
from validate_live_test_dataset import validate


def row(i):
    return dict(control_frame=np.asarray(i),source_seq=np.asarray(i),fresh_source=np.asarray(True),
                capture_read_end_ns=np.asarray(i*100+1,dtype=np.int64),
                inference_done_ns=np.asarray(i*100+2,dtype=np.int64),
                first_physics_step_ns=np.asarray(i*100+3,dtype=np.int64),
                physics_end_ns=np.asarray(i*100+4,dtype=np.int64),
                startup=np.asarray(False),paused=np.asarray(False),pose_valid=np.asarray(True),hand_valid=np.asarray(True),
                pose_raw=np.zeros((33,4)),right_hand_raw=np.zeros((21,3)),robot_qpos=np.zeros(18),
                robot_qvel=np.zeros(18),robot_ctrl=np.zeros(18),action=np.zeros(18),
                read_to_step_ms=np.asarray(.002),inference_ms=np.asarray(.001),gripper_closed=np.asarray(i%2))


class DatasetTests(unittest.TestCase):
    def test_segmented_video_and_last_partial_chunk_are_synchronized(self):
        with tempfile.TemporaryDirectory(prefix='openarm_recording_test_') as root:
            p=Path(root)/'dataset';writer=TestDatasetRecorder(p,{},chunk_rows=2)
            for i in range(5):
                self.assertTrue(writer.submit(row(i),np.full((48,64,3),i*40,np.uint8)))
            writer.close()
            self.assertIsNone(writer.error)
            report=validate(p)
            self.assertTrue(report['valid'],report)
            self.assertEqual(report['rows'],5)
            self.assertEqual(report['decoded_video_frames'],5)
            self.assertEqual(len(list(p.glob('human_camera_*.mp4'))),3)
            self.assertTrue(report['recording_finished'])

    def test_explicit_video_optout_still_records_numeric_data(self):
        with tempfile.TemporaryDirectory(prefix='openarm_numeric_test_') as root:
            p=Path(root)/'dataset';writer=TestDatasetRecorder(p,{},save_video=False)
            writer.submit(row(0),np.zeros((48,64,3),np.uint8));writer.close()
            self.assertFalse(list(p.glob('*.mp4')))
            self.assertTrue(validate(p)['valid'])
            self.assertFalse(writer.submit(row(1)))


if __name__=='__main__': unittest.main()
