"""Diagnostic events must not count duplicate, absent, or paused observations."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from live_test_cycle import analyze_useful
from live_bimanual_runtime import BimanualControlLoop
from live_motion_scene import build_motion_scene


class WristDiagnosticTests(unittest.TestCase):
    def data(self, path, extra=True):
        (path / 'manifest.json').write_text(json.dumps({'arms':['left','right']}), encoding='utf-8')
        values=dict(fresh_source=np.array([1,0,1,1,1], dtype=bool),
                    paused=np.array([0,0,0,1,0], dtype=bool), startup=np.zeros(5,dtype=bool),
                    side_pose_valid=np.array([[1,1],[1,1],[0,1],[1,1],[1,1]], dtype=bool),
                    side_hand_valid=np.array([[1,1],[1,1],[1,1],[1,1],[1,0]], dtype=bool),
                    side_angle_errors_deg=np.zeros((5,2,3)))
        if extra:
            for side in ('left','right'):
                values[f'{side}_wrist_rejected']=np.ones(5,dtype=bool)
                values[f'{side}_wrist_limited']=np.ones(5,dtype=bool)
        np.savez_compressed(path / 'chunk_00000.npz', **values)

    def test_only_fresh_unpaused_visible_observations_count(self):
        with tempfile.TemporaryDirectory(prefix='openarm_wrist_diag_') as directory:
            path=Path(directory);self.data(path)
            report=analyze_useful(path)
            self.assertEqual(report['wrist_diagnostics']['left']['rejected_observations'],2)
            self.assertEqual(report['wrist_diagnostics']['right']['rejected_observations'],2)
            self.assertEqual(report['wrist_diagnostics']['left']['joint_limit_observations'],2)

    def test_existing_recordings_without_new_fields_remain_readable(self):
        with tempfile.TemporaryDirectory(prefix='openarm_wrist_diag_') as directory:
            path=Path(directory);self.data(path,extra=False)
            self.assertIsNone(analyze_useful(path)['wrist_diagnostics'])

    def test_runtime_event_flags_are_not_repeated_or_counted_while_paused(self):
        model,data=build_motion_scene()
        loop=BimanualControlLoop(model,data,None,SimpleNamespace(save_video=False),Path('.'))
        for side in ('left','right'):
            loop.control.valid[side]=True;loop.control.hand_valid[side]=True
            loop.control.solvers[side].absolute_rejected_this_frame=True
        def event(fresh,paused):
            loop.paused=paused
            return loop._row(None,fresh,1,2,np.zeros((2,3)),0.,0.)
        self.assertFalse(event(False,False)['left_wrist_rejected'])
        self.assertFalse(event(True,True)['left_wrist_rejected'])
        self.assertTrue(event(True,False)['left_wrist_rejected'])
        loop.control.hand_valid['left']=False
        self.assertFalse(event(True,False)['left_wrist_rejected'])
        self.assertTrue(event(True,False)['right_wrist_rejected'])


if __name__=='__main__': unittest.main()
