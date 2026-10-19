from pathlib import Path
import tempfile
import time
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from bimanual_teleop import BimanualController
from live_motion_scene import build_motion_scene
from live_bimanual_runtime import BimanualControlLoop
from live_test_dataset import TestDatasetRecorder
from human_teleop_core import arm_frame


class MissingStream:
    error=None
    def snapshot(self): return None


class SyncTests(unittest.TestCase):
    def test_render_flags_use_scene_render_options_not_visual_flags(self):
        m,d=build_motion_scene();scene=mujoco.MjvScene(m,maxgeom=1)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        self.assertFalse(scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW])

    def test_extra_direction_filter_is_removed_only_from_live_bimanual(self):
        m,d=build_motion_scene();new=BimanualController(m,d);old=BimanualController(m,d,direction_tau=.065,arm_speed=1.8)
        for c in (new,old):
            solver=c.solvers['right'];u,v=solver.geometry();r=arm_frame(u,v)
            pose=dict(upper=u.copy(),fore=v.copy(),rotation=r,arm_rotation=r,palm=False)
            solver.update_target(pose,c.targets['right'])
            pose['upper']=Rotation.from_euler('x',30,degrees=True).apply(u)
            solver.update_target(pose,c.targets['right'])
            error=np.linalg.norm(solver.upper-pose['upper'])
            if c is new: self.assertLess(error,1e-10)
            else: self.assertGreater(error,.05)

    def test_slow_display_consumer_cannot_block_physics_owner(self):
        with tempfile.TemporaryDirectory(prefix='openarm_split_runtime_test_') as root:
            out=Path(root);recorder=TestDatasetRecorder(out/'dataset',{},save_video=False)
            m,d=build_motion_scene();loop=BimanualControlLoop(m,d,MissingStream(),recorder,out)
            loop.start()
            try:
                deadline=time.perf_counter()+3
                while loop.frames<2 and time.perf_counter()<deadline: time.sleep(.01)
                start_frames=loop.frames
                # Simulate a slow GUI waiting for a rendering operation. The GUI
                # never locks/steps live MjData; physics must keep advancing.
                time.sleep(.3)
                self.assertGreaterEqual(loop.frames-start_frames,3)
                snapshot=loop.snapshot()
                render=mujoco.MjData(m);render.qpos[:]=snapshot['qpos']
                original=d.qpos.copy();render.qpos[0]+=.1
                self.assertNotEqual(render.qpos[0],d.qpos[0])
                self.assertIsNone(loop.error)
            finally: loop.close();recorder.close()
            self.assertAlmostEqual(d.time,loop.frames*.05,places=7)
            self.assertGreater(recorder.saved_rows,0)


if __name__=='__main__': unittest.main()
