import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from live_motion_scene import build_motion_scene
from human_teleop_core import PadCartesianIK,ArmPoseIK,arm_frame
from member_b_actuators import ActuatorController


class TwistTests(unittest.TestCase):
    def setUp(self):
        self.model,self.data=build_motion_scene()
        self.pad=PadCartesianIK(self.model,self.data,initialize_grasp=False)
        self.ik=ArmPoseIK(self.pad,wrist_mode='twist')
        u,v=self.ik.geometry();self.frame=arm_frame(u,v)
        self.pose=dict(upper=u,fore=v,arm_rotation=self.frame,rotation=self.frame,palm=True)
        self.ik.update_target(self.pose,self.pad.ready_q)

    def test_turn_with_spurious_bend_cannot_command_wrist_collapse(self):
        for angle in (-45,45):
            self.pose['rotation']=self.frame @ Rotation.from_euler('zx',[angle,30],degrees=True).as_matrix()
            for _ in range(40): self.ik.update_target(self.pose,self.pad.ready_q)
            q=self.ik.solve(self.pad.ready_q,60)
            np.testing.assert_allclose(q[5:],0,atol=1e-12)
            self.assertGreater(abs(q[4]),.4)
            self.assertLess(self.ik.metrics['fore_deg'],3)

    def test_ambiguous_single_frame_flip_holds_previous_twist(self):
        self.pose['rotation']=self.frame @ Rotation.from_euler('z',180,degrees=True).as_matrix()
        self.ik.update_target(self.pose,self.pad.ready_q)
        self.assertAlmostEqual(self.ik.twist_target,0)

    def test_wrist_bend_physically_holds_neutral_while_turning(self):
        self.pose['rotation']=self.frame @ Rotation.from_euler('z',45,degrees=True).as_matrix()
        for _ in range(40): self.ik.update_target(self.pose,self.pad.ready_q)
        q=self.ik.solve(self.pad.ready_q,60)
        c=ActuatorController(self.model,self.data)
        c.set_targets({'left':self.data.qpos[c.specs['left'].qpos].copy(),'right':q},{'left':.044,'right':.044})
        for _ in range(2400): c.apply();mujoco.mj_step(self.model,self.data)
        actual=self.data.qpos[c.specs['right'].qpos]
        self.assertLess(np.max(abs(actual[5:])),np.deg2rad(3))
        self.assertLess(abs(actual[4]-q[4]),np.deg2rad(3))


if __name__=='__main__': unittest.main()
