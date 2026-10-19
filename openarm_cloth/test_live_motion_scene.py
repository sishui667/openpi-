import unittest
import mujoco
import numpy as np
from live_motion_scene import build_motion_scene
from human_teleop_core import PadCartesianIK,ArmPoseIK
from member_b_actuators import ActuatorController


class MotionSceneTests(unittest.TestCase):
    def test_no_task_objects_or_free_object_state(self):
        m,d=build_motion_scene()
        self.assertEqual(m.nq,18)
        self.assertEqual(m.npair,0)
        self.assertEqual(m.neq,2)  # finger mimic only, no grasp welds
        for kind,name in ((mujoco.mjtObj.mjOBJ_GEOM,'folding_table'),
                          (mujoco.mjtObj.mjOBJ_BODY,'droid_object'),
                          (mujoco.mjtObj.mjOBJ_SITE,'cube_spawn_reference')):
            self.assertEqual(mujoco.mj_name2id(m,kind,name),-1)

    def test_pose_controller_and_full_physics_work_without_table(self):
        m,d=build_motion_scene();ik=PadCartesianIK(m,d,initialize_grasp=False)
        np.testing.assert_allclose(ik.ready_q[4:],0)
        a=ArmPoseIK(ik);a.upper,a.fore=a.geometry()
        a.rotation=ik.kin.xmat[ik.spec.body_id].reshape(3,3).copy()
        q=a.solve(ik.ready_q)
        c=ActuatorController(m,d)
        c.set_targets({'left':d.qpos[c.specs['left'].qpos].copy(),'right':q},{'left':.044,'right':.044})
        for _ in range(400): c.apply();mujoco.mj_step(m,d)
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertAlmostEqual(d.time,.2,places=8)


if __name__=='__main__': unittest.main()
