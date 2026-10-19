"""Actual-model absolute wrist continuity regressions; no camera needed."""
import unittest

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import human_teleop_core as patched
from live_motion_scene import build_motion_scene


class AbsoluteWristTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model,cls.data=build_motion_scene()

    def make_solver(self,angle=0.,side='right',solver_class=None):
        pad=patched.PadCartesianIK(self.model,self.data,initialize_grasp=False,side=side)
        solver=(solver_class or patched.ArmPoseIK)(pad,'twist',direction_tau=0,palm_alignment='absolute')
        upper,fore=solver.geometry();frame=patched.arm_frame(upper,fore)
        pose=dict(upper=upper,fore=fore,rotation=frame,arm_rotation=frame,palm=False,palm_normal=None)
        solver.update_target(pose,pad.ready_q)
        base=float(np.arctan2(solver.robot_origin[1,0],solver.robot_origin[0,0]))
        self.observe(solver,pose,base,angle)
        return solver,pose,base

    def observe(self,solver,pose,base,angle,dt=.05):
        theta=base+np.deg2rad(angle)
        pose['palm']=True
        pose['palm_normal']=pose['arm_rotation'] @ np.array([np.cos(theta),np.sin(theta),0.])
        solver.update_target(pose,solver.pad.ready_q,dt)
        return float(np.rad2deg(solver.twist_target))

    def test_wrap_boundary_preserves_positive_stop(self):
        solver,pose,base=self.make_solver(179)
        before=solver.twist_target
        self.observe(solver,pose,base,-179)
        self.assertAlmostEqual(solver.twist_target,before,places=10)
        self.assertGreater(solver.absolute_angle,np.pi)

    def test_wrap_boundary_preserves_negative_stop(self):
        solver,pose,base=self.make_solver(-179,side='left')
        before=solver.twist_target
        self.observe(solver,pose,base,179)
        self.assertAlmostEqual(solver.twist_target,before,places=10)
        self.assertLess(solver.absolute_angle,-np.pi)

    def test_isolated_half_turn_cannot_change_command(self):
        solver,pose,base=self.make_solver(0)
        trace=[self.observe(solver,pose,base,a) for a in (180,0)]
        np.testing.assert_allclose(trace,0,atol=1e-10)
        self.assertIsNone(solver.absolute_candidate)

    def test_two_consistent_new_observations_confirm_reacquisition(self):
        solver,pose,base=self.make_solver(0)
        pose.update(palm=False,palm_normal=None)
        solver.update_target(pose,solver.pad.ready_q)
        self.assertAlmostEqual(self.observe(solver,pose,base,170),0)
        self.assertGreater(self.observe(solver,pose,base,175),89)
        self.assertIsNone(solver.absolute_candidate)
        self.assertEqual(solver.absolute_reacquisitions,1)

    def test_persistent_half_turn_during_continuous_visibility_is_not_confirmed(self):
        solver,pose,base=self.make_solver(0)
        trace=[self.observe(solver,pose,base,a) for a in (180,178,179,180,177)]
        np.testing.assert_allclose(trace,0,atol=1e-10)
        self.assertEqual(solver.absolute_rejected_total,5)
        self.assertEqual(solver.absolute_reacquisitions,0)
        self.assertTrue(solver.absolute_rejected_this_frame)
        self.assertIsNone(solver.absolute_candidate)
        self.assertAlmostEqual(self.observe(solver,pose,base,10),10)
        self.assertFalse(solver.absolute_rejected_this_frame)

    def test_repeated_fast_turns_cannot_freeze_gradual_real_rotation(self):
        solver,pose,base=self.make_solver(0)
        self.observe(solver,pose,base,180)
        self.observe(solver,pose,base,179)
        trace=[self.observe(solver,pose,base,a) for a in (15,30,45,60)]
        np.testing.assert_allclose(trace,[15,30,45,60],atol=1e-8)

    def test_missing_palm_clears_candidate_without_rezero(self):
        solver,pose,base=self.make_solver(20)
        pose.update(palm=False,palm_normal=None)
        solver.update_target(pose,solver.pad.ready_q)
        self.observe(solver,pose,base,-160)
        self.assertIsNotNone(solver.absolute_candidate)
        pose.update(palm=False,palm_normal=None)
        solver.update_target(pose,solver.pad.ready_q)
        self.assertIsNone(solver.absolute_candidate)
        self.assertAlmostEqual(np.rad2deg(solver.twist_target),20)
        self.assertAlmostEqual(self.observe(solver,pose,base,-155),20)
        self.assertLess(self.observe(solver,pose,base,-150),-89)

    def test_missing_pose_gap_does_not_confirm_old_candidate(self):
        solver,pose,base=self.make_solver(0)
        pose.update(palm=False,palm_normal=None)
        solver.update_target(pose,solver.pad.ready_q)
        self.observe(solver,pose,base,170)
        self.assertAlmostEqual(self.observe(solver,pose,base,175,dt=.15),0)
        self.assertGreater(self.observe(solver,pose,base,176),89)

    def test_source_gap_alone_enables_two_sample_reacquisition(self):
        solver,pose,base=self.make_solver(0)
        self.assertAlmostEqual(self.observe(solver,pose,base,170,dt=.15),0)
        self.assertGreater(self.observe(solver,pose,base,174),89)

    def test_degenerate_palm_projection_enables_reacquisition_without_reset(self):
        solver,pose,base=self.make_solver(10)
        pose['palm_normal']=pose['fore'].copy()
        solver.update_target(pose,solver.pad.ready_q)
        self.assertTrue(solver.absolute_reacquiring)
        self.assertAlmostEqual(np.rad2deg(solver.twist_target),10)
        self.assertAlmostEqual(self.observe(solver,pose,base,170),10)
        self.assertGreater(self.observe(solver,pose,base,175),89)

    def test_smooth_rotation_tracks_inside_limits_and_unwraps_beyond_them(self):
        solver,pose,base=self.make_solver(0)
        angles=[15,30,45,60,75,89,100,130,160,179,-179,-160]
        trace=[self.observe(solver,pose,base,a) for a in angles]
        np.testing.assert_allclose(trace[:6],angles[:6],atol=1e-8)
        self.assertTrue(all(89<t<=90 for t in trace[6:]))
        q=solver.solve(solver.pad.ready_q,60)
        np.testing.assert_allclose(q[5:],0,atol=1e-12)
        self.assertTrue(np.all(q>=solver.spec.q_range[:,0]))
        self.assertTrue(np.all(q<=solver.spec.q_range[:,1]))

    def test_reset_explicitly_clears_unwrapped_history(self):
        solver,pose,base=self.make_solver(179)
        self.observe(solver,pose,base,-179)
        solver.reset()
        self.assertIsNone(solver.absolute_previous)
        self.assertIsNone(solver.absolute_angle)
        self.assertIsNone(solver.absolute_candidate)
        self.assertIsNone(solver.absolute_arm_frame)
        self.observe(solver,pose,base,-179)
        self.assertLess(np.rad2deg(solver.twist_target),-89)

    def test_near_straight_elbow_frame_flip_cannot_turn_world_palm(self):
        pad=patched.PadCartesianIK(self.model,self.data,initialize_grasp=False,side='right')
        solver=patched.ArmPoseIK(pad,'twist',direction_tau=0,palm_alignment='absolute')
        upper=np.array([0.,0.,1.]);fore1=np.array([.06,0.,np.sqrt(1-.06**2)])
        fore2=fore1.copy();fore2[0]*=-1
        frame1=patched.arm_frame(upper,fore1);frame2=patched.arm_frame(upper,fore2)
        self.assertGreater(np.degrees(Rotation.from_matrix(frame2@frame1.T).magnitude()),179)
        pose=dict(upper=upper,fore=fore1,rotation=frame1,arm_rotation=frame1,palm=False,palm_normal=None)
        solver.update_target(pose,pad.ready_q)
        palm=frame1@solver.robot_origin[:,0]
        pose.update(palm=True,palm_normal=palm)
        for _ in range(30): solver.update_target(pose,pad.ready_q)
        before=solver.rotation.copy()
        pose.update(fore=fore2,rotation=frame2,arm_rotation=frame2)
        for _ in range(40): solver.update_target(pose,pad.ready_q)
        rotation_change=float(np.degrees(Rotation.from_matrix(solver.rotation@before.T).magnitude()))
        self.assertLess(rotation_change,10)
        self.assertGreater(float(np.dot(solver.rotation[:,0],palm)),.99)
        self.assertEqual(solver.absolute_rejected_total,0)
        turning=Rotation.from_rotvec(fore2*np.deg2rad(30)).apply(palm)
        pose['palm_normal']=turning
        for _ in range(30): solver.update_target(pose,pad.ready_q)
        self.assertGreater(float(np.dot(solver.rotation[:,0],turning)),.99)
        self.assertGreater(np.degrees(Rotation.from_matrix(solver.rotation@before.T).magnitude()),25)

    def test_absolute_frame_continuity_does_not_rewrite_input_pose(self):
        solver,pose,base=self.make_solver(0)
        original=pose['arm_rotation'].copy()
        solver._continuous_absolute_arm_frame(original @ np.diag([-1.,-1.,1.]))
        np.testing.assert_allclose(pose['arm_rotation'],original,atol=0)

if __name__=='__main__':
    unittest.main()
