import unittest
import mujoco
import numpy as np
from human_teleop_core import PadCartesianIK,ArmPoseIK,human_arm_pose
from live_motion_scene import build_motion_scene
from live_bimanual_stream import assign_candidates
from bimanual_teleop import BimanualController
from human_gripper_gestures import fist_features
from test_human_gripper_gestures import hand


class BimanualTests(unittest.TestCase):
    def setUp(self): self.model,self.data=build_motion_scene()

    def robot_pose(self,control,side,change=0):
        pad=control.pads[side];q=pad.ready_q.copy()
        q[1]+=change if side=='right' else -change;q[3]+=change
        pad.kin.qpos[pad.spec.qpos]=q;mujoco.mj_forward(self.model,pad.kin)
        u,v=control.solvers[side].geometry()
        from human_teleop_core import arm_frame
        frame=arm_frame(u,v)
        return dict(upper=u,fore=v,rotation=frame,arm_rotation=frame,palm=True)

    def observation(self,control,seq,left_closed=False,right_closed=False,change=0):
        return dict(seq=seq,read_end=1000000000+seq*50000000,arms={s:dict(
            pose=self.robot_pose(control,s,change),hand_valid=True,
            fist=fist_features(hand(4 if (left_closed if s=='left' else right_closed) else 0))) for s in ('left','right')})

    def test_each_model_side_solves_its_own_pose(self):
        for side in ('left','right'):
            pad=PadCartesianIK(self.model,self.data,initialize_grasp=False,side=side)
            solver=ArmPoseIK(pad,'twist');q=pad.ready_q.copy()
            q[1]+=.15 if side=='right' else -.15;q[3]+=.2;q[4]=.3
            pad.kin.qpos[pad.spec.qpos]=q;mujoco.mj_forward(self.model,pad.kin)
            solver.upper,solver.fore=solver.geometry()
            solver.rotation=pad.kin.xmat[pad.spec.body_id].reshape(3,3).copy()
            result=solver.solve(pad.ready_q,50)
            self.assertLess(solver.metrics['upper_deg'],2)
            self.assertLess(solver.metrics['fore_deg'],2)
            self.assertLess(solver.metrics['palm_deg'],3)
            np.testing.assert_allclose(result[5:],0,atol=1e-12)

    def test_fists_independently_close_the_corresponding_gripper(self):
        c=BimanualController(self.model,self.data)
        for seq in (1,2):
            obs=self.observation(c,seq,left_closed=True)
            c.update(obs,obs['read_end']+1000000)
        self.assertEqual(c.closed,{'left':1,'right':0})
        self.assertEqual(c.fingers,{'left':0.,'right':.044})
        for seq in (3,4):
            obs=self.observation(c,seq,right_closed=True)
            c.update(obs,obs['read_end']+1000000)
        self.assertEqual(c.closed,{'left':0,'right':1})
        for _ in range(1600): c.actuators.apply();mujoco.mj_step(self.model,self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())

    def test_missing_left_arm_holds_only_left(self):
        c=BimanualController(self.model,self.data);left=c.targets['left'].copy()
        obs=self.observation(c,1,change=.2);obs['arms']['left']['pose']=None
        c.update(obs,obs['read_end']+1000000)
        np.testing.assert_allclose(c.targets['left'],left)
        self.assertGreater(np.linalg.norm(c.targets['right']-c.pads['right'].ready_q),.01)

    def test_stale_observation_does_not_move_either_arm(self):
        c=BimanualController(self.model,self.data);before={s:c.targets[s].copy() for s in c.targets}
        obs=self.observation(c,1,change=.2);c.update(obs,obs['read_end']+1000000000)
        for s in c.targets: np.testing.assert_allclose(c.targets[s],before[s])

    def test_left_visibility_does_not_disable_valid_right_arm(self):
        w=np.zeros((33,4));w[:,3]=1
        w[11,:3]=[-.2,0,0];w[12,:3]=[.2,0,0];w[23,:3]=[-.15,-.5,0];w[24,:3]=[.15,-.5,0]
        w[13,:3]=[-.25,-.25,0];w[15,:3]=[-.25,-.25,-.25]
        w[14,:3]=[.25,-.25,0];w[16,:3]=[.25,-.25,-.25];w[13,3]=0
        self.assertIsNone(human_arm_pose(w,side='left'))
        self.assertIsNotNone(human_arm_pose(w,side='right'))


class AssociationTests(unittest.TestCase):
    def pose(self):
        p=np.zeros((33,4));p[:,3]=1;p[15,:2]=[.3,.5];p[16,:2]=[.7,.5];return p

    def candidate(self,wrist,label='Left',score=.9):
        return dict(wrist=np.asarray(wrist),label=label,score=score)

    def test_anatomical_side_follows_pose_not_display_mirror(self):
        p=self.pose();previous=np.full((2,2),np.nan)
        assigned=assign_candidates([self.candidate([.3,.5]),self.candidate([.7,.5],'Right')],p,previous)
        np.testing.assert_allclose(assigned['left']['wrist'],p[15,:2])
        np.testing.assert_allclose(assigned['right']['wrist'],p[16,:2])

    def test_duplicate_crop_cannot_drive_both_grippers(self):
        assigned=assign_candidates([self.candidate([.3,.5]),self.candidate([.305,.5])],self.pose(),np.full((2,2),np.nan))
        self.assertEqual(set(assigned),{'left'})

    def test_crossed_hands_keep_anatomical_identity(self):
        p=self.pose();p[[15,16],:2]=p[[16,15],:2]
        assigned=assign_candidates([self.candidate([.3,.5]),self.candidate([.7,.5],'Right')],p,np.array([[.3,.5],[.7,.5]]))
        np.testing.assert_allclose(assigned['left']['wrist'],[.7,.5])
        np.testing.assert_allclose(assigned['right']['wrist'],[.3,.5])


if __name__=='__main__': unittest.main()
