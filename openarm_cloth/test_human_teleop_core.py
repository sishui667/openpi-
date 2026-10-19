"""Controller checks; these do not establish live human grasp success."""
import unittest
import numpy as np
from human_teleop_core import RelativeWristMapper, PadCartesianIK, ArmPoseIK, human_right_pose, arm_frame
import mujoco
from scipy.spatial.transform import Rotation
from member_b_scene import build_scene


class MappingTests(unittest.TestCase):
    def test_first_observation_does_not_jump(self):
        m=RelativeWristMapper([.43,0,.50])
        np.testing.assert_allclose(m.update([8,2,-4]),[.43,0,.50])

    def test_invalid_holds_and_speed_is_limited(self):
        m=RelativeWristMapper([.43,0,.50]);m.update([0,0,0])
        old=m.target.copy();new=m.update([9,9,9])
        self.assertLessEqual(np.linalg.norm(new-old),.16*.05+1e-10)
        np.testing.assert_allclose(m.update([np.nan,0,0]),new)
        np.testing.assert_allclose(m.update(None),new)

    def test_recenter(self):
        m=RelativeWristMapper([.43,0,.50])
        m.recenter([3,4,5],[.44,-.02,.47])
        np.testing.assert_allclose(m.update([3,4,5]),[.44,-.02,.47])


class RobotIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.data=build_scene()
        cls.ik=PadCartesianIK(cls.model,cls.data)

    def test_center_low_target(self):
        _,error=self.ik.solve([.45,0,.43],self.ik.ready_q,100)
        self.assertLess(error,.002)

    def test_object_truth_cannot_change_robot_command(self):
        q1,_=self.ik.solve([.43,0,.46],self.ik.ready_q,80)
        j=self.model.joint('droid_object_freejoint').qposadr[0]
        self.data.qpos[j:j+3]=[7,8,9]
        q2,_=self.ik.solve([.43,0,.46],self.ik.ready_q,80)
        np.testing.assert_allclose(q1,q2,atol=1e-12)

    def test_pose_ik_follows_shoulder_elbow_and_wrist_rotation(self):
        pose_ik=ArmPoseIK(self.ik)
        for index,delta in ((1,.2),(3,.3),(6,.3)):
            target=self.ik.ready_q.copy();target[index]+=delta
            self.ik.kin.qpos[self.ik.spec.qpos]=target
            mujoco.mj_forward(self.model,self.ik.kin)
            pose_ik.upper,pose_ik.fore=pose_ik.geometry()
            pose_ik.rotation=self.ik.kin.xmat[self.ik.spec.body_id].reshape(3,3).copy()
            result=pose_ik.solve(self.ik.ready_q,45)
            self.assertGreater(np.linalg.norm(result-self.ik.ready_q),.1)
            self.assertLess(pose_ik.metrics['upper_deg'],2)
            self.assertLess(pose_ik.metrics['fore_deg'],2)
            self.assertLess(pose_ik.metrics['palm_deg'],3)

    def test_reacquiring_hand_does_not_jump_orientation(self):
        a=ArmPoseIK(self.ik)
        p=dict(upper=np.array([1.,0,0]),fore=np.array([0.,0,-1.]),rotation=np.eye(3),palm=False)
        a.update_target(p,self.ik.ready_q)
        before=a.rotation.copy()
        p['rotation']=Rotation.from_euler('x',90,degrees=True).as_matrix();p['palm']=True
        a.update_target(p,self.ik.ready_q)
        np.testing.assert_allclose(a.rotation,before,atol=1e-10)

    def test_neutral_palm_follows_arm_not_global_downward_ready_pose(self):
        a=ArmPoseIK(self.ik)
        q=self.ik.ready_q.copy();q[1]+=.2;q[3]+=.3;q[4:]=0
        a.kin.qpos[a.spec.qpos]=q;mujoco.mj_forward(self.model,a.kin)
        u,v=a.geometry();frame=arm_frame(u,v)
        pose=dict(upper=u,fore=v,rotation=frame,arm_rotation=frame,palm=True)
        for _ in range(40): a.update_target(pose,self.ik.ready_q)
        a.solve(self.ik.ready_q,50)
        self.assertLess(a.metrics['upper_deg'],1)
        self.assertLess(a.metrics['fore_deg'],1)
        self.assertLess(a.metrics['palm_deg'],3)

    def test_hand_occlusion_does_not_reset_wrist_zero(self):
        a=ArmPoseIK(self.ik)
        self.ik.kin.qpos[a.spec.qpos]=self.ik.ready_q
        mujoco.mj_forward(self.model,a.kin)
        u,v=a.geometry();frame=arm_frame(u,v)
        p=dict(upper=u,fore=v,rotation=frame,arm_rotation=frame,palm=True)
        a.update_target(p,self.ik.ready_q)
        origin=a.hand_origin.copy()
        p['rotation']=frame @ Rotation.from_euler('z',30,degrees=True).as_matrix()
        a.update_target(p,self.ik.ready_q)
        p['palm']=False;a.update_target(p,self.ik.ready_q)
        p['palm']=True;a.update_target(p,self.ik.ready_q)
        np.testing.assert_allclose(a.hand_origin,origin)
        self.assertGreater(Rotation.from_matrix(a.last_local_palm).magnitude(),.4)

    def test_neutral_wrist_turns_both_directions(self):
        a=ArmPoseIK(self.ik)
        neutral=self.ik.ready_q.copy();neutral[4:]=0
        self.ik.kin.qpos[a.spec.qpos]=neutral;mujoco.mj_forward(self.model,a.kin)
        u,v=a.geometry();a.upper=u;a.fore=v
        for angle in (-.5,.5):
            target=neutral.copy();target[6]=angle
            self.ik.kin.qpos[a.spec.qpos]=target;mujoco.mj_forward(self.model,a.kin)
            a.rotation=self.ik.kin.xmat[a.spec.body_id].reshape(3,3).copy()
            solution=a.solve(neutral,50)
            self.assertAlmostEqual(solution[6],angle,delta=.06)
            self.assertLess(a.metrics['palm_deg'],3)


class HumanPoseTests(unittest.TestCase):
    def body(self):
        w=np.zeros((33,4));w[:,3]=1
        w[11,:3]=[-.2,0,0];w[12,:3]=[.2,0,0]
        w[23,:3]=[-.15,-.5,0];w[24,:3]=[.15,-.5,0]
        w[14,:3]=[.25,-.25,0];w[16,:3]=[.25,-.25,-.25]
        return w

    def test_body_translation_does_not_change_pose(self):
        w=self.body();p=human_right_pose(w);w[:,:3]+=[7,3,-4];q=human_right_pose(w)
        for name in ('upper','fore','rotation'): np.testing.assert_allclose(p[name],q[name],atol=1e-12)

    def test_palm_turn_changes_orientation_with_fixed_wrist(self):
        w=self.body();h=np.zeros((21,3))
        h[9]=[0,.08,0];h[5]=[.04,.04,0];h[17]=[-.04,.04,0]
        p=human_right_pose(w,h)
        r=Rotation.from_euler('y',45,degrees=True).as_matrix()
        q=human_right_pose(w,h@r.T)
        np.testing.assert_allclose(p['upper'],q['upper'])
        self.assertTrue(p['palm'] and q['palm'])
        self.assertGreater(Rotation.from_matrix(q['rotation']@p['rotation'].T).magnitude(),.5)

    def test_straight_arm_is_finite(self):
        w=self.body();w[16,:3]=w[14,:3]+(w[14,:3]-w[12,:3])
        self.assertTrue(np.isfinite(human_right_pose(w)['rotation']).all())


if __name__=='__main__': unittest.main()
