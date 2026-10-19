import unittest
import numpy as np
import mujoco
from human_teleop_core import PadCartesianIK,ArmPoseIK,arm_frame,human_arm_pose
from live_motion_scene import build_motion_scene


class PalmTests(unittest.TestCase):
    def test_anatomical_palmar_signs_preserve_facing_hands(self):
        w=np.zeros((33,4));w[:,3]=1
        w[11,:3]=[-.2,0,0];w[12,:3]=[.2,0,0];w[23,:3]=[-.15,-.5,0];w[24,:3]=[.15,-.5,0]
        w[13,:3]=[-.3,-.3,0];w[15,:3]=[-.3,-.1,0]
        w[14,:3]=[.3,-.3,0];w[16,:3]=[.3,-.1,0]
        h=np.zeros((21,3));h[9]=[0,.08,0];h[5]=[0,.04,-.03];h[17]=[0,.04,.03]
        left=human_arm_pose(w,h,side='left');right=human_arm_pose(w,h,side='right')
        self.assertLess(np.dot(left['palm_normal'],right['palm_normal']),-.999)
        np.testing.assert_allclose(left['palm_normal'],[0,-1,0],atol=1e-10)
        np.testing.assert_allclose(right['palm_normal'],[0,1,0],atol=1e-10)

    def test_raised_robot_palms_face_each_other_without_wrist_collapse(self):
        model,data=build_motion_scene();normals=[]
        for s in ('left','right'):
            pad=PadCartesianIK(model,data,initialize_grasp=False,side=s)
            solver=ArmPoseIK(pad,'twist',direction_tau=0,palm_alignment='absolute')
            upper,_=solver.geometry();fore=np.array([.5,0,.866]);fore/=np.linalg.norm(fore)
            frame=arm_frame(upper,fore);normal=np.array([0.,-1 if s=='left' else 1.,0.])
            pose=dict(upper=upper,fore=fore,rotation=frame,arm_rotation=frame,palm=True,palm_normal=normal)
            for _ in range(40): solver.update_target(pose,pad.ready_q)
            q=solver.solve(pad.ready_q,100)
            actual=pad.kin.xmat[pad.spec.body_id].reshape(3,3)[:,0].copy();normals.append(actual)
            self.assertGreater(np.dot(actual,normal),.98)
            np.testing.assert_allclose(q[5:],0,atol=1e-12)
            # C resets tracking but must not erase absolute inter-palm facing.
            solver.reset()
            for _ in range(40): solver.update_target(pose,q)
            solver.solve(q,50)
            self.assertGreater(np.dot(pad.kin.xmat[pad.spec.body_id].reshape(3,3)[:,0],normal),.98)
        self.assertLess(np.dot(*normals),-.97)

    def test_full_mode_also_preserves_absolute_palm_normal(self):
        model,data=build_motion_scene();pad=PadCartesianIK(model,data,initialize_grasp=False,side='left')
        solver=ArmPoseIK(pad,'full',palm_alignment='absolute');u,v=solver.geometry();frame=arm_frame(u,v)
        normal=frame[:,0]
        pose=dict(upper=u,fore=v,rotation=frame,arm_rotation=frame,palm=True,palm_normal=normal)
        for _ in range(40): solver.update_target(pose,pad.ready_q)
        np.testing.assert_allclose(solver.rotation[:,0],normal,atol=1e-8)


if __name__=='__main__': unittest.main()
