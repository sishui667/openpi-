import unittest
import numpy as np
from hand_association import associate_hands
from collision_safety import InterArmCollisionProjector
from scripts.replay_mujoco import physics_steps_for_frame, validate_position_actuators
import mujoco


class HandAssociationTests(unittest.TestCase):
    def pose(self):
        p=np.zeros((33,4));p[15]=[.2,.5,0,1];p[16]=[.8,.5,0,1]
        return p

    def test_global_assignment_ignores_detection_order(self):
        self.assertEqual(associate_hands([[.8,.5],[.2,.5]],["Right","Left"],self.pose()),[1,0])

    def test_duplicate_not_forced_onto_opposite_hand(self):
        assigned=associate_hands([[.21,.5],[.22,.5]],["Right","Right"],self.pose())
        self.assertEqual(assigned.count(0),1)
        self.assertEqual(assigned.count(None),1)
        self.assertNotIn(1,assigned)

    def test_far_detection_missing_not_fabricated(self):
        self.assertEqual(associate_hands([[.5,.1]],["Right"],self.pose()),[None])

    def test_temporary_pose_loss_uses_temporal_wrist(self):
        self.assertEqual(associate_hands([[.21,.5]],["Left"],None,[[.2,.5],[.8,.5]]),[0])


class CollisionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.projector=InterArmCollisionProjector()

    def test_nonfinite_angles_never_certified_as_safe(self):
        with self.assertRaises(ValueError): self.projector.is_safe(np.full(14,np.nan))

    def test_nonfinite_and_bad_units_gripper_rejected(self):
        for grip in ([np.nan,0],[-.7854,0],[1.1,0]):
            with self.assertRaises(ValueError): self.projector.is_safe(np.zeros(14),grip)

    def test_nonfinite_margin_rejected(self):
        with self.assertRaises(ValueError): InterArmCollisionProjector(margin=np.nan)


class ReplayPhysicsTests(unittest.TestCase):
    def test_fractional_frame_clock_does_not_slow_physics(self):
        steps=[physics_steps_for_frame(i,1/29.375,.002) for i in range(1000)]
        self.assertLess(abs(sum(steps)*.002-1000/29.375),.002)
        self.assertGreater(min(steps),1)

    def test_torque_actuator_not_mistaken_for_position(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody><actuator><motor joint="j"/><position joint="j" kp="10"/></actuator></mujoco>')
        with self.assertRaises(ValueError): validate_position_actuators(model,[0])
        validate_position_actuators(model,[1])


if __name__=="__main__": unittest.main(verbosity=2)
