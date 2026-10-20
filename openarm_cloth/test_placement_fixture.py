"""Placement reset is a user fixture operation, not an execution cheat."""
import unittest
import mujoco
import numpy as np
from generic_object_scene import build_object_model,PlacementFixture


class PlacementTests(unittest.TestCase):
    def test_reset_restores_object_without_teleporting_robot(self):
        model=build_object_model('cylinder',[.045,.045,.075]);data=mujoco.MjData(model)
        fixture=PlacementFixture(model,data);original=data.qpos.copy()
        data.qpos[fixture.address:fixture.address+3]+=[.1,.1,-.4]
        data.qvel[fixture.dof:fixture.dof+6]=2.
        data.xfrc_applied[fixture.body]=3.
        data.qfrc_applied[fixture.dof:fixture.dof+6]=4.
        fixture.reset(data,placement_active=True)
        np.testing.assert_array_equal(data.qpos,original)
        np.testing.assert_array_equal(data.qvel[fixture.dof:fixture.dof+6],0.)
        np.testing.assert_array_equal(data.xfrc_applied[fixture.body],0.)

    def test_execution_cannot_use_reset_to_fake_a_grasp(self):
        model=build_object_model();data=mujoco.MjData(model);fixture=PlacementFixture(model,data)
        data.qpos[fixture.address+2]-=.4;saved=data.qpos.copy()
        with self.assertRaises(RuntimeError): fixture.reset(data,placement_active=False)
        np.testing.assert_array_equal(data.qpos,saved)


if __name__=='__main__': unittest.main()
