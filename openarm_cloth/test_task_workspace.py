"""Fixture-only invariants: physical table, inset and immutable calibration."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np

from demo_grasp_viewer import (SCENE, FINGER_OPEN, build_arm_specs,
                              build_finger_actuators, apply_controller,
                              initialize_observation_posture,
                              interactive_cube_placement,
                              repair_left_finger_position_servos,
                              place_cube_for_blind_test)
from task_workspace import (CONFIG, TABLE_TOP, TABLE_X, TABLE_Y,
                            constrain_initial_placement, load_task_model)


class WorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_task_model(SCENE)

    def test_physical_size_and_unchanged_cameras(self):
        old = mujoco.MjModel.from_xml_path(str(SCENE))
        np.testing.assert_allclose(self.model.geom("folding_table").pos,
                                   CONFIG["table_center_m"])
        np.testing.assert_allclose(self.model.geom("folding_table").size,
                                   CONFIG["table_half_size_m"])
        self.assertAlmostEqual(float(self.model.geom("folding_table").pos[2] +
                                     self.model.geom("folding_table").size[2]), TABLE_TOP)
        for name in ("cam_pos", "cam_quat", "cam_fovy", "cam_bodyid"):
            np.testing.assert_array_equal(getattr(old, name), getattr(self.model, name))

    def test_cube_diagonal_fits_at_all_corners(self):
        centre = np.asarray(CONFIG["table_center_m"][:2])
        half = np.asarray(CONFIG["table_half_size_m"][:2])
        diagonal = np.sqrt(2) * CONFIG["cube_half_size_m"]
        for x in TABLE_X:
            for y in TABLE_Y:
                self.assertTrue(np.all(np.abs([x, y] - centre) + diagonal < half))

    def test_trimmed_table_has_only_the_specified_support_rim(self):
        if "support_rim_m" not in CONFIG:
            self.skipTest("configuration does not specify a trimmed support rim")
        centre = np.asarray(CONFIG["table_center_m"][:2])
        half = np.asarray(CONFIG["table_half_size_m"][:2])
        lower = np.asarray([TABLE_X[0], TABLE_Y[0]])
        upper = np.asarray([TABLE_X[1], TABLE_Y[1]])
        np.testing.assert_allclose(lower - (centre - half), CONFIG["support_rim_m"], atol=1e-12)
        np.testing.assert_allclose((centre + half) - upper, CONFIG["support_rim_m"], atol=1e-12)

    def test_mouse_outside_is_clamped_only_in_fixture(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        j = m.joint("droid_object_freejoint")
        a = int(j.qposadr[0])
        d.qpos[a:a + 3] = [0.85, -0.65, 0.1]
        self.assertTrue(constrain_initial_placement(m, d))
        np.testing.assert_allclose(d.qpos[a:a + 2], [TABLE_X[1], TABLE_Y[0]])
        self.assertGreater(d.qpos[a + 2], TABLE_TOP)
        # The same pose remains unrestricted unless fixture code is called:
        # execution/IK has no hidden coordinate clamp or teleported grasp.
        d.qpos[a:a + 2] = [0.85, -0.65]
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.qpos[a:a + 2], [0.85, -0.65])

    def test_standby_does_not_knock_edge_cube_off(self):
        m = self.model
        specs = build_arm_specs(m)
        actuators = build_finger_actuators(m)
        repair_left_finger_position_servos(m, actuators)
        for x in TABLE_X:
            for y in TABLE_Y:
                d = mujoco.MjData(m)
                mujoco.mj_resetDataKeyframe(m, d, 0)
                initialize_observation_posture(m, d, specs)
                place_cube_for_blind_test(m, d, (x, y, 45))
                holds = {s: d.qpos[specs[s].qpos].copy() for s in specs}
                for _ in range(int(1.2 / m.opt.timestep)):
                    apply_controller(m, d, specs, actuators, holds,
                                     {"left": FINGER_OPEN, "right": FINGER_OPEN})
                    mujoco.mj_step(m, d)
                np.testing.assert_allclose(d.body("droid_object").xpos[:2], [x, y], atol=1e-3)
                self.assertGreater(d.body("droid_object").xpos[2], TABLE_TOP)

    def test_placement_ui_clamps_drag_before_confirmation(self):
        class Viewer:
            def __init__(self, model, data, key_callback):
                self.data, self.callback = data, key_callback
                self.address = int(model.joint("droid_object_freejoint").qposadr[0])
                self.cam = SimpleNamespace(lookat=np.zeros(3))
                self.frames = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self):
                return True

            def set_texts(self, *args):
                pass

            def sync(self):
                import glfw
                self.frames += 1
                self.data.qpos[self.address:self.address + 2] = [.85, -.65]
                if self.frames == 2:
                    self.callback(glfw.KEY_SPACE)

        with patch("demo_grasp_viewer.mujoco.viewer.launch_passive", side_effect=Viewer):
            x, y, _ = interactive_cube_placement(30)
        self.assertEqual(x, TABLE_X[1])
        self.assertEqual(y, TABLE_Y[0])

    def test_close_placement_cancels_instead_of_starting_grasp(self):
        viewer = SimpleNamespace(cam=SimpleNamespace(lookat=np.zeros(3)))
        viewer.is_running = lambda: False
        viewer.set_texts = lambda *args: None
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = viewer
        with patch("demo_grasp_viewer.mujoco.viewer.launch_passive", return_value=context):
            self.assertIsNone(interactive_cube_placement(30))


if __name__ == "__main__":
    unittest.main(verbosity=2)
