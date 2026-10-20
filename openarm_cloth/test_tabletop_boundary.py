"""Full-footprint editing boundaries are not grasp-time object constraints.

The low tabletop rails are physical static geometry in both execution and
camera-observed query worlds. Placement corrections are permitted only before
the user accepts the fixture, and never modify robot joint positions.
"""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import mujoco
import numpy as np

from generic_grasp_planner import plan_observed_grasp
from generic_object_scene import (PlacementFixture, PLACEMENT_INSET,
                                  TABLE_RIM_HEIGHT, TABLE_RIM_THICKNESS,
                                  build_object_model)
from task_workspace import CONFIG, TABLE_TOP
from teleop_towel_dataset import build_arm_specs


RIM_NAMES = tuple('tabletop_rim_' + side for side in ('near', 'far', 'left', 'right'))


class TabletopBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = build_object_model('box', [.070, .040, .055])
        cls.cylinder = build_object_model('cylinder', [.045, .045, .090])
        cls.center = np.asarray(CONFIG['table_center_m'][:2], dtype=float)
        cls.half = np.asarray(CONFIG['table_half_size_m'][:2], dtype=float)

    def fixture(self, model, position=None, quaternion=(1., 0., 0., 0.)):
        data = mujoco.MjData(model)
        fixture = PlacementFixture(model, data)
        if position is None:
            position = [*self.center, TABLE_TOP + .10]
        data.qpos[fixture.address:fixture.address + 3] = position
        data.qpos[fixture.address + 3:fixture.address + 7] = quaternion
        mujoco.mj_forward(model, data)
        return fixture, data

    def assert_footprint_inside(self, model, data, expected_extent):
        geom = model.geom('droid_object_geom').id
        measured_center = data.geom_xpos[geom]
        np.testing.assert_array_less(self.center - self.half + PLACEMENT_INSET - 1e-9,
                                     measured_center[:2] - expected_extent[:2])
        np.testing.assert_array_less(measured_center[:2] + expected_extent[:2],
                                     self.center + self.half - PLACEMENT_INSET + 1e-9)
        self.assertGreaterEqual(measured_center[2] - expected_extent[2], TABLE_TOP + .001 - 1e-9)

    def test_rotated_box_full_footprint_is_kept_inside_table_not_just_center(self):
        theta = np.pi / 4
        fixture, data = self.fixture(self.box, [2., -2., TABLE_TOP - .1],
                                     [np.cos(theta / 2), 0., 0., np.sin(theta / 2)])
        before = data.qpos.copy()
        data.qvel[fixture.dof:fixture.dof + 6] = [1, 2, -3, 4, 5, 6]
        data.xfrc_applied[fixture.body] = 5.
        data.qfrc_applied[fixture.dof:fixture.dof + 6] = 5.
        extent = np.array([(0.035 + 0.020) / np.sqrt(2)] * 2 + [.0275])
        self.assertTrue(fixture.constrain(data, placement_active=True))
        self.assert_footprint_inside(self.box, data, extent)
        np.testing.assert_array_equal(data.qpos[:fixture.address], before[:fixture.address])
        np.testing.assert_array_equal(data.qpos[fixture.address + 3:], before[fixture.address + 3:])
        np.testing.assert_array_equal(data.qvel[fixture.dof:fixture.dof + 3], 0.)
        np.testing.assert_array_equal(data.qvel[fixture.dof + 3:fixture.dof + 6], [4, 5, 6])
        np.testing.assert_array_equal(data.xfrc_applied[fixture.body], 0.)
        np.testing.assert_array_equal(data.qfrc_applied[fixture.dof:fixture.dof + 6], 0.)

    def test_tilted_tall_cylinder_uses_axis_extent_and_bottom_not_upright_radius(self):
        theta = np.pi / 3
        fixture, data = self.fixture(self.cylinder, [-2., 2., .05],
                                     [np.cos(theta / 2), 0., np.sin(theta / 2), 0.])
        # Exact support function of a cylinder rotated about world Y.
        radius, halfheight = .0225, .045
        extent = np.array([radius * np.cos(theta) + halfheight * np.sin(theta),
                           radius,
                           radius * np.sin(theta) + halfheight * np.cos(theta)])
        self.assertTrue(fixture.constrain(data, placement_active=True))
        self.assert_footprint_inside(self.cylinder, data, extent)
        self.assertAlmostEqual(data.qpos[fixture.address + 2], TABLE_TOP + extent[2] + .001)

    def test_valid_placement_is_not_mutated(self):
        fixture, data = self.fixture(self.box)
        data.qvel[fixture.dof:fixture.dof + 6] = np.arange(6)
        before = (data.qpos.copy(), data.qvel.copy())
        self.assertFalse(fixture.constrain(data, placement_active=True))
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])

    def test_direct_mouse_qpos_edit_is_constrained_even_with_stale_geom_xpos(self):
        fixture, data = self.fixture(self.box)
        geom = self.box.geom('droid_object_geom').id
        old_center = data.geom_xpos[geom].copy()
        theta = np.pi / 4
        data.qpos[fixture.address:fixture.address + 3] = [2., -2., TABLE_TOP - .1]
        data.qpos[fixture.address + 3:fixture.address + 7] = [np.cos(theta / 2), 0., 0., np.sin(theta / 2)]
        # A viewer edit can arrive after the last physics update. Deliberately
        # do not mj_forward here: derived geometry is still at its old pose.
        np.testing.assert_array_equal(data.geom_xpos[geom], old_center)
        self.assertTrue(fixture.constrain(data, placement_active=True))
        extent = np.array([(0.035 + 0.020) / np.sqrt(2)] * 2 + [.0275])
        self.assert_footprint_inside(self.box, data, extent)

    def test_upright_cylinder_yaw_does_not_inflate_its_circular_footprint(self):
        theta = np.pi / 4
        radius = .0225
        x = self.center[0] + self.half[0] - PLACEMENT_INSET - radius - .0002
        fixture, data = self.fixture(self.cylinder, [x, self.center[1], TABLE_TOP + .046],
                                     [np.cos(theta / 2), 0., 0., np.sin(theta / 2)])
        original = data.qpos.copy()
        self.assertFalse(fixture.constrain(data, placement_active=True))
        np.testing.assert_array_equal(data.qpos, original)
        self.assert_footprint_inside(self.cylinder, data, np.array([radius, radius, .045]))

    def test_normal_submillimetre_table_contact_is_left_to_physics(self):
        fixture, data = self.fixture(self.box, [*self.center, TABLE_TOP + .0275 - .0001])
        data.qvel[fixture.dof:fixture.dof + 6] = np.arange(6) + .25
        data.xfrc_applied[fixture.body] = 3.
        data.qfrc_applied[fixture.dof:fixture.dof + 6] = 4.
        before = (data.qpos.copy(), data.qvel.copy(),
                  data.xfrc_applied.copy(), data.qfrc_applied.copy())
        self.assertFalse(fixture.constrain(data, placement_active=True))
        for actual, saved in zip((data.qpos, data.qvel, data.xfrc_applied, data.qfrc_applied), before):
            np.testing.assert_array_equal(actual, saved)

    def test_execution_rejects_boundary_correction_without_mutating_state(self):
        fixture, data = self.fixture(self.box, [2., 2., .1])
        data.qvel[:] = 1.
        data.xfrc_applied[fixture.body] = 3.
        before = (data.qpos.copy(), data.qvel.copy(), data.xfrc_applied.copy())
        with self.assertRaises(RuntimeError):
            fixture.constrain(data, placement_active=False)
        for actual, saved in zip((data.qpos, data.qvel, data.xfrc_applied), before):
            np.testing.assert_array_equal(actual, saved)

    def test_accepting_placement_permanently_disables_reset_and_correction(self):
        fixture, data = self.fixture(self.box)
        data.qvel[fixture.dof:fixture.dof + 6] = 2.
        data.xfrc_applied[fixture.body] = 3.
        data.qfrc_applied[fixture.dof:fixture.dof + 6] = 4.
        fixture.finish(data)
        np.testing.assert_array_equal(data.qvel[fixture.dof:fixture.dof + 6], 0.)
        np.testing.assert_array_equal(data.xfrc_applied[fixture.body], 0.)
        np.testing.assert_array_equal(data.qfrc_applied[fixture.dof:fixture.dof + 6], 0.)
        data.qpos[fixture.address] = 2.
        saved = data.qpos.copy()
        with self.assertRaises(RuntimeError):
            fixture.constrain(data, placement_active=True)
        with self.assertRaises(RuntimeError):
            fixture.reset(data, placement_active=True)
        np.testing.assert_array_equal(data.qpos, saved)

    def test_physical_rims_are_static_colliding_boxes_at_all_four_table_edges(self):
        model = self.box
        cx, cy = self.center
        hx, hy = self.half
        for name in RIM_NAMES:
            geom = model.geom(name)
            self.assertEqual(int(geom.bodyid[0]), 0)
            self.assertEqual(int(geom.type[0]), mujoco.mjtGeom.mjGEOM_BOX)
            self.assertNotEqual(int(geom.contype[0]), 0)
            self.assertNotEqual(int(geom.conaffinity[0]), 0)
            self.assertAlmostEqual(geom.pos[2] - geom.size[2], TABLE_TOP)
            self.assertAlmostEqual(2 * geom.size[2], TABLE_RIM_HEIGHT)
            self.assertLess(TABLE_RIM_HEIGHT, .012)
        near, far = model.geom(RIM_NAMES[0]), model.geom(RIM_NAMES[1])
        left, right = model.geom(RIM_NAMES[2]), model.geom(RIM_NAMES[3])
        self.assertAlmostEqual(near.pos[0] + near.size[0], cx - hx)
        self.assertAlmostEqual(far.pos[0] - far.size[0], cx + hx)
        self.assertAlmostEqual(left.pos[1] - left.size[1], cy + hy)
        self.assertAlmostEqual(right.pos[1] + right.size[1], cy - hy)
        for geom, axis in ((near, 0), (far, 0), (left, 1), (right, 1)):
            self.assertAlmostEqual(2 * geom.size[axis], TABLE_RIM_THICKNESS)

    def test_rail_contact_is_real_not_a_display_only_marker(self):
        edge = self.center[0] + self.half[0]
        fixture, data = self.fixture(self.box, [edge - .025, 0., TABLE_TOP + .0276])
        target = self.box.geom('droid_object_geom').id
        rail = self.box.geom('tabletop_rim_far').id
        contacts = {frozenset((int(c.geom1), int(c.geom2))) for c in data.contact}
        self.assertIn(frozenset((target, rail)), contacts)

    def test_empty_table_chest_perception_does_not_detect_the_low_rims(self):
        from calibration_simulation import render_rgb_depth
        from demo_grasp_viewer import initialize_observation_posture, load_chest_calibration
        from tabletop_perception import detect_tabletop_objects

        model = self.box
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        initialize_observation_posture(model, data, build_arm_specs(model))
        address = int(model.joint('droid_object_freejoint').qposadr[0])
        # Test-only empty scene setup, not a controller observation/answer.
        data.qpos[address:address + 3] = [9., 9., .10]
        mujoco.mj_forward(model, data)
        transform, matrix, _ = load_chest_calibration()
        renderer = mujoco.Renderer(model, 480, 640)
        try:
            rgb, depth = render_rgb_depth(renderer, data, 'chest_d435i_preview')
        finally:
            renderer.close()
        bounds = tuple((float(c - h + .002), float(c + h - .002))
                       for c, h in zip(self.center, self.half))
        self.assertEqual(detect_tabletop_objects(rgb, depth, matrix, transform,
                                                TABLE_TOP, bounds), [])

    def test_private_visual_query_retains_same_physical_table_boundaries(self):
        model = self.box
        data = mujoco.MjData(model)
        spec = build_arm_specs(model)['right']
        seed = data.qpos[spec.qpos].copy()
        observation = SimpleNamespace(shape='box', size_xyz=[.06, .04, .05],
                                      center_world=np.array([.43, -.10, .425]), yaw_world_rad=0.)
        with patch('generic_grasp_planner.plan_face_grasp',
                   return_value=dict(direction=np.array([0., 1., 0.]))):
            plan = plan_observed_grasp(model, data, 'right', observation, seed, seed)
        query = plan['query_model']
        self.assertIsNot(query, model)
        for name in RIM_NAMES:
            actual, measured = model.geom(name), query.geom(name)
            np.testing.assert_array_equal(actual.pos, measured.pos)
            np.testing.assert_array_equal(actual.size, measured.size)
            np.testing.assert_array_equal(actual.contype, measured.contype)
            np.testing.assert_array_equal(actual.conaffinity, measured.conaffinity)


if __name__ == '__main__':
    unittest.main()
