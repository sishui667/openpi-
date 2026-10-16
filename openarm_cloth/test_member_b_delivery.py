"""B-side acceptance tests; camera hardware is tested separately, not mocked as proof."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from member_b_scene import build_scene
from member_b_actuators import ActuatorController
from pose2arm_driver import (Pose2ArmDriver, GripperHysteresis, load_pose2arm,
                            POSE2ARM_VEL_LIMITS)
from teleop_towel_dataset import apply_controller, build_arm_specs, build_finger_actuators


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.ready = build_scene()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "source.npz"
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.ready.qpos
        mujoco.mj_forward(self.model, self.data)
        self.specs = build_arm_specs(self.model)

    def fixture(self, **extra):
        q = np.zeros((4, 14))
        q[:, 0] = -2.5  # unambiguously LEFT first, outside right joint1 range
        blob = dict(joint_angles=q, target_obs=np.zeros((4, 18)),
                    frame_indices=np.arange(4), fps=30.0)
        blob.update(extra)
        np.savez(self.path, **blob)
        return self.path

    def test_hysteresis_exact_boundaries_and_missing(self):
        h = GripperHysteresis()
        self.assertEqual(h.update(0.025), 0.044)
        self.assertEqual(h.update(0.024), 0)
        self.assertEqual(h.update(0.045), 0)
        self.assertEqual(h.update(None), 0)
        self.assertEqual(h.update(float("nan")), 0)
        self.assertEqual(h.update(0.046), 0.044)

    def test_bad_numbers_and_shapes_rejected(self):
        for extra in (dict(joint_angles=np.full((4,14), np.nan)),
                      dict(target_obs=np.zeros((3,18))), dict(fps=0),
                      dict(frame_indices=np.arange(4, dtype=float)),
                      dict(hand_distance_m=np.full(4, -0.1))):
            self.fixture(**extra)
            with self.assertRaises(ValueError):
                load_pose2arm(self.path)

    def test_explicit_order_resolves_ambiguous_source(self):
        self.fixture(joint_angles=np.zeros((4,14)), joint_order="left-first")
        driver = Pose2ArmDriver(self.model, self.data, self.path)
        self.assertEqual(driver.order, "left-first")
        self.fixture(joint_angles=np.zeros((4,14)))
        with self.assertRaises(ValueError):
            Pose2ArmDriver(self.model, self.data, self.path)

    def test_order_contradiction_rejected(self):
        self.fixture(joint_order="right-first")
        with self.assertRaises(ValueError):
            Pose2ArmDriver(self.model, self.data, self.path)

    def test_first_frame_is_also_slew_limited(self):
        driver = Pose2ArmDriver(self.model, self.data, self.fixture())
        before = self.data.qpos[self.specs["right"].qpos].copy()
        targets, _ = driver.prepare_frame(0)
        self.assertTrue(np.all(np.abs(targets["right"] - before) <= POSE2ARM_VEL_LIMITS / 20 + 1e-12))
        self.assertEqual(driver.report.order, "left-first")

    def test_inactive_hold_is_fixed_not_current_drift(self):
        driver = Pose2ArmDriver(self.model, self.data, self.fixture())
        original = self.data.qpos[self.specs["left"].qpos].copy()
        self.data.qpos[self.specs["left"].qpos] += .03
        targets, fingers = driver.prepare_frame(0)
        np.testing.assert_array_equal(targets["left"], original)
        self.assertEqual(fingers["left"], .044)

    def test_human_distance_optional_input_takes_precedence(self):
        driver = Pose2ArmDriver(self.model, self.data,
                               self.fixture(hand_distance_m=np.array([.02,.035,.05,np.nan])))
        self.assertEqual(driver.prepare_frame(0)[1]["right"], 0)
        self.assertEqual(driver.prepare_frame(1)[1]["right"], 0)
        self.assertEqual(driver.prepare_frame(2)[1]["right"], .044)
        self.assertEqual(driver.prepare_frame(3)[1]["right"], .044)

    def test_full_physics_interval_and_frequency(self):
        driver = Pose2ArmDriver(self.model, self.data, self.fixture(), control_hz=10)
        start = self.data.time
        driver.step_frame(0)
        self.assertEqual(driver.steps_per_frame, 200)
        self.assertAlmostEqual(self.data.time - start, .1, places=10)
        with self.assertRaises(ValueError):
            Pose2ArmDriver(self.model, self.data, self.path, control_hz=30)

    def test_cached_pd_matches_original_for_100_physics_steps(self):
        other = mujoco.MjData(self.model)
        other.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, other)
        targets = {s: self.data.qpos[spec.qpos].copy() for s,spec in self.specs.items()}
        targets["right"] += .02
        fingers = {"left": .044, "right": .02}
        cached = ActuatorController(self.model, other)
        action = cached.set_targets(targets, fingers)
        for _ in range(100):
            original = apply_controller(self.model, self.data, self.specs,
                                        build_finger_actuators(self.model), targets, fingers)
            cached.apply()
            np.testing.assert_allclose(other.ctrl, self.data.ctrl, atol=1e-10)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_step(self.model, other)
        np.testing.assert_allclose(other.qpos, self.data.qpos, atol=1e-10)
        np.testing.assert_allclose(action, original)

    def test_monocular_tracking_checks_current_pixels(self):
        import workspace_camera as wc
        address = int(self.model.joint("droid_object_freejoint").qposadr[0])
        self.data.qpos[address:address+3] = [.45, 0, wc.TABLE_TOP + wc.CUBE_HALF]
        mujoco.mj_forward(self.model, self.data)
        r = mujoco.Renderer(self.model, 240, 320)
        try:
            camera = wc.Pinhole(self.model, self.data, wc.CAM_NAME, 320, 240)
            r.update_scene(self.data, camera=wc.CAM_NAME)
            mask = wc.red_mask(r.render())
            best, _ = wc.fit(camera, mask, topk=20)
            with patch.object(wc, "fit", side_effect=AssertionError("stationary refit")):
                stationary, score, _ = wc.fit_tracking(camera, mask, best)
            self.assertGreaterEqual(score, .97)
            np.testing.assert_allclose(stationary, best)
            # Hidden truth changed with pixels held fixed must not matter.
            self.data.qpos[address:address+2] = [.9, .9]
            mujoco.mj_forward(self.model, self.data)
            unchanged, _, _ = wc.fit_tracking(camera, mask, best)
            np.testing.assert_allclose(unchanged, best)
            # A genuinely changed image must update the measured position.
            self.data.qpos[address:address+2] = [.49, 0]
            mujoco.mj_forward(self.model, self.data)
            r.update_scene(self.data, camera=wc.CAM_NAME)
            moved, _, _ = wc.fit_tracking(camera, wc.red_mask(r.render()), best)
            self.assertGreater(moved[0]-best[0], .025)
            self.assertLess(abs(moved[0]-.49), .006)
        finally:
            r.close()

    def test_no_active_weld_and_target_zone_defined(self):
        self.assertGreaterEqual(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                                "member_b_target_zone"), 0)
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.assertFalse(self.data.eq_active[i])

    def test_freeze_detects_tampered_hash(self):
        from freeze_member_b_scene import inspect_scene, verify
        payload = inspect_scene()
        payload["files_sha256"]["member_b_scene.py"] = "0" * 64
        manifest = Path(self.temp.name) / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen file drift"):
            verify(manifest)

    def test_delivery_resolution_monocular_accuracy(self):
        import workspace_camera as wc
        address = int(self.model.joint("droid_object_freejoint").qposadr[0])
        renderer = mujoco.Renderer(self.model, 192, 256)
        try:
            for x, y, yaw in ((.36,-.12,0), (.43,0,30), (.50,.12,45),
                              (.36,.12,45), (.50,-.12,0)):
                # Truth is available ONLY to fixture placement and the assertion.
                self.data.qpos[address:address+3] = [x,y,wc.TABLE_TOP+wc.CUBE_HALF]
                angle = np.deg2rad(yaw) / 2
                self.data.qpos[address+3:address+7] = [np.cos(angle),0,0,np.sin(angle)]
                mujoco.mj_forward(self.model, self.data)
                renderer.update_scene(self.data, camera=wc.CAM_NAME)
                camera = wc.Pinhole(self.model, self.data, wc.CAM_NAME, 256, 192)
                best, score = wc.fit(camera, wc.red_mask(renderer.render()), topk=20)
                error = float(np.linalg.norm(np.asarray(best[:2])-[x,y]))
                print(f"RGB256 fixture=({x},{y},{yaw}) error_m={error:.6f} IoU={score:.4f}")
                self.assertGreaterEqual(score, .6)
                # This low-resolution benchmark is for interface/latency,
                # NOT the chest/wrist RGB-D precision-grasp controller.
                # Measured worst case is ~9 mm; do not claim a 6 mm guarantee.
                self.assertLess(error, .010)
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
