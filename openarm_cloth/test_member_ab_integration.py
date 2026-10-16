"""Actual A/B contract regressions; no detector/model download required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pose2arm-main"))
from member_b_export import adapt
from gripper_map import GripperMapper
from member_b_scene import build_scene
from pose2arm_driver import Pose2ArmDriver, load_pose2arm


class ABContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.ready = build_scene()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.npz"
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.ready.qpos
        mujoco.mj_forward(self.model,self.data)

    def blob(self, **extras):
        q = np.zeros((6,14)); q[:,0] = -2.5
        blob = dict(joint_angles=q, target_obs=np.zeros((6,18)),
                    frame_indices=np.arange(6), fps=np.array(30.),
                    joint_order=np.array("left-first"))
        blob.update(extras)
        return blob

    def driver(self, **extras):
        np.savez(self.path, **self.blob(**extras))
        return Pose2ArmDriver(self.model,self.data,self.path,gripper_mode="human-only")

    def test_right_gripper_is_column_zero_not_joint_order(self):
        d = self.driver(gripper_cmd=np.tile([1,0],(6,1)))
        _,f = d.prepare_frame(0)
        self.assertEqual(f["right"],0)
        self.assertEqual(f["left"],.044)

    def test_declared_left_first_gripper(self):
        d = self.driver(gripper_cmd=np.tile([0,1],(6,1)),gripper_order="left-first")
        self.assertEqual(d.prepare_frame(0)[1]["right"],0)

    def test_continuous_priority_and_slide_units(self):
        d = self.driver(gripper_cmd=np.ones((6,2)),
                        gripper_position=np.tile([.25,1],(6,1)))
        self.assertAlmostEqual(d.prepare_frame(0)[1]["right"],.033)
        self.assertEqual(d.report.gripper_source,"gripper_position")

    def test_missing_human_signal_never_uses_object_position(self):
        np.savez(self.path, **self.blob())
        with self.assertRaisesRegex(ValueError,"human-only"):
            Pose2ArmDriver(self.model,self.data,self.path,gripper_mode="human-only")
        d = Pose2ArmDriver(self.model,self.data,self.path,
                          object_centre_fn=lambda: np.zeros(3))
        self.assertEqual(d.prepare_frame(0, perceived_center=np.zeros(3))[1]["right"],.044)
        self.assertEqual(d.report.gripper_source,"unavailable")

    def test_bad_gripper_contract_rejected(self):
        for extras in (dict(gripper_position=np.full((6,2),np.nan)),
                       dict(gripper_position=np.full((6,2),-.7854)),
                       dict(gripper_cmd=np.full((6,2),.25)),
                       dict(gripper_cmd=np.ones((5,2))),
                       dict(gripper_order="unknown"),
                       dict(gripper_semantics="radians")):
            np.savez(self.path, **self.blob(**extras))
            with self.assertRaises(ValueError): load_pose2arm(self.path)

    def test_float32_missing_hand_counts_toward_release(self):
        mapper = GripperMapper(hold_frames=0,lost_frames=2)
        self.assertEqual(mapper.update(.02),1)
        self.assertEqual(mapper.update(np.float32(np.nan)),1)
        self.assertEqual(mapper.update(np.float32(np.nan)),1)
        self.assertEqual(mapper.update(np.float32(np.nan)),0)

    def test_exact_hysteresis_boundaries_and_invalid_signal(self):
        mapper = GripperMapper(hold_frames=0,lost_frames=0)
        self.assertEqual(mapper.update(.025),0)
        self.assertEqual(mapper.update(.02),1)
        self.assertEqual(mapper.update(.045),1)
        self.assertEqual(mapper.update(float("inf")),0)
        with self.assertRaises(ValueError): GripperMapper(hold_frames=-1)

    def test_resample_preserves_video_duration_and_separates_mirror(self):
        source = self.blob(frame_indices=np.array([0,1,3,0,1,3]),
                           gripper_cmd=np.tile([1,0],(6,1)))
        source["joint_angles"][3:,7:] = .5
        orig,mirror = adapt(source,20,"orig"),adapt(source,20,"mirror")
        self.assertTrue(np.all(orig["joint_angles"][:,7:]==0))
        self.assertTrue(np.all(mirror["joint_angles"][:,7:]==.5))
        self.assertLess(abs(len(orig["timestamps_s"])/20-4/30),.05)
        np.testing.assert_allclose(np.diff(orig["timestamps_s"]),.05)
        self.assertTrue(np.isin(orig["gripper_cmd"],[0,1]).all())

    def test_export_missing_signal_and_timing_mismatch_rejected(self):
        with self.assertRaises(ValueError): adapt(self.blob())
        np.savez(self.path, **self.blob(gripper_cmd=np.ones((6,2)),
                                       timestamps_s=np.arange(6)/30))
        with self.assertRaisesRegex(ValueError,"control_hz"):
            Pose2ArmDriver(self.model,self.data,self.path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
