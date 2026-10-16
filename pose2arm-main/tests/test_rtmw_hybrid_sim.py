"""Regression tests for the RTMW3D-to-simulation grip adapter."""

import unittest

import numpy as np

from scripts.build_rtmw_hybrid_sim import guarded_pinch_ratio, slew_limit


class HybridGripTests(unittest.TestCase):
    def test_rejects_implausible_wrist_jump_without_relabeling_valid_frames(self):
        xy = np.zeros((4, 133, 2), dtype=float)
        score = np.ones((4, 133), dtype=float)
        for i in range(4):
            xy[i, 112] = [100, 100]
            xy[i, 116] = [110, 100]
            xy[i, 120] = [120, 100]
            xy[i, 117] = [100, 110]
            xy[i, 129] = [140, 110]
        xy[2, 112:133] += [700, 0]
        result = guarded_pinch_ratio(xy, score, 112, 1080, 1920)
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isfinite(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isfinite(result[3]))

    def test_limits_single_frame_closure_change(self):
        out = slew_limit(np.array([0, 0, 1, 1, 0], dtype=float), 0.1)
        self.assertLessEqual(np.max(np.abs(np.diff(out))), 0.10001)
        self.assertEqual(out[0], 0)


if __name__ == "__main__":
    unittest.main()
