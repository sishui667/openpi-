import unittest
import numpy as np
from human_gripper_gestures import fist_features, FistGripperMapper


def hand(folded=0):
    h=np.zeros((21,3))
    for i,mcp in enumerate((5,9,13,17)):
        h[mcp]=[(i-1.5)*.018,.07,0]
        h[mcp+1]=h[mcp]+[0,.035,0]
        h[mcp+2]=h[mcp]+([0,.01,.01] if i<folded else [0,.06,0])
        h[mcp+3]=h[mcp]+([0,-.015,.015] if i<folded else [0,.09,0])
    return h


class GestureTests(unittest.TestCase):
    def test_open_hand_and_fist(self):
        self.assertEqual(fist_features(hand(0))['extended'],4)
        self.assertEqual(fist_features(hand(4))['folded'],4)

    def test_translation_scale_rotation_invariance(self):
        from scipy.spatial.transform import Rotation
        h=hand(4);r=Rotation.from_euler('xyz',[30,20,-60],degrees=True).as_matrix()
        a=fist_features(h);b=fist_features(h@r.T*3+[2,5,7])
        np.testing.assert_allclose(a['scores'],b['scores'],atol=1e-10)

    def test_thumb_index_pinch_is_not_fist(self):
        h=hand(0);h[4]=h[8]
        self.assertEqual(fist_features(h)['gesture'],'OPEN PALM')

    def test_two_fingers_do_not_close(self):
        self.assertEqual(fist_features(hand(2))['gesture'],'PARTIAL: HOLD')

    def test_new_frames_confirm_both_open_and_close(self):
        m=FistGripperMapper()
        self.assertEqual(m.update(fist_features(hand(4)),1,0),0)
        self.assertEqual(m.update(fist_features(hand(4)),1,.05),0)
        self.assertEqual(m.update(fist_features(hand(4)),2,.1),1)
        self.assertEqual(m.update(fist_features(hand(0)),3,.15),1)
        self.assertEqual(m.update(fist_features(hand(0)),4,.2),0)

    def test_missing_and_degenerate_are_not_fake_gestures(self):
        self.assertIsNone(fist_features(np.zeros((21,3))))
        self.assertIsNone(fist_features(None))
        m=FistGripperMapper()
        for seq in (1,2): m.update(fist_features(hand(4)),seq,seq*.05)
        self.assertEqual(m.update(None,3,.3),1)
        self.assertEqual(m.update(None,4,.8),0)


if __name__=='__main__': unittest.main()
