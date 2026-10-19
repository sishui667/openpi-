"""Camera-free deterministic regressions for causal hand ROI and association."""
from types import SimpleNamespace
import unittest

import numpy as np

import live_bimanual_stream as stream


def body(wrist=(.7,.4),side='left'):
    pose=np.zeros((33,4),dtype=float)
    pose[:,3]=1.
    pose[11,:2]=(.65,.4);pose[12,:2]=(.35,.4)
    pose[13,:2]=(.65,.55);pose[14,:2]=(.35,.55)
    pose[15,:2]=(.7,.4);pose[16,:2]=(.3,.4)
    pose[15 if side=='left' else 16,:2]=wrist
    return pose


class AdaptiveROI(unittest.TestCase):
    def assert_square_inside(self,bounds,width=1280,height=720):
        self.assertIsNotNone(bounds)
        lo,hi=bounds
        self.assertEqual(int(hi[0]-lo[0]),int(hi[1]-lo[1]))
        self.assertTrue(np.all(lo>=0));self.assertTrue(np.all(hi<=[width,height]))
        self.assertGreaterEqual(int(min(hi-lo)),100)

    def test_foreshortened_forearm_uses_image_context(self):
        p=body();p[13,:2]=p[15,:2]  # zero image-plane forearm length
        bounds=stream.roi_bounds(p,'left',1280,720)
        self.assert_square_inside(bounds)
        self.assertGreaterEqual(int(bounds[1][0]-bounds[0][0]),244)

    def test_shoulder_context_scales_crop(self):
        p=body();p[13,:2]=p[15,:2]
        p[11,:2]=(.2,.4);p[12,:2]=(.8,.4)
        lo,hi=stream.roi_bounds(p,'left',1280,720)
        self.assertGreaterEqual(int(hi[0]-lo[0]),537)

    def test_edge_shift_preserves_square_size(self):
        p=body();p[13,:2]=p[15,:2]
        lo,hi=stream.roi_bounds(p,'left',1280,720)
        size=int(hi[0]-lo[0])
        for wrist in [(.01,.01),(.99,.01),(.01,.99),(.99,.99)]:
            with self.subTest(wrist=wrist):
                edge=body(wrist);edge[13,:2]=edge[15,:2]
                b=stream.roi_bounds(edge,'left',1280,720)
                self.assert_square_inside(b)
                self.assertEqual(int(b[1][0]-b[0][0]),size)

    def test_recent_spread_hand_box_is_included(self):
        p=body((.7,.5))
        hand=np.zeros((21,3));hand[:,0]=np.linspace(.62,.78,21);hand[:,1]=np.linspace(.32,.55,21)
        hand[0,:2]=p[15,:2]  # no body/hand wrist translation
        lo,hi=stream.roi_bounds(p,'left',1280,720,hand)
        self.assert_square_inside((lo,hi))
        points=hand[:,:2]*[1280,720]
        self.assertTrue(np.all(points>=lo));self.assertTrue(np.all(points<=hi))

    def test_recent_hand_is_translated_to_new_wrist(self):
        p=body((.72,.5));hand=np.zeros((21,3))
        hand[:,0]=np.linspace(.6,.7,21);hand[:,1]=np.linspace(.4,.55,21)
        hand[0,:2]=(.65,.5)
        lo,hi=stream.roi_bounds(p,'left',1280,720,hand)
        translated=hand[:,:2]*[1280,720]+(p[15,:2]-hand[0,:2])*[1280,720]
        self.assertTrue(np.all(translated>=lo));self.assertTrue(np.all(translated<=hi))

    def test_invalid_hand_box_falls_back_to_body_crop(self):
        p=body();expected=stream.roi_bounds(p,'left',1280,720)
        for invalid in [np.zeros((20,3)),np.full((21,3),np.nan),np.full((21,3),np.inf)]:
            with self.subTest(shape=invalid.shape):
                result=stream.roi_bounds(p,'left',1280,720,invalid)
                np.testing.assert_array_equal(result[0],expected[0]);np.testing.assert_array_equal(result[1],expected[1])

    def test_rejects_bad_wrist_visibility_coordinates(self):
        for xy in [(-.01,.5),(1.01,.5),(.5,-.01),(.5,1.01)]:
            with self.subTest(xy=xy): self.assertIsNone(stream.roi_bounds(body(xy),'left',1280,720))
        for bad in [np.nan,np.inf,-np.inf]:
            p=body();p[15,0]=bad
            self.assertIsNone(stream.roi_bounds(p,'left',1280,720))
        p=body();p[15,3]=.599
        self.assertIsNone(stream.roi_bounds(p,'left',1280,720))
        self.assertIsNone(stream.roi_bounds(None,'left',1280,720))

    def test_nonfinite_elbow_or_shoulders_cannot_create_undefined_crop(self):
        for idx in [11,12,13]:
            p=body();p[idx,0]=np.nan
            with self.subTest(idx=idx): self.assertIsNone(stream.roi_bounds(p,'left',1280,720))

    def test_boundary_exactly_on_frame_can_have_clamped_crop(self):
        for xy in [(0.,0.),(1.,1.)]:
            b=stream.roi_bounds(body(xy),'left',1280,720)
            self.assert_square_inside(b)

    def test_low_resolution_is_handled_without_outside_bounds(self):
        for wh in [(640,360),(320,240),(720,1280)]:
            b=stream.roi_bounds(body(),'left',*wh)
            self.assert_square_inside(b,*wh)
        self.assertIsNone(stream.roi_bounds(body(),'left',60,50))


class AssociationInvariants(unittest.TestCase):
    @staticmethod
    def candidate(wrist,score=.9,label='Left',roi=None):
        return dict(wrist=np.array(wrist),score=score,label=label,roi=roi)

    def test_same_detection_in_two_rois_can_drive_only_one_side(self):
        c=[self.candidate((.7,.4),.9,roi='left'),self.candidate((.701,.4),.8,roi='right')]
        assigned=stream.assign_candidates(c,body(),np.full((2,2),np.nan))
        self.assertEqual(list(assigned),['left']);self.assertIs(assigned['left'],c[0])

    def test_crop_origin_does_not_decide_anatomical_side(self):
        c=[self.candidate((.7,.4),roi='right'),self.candidate((.3,.4),label='Right',roi='left')]
        assigned=stream.assign_candidates(c,body(),np.full((2,2),np.nan))
        self.assertEqual(set(assigned),{'left','right'})
        self.assertIs(assigned['left'],c[0]);self.assertIs(assigned['right'],c[1])

    def test_far_or_missing_candidates_do_not_force_either_arm(self):
        prior=np.full((2,2),np.nan)
        self.assertEqual(stream.assign_candidates([],body(),prior),{})
        self.assertEqual(stream.assign_candidates([self.candidate((.99,.99))],body(),prior),{})

    def test_full_inference_candidates_are_not_marked_as_roi(self):
        landmark=SimpleNamespace(x=.3,y=.4,z=.01)
        world=SimpleNamespace(x=.01,y=.02,z=.03)
        result=SimpleNamespace(hand_landmarks=[[landmark]*21],hand_world_landmarks=[[world]*21],
                               handedness=[[SimpleNamespace(category_name='Right',score=.97)]])
        detector=SimpleNamespace(detect_for_video=lambda image,timestamp:result)
        candidates=stream.full_inference(None,detector,object(),123)
        self.assertEqual(len(candidates),1);self.assertIsNone(candidates[0]['roi'])
        self.assertEqual(candidates[0]['norm'].shape,(21,3));self.assertEqual(candidates[0]['world'].shape,(21,3))
        np.testing.assert_allclose(candidates[0]['wrist'],[.3,.4])


if __name__=='__main__': unittest.main(verbosity=2)
