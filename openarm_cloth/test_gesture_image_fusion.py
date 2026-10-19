"""Synthetic geometry only; no personal webcam landmarks in these fixtures."""
import unittest
import numpy as np
from human_gripper_gestures import fist_features, image_extension_features, FistGripperMapper, gesture_diagnostics
from test_human_gripper_gestures import hand


def image_hand(folded=0,size=(1280,720)):
    # Pixel geometry is invariant to normalized-image aspect ratio.
    h=hand(folded)*1000+[400,250,0]
    return h/np.array([*size,1000])


class ImageFusionTests(unittest.TestCase):
    def test_clear_image_can_release_incorrect_world_fist(self):
        f=fist_features(hand(4),image_hand(),(1280,720))
        self.assertEqual(f['source'],'image_extension')
        self.assertEqual(f['gesture'],'OPEN PALM')
        self.assertEqual((f['world_scores']>=.6).sum(),4)
        self.assertEqual(f['image_extended'].sum(),4)

    def test_opening_still_requires_two_new_frames(self):
        m=FistGripperMapper()
        for seq in (1,2): m.update(fist_features(hand(4)),seq,.05*seq)
        f=fist_features(hand(4),image_hand(),(1280,720))
        self.assertEqual(m.update(f,3,.15),1)
        self.assertEqual(m.update(f,3,.16),1)
        self.assertEqual(m.update(f,4,.2),0)

    def test_image_fist_and_pinch_never_invent_opening(self):
        for folded in (2,3,4):
            f=fist_features(hand(4),image_hand(folded),(1280,720))
            self.assertEqual(f['gesture'],'FIST')
        h=image_hand(2);h[4]=h[8]
        self.assertEqual(fist_features(hand(2),h,(1280,720))['gesture'],'PARTIAL: HOLD')

    def test_three_visible_fingers_cannot_override_strong_world_fist(self):
        h=image_hand(1)
        self.assertEqual(image_extension_features(h,(1280,720))['extended'].sum(),3)
        self.assertEqual(fist_features(hand(4),h,(1280,720))['gesture'],'FIST')
        self.assertEqual(fist_features(hand(2),h,(1280,720))['gesture'],'OPEN PALM')

    def test_missing_world_is_not_replaced_by_image_only_gesture(self):
        for world in (None,np.zeros((21,3)),np.full((21,3),np.nan)):
            self.assertIsNone(fist_features(world,image_hand(),(1280,720)))

    def test_bad_tiny_clipped_or_edge_on_image_falls_back(self):
        normal=image_hand()
        tiny=(normal-normal[0])*.1+normal[0]
        clipped=normal.copy();clipped[20,0]=1.01
        boundary=normal.copy();boundary[20,1]=.975
        edge=normal.copy();edge[:,0]=edge[9,0]+.05*(edge[:,0]-edge[9,0])
        for h,size in ((None,(1280,720)),(np.zeros((3,2)),(1280,720)),
                       (normal,None),(normal,(0,720)),(np.full((21,3),np.nan),(1280,720)),
                       (tiny,(1280,720)),(clipped,(1280,720)),(boundary,(1280,720)),(edge,(1280,720))):
            with self.subTest(size=size):
                f=fist_features(hand(4),h,size)
                self.assertEqual(f['gesture'],'FIST')
                self.assertEqual(f['source'],'world')

    def test_normalized_aspect_ratio_is_not_treated_as_square(self):
        a=fist_features(hand(4),image_hand(size=(1280,720)),(1280,720))
        b=fist_features(hand(4),image_hand(size=(720,1280)),(720,1280))
        np.testing.assert_array_equal(a['image_extended'],b['image_extended'])
        self.assertEqual(a['gesture'],b['gesture'])

    def test_projected_shortening_does_not_close_open_world_hand(self):
        f=fist_features(hand(0),image_hand(4),(1280,720))
        self.assertEqual(f['gesture'],'OPEN PALM')
        self.assertEqual(f['source'],'world')

    def test_no_image_preserves_world_scores(self):
        for folded in range(5):
            f=fist_features(hand(folded))
            np.testing.assert_array_equal(f['scores'],f['world_scores'])

    def test_diagnostics_record_both_evidence_sources_without_object_arrays(self):
        for features in (None,fist_features(hand(4),image_hand(),(1280,720))):
            record=gesture_diagnostics(features)
            self.assertEqual(record['world_curl_scores'].shape,(4,))
            self.assertEqual(record['image_extended'].shape,(4,))
            self.assertTrue(all(v.dtype.kind!='O' for v in record.values()))

    def test_visible_extension_survives_image_roll_scale_and_translation(self):
        h=image_hand();pixels=h[:,:2]*[1280,720];center=pixels[0].copy()
        for degrees in (-70,0,65):
            r=np.radians(degrees);rotation=np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]])
            moved=(pixels-center)@rotation.T*.85+center+[200,40]
            self.assertEqual(fist_features(hand(4),moved/[1280,720],(1280,720))['gesture'],'OPEN PALM')

    def test_image_depth_does_not_enter_projected_extension_evidence(self):
        h=image_hand();h[:,2]=1e10
        self.assertEqual(fist_features(hand(4),h,(1280,720))['gesture'],'OPEN PALM')

    def test_missing_frame_resets_opening_confirmation(self):
        m=FistGripperMapper()
        for seq in (1,2): m.update(fist_features(hand(4)),seq,.05*seq)
        f=fist_features(hand(4),image_hand(),(1280,720))
        self.assertEqual(m.update(f,3,.15),1)
        self.assertEqual(m.update(None,4,.2),1)
        self.assertEqual(m.update(f,5,.25),1)
        self.assertEqual(m.update(f,6,.3),0)

    def test_opening_one_side_never_changes_the_other_mapper(self):
        mappers=[FistGripperMapper(),FistGripperMapper()]
        for mapper in mappers:
            for seq in (1,2): mapper.update(fist_features(hand(4)),seq,.05*seq)
        for seq in (3,4):
            mappers[0].update(fist_features(hand(4),image_hand(),(1280,720)),seq,.05*seq)
            mappers[1].update(fist_features(hand(4),image_hand(4),(1280,720)),seq,.05*seq)
        self.assertEqual([m.state for m in mappers],[0,1])

    def test_live_row_and_npz_preserve_opening_evidence(self):
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from live_bimanual_runtime import BimanualControlLoop
        from live_motion_scene import build_motion_scene
        from live_test_dataset import TestDatasetRecorder
        model,data=build_motion_scene()
        loop=BimanualControlLoop(model,data,None,SimpleNamespace(save_video=False),Path('.'))
        f=fist_features(hand(4),image_hand(),(1280,720))
        obs=dict(seq=1,read_end=1,done=2,pose_raw=None,pose_filtered=None,pose_norm=None,
                 arms={side:dict(hand_raw=hand(4),hand_filtered=hand(4),hand_norm=image_hand(),fist=f,pose=None)
                       for side in ('left','right')})
        try:
            rows=[loop._row(o,True,3,4,np.zeros((2,3)),0.,0.) for o in (None,obs)]
            with tempfile.TemporaryDirectory(prefix='openarm_fusion_record_') as tmp:
                writer=TestDatasetRecorder(Path(tmp)/'dataset',{},save_video=False,chunk_rows=2)
                for row in rows: writer.submit(row)
                writer.close();self.assertIsNone(writer.error)
                with np.load(Path(tmp)/'dataset'/'chunk_00000.npz',allow_pickle=False) as saved:
                    self.assertEqual(saved['left_gesture_source'].tolist(),['missing','image_extension'])
                    self.assertEqual(saved['right_gesture'].tolist(),['NO HAND','OPEN PALM'])
                    self.assertEqual(saved['left_image_extended'].shape,(2,4))
                    self.assertTrue((saved['left_world_curl_scores'][1]>=.6).all())
        finally: loop.control.close()


if __name__=='__main__': unittest.main()
