"""Synthetic RGB-D tests; no MuJoCo state, object dimensions, or colour cues.

The rasterized top faces have analytic metric extents under a downward-facing
calibrated pinhole camera. This tests geometric perception, not real-camera
recognition accuracy or physical grasp success.
"""
import copy
import unittest

import cv2
import numpy as np

from tabletop_perception import detect_tabletop_objects, select_tabletop_object, TargetMovedError


HEIGHT,WIDTH=240,320
TABLE_Z=.40
K=np.array([[600.,0.,160.],[0.,600.,120.],[0.,0.,1.]])
T=np.eye(4)
T[:3,:3]=np.diag([1.,-1.,-1.])
T[:3,3]=[0.,0.,1.]


def scene(objects=(),colour=(0,0,0)):
    # Constant RGB deliberately removes appearance boundaries and colour cues.
    rgb=np.empty((HEIGHT,WIDTH,3),dtype=np.uint8);rgb[:]=colour
    depth=np.full((HEIGHT,WIDTH),.60)
    for item in objects:
        center=item.get('center',(160,120));size=item.get('size',(60,40))
        mask=np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
        if item.get('shape','box')=='cylinder':
            cv2.circle(mask,tuple(map(int,center)),int(size[0]/2),255,-1)
        else:
            vertices=cv2.boxPoints((tuple(map(float,center)),tuple(map(float,size)),float(item.get('angle',0))))
            cv2.fillConvexPoly(mask,np.rint(vertices).astype(np.int32),255)
        depth[mask>0]=.60-float(item.get('height',.06))
    return rgb,depth


def observe(rgb,depth,**kwargs):
    return detect_tabletop_objects(rgb,depth,K,T,TABLE_Z,**kwargs)


def yaw_error(a,b):
    return abs((a-b+np.pi/2)%np.pi-np.pi/2)


class TabletopPerceptionTests(unittest.TestCase):
    def test_cube_display_label_comes_from_camera_dimensions_only(self):
        rgb,depth=scene([dict(size=(60,60),height=.054)])
        cube=observe(rgb,depth)[0]
        self.assertEqual(cube.shape,'box')
        self.assertEqual(cube.shape_label,'cube')
        self.assertEqual(cube.as_dict()['shape_label'],'cube')
        rgb,depth=scene([dict(size=(60,40),height=.054)])
        self.assertEqual(observe(rgb,depth)[0].shape_label,'box')

    def test_unsupported_flat_outlines_are_not_forced_into_boxes_or_cylinders(self):
        masks=[]
        ellipse=np.zeros((HEIGHT,WIDTH),np.uint8)
        cv2.ellipse(ellipse,(160,120),(38,21),22,0,360,255,-1);masks.append(ellipse)
        trapezoid=np.zeros_like(ellipse)
        cv2.fillConvexPoly(trapezoid,np.array([[125,95],[195,95],[182,145],[137,145]]),255)
        masks.append(trapezoid)
        hexagon=np.zeros_like(ellipse)
        angles=np.arange(6)*np.pi/3
        vertices=np.rint(np.c_[160+38*np.cos(angles),120+38*np.sin(angles)]).astype(np.int32)
        cv2.fillConvexPoly(hexagon,vertices,255);masks.append(hexagon)
        triangle=np.zeros_like(ellipse)
        cv2.fillConvexPoly(triangle,np.array([[160,85],[120,150],[200,150]]),255);masks.append(triangle)
        for index,mask in enumerate(masks):
            with self.subTest(outline=index):
                rgb,depth=scene();depth[mask>0]=.54
                self.assertEqual(observe(rgb,depth),[])

    def test_visible_relocated_object_is_reported_as_moved_not_invisible(self):
        rgb,depth=scene([dict(shape='box',size=(60,40),height=.06)])
        original=observe(rgb,depth)[0];moved=copy.deepcopy(original)
        moved.center_world[0]+=.18
        with self.assertRaisesRegex(TargetMovedError,'visible but moved 180 mm'):
            select_tabletop_object([moved],reference=original)

    def test_black_blue_green_white_objects_share_the_same_geometry(self):
        observations=[]
        for colour in ((0,0,0),(20,40,230),(20,200,30),(255,255,255),(220,30,10)):
            rgb,depth=scene([{}],colour)
            result=observe(rgb,depth)
            self.assertEqual(len(result),1)
            observations.append(result[0])
        for item in observations[1:]:
            np.testing.assert_allclose(item.center_world,observations[0].center_world,atol=1e-10)
            np.testing.assert_allclose(item.size_xyz,observations[0].size_xyz,atol=1e-10)
            self.assertEqual(item.shape,'box')

    def test_box_center_dimensions_height_and_mask_come_from_depth(self):
        rgb,depth=scene([dict(center=(180,105),size=(60,40),height=.06)])
        obj=observe(rgb,depth)[0]
        np.testing.assert_allclose(obj.center_world,[.018,.0135,.43],atol=.002)
        np.testing.assert_allclose(obj.size_xyz,[.054,.036,.06],atol=.003)
        self.assertEqual(obj.shape,'box')
        self.assertLess(yaw_error(obj.yaw_world_rad,0.),np.deg2rad(2))
        self.assertEqual(obj.mask.shape,depth.shape)
        self.assertEqual(int(np.count_nonzero(obj.mask)),obj.mask_pixels)
        self.assertEqual(len(obj.bbox),4)

    def test_rotated_long_rectangle_uses_its_long_axis_not_the_diagonal(self):
        rgb,depth=scene([dict(size=(78,36),angle=31)])
        obj=observe(rgb,depth)[0]
        self.assertEqual(obj.shape,'box')
        self.assertGreater(obj.size_xyz[0],obj.size_xyz[1])
        np.testing.assert_allclose(obj.size_xyz[:2],[.0702,.0324],atol=.004)
        self.assertLess(yaw_error(obj.yaw_world_rad,np.deg2rad(-31)),np.deg2rad(3))

    def test_round_top_is_a_cylinder_without_a_colour_or_known_radius(self):
        rgb,depth=scene([dict(shape='cylinder',center=(155,125),size=(46,46),height=.075)],(30,210,30))
        obj=observe(rgb,depth)[0]
        self.assertEqual(obj.shape,'cylinder')
        np.testing.assert_allclose(obj.size_xyz,[.04025,.04025,.075],atol=.003)
        np.testing.assert_allclose(obj.center_world,[-.004375,-.004375,.4375],atol=.002)
        self.assertEqual(obj.yaw_world_rad,0.)

    def test_multiple_objects_require_explicit_selection_and_keep_identity(self):
        rgb,depth=scene([dict(center=(85,100),size=(42,28)),
                         dict(center=(235,150),shape='cylinder',size=(44,44))])
        candidates=observe(rgb,depth)
        self.assertEqual(len(candidates),2)
        self.assertLess(candidates[0].center_world[0],candidates[1].center_world[0])
        with self.assertRaises(RuntimeError): select_tabletop_object(candidates)
        self.assertIs(select_tabletop_object(candidates,target_index=1),candidates[1])
        reference=copy.deepcopy(candidates[0]);reference.center_world[0]+=.005
        self.assertIs(select_tabletop_object(candidates,reference=reference),candidates[0])
        reference.center_world[0]+=.2
        with self.assertRaises(RuntimeError): select_tabletop_object(candidates,reference=reference)
        with self.assertRaises(ValueError): select_tabletop_object(candidates,target_index=2)

    def test_plain_table_invalid_depth_and_out_of_height_objects_are_not_targets(self):
        rgb,depth=scene(colour=(255,0,0))
        self.assertEqual(observe(rgb,depth),[])
        for value in (np.nan,np.inf,0.,-1.):
            with self.subTest(value=value):
                self.assertEqual(observe(rgb,np.full(depth.shape,value)),[])
        for height in (.005,.25):
            with self.subTest(height=height):
                rgb,depth=scene([dict(height=height)])
                self.assertEqual(observe(rgb,depth),[])

    def test_malformed_inputs_and_camera_calibration_are_rejected(self):
        rgb,depth=scene([{}])
        cases=[(rgb[:,:-1],depth,K,T),
               (rgb,depth[:,:,None],K,T),
               (rgb,depth,np.eye(2),T),
               (rgb,depth,K,np.eye(3))]
        zero=K.copy();zero[0,0]=0
        cases.append((rgb,depth,zero,T))
        stretched=T.copy();stretched[0,0]=2
        cases.append((rgb,depth,K,stretched))
        for a,b,c,d in cases:
            with self.subTest(shapes=(a.shape,b.shape,c.shape,d.shape)):
                with self.assertRaises(ValueError): detect_tabletop_objects(a,b,c,d,TABLE_Z)
        with self.assertRaises(ValueError): observe(rgb,depth,workspace_xy=((1,0),(0,1)))

    def test_input_rgb_depth_and_calibration_are_never_mutated(self):
        rgb,depth=scene([{}]);depth[15,20]=np.nan
        before=[value.copy() for value in (rgb,depth,K,T)]
        observe(rgb,depth)
        for value,original in zip((rgb,depth,K,T),before):
            np.testing.assert_array_equal(value,original)

    def test_different_unknown_dimensions_are_measured_not_reused_from_a_cube(self):
        for size,height in (((36,24),.035),((84,48),.095)):
            with self.subTest(size=size,height=height):
                rgb,depth=scene([dict(size=size,height=height)])
                obj=observe(rgb,depth)[0]
                expected=np.r_[np.array(size)*(.60-height)/600,height]
                np.testing.assert_allclose(obj.size_xyz,expected,atol=.003)
                self.assertAlmostEqual(obj.center_world[2],TABLE_Z+height/2,places=5)

    def test_nontrivial_camera_extrinsic_maps_to_world_not_pixel_coordinates(self):
        rgb,depth=scene([dict(center=(180,105),size=(60,40))])
        theta=.44;c,s=np.cos(theta),np.sin(theta)
        rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        transform=T.copy();transform[:3,:3]=rz@T[:3,:3];transform[:2,3]=[.2,-.1]
        obj=detect_tabletop_objects(rgb,depth,K,transform,TABLE_Z)[0]
        xy=(rz@np.array([.018,.0135,0.]))[:2]+[.2,-.1]
        np.testing.assert_allclose(obj.center_world,[*xy,.43],atol=.002)
        self.assertLess(yaw_error(obj.yaw_world_rad,theta),np.deg2rad(2))

    def test_workspace_excludes_external_and_straddling_geometry_instead_of_cropping(self):
        rgb,depth=scene([dict(size=(60,40))])
        self.assertEqual(len(observe(rgb,depth,workspace_xy=((-0.1,.1),(-.1,.1)))),1)
        self.assertEqual(observe(rgb,depth,workspace_xy=((.1,.2),(-.1,.1))),[])
        # Cutting a known 54 mm footprint in half must not invent a 27 mm box.
        self.assertEqual(observe(rgb,depth,workspace_xy=((0.,.1),(-.1,.1))),[])

    def test_tiny_and_image_clipped_targets_abstain(self):
        for item in (dict(size=(4,4)),dict(center=(5,120),size=(30,30))):
            with self.subTest(item=item):
                rgb,depth=scene([item]);self.assertEqual(observe(rgb,depth),[])
        with self.assertRaises(RuntimeError): select_tabletop_object([])


if __name__=='__main__': unittest.main()
