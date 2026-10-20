"""Colour-independent RGB-D geometry for isolated, upright tabletop solids.

No simulator, object IDs, object dimensions or object poses enter this module.
The support plane and workspace are measured calibration, not target locations.
This is geometric perception, not a learned semantic object recognizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ObjectObservation:
    center_world: np.ndarray
    size_xyz: np.ndarray
    yaw_world_rad: float
    shape: str
    mask_pixels: int
    bbox: tuple[int,int,int,int]
    mask: np.ndarray

    @property
    def shape_label(self):
        """Display-only subtype inferred from measured extents, not fixture type."""
        if self.shape == 'box':
            size = np.asarray(self.size_xyz, dtype=float)
            if float(np.ptp(size)) <= .10 * float(np.max(size)):
                return 'cube'
        return self.shape

    def as_dict(self):
        return dict(center_world=self.center_world.tolist(),size_xyz=self.size_xyz.tolist(),
                    yaw_world_rad=self.yaw_world_rad,shape=self.shape,shape_label=self.shape_label,
                    mask_pixels=self.mask_pixels,bbox=list(self.bbox),
                    source='RGB-D geometry; no colour class or simulator object state')


class TargetMovedError(RuntimeError):
    """A matching object is visible, but its old grasp plan is no longer valid."""


def detect_tabletop_objects(rgb,depth,camera_matrix,camera_to_world,table_z,
                            workspace_xy=None,min_height=.012,max_height=.20):
    """Return separate visible box/cylinder candidates, sorted by world x/y.

    Assumes opaque upright solids on the calibrated horizontal support plane.
    Top dimensions and height are estimated from depth, never supplied as a
    known cube size. Clipped, tiny, oversized and non-compact regions abstain.
    Touching objects or occluded top faces are NOT reliably instance-separated.
    ``workspace_xy`` is ((xmin,xmax),(ymin,ymax)), in calibrated robot metres.
    """
    rgb=np.asarray(rgb);depth=np.asarray(depth,dtype=float)
    k=np.asarray(camera_matrix,dtype=float);transform=np.asarray(camera_to_world,dtype=float)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]:
        raise ValueError('Expected RGB HxWx3 and aligned metric-depth HxW')
    if k.shape!=(3,3) or transform.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(transform).all():
        raise ValueError('Invalid camera calibration')
    if k[0,0]<=0 or k[1,1]<=0 or not np.isfinite(table_z) or not 0<min_height<max_height:
        raise ValueError('Invalid camera focal length or support-height interval')
    if not np.allclose(transform[3],[0,0,0,1]) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5):
        raise ValueError('camera_to_world must be a rigid homogeneous transform')
    height,width=depth.shape
    yy,xx=np.indices(depth.shape)
    valid=np.isfinite(depth)&(depth>.03)&(depth<5.)
    z=np.where(valid,depth,0.)
    local=np.stack(((xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z),axis=-1)
    points=local@transform[:3,:3].T+transform[:3,3]
    above=points[:,:,2]-float(table_z)
    valid&=(above>=min_height)&(above<=max_height)
    bounds=None
    if workspace_xy is not None:
        bounds=np.asarray(workspace_xy,dtype=float)
        if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1]):
            raise ValueError('workspace_xy must contain increasing x and y intervals')
    # Small holes from depth edges may be closed, but invalid pixels never
    # become geometric measurements just because morphology filled a mask.
    binary=cv2.morphologyEx(valid.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    result=[]
    for component in range(1,count):
        x,y,w,h,area=map(int,stats[component])
        if area<70 or min(w,h)<5 or x<2 or y<2 or x+w>width-2 or y+h>height-2: continue
        region=(labels==component)&valid
        cloud=points[region]
        if len(cloud)<70: continue
        # Do not crop an object at the workspace edge and then hallucinate a
        # smaller complete box. The entire measured component must fit.
        if bounds is not None and (np.any(cloud[:,:2]<bounds[:,0]) or np.any(cloud[:,:2]>bounds[:,1])): continue
        top_z=float(np.percentile(cloud[:,2],95))
        top=cloud[cloud[:,2]>=top_z-.002,:2]
        if len(top)<35: continue
        hull=cv2.convexHull(top.astype(np.float32))
        rectangle=cv2.minAreaRect(hull)
        sides=np.asarray(rectangle[1],dtype=float)
        if np.min(sides)<.012 or np.max(sides)>.16: continue
        area_hull=float(cv2.contourArea(hull));box_area=float(np.prod(sides))
        fill=area_hull/max(box_area,1e-10)
        if fill<.72: continue
        perimeter=float(cv2.arcLength(hull,True))
        vertices=len(cv2.approxPolyDP(hull,.015*perimeter,True))
        circularity=4*np.pi*area_hull/max(perimeter*perimeter,1e-10)
        aspect=float(max(sides)/min(sides))
        _,circle_radius=cv2.minEnclosingCircle(hull)
        circle_fill=area_hull/max(np.pi*circle_radius*circle_radius,1e-10)
        if vertices>=8 and circularity>.84 and circle_fill>.90 and aspect<1.2 and fill<.90:
            shape='cylinder'
        elif vertices==4 and fill>=.90:
            shape='box'
        else:
            # An unrecognised flat outline is not automatically a box. This
            # is conservative geometry fitting, not arbitrary-solid recognition:
            # objects with indistinguishable visible surfaces may still alias.
            continue
        yaw=np.deg2rad(float(rectangle[2]))
        if sides[0]<sides[1]: sides=sides[::-1];yaw+=np.pi/2
        yaw=float((yaw+np.pi/2)%np.pi-np.pi/2)
        if shape=='cylinder': sides[:]=np.mean(sides);yaw=0.
        object_height=top_z-float(table_z)
        center=np.array([*rectangle[0],float(table_z)+object_height/2])
        mask=(region.astype(np.uint8)*255)
        result.append(ObjectObservation(center,np.r_[sides,object_height],yaw,shape,
                                        int(region.sum()),(x,y,w,h),mask))
    return sorted(result,key=lambda obs:(float(obs.center_world[0]),float(obs.center_world[1])))


def select_tabletop_object(observations,reference=None,target_index=None,max_shift=.035):
    """Explicit initial index or unique target; later reobserve the SAME target.

    Never silently take the largest foreground blob or swap to another object.
    A target index is a user selection of visually enumerated candidates.
    """
    if not observations: raise RuntimeError('No complete supported tabletop object in the RGB-D image')
    if reference is not None:
        matching=[o for o in observations if o.shape==reference.shape and np.allclose(o.size_xyz,reference.size_xyz,atol=.012,rtol=.2)]
        candidates=[o for o in matching if np.linalg.norm(o.center_world-reference.center_world)<max_shift]
        if not candidates and len(matching)==1:
            distance=float(np.linalg.norm(matching[0].center_world-reference.center_world))
            raise TargetMovedError(f'Target is visible but moved {distance*1000:.0f} mm; re-observe and replan, do not use the old grasp')
        if len(candidates)!=1: raise RuntimeError('Target identity is ambiguous or its visible geometry is incomplete')
        return candidates[0]
    if target_index is not None:
        if target_index<0 or target_index>=len(observations): raise ValueError('Visual target index out of range')
        return observations[target_index]
    if len(observations)!=1: raise RuntimeError('Multiple tabletop objects; choose a visual --target-index')
    return observations[0]
