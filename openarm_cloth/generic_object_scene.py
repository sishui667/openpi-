"""Compile rigid box/cylinder fixtures without editing the frozen MJCF.

This factory receives a shape and dimensions, never an object's measured pose.
The same factory is used for physical test fixtures and for a separate visual
planning model; callers must not obtain planning dimensions from simulator
object state. Existing robot cameras, joints and contact physics are retained.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from member_b_scene import repair_finger_servos
from task_workspace import CONFIG, TABLE_TOP
from teleop_towel_dataset import build_finger_actuators


SCENE = Path(__file__).resolve().parent / "simulation/v1/droid_cube_scene_perception.xml"
TABLE_RIM_HEIGHT = .010
TABLE_RIM_THICKNESS = .006
PLACEMENT_INSET = .008


def add_table_boundary(spec):
    """Low visible physical rails, shared by execution and visual-query worlds.

    The rails' inner faces meet the table edges; they do not consume its usable
    surface. They are real static obstacles, not an invisible object constraint.
    Their 10 mm top stays below the detector's 12 mm foreground-height cutoff.
    """
    cx,cy=CONFIG['table_center_m'][:2];hx,hy=CONFIG['table_half_size_m'][:2]
    half=TABLE_RIM_THICKNESS/2;z=TABLE_TOP+TABLE_RIM_HEIGHT/2
    for name,pos,size in (
        ('near',[cx-hx-half,cy,z],[half,hy+TABLE_RIM_THICKNESS,TABLE_RIM_HEIGHT/2]),
        ('far',[cx+hx+half,cy,z],[half,hy+TABLE_RIM_THICKNESS,TABLE_RIM_HEIGHT/2]),
        ('right',[cx,cy-hy-half,z],[hx,half,TABLE_RIM_HEIGHT/2]),
        ('left',[cx,cy+hy+half,z],[hx,half,TABLE_RIM_HEIGHT/2]),
    ):
        spec.worldbody.add_geom(name='tabletop_rim_'+name,type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=pos,size=size,rgba=[.12,.70,.82,1],
                               contype=1,conaffinity=1,condim=3,group=0,
                               friction=[1,.01,.001],solref=[.005,1])
    # Lines indicate the allowed object footprint, not a fixed centre region.
    x0,x1=cx-hx+PLACEMENT_INSET,cx+hx-PLACEMENT_INSET
    y0,y1=cy-hy+PLACEMENT_INSET,cy+hy-PLACEMENT_INSET
    for name,pos,size in (
        ('near',[x0,cy,TABLE_TOP+.0006],[.001,(y1-y0)/2,.0005]),
        ('far',[x1,cy,TABLE_TOP+.0006],[.001,(y1-y0)/2,.0005]),
        ('right',[cx,y0,TABLE_TOP+.0006],[(x1-x0)/2,.001,.0005]),
        ('left',[cx,y1,TABLE_TOP+.0006],[(x1-x0)/2,.001,.0005]),
    ):
        spec.worldbody.add_site(name='tabletop_placement_'+name,type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=pos,size=size,rgba=[1,.72,.08,1])


class PlacementFixture:
    """User-operated object reset before camera-guided execution only.

    The saved pose belongs to scene setup, never perception or IK. There is no
    pose getter and reset is forbidden after placement has been accepted.
    """
    def __init__(self,model,data):
        joint=model.joint('droid_object_freejoint')
        self.address=int(joint.qposadr[0]);self.dof=int(joint.dofadr[0])
        self.body=model.body('droid_object').id
        self.initial=data.qpos[self.address:self.address+7].copy();self.model=model
        self.geom=model.geom('droid_object_geom').id;self._active=True
        if int(model.body_parentid[self.body])!=0:
            raise ValueError('Placement fixture requires a world-attached free body')
        self._geom_rotation=np.empty(9)
        mujoco.mju_quat2Mat(self._geom_rotation,model.geom_quat[self.geom])
        self._geom_rotation=self._geom_rotation.reshape(3,3)

    def constrain(self,data,*,placement_active):
        """Keep the *complete rotated object* over the table while editing only.

        This edits fixture initial conditions, not grasp dynamics. No centre,
        size, or pose measured here is ever returned to perception or IK.
        """
        if not placement_active or not self._active:
            raise RuntimeError('Placement boundary is allowed only during user placement, never during a grasp')
        # Read the editing freejoint directly, including a mouse update made
        # after the last mj_step. A full robot mj_forward on every no-op would
        # make the placement UI unnecessarily slow at the 2 kHz physics rate.
        body_rotation=np.empty(9)
        mujoco.mju_quat2Mat(body_rotation,data.qpos[self.address+3:self.address+7])
        body_rotation=body_rotation.reshape(3,3)
        rotation=body_rotation@self._geom_rotation
        size=self.model.geom_size[self.geom]
        if self.model.geom_type[self.geom]==mujoco.mjtGeom.mjGEOM_BOX:
            extent=np.abs(rotation)@size
        elif self.model.geom_type[self.geom]==mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis=rotation[:,2]
            extent=size[0]*np.sqrt(np.maximum(0.,1.-axis*axis))+size[1]*np.abs(axis)
        else:
            raise ValueError('Unsupported placement fixture geometry')
        center=data.qpos[self.address:self.address+3]+body_rotation@self.model.geom_pos[self.geom]
        table_center=np.asarray(CONFIG['table_center_m'][:2])
        table_half=np.asarray(CONFIG['table_half_size_m'][:2])
        lower=table_center-table_half+PLACEMENT_INSET+extent[:2]
        upper=table_center+table_half-PLACEMENT_INSET-extent[:2]
        if np.any(lower>upper):
            raise RuntimeError('Rotated object cannot fit inside the tabletop placement boundary; press R to restore')
        target=center.copy();target[:2]=np.clip(center[:2],lower,upper)
        # Let ordinary table contacts settle naturally. Only recover a mouse
        # edit that has actually pushed the bottom through the support plane.
        if center[2]-float(extent[2])<TABLE_TOP-.0005:
            target[2]=TABLE_TOP+float(extent[2])+.001
        correction=target-center;changed=bool(np.any(np.abs(correction)>1e-10))
        if changed:
            data.qpos[self.address:self.address+3]+=correction
            axes=np.flatnonzero(np.abs(correction)>1e-10)
            data.qvel[self.dof+axes]=0.
            # Do not keep an outwards mouse force after correcting the fixture.
            data.xfrc_applied[self.body]=0.
            data.qfrc_applied[self.dof:self.dof+6]=0.
            mujoco.mj_forward(self.model,data)
        return changed

    def finish(self,data):
        """Accept placement once, clear stale mouse forces, disable all edits."""
        self.constrain(data,placement_active=True)
        data.qvel[self.dof:self.dof+6]=0.
        data.xfrc_applied[self.body]=0.
        data.qfrc_applied[self.dof:self.dof+6]=0.
        self._active=False
        mujoco.mj_forward(self.model,data)

    def reset(self,data,*,placement_active):
        if not placement_active or not self._active: raise RuntimeError('Object reset is allowed only during user placement, never during a grasp')
        data.qpos[self.address:self.address+7]=self.initial
        data.qvel[self.dof:self.dof+6]=0.
        data.xfrc_applied[self.body]=0.
        data.qfrc_applied[self.dof:self.dof+6]=0.
        mujoco.mj_forward(self.model,data)


def validate_geometry(shape, size_xyz):
    shape = str(shape).lower()
    size = np.asarray(size_xyz, dtype=float)
    if shape not in ("box", "cylinder"):
        raise ValueError("Only box and upright cylinder geometries are supported")
    if size.shape != (3,) or not np.isfinite(size).all() or np.min(size) <= 0:
        raise ValueError("size_xyz must be three finite positive dimensions in metres")
    if shape == "cylinder" and not np.isclose(size[0], size[1], rtol=.02, atol=1e-5):
        raise ValueError("A cylinder requires equal X/Y diameters")
    return shape, size.copy()


def build_object_model(shape="box", size_xyz=(.0576, .0576, .0576),
                       rgba=(.85, .12, .08, 1), mass=.18):
    """Return an independent compiled model, preserving the base ready key.

    Box size is full local X/Y/Z extent. Cylinder size is diameter, diameter,
    full height, with its axis along local Z. Initial object placement remains
    the base keyframe plus the selected tabletop shift; fixture code is
    responsible for placing differently sized objects above their support.
    No temporary placement markers or target sites are introduced.
    """
    shape, size = validate_geometry(shape, size_xyz)
    color = np.asarray(rgba, dtype=float)
    if color.shape != (4,) or not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 1):
        raise ValueError("rgba must contain four finite components in [0, 1]")
    mass = float(mass)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive")

    spec = mujoco.MjSpec.from_file(str(SCENE))
    table = spec.geom("folding_table")
    original_top = float(table.pos[2] + table.size[2])
    table.pos = CONFIG["table_center_m"]
    table.size = CONFIG["table_half_size_m"]
    add_table_boundary(spec)
    geom = spec.geom("droid_object_geom")
    geom.rgba = color
    geom.meshname = ""
    if shape == "box":
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = size / 2
        inertia = mass / 12 * np.array([size[1] ** 2 + size[2] ** 2,
                                       size[0] ** 2 + size[2] ** 2,
                                       size[0] ** 2 + size[1] ** 2])
    else:
        radius, height = .25 * (size[0] + size[1]), size[2]
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [radius, height / 2, 0]
        inertia = mass * np.array([(3 * radius ** 2 + height ** 2) / 12,
                                   (3 * radius ** 2 + height ** 2) / 12,
                                   radius ** 2 / 2])
    body = spec.body("droid_object")
    body.mass = mass
    body.inertia = inertia
    body.ipos = [0, 0, 0]
    body.iquat = [1, 0, 0, 0]
    body.explicitinertial = True
    # Recompile rather than patching geom_size on a live model: MuJoCo must
    # update broad-phase bounds, rendering constants and inertia consistently.
    model = spec.compile()
    repair_finger_servos(model, build_finger_actuators(model))
    address = int(model.joint("droid_object_freejoint").qposadr[0])
    shift = TABLE_TOP - original_top
    model.qpos0[address + 2] += shift
    if model.nkey:
        model.key_qpos[:, address + 2] += shift
    return model
