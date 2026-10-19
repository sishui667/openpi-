"""Human relative-wrist teleoperation and robot-only pad IK.

No object state or target-object coordinate is used by these controllers.
Ground-truth grasp evaluation belongs in a separate component.
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent / "pose2arm-main"))
from retarget import compute_body_frame, OPENARM_BODY_TO_ROBOT
from teleop_towel_dataset import build_arm_specs
from handeye_grasp_red_cube import finger_joint_indices, solve_gripper_contact_proxy_q

PAD_PROXIES = (np.array([-.00018,-.00798,.045]),np.array([.00018,-.00798,.045]))


def human_right_wrist(world):
    """A torso-relative wrist displacement in robot axes, nominal model metres."""
    return human_wrist(world,'right')


def human_wrist(world,side='right'):
    if side not in ('left','right'): raise ValueError('unknown human arm side')
    shoulder,elbow,wrist=(11,13,15) if side=='left' else (12,14,16)
    if world is None: return None
    w=np.asarray(world,dtype=float)
    if w.shape!=(33,4): return None
    ids=[11,12,23,24,elbow,wrist]
    if not np.isfinite(w[ids]).all(): return None
    if np.min(w[[11,12,elbow,wrist],3])<.45: return None
    if np.linalg.norm(w[12,:3]-w[11,:3])<.05: return None
    if np.linalg.norm((w[11,:3]+w[12,:3]-w[23,:3]-w[24,:3])/2)<.05: return None
    try:
        origin,rotation=compute_body_frame(*[w[i,:3] for i in (11,12,23,24)])
    except ValueError:
        return None
    return OPENARM_BODY_TO_ROBOT @ (rotation.T @ (w[wrist,:3]-w[shoulder,:3]))


def human_right_pose(world, hand_world=None, previous_normal=None):
    """Torso-relative upper/forearm directions and palm attitude (not object truth)."""
    return human_arm_pose(world,hand_world,previous_normal,side='right')


def human_arm_pose(world,hand_world=None,previous_normal=None,side='right'):
    if human_wrist(world,side) is None: return None
    shoulder,elbow,wrist=(11,13,15) if side=='left' else (12,14,16)
    w=np.asarray(world,dtype=float)
    _,body=compute_body_frame(*[w[i,:3] for i in (11,12,23,24)])
    transform=OPENARM_BODY_TO_ROBOT @ body.T
    upper=transform @ (w[elbow,:3]-w[shoulder,:3])
    fore=transform @ (w[wrist,:3]-w[elbow,:3])
    if min(np.linalg.norm(upper),np.linalg.norm(fore))<.06: return None
    upper/=np.linalg.norm(upper);fore/=np.linalg.norm(fore)
    normal=np.cross(upper,fore)
    if np.linalg.norm(normal)<.05:
        normal=np.array([0.,1.,0.]) if previous_normal is None else previous_normal
    normal=normal-fore*np.dot(normal,fore)
    if np.linalg.norm(normal)<1e-5:
        normal=np.cross(fore,np.eye(3)[np.argmin(np.abs(fore))])
    normal/=np.linalg.norm(normal)
    x=np.cross(normal,fore);x/=np.linalg.norm(x)
    attitude=np.column_stack((x,np.cross(fore,x),fore))
    arm_rotation=attitude.copy()
    palm=False
    if hand_world is not None:
        h=np.asarray(hand_world,dtype=float)
        if h.shape==(21,3) and np.isfinite(h).all():
            z=transform @ (h[9]-h[0]);lateral=transform @ (h[5]-h[17])
            if min(np.linalg.norm(z),np.linalg.norm(lateral))>.01:
                z/=np.linalg.norm(z);x=lateral-z*np.dot(lateral,z)
                if np.linalg.norm(x)>.005:
                    x/=np.linalg.norm(x);attitude=np.column_stack((x,np.cross(z,x),z));palm=True
    # The radial/index axis has opposite anatomical handedness. Without this
    # side sign, facing palms can yield identical hand-frame orientations.
    palm_normal=(1. if side=='right' else -1.)*attitude[:,1] if palm else None
    return dict(upper=upper,fore=fore,rotation=attitude,arm_rotation=arm_rotation,palm=palm,normal=normal,palm_normal=palm_normal)


def arm_frame(upper,fore):
    z=fore/np.linalg.norm(fore)
    normal=np.cross(upper,z)
    if np.linalg.norm(normal)<1e-5: normal=np.cross(z,np.eye(3)[np.argmin(np.abs(z))])
    normal/=np.linalg.norm(normal)
    x=np.cross(normal,z)
    return np.column_stack((x,np.cross(z,x),z))


class ArmPoseIK:
    """Actual B-model direction IK: shoulder/elbow/wrist plus calibrated palm rotation.

    Anatomical direction targets replace the small tabletop wrist-only box.
    Uses robot encoders and geometry; never copies the free cube state.
    """
    def __init__(self,pad_ik,wrist_mode='full',direction_tau=.065,palm_alignment='relative'):
        if wrist_mode not in ('full','twist'): raise ValueError('unknown wrist mapping mode')
        self.wrist_mode=wrist_mode
        self.direction_tau=max(0.,float(direction_tau))
        if palm_alignment not in ('relative','absolute'): raise ValueError('unknown palm alignment')
        self.palm_alignment=palm_alignment
        self.palm_target=None
        self.pad=pad_ik;self.model=pad_ik.model;self.kin=pad_ik.kin;self.spec=pad_ik.spec
        self.side=pad_ik.side
        self.joints=[int(self.model.joint(f'openarm_{self.side}_joint{i}').id) for i in (1,4,7)]
        self.human_origin=None;self.robot_origin=None;self.rotation=None;self.palm_origin=False
        self.upper=None;self.fore=None
        self.hand_origin=None;self.last_local_palm=None
        self.twist_origin=None;self.twist_previous=None;self.twist_angle=None
        self.twist_target=0.;self.twist_limited=False
        self.absolute_previous=None;self.absolute_angle=None;self.absolute_candidate=None
        self.absolute_reacquiring=False
        self.absolute_arm_frame=None
        self.absolute_rejected_this_frame=False
        self.absolute_accepted_total=0;self.absolute_rejected_total=0
        self.absolute_reacquisitions=0
        self.palm_target=None
        self.metrics={}

    def reset(self):
        self.human_origin=None;self.rotation=None;self.upper=None;self.fore=None
        self.hand_origin=None;self.last_local_palm=None
        self.twist_origin=None;self.twist_previous=None;self.twist_angle=None
        self.twist_target=0.;self.twist_limited=False
        self.absolute_previous=None;self.absolute_angle=None;self.absolute_candidate=None
        self.absolute_reacquiring=False
        self.absolute_arm_frame=None
        self.absolute_rejected_this_frame=False
        self.palm_target=None

    def _mark_absolute_palm_missing(self):
        self.absolute_candidate=None
        if self.absolute_previous is not None: self.absolute_reacquiring=True

    def _continuous_absolute_arm_frame(self,frame):
        """Choose the continuous sign of an otherwise ambiguous arm plane.

        An almost straight elbow can change the cross-product normal's sign
        under tiny keypoint noise. Its +/-X,Y frames describe the same upper
        and forearm directions, but using the discontinuous frame would turn
        a held relative wrist command by 180 degrees in the world. This gauge
        choice changes neither direction targets nor the anatomical palm.
        """
        result=np.asarray(frame,dtype=float).copy()
        if self.absolute_arm_frame is not None:
            z=result[:,2]
            reference=self.absolute_arm_frame[:,0]
            reference=reference-z*np.dot(reference,z)
            if np.linalg.norm(reference)>1e-5 and np.dot(result[:,0],reference)<0:
                result[:,:2]*=-1
        self.absolute_arm_frame=result.copy()
        return result

    def _update_absolute_twist(self,angle,dt=.05):
        """Unwrap observed rotation before joint limits; confirm ambiguous flips.

        Called once per new palm observation by the controller. A wrapped
        +179 -> -179 degree estimate is a 2 degree turn, not a command to
        traverse both wrist stops. An isolated implausible >110 degree jump
        is held. Only after genuine palm loss/degeneracy or a source gap may
        two consistent new observations confirm a large reacquisition change.
        Repetition during uninterrupted visibility is not proof of a real turn.
        """
        raw=float(np.arctan2(np.sin(angle),np.cos(angle)))
        # A missing arm skips update_target entirely; its next valid update
        # carries the elapsed source interval. Do not count pre-occlusion
        # evidence as the first of two consecutive reacquisition observations.
        if dt>.10: self._mark_absolute_palm_missing()
        if self.absolute_previous is None:
            self.absolute_previous=raw;self.absolute_angle=raw
        else:
            increment=float(np.arctan2(np.sin(raw-self.absolute_previous),np.cos(raw-self.absolute_previous)))
            if abs(increment)>np.deg2rad(110):
                candidate=self.absolute_candidate
                consistent=self.absolute_reacquiring and candidate is not None and abs(float(np.arctan2(np.sin(raw-candidate),np.cos(raw-candidate))))<=np.deg2rad(25)
                if not consistent:
                    self.absolute_candidate=raw if self.absolute_reacquiring else None
                    self.absolute_rejected_this_frame=True
                    self.absolute_rejected_total+=1
                    return False
                self.absolute_reacquisitions+=1
            self.absolute_angle+=increment;self.absolute_previous=raw
        self.absolute_candidate=None;self.absolute_reacquiring=False
        self.absolute_accepted_total+=1
        self.twist_target=float(np.clip(self.absolute_angle,*self.spec.q_range[4]))
        self.twist_limited=abs(self.absolute_angle-self.twist_target)>1e-5
        return True

    def update_target(self,pose,robot_q,dt=.05):
        self.absolute_rejected_this_frame=False
        self.kin.qpos[self.spec.qpos]=robot_q;mujoco.mj_forward(self.model,self.kin)
        arm_rotation=pose.get('arm_rotation',arm_frame(pose['upper'],pose['fore']))
        if self.wrist_mode=='twist' and self.palm_alignment=='absolute':
            arm_rotation=self._continuous_absolute_arm_frame(arm_rotation)
        local_palm=arm_rotation.T @ pose['rotation']
        first_target=self.human_origin is None
        if self.human_origin is None:
            self.human_origin=np.eye(3)
            if self.rotation is None:
                # Neutral wrist attitude relative to the arm, not the unrelated
                # downward grasp-ready attitude in the global frame.
                neutral=np.asarray(robot_q).copy();neutral[4:]=0
                self.kin.qpos[self.spec.qpos]=neutral;mujoco.mj_forward(self.model,self.kin)
                u,v=self.geometry()
                self.robot_origin=arm_frame(u,v).T @ self.kin.xmat[self.spec.body_id].reshape(3,3)
                self.kin.qpos[self.spec.qpos]=robot_q;mujoco.mj_forward(self.model,self.kin)
                self.rotation=self.kin.xmat[self.spec.body_id].reshape(3,3).copy()
        if pose['palm']:
            if self.hand_origin is None:
                self.hand_origin=local_palm.copy()
                # First acquisition is continuous. Later reacquisition does NOT
                # reset the wrist zero; only explicit C performs recalibration.
                if not first_target and self.wrist_mode=='full': self.robot_origin=arm_rotation.T @ self.rotation
            self.last_local_palm=local_palm @ self.hand_origin.T
        wrist_delta=np.eye(3) if self.last_local_palm is None else self.last_local_palm
        if self.wrist_mode=='twist':
            # Lateral palm axis projected perpendicular to the forearm:
            # axial rotation is independent of hand longitudinal-axis tilt.
            absolute=self.palm_alignment=='absolute' and pose.get('palm_normal') is not None
            if not absolute and self.palm_alignment=='absolute': self._mark_absolute_palm_missing()
            if absolute:
                # Robot hand broad-face normal is TCP +X (the finger plane is
                # TCP YZ). Match absolute anatomical palm facing, rather than
                # erasing both hands' relative orientation at calibration.
                palm=np.asarray(pose['palm_normal'],dtype=float)
                projected=arm_rotation.T @ palm
                neutral=self.robot_origin[:,0]
                if np.linalg.norm(projected[:2])>.25:
                    angle=float(np.arctan2(projected[1],projected[0])-np.arctan2(neutral[1],neutral[0]))
                    if self._update_absolute_twist(angle,dt):
                        desired=arm_rotation @ np.r_[projected[:2]/np.linalg.norm(projected[:2]),0.]
                        self.palm_target=desired
                else: self._mark_absolute_palm_missing()
            elif pose['palm'] and np.linalg.norm(local_palm[:2,0])>.25:
                angle=float(np.arctan2(local_palm[1,0],local_palm[0,0]))
                if self.twist_origin is None:
                    self.twist_origin=angle;self.twist_previous=angle;self.twist_angle=angle
                else:
                    increment=float(np.arctan2(np.sin(angle-self.twist_previous),np.cos(angle-self.twist_previous)))
                    # A single 180-degree palm flip is unobservable/ambiguous,
                    # not evidence of a real forearm rotation within one frame.
                    if abs(increment)<np.deg2rad(110):
                        self.twist_angle+=increment;self.twist_previous=angle
                raw=self.twist_angle-self.twist_origin
                self.twist_target=float(np.clip(raw,*self.spec.q_range[4]))
                self.twist_limited=abs(raw-self.twist_target)>1e-5
            wrist_delta=Rotation.from_rotvec([0,0,self.twist_target]).as_matrix()
        wanted=arm_rotation @ wrist_delta @ self.robot_origin
        if self.wrist_mode=='full' and self.palm_alignment=='absolute' and pose.get('palm_normal') is not None:
            x=np.asarray(pose['palm_normal'],dtype=float);x=x/np.linalg.norm(x)
            z=pose['rotation'][:,2];z=z-x*np.dot(x,z);z=z/np.linalg.norm(z)
            wanted=np.column_stack((x,np.cross(z,x),z))
            self.palm_target=x.copy()
        delta=Rotation.from_matrix(wanted @ self.rotation.T).as_rotvec()
        self.rotation=Rotation.from_rotvec(delta*min(1.,4.*dt/max(np.linalg.norm(delta),1e-9))).as_matrix() @ self.rotation
        alpha=1. if self.direction_tau==0 else 1-np.exp(-dt/self.direction_tau)
        for name in ('upper','fore'):
            old=getattr(self,name)
            value=pose[name].copy() if old is None else old+alpha*(pose[name]-old)
            setattr(self,name,value/np.linalg.norm(value))

    def geometry(self,with_jac=False):
        points=[];jacs=[]
        for joint in self.joints:
            point=self.kin.xanchor[joint].copy();points.append(point)
            if with_jac:
                mujoco.mj_jac(self.model,self.kin,self.pad.jp,self.pad.jr,point,int(self.model.jnt_bodyid[joint]))
                jacs.append(self.pad.jp[:,self.spec.dof].copy())
        vectors=[points[1]-points[0],points[2]-points[1]]
        lengths=[np.linalg.norm(v) for v in vectors]
        units=[v/n for v,n in zip(vectors,lengths)]
        if not with_jac: return units
        dj=[(np.eye(3)-np.outer(u,u))/n @ (jacs[i+1]-jacs[i]) for i,(u,n) in enumerate(zip(units,lengths))]
        mujoco.mj_jacBody(self.model,self.kin,self.pad.jp,self.pad.jr,self.spec.body_id)
        return units,dj,self.pad.jr[:,self.spec.dof].copy()

    def solve(self,seed,iterations=10):
        initial=np.asarray(seed,dtype=float);q=initial.copy()
        active=np.arange(5 if self.wrist_mode=='twist' else 7)
        if self.wrist_mode=='twist': q[5:]=0
        for _ in range(iterations):
            self.kin.qpos[self.spec.qpos]=q;mujoco.mj_forward(self.model,self.kin)
            units,direction_jac,angular_jac=self.geometry(True)
            rotation=self.kin.xmat[self.spec.body_id].reshape(3,3)
            angular=Rotation.from_matrix(self.rotation @ rotation.T).as_rotvec()
            error=np.r_[.24*(self.upper-units[0]),.24*(self.fore-units[1]),.055*angular,.008*(initial-q)]
            jac=np.vstack([.24*direction_jac[0],.24*direction_jac[1],.055*angular_jac,.008*np.eye(7)])
            jac=jac[:,active]
            step=np.linalg.solve(jac.T@jac+np.eye(len(active))*.00008,jac.T@error)
            q[active]+=np.clip(step,-.12,.12)
            q=np.clip(q,self.spec.q_range[:,0]+1e-7,self.spec.q_range[:,1]-1e-7)
        self.kin.qpos[self.spec.qpos]=q;mujoco.mj_forward(self.model,self.kin)
        units=self.geometry()
        angle=lambda a,b:float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1))))
        self.metrics=dict(upper_deg=angle(units[0],self.upper),fore_deg=angle(units[1],self.fore),
                          palm_deg=float(np.degrees(Rotation.from_matrix(self.rotation @ self.kin.xmat[self.spec.body_id].reshape(3,3).T).magnitude())))
        return q


class RelativeWristMapper:
    def __init__(self,start,gain=.7,speed=.16,table_top=.4):
        self.target=np.asarray(start,dtype=float).copy()
        self.anchor=self.target.copy()
        self.origin=None
        self.filtered=None
        self.gain=float(gain);self.speed=float(speed)
        self.lower=np.array([.32,-.22,table_top+.018])
        self.upper=np.array([.54,.06,.69])

    def recenter(self,point,current):
        p=np.asarray(point,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all(): return False
        self.origin=p.copy();self.filtered=p.copy()
        self.anchor=np.asarray(current,dtype=float).copy()
        self.target=self.anchor.copy()
        return True

    def update(self,point,dt=.05):
        if point is None: return self.target.copy()
        p=np.asarray(point,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all(): return self.target.copy()
        if self.origin is None:
            self.recenter(p,self.target)
            return self.target.copy()
        alpha=1-np.exp(-dt/.10)
        self.filtered+=alpha*(p-self.filtered)
        goal=np.clip(self.anchor+self.gain*(self.filtered-self.origin),self.lower,self.upper)
        delta=goal-self.target
        norm=np.linalg.norm(delta)
        if norm>.001:
            self.target+=delta*min(1,self.speed*dt/max(norm,1e-12))
        return self.target.copy()


class PadCartesianIK:
    """Damped Jacobian IK on calibrated pad midpoint, constant wrist attitude."""
    def __init__(self,model,data,ready=np.array([.43,0.,.50]),initialize_grasp=True,side='right'):
        if side not in ('left','right'): raise ValueError('unknown robot arm side')
        if side=='left' and initialize_grasp: raise ValueError('Left arm currently supports motion-only IK, not a calibrated grasp')
        self.side=side
        self.model=model
        self.spec=build_arm_specs(model)[side]
        self.kin=mujoco.MjData(model)
        # Copy robot encoders ONLY. Free-object state is never copied.
        for j in range(model.njnt):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or ""
            if name.startswith("openarm_"):
                adr=int(model.jnt_qposadr[j]);self.kin.qpos[adr]=data.qpos[adr]
        self.finger_indices=finger_joint_indices(model,side)
        self.geoms=[int(model.geom(f"openarm_{side}_{finger}_finger_collision").id) for finger in ("right","left")]
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros((3,model.nv))
        if not initialize_grasp:
            self.ready_q=data.qpos[self.spec.qpos].copy()
            self.ready_q[4:]=0
            self.kin.qpos[self.spec.qpos]=self.ready_q
            self.kin.qpos[self.finger_indices[0]]=.044
            mujoco.mj_forward(model,self.kin)
            self.rotation=self.kin.geom_xmat[self.geoms[0]].reshape(3,3).copy()
            self.ready_center=self.midpoint()
            return
        mujoco.mj_forward(model,self.kin)
        first,second=self.points()
        span=float(np.linalg.norm(first-second))
        q,*_=solve_gripper_contact_proxy_q(
            model,self.kin,self.spec,tuple(self.geoms),self.finger_indices,PAD_PROXIES,
            np.array([.43,0.,.43]),1,-1.,span,.044,data.qpos[self.spec.qpos].copy(),
            closing_direction=np.array([0.,1.,0.]),approach_down_weight=.8,multi_start=False)
        self.kin.qpos[self.spec.qpos]=q
        self.kin.qpos[self.finger_indices[0]]=.044
        mujoco.mj_forward(model,self.kin)
        self.rotation=self.kin.geom_xmat[self.geoms[0]].reshape(3,3).copy()
        self.ready_q,error=self.solve(ready,q,iterations=100)
        if error>.015: raise RuntimeError(f"Manual ready IK failed: {error:.4f} m")
        self.ready_center=self.midpoint()

    def points(self):
        return tuple(self.kin.geom_xpos[g]+self.kin.geom_xmat[g].reshape(3,3)@p for g,p in zip(self.geoms,PAD_PROXIES))

    def midpoint(self):
        return .5*sum(self.points())

    def solve(self,goal,seed,iterations=12):
        q=np.clip(np.asarray(seed,dtype=float),self.spec.q_range[:,0]+1e-7,self.spec.q_range[:,1]-1e-7)
        goal=np.asarray(goal,dtype=float)
        if goal.shape!=(3,) or not np.isfinite(goal).all(): raise ValueError("invalid human EE target")
        self.kin.qpos[self.finger_indices[0]]=.044
        for _ in range(iterations):
            self.kin.qpos[self.spec.qpos]=q;mujoco.mj_forward(self.model,self.kin)
            position=goal-self.midpoint()
            frame=self.kin.geom_xmat[self.geoms[0]].reshape(3,3)
            angular=Rotation.from_matrix(self.rotation@frame.T).as_rotvec()
            if np.linalg.norm(position)<.0008 and np.linalg.norm(angular)<.025: break
            jpos=np.zeros((3,7));jrot=None
            for g,point in zip(self.geoms,self.points()):
                body=int(self.model.geom_bodyid[g])
                mujoco.mj_jac(self.model,self.kin,self.jp,self.jr,point,body)
                jpos+=.5*self.jp[:,self.spec.dof]
                if jrot is None: jrot=self.jr[:,self.spec.dof].copy()
            jac=np.vstack([jpos,.06*jrot]);error=np.r_[position,.06*angular]
            delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*.00004,error)
            q=np.clip(q+np.clip(delta,-.10,.10),self.spec.q_range[:,0]+1e-7,self.spec.q_range[:,1]-1e-7)
        self.kin.qpos[self.spec.qpos]=q;mujoco.mj_forward(self.model,self.kin)
        return q,float(np.linalg.norm(goal-self.midpoint()))
