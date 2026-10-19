"""Independent left/right human pose and fist controllers for the actual robot."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from human_teleop_core import PadCartesianIK,ArmPoseIK
from human_gripper_gestures import FistGripperMapper
from member_b_actuators import ActuatorController
from realtime_bimanual_planner import RealtimeBimanualPlanner

SIDES=('left','right')


class BimanualController:
    def __init__(self,model,data,wrist_mode='twist',direction_tau=0.,arm_speed=2.5,palm_alignment='absolute'):
        self.model=model;self.data=data;self.actuators=ActuatorController(model,data)
        self.specs=self.actuators.specs
        self.pads={s:PadCartesianIK(model,data,initialize_grasp=False,side=s) for s in SIDES}
        self.targets={s:data.qpos[self.specs[s].qpos].copy() for s in SIDES}
        self.fingers={s:.044 for s in SIDES};self.closed={s:0 for s in SIDES}
        self.gestures={s:FistGripperMapper() for s in SIDES}
        self.direction_tau=float(direction_tau);self.arm_speed=float(arm_speed)
        self.palm_alignment=palm_alignment
        self.last_seq=-1;self.wrist_mode=wrist_mode;self.reset()
        self.coordinator=RealtimeBimanualPlanner(model,data,self.specs)
        self.valid={s:False for s in SIDES};self.hand_valid={s:False for s in SIDES}

    def reset(self,wrist_mode=None):
        if hasattr(self,'coordinator'): self.coordinator.reset()
        if wrist_mode is not None: self.wrist_mode=wrist_mode
        self.solvers={s:ArmPoseIK(self.pads[s],wrist_mode=self.wrist_mode,direction_tau=self.direction_tau,palm_alignment=self.palm_alignment) for s in SIDES}
        self.cache={s:None for s in SIDES};self.last_stamp={s:None for s in SIDES}

    def update(self,obs,now_ns,paused=False):
        before={s:self.targets[s].copy() for s in SIDES}
        valid=obs is not None and now_ns-obs['read_end']<=350000000
        fresh=valid and obs['seq']!=self.last_seq
        for s in SIDES:
            arm=obs['arms'][s] if valid else None
            self.valid[s]=arm is not None and arm['pose'] is not None
            self.hand_valid[s]=arm is not None and arm['hand_valid']
            if paused: continue
            if not self.valid[s]: self.cache[s]=None
            elif fresh:
                dt=.05 if self.last_stamp[s] is None else float(np.clip((obs['read_end']-self.last_stamp[s])/1e9,.01,.15))
                self.solvers[s].update_target(arm['pose'],self.targets[s],dt)
                self.cache[s]=self.solvers[s].solve(self.targets[s],iterations=6)
                self.last_stamp[s]=obs['read_end']
            if self.valid[s] and self.cache[s] is not None:
                max_step=.05*np.array([self.arm_speed]*4+[4.4]*3)
                self.targets[s]+=np.clip(self.cache[s]-self.targets[s],-max_step,max_step)
            if fresh or not self.hand_valid[s]:
                features=arm['fist'] if self.hand_valid[s] else None
                self.closed[s]=self.gestures[s].update(features,obs['seq'] if valid else -1,now_ns/1e9)
            self.fingers[s]=.044*(1-self.closed[s])
        if valid: self.last_seq=obs['seq']
        goals={s:(self.cache[s].copy() if self.valid[s] and self.cache[s] is not None else before[s].copy()) for s in SIDES}
        wrist_speeds=[4.4,0.,0.] if self.wrist_mode=='twist' else [4.4]*3
        steps=np.tile(np.array([self.arm_speed]*4+wrist_speeds)*.05,2)
        for i,s in enumerate(SIDES):
            if not self.valid[s]: steps[7*i:7*(i+1)]=0
        self.targets=self.coordinator.coordinate(before,goals,self.data,steps,self.fingers,
                                                enabled=not paused and any(self.valid.values()))
        if not paused and not any(self.valid.values()): self.coordinator.status['mode']='input_missing'
        return self.actuators.set_targets(self.targets,self.fingers)

    def close(self): self.coordinator.close()

    def errors(self):
        errors=np.full((2,3),np.nan)
        for i,s in enumerate(SIDES):
            solver=self.solvers[s];pad=self.pads[s]
            if solver.upper is None: continue
            pad.kin.qpos[pad.spec.qpos]=self.data.qpos[pad.spec.qpos]
            mujoco.mj_forward(self.model,pad.kin)
            units=solver.geometry()
            errors[i,:2]=[float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1)))) for a,b in zip(units,(solver.upper,solver.fore))]
            errors[i,2]=float(np.degrees(Rotation.from_matrix(solver.rotation @ pad.kin.xmat[pad.spec.body_id].reshape(3,3).T).magnitude()))
        return errors
