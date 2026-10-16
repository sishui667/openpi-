"""Cached version of the existing PD + gravity controller; same physics rate."""
import numpy as np
from teleop_towel_dataset import build_arm_specs, build_finger_actuators


class ActuatorController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.specs = build_arm_specs(model)
        self.fingers = build_finger_actuators(model)
        self.qpos = np.concatenate([self.specs[s].qpos for s in ("left", "right")])
        self.dof = np.concatenate([self.specs[s].dof for s in ("left", "right")])
        self.actuators = np.concatenate([self.specs[s].actuators for s in ("left", "right")])
        self.finger_actuators = np.concatenate([self.fingers[s] for s in ("left", "right")])
        self.kp = np.tile([80, 80, 55, 55, 25, 20, 20], 2)
        self.kd = np.tile([8, 8, 5, 5, 3, 2, 2], 2)
        self.limits = model.actuator_forcerange[self.actuators].copy()
        self.target = data.qpos[self.qpos].copy()
        self.finger_target = np.full(4, 0.044)
        self.action = np.zeros(18)

    def set_targets(self, joints, fingers):
        self.target[:] = np.concatenate([joints[s] for s in ("left", "right")])
        self.finger_target[:] = np.repeat([np.clip(fingers[s], 0, 0.044) for s in ("left", "right")], 2)
        for side, start in (("left", 0), ("right", 9)):
            self.action[start:start + 7] = joints[side]
            self.action[start + 7:start + 9] = np.clip(fingers[side], 0, 0.044)
        return self.action

    def apply(self):
        data = self.data
        torque = (data.qfrc_bias[self.dof]
                  + self.kp * (self.target - data.qpos[self.qpos])
                  - self.kd * data.qvel[self.dof])
        data.ctrl[self.actuators] = np.clip(torque, self.limits[:, 0], self.limits[:, 1])
        data.ctrl[self.finger_actuators] = self.finger_target
