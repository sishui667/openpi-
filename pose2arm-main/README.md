# pose2arm

> Local A/B integration (2026-10-16): the current MuJoCo path uses offline IK/trajectory optimization plus human gripper signals, not a verified PPO deployment. `member_b_export.py` exports a single original/mirror clip at the receiving control rate without time stretching. See [integration status and limitations](../openarm_cloth/docs/member_AB_integration_20261016.md). The B sliding-gripper scene remains unchanged; rotary finger radians are not copied into it.

Real-time human arm motion retargeting onto a dual [OpenArm v2.0](https://github.com/OpenRobotLab/openarm) robot,
driven by a PPO policy trained in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

A webcam (or pre-recorded video) feeds into MediaPipe for skeleton extraction. The resulting
arm geometry is passed as an 18-D observation to a learned policy that outputs joint targets
for both arms simultaneously in Isaac Sim.

![Demo](docs/demo.gif)

---

## Pipeline

```
video / webcam
  → MediaPipe PoseLandmarker (LIVE_STREAM, GPU delegate)
  → torso-local shoulder / elbow / wrist geometry
  → One Euro smoothing
  → 18-D target obs  [upper_dir(3) | forearm_dir(3) | arm_plane(3)] × 2 arms
  → PPO policy  (92-D obs → 14-D joint action)
  → Isaac Sim dual-arm joint position targets
```

Full observation (92-D):

```
target_t (18) | target_t+1 (18) | sin(q) (14) | cos(q) (14) | q_vel/20.944 (14) | prev_action (14)
```

---

## Setup

### 1. Pose pipeline (`venv/`)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p models
curl -s -o models/pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
```

### 2. Isaac Lab (`isaac_venv/`)

Install [Isaac Sim 4.5 + Isaac Lab 0.54](https://isaac-sim.github.io/IsaacLab/) into `~/isaac_venv`.
Then install the RL extras:

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -r requirements-rl.txt
```

Convert the robot URDF to USD once (required before first training run):

```bash
source /opt/ros/humble/setup.bash
source ~/isaac_venv/bin/activate
python3 scripts/convert_urdf_local.py \
  --urdf gazebo_openarm/urdf/openarm_dual_arm_gazebo.urdf.xacro \
  --output rl/assets/openarm_bimanual.usd
```

---

## Dataset & Training

### 1. Extract poses from video

```bash
source venv/bin/activate
python3 scripts/generate_ik_dataset.py --input videos/sample01.mp4
```

Outputs `data/rl_poses.npz` with `target_obs (N, 18)` and `joint_angles (N, 14)`.
Includes left-right mirror augmentation (2× dataset size).

### 2. BC pre-training (optional warm start)

```bash
source ~/isaac_venv/bin/activate
python3 scripts/bc_pretrain.py --epochs 200
# → logs/bc_pretrained.pt
```

### 3. RL training

```bash
source ~/isaac_venv/bin/activate
python3 train_retarget.py --headless --num_envs 256
# Resume from checkpoint:
python3 train_retarget.py --headless --checkpoint logs/bc_pretrained.pt
```

Checkpoints are saved under `logs/skrl/openarm_retarget/*/checkpoints/`.
`best_agent.pt` tracks the highest mean episode reward (~127 total reward on RTX 5060 Ti).

---

## Demo

```bash
source ~/isaac_venv/bin/activate

python3 demo_isaac_gui.py                              # default: slow-motion sample video
python3 demo_isaac_gui.py --source videos/sample.mp4  # full-speed
python3 demo_isaac_gui.py --source 0                   # webcam
python3 demo_isaac_gui.py --checkpoint path/to/agent.pt
```

Two Isaac Sim windows open: the main 3D viewport (robot) and a "Reference Video" panel
showing the source video with skeleton overlay and colored arm-frame axes
(red = upper arm direction, green = forearm, blue = arm plane normal).

The demo auto-detects the latest checkpoint under `logs/skrl/openarm_retarget/`.

---

## Architecture

### Coordinate convention

`retarget.compute_body_frame()` defines a torso-local frame from shoulders and hips:

- `+body_x`: left shoulder → right shoulder
- `+body_y`: hip midpoint → shoulder midpoint
- `+body_z`: `cross(body_x, body_y)` (chest forward)

Body → robot mapping (same for both arms; symmetry is in the URDF joint axes):

```python
OPENARM_BODY_TO_ROBOT = [[ 0,  0, -1],
                          [-1,  0,  0],
                          [ 0,  1,  0]]
# human forward → robot +X
# human right   → robot -Y
# human up      → robot +Z
```

### OpenArm model

`gazebo_openarm/urdf/openarm_dual_arm_gazebo.urdf.xacro`:

- OpenArm v2.0 joint origins, axes, limits, velocities, inertials, and meshes.
- Bimanual base spacing: left `y = +0.031 m`, right `y = −0.031 m`.
- Positive collision mesh scales (DART rejects negative scales from upstream URDF).

`rl/assets/openarm_bimanual.usd` is the Isaac Sim USD derived from the above URDF.

### Reset state initialization (RSI)

Each episode resets the robot to the IK solution for that episode's start frame,
rather than a fixed home pose. This eliminates the large first-step rate penalty
that would otherwise discourage the policy from moving away from the home pose.

---

## Smoke Tests

```bash
source venv/bin/activate

# Syntax check
python3 -m py_compile extract_pose.py ik_solver.py openarm_common.py \
        openarm_live.py pose_constants.py retarget.py smoothing.py scripts/*.py

# Unit tests
python3 smoothing.py
python3 scripts/smoke_test_retarget_geometry.py
python3 scripts/smoke_test_openarm_ik.py
```

---

## Known Limitations

- Policy trained on `sample01.mp4` only; generalisation to other motions or body
  proportions is untested.
- MediaPipe `pose_world` landmarks assume GHUM body proportions; per-person
  accuracy is untested without camera calibration.
- Arm-plane singularity near fully extended elbows: the IK dataset and policy
  are trained on the sample video, which avoids extreme extensions.
- No explicit inter-arm collision cost in the RL reward; the policy relies on the
  physics simulation to avoid self-collision passively.

---

## Credits

- Training and demo video (`videos/sample.mp4`) sourced from [Pexels](https://www.pexels.com) under the [Pexels License](https://www.pexels.com/license/) (free for personal and commercial use, no attribution required).
