# pose2arm

> 2026-10-16 A/B 工程合并：当前 MuJoCo 路径是离线人体识别/IK/轨迹优化与夹爪信号，不代表以下原项目 PPO 训练路径已在本机验证。新增 `member_b_export.py` 可将原片或镜像片段按原视频时间导出为 B 的 20 Hz 输入；详见 [A/B 合并说明](../openarm_cloth/docs/member_AB_integration_20261016.md)。当前 B 场景保留其优化后的滑动夹爪，不复制本目录旧旋转夹爪的关节值。

实时人体手臂动作重定向到双臂 [OpenArm v2.0](https://github.com/OpenRobotLab/openarm) 机器人，由在 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 中训练的 PPO 策略驱动。

网络摄像头（或预录视频）接入 MediaPipe 进行骨架提取，得到的臂部几何信息作为 18 维观测输入学习策略，在 Isaac Sim 中同时输出双臂的关节目标。

![演示](docs/demo.gif)

---

## 流程

```
视频 / 摄像头
  → MediaPipe PoseLandmarker（LIVE_STREAM，GPU 委托）
  → 躯干局部系的肩 / 肘 / 腕几何
  → One Euro 平滑
  → 18 维目标观测  [上臂方向(3) | 前臂方向(3) | 臂平面(3)] × 2 臂
  → PPO 策略  (92 维观测 → 14 维关节动作)
  → Isaac Sim 双臂关节位置目标
```

完整观测（92 维）：

```
target_t (18) | target_t+1 (18) | sin(q) (14) | cos(q) (14) | q_vel/20.944 (14) | prev_action (14)
```

---

## 环境配置

### 1. 姿态流程（`venv/`）

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p models
curl -s -o models/pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
```

### 2. Isaac Lab（`isaac_venv/`）

将 [Isaac Sim 4.5 + Isaac Lab 0.54](https://isaac-sim.github.io/IsaacLab/) 安装到 `~/isaac_venv`，然后安装 RL 扩展包：

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -r requirements-rl.txt
```

首次训练前需将机器人 URDF 转换为 USD（一次性操作）：

```bash
source /opt/ros/humble/setup.bash
source ~/isaac_venv/bin/activate
python3 scripts/convert_urdf_local.py \
  --urdf gazebo_openarm/urdf/openarm_dual_arm_gazebo.urdf.xacro \
  --output rl/assets/openarm_bimanual.usd
```

---

## 数据集与训练

### 1. 从视频中提取姿态

```bash
source venv/bin/activate
python3 scripts/generate_ik_dataset.py --input videos/sample01.mp4
```

输出 `data/rl_poses.npz`，包含 `target_obs (N, 18)` 和 `joint_angles (N, 14)`。
包含左右镜像数据增强（数据集规模 ×2）。

### 2. BC 预训练（可选的热启动）

```bash
source ~/isaac_venv/bin/activate
python3 scripts/bc_pretrain.py --epochs 200
# → logs/bc_pretrained.pt
```

### 3. RL 训练

```bash
source ~/isaac_venv/bin/activate
python3 train_retarget.py --headless --num_envs 256
# 从检查点恢复：
python3 train_retarget.py --headless --checkpoint logs/bc_pretrained.pt
```

检查点保存在 `logs/skrl/openarm_retarget/*/checkpoints/` 下。
`best_agent.pt` 记录最高平均回合奖励（在 RTX 5060 Ti 上总奖励约 127）。

---

## 演示

```bash
source ~/isaac_venv/bin/activate

python3 demo_isaac_gui.py                              # 默认：慢速示例视频
python3 demo_isaac_gui.py --source videos/sample.mp4  # 全速
python3 demo_isaac_gui.py --source 0                   # 摄像头
python3 demo_isaac_gui.py --checkpoint path/to/agent.pt
```

会打开两个 Isaac Sim 窗口：主 3D 视口（机器人）和"参考视频"面板——显示源视频及骨架叠加和彩色臂部坐标系（红色 = 上臂方向，绿色 = 前臂，蓝色 = 臂平面法线）。

演示会自动检测 `logs/skrl/openarm_retarget/` 下最新的检查点。

---

## 架构

### 坐标约定

`retarget.compute_body_frame()` 由肩部和髋部定义躯干局部坐标系：

- `+body_x`：左肩 → 右肩
- `+body_y`：髋中点 → 肩中点
- `+body_z`：`cross(body_x, body_y)`（胸前向）

身体 → 机器人映射（双臂相同；对称性由 URDF 关节轴实现）：

```python
OPENARM_BODY_TO_ROBOT = [[ 0,  0, -1],
                          [-1,  0,  0],
                          [ 0,  1,  0]]
# 人体前方 → 机器人 +X
# 人体右方 → 机器人 -Y
# 人体上方 → 机器人 +Z
```

### OpenArm 模型

`gazebo_openarm/urdf/openarm_dual_arm_gazebo.urdf.xacro`：

- OpenArm v2.0 的关节原点、轴、限位、速度、惯量和网格。
- 双臂基座间距：左臂 `y = +0.031 m`，右臂 `y = −0.031 m`。
- 正的碰撞网格缩放（DART 拒绝上游 URDF 中的负缩放）。

`rl/assets/openarm_bimanual.usd` 是由上述 URDF 导出的 Isaac Sim USD。

### 重置状态初始化（RSI）

每个回合开始时，机器人被重置为该回合起始帧对应的 IK 解，而非固定的初始位姿。这消除了原本会阻碍策略离开初始位姿的大幅首步速率惩罚。

---

## 冒烟测试

```bash
source venv/bin/activate

# 语法检查
python3 -m py_compile extract_pose.py ik_solver.py openarm_common.py \
        openarm_live.py pose_constants.py retarget.py smoothing.py scripts/*.py

# 单元测试
python3 smoothing.py
python3 scripts/smoke_test_retarget_geometry.py
python3 scripts/smoke_test_openarm_ik.py
```

---

## 已知局限

- 策略仅在 `sample01.mp4` 上训练；对其他动作或体型比例的泛化能力未经测试。
- MediaPipe `pose_world` 关键点假设 GHUM 体型比例；未经相机标定时单人精度未验证。
- 肘部接近完全伸直时臂平面存在奇异性：IK 数据集和策略基于示例视频训练，避开了极端伸展。
- RL 奖励中没有显式的双臂间碰撞代价；策略依赖物理仿真被动避免自碰撞。

---

## 致谢

- 训练与演示视频（`videos/sample.mp4`）来自 [Pexels](https://www.pexels.com)，遵循 [Pexels 许可证](https://www.pexels.com/license/)（个人和商业使用免费，无需署名）。
