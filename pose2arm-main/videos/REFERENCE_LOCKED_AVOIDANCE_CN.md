# Pose2Arm 参考锁定避障与回放

当前稳定输出：`data\p2a_test1_reference_locked.npz`

## 改动内容

新版处理链路不再直接执行 RRT 关节路径：

1. MediaPipe 肩、肘、腕按左右臂分别做 visibility 门控；低可信度帧保持该侧最近可信姿态。
2. 离线关节轨迹使用 9 帧二阶 Savitzky-Golay 零相位整形，降低 jerk，不引入播放时延。
3. 滑动窗口优化和硬安全投影产生初始安全轨迹。
4. RRT-Connect 只负责为双手换位提供无碰撞拓扑种子。
5. 控制点轨迹优化重新约束腕部位置、上臂/前臂方向、臂平面、参考关节、边界速度、加速度和 jerk。
6. 只有通过动作保真预算和 MuJoCo 连续边碰撞检查的窗口才会写入输出。

该流程适用于离线视频。第 2 步是零相位滤波，不能直接用于实时控制；实时版本应改用因果限 jerk 控制器或短时域 MPC。

## 从视频完整生成

在 `D:\openpi\pose2arm-main` 中运行：

```powershell
$python = "D:\ProgramData\Anaconda3\envs\openpi\python.exe"

& $python scripts\generate_ik_dataset.py `
  --input videos\p2a_test1.mp4 `
  --output data\p2a_test1_reference_base.npz `
  --min-arm-visibility 0.55

& $python scripts\apply_exchange_rrt.py `
  --input data\p2a_test1_reference_base.npz `
  --output data\p2a_test1_reference_locked.npz `
  --margin 0.005 `
  --padding 8 `
  --proximity 0.40 `
  --max-iterations 1800 `
  --trajopt-iterations 8
```

新版生成器已经内置零相位关节整形。`rebuild_reference_trajectory.py` 只用于迁移已有的、尚未整形的 NPZ，不是新视频的必需步骤。

## 查看原视频与机械臂同步回放

```powershell
$python = "D:\ProgramData\Anaconda3\envs\openpi\python.exe"

& $python scripts\replay_mujoco.py `
  --npz data\p2a_test1_reference_locked.npz `
  --video videos\p2a_test1.mp4 `
  --clip orig `
  --gripper-side both
```

半速查看动作细节：

```powershell
& $python scripts\replay_mujoco.py `
  --npz data\p2a_test1_reference_locked.npz `
  --video videos\p2a_test1.mp4 `
  --clip orig `
  --gripper-side both `
  --speed 0.5
```

## 验证

```powershell
& $python scripts\diagnose_interarm_collision.py `
  --npz data\p2a_test1_reference_locked.npz `
  --margin 0.005

& $python scripts\smoke_test_reference_locked_output.py
```

本次 `p2a_test1` 验证结果：碰撞 mesh 接触 0 帧、渲染 mesh 接触 0 帧、5 mm 安全余量违规 0 帧。相对旧 `p2a_test1_exchange_rrt.npz`，明显失真帧减少约 26%，jerk p95 降低约 56%–59%。
