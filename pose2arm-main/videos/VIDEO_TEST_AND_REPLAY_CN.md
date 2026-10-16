# Pose2Arm 更换视频、生成数据与查看回放

以下命令均在 PowerShell 中执行，工作目录为：

```powershell
cd D:\openpi\pose2arm-main
```

项目使用的 Python 环境：

```powershell
$python = 'D:\ProgramData\Anaconda3\envs\openpi\python.exe'
```

## 1. 更换输入视频路径

不需要修改 Python 源码，只要替换 `--input` 后的视频路径。建议每个视频使用独立的 `.npz` 输出，避免覆盖旧结果。

本次 `p2a_test1.mp4` 的快速测试命令：

```powershell
& $python scripts\generate_ik_dataset.py `
  --input videos\p2a_test1.mp4 `
  --output data\p2a_test1_coupled.npz `
  --restarts 2 `
  --restarts-14d 1 `
  --max-nfev 300 `
  --max-nfev-14d 500 `
  --collision-margin 0.005 `
  --collision-policy project
```

生成脚本现在使用左右臂共享的人体缩放比例，并保留人体双腕在躯干坐标系中的
相对位置；接近机械臂最大臂展时才进行连续的工作空间压缩。之后默认运行
“滑动窗口局部轨迹优化”，再运行硬安全投影兜底。
前者尽量让机械臂在碰撞片段附近平滑绕开，后者保证最终输出不穿模。局部优化是
随机采样算法；默认随机种子固定，因此相同输入和参数可复现。

换成其他视频时，只需同时修改这两项：

```powershell
& $python scripts\generate_ik_dataset.py `
  --input videos\你的视频.mp4 `
  --output data\你的视频.npz `
  --restarts 2 `
  --restarts-14d 1 `
  --max-nfev 300 `
  --max-nfev-14d 500
```

如果更看重 IK 质量而不介意等待较长时间，省略快速参数，使用默认的 20 次单臂重启和 5 次双臂重启：

```powershell
& $python scripts\generate_ik_dataset.py `
  --input videos\你的视频.mp4 `
  --output data\你的视频.npz
```

## 2. 查看原始动作回放

下面会同时打开 MuJoCo 机器人窗口和原视频对比窗口，并循环播放：

```powershell
& $python scripts\replay_mujoco.py `
  --npz data\p2a_test1_reference_locked.npz `
  --video videos\p2a_test1.mp4 `
  --clip orig
```

- 在视频窗口按 `q` 退出。
- 关闭 MuJoCo 窗口也会结束回放。
- `--speed 0.5` 为半速，`--speed 2` 为两倍速。
- `--stride 2` 表示每隔一帧播放一次。
- `--bg light`、`--bg dark`、`--bg gray` 可更换 MuJoCo 背景。

示例：半速回放并使用浅色背景：

```powershell
& $python scripts\replay_mujoco.py `
  --npz data\p2a_test1_reference_locked.npz `
  --video videos\p2a_test1.mp4 `
  --clip orig `
  --speed 0.5 `
  --bg light
```

## 3. 查看镜像片段或全部数据

镜像片段：

```powershell
& $python scripts\replay_mujoco.py --npz data\p2a_test1_reference_locked.npz --video videos\p2a_test1.mp4 --clip mirror
```

依次播放原始片段和镜像片段：

```powershell
& $python scripts\replay_mujoco.py --npz data\p2a_test1_reference_locked.npz --video videos\p2a_test1.mp4 --clip all
```

## 4. 可选检查

生成后检查双臂碰撞：

```powershell
& $python scripts\diagnose_interarm_collision.py --npz data\p2a_test1_reference_locked.npz --margin 0.005
```

第一阶段新版数据同时保存 `gripper_position`（0=全开，1=全闭）和兼容旧工具的
`gripper_cmd`（二值）通道。新版回放优先使用连续通道，默认夹爪斜坡为 0.15 秒。

本次测试结果：视频共 612 帧，612 帧均检测成功；生成数据包含 1224 帧（原始 612 + 镜像 612），所有数值有限。渲染网格检测为 0 个穿模帧。

`--collision-margin 0.005` 表示左右臂碰撞几何之间保留 5 mm 硬安全余量。
`--collision-policy project` 会沿“上一安全姿态 → 当前目标姿态”扫描；发现碰撞后，
通过二分搜索回退到最近安全位置。若不能安全前进，则保持上一姿态。

若已有第一阶段 NPZ，可直接重新应用包含真实指尖开合的安全层，无需重跑 IK：

```powershell
& $python scripts\apply_collision_safety.py `
  --input data\p2a_test1_stage1.npz `
  --output data\p2a_test1_fingertip_safe.npz `
  --margin 0.005 `
  --policy project
```

推荐使用下面的新后处理命令。它先对碰撞窗口做平滑绕障，再执行硬安全兜底：

```powershell
& $python scripts\plan_local_avoidance.py `
  --input data\p2a_test1_stage1.npz `
  --output data\p2a_test1_local_avoidance.npz `
  --margin 0.005 `
  --padding 10 `
  --population 24 `
  --iterations 10
```

若只想使用旧的硬投影，可以在完整视频生成命令中加入：

```powershell
--disable-local-avoidance
```

本次平滑版后处理实测：原始片段直接碰撞姿态由 130 帧降到 50 帧，镜像片段由
130 帧降到 45 帧；硬兜底的保持帧由旧方案的 224/225 帧降到 66/113 帧，
随后检查最终 5 mm 安全余量违规为 0。

加入共享尺度、双腕相对位置和工作空间软压缩后，原始/镜像片段的碰撞候选从
130/130 帧降为 89/89 帧，局部优化后为 30/32 帧，最终硬兜底保持帧从
66/113 帧进一步降为 40/45 帧。最终实体接触和 5 mm 安全余量违规均为 0。

对双腕靠近时的连续保持窗口运行安全直连/RRT-Connect 后，原始和镜像片段的保持帧
从 40/45 进一步降为 0/0；其中 9 个窗口可安全直连，1 个窗口使用 RRT 绕行。
完整方法与参数见 `RRT_CONNECT_EXCHANGE_CN.md`。
