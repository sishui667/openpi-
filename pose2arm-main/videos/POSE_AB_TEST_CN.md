# 人体关键点 A/B 测试与回放

本测试在同一视频上对比原项目 MediaPipe 与 RTMW3D 的二维关键点可视化。它是**识别诊断**，不改变当前机械臂的关节输出、避碰算法或坐标映射。RTMW3D 的三维输出为模型坐标，不能直接当作米制机械臂目标。

## 当前视频

在 PowerShell 中进入项目目录：

```powershell
Set-Location 'D:\openpi\pose2arm-main'
```

本机已建好独立环境 `.venv-pose3d`。迁移到其他 Windows 机器时，可用 Python 3.11 创建环境并安装依赖（MediaPipe 仍使用原项目环境）：

```powershell
& 'D:\ProgramData\Anaconda3\envs\python-course-py3112\python.exe' -m venv .venv-pose3d
& '.\.venv-pose3d\Scripts\python.exe' -m pip install rtmlib==0.0.16 onnxruntime-directml==1.23.0 opencv-python numpy
```

本次输入为 `videos\p2a_test1.mp4`，输出在 `out\pose_ab\`：

```powershell
& 'D:\ProgramData\Anaconda3\envs\openpi\python.exe' scripts\pose_ab_extract.py --backend mediapipe --input videos\p2a_test1.mp4 --output out\pose_ab\mediapipe.npz
& '.\.venv-pose3d\Scripts\python.exe' scripts\pose_ab_extract.py --backend rtmw3d --device dml --input videos\p2a_test1.mp4 --output out\pose_ab\rtmw3d.npz
& '.\.venv-pose3d\Scripts\python.exe' scripts\pose_ab_compare.py --video videos\p2a_test1.mp4 --mediapipe out\pose_ab\mediapipe.npz --rtmw3d out\pose_ab\rtmw3d.npz --output-dir out\pose_ab
```

查看并排回放（左 MediaPipe，右 RTMW3D）：

PowerShell：

```powershell
& '.\.venv-pose3d\Scripts\python.exe' scripts\play_video.py out\pose_ab\p2a_test1_pose_ab.mp4
```

如果使用的是 CMD（提示符形如 `D:\...>`，而不是 `PS D:\...>`），使用双引号且不要前面的 `&`：

```bat
"D:\openpi\pose2arm-main\.venv-pose3d\Scripts\python.exe" "D:\openpi\pose2arm-main\scripts\play_video.py" "D:\openpi\pose2arm-main\out\pose_ab\p2a_test1_pose_ab.mp4"
```

查看数值报告：`out\pose_ab\p2a_test1_pose_ab_report.md` 和 `out\pose_ab\p2a_test1_pose_ab_metrics.csv`。

## 更换视频

把三条命令的 `--input` / `--video` 换成新视频路径，并把 `--output` 改成不同的 NPZ 文件名，避免新旧结果混用。比较脚本会根据视频文件名生成对应的回放及报告。例如：

```powershell
& 'D:\ProgramData\Anaconda3\envs\openpi\python.exe' scripts\pose_ab_extract.py --backend mediapipe --input videos\new_take.mp4 --output out\pose_ab\new_take_mediapipe.npz
& '.\.venv-pose3d\Scripts\python.exe' scripts\pose_ab_extract.py --backend rtmw3d --device dml --input videos\new_take.mp4 --output out\pose_ab\new_take_rtmw3d.npz
& '.\.venv-pose3d\Scripts\python.exe' scripts\pose_ab_compare.py --video videos\new_take.mp4 --mediapipe out\pose_ab\new_take_mediapipe.npz --rtmw3d out\pose_ab\new_take_rtmw3d.npz --output-dir out\pose_ab
& '.\.venv-pose3d\Scripts\python.exe' scripts\play_video.py out\pose_ab\new_take_pose_ab.mp4
```

短片段试运行可在两条提取命令后加 `--max-frames 30`。两个结果必须来自**同一视频、同一帧序列**。首次运行 RTMW3D 会下载 ONNX 模型到 `models\rtmlib\`，需联网；后续复用本地模型。此机使用 Windows DirectML GPU（`--device dml`）；若设备不支持，改为 `--device cpu`，但全片耗时会明显增加。

## 如何解释结果

- `complete_pct`：手/臂所需关节同时可见的帧比例。
- `max_missing_run`：最长连续丢失帧数。越短通常越利于稳定跟踪。
- `wrist_jump_p95_px`：相邻可见帧腕部二维位移的第 95 百分位。大跳变值得回看，但真实快速动作也会提高此值。
- `forearm_2d_cv`：二维前臂长度波动；会受透视、遮挡与人手朝向影响。
- `pinch_ratio`：拇指尖至食指尖距离除以掌宽，可观察开合趋势，不能直接当作夹爪控制量。

本测试没有人工标注真值，因此这些指标不能单独证明 RTMW3D 比 MediaPipe 更准确。两模型的置信度不在同一标尺；MediaPipe 手部没有逐点分数。应重点查看双手交叉、手指开合和遮挡片段的并排回放。确认提升后，下一阶段才将新检测结果接入坐标归一化与机械臂重定向，并做独立安全验证。

## RTMW3D 接入 MuJoCo 仿真

当前采用安全的混合方案：双臂继续使用已验证的米制 MediaPipe 重定向轨迹，RTMW3D 的 133 点结果只用于夹爪连续开合。接入器会拒绝异常跳变、做时间滤波和变化率限制，并逐帧回溯 RTMW3D 的影响强度，直到在固定双臂轨迹下通过含夹爪的扫掠碰撞检查。RTMW3D 的裁剪空间三维值不直接输入 IK。

重新生成混合仿真数据：

```powershell
& 'D:\ProgramData\Anaconda3\envs\openpi\python.exe' scripts\build_rtmw_hybrid_sim.py
& 'D:\ProgramData\Anaconda3\envs\openpi\python.exe' scripts\smoke_test_rtmw_hybrid_sim.py
```

PowerShell 中启动 MuJoCo 原片同步回放：

```powershell
& 'D:\ProgramData\Anaconda3\envs\openpi\python.exe' scripts\replay_mujoco.py --npz data\p2a_test1_rtmw_hybrid_sim.npz --video videos\p2a_test1.mp4 --clip orig --gripper auto --gripper-side both
```

CMD 中启动：

```bat
"D:\ProgramData\Anaconda3\envs\openpi\python.exe" "D:\openpi\pose2arm-main\scripts\replay_mujoco.py" --npz "D:\openpi\pose2arm-main\data\p2a_test1_rtmw_hybrid_sim.npz" --video "D:\openpi\pose2arm-main\videos\p2a_test1.mp4" --clip orig --gripper auto --gripper-side both
```

输出报告在 `out\pose_ab\rtmw_hybrid_sim_report.md`。当前验证保证的是离线 MuJoCo 模型中的轨迹后置条件，不等同于实体机械臂的绝对安全保证；实体部署还需要控制周期、急停、速度/力矩限制和现场安全区验证。
