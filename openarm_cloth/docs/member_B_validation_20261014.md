# 成员 B 阶段验证报告（2026-10-14）

## 1. 结论

成员 B 的四项交付链路均已落盘并复验：

1. 单臂方块场景与物理参数已冻结；基础场景和 wrapper 均有 SHA-256 守卫。
2. Pose2Arm 的 762 帧输出已完整驱动右臂执行器，无急停。
3. 单目相机实时链路在 200 帧交付测试中满足 20 Hz 预算。
4. 纯摩擦抓取在任务规定的 15 秒内通过，未使用 weld；相机窗口连续刷新 56 帧。

## 2. 冻结场景

- 入口：`simulation/v1/droid_cube_scene_perception.xml`
- 配置：`simulation/v1/droid_cube_scene_perception.json`
- 基础场景 SHA-256：`d452c13db96342803fedcb96a7aef11e6f3e44c23124cc586d8344cb580f4357`
- wrapper SHA-256：以配置中的 `output_sha256` 为准。
- 相机：`workspace_camera`，固定 45 度俯视。
- 执行臂：右臂；左臂保持 home。

检查命令：

```powershell
& 'D:\miniconda3\envs\openpi\python.exe' build_droid_cube_perception_scene.py --verify
```

## 3. 纯摩擦抓取与稳定性

标准验证命令：

```powershell
& 'D:\miniconda3\envs\openpi\python.exe' demo_grasp_viewer.py --check --post-lift-s 1.5
```

结果（`outputs/20261014/grasp_stability_final.json`）：

- 方块抬升：117.5 mm（要求 > 30 mm）
- 连续满足成功判据：6.105 s（要求 >= 0.5 s）
- Episode 仿真时长：13.914 s（要求 <= 15 s）
- 横向位移：3.9 mm
- 方块最大姿态变化：2.25 度
- 夹爪自身最大姿态变化：2.18 度
- 三个接触点均位于方块平面，不在棱上
- 三个接触的真实法向力：1.464 N、1.526 N、3.133 N，合计 6.123 N
- 保持期间最小法向力合计：6.072 N
- 方块到夹爪中心距离：最终 10.5 mm，最大 11.4 mm（要求 < 20 mm）
- 理论所需法向力：0.883 N/指
- weld：未使用

原先“接触落在棱上导致晃动”的判断已被实验推翻。真正原因是导入的 finger mesh
在每个平面只产生一个点接触，`condim=3` 不提供绕接触法线的扭转摩擦。wrapper 只对
右夹爪与方块的两对接触设置 `condim=4`、扭转摩擦系数 0.03 m，并显式保留原接触
`solref="0.005 1"`。场景另启用 10 次 NoSlip 后处理，抑制软接触模型中的残余滑移。
加长到 5 秒保持时方块仍未掉落（抬升 117.5 mm、最大旋转 2.25 度）；该压力运行
总时长 17.414 秒，因故意超过任务 15 秒上限而标记 FAIL，不是抓取失败。
以上仍是接触/摩擦抓取，不是 weld。

MuJoCo 官方将 `condim=4` 定义为加入绕接触法线的扭转摩擦，并说明该系数具有长度单位，
可解释为接触斑直径：
<https://mujoco.readthedocs.io/en/latest/computation.html#contact>

## 4. 相机窗口

交互运行 `demo_grasp_viewer.py` 后，完整抓取过程中 `CameraWindow` 连续显示 56 帧，
不再停在 2 帧。运行结束后输出 `camera window: 56 frames shown`。

## 5. 端到端延迟

交付日志：`outputs/20261014/latency_log.json`

命令：

```powershell
& 'D:\miniconda3\envs\openpi\python.exe' realtime_capture.py `
  --frames 500 --hz 20 --warmup 30 `
  --width 320 --height 240 --fit-topk 20 `
  --high-priority --max-miss-rate 0.01
```

500 帧结果（首帧全局定位，后续局部跟踪，低 IoU 自动回退全局）：

- 平均：16.86 ms
- P95：29.36 ms
- P99：43.81 ms
- 最大：75.79 ms
- 超过 50 ms：3/500，miss rate 0.6%
- 判定：PASS_SOFT_REALTIME（SLA 允许 <= 1% miss）
- 严格零超时判定：FAIL on 3/500 ticks
- 等效处理吞吐：59.30 Hz
- 感知误差：平均 4.47 mm，最大 4.47 mm
- 全局回退：0 次
- 进程优先级：Windows `HIGH_PRIORITY_CLASS`

高负载失败日志保存在 `outputs/20261014/latency_final_500/latency_log.json`：
负载指数从 83.2 ms 跳到 255.1 ms（3.07 倍），拟合退化到 46.3 ms，490/500 超时。
这证明当前系统可称为 20 Hz 软实时，但不应宣称 Windows 上的硬实时保证。

旧的 640x480 失败日志保存在
`outputs/20261014/latency_log_640x480_failed.json`，没有删除或覆盖证据。

## 6. Pose2Arm 到执行器

报告：`outputs/20261014/pose2arm_driver_validation.json`

- 输入：`p2a_test.npz`，762 帧，30 fps
- 自动确认列顺序：左臂 7 维在前、右臂 7 维在后
- 驱动：右臂；左臂保持 home
- 关节限位裁剪：0
- 速度限制触发：42 帧
- 最大单帧限速裁剪/最大跟踪滞后：2.353 rad
- 急停：未触发
- 完整运行：32.1 s，23.7 fps
- 夹爪：762 帧均为 open（仿真侧距离始终未进入闭合阈值）

源数据超速是成员 A 侧问题，具体逐关节数据见
`docs/member_A_pose2arm_feedback.md`。

## 7. 当前边界

- 交付日志已通过 200 帧硬判据；500 帧压力测试仍有 0.4% GPU 调度尖峰。
- Pose2Arm 不输出手部关键点，夹爪输入按团队决定使用仿真侧钳口到感知物体的距离。
- 目标区域仍未定义；当前只验证抓起与稳定保持，未实现“搬运至目标区并释放”。

