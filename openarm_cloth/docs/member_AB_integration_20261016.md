# 成员 A/B 合并与完善说明（2026-10-16）

## 当前结论

已将成员 A 的 `dazhaxie-666/openpi-` 提交 `c19e08c13770a40afb071ec1ecb3a12f070f33d7` 合并到成员 B 仓库，并完善 A 的输入处理与 A→B 接口。保留上游历史，不直接修改 A 的远程仓库。

**你的优化场景没有被 A 的旧模型替换。** 合并验收逐项比较物理参数、桌面、相机、执行器和控制器；均与此前 B 冻结场景一致。桌面仍是 24.6×44.6 cm、顶面 40 cm，单臂执行、真实接触摩擦，不激活物体 weld。

这里完成的是**离线人体示范轨迹/夹爪信号→B 执行器**。它不是“随便挥手就能自动抓到任意位置方块”，也不是已经完成真人摄像头在线推理或 15 秒搬运释放。

## 本次具体完善

### A 侧

1. 修复 `GripperMapper` 对 `np.float32(NaN)` 不计入丢失超时的问题；Inf、负距离也不再触发错误闭合。校验保持/丢失帧数和连续映射 FPS。
2. 新增一对一手腕关联，结合当前人体腕点与短期历史腕点，移除“检测重复就强塞到另一只手”的逻辑。无法匹配的手记为缺失；持续丢失会清除旧关联。
3. 碰撞检查拒绝 NaN/Inf 关节角、错误形状、非归一化夹爪值及 NaN 余量，避免无效几何被误判安全。
4. 碰撞模型和回放默认 XML 改为仓库相对路径，不依赖 A 开发电脑的 D 盘目录。
5. 新生成/补丁 NPZ 显式声明 `joint_order`、`gripper_order`、`gripper_semantics`。
6. 新增 `member_b_export.py`：按源帧号/FPS 保留原视频时间，单独选择原片或镜像片段，再重采样到 B 的 20 Hz。二值信号采用前值保持，连续闭合量插值；不把镜像段接成连续动作。
7. 修复 A 独立 `replay_mujoco.py --dynamic` 每视频帧只推进一个物理步的问题：用累计时钟分配物理子步，先写臂/爪命令再推进。位置回放明确拒绝力矩执行器；B 必须走专用 PD 驱动。

手腕关联目前通过合成回归测试，**未重新跑整段真实视频证明识别精度提升**；不承诺遮挡时不换手。新的导出样例来自 A 已公开的现有 NPZ，不假称已用本次关联算法重新提取。

### B 侧

- 优先使用 A 的连续 `gripper_position`，其次二值 `gripper_cmd`，再使用原始米制 `hand_distance_m`。
- A 的关节角默认 `[左7,右7]`，夹爪默认 `[右,左]`；分别解析，不能共用列序。支持显式声明两种顺序。
- A 输出闭合量 `c∈[0,1]`：0 张开、1 闭合。B 每指滑动目标为 `q=.044*(1-c)` m。不能把 A 原模型的 ±.7854 rad 旋转关节值直接用于 B。
- 缺少人体信号时默认张开；`human-only` 会拒绝缺失输入。旧的机器人—方块距离代理仅 `--gripper-mode sim-proxy` 显式启用，不再默默代替真人手势。
- 保留首帧/后续帧限速、关节限位、力矩限幅及固定左臂待机。每个 20 Hz 周期完整推进 100 个物理步。
- 新增 `demo_member_ab.py` 物理回放窗口，播放完不自动关闭；空格暂停/继续，R 从初始化状态重播，Q 或关闭窗口退出。

## 两种演示不要混淆

| 入口 | 控制来源 | 当前用途 |
|---|---|---|
| `demo_grasp_viewer.py` | 胸部/腕部 RGB-D → 视觉 IK | 保留的自主方块抓取演示，桌面参数未变 |
| `demo_member_ab.py` | A 人体轨迹和夹爪通道 → B 执行器 | 真人示范离线回放，不能当成物体视觉重定位 |
| `realtime_capture.py --source usb` | 真实 USB 摄像头 | 采集测试；尚未连接 A 在线人体推理 |

关节角回放不读取方块真值生成控制目标，但也**没有**将人体轨迹锚定到当前方块。完整的“示范末端增量 + 胸腕视觉目标 + IK”仍是下一步融合策略，而非本次已实现功能。

## 验证与证据

当前交付目录：`deliverables/member_AB_20261016/`；此前 `member_B_20261016/` 是历史版本，保留原记录，不重写旧结果。

- A 专项/混合夹爪共 11 项单元测试，A/B 与原 B/桌面共 29 项，合计 40 项。
- A 的离线平滑、姿态质量门控和参考锁定输出回归另行通过；原参考锁定样例静态碰撞检查 unsafe=0。这是 A 原旋转夹爪模型中的结果，**不能直接转移成 B 滑动夹爪/桌面场景的无碰撞保证**。
- 参考锁定与 hybrid 原片各导出 417 帧。原视频片段约 20.834 s，控制回放 20.85 s，差异小于一个 50 ms 周期。
- 两种信号分别进行完整物理执行。逐帧审计检查 A 闭合量、B 动作和实际 `ctrl` 完全对应，左臂目标保持不变，仿真时间每步增加 .05 s。
- 物理执行中的指关节实际跟踪误差与命令分别记录；不能把 ctrl 正确等同于指位置完全准确。
- 最终延迟见 `acceptance_summary.json` 和 `latency_final/`。早期两轮保留在 `latency/`，对应 `scene_manifest_initial.json`；不能删去失败结果。

目前带完整顶视单目定位的长测有超过 50 ms 的工作帧；若最终报告为 FAIL，就是延迟预算未通过。平均频率接近 20 Hz 不能替代逐帧时限验收。人体关键点提取和 IK 已离线完成，不包含在这次耗时中。

最终两轮平均约 30.82 / 30.59 ms，P95 47.75 / 49.45 ms，超预算 11/417 与 19/417，因此均为 FAIL。两种夹爪实际跟踪 RMS 约 2.15 / 3.34 mm；软关节限位下观察到瞬时超过 .044 m，记录保留，不将命令限幅当作实际位置严格限幅。源轨迹重采样后部分 R3/R4 步长仍超速，B 首帧过渡与后续限速共干预 14 帧，不能声称源示范已完全满足速度约束。

## 本机运行

```powershell
Set-Location D:\openpi\openarm_cloth
& D:\miniconda3\envs\openpi\python.exe demo_member_ab.py
```

原来的“打开抓取仿真”快捷方式仍启动胸部/腕部自主抓取，不改成 NPZ 回放。两种入口都保留。

另一台电脑克隆后，安装 `openarm_cloth/requirements_simulation.txt`，在 `openarm_cloth` 内运行：

```powershell
.\.venv\Scripts\python.exe freeze_member_b_scene.py
.\.venv\Scripts\python.exe demo_member_ab.py
.\.venv\Scripts\python.exe -m unittest -v test_member_ab_integration test_member_b_delivery test_task_workspace
.\.venv\Scripts\python.exe audit_member_ab_delivery.py
```

从新的 A 结果导出时（输出必须是新文件）：

```powershell
Set-Location ..\pose2arm-main
..\openarm_cloth\.venv\Scripts\python.exe member_b_export.py --input data/p2a_test1_rtmw_hybrid_sim.npz --output data/new_B_orig_20hz.npz --hz 20 --clip orig
```

重采样会改变帧间路径；导出程序明确标记“接收场景仍需检查碰撞”。A 源轨迹及首帧到 B 待机位的过渡仍可能需要限速，实际干预计数见报告。

## 仍未完成

真人在线推理→B 的端到端实时闭环；识别精度与跨视频泛化验收；A 示范相对末端轨迹对方块的视觉重定位；手势驱动的三次成功抓取及 15 秒搬运释放。不能用既有自主抓取 19 例或 A 的运动学回放替代这些验收。
