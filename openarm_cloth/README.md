# OpenArm 双相机视觉抓取仿真（成员 B）

最新 A/B 合并版：见 [合并说明与运行入口](docs/member_AB_integration_20261016.md)。A 的轨迹/夹爪输出已接入；优化后的胸部/腕部自主抓取场景保持不变。`demo_member_ab.py` 是离线人体示范物理回放，`demo_grasp_viewer.py` 仍是自主抓取，两者不混称。当前冻结清单默认在 `deliverables/member_AB_20261016/`，原 B 交付目录作为历史证据保留。

MuJoCo 物理仿真：胸部 RGB-D 粗定位、腕部 RGB-D 精定位与方块朝向估计、多候选相对面抓取 IK、路径碰撞排序、接近后视觉校正和纯摩擦夹持。支持交互设置方块位置与旋转角度。

这是成员 B 的当前项目快照。仓库根目录中原有的 `pose2arm-main/`、`Joint-action-n-pose-estimation-main/` 和 `openarm_cloth_with_dependencies_20260929/` 保留不变。

## 成员 B 任务书交付（2026-10-16）

场景冻结、NPZ→右臂执行器接口、仿真单目控制延迟、真实 USB 单目采集脚本及逐帧日志集中在 `deliverables/member_B_20261016/README.md`，接口见 `docs/member_B_interface_20261016.md`。入口为 `realtime_capture.py`；历史实现保留为 `legacy_realtime_capture_20261014.py`，不用于当前验收。

GitHub 同步版本、两种运行模式与可点击证据索引见 [版本说明](docs/github_snapshot_20261016.md)。

注意：A 当前只有离线 NPZ，尚不能宣称真人摄像头→Pose2Arm 在线推理→机械臂的完整实时闭环。以下胸部/腕部 RGB-D 自主抓取是扩展验证，不替代 A→B 动作接入。

## 安装

建议使用 Python 3.12，模型与相机渲染需要可用的 OpenGL 图形驱动。

```powershell
cd openarm_cloth
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_simulation.txt
```

已有 `openpi` Conda 环境时，也可激活后安装这些依赖。本机验证解释器为 `D:\miniconda3\envs\openpi\python.exe`；其他电脑使用自己的解释器路径。

## 打开仿真

```powershell
.\.venv\Scripts\python.exe demo_grasp_viewer.py --interactive-place-s 86400 --preview-initial-s 5 --post-lift-s 1.5
```

也可以在 PowerShell 中执行 `./start_simulation.ps1`，或使用自己的 Python 解释器运行 `launch_grasp_simulation.pyw`。图形环境下才会显示 MuJoCo 窗口。

| 操作 | 功能 |
|---|---|
| W/S/A/D 或方向键 | 移动方块，每次 2 cm |
| Q/E | 旋转方块 |
| 双击方块，Ctrl + 鼠标右键拖动 | 鼠标调整初始状态 |
| 空格 | 接受摆位，开始摄像头定位、规划与抓取 |
| 关闭窗口 | 退出程序 |

方块初始坐标仅用于设置测试环境。运行规划使用摄像头估计的方块位姿；仿真真实物体位置、接触点只用于评估记录。模型不通过 weld 固定方块。

当前演示使用 **24.6×44.6 cm 的实体桌面，桌面高度 40 cm**。黄色内框为方块中心的初始摆位区，16×36 cm；相比旧的 18×26 cm 区域面积增加约 23%，并非每个方向都扩大。实体桌面已裁剪到内框四周仅各留 4.3 cm 支撑带，容纳 57.6 mm 方块旋转到 45° 后的完整底面；支撑带不是额外的中心摆位区。鼠标和键盘摆位都受内框限制。双臂在初始化时采用高位待机，避免还未观测就撞到方块。这个保守测试区域不等于机械臂完整可达工作空间，也不是任意位置成功的数学保证。

尺寸集中在 `simulation/v1/task_workspace.json`。主程序通过 `task_workspace.py` 在编译前调整桌子的真实碰撞与渲染几何，原冻结 XML 不变。此限制只用于初始摆位；抓取过程中不裁剪、不瞬移方块，也不把真实坐标传给 IK。桌高改变时，初始方块随桌面升高，防止出生在桌内；视觉目标 Z 仍由 RGB-D 估计。相机安装和标定保持不变，已在新高度实测验证。

35.6/40/45/50 cm 的比较及最终选择见 `docs/table_height_comparison_20261016.md`。旧配置保留在 `simulation/v1/task_workspace_h356_compact.json`。测试子进程可用环境变量 `OPENARM_TASK_WORKSPACE_CONFIG` 指向独立配置，避免批量测试改动正在使用的默认桌面。

最新桌面裁剪与靠近机器人 3/5 cm 的对照见 `docs/table_footprint_layout_20261016.md`。单独靠近时近侧角点出现相机盲区，未采用为默认；不要把“视觉不可见”直接解释成“物理不可达”。

首次胸部识别失败时会尝试抬臂消除遮挡并使用两侧腕部相机搜索；仍不可见则保留窗口，按 R/空格重新观察，或按 C 重新摆位，不再直接弹出未处理异常并退出。

## 验证

```powershell
.\.venv\Scripts\python.exe build_droid_cube_perception_scene.py --verify
.\.venv\Scripts\python.exe verify_visual_only_contract.py --report outputs/visual_only_contract.json
.\.venv\Scripts\python.exe demo_grasp_viewer.py --check --cube-test-pose 0.365 -0.060 45 --report outputs/check.json
.\.venv\Scripts\python.exe validate_visual_face_grasps.py
.\.venv\Scripts\python.exe test_task_workspace.py
```

`--check` 不打开交互窗口，但仍通过 OpenGL 渲染相机图像。Windows 无显示/GPU 驱动的 CI 不一定能够运行。

当前回归脚本默认测试任务桌面的四角（分别 0°、45°）及中心，共九例，按桌高分别保存，避免覆盖旧证据。`--case-set legacy` 保留历史位置，不受当前初始摆位内框限制；仅作额外诊断，不能自动作为当前允许摆位区域。

2026-10-16 裁剪后的 40 cm 桌高保守区，九个位置分别 0°/45° 加中心 30°，重新验证 19 例通过；证据位于 `outputs/20261016/table_h400_trimmed_validation/` 和 `outputs/20261016/table_h400_trimmed_centre30/`。裁剪前的 29×46 cm/40 cm 桌面也曾通过相同 19 例，证据保留。旧的 35.6 cm 小桌面九例也曾通过，历史证据见 `docs/compact_task_workspace_20261016.md`。这些都不是连续空间内任意位置/姿态的完备性保证。

2026-10-15 最终六组案例均通过；平移位置和旋转角度见 `docs/visual_face_grasp_fix_20261015.md`。动作阶段仿真耗时约 14.28–14.46 秒，不含此前的观察与规划，也不等于端到端耗时。六组通过不代表任意可达位置均有成功保证。

发布快照的验证证据位于 `validation/face_grasp_regression/`；正常运行生成的 `outputs/`、历史采集 `datasets/` 和 Python 缓存不提交。

视觉来源与坐标隔离说明见 `docs/visual_only_contract.md`。腕部图像不可用时会退回胸部 RGB-D，并在报告中标记来源；不是每次运行都强制融合两路图像。

## 文件

- `demo_grasp_viewer.py`：交互摆位、双相机观测、执行及评估。
- `visual_grasp_planner.py`：相对面候选、腕姿比较与路径碰撞排序。
- `handeye_grasp_red_cube.py`：RGB-D 点云、方块朝向、IK 求解。
- `simulation/v1/`：仿真场景及依赖模型；不要手工修改冻结场景。
- `task_workspace.py`、`benchmark_table_heights.py`：真实桌面配置与隔离式桌高对照实验。
- `calibration/`：相机与手眼标定，包含运行所需 JSON。
- `pose2arm_driver.py`：成员 A 的 Pose2Arm 轨迹接入接口。
- `docs/member_A_B_integration.md`：A/B 融合方案；部分是规划方案，并未全部实现。
- `docs/visual_face_grasp_fix_20261015.md`：最新修正、实测结果与边界。
- `HANDOFF_member_B.md`：早期交接记录；当前算法以最新代码与上述验证文档为准。

部分历史 JSON、日志和交接文档记录了原开发电脑的绝对路径，作为历史元数据保留；主程序从当前项目目录解析场景和标定文件。

## 第三方资源

保留 `official_openarm_mujoco/LICENSE` 和原始 README。该目录的第三方资源按其原有 Apache-2.0 许可证提供；本次上传不为整个团队项目额外指定新的许可证。
