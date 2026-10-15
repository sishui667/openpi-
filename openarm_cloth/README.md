# OpenArm 双相机视觉抓取仿真（成员 B）

MuJoCo 物理仿真：胸部 RGB-D 粗定位、腕部 RGB-D 精定位与方块朝向估计、多候选相对面抓取 IK、路径碰撞排序、接近后视觉校正和纯摩擦夹持。支持交互设置方块位置与旋转角度。

这是成员 B 的当前项目快照。仓库根目录中原有的 `pose2arm-main/`、`Joint-action-n-pose-estimation-main/` 和 `openarm_cloth_with_dependencies_20260929/` 保留不变。

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

## 验证

```powershell
.\.venv\Scripts\python.exe build_droid_cube_perception_scene.py --verify
.\.venv\Scripts\python.exe verify_visual_only_contract.py --report outputs/visual_only_contract.json
.\.venv\Scripts\python.exe demo_grasp_viewer.py --check --cube-test-pose 0.365 -0.060 45 --report outputs/check.json
.\.venv\Scripts\python.exe validate_visual_face_grasps.py
```

`--check` 不打开交互窗口，但仍通过 OpenGL 渲染相机图像。Windows 无显示/GPU 驱动的 CI 不一定能够运行。

2026-10-15 最终六组案例均通过；平移位置和旋转角度见 `docs/visual_face_grasp_fix_20261015.md`。动作阶段仿真耗时约 14.28–14.46 秒，不含此前的观察与规划，也不等于端到端耗时。六组通过不代表任意可达位置均有成功保证。

发布快照的验证证据位于 `validation/face_grasp_regression/`；正常运行生成的 `outputs/`、历史采集 `datasets/` 和 Python 缓存不提交。

视觉来源与坐标隔离说明见 `docs/visual_only_contract.md`。腕部图像不可用时会退回胸部 RGB-D，并在报告中标记来源；不是每次运行都强制融合两路图像。

## 文件

- `demo_grasp_viewer.py`：交互摆位、双相机观测、执行及评估。
- `visual_grasp_planner.py`：相对面候选、腕姿比较与路径碰撞排序。
- `handeye_grasp_red_cube.py`：RGB-D 点云、方块朝向、IK 求解。
- `simulation/v1/`：仿真场景及依赖模型；不要手工修改冻结场景。
- `calibration/`：相机与手眼标定，包含运行所需 JSON。
- `pose2arm_driver.py`：成员 A 的 Pose2Arm 轨迹接入接口。
- `docs/member_A_B_integration.md`：A/B 融合方案；部分是规划方案，并未全部实现。
- `docs/visual_face_grasp_fix_20261015.md`：最新修正、实测结果与边界。
- `HANDOFF_member_B.md`：早期交接记录；当前算法以最新代码与上述验证文档为准。

部分历史 JSON、日志和交接文档记录了原开发电脑的绝对路径，作为历史元数据保留；主程序从当前项目目录解析场景和标定文件。

## 第三方资源

保留 `official_openarm_mujoco/LICENSE` 和原始 README。该目录的第三方资源按其原有 Apache-2.0 许可证提供；本次上传不为整个团队项目额外指定新的许可证。
