# OpenArm 叠毛巾场景

## 运行

在 `openpi` 环境中执行：

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\build_towel_scene_v25_4.py
python .\launch_towel_scene.py
```

`launch_towel_scene.py` 默认加载 `simulation\v1` 中最新的
`folding_scene_V25_4_TOWEL_SUPPORT_*.xml`，并自动恢复 `towel_ready` 双臂准备姿态。
若希望让毛巾自由落到支撑层，可执行：

```powershell
python .\launch_towel_scene.py --dynamic
```

也可以直接打开指定版本：

```powershell
python .\launch_towel_scene.py --scene simulation\v1\folding_scene_V25_4_TOWEL_SUPPORT_04.xml
```

## 本次修复

- 毛巾位置改为以桌面中心为基准，范围为 `x=[0.28, 0.62] m`、`y=[-0.21, 0.21] m`。
- 增加 `towel_ready` 双臂准备姿态，避免打开场景时机器人停在零位。
- 使用独立碰撞位和桌面上方 5 mm 的不可见毛巾支撑层，避免柔性体穿桌或数值发散。
- 将时间步设为 `0.0005`，使用 `CG + implicitfast`，并在构建时进行 1.5 s 头less 稳定性检查。

当前版本是稳定的平铺/叠毛巾场景基线；机器人-毛巾抓取碰撞和自动折叠轨迹仍需后续控制脚本接入。

## 键盘遥操作与数据采集

使用最终 V25.4 场景启动双臂笛卡尔遥操作：

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\teleop_towel_dataset.py --robot-cloth-contact
```

如果希望使用按钮控制，可打开旁边的控制面板：

```powershell
python .\teleop_towel_dataset.py --robot-cloth-contact --gui
```

控制面板会与 MuJoCo 可视化窗口同步，提供左臂、右臂、双臂选择，末端移动/姿态调整，夹爪开合，复位、暂停和录制按钮。移动和姿态按钮支持按住连续运动，松开即停止。

面板中的速度参数：

- `平移步长(mm)`：每次控制增量，范围 1--50 mm。
- `旋转步长(度)`：每次姿态增量，范围 1--30 度。
- `连续间隔(ms)`：按住按钮时的重复间隔，越小越快，范围 20--250 ms。
- `慢速/标准/快速`：快速预设为 25 mm、12 度、35 ms。

由于 MuJoCo 官方 Python viewer 没有开放自定义控件插入接口，按钮显示在 MuJoCo 窗口旁边，但控制的是同一个仿真状态。

如果同时采集三路相机图像：

```powershell
python .\teleop_towel_dataset.py --robot-cloth-contact --images --record
```

键位说明：

- `1/2/3`：选择左臂、右臂、双臂；双臂模式下 `M` 切换 Y 方向镜像。
- `1/2/3` 只负责选择控制对象，本身不会移动机械臂；选择后还要按下面的移动键。
- `I/K`：末端沿 X 轴前后移动，X 正方向是毛巾折叠方向。
- `J/L`：末端沿 Y 轴移动；`U/O`：沿 Z 轴升降。
- `Q/E`：偏航；`Z/C`：俯仰；`X/V`：滚转。
- `G/H`：闭合/打开当前选中夹爪；`R`：恢复 `towel_ready`；`P`：暂停。
- `SPACE`：开始/停止录制；`[`、`]`：减小/增大单次移动步长。

默认输出到 `datasets/towel_teleop/episode_时间戳/`：

- `episode.npz`：时间、完整仿真状态、控制量、目标动作、双臂末端位姿和毛巾中心。
- `metadata.json`：场景、关节顺序、相机和键位定义。
- `events.json`：录制期间的键盘事件。
- `images/`：启用 `--images` 后保存的相机 PNG 帧。

当前脚本采用“末端目标 + 阻尼最小二乘 IK + 重力补偿 PD 力矩控制”。`--robot-cloth-contact` 只在运行时打开机器人与毛巾的独立碰撞位，不修改已验证的 V25.4 XML；建议正式采集时使用该参数，并先录制短片段检查夹爪是否稳定夹住毛巾。

控制器采用事件触发 IK：只在按下移动或姿态键时更新对应手臂目标，单独选择某一只手臂不会改变另一只手臂。若末端超出安全工作空间，仿真会自动暂停；此时按 `R` 复位，再按 `P` 继续。
