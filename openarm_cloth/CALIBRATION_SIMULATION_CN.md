# OpenArm 仿真相机标定数据采集

## 启动检查

在 `openpi` 环境中运行：

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\calibration_simulation.py --preview
```

程序会自动生成并加载：

```text
simulation/v1/calibration_board_scene.xml
simulation/v1/calibration_board_scene.board.json
```

标定板是固定在桌面上的 9×6 非对称棋盘格，内角点为 8×5，方格边长为 15 mm。非对称列数用于消除棋盘旋转 180° 的方向歧义。

## 操作

```text
1 / 2       选择左臂 / 右臂
I / K       TCP X+ / X-
J / L       TCP Y- / Y+
U / O       TCP Z+ / Z-
Q / E       yaw+ / yaw-
Z / C       pitch+ / pitch-
X / V       roll+ / roll-
N           记录一个同步样本
SPACE       开始 / 暂停连续记录
R           复位到 ready 姿态
P           暂停 / 恢复物理仿真
F1          显示帮助
```

推荐先使用 `N` 手动采样。每次移动机械臂并等待稳定后按一次 `N`，左右腕部相机分别采集 20～40 个不同姿态。连续记录模式适合快速检查，不建议直接用于最终手眼标定。

## 左右臂自动采集并标定

当前已提供左右臂自动流程。它会自动生成可达姿态，先用 IK 检查对应腕部相机能否完整看到标定板，再按“移动—稳定—采集”的顺序保存胸部相机和对应腕部相机的 RGB-D、关节状态、TCP 位姿及相机位姿，最后自动完成棋盘角点检测、手眼标定和深度一致性检查。

带相机预览运行：

```powershell
python .\auto_calibrate_left.py --preview
```

右臂运行：

```powershell
python .\auto_calibrate_right.py --preview
```

无窗口测试：

```powershell
python .\auto_calibrate_left.py --headless --pose-count 15
```

右臂当前固定标定板可稳定采集 10 个有效姿态：

```powershell
python .\auto_calibrate_right.py --headless --pose-count 10
```

左臂结果位于 `calibration/auto_left/session_*/calibration_left_wrist.json`，右臂结果位于 `calibration/auto_right/session_*/calibration_right_wrist.json`。仿真相机是无畸变针孔模型，默认固定仿真内参和零畸变；如果以后把同一求解器用于真实相机，可额外尝试 `--free-distortion`。

注意：自动流程目前是仿真中的关节位姿插值，不会连接真实机械臂。

## 胸部相机自动标定

胸部相机是固定相机，不做腕部相机的 TCP 手眼标定。程序会自动移动仿真标定板，采集不同位置和角度的胸部 RGB-D 图像，然后求解：

```text
相机内参、畸变参数、T_base_chest_camera_opencv、深度一致性误差
```

推荐使用 640×480 采集，以提高棋盘角点精度：

```powershell
python .\auto_calibrate_chest.py --preview --pose-count 20 --image-width 640 --image-height 480
```

无窗口测试：

```powershell
python .\auto_calibrate_chest.py --headless --pose-count 20 --image-width 640 --image-height 480
```

结果位于 `calibration/auto_chest/session_*/calibration_chest.json`。仿真相机默认使用固定针孔内参和零畸变；真实相机数据可追加 `--free-distortion` 进行 OpenCV 畸变估计。

## 输出内容

每次运行生成：

```text
calibration/sim_data/session_*/
├── metadata.json
├── frames.json
├── samples.npz
└── images/
    ├── chest_d435i_preview/rgb/*.png
    ├── chest_d435i_preview/depth/*.npy
    ├── v22_left_d405_rgb/rgb/*.png
    ├── v22_left_d405_rgb/depth/*.npy
    ├── v22_right_d405_rgb/rgb/*.png
    └── v22_right_d405_rgb/depth/*.npy
```

`samples.npz` 包含：

```text
time_s
qpos / qvel / ctrl
tcp_pose_world
board_pose_world
board_inner_corners_world
camera_pose_world_mujoco
camera_pose_world_opencv
```

深度文件是 `float32` 米单位。仿真相机无畸变，内参和坐标系说明保存在 `metadata.json`。

## 无窗口自检

```powershell
python .\calibration_simulation.py `
  --headless `
  --samples 2 `
  --output-dir .\calibration\sim_test
```

这个命令用于确认 XML、RGB、Depth 和 NPZ 记录链路，不用于产生正式标定数据。

