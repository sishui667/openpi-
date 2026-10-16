# Pose2Arm 人体手臂与手部识别诊断及升级方案

更新时间：2026-10-16

## 结论

可以实现明显更准确的识别，但不能只继续调整 IK、碰撞阈值或平滑参数。当前主要误差已经发生在机械臂映射之前：MediaPipe 的肩、肘、腕三维点在遮挡时跳变，独立的 Hand Landmarker 又会丢手和发生左右手关联不稳定。下游的保持上一帧、平滑和避碰只能掩盖输入错误，并会产生滞后或停住。

推荐先做一个不影响现有稳定输出的 A/B 原型：在同一段 `p2a_test1.mp4` 上并行运行当前 MediaPipe 与 RTMW3D-L，输出两套人体骨架叠加视频和统一指标。若 RTMW3D 达到验收标准，再替换生产提取器。

## 当前开源 Pose2Arm 实际做了什么

仓库并不是一个新的人体手部识别网络。其公开流程是：

```text
视频/摄像头
  -> MediaPipe PoseLandmarker
  -> 肩、肘、腕组成的 18 维方向特征
  -> PPO 策略输出双臂 14 个关节目标
```

18 维特征为左右两臂的上臂方向、前臂方向和手臂平面法向量。手指不在 PPO 的输入中；夹爪开合是本地后续增加的独立 Hand Landmarker 链路。

公开视频 `pose2arm V1：这次不靠硬解 IK 了` 展示的是 Isaac Sim 中的大尺度手臂方向跟随，并未给出手指精度、腕部三维误差、延迟或跨视频泛化指标。仓库 README 还明确说明策略只在 `sample01.mp4` 上训练，其他动作和人体比例的泛化未验证。因此该视频可以证明 PPO 能学习关节映射，但不能证明 MediaPipe 在新视频上的手部识别已经足够准确。

本机当前回放的 `p2a_test1_reference_locked.npz` 仍来自几何重定向、IK、轨迹优化和碰撞投影；本地没有公开演示所需的 PPO checkpoint。因此“开源仓库不靠硬解 IK”的训练路径目前没有实际替代 MuJoCo 回放链路。

## `p2a_test1` 实测证据

视频参数：1080 x 1920，612 帧，约 29.38 FPS。手在画面中的中位对角尺寸约为 292～310 像素，分辨率足够，问题不能简单归因于“手太小”。

### 手臂关键点

| 指标 | 左臂 | 右臂 |
|---|---:|---:|
| 肩/肘/腕最低可见度低于 0.55 | 64 / 612 帧 | 34 / 612 帧 |
| 单帧腕部最大跳变 | 599 px | 334 px |
| 前臂估计长度变异系数 | 12.2% | 10.0% |

真实人的骨长不会逐帧改变。较大的骨长漂移和腕部跳变说明单目 `pose_world` 深度与遮挡恢复不稳定。现有 `gate_arm_landmarks()` 在低置信度时保持上一帧，所以输入错误会表现为机械臂滞后或卡住。

### 手与夹爪信号

| 指标 | 左手 | 右手 |
|---|---:|---:|
| Hand Landmarker 检出率 | 87.25% | 93.79% |
| 最长连续丢失 | 51 帧（约 1.74 秒） | 19 帧（约 0.65 秒） |
| 手检测腕点与人体腕点差异 p95 | 81 px | 75 px |
| 最大差异 | 122 px | 307 px |
| 归一化捏合量单帧变化 p95 | 0.278 掌宽 | 0.282 掌宽 |

当前手检测和人体检测分两次独立运行，再按单帧腕部最近邻分配左右手。双手靠近或交叉时没有跨帧身份约束，容易丢手、换手或将错误手势传给另一侧夹爪。

## 论文与可用开源实现

### 1. RTMW / RTMW3D：首选替换检测器

- 论文：Tao Jiang 等，*RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation*，2024。
- 代码与权重：OpenMMLab MMPose 的 `projects/rtmpose3d`。
- 能力：一套模型输出身体、双手、脸和脚的全身关键点；RTMW3D-L 公开模型在 H3WB 上报告 0.056 m MPJPE。
- 对本项目的价值：身体和手来自同一个人体实例与坐标系，可去掉当前“独立检测两遍再按腕部最近邻配对”的薄弱环节。

### 2. MotionBERT / VideoPose3D：利用时间信息恢复稳定三维姿态

- MotionBERT 使用时空 Transformer，从有噪声或缺失的二维序列恢复三维运动。
- VideoPose3D 使用扩张时序卷积，从一段二维关键点序列估计三维姿态。
- 对本项目的价值：人体动作本质上连续，连续帧能补偿单帧遮挡和深度歧义。它们更适合身体主关节；手指仍应使用 RTMW3D 的手部点或独立手部模型。

### 3. HaMeR / Hand4Whole：高精度手部的离线或第二阶段方案

- HaMeR 使用大规模视觉 Transformer 从单目图像重建三维手网格。
- Hand4Whole 强调身体特征不足以准确预测手指，需要专门的手部特征并与腕部运动学对齐。
- 对本项目的价值：可以作为离线高精度对照或困难帧回退，但对只有一个开合自由度的 OpenArm 夹爪而言，直接实时部署 HaMeR 成本偏高，第一阶段没有必要。

### 4. 遮挡感知时序优化

Occlusion-Aware Networks 和 STRIDE 的共同结论是：不能把低置信度关节与正常关节等权处理，也不能仅用低通滤波。应显式使用可见度/不确定度，并用运动先验补全短时遮挡。

## 推荐的新识别链路

```text
视频帧
  -> 人体框检测与跨帧 identity tracking
  -> RTMW3D-L：133 个全身 3D 关键点 + 每点置信度
  -> 置信度加权的 9～15 帧时序优化
       - 固定骨长
       - 速度/加速度连续
       - 2D 重投影一致
       - 短时遮挡插值，长期遮挡安全回中
  -> 躯干局部坐标系与人体比例归一化
  -> 肩/肘/腕方向与双腕相对位置
  -> 现有 IK/PPO 与碰撞安全层
```

时序优化建议最小化：

```text
E = E_keypoint(confidence)
  + lambda_bone * E_fixed_bone_length
  + lambda_velocity * E_velocity
  + lambda_acceleration * E_acceleration
  + lambda_reprojection * E_2d_reprojection
```

离线视频可使用前后帧的零相位窗口，不产生相位滞后；实时模式使用 7～9 帧固定延迟窗口，约增加 0.10～0.14 秒延迟。

## 夹爪开合改进

不要只使用拇指尖到食指尖一个距离。改用同一只手的多个归一化特征，例如：

- 拇指尖—食指尖距离 / 掌宽；
- 拇指尖—中指尖距离 / 掌宽；
- 四个指尖到掌心的平均距离 / 掌宽；
- 各特征的关键点置信度。

把这些特征融合成一个 `[0, 1]` 的手部闭合量，再做小迟滞和短时序滤波。OpenArm 夹爪只有一个开合自由度，因此无需恢复并映射全部手指关节，但应使用全部可靠手指点共同判断开合。

## 分阶段实施建议

### 阶段 A：只做 A/B 识别验证，不改机械臂输出

1. 新建独立 `pose3d` 环境，避免破坏当前 MediaPipe 环境。
2. 安装 RTMW3D-L 及公开权重。
3. 对 `p2a_test1.mp4` 输出 MediaPipe 与 RTMW3D 双路骨架叠加视频。
4. 输出检出率、左右手身份连续性、骨长漂移、腕部跳变、捏合曲线。

### 阶段 B：接入人体比例归一化和时序优化

1. 用肩宽、上臂和前臂中位长度建立每人标定。
2. 先在人体关键点空间优化，再进入机器人映射；不要在错误关节角生成后才平滑。
3. 保留现有参考锁定避碰作为末端安全层。

### 阶段 C：再评估 PPO

识别输入稳定后，再决定是否训练 PPO 替代逐帧 IK。PPO 解决的是人体目标到机器人关节的映射和动力学连续性，不能修复错误的人体关键点。

## 建议验收标准

- 两手各自检出率不低于 98%，最长连续丢失不超过 5 帧；
- 上臂、前臂估计长度变异系数低于 3%；
- 无左右手身份交换；
- 人工标注的开/合事件 F1 不低于 0.95；
- 源视频骨架叠加检查中，肩、肘、腕不出现明显跨肢体跳点；
- 接入机器人后再检查腕部任务空间误差、方向误差、碰撞率和端到端延迟，识别升级不得以更大的滞后换取表面平滑。

## 参考资料

- Pose2Arm 开源仓库：https://github.com/angli112397/pose2arm
- 示例视频：https://www.bilibili.com/video/BV1NETk62EyA/
- RTMW 论文：https://arxiv.org/abs/2407.08634
- RTMW3D 代码与权重：https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d
- RTMPose 论文：https://arxiv.org/abs/2303.07399
- DWPose 论文：https://arxiv.org/abs/2307.15880
- MotionBERT 论文：https://arxiv.org/abs/2210.06551
- VideoPose3D 论文：https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html
- HaMeR 论文：https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html
- Hand4Whole 论文：https://arxiv.org/abs/2011.11534
- Occlusion-Aware Networks：https://openaccess.thecvf.com/content_ICCV_2019/html/Cheng_Occlusion-Aware_Networks_for_3D_Human_Pose_Estimation_in_Video_ICCV_2019_paper.html
- STRIDE：https://openaccess.thecvf.com/content/WACV2025/html/Lal_STRIDE_Single-Video_Based_Temporally_Continuous_Occlusion-Robust_3D_Pose_Estimation_WACV_2025_paper.html
