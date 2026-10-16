# p2a_test1 人体关键点 A/B 测试

视频：`D:\openpi\pose2arm-main\videos\p2a_test1.mp4`；比较帧数：612；帧率：29.375 FPS。

并排回放：`D:\openpi\pose2arm-main\out\pose_ab\p2a_test1_pose_ab.mp4`。左为 MediaPipe，右为 RTMW3D。

| 指标 | MediaPipe | RTMW3D |
|---|---:|---:|
| left_arm_complete_pct | 90.03 | 100.0 |
| left_arm_max_missing_run | 47 | 0 |
| right_arm_complete_pct | 95.92 | 100.0 |
| right_arm_max_missing_run | 8 | 0 |
| left_hand_complete_pct | 83.99 | 87.58 |
| left_hand_max_missing_run | 51 | 41 |
| right_hand_complete_pct | 93.46 | 95.1 |
| right_hand_max_missing_run | 19 | 14 |
| left_wrist_jump_p95_px | 37.2 | 31.25 |
| left_wrist_jump_max_px | 68.05 | 79.88 |
| left_wrist_jump_max_frames | 57-58 | 57-58 |
| left_forearm_2d_cv | 0.267 | 0.27 |
| left_pinch_ratio_p05 | 0.127 | 0.167 |
| left_pinch_ratio_p95 | 1.787 | 2.237 |
| right_wrist_jump_p95_px | 34.05 | 34.28 |
| right_wrist_jump_max_px | 225.58 | 881.21 |
| right_wrist_jump_max_frames | 8-9 | 8-9 |
| right_forearm_2d_cv | 0.275 | 0.286 |
| right_pinch_ratio_p05 | 0.22 | 0.205 |
| right_pinch_ratio_p95 | 2.203 | 2.521 |

指标解释：complete_pct 是关节同时落在画面内且达到各自阈值的比例；max_missing_run 是最长连续缺失帧数；wrist_jump 是相邻可见帧像素位移；forearm_2d_cv 是前臂二维投影长度变异系数；pinch_ratio 是拇指与食指指尖距离除以掌宽。

注意：没有人工标注真值，不能据此断言哪个模型更准确。两模型分数未校准，MediaPipe 手部只提供整手检出标记；二维骨长受深度、透视和遮挡影响。RTMW3D 的 model_xyz 是模型坐标，不是机械臂坐标或米制空间，不可直接输入控制器。

本次重点复核：右腕最大跳变对应的帧对见上表；大跳变可能是身份切换、遮挡后重定位或真实快速动作，须结合并排视频判断。

## 本次观察

RTMW3D 在本片的双臂完整帧占比为 100%，MediaPipe 左/右臂分别为 90.03% / 95.92%；双手完整帧占比也略高（左 87.58% 对 83.99%，右 95.10% 对 93.46%）。但 RTMW3D 并非全程更稳：右腕最大跳变发生在第 8→9 帧（881.21 px）。画面中手突然靠近镜头并严重模糊，RTMW3D 第 8 帧把腕点放在画面底部，第 9 帧重新定位到手部；这会让直接驱动机械臂产生危险突变。不能仅凭检出率就把新模型接入控制器，应先加入时间一致性检查、异常跳变拒绝和短时预测。
