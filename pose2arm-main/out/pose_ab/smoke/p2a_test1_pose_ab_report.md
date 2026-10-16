# p2a_test1 人体关键点 A/B 测试

视频：`D:\openpi\pose2arm-main\videos\p2a_test1.mp4`；比较帧数：10；帧率：29.375 FPS。

并排回放：`D:\openpi\pose2arm-main\out\pose_ab\smoke\p2a_test1_pose_ab.mp4`。左为 MediaPipe，右为 RTMW3D。

| 指标 | MediaPipe | RTMW3D |
|---|---:|---:|
| left_arm_complete_pct | 0.0 | 100.0 |
| left_arm_max_missing_run | 10 | 0 |
| right_arm_complete_pct | 30.0 | 100.0 |
| right_arm_max_missing_run | 7 | 0 |
| left_hand_complete_pct | 0.0 | 0.0 |
| left_hand_max_missing_run | 10 | 10 |
| right_hand_complete_pct | 10.0 | 10.0 |
| right_hand_max_missing_run | 9 | 9 |
| left_wrist_jump_p95_px | nan | 13.79 |
| left_wrist_jump_max_px | nan | 15.54 |
| left_forearm_2d_cv | nan | 0.023 |
| left_pinch_ratio_p05 | nan | nan |
| left_pinch_ratio_p95 | nan | nan |
| right_wrist_jump_p95_px | 218.41 | 548.04 |
| right_wrist_jump_max_px | 225.58 | 881.21 |
| right_forearm_2d_cv | 0.133 | 0.673 |
| right_pinch_ratio_p05 | 0.312 | 0.339 |
| right_pinch_ratio_p95 | 0.312 | 0.577 |

指标解释：complete_pct 是关节同时落在画面内且达到各自阈值的比例；max_missing_run 是最长连续缺失帧数；wrist_jump 是相邻可见帧像素位移；forearm_2d_cv 是前臂二维投影长度变异系数；pinch_ratio 是拇指与食指指尖距离除以掌宽。

注意：没有人工标注真值，不能据此断言哪个模型更准确。两模型分数未校准，MediaPipe 手部只提供整手检出标记；二维骨长受深度、透视和遮挡影响。RTMW3D 的 model_xyz 是模型坐标，不是机械臂坐标或米制空间，不可直接输入控制器。
