# A/B 合并交付（2026-10-16）

入口：[完整合并说明](../../docs/member_AB_integration_20261016.md)。

- `scene_manifest.json`：当前合并版本，68 个代码/模型资源哈希；物理和相机与优化后的 B 场景一致。
- `a_reference_orig_20hz.npz` / `a_hybrid_orig_20hz.npz`：从 A 公开样例导出的原片，417 帧、20 Hz，保留原视频时间。
- `latency_final/`：最终代码的两轮完整物理执行，包含逐帧 CSV、动作/关节/执行器 ctrl NPZ 与延迟判定。
- `acceptance_summary.json`：逐帧人体夹爪信号映射、物理步数和场景不变性审计。
- `tests_ab.txt`：29 项 A/B 与原 B 回归；`tests_a.txt`：11 项 A 专项和额外离线回归。
- `viewer_physics_smoke.json`：新回放入口的 80 帧无窗口物理冒烟检查，不代表 GUI 交互验收或抓取成功。
- `latency/`、`scene_manifest_initial.json`：初始集成测量，保留失败证据。

`demo_member_ab.py` 展示 A 轨迹如何驱动 B，不是自主抓取算法。真实 USB 在线人体推理尚未接通；人体识别耗时不在这些延迟内。查阅报告中的 FAIL 与未验证项，不把“40 项测试通过”解释成全部项目目标完成。
