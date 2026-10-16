# GitHub 同步版本说明（2026-10-16）

> 历史版本：本文记录合并 A 之前的 `a7c5c06` 同步。随后 A/B 合并见 `member_AB_integration_20261016.md`；新版本保留 A 上游已公开的 NPZ、视频和 Git 历史，但没有上传本机新增真实摄像头采集画面。下面“原始 NPZ 未上传”仅指前一次 B 单独交付及其本机旧输入。

本次同步的是权威目录 `D:\openpi\openarm_cloth` 的当前优化场景与成员 B 交付，不是早期大桌面快照。

## 两个运行模式

- `demo_grasp_viewer.py`：交互摆位，胸部 RGB-D 寻物，腕部 RGB-D 精定位/朝向估计，视觉 IK 相对面抓取。沿用你优化后的桌面与防遮挡/待机初始化。
- `realtime_capture.py`：成员 B 接口/性能验收。NPZ 固定右臂执行，采用独立顶视单目 RGB；真实 USB 模式只采集，不假称 A 的在线人体推理已接入。

共用桌面配置为 `simulation/v1/task_workspace.json`：24.6×44.6 cm、顶面 40 cm，方块中心摆位区 X `[.35,.51]`、Y `[-.18,.18]` m。新桌面是编译前覆写真实碰撞几何，**只加载基础 XML 不会自动获得这张桌面**；必须经上述程序入口或 `task_workspace.load_task_model()`。

## GitHub 中可直接查看的证据

- [成员 B 交付说明](../deliverables/member_B_20261016/README.md)
- [延迟及接口审计汇总](../deliverables/member_B_20261016/acceptance_summary.json)
- [冻结场景与哈希](../deliverables/member_B_20261016/scene_manifest.json)
- [18 个桌面抓取案例](../validation/table_h400_trimmed_validation/totals.json)，加 [中心旋转 30 度案例](../validation/table_h400_trimmed_centre30/totals.json)，合计 19/19；是抽样验证，不是任意位置保证。
- [探索阶段失败延迟记录](../validation/member_b_exploratory_latency/)，与正式串行长测同时保留。

以上 `validation/` 是原 `outputs/20261016/` 的证据副本；历史 JSON 里的开发电脑路径保留作来源，不代表他人机器具有相同路径。运行产生的整个 `outputs/`、真实照片、训练数据和临时环境不上传。

## 换一台电脑运行

克隆仓库后进入 `openarm_cloth`，创建 Python 3.12 环境并安装 `requirements_simulation.txt`，按主 README 打开仿真。Windows 需要可用 OpenGL 驱动。

```powershell
git clone https://github.com/sishui667/openpi-.git
Set-Location openpi-/openarm_cloth
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_simulation.txt
.\.venv\Scripts\python.exe freeze_member_b_scene.py
.\.venv\Scripts\python.exe demo_grasp_viewer.py --interactive-place-s 86400
```

成员 A 的原始 NPZ 没有随这次同步上传；要重新运行 NPZ 回放或完整的 `audit_member_b_delivery.py` 输入哈希核对，需要另行提供样例并调整其本机路径。当前逐帧日志/动作记录和已生成审计汇总可直接查看；冻结校验和 20 项单元/场景测试不依赖该原始样例。

`.gitattributes` 为项目禁用自动换行转换，保持 SHA256 所需的原始字节；不要为了通过校验而直接重写 manifest。真人摄像头完整在线闭环、人体夹爪意图和 15 秒搬运释放仍未验收，具体边界见交付说明。
