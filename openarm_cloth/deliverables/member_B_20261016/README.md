# 成员 B 交付包：场景、NPZ 执行器接口与实时采集

> 历史 B 单独交付，运行代码和冻结哈希对应 Git 版本 `a7c5c06`。当前 A 已提供夹爪通道并接通 B，最新说明在 `../member_AB_20261016/README.md`；当前默认冻结清单也改为该目录。旧测量不重写，不应拿旧“缺少手指信号”结论描述 A 的新版本。

日期：2026-10-16。权威工程：`D:\openpi\openarm_cloth`。

结论：**离线 NPZ 接入版已实现并测试；真实单目摄像头采集可运行。由于 A 目前只提供 NPZ，真人摄像头→Pose2Arm 在线推理→机器人这一完整链路尚不能验收。**

## 交付对应

| 成员 B 任务 | 实际交付 | 当前状态 |
|---|---|---|
| 固定单臂方块场景与物理参数 | `scene_manifest.json` + 工程内 XML / `task_workspace.json` / `member_b_scene.py` | 冻结并校验，右臂活动、左臂固定待机 |
| 接入 Pose2Arm→执行器 | `pose2arm_driver.py`、`member_b_actuators.py` | 762 行真实 NPZ 完整执行；限位/限速/力矩限幅 |
| 单目实时流及延迟测试 | `realtime_capture.py`、`member_b_realtime.py`、`latency/` | 仿真控制链与真实 USB 采集分别实测 |
| 证明与复核 | `acceptance_summary.json`、逐帧 CSV、执行器 NPZ、`tests.txt` | 时间戳、物理步数、动作边界审计通过 |

这些脚本在工程根目录，本目录是配置清单与测试证据目录，**不是脱离工程即可运行的独立压缩包**。不修改 A 的原始 NPZ，不上传 GitHub，不保存真实摄像头原始画面。

## 本次正式串行长测

仿真分辨率 256×192 RGB，20 Hz，每周期完整 100×0.0005 s 物理步；30 帧预热单独记录并排除。Windows 仅测试进程使用 HIGH 优先级，没有结束用户程序。USB 实测 640×480、自动曝光、MJPG、普通进程优先级。

| 测试 | 帧数 | 平均 ms | P95 ms | 最大 ms | 超过 50 ms | 实际频率 Hz |
|---|---:|---:|---:|---:|---:|---:|
| 仿真第 1 轮 | 500 | 23.03 | 31.69 | 52.28 | 3/500 | 20.000 |
| 仿真第 2 轮 | 500 | 22.04 | 28.47 | 49.73 | 0/500 | 20.000 |
| 仿真完整 NPZ | 762 | 22.11 | 33.59 | 56.68 | 5/762 | 20.000 |
| USB 主机取最新帧 | 500 | 0.41 | 0.35 | 58.35 | 1/500 | 20.000 |

仿真统计范围：**请求渲染 RGB→分割/拟合→NPZ 目标映射→执行器写入→完整控制周期物理推进完成**。完整 NPZ 中，首次执行器命令被物理引擎消费平均 7.73 ms，完整周期平均 22.11 ms；不是把首次写入时刻混成完整推进耗时。

USB 表中的 0.41 ms 只是主机从后台“最新帧槽”取得新帧的等待/复制耗时，**绝不是摄像头曝光到机器人动作的延迟**。摄像头 API 读帧平均 33.60 ms、P95 47.81 ms；交付时主机帧龄平均 17.64 ms、P95 34.67 ms，分别保留。光学曝光/传感器内部缓存延迟不可从这些值推导。源帧序号递增，不能重复旧图冒充新帧；20 Hz 消费会主动跳过约 30 fps 源流中的中间帧，不累积队列。

程序软实时口径是 P95≤50 ms 且超预算工作帧≤1%；另存 `strict_zero_miss_pass`。按此软口径四轮通过，但按**每帧均≤50 ms**的严格口径，只有仿真第 2 轮通过。普通 Windows 环境不承诺硬实时。CSV 写入不计入 `tick_work_ms`，实际带节拍频率包含写入；模型初始化、预热、显示时延及 A 的离线人体提取/IK 不计入上述控制链。

`acceptance_summary.json` 的 `evidence_audit=PASS` 仅表示日志自洽、完整物理推进和动作边界检查通过，不等于抓取成功或严格实时通过。

## A 的样例暴露的问题

- 样例 SHA256：`36b1d25b7115612f9e15e7fea219b73fc5b4ac559aad5d5bab12dc4d76b2a324`，源路径见汇总文件；未复制或修改原件。
- R1 单帧最大 3.191 rad，是 20 Hz 速度上限的 3.81 倍；R3 是 8.80 倍。48 帧触发限速，源目标到限速后命令最大差 2.353 rad。
- 机器人实际角度到限速后命令的全序列 RMS 误差 .256 rad、最大 2.063 rad；与“限速导致的源目标滞后”分开报告。此轨迹不能当作高精度跟踪示范。
- 762 帧中视觉位置有效 755 帧，存在遮挡，未伪造全程识别成功。跟踪 748 帧复用前均验证当前 RGB 掩码，另有局部/全局重定位。
- 762 帧夹爪均保持打开；样例没有 `hand_distance_m`，**本次没有证明真人夹爪控制或 NPZ 成功抓起方块**。

## 复现

从工程根目录执行；本机 `python` 命令是不可用存根，务必用下列解释器，其他电脑替换成自己的 Python 3.12 环境。

```powershell
Set-Location D:\openpi\openarm_cloth
$memberBPython = 'D:\miniconda3\envs\openpi\python.exe'
$memberBNpz = 'C:\Users\26473\Desktop\openpi\openpi\pose2arm-main\data\p2a_test.npz'

# 只校验，不重写冻结清单
& $memberBPython freeze_member_b_scene.py
& $memberBPython test_member_b_delivery.py
& $memberBPython test_task_workspace.py
& $memberBPython audit_member_b_delivery.py

# 新日志另存，保留正式证据
& $memberBPython realtime_capture.py --source simulation --npz $memberBNpz --frames 762 --hz 20 --warmup 30 --width 256 --height 192 --high-priority --scene-manifest deliverables/member_B_20261016/scene_manifest.json --outdir outputs/member_B_manual_sim

# 真实摄像头预览；Q 结束，默认不保存真实照片
& $memberBPython realtime_capture.py --source usb --device 0 --frames 500 --hz 20 --warmup 30 --width 640 --height 480 --preview --outdir outputs/member_B_manual_usb
```

相机必须没有被其他程序独占。人体摄像头的正面/斜上方位置及整手入画需要操作者布置；本次验证设备可读和实时采集，未声称已验证手指识别。

正式证据不应覆盖。只有团队明确改变冻结版本时才运行 `freeze_member_b_scene.py --write` 并重新测试；普通启动只验证。包装 XML 被旧临时脚本覆盖也会触发 SHA256 不一致。

## 参数与证据说明

- `scene_manifest.json`：当前编译后实际物理参数、6 路相机、右臂/待机策略、目标区与任务定义、60 个文件哈希。
- `latency/sim_run_1/`、`sim_run_2/`、`sim_full_npz/`：逐帧时钟、阶段耗时、仿真时间以及动作/关节/ctrl 记录。
- `latency/usb_capture/`：真实采集逐帧时钟与源序号，不含机器人动作。
- `tests.txt`：13 项 B 专项和 7 项当前桌面/交互回归的真实运行输出。
- 当前接口完整说明在 `docs/member_B_interface_20261016.md`。历史 `legacy_realtime_capture_20261014.py` 每 tick 仅一个物理步，不再用于验收。

现场精抓取扩展仍用 `demo_grasp_viewer.py` 的胸部/腕部 RGB-D。当前低分辨率单目在 5 个位姿上误差 2.24–8.94 mm，不能宣称全小于 6 mm；离开桌面、倾斜或严重遮挡时不适用桌面平面模型。过去 19 例自主抓取测试为另一个控制模式，不能替代这里的 A/B 闭环证据。

未完成、不能包装为完成：A 在线人体推理延迟、真人手指开闭链路、相对末端示范的物体坐标重定位、NPZ 抓起并搬运到目标区释放、15 秒完整任务验收、连续工作空间任意位置的成功保证。B 已提供目标区和判定阈值契约，不等于已实现完整成功判定器或策略。

## 保留的探索/失败记录

所有旧日志保留在 `outputs/20261016/`。包括 `member_b_npz_long_baseline`（762 帧，17 帧超预算）、`member_b_cached_smoke`（50 帧，2% 超预算）、`member_b_final_sim_run2`（并发相机诊断时失败）、`member_b_final_usb`（相机设置优化前失败）。本目录三轮仿真加一轮 USB 是固定代码/参数后的串行长测，不能宣称旧失败不存在。

相机设置参考 OpenCV 4.12 的 [DirectShow 实现](https://github.com/opencv/opencv/blob/4.12.0/modules/videoio/src/cap_dshow.cpp)：FPS 设置可能重建设备，因此编码设置放在帧率之后，自动曝光采用该后端支持的设置。参数 setter 返回成功不等于设备曝光可独立测量；本次没有用黑暗手动曝光画面换取帧率。
