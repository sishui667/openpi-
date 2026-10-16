# 双手靠近与交换动作：轻量 RRT-Connect 方案

## 问题

逐帧硬安全投影会沿“上一安全姿态 → 当前目标”前进。一旦停在碰撞边界，后续若仍要求
穿过同一障碍区域，就可能连续保持上一帧。滑动窗口随机优化能修复局部偏差，但对于必须
从另一侧绕行的动作，可能困在原轨迹的拓扑邻域。

## 论文依据与取舍

1. Kuffner 与 LaValle 的 RRT-Connect（ICRA 2000）从起点和终点建立两棵树，并贪婪地
   尝试连接，适合高维、单次查询的机械臂规划。本项目采用它处理 14-DOF 双臂短窗口。
   论文：https://www.clear.rice.edu/comp450/papers/kuffner_lavalle_00.pdf
2. Kalakrishnan 等人的 STOMP（ICRA 2011）使用无梯度随机轨迹扰动，同时优化障碍与
   平滑代价。本项目已有的滑动窗口优化属于这一思路；它适合局部改善，但依赖初始轨迹，
   因而保留 RRT-Connect 作为直连失败后的拓扑逃逸层。
   论文：https://ftp.oregonstate.edu/pub/ros/ros_wiki_mirror/attachments/Papers%282f%29ICRA2011_Kalakrishnan/kalakrishnan_icra2011.pdf
3. 没有采用 RRT*：短交换窗口首先要求快速找到可行路径，而不是渐近最优。找到路径后用
   shortcut 和固定时长重采样，成本更低、更符合当前离线视频流程。

## 实际执行顺序

```text
共享尺度 + 双腕相对位置 IK
        ↓
滑动窗口轨迹优化
        ↓
硬安全投影（产生可识别的停住窗口）
        ↓
双腕靠近且连续停住？
  ├─ 否：保留原轨迹
  └─ 是：先检查安全直连
          ├─ 可直连：固定时长重采样
          └─ 不可直连：14-DOF RRT-Connect
                       ↓
                    shortcut
                       ↓
                  固定时长重采样
        ↓
MuJoCo 连续边碰撞复检 + 硬安全兜底
```

RRT 搜索以原轨迹附近采样为主，少量目标偏置和全局采样。树节点使用 14 个双臂关节角，
夹爪开合不参与搜索，但重采样后的每一条边均使用对应视频时刻的真实夹爪开度检查。

对于“伸出再收回”这类窗口，起点和终点可能几乎相同，不能仅连接两端。实现会检测窗口内
双腕目标的最大偏移帧，将其作为语义中间目标，把动作拆成“绕到运动峰值 → 安全返回”。
若双臂同步路径仍显得阻塞，还会分别测试左臂先让路、右臂先让路的优先级协调路径。

## 本次 `p2a_test1` 结果

| 片段 | 输入停住帧 | 输出停住帧 | 安全直连窗口 | RRT 窗口 | 失败 |
|---|---:|---:|---:|---:|---:|
| 原始 | 40 | 0 | 5 | 0 | 0 |
| 镜像 | 45 | 0 | 4 | 1 | 0 |

镜像片段的 RRT 窗口在固定随机种子下使用 1054 次迭代、412 个树节点找到路径，随后
shortcut 将 11 个路点缩减为 3 个。最终实体网格接触为 0，5 mm 安全余量违规为 0。

后续针对视频约 5.4–6.6 秒的“单手前伸再收回”动作加入语义中间目标后，该窗口原始/镜像
片段的双臂同时低速帧从 37/36 降为 0/0。原片由一侧机械臂主动让路，镜像片段自动选择
对称的另一侧；窗口平均腕部误差仅由约 9.3 cm 变为 9.5 cm。

## 生成命令

先生成包含共享尺度、双腕位置、局部优化和硬安全层的数据：

```powershell
cd D:\openpi\pose2arm-main
$python = 'D:\ProgramData\Anaconda3\envs\openpi\python.exe'

& $python scripts\generate_ik_dataset.py `
  --input videos\p2a_test1.mp4 `
  --output data\p2a_test1_coupled.npz `
  --restarts 2 `
  --restarts-14d 1 `
  --max-nfev 300 `
  --max-nfev-14d 500
```

再修复双手靠近/交换时的连续停住窗口：

```powershell
& $python scripts\apply_exchange_rrt.py `
  --input data\p2a_test1_coupled.npz `
  --output data\p2a_test1_reference_locked.npz `
  --margin 0.005 `
  --padding 8 `
  --proximity 0.40 `
  --max-iterations 1800
```

查看回放：

```powershell
& $python scripts\replay_mujoco.py `
  --npz data\p2a_test1_reference_locked.npz `
  --video videos\p2a_test1.mp4 `
  --clip orig `
  --gripper-side both
```

验证安全性：

```powershell
& $python scripts\diagnose_interarm_collision.py `
  --npz data\p2a_test1_reference_locked.npz `
  --margin 0.005
```

## 当前边界

- 当前实现面向离线视频，不保证实时搜索延迟。
- RRT 只在安全直连失败时启动，避免不必要地改变人体动作。
- 输出仍必须通过硬安全层；RRT 成功不等于可以绕过最终安全检查。
- 如果未来改用 PPO 实时输出，应把该规划器放在 PPO 输出与机械臂关节命令之间，并设置
  严格的时间预算和上一周期路径缓存。
