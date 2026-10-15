# OpenArm 单臂 DROID 联调流程

本目录现在保留两条远程策略流程：

```text
openarm_openpi_client.py  -> pi0_aloha_towel  -> 双臂 14 维 ALOHA 动作
openarm_droid_client.py   -> pi05_droid       -> 单臂 7 关节速度 + 1 夹爪
```

本文件对应第二条流程。DROID 是单臂策略，因此本地场景仍然保留双臂 OpenArm，但只有 `--arm` 指定的手臂接收远程动作，另一只手臂保持安全目标。DROID 默认使用刚性红色方块测试，避免通用单臂策略一开始就面对柔性毛巾。

## 1. 启动 DGX 服务

在本地 PowerShell 执行：

```powershell
cd D:\openpi
.\openarm_cloth\deploy_pi05_droid_remote.ps1
```

脚本会：

1. 停止原来的 `openpi-aloha-towel` 服务，但不会删除它；
2. 启动 `openpi-droid` 容器；
3. 使用共享缓存下载 `pi05_droid`；
4. 在 DGX 的 `8000` 端口提供 WebSocket 推理服务。

切换回原来的 ALOHA 毛巾服务时，使用：

```powershell
ssh jnu@100.122.120.76 "docker stop openpi-droid; docker start openpi-aloha-towel"
```

原来的 `openpi-aloha-towel` 容器只被停止，没有删除。

## 2. 本地预览联调

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\openarm_droid_client.py --headless --steps 60
```

预览模式会连接远程模型并执行 MuJoCo，但不会使用模型动作移动目标关节。正常输出应包含：

```text
Connected to DGX OpenPI
action_shape=(15, 8)
error=None
```

## 3. 单臂仿真测试

第一次建议只控制左臂、启用毛巾接触，并降低速度：

```powershell
python .\openarm_droid_client.py `
  --arm left `
  --object cube `
  --prompt "pick up the red cube" `
  --apply `
  --contact `
  --velocity-scale 1.2
```

右臂测试：

```powershell
python .\openarm_droid_client.py --arm right --apply --contact --velocity-scale 1.2
```

可用测试物体：

```powershell
# 红色方块，默认推荐
python .\openarm_droid_client.py --object cube --prompt "pick up the red cube" --apply --contact

# 蓝色瓶子
python .\openarm_droid_client.py --object bottle --prompt "pick up the blue bottle" --apply --contact

# 黄色球
python .\openarm_droid_client.py --object ball --prompt "pick up the yellow ball" --apply --contact

# 恢复原来的毛巾场景
python .\openarm_droid_client.py --object towel --prompt "pick up the towel" --apply --contact
```

MuJoCo 窗口快捷键：

```text
Space  暂停/继续远程 DROID 动作
P      暂停/继续物理仿真
R      恢复 towel_ready 姿态
```

窗口出现异常运动时，先按 `Space`，再按 `R`。关闭窗口即可退出。

## 4. 动作适配逻辑

```text
DROID observation/joint_position       7 维 OpenArm 关节角
DROID observation/gripper_position     1 维，0=闭合，1=张开
DROID actions                         7 维归一化关节速度 + 1 维夹爪
                                      |
                                      v
                               速度 × velocity_scale
                                      |
                                      v
                          积分为 OpenArm 关节目标角
                                      |
                                      v
                   MuJoCo 重力补偿 PD + 关节范围安全裁剪
```

`--velocity-scale` 的单位是“DROID 归一化速度为 1 时的 OpenArm 关节弧度/秒”。建议范围：

```text
0.8  初次测试
1.2  默认推荐
1.5  较快，但需要确认相机和关节符号
```

如果某个关节方向相反，可以分别传入 7 个符号，例如：

```powershell
python .\openarm_droid_client.py --apply --left-signs 1,-1,-1,1,1,1,1
```

## 5. 当前验证边界

这一步验证的是：

- DGX 的 DROID 权重能否启动；
- 本地 OpenArm 是否能正确发送 DROID 观察格式；
- 服务是否返回 8 维单臂动作；
- 速度动作是否能安全转换为 OpenArm 关节目标；
- MuJoCo 物理和安全暂停是否工作。

它不等于已经完成 OpenArm 上的真实叠毛巾。DROID 是通用单臂模型，OpenArm 的相机视角、关节范围、夹爪形状和毛巾场景都与训练数据不同。若单臂基础任务稳定，下一步应采集 OpenArm 原生数据，再用 `pi05_base` 或 `pi0_base` 微调双臂 18 维策略。
