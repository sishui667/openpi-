# 本地 OpenArm 仿真 + 远程 DGX OpenPI 流程

当前部署拓扑：

```text
本地 Windows / conda openpi
  openarm_openpi_client.py
  MuJoCo V25.4 + OpenArm 双臂 + 毛巾
       │  WebSocket :8000
       ▼
DGX 100.122.120.76 / Docker
  openpi-aloha-towel
  pi0_aloha_towel
```

## 1. DGX 端

当前 DGX 上已经启动最终服务：

```text
容器：openpi-aloha-towel
模型：pi0_aloha_towel
监听：0.0.0.0:8000
重启策略：unless-stopped
```

检查服务：

```powershell
ssh jnu@100.122.120.76 "docker ps --filter name=openpi-aloha-towel"
ssh jnu@100.122.120.76 "docker logs --tail 80 openpi-aloha-towel"
```

GB10 使用了下面两个运行参数：

```text
OPENPI_FORCE_FLOAT32=1
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

第一个参数规避 GB10 上的 `bf16 -> f16` 编译错误；后两个参数避免 JAX 一次性预占整块显存。毛巾模型权重位于 DGX 的 OpenPI 缓存中，不需要每次重新下载。

## 2. 本地先做预览测试

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\openarm_openpi_client.py --headless --steps 40
```

预览模式会加载本地 MuJoCo 场景、渲染胸部和双腕相机、调用 DGX，但不会把模型动作写入 OpenArm 目标关节。看到下面几类信息就表示链路正常：

```text
Connected to DGX OpenPI
DGX metadata: {'reset_pose': ...}
action_shape=(50, 14)
error=None
```

## 3. 本地打开可视化并执行远程动作

确认预览正常后再执行：

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\openarm_openpi_client.py --apply --contact
```

关闭 MuJoCo 窗口即可退出。建议第一次执行时保留较保守参数：

```powershell
python .\openarm_openpi_client.py `
  --apply `
  --contact `
  --policy-hz 2 `
  --action-gain 0.20 `
  --max-step-deg 2
```

参数含义：

- `--apply`：允许 DGX 动作控制本地仿真；不加时是只读预览。
- `--contact`：运行时启用 OpenArm 与毛巾碰撞。
- `--policy-hz`：向 DGX 发起推理请求的频率。当前模型单次约 0.5 秒，推荐 1.5--2 Hz。
- `--action-gain`：每次收到动作后向目标关节靠近的比例，范围 `(0,1]`。
- `--max-step-deg`：每个本地控制周期的关节目标最大变化量。
- `--headless --steps N`：不打开窗口，运行 N 个控制周期，用于联调。

## 4. 如果 8000 端口不可达

可以使用 SSH 隧道，不需要开放 DGX 的 8000 端口：

```powershell
ssh -N -L 8000:127.0.0.1:8000 jnu@100.122.120.76
```

保持该窗口运行，另开一个 PowerShell：

```powershell
conda activate openpi
cd D:\openpi\openarm_cloth
python .\openarm_openpi_client.py --host 127.0.0.1 --port 8000 --headless --steps 40
```

## 5. OpenArm 与 ALOHA 动作的适配逻辑

`pi0_aloha_towel` 的接口是每只手臂 6 个关节加 1 个夹爪，共 14 维；本地 OpenArm 是每只手臂 7 个关节加 2 个手指滑动关节，共 18 维。桥接脚本采用以下安全基线：

```text
OpenArm left  : q1 q2 q3 q4 q5 q6 | q7 | finger1 finger2
OpenPI left   : q1 q2 q3 q4 q5 q6 | gripper

OpenArm right : q1 q2 q3 q4 q5 q6 | q7 | finger1 finger2
OpenPI right  : q1 q2 q3 q4 q5 q6 | gripper
```

- OpenArm 的第 7 个关节保持当前安全目标，不直接接收 ALOHA 动作。
- ALOHA 的一个夹爪值转换为 OpenArm 的两个手指相同目标。
- 每个动作经过关节范围裁剪、比例靠近和每周期角度限幅。
- 左右关节符号默认是 `+1`；如果实机/模型坐标方向相反，可以分别调整：

```powershell
python .\openarm_openpi_client.py --apply --left-signs 1,-1,-1,1,1,1 --right-signs 1,-1,-1,1,1,1
```

这是跨机器人验证用的初始 retargeting，不等同于已经完成 OpenArm 的策略迁移。当前流程用于检查远程推理、观测格式、动作返回和仿真执行；要获得稳定的真实叠毛巾效果，还需要用 OpenArm 数据做动作坐标标定或微调。

## 6. 依赖

本地 `openpi` 环境已经安装 `websockets`，其余 MuJoCo、NumPy、GLFW、Pillow 已存在：

```powershell
conda activate openpi
python -c "import mujoco, numpy, glfw, PIL, websockets; print('dependencies ok')"
```

