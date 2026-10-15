# OpenArm 示范数据格式规范（dataset_schema.md）

> 成员 B（数据与工具）阶段一交付物。
> 本文件是 episode 数据的**唯一契约**：录什么、怎么存、怎么编号、怎么标记损坏。
> 校验脚本 `scripts/validate_episode.py`、录制脚本的字段补全都必须以本文为准。

**文档状态**：v3 草稿。表中标 `新增` 的字段**尚未实现**，计划在下一步改进 `teleop_towel_dataset.py` 时补上；标 `现有` 的字段当前已在落盘。

---

## 1. 范围与版本

### 1.1 格式串

| 格式串 | 来源 | 状态 |
|---|---|---|
| `openarm_towel_teleop_v1` | 磁盘上 2 条旧 episode | 可读，字段不全 |
| `openarm_towel_teleop_v2` | 磁盘上 1 条旧 episode；当前脚本 `make_metadata()` 仍写这个串 | 可读，字段不全 |
| `openarm_teleop_v3` | **本文档定义的冻结格式** | 待实现 |

> **坑**：当前 `teleop_towel_dataset.py` 的 `make_metadata()` 写的是 `openarm_towel_teleop_v2`，但它比磁盘上那条 v2 episode 多写一个 `motion_defaults` 段 —— 说明磁盘上 3 条 episode 都是**更早版本**的脚本录的。同一个格式串对应过不同字段集，所以校验时**不能只信格式串**，必须逐字段检查。

### 1.2 版本升级规则

- 已有 episode **永不覆盖**。修订后的数据写到新的版本目录（见 §3）。
- 增加可选字段 = 次版本号不变，消费者必须容忍未知键。
- 删除字段、改变已有字段语义或形状 = 必须换格式串。

---

## 2. 最低字段对照表

计划书阶段一要求的最小字段集，对照现状：

| 计划要求的字段 | 现有来源 | 状态 | 说明 |
|---|---|---|---|
| `timestamp` | npz `time`（仿真时间）；`events.json[].wall_time`（墙钟，仅事件） | **部分** | 缺**逐帧**墙钟时间 → `新增` `wall_time` |
| RGB 图像路径 | `images/<camera>/%06d.png` 目录约定 | **可复用** | 无显式路径字段，靠约定 + `metadata.cameras` 推导；见 §7.3 |
| 机器人关节状态 | npz `qpos`（全模型） | **可复用** | 但需索引才能切片 → `新增` `metadata.qpos_index` |
| 末端状态 | npz `ee_pose` (F,2,7) | **可复用** | 双臂 TCP 位置+姿态 |
| 目标动作 | npz `action_target` (F,18)、`ee_target` (F,2,7) | **可复用** | 关节空间与笛卡尔空间两种粒度都有 |
| 实际动作 | npz `ctrl` (F,18) | **可复用** | 注意臂列是**力矩**不是位置，见 §6.3 |
| 夹爪状态 | `action_target`/`ctrl` 的列 7,8（左）与 16,17（右） | **可复用** | 需在规范里写死索引 |
| 任务状态 | — | **必须新增** | `metadata.task` 段 |
| 成功标签 | — | **必须新增** | `metadata.task.success` |

**结论**：9 项里 6 项可直接复用，2 项（任务状态、成功标签）必须新增，1 项（逐帧时间戳）只有部分。

---

## 3. 目录结构

沿用计划书的命名规则：

```
openarm_cloth/
  episodes/                              # 新的录制输出根目录
    episode_YYYYMMDD_HHMMSS/
      metadata.json                      # 契约元数据（见 §5）
      episode.npz                        # 数值轨迹（见 §4）
      events.json                        # 操作事件流
      images/
        <camera_name>/
          000000.png                     # %06d，见 §7.3
          000001.png
          ...
  data/
    raw/                                 # 不修改的原始记录（只读）
    processed/v01/                       # 清洗/筛选后的独立版本
  datasets/
    towel_teleop/                        # 【历史路径】早期录制的毛巾 episode，保持着不动
  docs/
    dataset_schema.md                    # 本文件
```

**约定**：

- `episodes/` 是**新**录制的唯一输出根目录。`datasets/towel_teleop/` 是早期路径，其中的记录按「不得覆盖原始 episode」原则**只读保留**，不再写入。
- `data/raw/` 中的内容一律不改。任何清洗、筛选、截断都产出到 `data/processed/<版本>/`。
- episode 目录名由录制开始时刻生成，格式 `episode_%Y%m%d_%H%M%S`（旧目录多一段毫秒后缀，不强制）。

---

## 4. `episode.npz` 字段表

`F` = 帧数；`nq`/`nv`/`nu` 取自 `metadata.json`。

| 键 | 形状 | dtype | 单位 / 坐标系 | 状态 | 含义 |
|---|---|---|---|---|---|
| `time` | (F,) | float64 | 秒，**仿真时间** | 现有 | 权威时间轴，见 §7.1 |
| `wall_time` | (F,) | float64 | 秒，`time.perf_counter()` | **新增** | 逐帧墙钟，用于与外部视频/人手动作对齐 |
| `qpos` | (F, nq) | float64 | rad / m | 现有 | 全模型广义坐标 |
| `qvel` | (F, nv) | float64 | rad/s / m/s | 现有 | 全模型广义速度 |
| `ctrl` | (F, nu) | float64 | 臂：力矩 N·m；指：位置 m | 现有 | 该帧**实际下发**的控制量 |
| `action_target` | (F, 18) | float64 | 臂：关节角 rad；指：位置 m | 现有 | 该帧**期望**的目标，见 §6 |
| `ee_pose` | (F, 2, 7) | float64 | 位置 m + 四元数 wxyz，world | 现有 | 实际末端位姿，顺序 `[左, 右]` |
| `ee_target` | (F, 2, 7) | float64 | 同上 | 现有 | 目标末端位姿，顺序 `[左, 右]` |
| `towel_center` | (F, 3) | float64 | m, world | 现有 | 毛巾（flex）顶点质心；仅毛巾场景 |
| `object_center` | (F, 3) | float64 | m, world | **新增** | 被跟踪刚体 body 原点，见 §4.2 |
| `frame_gap` | (F,) | bool | — | **新增** | 该帧与前帧的仿真时间间隔超阈值，见 §8 |
| `towel_vertices` | (F, V, 3) | float64 | m, world | 现有（可选） | 仅 `--save-cloth-state` 时存在 |

### 4.1 形状纪律

- **不要假设 `nq == nv`**。实测：毛巾场景 681/681，方块场景 **25/24**（free joint 是 7 个位置、6 个速度）。一律从 `metadata.json` 读 `nq`/`nv`。
- **不要假设 `qpos` 宽度 == `action_target` 宽度**。方块场景 `nq=25`，动作仍是 18。
- `ctrl` 宽度 == `model.nu`，当前两个场景都是 18。**`nu` 目前没有写进 metadata**（见 §11 待定项），所以校验脚本暂时只能按 18 硬判。

### 4.2 被跟踪物体

- 毛巾场景：记录 `towel_center`（按 flex 名 `folding_towel_v25_4` 查找顶点批量质心）。
- 刚体场景（如方块）：记录 `object_center`，目标由 `--object-body` 指定（方块为 `droid_object`），取 `data.xpos[body]`。
- **现状缺陷**：`towel_vertices()` 写死 flex 名且在 `append()` 里无条件调用，在无毛巾的场景上会直接抛异常。补字段时必须一并改为按需查找。

---

## 5. `metadata.json` 字段表

### 5.1 现有字段

| 键 | 类型 | 说明 |
|---|---|---|
| `format` | str | 格式串，见 §1.1 |
| `scene` | str | 场景 XML 的**绝对路径** |
| `mujoco_version` | str | 录制时的 MuJoCo 版本 |
| `nq` / `nv` | int | 广义坐标/速度维度 |
| `timestep_s` | float | 物理步长（当前 0.0005） |
| `control_hz` | int | 记录频率（当前 20） |
| `robot_cloth_contact` | bool | 是否开启了机器人与布料碰撞 |
| `cameras` | list[str] | 实际保存的相机名 |
| `image_size` | [w, h] 或 null | 图像分辨率 |
| `image_stride` | int | 每隔几帧存一张图 |
| `save_cloth_state` | bool | 是否存了 `towel_vertices` |
| `motion_defaults` | dict | 遥操作默认步长等（仅新版脚本写） |
| `arm_joint_order` | dict | `{left:[7 个关节名], right:[...]}` |
| `action_layout` | str | 动作布局的文字描述，见 §6 |
| `keyboard` | dict | 键位表 |
| `frames` | int | 帧数（= 所有数组首维） |
| `duration_s` | float | `time[-1] - time[0]` |

### 5.2 新增字段

| 键 | 类型 | 说明 |
|---|---|---|
| `qpos_index` | `{left:[7], right:[7]}` int | 臂关节在 `qpos` 中的下标，来自 `ArmSpec.qpos` |
| `dof_index` | `{left:[7], right:[7]}` int | 臂关节在 `qvel` 中的下标，来自 `ArmSpec.dof` |
| `actuator_index` | `{left:[7], right:[7]}` int | 臂执行器在 `ctrl` 中的下标，来自 `ArmSpec.actuators` |
| `finger_actuator_index` | `{left:[2], right:[2]}` int | 夹爪执行器在 `ctrl` 中的下标 |
| `finger_qpos_index` | `{left:[2], right:[2]}` int | 指关节在 `qpos` 中的下标（**导出 state 必需**，见 §10） |
| `object_body` | str 或 null | 被跟踪物体 body 名（方块场景为 `droid_object`） |
| `task` | dict | 任务段，见 §5.3 |

> **为什么必须有 `qpos_index`**：当前 metadata 只存关节**名字**。消费者想在 `qpos` 里切出 7 个臂关节，必须重新加载同一个 XML 才能把名字解成下标。一旦 XML 变了或消费者没有那个文件，数据就不可用。存下标就解开了这个耦合。

### 5.3 `task` 段

```json
{
  "task_id": "single_arm_cube_lift",
  "operator": "手动录入",
  "object": "droid_object",
  "start_pose": [0.45, 0.0, 0.401],
  "goal": null,
  "success": true,
  "success_rule": "lifted = z_final > z_initial + 0.025; near_hand = |p_obj - p_finger_mid| < 0.18; success = lifted and near_hand",
  "success_detail": {
    "z_initial": 0.401,
    "z_final": 0.5123,
    "lift_m": 0.1113,
    "near_hand_m": 0.0312,
    "contact": true
  }
}
```

- `goal` 当前为 `null`：方块场景里**没有定义目标区**。唯一先例是官方 `official_openarm_mujoco/v2/pedestal/move_puck_scene.xml` 里 `goal_a`/`goal_b` 两个非碰撞可视化 site。若后续要做「移动至目标区」，应照此在场景里加 goal site 并把坐标写进这个字段。
- `success` 三态：`true` / `false` / `null`（未判定）。
- `success_rule` 存**判据原文**，保证以后改判据时旧数据仍可解释。

---

## 6. 动作布局

### 6.1 18 维索引

| 下标 | 含义 |
|---|---|
| 0–6 | 左臂 `openarm_left_joint1..7` 的目标关节角（rad） |
| 7 | 左指 `openarm_left_finger_joint1` 目标位置（m） |
| 8 | 左指 `openarm_left_finger_joint2` 目标位置（m） |
| 9–15 | 右臂 `openarm_right_joint1..7` 目标关节角（rad） |
| 16 | 右指 `openarm_right_finger_joint1` 目标位置（m） |
| 17 | 右指 `openarm_right_finger_joint2` 目标位置（m） |

### 6.2 夹爪耦合

两个指关节**在代码里被赋同一个值**（`apply_controller` 里 `action[base+7:base+9] = finger_value`），但模型里二者是 `polycoef="0 1 0 0 0"` 的 mimic 约束。因此：

> **不要把下标 7 和 8（或 16 和 17）当成两个独立通道**。它们当前恒等，是同一个自由度的冗余表示。

### 6.3 `ctrl` 与 `action_target` 的区别（重要）

| | `action_target` | `ctrl` |
|---|---|---|
| 含义 | 想要多少 | 实际发了多少 |
| 臂列 | 目标**关节角**（rad） | 目标**力矩**（N·m），已 clip 到执行器 `forcerange` |
| 指列 | 目标位置（m） | 目标位置（m） |
| 来源 | `apply_controller()` 返回值 | `data.ctrl.copy()` |

即：臂执行器在 MJCF 里是**力矩电机**（`motor_DM8009`/`DM4340`/`DM4310`），由「笛卡尔目标 → 阻尼最小二乘 IK → 重力补偿 PD」这条链转成力矩；手指执行器是**位置**执行器。做行为克隆时，`action_target` 是更干净的动作标签，`ctrl` 是真实下发的量。

### 6.4 夹爪取值范围

- 硬限位（MJCF `finger_joint` 的 `range`）：`0 … 0.044` m
- 约定：`0.044` = 张开，`0.020` = 轻夹，`0.000` = 完全闭合
- 遥操作脚本的闭合/张开键分别写 `0.000` / `0.044`（`teleop_towel_dataset.py:314,319`）

---

## 7. 时间源与帧编号

### 7.1 时间源

**权威时间轴是仿真时间 `time`**，不是墙钟。理由：仿真里物理步长固定（`timestep_s`），`time` 严格等于「已推进的物理步数 × 步长」，可精确复现；墙钟受机器负载影响。

`wall_time` 是**旁证**，用于：
- 与外部录制的视频或人体动作对齐
- 事后诊断「这一帧为什么慢了」

两者**不保证线性关系**。

### 7.2 帧编号

- **帧号 = `append()` 的调用序号**，从 0 开始，等于该帧在所有 `(F, ...)` 数组里的行下标。
- 帧号**不**存在单独字段里 —— 它就是数组下标。
- `frame_indices` 这类字段在本格式中**不存在**（那是 pose2arm 的 npz 格式，不要混淆）。

### 7.3 图像命名与对齐

- 路径 = `images/<camera_name>/` + `f"{frame_index:06d}.png"`。
- `frame_index` 是**帧号**，不是被整除后的序号 —— 只有满足 `frame_index % image_stride == 0` 的帧才落图。
- 因此某相机的图像**张数** = `ceil(F / image_stride)`，图像序列对应帧号 `0, stride, 2·stride, …`。
- 图像是 RGB、无 alpha、`.png`，尺寸 = `metadata.image_size` 的 `[w, h]`。
- **缺图不等于缺帧**：图像按 stride 抽稀是设计行为，不是损坏。判断要按上面公式算期望张数。

### 7.4 采样节拍

记录由仿真时间驱动：主循环每 tick 推进 `steps_per_tick × timestep` 的仿真时间，当 `data.time >= sample_time` 时记一帧，然后 `sample_time += 1/control_hz`。

理想情况下相邻帧的 `time` 间隔恰为 `1/control_hz`（当前 0.05 s）。**实际不保证**，见 §8。

---

## 8. 缺帧 / 缺状态标记

### 8.1 缺帧

因为采样按「`data.time` 追上 `sample_time`」触发，若某个 tick 超时（机器卡顿、渲染慢），下一帧的 `time` 间隔会**大于** `1/control_hz`。当前代码对此**没有任何检测**。

**标记规则**：

1. 令 `dt_i = time[i] - time[i-1]`，`expected = 1 / control_hz`。
2. 若 `dt_i > 1.5 × expected`，则 `frame_gap[i] = true`，表示「第 i 帧之前丢了东西」。
3. `frame_gap[0] = false`（首帧无前驱）。
4. `metadata` 汇总 `gap_count`（次数）与 `gap_frames`（帧号列表）。

**语义**：`frame_gap[i] = true` 表示**时间轴上有真实跳跃**，第 `i-1` 帧与第 `i` 帧之间在仿真里不是连续的 1 拍。做行为克隆时，跨越 gap 的 `(obs, action)` 对**不应**当成相邻转移。

> 注意区分两件事：`frame_gap` 标记的是**时间轴跳跃**（真丢失）；图像的 stride 抽稀是**设计行为**（不是丢失）。二者互不影响。

### 8.2 缺状态

本格式中所有状态数组与 `time` **等长且同下标**，不存在「某帧缺某个字段」的情况。因此：

- 若某帧确实取不到状态 → **不记这一帧**，让时间轴自然产生 gap，由 §8.1 标记。**不允许**用 NaN 占位。
- 校验时任何数组出现 NaN / Inf 都判 **FAIL**，视同损坏（见校验项 §9）。

### 8.3 缺图

- 设计上按 stride 抽稀，按 §7.3 公式算期望张数。
- 期望存在的文件缺失 → **FAIL**。
- 存在但尺寸与 `image_size` 不符 → **FAIL**。
- 存在但没有对应帧号（多出文件）→ **WARN**。

---

## 9. 校验规则（供 `scripts/validate_episode.py` 实现）

| # | 检查 | 级别 |
|---|---|---|
| 1 | `metadata.json` 存在、可解析、`format` 已知 | FAIL |
| 2 | `episode.npz` 存在、可加载、必需键齐全 | FAIL |
| 3 | 所有数组首维一致且 == `metadata.frames` | FAIL |
| 4 | `time` 单调不减；按 §8.1 标出 gap | WARN |
| 5 | `duration_s` 与 `time` 自洽 | WARN |
| 6 | `qpos`/`qvel` 宽度 == `metadata.nq`/`nv` | FAIL |
| 7 | `action_target`/`ctrl` 宽度 == 18；夹爪列落在 `0…0.044` | FAIL |
| 8 | 全数组无 NaN / Inf | FAIL |
| 9 | 每个相机图像数 == `ceil(frames/stride)`，文件存在且尺寸 == `image_size` | FAIL |
| 10 | `events.json` 可解析、`wall_time` 不减 | WARN |
| 11 | （需 `--scene`）臂关节 `qpos` 落在 `jnt_range` 内 | WARN |

有 FAIL → 退出码非 0，并写 `quality_report.json`。

---

## 10. 导出目标：LeRobot / π0.5 微调

### 10.1 为什么有这一节

openpi 的训练入口是 **LeRobot 数据集格式**，不是本格式的 npz。本 episode 是**转换脚本的输入**，不是训练的直接输入：

```
episode.npz + images/   →   转换脚本   →   LeRobotDataset   →   compute_norm_stats   →   微调 π0.5
```

因此下面这些取值必须在**采集端冻结**。官方文档明确要求训练与部署两侧一致，事后改要重导数据。

### 10.2 五个冻结项

| 项 | 本规范的取值 | 备注 |
|---|---|---|
| 相机键名 | `v22_left_d405_rgb`、`v22_right_d405_rgb`、`chest_d435i_preview` | 训练与部署必须**同名同数量** |
| state 维度与含义 | **18 维关节位置**，见 §10.3 | 逐维物理含义已定 |
| action 语义 | **绝对关节角**（非增量），见 §10.4 | |
| 夹爪归一化 | `g_m / 0.044 → 0…1`，见 §10.5 | |
| task 语言指令 | 必须新增 `task.prompt`，见 §10.6 | 当前缺失 |

### 10.3 state 向量（18 维）

`observation.state` 从 `qpos` 按下列顺序切出，与 §6.1 的 `action_target` **逐位对齐**：

| 下标 | 来源 | 含义 |
|---|---|---|
| 0–6 | `qpos[qpos_index.left]` | 左臂关节角（rad） |
| 7–8 | `qpos[finger_qpos_index.left]` | 左指位置（m） |
| 9–15 | `qpos[qpos_index.right]` | 右臂关节角（rad） |
| 16–17 | `qpos[finger_qpos_index.right]` | 右指位置（m） |

> **这正是要 `qpos_index` / `finger_qpos_index` 的原因**：没有它们，转换脚本必须重新加载同一个 XML 才能从 `qpos`（方块场景 25 维、毛巾场景 681 维）里切出这 18 维。

### 10.4 绝对还是增量（重要）

本规范存的是**绝对关节角**。这是刻意的 —— 绝对量可以事后求差得到增量，反之不行。

但转换时**必须显式记录**用了哪一种，并让训练配置一致，否则训练和部署对不上。

> ⚠️ **警告**：远程 `pi05_droid` 那条路径的契约是「7 关节**速度** + 1 夹爪」，与这里的「双臂 18 维绝对关节**位置**」**不是一回事**。本规范选定的是后者，对应 `pi05_base` / `pi0_base` 微调，**不能**直接套用 `pi05_droid` 的配置。

### 10.5 夹爪

- 原始存储：米，`0 … 0.044`
- 归一化到 0–1：`g_norm = g_m / 0.044`
- 两个指关节是 mimic 耦合（§6.2），值恒等。本规范**保留 2 维**，故动作/状态空间为 **18**。若要压成 1 维（变 16），必须换格式串并同步改 §10.3。

### 10.6 task 语言指令

必须新增 `metadata.task.prompt`，存自然语言指令原文，例如：

```json
"prompt": "pick up the red cube"
```

现有的 `task_id`（`single_arm_cube_lift`）是机器标识，**不能替代**语言指令 —— LeRobot 的 `add_frame` 每一帧都要一个 task 字符串，π0.5 是靠它做条件生成的。

### 10.7 归一化：π0.5 用分位数

π0.5 默认 **分位数归一化（q01/q99）**，不是 π0 用的均值方差：

```
x_norm = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1
```

norm stats 由 `scripts/compute_norm_stats.py` 计算。**注意坑**：算出来不等于训练会用它 —— 还得在训练 config 的 `AssetsConfig` 里指过去，否则训的是旧 stats。

### 10.8 采集量目标

LeRobot 的建议规模是 **600+ 帧**级别。现有 3 条 episode 是 **19~37 帧**，差一个数量级。

按 20 Hz 换算：**600 帧 = 单条 30 秒**。

→ 采集脚本的设计目标应是「**单条 ≥ 30 秒、总量到几千帧**」，而不是现在这种 1~2 秒的试录。

### 10.9 本节尚未在本机验证

§10.1–§10.8 的 openpi / LeRobot 细节来自 openpi 官方 README 与实践指南，**未在本机核实**（本机没有 openpi 训练源码：`openpi.zip` 只是外层文件夹的重新打包，其中 lerobot 相关条目为 0）。真正落实要等能访问 DGX 上的 openpi 仓库 —— 按现有约定是假期之后。在此之前，本节的取值应视为**待确认**，不要据此冻结格式串。

---

## 附录 A：OpenArm 侧接口

均为 `droid_cube_scene.xml` / `folding_scene_V25_4_TOWEL_SUPPORT_04.xml` 实测值。

### A.1 命名约定

```
关节    openarm_{left|right}_joint1..7
手指    openarm_{left|right}_finger_joint1..2
末端    openarm_{left|right}_hand_tcp          ← 记录 EE 位姿用的 body
臂执行器  {left|right}_joint1..7_ctrl
指执行器  {left|right}_finger1..2_ctrl
```

### A.2 关节限位（rad）

| 关节 | 左臂 | 右臂 |
|---|---|---|
| joint1 | `-3.4907 … 1.3963` | `-1.3963 … 3.4907` |
| joint2 | `-3.3161 … 0.1745` | `-0.1745 … 3.3161` |
| joint3 | `-1.5708 … 1.5708` | 同左 |
| joint4 | `0.0 … 2.4435` | 同左 |
| joint5 | `-1.5708 … 1.5708` | 同左 |
| joint6 | `-0.7854 … 0.7854` | 同左 |
| joint7 | `-1.5708 … 1.5708` | 同左 |
| finger_joint1/2 | `0.0 … 0.044` | 同左 |

注意 joint1/joint2 左右**不对称**（左臂关节2上限仅 0.1745 rad）。

### A.3 电机与控制器

| 关节 | 电机类 | 力矩范围 |
|---|---|---|
| joint1–2 | `motor_DM8009` | ±40 |
| joint3–4 | `motor_DM4340` | ±27 |
| joint5–7 | `motor_DM4310` | ±7 |
| 手指 | position，`kp=100` | `ctrlrange 0…0.044` |

臂用重力补偿 PD（`kp=[80,80,55,55,25,20,20]`，`kd=[8,8,5,5,3,2,2]`），输出力矩 clip 到 `forcerange`。

### A.4 控制频率与安全盒

- 记录频率 `control_hz = 20`；物理步长 `timestep_s = 0.0005`。
- Cartesian 安全盒（`safety_reason()`，末端超出即暂停物理）：`x ∈ [0.15, 0.80]`，`y ∈ [-0.50, 0.50]`，`z ∈ [0.08, 0.85]`。

### A.5 相机

场景共 5 个：`front_camera`、`side_camera`、`v22_left_d405_rgb`、`v22_right_d405_rgb`、`chest_d435i_preview`。默认录制后三个。

---

## 附录 B：冻结的单臂方块任务定义

**任务**：单臂抓取红色方块并抬起。

| 项 | 值 | 出处 |
|---|---|---|
| 场景 | `simulation/v1/droid_cube_scene.xml` | — |
| 物体 body | `droid_object`（freejoint） | `droid_cube_scene.xml:305` |
| 方块尺寸 | 半边长 0.0288 m（边长 57.6 mm） | 同上 |
| 质量 | 0.18 kg | 同上 |
| 起始位置 | `0.45 0.0 0.401` | `droid_cube_scene.xml:305`（也写在 keyframe `towel_ready`） |
| 抓取 weld | `left_grasp_weld` / `right_grasp_weld`，`active=false` | `droid_cube_scene.xml:332-333` |
| 抬起高度 | `--lift-height` 默认 0.12 m | `handeye_grasp_red_cube.py:1810` |
| 目标区 | **无** | 场景里没有任何 goal/region 标记 |

**成功判据**（`handeye_grasp_red_cube.py:1735-1737`）：

```python
lifted    = float(final_object[2]) > initial_object_z + 0.025   # 抬起 > 25 mm
near_hand = float(np.linalg.norm(final_object - finger_midpoint)) < 0.18   # 距离指中点 < 18 cm
success   = bool(lifted and near_hand)
```

其中 `finger_midpoint = 0.5*(geom_xpos[f0] + geom_xpos[f1])` 为两指碰撞几何中心的中点。

> **注意**：同一判据在 `openarm_droid_client.py:543-561` 的 `GraspMonitor.is_grasped` 里有**第二份拷贝**，且多一个 `gripper_closed or 接触` 条件；毛巾任务用的是放宽版（`+0.012` 和 `0.24`）。写入 `task.success_rule` 时必须记清楚用的是哪一份，否则数据无法互相解释。

### B.1 与「起始区 → 目标区」的差距

计划书阶段一写的最小任务是「单臂将方块从起始区移动至目标区」。当前场景**只有起点、没有终点**。若要严格贴合，需要在场景里新增 goal site（参考 `official_openarm_mujoco/v2/pedestal/move_puck_scene.xml:72-78` 的 `goal_a`/`goal_b` 可视化圆柱 site 写法），并同步：判据从「抬起高度」改为「方块进入目标区半径内」。**这是一个待定项，见 §11。**

---

## 11. 待定项

### 11.1 已定

- **微调目标动作空间 = 双臂 18 维绝对关节位置**（见 §10.2–§10.4），对应 `pi05_base` / `pi0_base`，不走 `pi05_droid` 那条契约。

### 11.2 待确认（需与成员 A / C 定）

1. **`nu` 是否写进 metadata** —— 现在 `ctrl` 宽度没有对应的元数据可校验。建议加。
2. **成功判据以哪一份为准** —— `handeye_grasp_red_cube.py` 版还是 `openarm_droid_client.py` 版（后者多夹爪条件）。三处拷贝应统一成一处并被引用。
3. **最小任务是否要真正的目标区** —— 计划书写的是「移动至目标区」，现状是「抓取并抬起」。若要用前者，需加 goal site 并改判据（参考 §附录 B）。
4. **`task.prompt` 的措辞** —— §10.6 要求自然语言指令，但措辞属于任务定义（成员 C）。需要一份 prompt 清单，且训练与部署用同一句。
5. **单条采集时长目标** —— §10.8 建议单条 ≥ 30 秒。需确认是否按此改采集脚本设计。
6. **§10 全部取值的最终确认** —— 受 §10.9 限制，等能访问 DGX 上的 openpi 仓库后再定稿，**在此之前不要据此冻结格式串**。

### 11.3 已建议（不阻塞）

- **旧 episode 不迁移** —— 磁盘上 3 条 v1/v2 episode 缺新字段，保持只读，新数据一律 `v3`。
- **`towel_center` 与 `object_center` 并存** —— 毛巾场景写前者，刚体场景写后者，都写也无害。
