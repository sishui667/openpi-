# -*- coding: utf-8 -*-
r"""在 MuJoCo 中回放 pose2arm 生成的双臂关节角（目标 b：桥接脚本）。

把 data/rl_poses_output.npz 中的 (2N, 14) joint_angles 逐帧喂给
openarm_bimanual.xml 双臂模型，让机械臂在仿真里跟随人体动作。

功能:
  - 可选 --video: OpenCV 窗口同步播放原视频，与 MuJoCo 窗口并排对比
    （按 npz 里的 frame_indices 帧级对齐; mirror 片段自动水平翻转视频）
  - 场景美化: 仅替换背景色（skybox 渐变）+ 增强灯光，不加地面
    （写入原 XML 同目录 openarm_bimanual_replay.xml; --plain 关闭）

用法（pose2arm-main 根目录下，openpi 环境）:
    python scripts\replay_mujoco.py --video videos\waving_demo.mp4
    python scripts\replay_mujoco.py --npz data\xxx.npz --mirror
    python scripts\replay_mujoco.py --dynamic --stride 2 --speed 0.5
    python scripts\replay_mujoco.py --xml 路径\场景.xml --plain
    python scripts\replay_mujoco.py --gripper auto --gripper-side right   # 夹爪随视频开合

键位: 视频窗口按 q 退出; 关闭任一窗口即结束; 循环播放直到手动关闭。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = (REPO_ROOT.parent /
               "openarm_cloth_with_dependencies_20260929/openarm_cloth/"
               "official_openarm_mujoco/v2/openarm_bimanual.xml")
DEFAULT_VIDEO = REPO_ROOT / "videos" / "waving_demo.mp4"

JOINT_NAMES = [f"openarm_{side}_joint{i}" for side in ("left", "right")
               for i in range(1, 8)]

# 夹爪关节映射（实测验证: qpos=0 指尖间距 9.1mm=全闭合, 极值=152.5mm 全张开）
GRIPPER_OPEN_QPOS  = {"left": 0.7854, "right": -0.7854}   # 全张开
GRIPPER_CLOSE_QPOS = {"left": 0.0,    "right": 0.0}       # 全闭合
GRIPPER_COL = {"right": 0, "left": 1}  # gripper_cmd 列序 [右臂, 左臂]

CV_WIN = "pose2arm replay - source video (q to quit)"


def physics_steps_for_frame(index, period, timestep):
    """Carry fractional steps across frames instead of advancing just once."""
    if index < 0 or not np.isfinite(period) or not np.isfinite(timestep) or period <= 0 or timestep <= 0:
        raise ValueError("invalid physics clock")
    return int(np.floor((index+1)*period/timestep+1e-9)-np.floor(index*period/timestep+1e-9))


def validate_position_actuators(model, ids):
    for aid in ids:
        if model.actuator_gainprm[aid,0] <= 0 or model.actuator_biasprm[aid,1] >= 0:
            raise ValueError("A dynamic replay requires POSITION servos; use B driver for torque-controlled models")

# ── 场景美化: 仅注入 skybox 背景色 + 灯光（无地面） ───────────────────────────
# 配色方案: key = --bg 参数值。机械臂材质亮度: matte_black≈0.25, silver≈0.65-0.75
#   blue  饱和蓝(默认): 无彩色臂身与蓝色背景形成色相差异, 黑/银件双向清晰
#   light 浅灰白: 黑色部件对比最强, 适合看 matte_black 壳体
#   dark  深蓝黑: 银色部件对比最强, 高光立体感最好
#   gray  中性灰: 折中方案

_THEMES = {
    "blue":  ("0.32 0.40 0.58", "0.20 0.25 0.40",
              "0.30 0.30 0.30", "0.90 0.90 0.90", "0.35 0.35 0.35"),
    "light": ("0.82 0.84 0.88", "0.68 0.71 0.76",
              "0.45 0.45 0.45", "0.75 0.75 0.75", "0.15 0.15 0.15"),
    "dark":  ("0.16 0.18 0.24", "0.08 0.09 0.12",
              "0.22 0.22 0.22", "0.95 0.95 0.95", "0.40 0.40 0.40"),
    "gray":  ("0.52 0.55 0.60", "0.40 0.42 0.47",
              "0.25 0.25 0.25", "0.85 0.85 0.85", "0.25 0.25 0.25"),
}


def make_scenic_xml(xml_path: Path, theme: str = "blue") -> Path:
    """读取原始 XML，按配色主题注入背景色与灯光，生成美化版场景文件（同目录）。"""
    sky_top, sky_bot, amb, diff, spec = _THEMES[theme]
    asset = (
        f'    <texture name="replay_sky" type="skybox" builtin="gradient" '
        f'rgb1="{sky_top}" rgb2="{sky_bot}" width="512" height="512"/>\n'
    )
    visual = (f'  <visual>\n'
              f'    <headlight ambient="{amb}" diffuse="{diff}" specular="{spec}"/>\n'
              f'    <map znear="0.01"/>\n'
              f'    <quality shadowsize="2048"/>\n'
              f'  </visual>\n')

    text = xml_path.read_text(encoding="utf-8")
    if "</asset>" in text:
        text = text.replace("</asset>", asset + "  </asset>", 1)
    else:
        text = text.replace("</mujoco>", "  <asset>\n" + asset + "  </asset>\n</mujoco>", 1)
    text = text.replace("</mujoco>", visual + "</mujoco>", 1)

    out = xml_path.with_name(xml_path.stem + "_replay.xml")
    out.write_text(text, encoding="utf-8")
    return out


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_frames(npz_path: Path, clip: str) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """读取 npz，返回待回放的 (F,14) 帧序列、对应原视频帧号、源 fps、夹爪通道 (F,2) 或 None。"""
    d = np.load(npz_path)
    joint_angles = np.asarray(d["joint_angles"], dtype=float)
    if joint_angles.ndim != 2 or joint_angles.shape[1] != 14:
        raise ValueError(f"joint_angles 形状应为 (F,14)，实际 {joint_angles.shape}")
    fids = np.asarray(d["frame_indices"], dtype=int)
    n = len(joint_angles) // 2
    if clip == "orig":
        frames, seg_fids = joint_angles[:n], fids[:n]
    elif clip == "mirror":
        frames, seg_fids = joint_angles[n:], fids[n:]
    else:  # all
        frames, seg_fids = joint_angles, fids
    grip = None
    grip_key = "gripper_position" if "gripper_position" in d.files else "gripper_cmd"
    if grip_key in d.files:
        g = np.asarray(d[grip_key], dtype=float)
        if g.ndim == 2 and g.shape[0] == len(joint_angles):
            grip = g[:n] if clip == "orig" else (g[n:] if clip == "mirror" else g)
        else:
            print(f"[WARN] {grip_key} 形状 {g.shape} 与 joint_angles 不匹配，忽略")
    return frames, seg_fids, float(d["fps"]), grip


def joint_qpos_ids(model, names: list[str]) -> np.ndarray:
    """按名字解析 14 个关节的 qpos 地址，缺失即报错。"""
    ids = []
    for name in names:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise ValueError(f"模型中找不到关节 {name!r}")
        ids.append(int(model.jnt_qposadr[jid]))
    return np.array(ids)


def finger_qpos_ids(model, side: str) -> np.ndarray:
    """该侧夹爪两个 finger 关节的 qpos 地址（mimic 联动，运动学模式两关节同值直写）。"""
    ids = []
    for j in (1, 2):
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"openarm_{side}_finger_joint{j}"))
        if jid < 0:
            raise ValueError(f"模型中找不到夹爪关节 openarm_{side}_finger_joint{j}")
        ids.append(int(model.jnt_qposadr[jid]))
    return np.array(ids)


def finger_actuator_id(model, side: str) -> int | None:
    """该侧夹爪位置执行器 id（{side}_finger1_ctrl），缺失返回 None。"""
    aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"{side}_finger1_ctrl"))
    return aid if aid >= 0 else None


def clip_to_limits(frames: np.ndarray, model, qpos_ids: np.ndarray) -> np.ndarray:
    """把关节角裁剪到模型关节限位内，返回裁剪后的副本并打印统计。"""
    lim_lo = np.zeros(len(qpos_ids))
    lim_hi = np.zeros(len(qpos_ids))
    for k, adr in enumerate(qpos_ids):
        jid = int(np.flatnonzero(model.jnt_qposadr == adr)[0])
        lim_lo[k], lim_hi[k] = model.jnt_range[jid]
    n_clip = int(((frames < lim_lo) | (frames > lim_hi)).sum())
    if n_clip:
        print(f"[WARN] {n_clip}/{frames.size} 个关节角超限，已裁剪到模型限位")
    return np.clip(frames, lim_lo, lim_hi)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path,
                        default=REPO_ROOT / "data" / "rl_poses_output.npz",
                        help="pose2arm 输出的 npz 路径")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML,
                        help="MuJoCo 模型 XML（需含 openarm_left/right_joint1..7）")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help="源视频路径（配合 npz 帧号同步播放对比窗口; 置空字符串禁用）")
    parser.add_argument("--clip", choices=["orig", "mirror", "all"], default="orig",
                        help="回放片段: orig=原始(前 N), mirror=镜像(后 N), all=全部")
    parser.add_argument("--dynamic", action="store_true",
                        help="物理模式: position 控制器 + mj_step（默认纯运动学 qpos 直写）")
    parser.add_argument("--stride", type=int, default=1,
                        help="帧抽稀步长（>1 加速回放，默认 1）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="回放倍速（默认 1.0）")
    parser.add_argument("--bg", choices=list(_THEMES), default="blue",
                        help="背景配色主题: blue=饱和蓝(默认) light=浅灰白 dark=深蓝黑 gray=中灰")
    parser.add_argument("--plain", action="store_true",
                        help="禁用背景美化，直接用原 XML")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="循环播放直到关闭窗口（默认开）")
    parser.add_argument("--gripper", choices=["auto", "open", "close"], default="auto",
                        help="夹爪: auto=跟随 npz gripper_cmd(无通道则常开); "
                             "open/close=强制常开/常闭（调试用）")
    parser.add_argument("--gripper-side", choices=["right", "left", "both"], default="right",
                        help="驱动哪侧夹爪（默认 right，单臂右手任务）")
    parser.add_argument("--gripper-ramp", type=float, default=None,
                        help="夹爪全行程斜坡时间（秒）；连续安全通道默认不二次限速，旧二值通道默认 0.15")
    args = parser.parse_args()

    if not args.npz.is_file():
        raise SystemExit(f"npz 不存在: {args.npz}")
    if not args.xml.is_file():
        raise SystemExit(f"XML 不存在: {args.xml}")

    frames, fids, src_fps, grip = load_frames(args.npz, args.clip)
    continuous_grip = "gripper_position" in np.load(args.npz).files
    gripper_ramp = args.gripper_ramp
    if gripper_ramp is None:
        gripper_ramp = 0.0 if continuous_grip else 0.15
    frames = frames[::max(1, args.stride)]
    fids = fids[::max(1, args.stride)]
    if grip is not None:
        grip = grip[::max(1, args.stride)]
    dt = args.stride / (src_fps * args.speed)  # 每帧目标停留时长

    # 场景美化
    xml_path = args.xml
    if not args.plain:
        try:
            xml_path = make_scenic_xml(args.xml, args.bg)
            print(f"[SCENE] 配色主题={args.bg}: 背景色已替换 → {xml_path.name}")
        except Exception as e:  # 美化失败不阻塞回放
            print(f"[WARN] 场景美化失败 ({e})，回退原始 XML")
            xml_path = args.xml

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qpos_ids = joint_qpos_ids(model, JOINT_NAMES)
    frames = clip_to_limits(frames, model, qpos_ids)

    # 物理模式: 通过 trnid 建立 qpos 地址 → 执行器 的映射
    ctrl_ids = None
    if args.dynamic:
        adr_to_actuator = {int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])]): a
                           for a in range(model.nu)}
        ctrl_ids = np.array([adr_to_actuator.get(int(adr), -1) for adr in qpos_ids])
        if (ctrl_ids < 0).any():
            missing = [JOINT_NAMES[k] for k, c in enumerate(ctrl_ids) if c < 0]
            raise SystemExit(f"以下关节无对应执行器，--dynamic 不可用: {missing}")
        validate_position_actuators(model, ctrl_ids)

    # ── 夹爪配置 ────────────────────────────────────────────────────────────────
    driven = ["right", "left"] if args.gripper_side == "both" else [args.gripper_side]
    grip_qpos = {s: finger_qpos_ids(model, s) for s in driven}
    grip_ctrl = {s: finger_actuator_id(model, s) for s in driven}
    if args.dynamic and any(grip_ctrl[s] is None for s in driven):
        missing = [s for s in driven if grip_ctrl[s] is None]
        raise SystemExit(f"以下侧夹爪无位置执行器，--dynamic 下无法驱动: {missing}")
    if args.dynamic:
        validate_position_actuators(model, [grip_ctrl[s] for s in driven])

    # 未驱动侧夹爪初始化为全张开（qpos/ctrl 默认 0 = 全闭合，避免静止臂显示为捏死）
    for s in ("right", "left"):
        if s not in driven:
            data.qpos[finger_qpos_ids(model, s)] = GRIPPER_OPEN_QPOS[s]
            if args.dynamic:
                aid = finger_actuator_id(model, s)
                if aid is not None:
                    data.ctrl[aid] = GRIPPER_OPEN_QPOS[s]

    if args.gripper == "open":
        grip_desc = "强制常开"
    elif args.gripper == "close":
        grip_desc = "强制常闭"
    elif grip is not None:
        grip_desc = "跟随连续 gripper_position（旧数据兼容 gripper_cmd）"
    else:
        grip_desc = "npz 无 gripper_cmd 通道 → 常开"
    print(f"[GRIPPER] side={args.gripper_side}  mode={args.gripper}({grip_desc})  "
          f"ramp={gripper_ramp}s")

    def gripper_target(side: str, i: int) -> float:
        """该回放帧的夹爪 qpos/ctrl 目标。"""
        if args.gripper == "open":
            return GRIPPER_OPEN_QPOS[side]
        if args.gripper == "close":
            return GRIPPER_CLOSE_QPOS[side]
        if grip is not None:
            closure = float(np.clip(grip[i % len(grip), GRIPPER_COL[side]], 0.0, 1.0))
            return (GRIPPER_OPEN_QPOS[side]
                    + closure * (GRIPPER_CLOSE_QPOS[side] - GRIPPER_OPEN_QPOS[side]))
        return GRIPPER_OPEN_QPOS[side]

    # 斜坡限速: 全行程 / ramp 秒 → 每回放帧最大步长
    grip_step = {s: abs(GRIPPER_OPEN_QPOS[s] - GRIPPER_CLOSE_QPOS[s])
                 / max(1.0, gripper_ramp * src_fps / max(1, args.stride))
                 for s in driven}
    grip_pos = {s: GRIPPER_OPEN_QPOS[s] for s in driven}  # 初始张开
    grip_switches = {s: 0 for s in driven}
    grip_prev = {s: None for s in driven}

    # 视频对比窗口
    cap = None
    cap_next_fid = 0
    if args.video and Path(args.video).is_file():
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频 {args.video}，跳过对比窗口")
            cap = None
        else:
            cv2.namedWindow(CV_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(CV_WIN, 640, 360)
            print(f"[VIDEO] 对比窗口已开启: {args.video}（按 q 退出）")
    elif args.video:
        print(f"[WARN] 视频不存在: {args.video}，跳过对比窗口")

    print(f"[DATA] {args.npz.name} clip={args.clip} frames={len(frames)} "
          f"(src {src_fps:.0f}fps, stride {args.stride}, speed {args.speed}x)")
    print(f"[MODEL] {xml_path}")
    print(f"[MODE] {'dynamic (position ctrl + mj_step)' if args.dynamic else 'kinematic (qpos + mj_forward)'}")

    mirror = args.clip in ("mirror", "all")
    physical_frame = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.30]
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.sync()

        while viewer.is_running():
            stop = False
            for i, frame in enumerate(frames):
                if not viewer.is_running():
                    stop = True
                    break
                t0 = time.perf_counter()

                if args.dynamic:
                    data.ctrl[ctrl_ids] = frame
                else:
                    data.qpos[qpos_ids] = frame

                # 夹爪驱动（斜坡限速趋近目标）
                for s in driven:
                    tgt = gripper_target(s, i)
                    grip_pos[s] += float(np.clip(tgt - grip_pos[s],
                                                 -grip_step[s], grip_step[s]))
                    if args.dynamic:
                        data.ctrl[grip_ctrl[s]] = grip_pos[s]
                    else:
                        data.qpos[grip_qpos[s]] = grip_pos[s]  # joint1/2 同值(mimic)
                    if grip_prev[s] is not None and tgt != grip_prev[s]:
                        grip_switches[s] += 1
                    grip_prev[s] = tgt

                # Write arm AND gripper commands before the physics interval.
                if args.dynamic:
                    for _ in range(physics_steps_for_frame(physical_frame, dt, model.opt.timestep)):
                        mujoco.mj_step(model, data)
                    physical_frame += 1
                else:
                    mujoco.mj_forward(model, data)

                viewer.sync()

                # 同步视频帧（按 npz 记录的原视频帧号对齐）
                if cap is not None:
                    fid = int(fids[i % len(fids)])
                    # Normal playback is sequential. Avoid a costly random seek on
                    # every frame; seek only for a loop restart, mirror/all boundary,
                    # or deliberately sparse/non-consecutive frame indices.
                    if fid != cap_next_fid:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    ok, img = cap.read()
                    cap_next_fid = fid + 1
                    if ok:
                        if mirror and args.clip == "mirror":
                            img = cv2.flip(img, 1)  # 镜像动作 ↔ 水平翻转视频
                        label = f"frame {fid}/{len(fids)}  clip={args.clip}"
                        cv2.putText(img, label, (12, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow(CV_WIN, img)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        stop = True
                        break

                elapsed = time.perf_counter() - t0
                if dt > elapsed:
                    time.sleep(dt - elapsed)
            if stop or not args.loop:
                break

    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    print("[DONE] 回放结束")
    sw = ", ".join(f"{s}: {grip_switches[s]} 次" for s in driven)
    print(f"[GRIPPER] 连续目标更新统计: {sw}")


if __name__ == "__main__":
    main()
