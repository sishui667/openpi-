# -*- coding: utf-8 -*-
"""手势捏合 → 二指夹爪开合状态映射（0/1 全开全闭版，成员A）。

链路:
    MediaPipe Hand Landmarker world 关键点 (T, 2, 21, 3)
        ↓ 拇指尖-食指尖欧氏距离 D(t)，米制，与 pose 提取同帧时钟
    双阈值迟滞状态机 + 最小保持时间 + 丢失回落
        ↓
    gripper_cmd 通道 (2N, 2)，列序 [右臂, 左臂]，1=闭合 0=张开
    （mirror 片段左右手对调: 机器人右臂 ← 原视频左手）

D 来源说明: BlazePose 自带拇指/食指点噪声过大（实测最小 D 23cm，物理不可能），
thumb_index_distance() 仅留作对照诊断; 生产链路用 extract_hand.pinch_distance()。

MuJoCo 侧关节映射（实测: qpos=0 指尖间距 9.1mm=闭合, 极值 152.5mm=张开）:
    右手: 张开=-0.7854, 闭合=0     左手: 张开=+0.7854, 闭合=0
    关节: openarm_{side}_finger_joint1/2（mimic 联动, 两关节同值）
    执行器: {side}_finger1_ctrl, ctrl 目标 = qpos 目标

CLI（在 pose2arm-main 根目录、openpi 环境下运行）:
    python gripper_map.py --selftest                            # 状态机自测
    python gripper_map.py --input videos/xxx.mp4                # D(t) 统计（阈值调优依据）
    python gripper_map.py --input videos/xxx.mp4 --update-npz data/xxx.npz
                                                                # 给已有关节角 npz 补夹爪通道（免重跑 IK）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pose_constants import LANDMARK
from smoothing import OneEuroFilter

# 手部关键点可见度低于该值 → 视为不可测（D=NaN，状态保持）
VIS_THRESHOLD = 0.3

# 默认参数（工作流程规定值; T1 实测后可通过 CLI 微调，不写死）
DEFAULT_CLOSE_M     = 0.025   # D < 2.5cm → 闭合
DEFAULT_OPEN_M      = 0.045   # D > 4.5cm → 张开
DEFAULT_HOLD_FRAMES = 3       # 状态切换后最小保持帧数（30fps ≈ 100ms）
DEFAULT_LOST_FRAMES = 15      # 手部连续不可测超过该帧数 → 回落张开（安全侧, 30fps ≈ 0.5s）
DEFAULT_CLOSE_RATIO = 0.25    # 归一化捏合距离 <= 此值 → 全闭
DEFAULT_OPEN_RATIO  = 1.00    # 归一化捏合距离 >= 此值 → 全开
DEFAULT_GRIPPER_MIN_CUTOFF = 1.5
DEFAULT_GRIPPER_BETA = 0.5


# ── D(t) 计算 ──────────────────────────────────────────────────────────────────

def hold_indices(present) -> np.ndarray:
    """与 generate_ik_dataset 相同的 hold 逻辑: 丢失帧沿用最近一次检测成功帧。"""
    T = len(present)
    hold = np.full(T, -1, dtype=int)
    last = -1
    for t in range(T):
        if present[t]:
            last = t
        hold[t] = last
    valid = hold >= 0
    if valid.any():
        hold[~valid] = hold[valid][0]
    return hold


def thumb_index_distance(pose_world, present, side) -> np.ndarray:
    """逐帧拇指尖-食指尖欧氏距离 D(t)，单位米; 不可测帧为 NaN。

    pose_world: (T, 33, 4) MediaPipe world 关键点（米制，髋部原点）
    present:    (T,) bool  人体检测成功标记
    side:       "left" / "right"
    """
    i_thumb = LANDMARK[f"{side.upper()}_THUMB"]
    i_index = LANDMARK[f"{side.upper()}_INDEX"]
    hold = hold_indices(present)
    T = len(pose_world)
    D = np.full(T, np.nan)
    for t in range(T):
        if hold[t] < 0:
            continue
        f = pose_world[hold[t]]
        pt, pi = f[i_thumb], f[i_index]
        if pt[3] < VIS_THRESHOLD or pi[3] < VIS_THRESHOLD:
            continue
        D[t] = float(np.linalg.norm(pt[:3] - pi[:3]))
    return D


# ── 双阈值迟滞状态机 ───────────────────────────────────────────────────────────

class GripperMapper:
    """迟滞 + 最小保持时间 + 丢失回落。

    state: 1=闭合, 0=张开
      D < close_m            → 闭合
      D > open_m             → 张开
      close_m <= D <= open_m → 保持上一状态（迟滞带）
      D 为 NaN               → 保持; 连续 NaN 超过 lost_frames 帧 → 张开
      状态切换后 hold_frames 帧内禁止再次切换（滤单帧噪声尖峰）
    """

    def __init__(self, close_m=DEFAULT_CLOSE_M, open_m=DEFAULT_OPEN_M,
                 hold_frames=DEFAULT_HOLD_FRAMES, lost_frames=DEFAULT_LOST_FRAMES):
        if not 0.0 < close_m < open_m:
            raise ValueError(f"需满足 0 < close_m < open_m，实际 close={close_m} open={open_m}")
        self.close_m = float(close_m)
        self.open_m = float(open_m)
        self.hold_frames = int(hold_frames)
        self.lost_frames = int(lost_frames)
        if self.hold_frames < 0 or self.lost_frames < 0:
            raise ValueError("hold_frames/lost_frames must be nonnegative")
        self.state = 0             # 初始张开（安全侧）
        self._hold_left = 0        # 最小保持剩余帧数
        self._lost_count = 0       # 连续不可测帧计数

    def update(self, d) -> int:
        # NumPy float32 NaN is not a Python float. It must count as lost too.
        if d is None or not np.isfinite(d) or d < 0:
            self._lost_count += 1
            if self._lost_count > self.lost_frames and self.state == 1:
                self.state = 0     # 手消失太久，安全张开
                self._hold_left = self.hold_frames
        else:
            self._lost_count = 0
            if self._hold_left <= 0:
                if d < self.close_m and self.state != 1:
                    self.state = 1
                    self._hold_left = self.hold_frames
                elif d > self.open_m and self.state != 0:
                    self.state = 0
                    self._hold_left = self.hold_frames
        if self._hold_left > 0:
            self._hold_left -= 1
        return self.state


def map_states(D, close_m=DEFAULT_CLOSE_M, open_m=DEFAULT_OPEN_M,
               hold_frames=DEFAULT_HOLD_FRAMES,
               lost_frames=DEFAULT_LOST_FRAMES) -> np.ndarray:
    """逐帧映射 D(t) → 0/1 开合状态序列 (T,)。"""
    m = GripperMapper(close_m, open_m, hold_frames, lost_frames)
    return np.array([m.update(d) for d in D], dtype=np.float32)


class ContinuousGripperMapper:
    """归一化捏合距离 → 连续夹爪闭合量，1=全闭、0=全开。

    输入先按两个端点线性映射，再用响应快速运动的 One Euro Filter 抑制
    单帧抖动。短暂丢失保持最近值，持续丢失则平滑回到安全张开状态。
    """

    def __init__(self, fps, close_ratio=DEFAULT_CLOSE_RATIO,
                 open_ratio=DEFAULT_OPEN_RATIO,
                 min_cutoff=DEFAULT_GRIPPER_MIN_CUTOFF,
                 beta=DEFAULT_GRIPPER_BETA,
                 lost_frames=DEFAULT_LOST_FRAMES):
        if not 0.0 <= close_ratio < open_ratio:
            raise ValueError("需满足 0 <= close_ratio < open_ratio")
        self.close_ratio = float(close_ratio)
        self.open_ratio = float(open_ratio)
        self.lost_frames = int(lost_frames)
        if not np.isfinite(fps) or fps <= 0 or self.lost_frames < 0:
            raise ValueError("fps must be positive and lost_frames nonnegative")
        self._lost = 0
        self._last = 0.0
        self._filter = OneEuroFilter(float(fps), min_cutoff, beta)

    def update(self, ratio) -> float:
        if ratio is None or not np.isfinite(ratio):
            self._lost += 1
            raw = self._last if self._lost <= self.lost_frames else 0.0
        else:
            self._lost = 0
            raw = np.clip(
                (self.open_ratio - float(ratio)) /
                (self.open_ratio - self.close_ratio), 0.0, 1.0)
        self._last = float(np.clip(self._filter.filter(float(raw)), 0.0, 1.0))
        return self._last


def map_positions(ratios, fps, close_ratio=DEFAULT_CLOSE_RATIO,
                  open_ratio=DEFAULT_OPEN_RATIO,
                  min_cutoff=DEFAULT_GRIPPER_MIN_CUTOFF,
                  beta=DEFAULT_GRIPPER_BETA,
                  lost_frames=DEFAULT_LOST_FRAMES) -> np.ndarray:
    mapper = ContinuousGripperMapper(
        fps, close_ratio, open_ratio, min_cutoff, beta, lost_frames)
    return np.array([mapper.update(r) for r in ratios], dtype=np.float32)


# ── 分析 / npz 补丁 ────────────────────────────────────────────────────────────

def analyze(D, close_m, open_m, label="") -> None:
    """打印 D(t) 统计，用于 T1 阈值调优。"""
    valid = D[np.isfinite(D)]
    total = len(D)
    head = f"{label} " if label else ""
    if len(valid) == 0:
        print(f"  {head}无可测帧（手部关键点不可见或无人体检测）")
        return
    pct = np.percentile(valid, [5, 25, 50, 75, 95])
    print(f"  {head}可测帧 {len(valid)}/{total} ({100 * len(valid) / max(total, 1):.1f}%)")
    print(f"  {head}D 分位数(cm): p5={pct[0]*100:.1f}  p25={pct[1]*100:.1f}  p50={pct[2]*100:.1f}  "
          f"p75={pct[3]*100:.1f}  p95={pct[4]*100:.1f}")
    lo = float((valid < close_m).mean())
    band = float(((valid >= close_m) & (valid <= open_m)).mean())
    hi = float((valid > open_m).mean())
    print(f"  {head}帧占比: 捏合区(<{close_m*100:.1f}cm)={100*lo:.0f}%  "
          f"迟滞带={100*band:.0f}%  张开区(>{open_m*100:.1f}cm)={100*hi:.0f}%")
    states = map_states(D, close_m, open_m)
    switches = int((np.diff(states) != 0).sum())
    print(f"  {head}状态切换次数: {switches}（越少越稳）")
    if pct[0] >= close_m:
        print(f"  [!] {head}p5={pct[0]*100:.1f}cm 未低于闭合阈值 → 视频可能无捏合动作，或需下调 --close")
    if pct[4] <= open_m:
        print(f"  [!] {head}p95={pct[4]*100:.1f}cm 未超过张开阈值 → 视频中手一直较合拢，或需下调 --open")


def update_npz(npz_path: Path, D, close_m, open_m, hold_frames, lost_frames,
               ratios=None, fps=30.0) -> None:
    """给已有关节角 npz 补 gripper_cmd 通道（复用已算好的 IK，免重跑）。

    D: {side: (T,)} 拇指-食指捏合距离序列（全视频时间轴，来自 extract_hand）
    npz 新增:
      gripper_cmd   (2N, 2) float32, 列 [右臂, 左臂], 1=闭合 0=张开
      gripper_params (4,)  float32, [close_m, open_m, hold_frames, lost_frames]
    """
    d = dict(np.load(npz_path))
    if "joint_angles" not in d:
        raise SystemExit(f"{npz_path} 没有 joint_angles，不是 pose2arm 数据集")
    if "gripper_cmd" in d:
        print(f"  [WARN] 已含 gripper_cmd，将覆盖")

    fids = np.asarray(d["frame_indices"], dtype=int)
    n = len(fids) // 2
    T = len(next(iter(D.values())))
    if fids.max() >= T:
        raise SystemExit(f"frame_indices 最大 {fids.max()} 超出视频帧数 {T}，视频与 npz 不匹配")

    states = {}
    for side in ("right", "left"):
        states[side] = map_states(D[side], close_m, open_m, hold_frames, lost_frames)
        analyze(D[side], close_m, open_m, label=f"[{side}]")

    gripper = np.zeros((len(fids), 2), dtype=np.float32)
    gripper[:n, 0] = states["right"][fids[:n]]   # orig 片段: 机器人右臂 ← 人右手
    gripper[:n, 1] = states["left"][fids[:n]]
    gripper[n:, 0] = states["left"][fids[n:]]    # mirror 片段: 左右手对调
    gripper[n:, 1] = states["right"][fids[n:]]

    d["gripper_cmd"] = gripper
    d.setdefault("joint_order", np.array("left-first"))
    d["gripper_order"] = np.array("right-first")
    d["gripper_semantics"] = np.array("normalized_closure_0_open_1_closed")
    d["gripper_params"] = np.array(
        [close_m, open_m, hold_frames, lost_frames], dtype=np.float32)
    if ratios is not None:
        positions = {
            side: map_positions(ratios[side], fps, lost_frames=lost_frames)
            for side in ("right", "left")
        }
        continuous = np.zeros((len(fids), 2), dtype=np.float32)
        continuous[:n, 0] = positions["right"][fids[:n]]
        continuous[:n, 1] = positions["left"][fids[:n]]
        continuous[n:, 0] = positions["left"][fids[n:]]
        continuous[n:, 1] = positions["right"][fids[n:]]
        d["gripper_position"] = continuous
        d["gripper_position_params"] = np.array([
            DEFAULT_CLOSE_RATIO, DEFAULT_OPEN_RATIO,
            DEFAULT_GRIPPER_MIN_CUTOFF, DEFAULT_GRIPPER_BETA,
            lost_frames], dtype=np.float32)
    np.savez(npz_path, **d)
    n_close = int((gripper[:n, 0] > 0.5).sum())
    print(f"  [OK] gripper_cmd 已写入 {npz_path}，orig 片段右臂闭合帧 {n_close}/{n}")


# ── 自测 ───────────────────────────────────────────────────────────────────────

def _expect(name, got, want):
    ok = list(got) == list(want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={list(got)} want={list(want)}")
    return ok


def selftest() -> int:
    nan = float("nan")
    ok = True

    # 1. 基本迟滞（hold=0 隔离最小保持逻辑）
    m = GripperMapper(0.025, 0.045, hold_frames=0)
    seq = [0.06, 0.05, 0.02, 0.03, 0.04, 0.05, 0.06]
    got = [m.update(d) for d in seq]
    ok &= _expect("基本迟滞: 捏合→带内保持→张开", got, [0, 0, 1, 1, 1, 0, 0])

    # 2. 单帧噪声尖峰被最小保持时间滤除
    m = GripperMapper(0.025, 0.045, hold_frames=3)
    seq = [0.06, 0.02, 0.06, 0.06, 0.06, 0.06]
    got = [m.update(d) for d in seq]
    ok &= _expect("最小保持: 闭合后 1 帧尖峰被忽略", got, [0, 1, 1, 1, 0, 0])

    # 3. NaN 保持 + 超时回落张开
    m = GripperMapper(0.025, 0.045, hold_frames=0, lost_frames=3)
    seq = [0.02, nan, nan, nan, nan]
    got = [m.update(d) for d in seq]
    ok &= _expect("丢失: 短暂 NaN 保持, 超时张开", got, [1, 1, 1, 1, 0])

    # 4. 迟滞带内不受反向尖峰影响（尖峰未穿过对侧阈值）
    m = GripperMapper(0.025, 0.045, hold_frames=0)
    seq = [0.06, 0.02, 0.035, 0.03, 0.05, 0.06]
    got = [m.update(d) for d in seq]
    ok &= _expect("带内抖动不影响状态", got, [0, 1, 1, 1, 0, 0])

    # 5. 参数校验
    try:
        GripperMapper(0.05, 0.02)
        ok &= _expect("非法阈值应报错", [False], [True])
    except ValueError:
        print("  PASS  非法阈值报错")
    except Exception:
        ok &= _expect("非法阈值应报 ValueError", [False], [True])

    # 6. 连续映射端点、单调性与短暂丢失保持
    p = map_positions([1.2, 1.0, 0.625, 0.25, 0.1], fps=30,
                      min_cutoff=1000.0, beta=0.0, lost_frames=2)
    monotonic = np.all(np.diff(p) >= -1e-6) and p[0] < 0.01 and p[-1] > 0.99
    ok &= _expect("连续夹爪映射单调且覆盖开闭端点", [monotonic], [True])
    m = ContinuousGripperMapper(30, min_cutoff=1000.0, beta=0.0, lost_frames=2)
    held = [m.update(v) for v in (0.25, nan, nan)]
    ok &= _expect("连续夹爪短暂丢失保持", [abs(held[2] - held[0]) < 1e-3], [True])

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="运行状态机单元测试")
    p.add_argument("--input", type=Path, default=None, help="源视频（D 分析 / npz 补丁）")
    p.add_argument("--update-npz", type=Path, default=None,
                   help="要补 gripper_cmd 的 npz 路径（需配合 --input 指定其源视频）")
    p.add_argument("--close", type=float, default=DEFAULT_CLOSE_M,
                   help=f"闭合阈值 D 小于该值闭合，米（默认 {DEFAULT_CLOSE_M}）")
    p.add_argument("--open", type=float, default=DEFAULT_OPEN_M,
                   help=f"张开阈值 D 大于该值张开，米（默认 {DEFAULT_OPEN_M}）")
    p.add_argument("--hold", type=int, default=DEFAULT_HOLD_FRAMES,
                   help=f"切换后最小保持帧数（默认 {DEFAULT_HOLD_FRAMES}）")
    p.add_argument("--lost", type=int, default=DEFAULT_LOST_FRAMES,
                   help=f"连续不可测回落张开的帧数（默认 {DEFAULT_LOST_FRAMES}）")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    if not args.input:
        p.print_help()
        return

    from extract_pose import extract_from_video
    from extract_hand import (
        extract_hands_from_video, normalized_pinch_distance, pinch_distance)

    print(f"[POSE] 提取 {args.input} ...")
    raw = extract_from_video(str(args.input))
    print(f"  {len(raw['pose_present'])} 帧 @ {raw['fps']:.1f} fps")

    print(f"[HAND] Hand Landmarker 提取 ...")
    hands = extract_hands_from_video(str(args.input), pose_norm=raw["pose"])
    for i, side in enumerate(("left", "right")):
        n_det = int(hands["hand_present"][:, i].sum())
        print(f"  [{side}] 手部检测帧 {n_det}/{len(hands['hand_present'])}")

    D = {s: pinch_distance(hands["hand_world"], s) for s in ("right", "left")}
    ratios = {
        s: normalized_pinch_distance(hands["hand_world"], s)[0]
        for s in ("right", "left")
    }
    for side in ("right", "left"):
        analyze(D[side], args.close, args.open, label=f"[{side}]")

    if args.update_npz:
        print(f"\n[PATCH] 更新 {args.update_npz} ...")
        update_npz(args.update_npz, D,
                   args.close, args.open, args.hold, args.lost,
                   ratios=ratios, fps=float(raw["fps"]))


if __name__ == "__main__":
    main()
