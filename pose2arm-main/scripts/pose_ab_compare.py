"""Render side-by-side pose detections and report temporal/coverage proxies.

These are diagnostics, not ground-truth accuracy or robot retargeting metrics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
GROUPS = {"left_arm": [5, 7, 9], "right_arm": [6, 8, 10],
          "left_hand": list(range(91, 112)),
          "right_hand": list(range(112, 133))}


def load(path: Path) -> dict:
    with np.load(path) as archive:
        return {k: archive[k].copy() for k in archive.files}


def valid(data: dict, threshold: float) -> np.ndarray:
    xy = data["image_xy"]
    size = np.array([int(data["frame_width"]), int(data["frame_height"])])
    return (np.isfinite(xy).all(axis=-1) & (data["score"] >= threshold)
            & (xy >= 0).all(axis=-1) & (xy[..., 0] < size[0])
            & (xy[..., 1] < size[1]))


def longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for item in mask:
        current = 0 if item else current + 1
        longest = max(longest, current)
    return longest


def percentile_or_nan(x: np.ndarray, percentile: int) -> float:
    return float(np.percentile(x, percentile)) if len(x) else float("nan")


def row_metrics(data: dict, mask: np.ndarray) -> dict:
    xy = data["image_xy"][:len(mask)]
    count = len(xy)
    result = {"backend": str(data["backend"]), "frames": count}
    for group, indices in GROUPS.items():
        all_visible = mask[:, indices].all(axis=1)
        result[f"{group}_complete_pct"] = round(100 * float(all_visible.mean()), 2)
        result[f"{group}_max_missing_run"] = longest_false_run(all_visible)
    for side, wrist, elbow, hand_start in (("left", 9, 7, 91),
                                            ("right", 10, 8, 112)):
        adjacent = mask[1:, wrist] & mask[:-1, wrist]
        jumps = np.linalg.norm(np.diff(xy[:, wrist], axis=0), axis=1)[adjacent]
        result[f"{side}_wrist_jump_p95_px"] = round(percentile_or_nan(jumps, 95), 2)
        result[f"{side}_wrist_jump_max_px"] = round(float(jumps.max()), 2) if len(jumps) else float("nan")
        result[f"{side}_wrist_jump_max_frames"] = (
            f"{int(np.flatnonzero(adjacent)[np.argmax(jumps)])}-"
            f"{int(np.flatnonzero(adjacent)[np.argmax(jumps)]) + 1}"
            if len(jumps) else "none")
        bone_ok = mask[:, wrist] & mask[:, elbow]
        lengths = np.linalg.norm(xy[:, wrist] - xy[:, elbow], axis=1)[bone_ok]
        result[f"{side}_forearm_2d_cv"] = round(float(lengths.std() / lengths.mean()), 3) if len(lengths) and lengths.mean() else float("nan")
        # Thumb/index distance relative to palm width. A detection, not a binary grasp label.
        ids = np.array([hand_start, hand_start + 4, hand_start + 8, hand_start + 5,
                        hand_start + 17])
        pinch_ok = mask[:, ids].all(axis=1)
        hand = xy[pinch_ok]
        gap = np.linalg.norm(hand[:, ids[1]] - hand[:, ids[2]], axis=1)
        palm = np.linalg.norm(hand[:, ids[3]] - hand[:, ids[4]], axis=1)
        ratio = gap / np.maximum(palm, 1.0)
        result[f"{side}_pinch_ratio_p05"] = round(percentile_or_nan(ratio, 5), 3)
        result[f"{side}_pinch_ratio_p95"] = round(percentile_or_nan(ratio, 95), 3)
    return result


def draw_skeleton(image: np.ndarray, xy: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int], scale: float) -> None:
    points = xy * scale
    edges = BODY_EDGES + [(start + a, start + b)
                          for start in (91, 112) for a, b in HAND_EDGES]
    edges += [(9, 91), (10, 112)]
    for a, b in edges:
        if mask[a] and mask[b]:
            pa = tuple(np.rint(points[a]).astype(int))
            pb = tuple(np.rint(points[b]).astype(int))
            cv2.line(image, pa, pb, color, 2, cv2.LINE_AA)
    for idx in list(range(5, 17)) + list(range(91, 133)):
        if mask[idx]:
            point = tuple(np.rint(points[idx]).astype(int))
            cv2.circle(image, point, 3 if idx < 17 else 2, (255, 255, 255), -1,
                       cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mediapipe", type=Path, required=True)
    parser.add_argument("--rtmw3d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtmw-threshold", type=float, default=0.3)
    args = parser.parse_args()
    mp, rt = load(args.mediapipe), load(args.rtmw3d)
    n = min(len(mp["image_xy"]), len(rt["image_xy"]))
    if not n:
        raise ValueError("no extracted frames")
    if (int(mp["frame_width"]), int(mp["frame_height"])) != (
            int(rt["frame_width"]), int(rt["frame_height"])):
        raise ValueError("video sizes do not match")
    mp_mask = valid(mp, 0.5)[:n]
    rt_mask = valid(rt, args.rtmw_threshold)[:n]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    output_video = args.output_dir / f"{stem}_pose_ab.mp4"
    source = cv2.VideoCapture(str(args.video))
    if not source.isOpened():
        raise FileNotFoundError(args.video)
    width, height = int(mp["frame_width"]), int(mp["frame_height"])
    panel_width = min(width, 540)
    scale = panel_width / width
    panel_height = round(height * scale)
    fps = float(mp["fps"])
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (panel_width * 2, panel_height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {output_video}")
    try:
        for frame_id in range(n):
            ok, frame = source.read()
            if not ok:
                raise RuntimeError(f"source ended at frame {frame_id}")
            base = cv2.resize(frame, (panel_width, panel_height),
                              interpolation=cv2.INTER_AREA)
            panels = [base.copy(), base.copy()]
            for panel, data, mask, label, color in (
                    (panels[0], mp, mp_mask, "MediaPipe", (0, 210, 255)),
                    (panels[1], rt, rt_mask, "RTMW3D", (255, 170, 0))):
                draw_skeleton(panel, data["image_xy"][frame_id], mask[frame_id],
                              color, scale)
                cv2.rectangle(panel, (0, 0), (panel_width, 64), (0, 0, 0), -1)
                cv2.putText(panel, f"{label}  frame {frame_id}  {frame_id/fps:.2f}s",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, "2D overlay / no robot command change", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1,
                            cv2.LINE_AA)
            writer.write(np.concatenate(panels, axis=1))
            if (frame_id + 1) % 100 == 0:
                print(f"rendered {frame_id + 1}/{n}", flush=True)
    finally:
        writer.release()
        source.release()
    rows = [row_metrics(mp, mp_mask), row_metrics(rt, rt_mask)]
    csv_path = args.output_dir / f"{stem}_pose_ab_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)
    report_path = args.output_dir / f"{stem}_pose_ab_report.md"
    with report_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# {stem} 人体关键点 A/B 测试\n\n")
        stream.write(f"视频：`{args.video.resolve()}`；比较帧数：{n}；帧率：{fps:.3f} FPS。\n\n")
        stream.write(f"并排回放：`{output_video.resolve()}`。左为 MediaPipe，右为 RTMW3D。\n\n")
        stream.write("| 指标 | MediaPipe | RTMW3D |\n|---|---:|---:|\n")
        for key in list(rows[0])[2:]:
            stream.write(f"| {key} | {rows[0][key]} | {rows[1][key]} |\n")
        stream.write("\n指标解释：complete_pct 是关节同时落在画面内且达到各自阈值的比例；"
                     "max_missing_run 是最长连续缺失帧数；wrist_jump 是相邻可见帧像素位移；"
                     "forearm_2d_cv 是前臂二维投影长度变异系数；pinch_ratio 是拇指与食指指尖距离除以掌宽。\n\n")
        stream.write("注意：没有人工标注真值，不能据此断言哪个模型更准确。"
                     "两模型分数未校准，MediaPipe 手部只提供整手检出标记；"
                     "二维骨长受深度、透视和遮挡影响。RTMW3D 的 model_xyz 是模型坐标，"
                     "不是机械臂坐标或米制空间，不可直接输入控制器。\n")
        stream.write("\n本次重点复核：右腕最大跳变对应的帧对见上表；"
                     "大跳变可能是身份切换、遮挡后重定位或真实快速动作，"
                     "须结合并排视频判断。\n")
    print(f"video: {output_video}\nmetrics: {csv_path}\nreport: {report_path}")


if __name__ == "__main__":
    main()
