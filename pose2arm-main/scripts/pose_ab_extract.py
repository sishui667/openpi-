"""Extract comparable 133-point whole-body arrays from either pose backend.

The RTMW3D 3D output is retained as model coordinates, not assumed metric.
Run each backend with its own Python environment; the NPZ schema is shared.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)


def video_info(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return width, height, frames, fps


def extract_mediapipe(path: Path, width: int, height: int, max_frames: int):
    import sys
    sys.path.insert(0, str(ROOT))
    from extract_pose import extract_from_video
    from extract_hand import extract_hands_from_video

    raw = extract_from_video(str(path))
    hands = extract_hands_from_video(str(path), pose_norm=raw["pose"])
    total = len(raw["pose"])
    limit = total if max_frames <= 0 else min(total, max_frames)
    points = np.full((limit, 133, 2), np.nan, np.float32)
    scores = np.zeros((limit, 133), np.float32)
    world = np.full((limit, 133, 3), np.nan, np.float32)
    size = np.array([width, height], np.float32)
    body = raw["pose"][:limit, BODY_MP]
    body_world = raw["pose_world"][:limit, BODY_MP]
    points[:, :17] = body[..., :2] * size
    scores[:, :17] = np.clip(body[..., 3], 0, 1)
    world[:, :17] = body_world[..., :3]
    for hand_id, target in enumerate((LEFT_HAND, RIGHT_HAND)):
        present = hands["hand_present"][:limit, hand_id]
        points[present, target] = hands["hand_img"][:limit, hand_id][present] * size
        world[present, target] = hands["hand_world"][:limit, hand_id][present]
        scores[present, target] = 1.0  # model has detection flag, no per-point confidence
    return points, scores, world, limit


def extract_rtmw3d(path: Path, max_frames: int, device: str):
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "rtmlib"))
    if device == "dml":
        import onnxruntime as ort
        from rtmlib.tools.base import RTMLIB_SETTINGS
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML provider unavailable; install onnxruntime-directml")
        RTMLIB_SETTINGS["onnxruntime"]["dml"] = "DmlExecutionProvider"
    from rtmlib import Wholebody3d

    detector = Wholebody3d(backend="onnxruntime", device=device)
    cap = cv2.VideoCapture(str(path))
    points_list, scores_list, world_list = [], [], []
    last_center = None
    frame_id = 0
    try:
        while True:
            if max_frames > 0 and frame_id >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            xyz, scores, _, image_xy = detector(frame)
            if len(image_xy):
                centers = np.nanmean(image_xy[:, [5, 6, 11, 12]], axis=1)
                strength = np.nanmean(scores[:, [5, 6, 7, 8, 9, 10]], axis=1)
                if last_center is None:
                    choice = int(np.argmax(strength))
                else:
                    distances = np.linalg.norm(centers - last_center, axis=1)
                    choice = int(np.argmax(strength - 0.25 * distances / max(width, height)))
                last_center = centers[choice]
                points_list.append(np.asarray(image_xy[choice], np.float32))
                scores_list.append(np.asarray(scores[choice], np.float32))
                world_list.append(np.asarray(xyz[choice], np.float32))
            else:
                points_list.append(np.full((133, 2), np.nan, np.float32))
                scores_list.append(np.zeros(133, np.float32))
                world_list.append(np.full((133, 3), np.nan, np.float32))
            frame_id += 1
            if frame_id % 25 == 0:
                print(f"RTMW3D: {frame_id} frames", flush=True)
    finally:
        cap.release()
    return (np.asarray(points_list), np.asarray(scores_list),
            np.asarray(world_list), frame_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mediapipe", "rtmw3d"), required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "videos" / "p2a_test1.mp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "dml"), default="cpu",
                        help="RTMW3D ONNX device; dml uses Windows DirectML GPU")
    args = parser.parse_args()
    width, height, _, fps = video_info(args.input)
    if args.backend == "mediapipe":
        points, scores, world, total = extract_mediapipe(
            args.input, width, height, args.max_frames)
    else:
        points, scores, world, total = extract_rtmw3d(
            args.input, args.max_frames, args.device)
    if points.shape != (total, 133, 2) or scores.shape != (total, 133):
        raise RuntimeError(f"unexpected extraction shape: {points.shape}, {scores.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, image_xy=points, score=scores,
                        model_xyz=world, fps=fps,
                        frame_width=width, frame_height=height,
                        backend=args.backend)
    print(f"saved {total} frames to {args.output}", flush=True)


if __name__ == "__main__":
    main()
