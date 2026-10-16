# -*- coding: utf-8 -*-
"""Per-frame hand keypoint extraction with MediaPipe Hand Landmarker (21 点/手).

用途: 拇指尖(#4)-食指尖(#8) 的 world 坐标欧氏距离 = 真实捏合距离 D(t)，
是夹爪开合映射(gripper_map.py)的首选信号源。BlazePose 自带的拇指/食指点
实测噪声过大（waving_demo / p2a_test 两段视频 D 最小 23cm，物理上不可能），
故引入专用手部模型。

左右手归属: 优先用 Pose 腕部(15/16) normalized 坐标最近邻关联 —— 非镜像视频下
MediaPipe handedness 假设自拍镜像，label 语义易混淆; 无 pose 腕部时才回退
handedness（按非镜像视频取反）。
"""
from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from extract_pose import _delegate_from_name, _model_path, _to_mp_image
from pose_constants import LANDMARK
from hand_association import associate_hands

NUM_HAND_LANDMARKS = 21
THUMB_TIP = 4   # 拇指尖
INDEX_TIP = 8   # 食指尖
INDEX_MCP = 5   # 食指掌指关节
PINKY_MCP = 17  # 小指掌指关节
WRIST     = 0   # 手腕（关联用）

HAND_MODEL_PATH = _model_path("hand_landmarker.task")

# 输出数组的手顺序: [0]=left, [1]=right
SIDES = ("left", "right")


def _make_hand_landmarker(model_path=HAND_MODEL_PATH, delegate="cpu", num_hands=2):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path,
                                 delegate=_delegate_from_name(delegate)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def _assign_side(himg_wrist, handedness_label, pose_frame):
    """把检测到的手归属到 left(0) / right(1)。

    himg_wrist:        该手手腕 normalized 坐标 (2,)
    handedness_label:  "Left"/"Right"（MediaPipe 镜像假设下的输出）
    pose_frame:        (33,4) normalized pose landmarks，或 None
    """
    if pose_frame is not None:
        best, best_d = None, 1e9
        for s, lm in ((0, LANDMARK["LEFT_WRIST"]), (1, LANDMARK["RIGHT_WRIST"])):
            p = pose_frame[lm]
            if not np.isfinite(p[0]) or p[3] < 0.3:
                continue
            d = float(np.linalg.norm(himg_wrist - p[:2]))
            if d < best_d:
                best_d, best = d, s
        if best is not None:
            return best
    # 回退: MediaPipe 假设镜像输入 → 非镜像视频中 "Left" 对应人的右手
    return 1 if handedness_label == "Left" else 0


def extract_hands_from_video(video_path, pose_norm=None, delegate="cpu"):
    """逐帧手部关键点提取（VIDEO 模式，与 extract_pose 同帧时钟）。

    pose_norm: extract_from_video()["pose"]，(T,33,4) normalized 坐标，
               用于左右手腕部关联（推荐传入; None 时回退 handedness）。
    返回 dict:
      hand_world   (T, 2, 21, 3)  world landmarks（米，原点=手部几何中心）
      hand_img     (T, 2, 21, 2)  normalized image 坐标
      hand_present (T, 2) bool    列 [0]=left [1]=right
      fps          float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker = _make_hand_landmarker(delegate=delegate)
    world, img, present = [], [], []
    t = 0
    previous_wrist = np.full((2,2),np.nan)
    missing_counts = np.zeros(2,dtype=int)
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            ts_ms = int(round((t / fps) * 1000))
            res = landmarker.detect_for_video(_to_mp_image(frame_bgr), ts_ms)

            fw = np.full((2, NUM_HAND_LANDMARKS, 3), np.nan)
            fi = np.full((2, NUM_HAND_LANDMARKS, 2), np.nan)
            fp = np.zeros(2, dtype=bool)

            pose_frame = None
            if pose_norm is not None and t < len(pose_norm):
                pose_frame = pose_norm[t]

            wrists = [[lms[WRIST].x,lms[WRIST].y] for lms in res.hand_landmarks]
            labels = [h[0].category_name if h else "" for h in res.handedness]
            assignments = associate_hands(wrists,labels,pose_frame,previous_wrist)
            for k,side in enumerate(assignments):
                if side is None:
                    continue
                lms, wlms = res.hand_landmarks[k], res.hand_world_landmarks[k]
                fi[side] = [(l.x, l.y) for l in lms]
                fw[side] = [(w.x, w.y, w.z) for w in wlms]
                fp[side] = True
                previous_wrist[side] = wrists[k]
            missing_counts = np.where(fp,0,missing_counts+1)
            previous_wrist[missing_counts>15] = np.nan

            world.append(fw)
            img.append(fi)
            present.append(fp)
            t += 1
    finally:
        cap.release()
        landmarker.close()

    return {
        "hand_world":   np.array(world, dtype=float),
        "hand_img":     np.array(img, dtype=float),
        "hand_present": np.array(present, dtype=bool),
        "fps": fps,
    }


def pinch_distance(hand_world, side):
    """拇指尖-食指尖 world 欧氏距离 D(t)，单位米; 未检测帧为 NaN。

    hand_world: (T, 2, 21, 3); side: "left" / "right"
    """
    s = SIDES.index(side)
    a = hand_world[:, s, THUMB_TIP]
    b = hand_world[:, s, INDEX_TIP]
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def normalized_pinch_distance(hand_world, side):
    """返回与手大小无关的捏合距离和手掌宽度。

    ratio = |thumb_tip-index_tip| / |index_mcp-pinky_mcp|。Hand Landmarker 的
    world 坐标以手部为局部原点，因此这个比值同时消除了人的手掌尺寸和画面缩放。
    无法检测或手掌宽度退化的帧返回 NaN。
    """
    s = SIDES.index(side)
    hand = np.asarray(hand_world[:, s], dtype=float)
    pinch = np.linalg.norm(hand[:, THUMB_TIP] - hand[:, INDEX_TIP], axis=-1)
    palm_width = np.linalg.norm(hand[:, INDEX_MCP] - hand[:, PINKY_MCP], axis=-1)
    ratio = np.full(len(hand), np.nan, dtype=float)
    valid = np.isfinite(pinch) & np.isfinite(palm_width) & (palm_width > 1e-6)
    ratio[valid] = pinch[valid] / palm_width[valid]
    return ratio, palm_width
