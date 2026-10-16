"""Shared MediaPipe pose landmark constants."""

NUM_POSE_LANDMARKS = 33

LANDMARK = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    # 手部关键点（BlazePose 自带，精度低于专用 Hand Landmarker，用于夹爪开合映射）
    "LEFT_THUMB": 21,
    "LEFT_INDEX": 22,
    "RIGHT_THUMB": 24,
    "RIGHT_INDEX": 25,
}

# Body-only skeleton used for overlays and sanity checks.
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]
