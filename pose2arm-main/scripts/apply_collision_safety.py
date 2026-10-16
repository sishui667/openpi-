"""Apply hard inter-arm + fingertip collision projection to an existing NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_safety import InterArmCollisionProjector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--policy", choices=["project", "hold", "error"],
                        default="project")
    args = parser.parse_args()

    data = dict(np.load(args.input))
    joints = np.asarray(data["joint_angles"], dtype=float)
    if "gripper_position" not in data:
        raise SystemExit("输入 NPZ 缺少 gripper_position，无法验证真实指尖姿态")
    gripper = np.asarray(data["gripper_position"], dtype=float)
    if len(joints) != len(gripper) or len(joints) % 2:
        raise SystemExit("joint_angles / gripper_position 长度不一致或不是偶数")

    split = len(joints) // 2
    projector = InterArmCollisionProjector(margin=args.margin)
    safe_parts, grip_parts, rows = [], [], []
    for name, sl in (("orig", slice(0, split)), ("mirror", slice(split, None))):
        safe, safe_grip, stats = projector.project_trajectory(
            joints[sl], policy=args.policy, gripper_positions=gripper[sl])
        safe_parts.append(safe)
        grip_parts.append(safe_grip)
        rows.append([stats.unsafe_candidates, stats.projected_frames,
                     stats.held_frames])
        print(f"[{name}] unsafe={stats.unsafe_candidates} "
              f"projected={stats.projected_frames} held={stats.held_frames}")

    data["joint_angles"] = np.concatenate(safe_parts, axis=0)
    data["gripper_position"] = np.concatenate(grip_parts, axis=0)
    data["collision_safety_params"] = np.array(
        [args.margin, projector.sweep_samples, projector.bisection_steps],
        dtype=np.float32)
    data["collision_safety_stats"] = np.asarray(rows, dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)

    unsafe_after = sum(
        not projector.is_safe(q, g)
        for q, g in zip(data["joint_angles"], data["gripper_position"]))
    if unsafe_after:
        raise RuntimeError(f"安全后置条件失败：仍有 {unsafe_after} 帧违规")
    print(f"[OK] {len(joints)} frames, unsafe_after=0 → {args.output}")


if __name__ == "__main__":
    main()
