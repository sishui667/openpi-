"""A -> B offline contract: split augmentation clips and preserve video time.

No MuJoCo state, object truth or detector is used here. Interpolation is NOT
collision certification: the receiving scene must check its own geometry.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def adapt(blob, hz=20.0, clip="orig"):
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be positive")
    q = np.asarray(blob["joint_angles"], dtype=float)
    obs = np.asarray(blob["target_obs"], dtype=float)
    fids = np.asarray(blob["frame_indices"])
    fps = float(blob["fps"])
    if (q.ndim != 2 or q.shape[1] != 14 or not len(q)
            or obs.shape != (len(q),18) or fids.shape != (len(q),)):
        raise ValueError("invalid NPZ shapes")
    if (not np.isfinite(q).all() or not np.isfinite(obs).all()
            or not np.issubdtype(fids.dtype, np.integer) or np.any(fids < 0)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("invalid source values/fps")
    boundaries = np.r_[0, np.flatnonzero(np.diff(fids) <= 0)+1, len(q)]
    if len(boundaries) > 3:
        raise ValueError("ambiguous clip layout: expected one clip or original+mirror")
    segment = 0 if clip == "orig" else 1 if clip == "mirror" else -1
    if segment < 0 or segment >= len(boundaries)-1:
        raise ValueError("requested clip unavailable; never concatenate mirror as continuous motion")
    lo, hi = boundaries[segment:segment+2]
    source_time = (fids[lo:hi]-fids[lo])/fps
    # Every exported sample represents one complete control interval. A held
    # final sample covers the video endpoint; no clip-boundary interpolation.
    duration = float(source_time[-1] + 1/fps)
    time = np.arange(max(1, int(np.ceil(duration*hz-1e-9))), dtype=float)/hz
    source_rows = lo + np.clip(np.searchsorted(source_time, time, side="right")-1, 0, hi-lo-1)
    order = str(np.asarray(blob.get("joint_order", "left-first")).item())
    if order not in ("left-first", "right-first"):
        raise ValueError("invalid joint_order")
    result = {
        "joint_angles": np.column_stack([np.interp(time,source_time,q[lo:hi,j]) for j in range(14)]),
        "target_obs": obs[source_rows].copy(),
        "frame_indices": np.arange(len(time), dtype=np.int64),
        "timestamps_s": time, "fps": np.array(hz), "source_fps": np.array(fps),
        "source_frame_indices": fids[source_rows].copy(),
        "source_rows": source_rows, "joint_order": np.array(order),
        "source_clip": np.array(clip), "source_duration_s": np.array(duration),
        "schema_version": np.array(1),
        "processing_mode": np.array("offline_resampled_not_online_inference"),
        "collision_validation": np.array("receiving_scene_required_after_resampling"),
    }
    grip_order = str(np.asarray(blob.get("gripper_order", "right-first")).item())
    if grip_order not in ("left-first", "right-first"):
        raise ValueError("invalid gripper_order")
    for key in ("gripper_position", "gripper_cmd"):
        if key not in blob:
            continue
        g = np.asarray(blob[key], dtype=float)
        if (g.shape != (len(q),2) or not np.isfinite(g).all()
                or np.any((g < 0) | (g > 1))):
            raise ValueError(f"{key} must be finite normalized (F,2)")
        if key == "gripper_cmd" and not np.isin(g,[0,1]).all():
            raise ValueError("gripper_cmd must be binary")
        result[key] = (np.column_stack([np.interp(time,source_time,g[lo:hi,j]) for j in range(2)])
                       if key == "gripper_position" else g[source_rows].copy())
    if "gripper_position" not in result and "gripper_cmd" not in result:
        raise ValueError("A output has no human gripper signal; do not invent it from object distance")
    result["gripper_order"] = np.array(grip_order)
    result["gripper_semantics"] = np.array("normalized_closure_0_open_1_closed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--clip", choices=("orig","mirror"), default="orig")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve() or args.output.exists():
        parser.error("output must be a NEW file, not the original or an existing artifact")
    with np.load(args.input, allow_pickle=False) as source:
        result = adapt(dict(source), args.hz, args.clip)
    result["source_sha256"] = np.array(hashlib.sha256(args.input.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(json.dumps({"output":str(args.output), "frames":len(result["joint_angles"]),
                      "hz":args.hz, "source_duration_s":float(result["source_duration_s"]),
                      "playback_duration_s":len(result["joint_angles"])/args.hz,
                      "clip":args.clip, "mode":str(result["processing_mode"])}))


if __name__ == "__main__":
    main()
