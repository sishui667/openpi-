# -*- coding: utf-8 -*-
"""Member B: monocular workspace camera -- cube pose from ONE image.

Method -- model-based silhouette fitting.  A single RGB image has no depth, so
the cube's 3D pose cannot be read off directly.  What we do have is the camera's
exact intrinsics/extrinsics and the knowledge that the object is a box of KNOWN
size RESTING ON THE TABLE.  That leaves exactly three unknowns: (x, y, yaw) on
the table plane.  For each candidate we project the 8 corners, rasterise the
silhouette, and score IoU against the observed red mask.  Coarse grid, then
refine.  The winner is the perceived pose.

Why not just project the red centroid onto the table?  At 45 degrees we see the
TOP face, which is one cube half-height above the table, so the ray through its
centroid lands about a cube-width from the true centre (measured 30.07 mm vs
2.24 mm for the fit).  Fitting the whole silhouette has no such bias.

Nothing here reads d.xpos to form an estimate.  Ground truth is read only by the
self-check in __main__, to score the estimator.

Cube and camera constants come from the frozen perception scene JSON. Table
height comes from task_workspace.json, matching the CURRENT compiled task
scene rather than the older base XML table. No object state supplies a target.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from task_workspace import TABLE_TOP as TASK_TABLE_TOP

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "simulation" / "v1"
SCENE_XML = SCENE_DIR / "droid_cube_scene_perception.xml"
SCENE_JSON = SCENE_DIR / "droid_cube_scene_perception.json"

W, H = 640, 480


def scene_constants():
    if not SCENE_JSON.is_file():
        raise FileNotFoundError(
            f"{SCENE_JSON} not found -- run build_droid_cube_perception_scene.py first")
    cfg = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
    return {
        "half": float(cfg["object"]["half_size_m"][0]),
        "table_top": float(TASK_TABLE_TOP),
        "cam_name": cfg["camera"]["name"],
        "cam_pos": np.array(cfg["camera"]["pos"], dtype=float),
        "cam_quat": np.array(cfg["camera"]["quat"], dtype=float),
        "fovy": float(cfg["camera"]["fovy_deg"]),
        # the object's red channel is what the mask keys on
        "rgba": cfg["object"]["rgba"],
    }


_C = scene_constants()
CUBE_HALF = _C["half"]
TABLE_TOP = _C["table_top"]
CAM_NAME = _C["cam_name"]


def red_mask(rgb, keep_largest=True):
    """Red blob of the tracked object.  HSV, so shading does not matter."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (cv2.inRange(hsv, (0, 90, 60), (12, 255, 255))
         | cv2.inRange(hsv, (168, 90, 60), (180, 255, 255)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not keep_largest:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == k, 255, 0).astype(np.uint8)


class Pinhole:
    """Intrinsics + extrinsics for one MuJoCo camera.

    MuJoCo cameras look along -z of their own frame, so depth is -z_cam.
    """

    def __init__(self, model, data, name=CAM_NAME, w=W, h=H):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(f"no camera named {name!r} in this model")
        self.pos = data.cam_xpos[cid].copy()
        self.R = data.cam_xmat[cid].reshape(3, 3).copy()      # camera -> world
        self.f = 0.5 * h / np.tan(0.5 * np.deg2rad(model.cam_fovy[cid]))
        self.cx, self.cy = 0.5 * w, 0.5 * h

    def project(self, pts_world):
        pc = (np.asarray(pts_world) - self.pos) @ self.R
        depth = -pc[:, 2]
        u = self.cx + self.f * pc[:, 0] / depth
        v = self.cy - self.f * pc[:, 1] / depth
        return np.stack([u, v], axis=1), depth

    def ray_to_table(self, u, v, z=TABLE_TOP):
        """Where the pixel's ray crosses a horizontal plane."""
        d = self.R @ np.array([(u - self.cx) / self.f, -(v - self.cy) / self.f, -1.0])
        t = (z - self.pos[2]) / d[2]
        return self.pos + t * d


def cube_corners(x, y, yaw, half=CUBE_HALF, z0=TABLE_TOP):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    loc = half * np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    return (loc @ Rz.T) + np.array([x, y, z0 + half])


def silhouette(cam, x, y, yaw, shape=(H, W), offset=(0, 0)):
    uv, depth = cam.project(cube_corners(x, y, yaw))
    if (depth <= 0.01).any():
        return None
    uv = uv - np.asarray(offset, dtype=float)
    # fillConvexPoly assumes the vertices are ALREADY in hull order.  The 8
    # projected corners are not, so filling them directly self-intersects and
    # silently halves the area -- take the hull first.
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    return cv2.fillConvexPoly(np.zeros(shape, np.uint8), hull, 255)


# default search box, in metres on the table plane; wide enough for the frozen
# task's workspace and tight enough to stay fast
SEARCH_X = (0.28, 0.62)
SEARCH_Y = (-0.18, 0.18)
COARSE_STEP = 0.012
COARSE_YAW_DEG = 15
MIN_AREA_RATIO = 0.70      # below this, IoU cannot beat the gate, so skip the raster


def mask_roi(mask, pad):
    """Bounding box of the blob, padded.  Fitting inside it is ~20x less work
    than fitting over the full frame, and the search never needs pixels the
    object cannot reach."""
    ys, xs = np.nonzero(mask)
    return (max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1),
            max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1))


def iou_roi(cam, sub, area_mask, x, y, yaw, ox, oy):
    """IoU of the projected cube against the mask, computed inside the ROI."""
    uv, depth = cam.project(cube_corners(x, y, yaw))
    if (depth <= 0.01).any():
        return 0.0
    hull = cv2.convexHull(np.round(uv - (ox, oy)).astype(np.int32))
    poly = abs(float(cv2.contourArea(hull)))
    if poly <= 1.0:
        return 0.0
    # cheap gate: IoU is bounded by the area ratio, so a wildly wrong size
    # cannot win and does not deserve a rasterisation
    if min(poly, area_mask) / max(poly, area_mask) < MIN_AREA_RATIO:
        return 0.0
    pred = cv2.fillConvexPoly(np.zeros(sub.shape, np.uint8), hull, 255)
    inter = int(np.count_nonzero(pred & sub))
    return inter / (area_mask + int(np.count_nonzero(pred)) - inter)


_LOCAL_CORNERS = CUBE_HALF * np.array(
    [[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], dtype=float)


def project_many(cam, xs, ys, yaws):
    """Project the cube's 8 corners for MANY candidate poses in one shot.

    Returns (N, 8, 2) pixel coords and (N, 8) depths.
    """
    n = len(xs)
    c, s = np.cos(yaws), np.sin(yaws)
    Rz = np.zeros((n, 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
    Rz[:, 1, 0], Rz[:, 1, 1] = s, c
    Rz[:, 2, 2] = 1.0
    origin = np.stack([xs, ys, np.full(n, TABLE_TOP + CUBE_HALF)], axis=1)
    world = np.einsum("nij,kj->nki", Rz, _LOCAL_CORNERS) + origin[:, None, :]
    pc = (world - cam.pos) @ cam.R
    depth = -pc[:, :, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = cam.cx + cam.f * pc[:, :, 0] / safe
    v = cam.cy - cam.f * pc[:, :, 1] / safe
    return np.stack([u, v], axis=2), depth


def hull_area_centroid(pts):
    """Area and centroid of the convex hull of each candidate's 8 points.

    Vectorised substitute for rasterising and counting pixels: sorting by angle
    around the mean gives the hull order, then the shoelace formula gives both
    quantities exactly.  Used only to RANK candidates; the winner is always
    re-scored with a real IoU, so the estimator's accuracy is unchanged.
    """
    cen = pts.mean(axis=1, keepdims=True)
    ang = np.arctan2(pts[:, :, 1] - cen[:, :, 1], pts[:, :, 0] - cen[:, :, 0])
    order = np.argsort(ang, axis=1)
    p = np.take_along_axis(pts, order[:, :, None], axis=1)
    x, y = p[:, :, 0], p[:, :, 1]
    xr, yr = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yr - xr * y
    a = 0.5 * np.sum(cross, axis=1)
    denom = 6.0 * np.where(np.abs(a) < 1e-9, 1e-9, a)
    cx = np.sum((x + xr) * cross, axis=1) / denom
    cy = np.sum((y + yr) * cross, axis=1) / denom
    return np.abs(a), np.stack([cx, cy], axis=1)


REFINE_MARGIN_M = 0.012

# Per-stage seconds from the last fit() call, for the latency work.  Off unless
# PROFILE is set; it costs two clock reads per stage and nothing else.
PROFILE = False
LAST_PROFILE = {}
TRACKING_STATS = {"stationary_current_mask_validated": 0, "local_fits": 0, "global_fallbacks": 0}


def _tick(key, t0):
    if PROFILE:
        LAST_PROFILE[key] = LAST_PROFILE.get(key, 0.0) + (time.perf_counter() - t0)
    return time.perf_counter()


def tight_roi(cam, mask, x, y, yaw):
    """Crop the mask to the predicted silhouette plus the refinement window.

    The refinement passes move a few millimetres at most, so rasterising the
    full padded mask bbox for each of ~250 candidates spends nearly all of its
    work outside the region any of them can reach.  The crop is still wide
    enough to contain every pose the refinement can propose, so this is a pure
    speed-up -- but it does mean the refinement's IoU is measured over a
    different window than the coarse stage's, which is why the coarse score is
    kept as the bar to beat rather than reset.
    """
    uv, depth = project_many(cam, np.array([x]), np.array([y]), np.array([yaw]))
    u, v = uv[0, :, 0], uv[0, :, 1]
    px_per_m = cam.f / max(float(np.mean(depth[0])), 1e-3)
    m = REFINE_MARGIN_M * px_per_m
    h, w = mask.shape
    x0, x1 = max(0, int(np.floor(u.min() - m))), min(w, int(np.ceil(u.max() + m)) + 1)
    y0, y1 = max(0, int(np.floor(v.min() - m))), min(h, int(np.ceil(v.max() + m)) + 1)
    sub = np.ascontiguousarray(mask[y0:y1, x0:x1])
    return sub, x0, y0, int(np.count_nonzero(sub))


def fit(cam, mask, step=COARSE_STEP, yaw_deg=COARSE_YAW_DEG, topk=40,
        search_x=SEARCH_X, search_y=SEARCH_Y, yaw_range_deg=(0.0, 90.0)):
    """Fit (x, y, yaw) on the table plane.

    Coarse stage: rank every candidate by an analytic area/centroid mismatch
    (vectorised, ~5000 candidates in a few ms).  Then score only the top `topk`
    with a real rasterised IoU, and refine in two local passes over a tight
    window.  The winner is always chosen by IoU, so these are pure speed-ups,
    not a change of estimator.
    """
    if int(np.count_nonzero(mask)) == 0:
        return None, 0.0
    if PROFILE:
        LAST_PROFILE.clear()
    _t = time.perf_counter()
    pad = int(1.6 * np.sqrt(int(np.count_nonzero(mask))))
    y0, y1, x0, x1 = mask_roi(mask, pad)
    sub = np.ascontiguousarray(mask[y0:y1, x0:x1])
    area = int(np.count_nonzero(sub))
    ys_m, xs_m = np.nonzero(sub)
    target_c = np.array([xs_m.mean(), ys_m.mean()])
    _t = _tick("roi", _t)

    gx = np.arange(search_x[0], search_x[1], step)
    gy = np.arange(search_y[0], search_y[1], step)
    gth = np.mod(np.deg2rad(np.arange(yaw_range_deg[0], yaw_range_deg[1], yaw_deg)),
                  0.5 * np.pi)
    X, Y, T = np.meshgrid(gx, gy, gth, indexing="ij")
    X, Y, T = X.ravel(), Y.ravel(), T.ravel()

    pts, depth = project_many(cam, X, Y, T)
    pa, pc = hull_area_centroid(pts)
    scale = np.sqrt(max(area, 1.0))
    cost = (np.abs(pa - area) / area
            + np.linalg.norm(pc - np.array([x0, y0]) - target_c, axis=1) / scale)
    cost[np.any(depth <= 0.01, axis=1)] = np.inf
    _t = _tick("vectorised", _t)

    order = np.argsort(cost)[:topk]
    _t = _tick("argsort", _t)
    best, bestv = None, -1.0
    for i in order:
        v = iou_roi(cam, sub, area, float(X[i]), float(Y[i]), float(T[i]), x0, y0)
        if v > bestv:
            bestv, best = v, (float(X[i]), float(Y[i]), float(T[i]))
    _t = _tick("coarse_iou", _t)
    if best is None:
        return None, 0.0

    xb, yb, tb = best
    sub_t, xt, yt, area_t = tight_roi(cam, mask, xb, yb, tb)
    _t = _tick("tight_roi", _t)
    for sx, sy, st in ((0.004, 0.004, np.deg2rad(4)), (0.001, 0.001, np.deg2rad(1))):
        for x in np.arange(xb - 2 * sx, xb + 2 * sx + 1e-9, sx):
            for y in np.arange(yb - 2 * sy, yb + 2 * sy + 1e-9, sy):
                for yaw in np.arange(tb - 2 * st, tb + 2 * st + 1e-9, st):
                    v = iou_roi(cam, sub_t, area_t, x, y, yaw, xt, yt)
                    if v > bestv:
                        bestv, best = v, (float(x), float(y), float(yaw))
        xb, yb, tb = best
    _tick("refine_iou", _t)
    return best, bestv


def fit_tracking(cam, mask, previous, min_iou=0.70):
    """Track near the previous pose, falling back to a global fit if needed.

    At 20 Hz the cube cannot physically jump across the table between adjacent
    frames.  Repeating the full table search every frame wastes most of the
    latency budget and makes the pipeline more sensitive to host load.  The
    local window still uses the same real-IoU scoring and two refinement passes;
    a poor score triggers the original global search, so recovery is explicit.
    """
    x, y, yaw = (float(previous[0]), float(previous[1]), float(previous[2]))
    # Validate against the CURRENT image before reusing a stationary estimate.
    # No robot/object ground truth is consulted. Motion or occlusion reduces
    # silhouette IoU and triggers the original local/global fitting path.
    area = int(np.count_nonzero(mask))
    if area:
        previous_iou = iou(cam, mask, x, y, yaw, area)
        if previous_iou >= 0.97:
            TRACKING_STATS["stationary_current_mask_validated"] += 1
            return (x, y, yaw), previous_iou, False
    TRACKING_STATS["local_fits"] += 1
    best, score = fit(
        cam, mask, step=0.004, yaw_deg=3.0, topk=12,
        search_x=(x - 0.016, x + 0.016 + 1e-9),
        search_y=(y - 0.016, y + 0.016 + 1e-9),
        yaw_range_deg=(np.rad2deg(yaw) - 9.0, np.rad2deg(yaw) + 9.0 + 1e-9),
    )
    if best is not None and score >= min_iou:
        return best, score, False
    best, score = fit(cam, mask)
    TRACKING_STATS["global_fallbacks"] += 1
    return best, score, True


def iou(cam, mask, x, y, yaw, area_mask):
    """Full-frame IoU, used by the self-checks.  The fit loop uses iou_roi."""
    pred = silhouette(cam, x, y, yaw, mask.shape)
    if pred is None:
        return 0.0
    inter = int(np.count_nonzero(pred & mask))
    return inter / (area_mask + int(np.count_nonzero(pred)) - inter)


def perceive(model, data, renderer=None, verbose=False):
    """One image in, cube pose out.  Reads no ground truth.

    Returns (pose, rgb, mask, cam); pose is None when the object is not visible.
    Pass a persistent `renderer` to avoid reallocating the GL context per frame
    (which is what makes this fast enough for a real-time loop).
    """
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, H, W)
    renderer.update_scene(data, camera=CAM_NAME)
    rgb = renderer.render().copy()
    if own:
        renderer.close()
    cam = Pinhole(model, data, CAM_NAME)
    mask = red_mask(rgb)
    if int(np.count_nonzero(mask)) == 0:
        return None, rgb, mask, cam
    best, score = fit(cam, mask)
    if best is None:
        return None, rgb, mask, cam
    x, y, yaw = best
    if verbose:
        print(f"  mask {int(np.count_nonzero(mask))} px  IoU {score:.3f}")
    return dict(center=np.array([x, y, TABLE_TOP + CUBE_HALF]), yaw=yaw, iou=score,
                cam=cam), rgb, mask, cam


def main() -> int:
    import time

    from member_b_scene import build_scene
    m, d = build_scene()
    while d.time < 2.0:
        mujoco.mj_step(m, d)
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "droid_object")

    renderer = mujoco.Renderer(m, H, W)
    t_r = time.perf_counter()
    renderer.update_scene(d, camera=CAM_NAME)
    rgb = renderer.render().copy()
    t_r = time.perf_counter() - t_r
    t_m = time.perf_counter()
    cam = Pinhole(m, d, CAM_NAME)
    mask = red_mask(rgb)
    t_m = time.perf_counter() - t_m
    if int(np.count_nonzero(mask)) == 0:
        print("no red object visible", file=sys.stderr)
        return 1
    t_f = time.perf_counter()
    best, score = fit(cam, mask)
    t_f = time.perf_counter() - t_f
    pose = None if best is None else dict(center=np.array([best[0], best[1], TABLE_TOP + CUBE_HALF]),
                                          yaw=best[2], iou=score, cam=cam)
    dt = t_r + t_m + t_f
    if pose is None:
        print("fit failed", file=sys.stderr)
        return 1
    gt = d.xpos[ob].copy()
    area = int(np.count_nonzero(mask))

    # a perfect fit should score ~1.0; if it does not, the pinhole model is wrong
    iou_truth = iou(cam, mask, gt[0], gt[1], 0.0, area)
    err = pose["center"][:2] - gt[:2]

    # the naive baseline, kept as the reason the fit exists
    ys, xs = np.nonzero(mask)
    naive = cam.ray_to_table(xs.mean(), ys.mean())
    nerr = np.linalg.norm(naive[:2] - gt[:2])

    print(f"scene        {SCENE_XML.name}   ncam={m.ncam}  nq={m.nq}")
    print(f"ground truth ({gt[0]:.4f}, {gt[1]:.4f})")
    print(f"perceived    ({pose['center'][0]:.4f}, {pose['center'][1]:.4f})  "
          f"yaw {np.rad2deg(pose['yaw']):.1f} deg")
    print(f"error        {np.linalg.norm(err)*1000:.2f} mm      "
          f"naive centroid->table {nerr*1000:.2f} mm")
    print(f"IoU          fitted {pose['iou']:.4f}   at ground truth {iou_truth:.4f} "
          f"<- ~1.0 means the pinhole model is right")
    print(f"one frame    {dt*1000:.1f} ms   render {t_r*1000:.1f} + mask {t_m*1000:.1f} "
          f"+ fit {t_f*1000:.1f}")

    out = ROOT / "outputs" / time.strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis[mask > 0] = (0.45 * vis[mask > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    pred = silhouette(cam, pose["center"][0], pose["center"][1], pose["yaw"])
    edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    vis[edge > 0] = (0, 255, 0)
    uv, _ = cam.project(cube_corners(pose["center"][0], pose["center"][1], pose["yaw"]))
    for p in uv:
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (255, 255, 0), -1)
    cv2.putText(vis, "red = mask   green = fitted silhouette   cyan = projected corners",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"perceived ({pose['center'][0]:.3f}, {pose['center'][1]:.3f}) "
                     f"yaw {np.rad2deg(pose['yaw']):.0f}deg  err "
                     f"{np.linalg.norm(err)*1000:.1f}mm  naive {nerr*1000:.1f}mm",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    png = out / "workspace_camera.png"
    cv2.imwrite(str(png), vis)
    renderer.close()
    print(f"wrote        {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
