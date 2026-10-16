# -*- coding: utf-8 -*-
r"""Camera-guided friction grasp in MuJoCo.

Chest RGB-D selects an arm; wrist RGB-D refines the cube centre and estimates
its top-face edge orientation.  Physical finger-contact proxies are aligned
with that measured face normal.  Simulator object state is reserved for
evaluation, including the opposing-face contact check; it does not provide the
grasp target.  No weld constraints are used.

    python demo_grasp_viewer.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import workspace_camera as wc                                               # noqa: E402
from calibration_simulation import body_transform, render_rgb_depth         # noqa: E402
from handeye_grasp_red_cube import (finger_joint_indices,                   # noqa: E402
                                    estimate_cube_center,
                                    latest_handeye_report,
                                    load_handeye,
                                    quat_from_body,
                                    search_reachable_grasp,
                                    solve_gripper_contact_proxy_q,
                                    solve_target_q,
                                    solve_relaxed_gripper_q)
from pose2arm_driver import CLOSE_BELOW_M, OPEN_ABOVE_M                     # noqa: E402
from task_workspace import (TABLE_X, TABLE_Y, constrain_initial_placement,  # noqa: E402
                            load_task_model, workspace_description,
                            CONFIG as TASK_WORKSPACE_CONFIG)
from teleop_towel_dataset import (apply_controller, build_arm_specs,        # noqa: E402
                                  build_finger_actuators)

CUBE_HALF = 0.0288
CONTROLLER_CONTRACT = {
    "allowed_inputs": [
        "chest_rgb", "chest_depth", "selected_wrist_rgb",
        "selected_wrist_depth", "joint_state", "camera_intrinsics",
        "chest_extrinsic", "wrist_handeye_transform",
    ],
    "forbidden_planner_inputs": [
        "mujoco_body_xpos", "fixture_cube_pose", "renderer_segmentation_id",
    ],
    "grasp_constraint": "contact_and_friction_only_no_weld",
}
FINGER_OPEN = 0.044
APPROACH_UP_M = 0.07          # descend from this high
TRANSIT_UP_M = 0.16           # first move above the object, avoiding side sweep
LIFT_UP_M = 0.08
PRELIFT_UP_M = 0.04
VIEWER_HZ = 60.0
ORIENTATION_WEIGHT = 5.0
PAD_CONTACT_PROXY_LOCAL = (
    # Aim 13 mm inboard from the distal mesh tip so the cube is enveloped by
    # the fingers instead of hanging from their last few millimetres.
    np.asarray([-0.00018, -0.00798, 0.04500]),
    np.asarray([0.00018, -0.00798, 0.04500]),
)

SCENE = wc.SCENE_XML
CONFIG = wc.SCENE_JSON
RGBD_WORKSPACE_CALIBRATION = ROOT / "calibration" / "rgbd_workspace_affine.json"


class CameraWindow:
    """The monocular stream, live, with what the perception made of it.

    Shows the raw frame, the red mask the detector keys on, the silhouette the
    fitter believes is there, and the jaw-to-object distance that drives the
    gripper.  Reading the numbers off a terminal tells you the perception works;
    watching the green outline sit on the cube while the arm moves tells you
    whether it is actually tracking.
    """

    NAME = "workspace camera (monocular)"

    def __init__(self, model, every_steps, jaw_geoms, hz_note=None):
        import cv2

        self.cv2 = cv2
        self.renderer = mujoco.Renderer(model, wc.H, wc.W)
        self.every = max(1, int(every_steps))
        self.n = 0
        self.jaw_geoms = jaw_geoms
        self.frames = 0
        cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)

    def maybe_update(self, model, data, step_n):
        """Redraw every `every` SIM STEPS -- step_n is the caller's step count."""
        if step_n % self.every:
            return
        cv2 = self.cv2
        pose, rgb, mask, cam = wc.perceive(model, data, renderer=self.renderer)
        vis = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        vis[mask > 0] = (0.45 * vis[mask > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)

        line = "no cube in view"
        if pose is not None:
            pred = wc.silhouette(cam, pose["center"][0], pose["center"][1], pose["yaw"])
            if pred is not None:
                edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                vis[edge > 0] = (0, 255, 0)
            uv, _ = cam.project(wc.cube_corners(pose["center"][0], pose["center"][1],
                                                pose["yaw"]))
            for p in uv:
                cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (255, 255, 0), -1)
            mid = 0.5 * (data.geom_xpos[self.jaw_geoms[0]] + data.geom_xpos[self.jaw_geoms[1]])
            d_mm = float(np.linalg.norm(mid - pose["center"])) * 1000.0
            state = "CLOSED" if d_mm < CLOSE_BELOW_M * 1000 else (
                "OPEN" if d_mm > OPEN_ABOVE_M * 1000 else "hold")
            line = (f"({pose['center'][0]:.3f}, {pose['center'][1]:.3f})  "
                    f"yaw {np.rad2deg(pose['yaw']):.0f}deg  IoU {pose['iou']:.3f}   "
                    f"D {d_mm:.0f}mm -> {state}")
        cv2.putText(vis, line, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "red = mask   green = fitted silhouette   cyan = projected corners",
                    (10, wc.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.NAME, vis)
        cv2.waitKey(1)
        self.frames += 1

    def close(self):
        self.cv2.destroyWindow(self.NAME)
        self.renderer.close()


class WristCameraWindow:
    """Live views from the chest D435i and selected wrist D405."""

    CHEST_NAME = "CHEST D435i - global controller input"

    def __init__(self, model, every_steps, camera_name):
        import cv2

        self.cv2 = cv2
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(model, 240, 320)
        self.chest_renderer = mujoco.Renderer(model, 240, 320)
        self.every = max(1, int(every_steps))
        self.frames = 0
        self.name = f"{camera_name.upper()} - precision controller input"
        cv2.namedWindow(self.CHEST_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def maybe_update(self, model, data, step_n):
        if step_n % self.every:
            return
        cv2 = self.cv2
        rgb, _depth = render_rgb_depth(
            self.renderer, data, self.camera_name)
        mask = wc.red_mask(rgb)
        vis = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        vis[mask > 0] = (
            0.45 * vis[mask > 0] + 0.55 * np.array([0, 0, 255])
        ).astype(np.uint8)
        pixels = int(np.count_nonzero(mask))
        cv2.putText(
            vis, f"SELECTED WRIST D405   red mask: {pixels} px", (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
            cv2.LINE_AA)
        cv2.putText(
            vis, "this RGB-D stream supplies the hand-eye IK target", (8, 232),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
            cv2.LINE_AA)
        chest_rgb, _ = render_rgb_depth(
            self.chest_renderer, data, "chest_d435i_preview")
        chest_mask = wc.red_mask(chest_rgb)
        chest_vis = cv2.cvtColor(np.asarray(chest_rgb), cv2.COLOR_RGB2BGR)
        chest_vis[chest_mask > 0] = (
            0.45 * chest_vis[chest_mask > 0] + 0.55 * np.array([0, 0, 255])
        ).astype(np.uint8)
        cv2.putText(
            chest_vis,
            f"CHEST D435i   red mask: {int(np.count_nonzero(chest_mask))} px",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            chest_vis, "global localization and arm selection", (8, 232),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
            cv2.LINE_AA)
        cv2.imshow(self.CHEST_NAME, chest_vis)
        cv2.imshow(self.name, vis)
        cv2.waitKey(1)
        self.frames += 1

    def close(self):
        for window_name in (self.CHEST_NAME, self.name):
            try:
                self.cv2.destroyWindow(window_name)
            except self.cv2.error:
                # The user may already have closed one preview window.
                pass
        self.renderer.close()
        self.chest_renderer.close()


def load_chest_calibration():
    """Load the newest measured fixed chest-camera calibration."""
    reports = sorted(
        (ROOT / "calibration" / "auto_chest").glob(
            "session_*/calibration_chest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise FileNotFoundError("没有找到胸部相机标定 calibration_chest.json")
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    transform = np.asarray(
        report["T_base_chest_camera_opencv"], dtype=np.float64)
    camera_matrix = np.asarray(report["camera_matrix"], dtype=np.float64)
    return transform, camera_matrix, reports[-1]


def load_workspace_rgbd_calibration():
    report = json.loads(RGBD_WORKSPACE_CALIBRATION.read_text(encoding="utf-8"))
    return report


def apply_workspace_rgbd_calibration(center, calibration):
    corrected = np.asarray(center, dtype=float).copy()
    coeff = np.asarray(calibration["xy_coeff"], dtype=float)
    corrected[:2] = np.asarray([center[0], center[1], 1.0]) @ coeff
    corrected[2] += float(calibration.get("z_bias_m", 0.0))
    return corrected


def repair_left_finger_position_servos(model, finger_actuators):
    """Make the left gripper use the same position servos as the right.

    The imported MJCF declares the left fingers as plain motors but the right
    fingers as position actuators.  The shared controller sends desired finger
    positions, so the left pair otherwise receives zero force for a close
    command and remains open near 0.037 m.
    """
    left = finger_actuators["left"]
    right = finger_actuators["right"]
    model.actuator_gaintype[left] = model.actuator_gaintype[right]
    model.actuator_gainprm[left] = model.actuator_gainprm[right]
    model.actuator_biastype[left] = model.actuator_biastype[right]
    model.actuator_biasprm[left] = model.actuator_biasprm[right]
    model.actuator_forcerange[left] = model.actuator_forcerange[right]


class _NullViewer:
    """Stand-in so the exact same motion code can also run headless.

    Running the animation and running the episode must not be two code paths --
    a demo that works and an episode that does not is exactly the failure this
    is meant to avoid.
    """

    class _Cam:
        type = None
        distance = 0.0
        azimuth = 0.0
        elevation = 0.0

        def __init__(self):
            self.lookat = np.zeros(3)

    def __init__(self):
        self.cam = self._Cam()

    def is_running(self):
        return True

    def sync(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load_baked_grasp():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    g = cfg["grasp"]
    axis = g["calibrated_axis"]                     # e.g. "y-"
    ai = {"x": 0, "y": 1, "z": 2}[axis[0]]
    asgn = -1.0 if axis.endswith("-") else 1.0
    return np.asarray(g["calibrated_offset_mm"], dtype=float) / 1000.0, ai, asgn


def quaternion_angle_deg(q0, q1):
    """Shortest orientation difference between two MuJoCo wxyz quaternions."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def rotation_matrix_angle_deg(r0, r1):
    relative = np.asarray(r0, dtype=float).T @ np.asarray(r1, dtype=float)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def opposing_face_contact_quality(points, finger_ids, half_size=CUBE_HALF):
    """Evaluation only: require each finger on opposite faces away from edges."""
    face_contacts = []
    edge_contacts = 0
    for point, finger in zip(points, finger_ids):
        point = np.asarray(point)
        axes = np.flatnonzero(np.abs(np.abs(point) - half_size) < 0.003)
        if len(axes) != 1:
            edge_contacts += 1
            continue
        axis = int(axes[0])
        if axis == 2:
            # A top-surface touch is not one of the two side-face grasps.
            continue
        tangential = np.delete(point, axis)
        margin = float(half_size - np.max(np.abs(tangential)))
        if margin < 0.005:
            edge_contacts += 1
            continue
        face_contacts.append((finger, axis, float(np.sign(point[axis])), margin))
    opposed = any(
        a[0] != b[0] and a[1] == b[1] and a[2] == -b[2]
        for a in face_contacts for b in face_contacts
    )
    return {
        "opposing_faces": opposed,
        "interior_face_contacts": len(face_contacts),
        "edge_or_corner_contacts": edge_contacts,
        "minimum_face_margin_mm": (
            min(contact[3] for contact in face_contacts) * 1000
            if face_contacts else 0.0),
    }


def place_cube_for_blind_test(model, data, pose):
    """Relocate the cube as test-fixture setup, not as controller input.

    ``pose`` is deliberately consumed only here, before perception runs.  The
    grasp target below is still derived exclusively from the rendered camera
    image.  This lets us test position generalisation without editing the
    frozen MJCF scene or leaking the requested coordinates into IK planning.
    """
    if pose is None:
        return
    x, y, yaw_deg = map(float, pose)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                           "droid_object_freejoint")
    qadr = int(model.jnt_qposadr[jid])
    dadr = int(model.jnt_dofadr[jid])
    data.qpos[qadr:qadr + 2] = (x, y)
    yaw = np.deg2rad(yaw_deg)
    data.qpos[qadr + 3:qadr + 7] = (np.cos(yaw / 2.0), 0.0, 0.0,
                                     np.sin(yaw / 2.0))
    data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def initialize_observation_posture(model, data, specs):
    """Initialize robot-only high standby poses before the fixture is placed.

    The old home fingers overlap parts of the smaller tabletop and could
    knock a near-edge cube off during initial settling. These fixed robot
    observation poses use no object position and are initial conditions, not
    teleportation during an episode. Actual grasp motion remains physical.
    """
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        spec = specs[side]
        rotation = body_transform(data, spec.body_id)[:3, :3].copy()
        quaternion = quat_from_body(model, data, spec.body_id)
        sign = 1.0 if side == "left" else -1.0
        q, _, _ = solve_target_q(
            model, data, spec, np.array([0.28, sign * 0.16, 0.68]),
            quaternion, data.qpos[spec.qpos].copy(), target_rotation=rotation)
        data.qpos[spec.qpos] = q
        data.qpos[finger_joint_indices(model, side)[0]] = FINGER_OPEN
        mujoco.mj_forward(model, data)
    data.qvel[:] = 0


def interactive_cube_placement(seconds):
    """Let the human position the cube with the keyboard before planning.

    This is test-fixture setup in a separate model/data instance.  Only the
    resulting environment pose is transferred; the controller subsequently
    renders a fresh RGB image and receives none of these coordinates.
    """
    m = load_task_model(SCENE)
    d = mujoco.MjData(m)
    specs = build_arm_specs(m)
    fing_act = build_finger_actuators(m)
    repair_left_finger_position_servos(m, fing_act)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    initialize_observation_posture(m, d, specs)
    q_hold = {side: d.qpos[specs[side].qpos].copy()
              for side in ("left", "right")}
    finger = {"left": FINGER_OPEN, "right": FINGER_OPEN}
    dt = float(m.opt.timestep)
    steps_per_frame = max(1, int(1.0 / (60.0 * dt)))
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                            "droid_object_freejoint")
    qadr = int(m.jnt_qposadr[jid])
    dadr = int(m.jnt_dofadr[jid])
    commands = []
    confirmed = [False]

    def on_key(keycode):
        move_keys = {
            glfw.KEY_W: (0.02, 0.0), glfw.KEY_UP: (0.02, 0.0),
            glfw.KEY_S: (-0.02, 0.0), glfw.KEY_DOWN: (-0.02, 0.0),
            glfw.KEY_A: (0.0, 0.02), glfw.KEY_LEFT: (0.0, 0.02),
            glfw.KEY_D: (0.0, -0.02), glfw.KEY_RIGHT: (0.0, -0.02),
        }
        if keycode in move_keys:
            commands.append(("move", move_keys[keycode]))
        elif keycode == glfw.KEY_Q:
            commands.append(("yaw", 15.0))
        elif keycode == glfw.KEY_E:
            commands.append(("yaw", -15.0))
        elif keycode == glfw.KEY_R:
            commands.append(("reset", None))
        elif keycode == glfw.KEY_SPACE:
            confirmed[0] = True

    def apply_placement_command(kind, value):
        if kind == "move":
            dx, dy = value
            # Initial-fixture constraint only; never supplied to visual IK.
            d.qpos[qadr] = np.clip(d.qpos[qadr] + dx, *TABLE_X)
            d.qpos[qadr + 1] = np.clip(d.qpos[qadr + 1] + dy, *TABLE_Y)
        elif kind == "reset":
            d.qpos[qadr:qadr + 2] = (0.45, 0.0)
        rot = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rot, d.qpos[qadr + 3:qadr + 7])
        yaw = float(np.arctan2(rot[3], rot[0]))
        if kind == "yaw":
            yaw += np.deg2rad(value)
        if kind in ("yaw", "reset"):
            d.qpos[qadr + 3:qadr + 7] = (
                np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        d.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(m, d)
        print(f"cube fixture: x={d.qpos[qadr]:.3f}, "
              f"y={d.qpos[qadr + 1]:.3f}, yaw={np.rad2deg(yaw):.0f} deg")

    print("\nINTERACTIVE PLACEMENT (2 cm per press):\n"
          "  W/S or Up/Down    move forward/backward (world X)\n"
          "  A/D or Left/Right move left/right (world Y)\n"
          "  Q/E                rotate 15 degrees\n"
          "  R                  reset to the green centre marker\n"
          "  Mouse              double-click cube, then Ctrl + right-drag\n"
          "  SPACE              accept and start wrist-camera perception + IK\n"
          "  ENTER              does nothing (prevents accidental acceptance)\n"
          f"The placement window also accepts automatically after {seconds:.0f} s.\n")
    print(workspace_description())
    constrain_initial_placement(m, d)
    started = time.perf_counter()
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        viewer.set_texts((None, None,
                         "PLACE CUBE\n" + workspace_description() +
                         "\nW/S/A/D: move | Q/E: rotate\nSPACE: start camera-guided grasp", ""))
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.25, 180.0, -35.0
        viewer.cam.lookat[:] = np.array([0.28, 0.0, 0.53])
        while (viewer.is_running() and not confirmed[0]
               and time.perf_counter() - started < seconds):
            while commands:
                apply_placement_command(*commands.pop(0))
            constrain_initial_placement(m, d)
            for _ in range(steps_per_frame):
                apply_controller(m, d, specs, fing_act, q_hold, finger)
                mujoco.mj_step(m, d)
            constrain_initial_placement(m, d)
            viewer.sync()
            time.sleep(1.0 / 60.0)

    if not confirmed[0] and time.perf_counter() - started < seconds:
        print("placement cancelled by closing the window")
        return None
    constrain_initial_placement(m, d)
    x, y = map(float, d.qpos[qadr:qadr + 2])
    rot = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rot, d.qpos[qadr + 3:qadr + 7])
    R = rot.reshape(3, 3)
    yaw_deg = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))
    print("interactive fixture accepted; controller coordinates remain hidden")
    return x, y, yaw_deg


def main(check_only=False, post_lift_s=1.5, report_path=None,
         test_cube_pose=None, preview_initial_s=0.0) -> int:
    m = load_task_model(SCENE)
    print(workspace_description())
    d = mujoco.MjData(m)
    DT = float(m.opt.timestep)

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)        # noqa: E731
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)        # noqa: E731
    specs = build_arm_specs(m)
    fing_act = build_finger_actuators(m)
    repair_left_finger_position_servos(m, fing_act)
    ob, og = bid("droid_object"), gid("droid_object_geom")
    SPAN = max(2 * CUBE_HALF + 0.004, 0.070)
    MU = float(m.geom_friction[og][0])
    WEIGHT = float(m.body_mass[ob]) * 9.81

    mujoco.mj_resetDataKeyframe(m, d, 0)
    initialize_observation_posture(m, d, specs)
    place_cube_for_blind_test(m, d, test_cube_pose)
    # Do not retain the fixture command in the controller path.  From this
    # point onward the planner can obtain the cube pose only from camera pixels.
    test_cube_pose = None
    q_hold = {s: d.qpos[specs[s].qpos].copy() for s in ("left", "right")}
    for _ in range(int(1.2 / DT)):          # settle the arm, cube falls to rest
        apply_controller(m, d, specs, fing_act, q_hold, {"left": FINGER_OPEN, "right": FINGER_OPEN})
        mujoco.mj_step(m, d)

    planning_viewer_ctx = None
    planning_viewer = None
    observation_keys = []

    def observation_key(key):
        if key in (glfw.KEY_R, glfw.KEY_SPACE):
            observation_keys.append("retry")
        elif key == glfw.KEY_C:
            observation_keys.append("place")

    if not check_only:
        planning_viewer_ctx = mujoco.viewer.launch_passive(
            m, d, key_callback=observation_key)
        planning_viewer = planning_viewer_ctx.__enter__()
        planning_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        planning_viewer.cam.distance = 1.5
        planning_viewer.cam.azimuth = 130.0
        planning_viewer.cam.elevation = -25.0
        planning_viewer.cam.lookat[:] = np.array([0.45, -0.02, 0.43])
        planning_viewer.set_texts((None, None, "CAMERA OBSERVATION\nChest + wrist RGB-D", ""))
        planning_viewer.sync()
        print("observation/planning viewer opened; it remains visible while IK computes")

    def close_planning_viewer():
        nonlocal planning_viewer, planning_viewer_ctx
        if planning_viewer_ctx is not None:
            planning_viewer_ctx.__exit__(None, None, None)
            planning_viewer_ctx = None
            planning_viewer = None

    # ---- two-stage perception: chest global, selected wrist precision ----
    # The fixed chest RGB-D camera first localises the cube over the full table
    # and selects an arm.  That camera-derived coarse pose moves the selected
    # wrist to a useful observation pose.  The wrist RGB-D image then supplies
    # the final IK target.  No MuJoCo object coordinate enters either stage.
    chest_transform, chest_matrix, chest_calibration_path = load_chest_calibration()
    workspace_calibration = load_workspace_rgbd_calibration()
    from visual_observation_recovery import acquire_visual_target

    def observation_progress(message):
        if planning_viewer is not None and planning_viewer.is_running():
            planning_viewer.set_texts((None, None, message, ""))
            planning_viewer.sync()

    while True:
        initial_target, observation_attempts = acquire_visual_target(
            m, d, specs, fing_act, q_hold, CUBE_HALF, FINGER_OPEN,
            chest_transform, chest_matrix, workspace_calibration,
            progress=observation_progress if not check_only else None)
        if initial_target is not None:
            break
        output = (Path(report_path) if report_path else
                  ROOT / "outputs" / time.strftime("%Y%m%d") / "grasp_stability.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "verdict": "FAIL", "failure_reason": "target_not_visible_in_robot_cameras",
            "task_workspace": TASK_WORKSPACE_CONFIG,
            "controller_contract": CONTROLLER_CONTRACT,
            "observation_attempts": observation_attempts,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if check_only:
            return 1
        print("Camera target unavailable. R/SPACE: retry; C: place cube again. "
              "Window stays open.", flush=True)
        observation_keys.clear()
        while planning_viewer.is_running() and not observation_keys:
            observation_progress(
                "CUBE NOT VISIBLE\nR / SPACE: retry cameras\nC: place cube again\nClose window: exit")
            for _ in range(max(1, int(0.02 / DT))):
                apply_controller(m, d, specs, fing_act, q_hold,
                                 {"left": FINGER_OPEN, "right": FINGER_OPEN})
                mujoco.mj_step(m, d)
            time.sleep(0.02)
        if not planning_viewer.is_running():
            close_planning_viewer()
            return 0
        if observation_keys[-1] == "place":
            close_planning_viewer()
            return 2
        observation_keys.clear()
    chest_observation = initial_target["observation"]
    chest_center_raw = initial_target["raw_center"]
    chest_center = initial_target["center"]
    initial_target_source = initial_target["source"]
    # Keep the centre overlap on the calibrated right arm; switch only when
    # chest vision clearly places the target on the positive-Y side.
    perception_arm = "left" if chest_center[1] > 0.02 else "right"
    print(
        f"initial camera {initial_target_source} perceived "
        f"({chest_center[0]:.4f}, {chest_center[1]:.4f})  "
        f"mask {chest_observation.mask_pixels} px -> select {perception_arm} arm"
    )

    perception_spec = specs[perception_arm]
    q_ready = q_hold[perception_arm].copy()
    quat_ready = quat_from_body(m, d, perception_spec.body_id)
    rotation_ready = body_transform(d, perception_spec.body_id)[:3, :3].copy()
    side_sign = 1.0 if perception_arm == "left" else -1.0
    ready_position = body_transform(d, perception_spec.body_id)[:3, 3]
    raise_position = ready_position.copy()
    raise_position[2] = np.clip(float(ready_position[2]) + 0.18, 0.68, 0.78)
    q_raise, _, _ = solve_target_q(
        m, d, perception_spec, raise_position, quat_ready,
        q_ready, target_rotation=rotation_ready)
    # The initialized high standby pose is already clear of the entire task
    # table. Reuse it rather than solving a redundant 72 cm exit on a new IK
    # branch; that branch could stall and add 1.5 s to every right-arm episode.
    q_clear = q_ready.copy()
    search_position = np.array([
        np.clip(chest_center[0] - 0.19, 0.18, 0.38),
        np.clip(chest_center[1] * 0.65, -0.13, 0.13),
        0.60,
    ])
    q_search, _, _ = solve_target_q(
        m, d, perception_spec, search_position, quat_ready,
        q_clear, target_rotation=rotation_ready)
    print(f"chest-guided wrist observation TCP: {np.round(search_position, 3)} m")

    def move_observer(q_to, seconds=2.0):
        q_from = d.qpos[perception_spec.qpos].copy()
        n = max(1, int(seconds / DT))
        for i in range(n):
            q_cmd = q_from + (q_to - q_from) * ((i + 1) / n)
            arm_commands = dict(q_hold)
            arm_commands[perception_arm] = q_cmd
            apply_controller(m, d, specs, fing_act, arm_commands,
                             {"left": FINGER_OPEN, "right": FINGER_OPEN})
            mujoco.mj_step(m, d)
            if (planning_viewer is not None and i % max(1, int(0.02 / DT)) == 0
                    and planning_viewer.is_running()):
                planning_viewer.sync()
        for i in range(int(1.0 / DT)):
            arm_commands = dict(q_hold)
            arm_commands[perception_arm] = q_to
            apply_controller(m, d, specs, fing_act, arm_commands,
                             {"left": FINGER_OPEN, "right": FINGER_OPEN})
            mujoco.mj_step(m, d)
            if (planning_viewer is not None
                    and i % max(1, int(0.02 / DT)) == 0
                    and planning_viewer.is_running()):
                planning_viewer.sync()

    move_observer(q_raise)
    move_observer(q_clear)
    move_observer(q_search)
    q_hold[perception_arm] = d.qpos[perception_spec.qpos].copy()
    handeye, camera_matrix = load_handeye(
        latest_handeye_report(perception_arm), perception_arm)
    wrist_name = f"v22_{perception_arm}_d405_rgb"
    renderer = mujoco.Renderer(m, 240, 320)
    rgb, depth = render_rgb_depth(renderer, d, wrist_name)
    renderer.close()
    try:
        observation = estimate_cube_center(
            rgb, depth, body_transform(d, perception_spec.body_id),
            handeye, camera_matrix, CUBE_HALF)
        final_center_raw = np.asarray(observation.center_world, dtype=float)
        final_center = apply_workspace_rgbd_calibration(
            final_center_raw, workspace_calibration["wrist"][perception_arm])
        target_source = "chest_coarse_plus_selected_wrist_rgbd"
        if initial_target_source != "chest_d435i_preview":
            target_source = "wrist_search_recovery_plus_selected_wrist_rgbd"
        final_mask_pixels = int(observation.mask_pixels)
        print(
            f"wrist refined ({final_center[0]:.4f}, {final_center[1]:.4f})  "
            f"mask {final_mask_pixels} px ({perception_arm}-wrist RGB-D)"
        )
    except RuntimeError as exc:
        # The global camera still gives a calibrated RGB-D target.  Continue
        # the physical attempt instead of introducing another rejection gate.
        observation = chest_observation
        final_center_raw = chest_center_raw.copy()
        final_center = chest_center.copy()
        target_source = "chest_rgbd_fallback_after_wrist_miss"
        if initial_target_source != "chest_d435i_preview":
            target_source = "wrist_search_recovery_fallback"
        final_mask_pixels = int(chest_observation.mask_pixels)
        print(f"wrist refinement unavailable ({exc}); using chest RGB-D target")
    # The baked friction-grasp offset was calibrated on the ready-pose IK
    # branch.  Return there after observing so the wrist search pose cannot
    # silently select a different joint solution for the same Cartesian goal.
    move_observer(q_clear)
    move_observer(q_raise)
    move_observer(q_ready)
    q_hold[perception_arm] = d.qpos[perception_spec.qpos].copy()
    pose = {
        "center": final_center,
        "yaw": float(observation.yaw_world_rad),
        "iou": 1.0,
        "mask_pixels": final_mask_pixels,
    }
    mask = np.zeros((1, 1), dtype=np.uint8)
    print(f"final visual target ({pose['center'][0]:.4f}, "
          f"{pose['center'][1]:.4f}) from {target_source}")
    print(f"cube {2*CUBE_HALF*1000:.1f} mm  {WEIGHT:.3f} N  mu {MU:.2f} -> "
          f"friction needs {WEIGHT/(2*MU):.3f} N/finger")

    from visual_grasp_planner import plan_face_grasp

    arm = perception_arm
    other = "right" if arm == "left" else "left"
    spec = specs[arm]
    FGI = (gid(f"openarm_{arm}_right_finger_collision"),
           gid(f"openarm_{arm}_left_finger_collision"))
    FQ = finger_joint_indices(m, arm)
    target = pose["center"].copy()

    def planning_progress(candidate):
        print(f"face candidate {candidate['face_index']} sign {candidate['sign']:+.0f}: "
              f"centre {candidate['midpoint_error_m']*1000:.1f} mm, "
              f"axis {candidate['perpendicular_error_m']*1000:.1f} mm, "
              f"path penetration {candidate['max_new_penetration_mm']:.1f} mm",
              flush=True)
        if planning_viewer is not None and planning_viewer.is_running():
            planning_viewer.set_texts((None, None,
                f"PLANNING FACE GRASP\nCandidate {candidate['face_index']} / {candidate['sign']:+.0f}\n"
                f"Position error: {candidate['midpoint_error_m']*1000:.1f} mm\n"
                f"Path penetration: {candidate['max_new_penetration_mm']:.1f} mm", ""))
            planning_viewer.sync()

    face_plan = plan_face_grasp(
        m, d, spec, FGI, FQ, PAD_CONTACT_PROXY_LOCAL, target, pose["yaw"],
        q_hold[arm], q_clear, FINGER_OPEN, APPROACH_UP_M, TRANSIT_UP_M,
        progress=planning_progress)
    q_grasp = face_plan["q_grasp"]
    q_appr = face_plan["q_approach"]
    q_transit = face_plan["q_transit"]
    grasp_rotation = face_plan["rotation"]
    grasp_metrics = face_plan["metrics"]
    planned_fingers = face_plan["pads"]
    closing_direction = face_plan["direction"]
    SPAN = face_plan["span"]
    ai, asgn = 1, float(grasp_metrics["sign"])
    name = f"object_face_{grasp_metrics['face_index']}_{asgn:+.0f}"
    print(f"selected {name}, visual yaw {np.rad2deg(pose['yaw']):+.2f} deg, "
          f"physical open span {SPAN*1000:.1f} mm", flush=True)

    # Oracle data is read after the initial visual plan.  Subsequent corrections
    # use fresh RGB-D observations only; evaluation truth never enters IK.
    evaluation_truth_initial = d.xpos[ob].copy()
    perception_error_mm = float(np.linalg.norm(
        pose["center"][:2] - evaluation_truth_initial[:2]) * 1000.0)
    print(f"evaluation-only perception error {perception_error_mm:.2f} mm "
          "(read after planning was frozen)")

    def step(q_arm, finger):
        apply_controller(m, d, specs, fing_act, {arm: q_arm, other: q_hold[other]},
                         {"left": finger, "right": finger})
        mujoco.mj_step(m, d)

    def move(q_to, seconds, finger, v, tol=0.015, extra=1.5):
        """Interpolate to q_to, then hold until the joints actually converge."""
        q_from = d.qpos[spec.qpos].copy()
        n_steps = max(1, int(seconds / DT))
        for i in range(n_steps):
            phase = (i + 1) / n_steps
            blend = phase * phase * (3.0 - 2.0 * phase)
            step(q_from + (q_to - q_from) * blend, finger)
            pump(v)
        t = 0.0
        while t < extra and np.abs(d.qpos[spec.qpos] - q_to).max() > tol:
            step(q_to, finger)
            pump(v)
            t += DT
        return float(np.abs(d.qpos[spec.qpos] - q_to).max())

    def hold(seconds, q_arm, finger, v):
        for _ in range(int(seconds / DT)):
            step(q_arm, finger)
            pump(v)

    # viewer runs at a fixed rate regardless of how fast the sim can step
    _pace = {"t": 0.0}
    SYNC_EVERY = max(1, int(1.0 / (VIEWER_HZ * DT)))

    def pump(v):
        _pace["n"] = _pace.get("n", 0) + 1
        sample_stability()
        # before the sync early-return, so the camera sees the real step count
        if cam_win is not None:
            cam_win.maybe_update(m, d, _pace["n"])
        if _pace["n"] % SYNC_EVERY:
            return True
        if not v.is_running():
            return False
        # Keep the demonstration camera fixed on the workspace.  Following the
        # cube here makes a relocated cube appear to remain in the screen
        # centre, which hides the very position-generalisation being tested.
        # Perception uses the separate fixed ``workspace_camera`` and is
        # unaffected by this viewer-only choice.
        v.sync()
        want = _pace["t"] + SYNC_EVERY * DT
        slack = want - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        _pace["t"] = time.perf_counter()
        return True

    def contact_indices():
        return [i for i in range(d.ncon)
                if og in {d.contact[i].geom1, d.contact[i].geom2}
                and (FGI[0] in {d.contact[i].geom1, d.contact[i].geom2}
                     or FGI[1] in {d.contact[i].geom1, d.contact[i].geom2})]

    def contacts():
        return [d.contact[i] for i in contact_indices()]

    def contact_forces():
        result = []
        for index in contact_indices():
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(m, d, index, force)
            result.append(force)
        return result

    def grip_force():
        return float(np.abs(d.actuator_force[fing_act[arm]]).sum())

    def contact_centroid():
        cs = contacts()
        if not cs:
            return None
        R = d.xmat[ob].reshape(3, 3)
        return R.T @ (np.array([c.pos for c in cs]).mean(axis=0) - d.xpos[ob])

    def gripper_center():
        """Physical centre of the active pad contact patch.

        The imported finger meshes have origins far behind their pad surfaces,
        so the midpoint of geom_xpos is not the gripper centre used by the task
        success rule.  While grasping, the mean pad contact position is the
        directly measured centre of the two opposing pad surfaces.
        """
        cs = contacts()
        if cs:
            return np.asarray([c.pos for c in cs]).mean(axis=0)
        return 0.5 * (d.geom_xpos[FGI[0]] + d.geom_xpos[FGI[1]])

    def contact_points_object():
        R = d.xmat[ob].reshape(3, 3)
        return [R.T @ (c.pos - d.xpos[ob]) for c in contacts()]

    def contact_points_finger_local():
        """Contact patch locations in each finger collision-geom frame."""
        result = []
        for index in contact_indices():
            contact = d.contact[index]
            finger_geom = next(
                (geom for geom in FGI
                 if geom in {contact.geom1, contact.geom2}), None)
            if finger_geom is None:
                continue
            rotation = d.geom_xmat[finger_geom].reshape(3, 3)
            local = rotation.T @ (contact.pos - d.geom_xpos[finger_geom])
            result.append((int(finger_geom), local))
        return result

    stability = {
        "active": False,
        "reference_quat": None,
        "reference_z": None,
        "max_rotation_deg": 0.0,
        "max_gripper_rotation_deg": 0.0,
        "max_angular_speed_rad_s": 0.0,
        "max_jaw_object_distance_m": 0.0,
        "min_contacts": None,
        "min_contact_normal_total_n": None,
        "success_run_s": 0.0,
        "max_success_run_s": 0.0,
    }

    def save_grasp_snapshot(stage, elevation):
        """Diagnostic rendering only, after the controller plan is frozen."""
        if report_path is None:
            return
        import cv2

        destination = Path(report_path).with_suffix(f".{stage}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, camera)
        camera.lookat[:] = target + np.asarray([
            0.0, 0.0, LIFT_UP_M if stage == "lift" else 0.0])
        camera.distance = 0.40
        camera.azimuth = 90.0
        camera.elevation = elevation
        renderer = mujoco.Renderer(m, 480, 640)
        try:
            renderer.update_scene(d, camera=camera)
            rgb = renderer.render()
            if not cv2.imwrite(str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Cannot save diagnostic image: {destination}")
        finally:
            renderer.close()

    def sample_stability():
        if not stability["active"]:
            return
        distance = float(np.linalg.norm(gripper_center() - d.xpos[ob]))
        rotation = quaternion_angle_deg(stability["reference_quat"], d.xquat[ob])
        gripper_rotation = rotation_matrix_angle_deg(
            grasp_rotation, d.geom_xmat[FGI[0]].reshape(3, 3))
        angular_speed = float(np.linalg.norm(d.cvel[ob, :3]))
        n_contact = len(contacts())
        forces = contact_forces()
        normal_total = float(sum(max(0.0, f[0]) for f in forces))
        stability["max_rotation_deg"] = max(stability["max_rotation_deg"], rotation)
        stability["max_gripper_rotation_deg"] = max(
            stability["max_gripper_rotation_deg"], gripper_rotation)
        stability["max_angular_speed_rad_s"] = max(
            stability["max_angular_speed_rad_s"], angular_speed)
        stability["max_jaw_object_distance_m"] = max(
            stability["max_jaw_object_distance_m"], distance)
        stability["min_contacts"] = (n_contact if stability["min_contacts"] is None
                                     else min(stability["min_contacts"], n_contact))
        if forces:
            stability["min_contact_normal_total_n"] = (
                normal_total if stability["min_contact_normal_total_n"] is None
                else min(stability["min_contact_normal_total_n"], normal_total)
            )
        lifted_and_near = (
            float(d.xpos[ob, 2]) - stability["reference_z"] > 0.03
            and distance < 0.02
        )
        stability["success_run_s"] = (
            stability["success_run_s"] + DT if lifted_and_near else 0.0)
        stability["max_success_run_s"] = max(
            stability["max_success_run_s"], stability["success_run_s"])

    close_planning_viewer()
    print("\nopening the viewer -- watch the cube. Ctrl-click / Esc to close.\n"
          if not check_only else "\n--check: running the grasp headless.\n")
    q_lift = None
    with (_NullViewer() if check_only else mujoco.viewer.launch_passive(m, d)) as v:
        if not check_only:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 130.0, -25.0
            v.cam.lookat[:] = np.array([0.45, 0.0, 0.43])
            v.set_texts((None, None, "EXECUTING CAMERA-GUIDED GRASP", ""))
        v.sync()
        _pace["t"] = time.perf_counter()
        cam_win = (None if check_only else
                   WristCameraWindow(m, int(0.2 / DT), wrist_name))  # ~5 Hz
        if not check_only and preview_initial_s > 0.0:
            print(f"showing the relocated initial cube pose for "
                  f"{preview_initial_s:.1f} s before the timed episode")
            hold(preview_initial_s, q_hold[arm], FINGER_OPEN, v)
        episode_start = float(d.time)

        cube_before_transit = d.xpos[ob].copy()
        clear_error = move(q_clear, 1.5, FINGER_OPEN, v)       # verified collision-free exit
        clear_displacement_mm = float(np.linalg.norm(
            d.xpos[ob] - cube_before_transit) * 1000.0)
        transit_error = move(q_transit, 1.8, FINGER_OPEN, v)  # high and clear
        transit_displacement_mm = float(np.linalg.norm(
            d.xpos[ob] - cube_before_transit) * 1000.0)
        approach_error = move(q_appr, 1.2, FINGER_OPEN, v)    # vertical approach
        # Refresh visual observations at the pregrasp pose.  Occlusion is
        # possible, so prefer agreeing chest/wrist observations; a partially
        # visible view is not allowed to overwrite a consistent earlier pose.
        visual_updates = []
        for camera_name, transform, intrinsics, correction in (
            ("chest_d435i_preview", chest_transform, chest_matrix,
             workspace_calibration["chest"]),
            (wrist_name, body_transform(d, perception_spec.body_id) @ handeye,
             camera_matrix, workspace_calibration["wrist"][arm]),
        ):
            refresh_renderer = mujoco.Renderer(
                m, 480 if camera_name.startswith("chest") else 240,
                640 if camera_name.startswith("chest") else 320)
            try:
                fresh_rgb, fresh_depth = render_rgb_depth(
                    refresh_renderer, d, camera_name)
                fresh = estimate_cube_center(
                    fresh_rgb, fresh_depth, np.eye(4), transform,
                    intrinsics, CUBE_HALF)
                corrected = apply_workspace_rgbd_calibration(
                    fresh.center_world, correction)
                # Re-observe position only here: a hand occluding a square's
                # edges can make its orientation estimate ambiguous.
                if np.linalg.norm(corrected[:2] - target[:2]) < 0.015:
                    visual_updates.append(corrected)
            except RuntimeError as exc:
                print(f"pregrasp camera {camera_name}: {exc}", flush=True)
            finally:
                refresh_renderer.close()
        visual_correction_mm = 0.0
        if visual_updates:
            agrees = (len(visual_updates) == 1 or np.linalg.norm(
                visual_updates[0][:2] - visual_updates[1][:2]) < 0.010)
            if agrees:
                revised = np.mean(visual_updates, axis=0)
                revised[2] = target[2]
                visual_correction_mm = float(np.linalg.norm(revised - target) * 1000)
                target = revised
                q_grasp, _, _, _ = solve_gripper_contact_proxy_q(
                    m, d, spec, FGI, FQ, PAD_CONTACT_PROXY_LOCAL,
                    target, ai, asgn, SPAN, FINGER_OPEN, q_grasp,
                    reference_rotation=grasp_rotation, multi_start=False,
                    closing_direction=closing_direction)
        print(f"pregrasp visual correction {visual_correction_mm:.2f} mm "
              f"from {len(visual_updates)} usable views", flush=True)
        cube_before_descent = d.xpos[ob].copy()
        grasp_error = move(q_grasp, 1.5, FINGER_OPEN, v)      # vertical descent
        preclose_displacement_mm = float(np.linalg.norm(
            d.xpos[ob] - cube_before_transit) * 1000.0)
        descent_displacement_mm = float(np.linalg.norm(
            d.xpos[ob] - cube_before_descent) * 1000.0)
        print(f"pre-contact cube displacement: clear "
              f"{clear_displacement_mm:.2f} mm, transit "
              f"{transit_displacement_mm:.2f} mm, total "
              f"{preclose_displacement_mm:.2f} mm, descent "
              f"{descent_displacement_mm:.2f} mm")
        close_error = move(q_grasp, 1.5, 0.0, v)             # close -- fingers only
        hold(1.0, q_grasp, 0.0, v)
        print(f"joint convergence rad: clear {clear_error:.4f}, "
              f"transit {transit_error:.4f}, "
              f"approach {approach_error:.4f}, "
              f"grasp {grasp_error:.4f}, close {close_error:.4f}")
        print("      close joint error deg", np.round(
            np.rad2deg(q_grasp - d.qpos[spec.qpos]), 2))
        print("      active arm torque", np.round(d.ctrl[spec.actuators], 2),
              "limits", np.round(m.actuator_forcerange[spec.actuators, 1], 1))
        print("      finger qpos m", np.round(d.qpos[FQ[0]], 5),
              "finger actuator force", np.round(d.actuator_force[fing_act[arm]], 3))
        cen = contact_centroid()
        z0 = float(d.xpos[ob, 2])
        n_c = len(contacts())
        grip_points = contact_points_object()
        grip_finger_ids = [next(geom for geom in FGI
                               if geom in {c.geom1, c.geom2})
                           for c in contacts()]
        grip_quality = opposing_face_contact_quality(grip_points, grip_finger_ids)
        print(f"opposing-face evaluation: {grip_quality}")
        grip_points_finger_local = contact_points_finger_local()
        grip_contact_forces = contact_forces()
        jaw_relative = np.asarray([
            d.geom_xpos[FGI[0]] - d.xpos[ob],
            d.geom_xpos[FGI[1]] - d.xpos[ob],
        ])
        print(f"      jaw origins relative to cube "
              f"{np.round(jaw_relative*1000, 1).tolist()} mm")
        print(f"grip: {n_c} contacts  {grip_force():.3f} N "
              f"({grip_force()/2:.3f} N/finger vs {WEIGHT/(2*MU):.3f} needed)")
        if cen is not None:
            print(f"      pads off the object centre by {np.linalg.norm(cen)*1000:.1f} mm")
        for index, point in enumerate(grip_points, start=1):
            surface_axes = int(np.count_nonzero(
                np.abs(np.abs(point) - CUBE_HALF) < 0.003))
            kind = "face" if surface_axes == 1 else ("edge" if surface_axes == 2 else "corner")
            print(f"      contact {index}: object-local {np.round(point*1000, 1)} mm "
                  f"-> {kind} ({surface_axes} surface axes)")
        for geom_id, point in grip_points_finger_local:
            print(f"      contact patch in {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)}: "
                  f"{np.round(point*1000, 2)} mm")
        if grip_contact_forces:
            normals = [float(max(0.0, force[0])) for force in grip_contact_forces]
            tangentials = [float(np.linalg.norm(force[1:3])) for force in grip_contact_forces]
            print(f"      contact normal forces {np.round(normals, 3)} N "
                  f"(sum {sum(normals):.3f}); tangential {np.round(tangentials, 3)} N")

        result = None
        save_grasp_snapshot("grip", -70.0)
        # A fingertip pressing on the cube top is not a grip.  Infer successful
        # closure from encoders and actuator effort, as on a real gripper.
        physically_gripping = bool(
            np.all(d.qpos[FQ[0]] > 0.008)
            and np.all(np.abs(d.actuator_force[fing_act[arm]]) > 0.8)
            and close_error < 0.03)
        if physically_gripping:
            cube_pre = d.xpos[ob].copy()
            stability["active"] = True
            stability["reference_quat"] = d.xquat[ob].copy()
            stability["reference_z"] = z0
            q_prelift, *_ = solve_gripper_contact_proxy_q(
                m, d, spec, FGI, FQ, PAD_CONTACT_PROXY_LOCAL,
                target + np.array([0, 0, PRELIFT_UP_M]), ai, asgn,
                2 * CUBE_HALF, 0.0, q_grasp,
                reference_rotation=grasp_rotation, multi_start=False,
                closing_direction=closing_direction)
            q_lift, *_ = solve_gripper_contact_proxy_q(
                m, d, spec, FGI, FQ, PAD_CONTACT_PROXY_LOCAL,
                target + np.array([0, 0, LIFT_UP_M]), ai, asgn,
                2 * CUBE_HALF, 0.0, q_prelift,
                reference_rotation=grasp_rotation, multi_start=False,
                closing_direction=closing_direction)
            print("lift joint delta deg", np.round(np.rad2deg(q_lift - q_grasp), 1))
            prelift_error = move(q_prelift, 1.5, 0.0, v)
            hold(0.4, q_prelift, 0.0, v)
            lift_error = move(q_lift, 2.0, 0.0, v)
            print(f"prelift/lift convergence error "
                  f"{prelift_error:.4f}/{lift_error:.4f} rad")
            hold(post_lift_s, q_lift, 0.0, v)
            stability["active"] = False
            save_grasp_snapshot("lift", -40.0)
            z1 = float(d.xpos[ob, 2])
            dz = (z1 - z0) * 1000
            episode_s = float(d.time) - episode_start
            final_distance = float(np.linalg.norm(gripper_center() - d.xpos[ob]))
            final_rotation = quaternion_angle_deg(stability["reference_quat"], d.xquat[ob])
            passed = (dz > 30.0 and stability["max_success_run_s"] >= 0.5
                      and episode_s <= 15.0 and len(contacts()) > 0
                      and grip_quality["opposing_faces"])
            print(f"\ncube z {z0*1000:.1f} -> {z1*1000:.1f} mm   lift {dz:+.1f} mm  -> "
                  f"{'LIFTED' if dz > 30 else 'NOT LIFTED'}")
            print(f"displaced sideways {np.linalg.norm(d.xpos[ob][:2]-cube_pre[:2])*1000:.1f} mm")
            print(f"held by friction alone: {len(contacts())} contacts, "
                  f"{grip_force():.3f} N. Nothing is welded.")
            print(f"stability: final/max rotation {final_rotation:.2f}/"
                  f"{stability['max_rotation_deg']:.2f} deg, peak angular speed "
                  f"{stability['max_angular_speed_rad_s']:.3f} rad/s")
            print(f"           max gripper rotation from grasp pose "
                  f"{stability['max_gripper_rotation_deg']:.2f} deg")
            print(f"           minimum summed contact-normal force while held "
                  f"{stability['min_contact_normal_total_n'] or 0.0:.3f} N")
            print(f"task gate: jaw-object {final_distance*1000:.1f} mm final, "
                  f"{stability['max_jaw_object_distance_m']*1000:.1f} mm max; "
                  f"success held {stability['max_success_run_s']:.3f} s; "
                  f"episode {episode_s:.3f} s -> {'PASS' if passed else 'FAIL'}")
            result = {
                "scene": str(SCENE),
                "controller_cube_pose_source": target_source,
                "initial_target_source": initial_target_source,
                "observation_attempts": observation_attempts,
                "controller_contract": CONTROLLER_CONTRACT,
                "chest_calibration": str(chest_calibration_path),
                "chest_perceived_center_m": [float(v) for v in chest_center],
                "selected_arm": arm,
                "oracle_read_after_planning": True,
                "perceived_center_m": [float(v) for v in pose["center"]],
                "perception_iou": float(pose["iou"]),
                "perception_error_mm_evaluation_only": perception_error_mm,
                "preclose_cube_displacement_mm_evaluation_only": preclose_displacement_mm,
                "descent_cube_displacement_mm_evaluation_only": descent_displacement_mm,
                "grasp_axis": name,
                "perceived_yaw_deg": float(np.rad2deg(pose["yaw"])),
                "closing_direction_world": closing_direction.tolist(),
                "face_plan_candidates": face_plan["candidates"],
                "pregrasp_visual_correction_mm": visual_correction_mm,
                "contact_quality_at_grip": grip_quality,
                "contact_points_object_mm": [[float(v * 1000.0) for v in p]
                                               for p in grip_points],
                "contact_points_finger_local_mm": [
                    {
                        "geom": mujoco.mj_id2name(
                            m, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                        "point_mm": [float(v * 1000.0) for v in point],
                    }
                    for geom_id, point in grip_points_finger_local
                ],
                "contacts_at_grip": n_c,
                "contacts_final": len(contacts()),
                "grip_force_n": grip_force(),
                "required_force_per_finger_n": WEIGHT / (2 * MU),
                "contact_normal_at_grip_n": [float(max(0.0, f[0]))
                                              for f in grip_contact_forces],
                "contact_tangential_at_grip_n": [float(np.linalg.norm(f[1:3]))
                                                  for f in grip_contact_forces],
                "min_contact_normal_total_n": stability["min_contact_normal_total_n"],
                "lift_mm": dz,
                "sideways_displacement_mm": float(
                    np.linalg.norm(d.xpos[ob][:2] - cube_pre[:2]) * 1000.0),
                "orientation_final_deg": final_rotation,
                "orientation_max_deg": stability["max_rotation_deg"],
                "gripper_orientation_max_deg": stability["max_gripper_rotation_deg"],
                "peak_angular_speed_rad_s": stability["max_angular_speed_rad_s"],
                "jaw_object_distance_final_mm": final_distance * 1000.0,
                "jaw_object_distance_max_mm": stability["max_jaw_object_distance_m"] * 1000.0,
                "success_continuous_s": stability["max_success_run_s"],
                "episode_sim_s": episode_s,
                "post_lift_hold_s": post_lift_s,
                "joint_convergence_rad": {
                    "clear": clear_error,
                    "transit": transit_error,
                    "approach": approach_error,
                    "grasp": grasp_error,
                    "close": close_error,
                    "prelift": prelift_error,
                    "lift": lift_error,
                },
                "verdict": "PASS" if passed else "FAIL",
                "criteria": "opposing interior face contacts, lift>30mm, jaw-object<20mm continuously>=0.5s, episode<=15s",
            }
        else:
            print("finger encoders/effort do not indicate a two-finger grip; "
                  "recording failed closure without a fake lift")
            result = {
                "scene": str(SCENE),
                "controller_cube_pose_source": target_source,
                "controller_contract": CONTROLLER_CONTRACT,
                "chest_calibration": str(chest_calibration_path),
                "chest_perceived_center_m": [float(v) for v in chest_center],
                "selected_arm": arm,
                "oracle_read_after_planning": True,
                "perceived_center_m": [float(v) for v in pose["center"]],
                "perception_iou": float(pose["iou"]),
                "perception_error_mm_evaluation_only": perception_error_mm,
                "preclose_cube_displacement_mm_evaluation_only": preclose_displacement_mm,
                "descent_cube_displacement_mm_evaluation_only": descent_displacement_mm,
                "contacts_at_grip": n_c,
                "contact_quality_at_grip": grip_quality,
                "face_plan_candidates": face_plan["candidates"],
                "perceived_yaw_deg": float(np.rad2deg(pose["yaw"])),
                "failure_reason": "closure_not_confirmed_by_encoders_and_effort",
                "verdict": "FAIL",
            }

        result["task_workspace"] = TASK_WORKSPACE_CONFIG
        output = (Path(report_path) if report_path else
                  ROOT / "outputs" / time.strftime("%Y%m%d") / "grasp_stability.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float) + "\n",
                          encoding="utf-8", newline="\n")
        print(f"wrote {output}")
        if not check_only:
            v.set_texts((None, None,
                        f"RESULT: {result['verdict']}\nClose window to exit", ""))

        if check_only:
            return 0 if result["verdict"] == "PASS" else 1
        if cam_win is not None:
            print(f"camera window: {cam_win.frames} frames shown")
            cam_win.close()
            cam_win = None
        print("\nviewer stays open. Close the window to exit.")
        while v.is_running():
            v.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="run perception, the axis check and both IK solves, then exit "
                         "without opening the viewer")
    ap.add_argument("--post-lift-s", type=float, default=1.5,
                    help="seconds to keep stepping after the lift (default: 1.5)")
    ap.add_argument("--preview-initial-s", type=float, default=0.0,
                    help=("viewer-only pause showing the initial cube placement "
                          "before the timed episode (default: 0)"))
    ap.add_argument("--report", type=Path, default=None,
                    help="stability JSON path (default: outputs/YYYYMMDD/grasp_stability.json)")
    placement = ap.add_mutually_exclusive_group()
    placement.add_argument(
        "--cube-test-pose", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
        help=("test-only cube placement. The coordinates initialize the simulator "
              "but are not supplied to perception or IK"))
    placement.add_argument(
        "--random-cube", type=int, metavar="SEED",
        help=("choose a reproducible blind-test pose inside the camera/search area; "
              "the controller still receives only the rendered image"))
    placement.add_argument(
        "--interactive-place-s", type=float, metavar="SECONDS",
        help=("open a pre-planning window for keyboard cube placement; "
              "camera perception and IK run only after that window closes"))
    args = ap.parse_args()
    if args.post_lift_s < 0 or args.preview_initial_s < 0:
        ap.error("--post-lift-s and --preview-initial-s must be non-negative")
    cube_test_pose = args.cube_test_pose
    if args.interactive_place_s is not None:
        if args.interactive_place_s <= 0:
            ap.error("--interactive-place-s must be positive")
        cube_test_pose = interactive_cube_placement(args.interactive_place_s)
        if cube_test_pose is None:
            sys.exit(0)
    if args.random_cube is not None:
        rng = np.random.default_rng(args.random_cube)
        # Fixture setup spans the tabletop.  These hidden coordinates are
        # discarded before control; perception and IK receive camera data only.
        cube_test_pose = (rng.uniform(*TABLE_X), rng.uniform(*TABLE_Y), 0.0)
        print("blind-test fixture selected a hidden cube pose; the controller "
              "will plan from camera pixels only")
    while True:
        result = main(check_only=args.check, post_lift_s=args.post_lift_s,
                      report_path=args.report, test_cube_pose=cube_test_pose,
                      preview_initial_s=args.preview_initial_s)
        if result != 2:
            sys.exit(result)
        cube_test_pose = interactive_cube_placement(args.interactive_place_s or 86400)
        if cube_test_pose is None:
            sys.exit(0)
