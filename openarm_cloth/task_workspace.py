"""Physical task table and initial-placement fixture; never a vision input.

The frozen robot MJCF remains unchanged. Edit the table in MjSpec *before*
compiling so collision bounds, inertia constants and render geometry agree.
Object state is used only by the human-placement fixture, not by planning.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "simulation/v1/task_workspace.json"
# Each benchmark subprocess has its own config; never rewrite the live default
# between runs or let parallel tests race to change the current scene.
CONFIG_PATH = Path(os.environ.get("OPENARM_TASK_WORKSPACE_CONFIG", DEFAULT_CONFIG_PATH))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
TABLE_X = tuple(CONFIG["placement_x_m"])
TABLE_Y = tuple(CONFIG["placement_y_m"])
TABLE_TOP = CONFIG["table_center_m"][2] + CONFIG["table_half_size_m"][2]


def load_task_model(scene, target_zone=None):
    """Compile the actual smaller tabletop, with non-colliding inset lines."""
    spec = mujoco.MjSpec.from_file(str(scene))
    table = spec.geom("folding_table")
    original_top = float(table.pos[2] + table.size[2])
    table.pos = CONFIG["table_center_m"]
    table.size = CONFIG["table_half_size_m"]
    x0, x1 = TABLE_X
    y0, y1 = TABLE_Y
    z = TABLE_TOP + 0.0006
    if target_zone is not None:
        spec.worldbody.add_site(
            name="member_b_target_zone", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[*target_zone["center_xy_m"], z],
            size=[target_zone["radius_m"], 0.0005, 0], rgba=[0.1, 0.8, 0.3, 0.65])
    for name, pos, size in (
        ("near", [x0, (y0 + y1) / 2, z], [0.001, (y1 - y0) / 2, 0.0005]),
        ("far", [x1, (y0 + y1) / 2, z], [0.001, (y1 - y0) / 2, 0.0005]),
        ("left", [(x0 + x1) / 2, y1, z], [(x1 - x0) / 2, 0.001, 0.0005]),
        ("right", [(x0 + x1) / 2, y0, z], [(x1 - x0) / 2, 0.001, 0.0005]),
    ):
        spec.worldbody.add_site(name=f"placement_boundary_{name}",
                                type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=pos, size=size, rgba=[0.95, 0.65, 0.12, 1])
    model = spec.compile()
    # Reset the *fixture* above the selected physical tabletop, not inside it.
    # The unchanged 16.2 mm initial drop preserves the existing baseline.
    # These are initial conditions only; camera perception still estimates Z.
    address = int(model.joint("droid_object_freejoint").qposadr[0])
    shift = TABLE_TOP - original_top
    model.qpos0[address + 2] += shift
    model.key_qpos[:, address + 2] += shift
    return model


def constrain_initial_placement(model, data):
    """Clamp mouse/keyboard fixture ONLY, before any camera-guided episode.

    No calls to this function are permitted during grasp execution. The
    inset accommodates the cube's full diagonal, not just its centre point.
    """
    joint = model.joint("droid_object_freejoint")
    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    old_xy = data.qpos[qadr:qadr + 2].copy()
    new_xy = np.clip(old_xy, [TABLE_X[0], TABLE_Y[0]], [TABLE_X[1], TABLE_Y[1]])
    changed = bool(np.any(new_xy != old_xy))
    if changed:
        data.qpos[qadr:qadr + 2] = new_xy
        data.qvel[dadr:dadr + 2] = 0
    # Recover an object dragged below the tabletop during fixture editing.
    if data.qpos[qadr + 2] < TABLE_TOP:
        data.qpos[qadr + 2] = TABLE_TOP + CONFIG["cube_half_size_m"] + 0.001
        data.qvel[dadr:dadr + 6] = 0
        changed = True
    if changed:
        mujoco.mj_forward(model, data)
    return changed


def workspace_description():
    half = CONFIG["table_half_size_m"]
    return (f"Table {half[0] * 200:.1f} x {half[1] * 200:.1f} cm | "
            f"top {TABLE_TOP * 100:.1f} cm\n"
            "Cube centre stays inside the amber outline")
