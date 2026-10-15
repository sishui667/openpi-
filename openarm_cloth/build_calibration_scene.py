# -*- coding: utf-8 -*-
"""Build a MuJoCo scene containing a fixed checkerboard calibration target.

The scene keeps the calibrated OpenArm bodies and cameras from the verified
V25.4 scene, removes the deformable towel, and adds a static checkerboard on
the tabletop.  The checkerboard is intentionally made from separate black and
white geoms so it is visible without relying on external texture files and can
be detected with standard chessboard corner detectors later.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "simulation" / "v1"
SOURCE_SCENE = SCENE_DIR / "folding_scene_V25_4_TOWEL_SUPPORT_04.xml"
OUTPUT_SCENE = SCENE_DIR / "calibration_board_scene.xml"

BOARD = {
    "name": "calibration_board",
    # Place the board on the left-arm working side so it is already inside
    # the left D405 view at the ready pose and remains reachable for auto-poses.
    "body_pos": [0.53, 0.08, 0.356],
    # An odd number of columns removes the 180-degree checkerboard ambiguity.
    "columns": 9,
    "rows": 6,
    "square_size_m": 0.015,
    "backing_thickness_m": 0.004,
}


def board_metadata() -> dict:
    columns = int(BOARD["columns"])
    rows = int(BOARD["rows"])
    square = float(BOARD["square_size_m"])
    surface_z = float(BOARD["backing_thickness_m"]) + 0.001
    width = columns * square
    height = rows * square
    inner_corners = [
        [(-width / 2.0) + column * square, (-height / 2.0) + row * square, surface_z]
        for row in range(1, rows)
        for column in range(1, columns)
    ]
    return {
        **BOARD,
        "width_m": width,
        "height_m": height,
        "inner_corners_columns": columns - 1,
        "inner_corners_rows": rows - 1,
        "inner_corners_board_xyz_m": inner_corners,
        "pattern": "checkerboard",
        "orientation_marker": "red square at board local lower-left outer cell",
        "coordinate_frame": "board local XY plane, +Z is board normal",
    }


def _materials() -> str:
    return """  <material name="calib_board_backing" rgba="0.12 0.12 0.12 1" />
  <material name="calib_board_black" rgba="0.015 0.015 0.015 1" />
  <material name="calib_board_white" rgba="0.95 0.95 0.95 1" />
  <material name="calib_board_marker" rgba="0.95 0.04 0.04 1" />
"""


def _board_body() -> str:
    columns = int(BOARD["columns"])
    rows = int(BOARD["rows"])
    square = float(BOARD["square_size_m"])
    width = columns * square
    height = rows * square
    thickness = float(BOARD["backing_thickness_m"])
    squares: list[str] = []
    for row in range(rows):
        for column in range(columns):
            if row == 0 and column == 0:
                material = "calib_board_marker"
            else:
                material = "calib_board_white" if (row + column) % 2 == 0 else "calib_board_black"
            x = -width / 2.0 + (column + 0.5) * square
            y = -height / 2.0 + (row + 0.5) * square
            squares.append(
                f'    <geom name="calib_square_{row}_{column}" type="box" '
                f'pos="{x:.6f} {y:.6f} {thickness + 0.0005:.6f}" '
                f'size="{square / 2.0:.6f} {square / 2.0:.6f} 0.0005" '
                f'material="{material}" contype="0" conaffinity="0" />'
            )
    x, y, z = (float(value) for value in BOARD["body_pos"])
    return (
        f'  <body name="{BOARD["name"]}" pos="{x:.6f} {y:.6f} {z:.6f}">\n'
        f'    <geom name="calib_board_backing" type="box" pos="0 0 {thickness / 2.0:.6f}" '
        f'size="{width / 2.0 + 0.008:.6f} {height / 2.0 + 0.008:.6f} {thickness / 2.0:.6f}" '
        'material="calib_board_backing" contype="0" conaffinity="0" />\n'
        + "\n".join(squares)
        + "\n"
        '    <site name="calibration_board_origin" pos="0 0 0" size="0.004" '
        'rgba="1 0 0 1" />\n'
        "  </body>\n"
    )


def build_scene(output: Path | None = None) -> Path:
    if not SOURCE_SCENE.is_file():
        raise FileNotFoundError(f"Verified source scene not found: {SOURCE_SCENE}")

    text = SOURCE_SCENE.read_text(encoding="utf-8")
    # The calibration scene only needs the rigid robot, table, and cameras.
    text = re.sub(r"\s*<flexcomp\b.*?</flexcomp>", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r'\s*<geom name="towel_support_v25_4"[^>]*/>', "", text, count=1)

    key_match = re.search(r'<key name="towel_ready" qpos="([^"]+)"\s*/>', text)
    if key_match is None:
        raise RuntimeError("The source scene has no towel_ready keyframe")
    robot_qpos = key_match.group(1).split()[:18]
    text = (
        text[: key_match.start()]
        + f'<key name="towel_ready" qpos="{" ".join(robot_qpos)}" />'
        + text[key_match.end() :]
    )

    if "  </asset>" not in text:
        raise RuntimeError("The source scene has no asset closing tag")
    text = text.replace("  </asset>", _materials() + "  </asset>", 1)
    if "  </worldbody>" not in text:
        raise RuntimeError("The source scene has no worldbody closing tag")
    text = text.replace("  </worldbody>", _board_body() + "  </worldbody>", 1)

    output = output or OUTPUT_SCENE
    output.write_text(text, encoding="utf-8", newline="\n")
    metadata_path = output.with_suffix(".board.json")
    metadata_path.write_text(
        json.dumps(board_metadata(), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    return output


def ensure_scene() -> Path:
    metadata_path = OUTPUT_SCENE.with_suffix(".board.json")
    generated = OUTPUT_SCENE.read_text(encoding="utf-8") if OUTPUT_SCENE.is_file() else ""
    board_pos_text = " ".join(f"{float(value):.6f}" for value in BOARD["body_pos"])
    stale = (
        'site name="calibration_board_origin"' not in generated
        or ('site name="calibration_board_origin"' in generated and 'contype="0"' in generated)
        or f'<body name="{BOARD["name"]}" pos="{board_pos_text}">' not in generated
        or f'size="{float(BOARD["square_size_m"]) / 2.0:.6f} {float(BOARD["square_size_m"]) / 2.0:.6f} 0.0005"' not in generated
        or f'calib_square_0_{int(BOARD["columns"]) - 1}' not in generated
        or 'material="calib_board_marker"' not in generated
    )
    if (
        not OUTPUT_SCENE.is_file()
        or OUTPUT_SCENE.stat().st_mtime < SOURCE_SCENE.stat().st_mtime
        or not metadata_path.is_file()
        or stale
    ):
        return build_scene()
    return OUTPUT_SCENE


if __name__ == "__main__":
    print(ensure_scene())
