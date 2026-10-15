# -*- coding: utf-8 -*-
"""Generate simple rigid-object variants of the OpenArm table scene.

The verified towel XML contains a MuJoCo flex body and a towel-specific
keyframe.  DROID is easier to diagnose with a rigid object, so this module
creates small cube, bottle, and ball variants while preserving the calibrated
OpenArm bodies and cameras.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "simulation" / "v1"
SOURCE_SCENE = SCENE_DIR / "folding_scene_V25_4_TOWEL_SUPPORT_04.xml"

OBJECTS: dict[str, dict[str, str]] = {
    "cube": {
        "label": "red cube",
        "body_pos": "0.45 0.0 0.401",
        "inertial": 'mass="0.18" diaginertia="0.00012 0.00012 0.00012"',
        "geom": 'type="box" size="0.0288 0.0288 0.0288" rgba="0.85 0.12 0.08 1"',
    },
    "bottle": {
        "label": "blue bottle",
        "body_pos": "0.45 0.0 0.445",
        "inertial": 'mass="0.12" diaginertia="0.00018 0.00018 0.00006"',
        "geom": 'type="cylinder" size="0.035 0.089" rgba="0.08 0.25 0.85 1"',
    },
    "ball": {
        "label": "yellow ball",
        "body_pos": "0.45 0.0 0.406",
        "inertial": 'mass="0.10" diaginertia="0.000064 0.000064 0.000064"',
        "geom": 'type="sphere" size="0.05" rgba="0.95 0.72 0.05 1"',
    },
}


def _object_body(kind: str) -> str:
    spec = OBJECTS[kind]
    return f'''    <body name="droid_object" pos="{spec["body_pos"]}">
      <freejoint name="droid_object_freejoint" />
      <inertial pos="0 0 0" {spec["inertial"]} />
      <geom name="droid_object_geom" {spec["geom"]} contype="1" conaffinity="1"
        condim="3" friction="1.0 0.02 0.01" solref="0.004 1" solimp="0.95 0.99 0.001" />
    </body>
'''


def build_scene(kind: str, output: Path | None = None) -> Path:
    if kind not in OBJECTS:
        raise ValueError(f"Unsupported DROID object {kind!r}; choose from {sorted(OBJECTS)}")
    if not SOURCE_SCENE.is_file():
        raise FileNotFoundError(f"Verified source scene not found: {SOURCE_SCENE}")

    text = SOURCE_SCENE.read_text(encoding="utf-8")
    text = re.sub(r"\s*<flexcomp\b.*?</flexcomp>", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"\s*<geom name=\"towel_support_v25_4\"[^>]*/>", "", text, count=1)

    key_match = re.search(r'<key name="towel_ready" qpos="([^"]+)"\s*/>', text)
    if key_match is None:
        raise RuntimeError("The source scene has no towel_ready keyframe")
    robot_qpos = key_match.group(1).split()[:18]
    object_qpos = OBJECTS[kind]["body_pos"].split() + ["1", "0", "0", "0"]
    qpos = " ".join(robot_qpos + object_qpos)
    text = text[: key_match.start()] + f'<key name="towel_ready" qpos="{qpos}" />' + text[key_match.end() :]

    marker = "  </worldbody>"
    if marker not in text:
        raise RuntimeError("The source scene has no worldbody closing tag")
    text = text.replace(marker, _object_body(kind) + marker, 1)

    output = output or (SCENE_DIR / f"droid_{kind}_scene.xml")
    output.write_text(text, encoding="utf-8", newline="\n")
    return output


def ensure_scene(kind: str) -> Path:
    output = SCENE_DIR / f"droid_{kind}_scene.xml"
    stale = True
    if output.is_file() and output.stat().st_mtime >= SOURCE_SCENE.stat().st_mtime:
        generated = output.read_text(encoding="utf-8")
        stale = "<flexcomp" in generated or "towel_support_v25_4" in generated
    if not output.is_file() or output.stat().st_mtime < SOURCE_SCENE.stat().st_mtime or stale:
        return build_scene(kind, output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(OBJECTS))
    args = parser.parse_args()
    print(ensure_scene(args.kind))
