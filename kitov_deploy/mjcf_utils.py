"""Helpers for loading MJCF assets that need runtime path fixes."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]


G1_SCENE_RELATIVE = Path("Glush_Zoo/g1_description/mjcf/scene_29dof.xml")
G1_ROBOT_RELATIVE = Path("Glush_Zoo/g1_description/mjcf/g1_29dof_rev_1_0.xml")


G1_MESH_FILE_RENAMES = {
    "waist_yaw_link.STL": "waist_yaw_link_rev_1_0.STL",
    "waist_roll_link.STL": "waist_roll_link_rev_1_0.STL",
    "torso_link.STL": "torso_link_rev_1_0.STL",
    "waist_support_link.STL": "torso_link_rev_1_0.STL",
}


def _is_g1_glush_scene(path: Path) -> bool:
    try:
        return path.expanduser().resolve() == (REPO_ROOT / G1_SCENE_RELATIVE).resolve()
    except FileNotFoundError:
        return False


@contextmanager
def prepared_mjcf_path(mjcf_path: str | Path) -> Iterator[Path]:
    """Yield a MuJoCo-loadable XML path without modifying source assets."""

    source_path = Path(mjcf_path).expanduser()
    if not source_path.is_absolute():
        source_path = (REPO_ROOT / source_path).resolve(strict=False)

    if not _is_g1_glush_scene(source_path):
        yield source_path
        return

    robot_xml = (REPO_ROOT / G1_ROBOT_RELATIVE).resolve(strict=False)
    mesh_dir = (robot_xml.parent / "../meshes").resolve(strict=False)
    robot_text = robot_xml.read_text(encoding="utf-8")
    for old, new in G1_MESH_FILE_RENAMES.items():
        robot_text = robot_text.replace(f'file="{old}"', f'file="{new}"')
    robot_text = robot_text.replace('meshdir="../meshes"', f'meshdir="{mesh_dir.as_posix()}"')

    scene_text = source_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="kitov_g1_mjcf_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        patched_robot = temp_dir / "g1_29dof_rev_1_0.xml"
        patched_scene = temp_dir / "scene_29dof.xml"
        patched_robot.write_text(robot_text, encoding="utf-8")
        patched_scene.write_text(scene_text, encoding="utf-8")
        yield patched_scene
