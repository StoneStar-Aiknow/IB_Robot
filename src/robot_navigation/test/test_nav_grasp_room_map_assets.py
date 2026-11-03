"""The shipped nav_grasp_room map pair stays regenerable from the layout.

Review r2 finding 13: sim_models' own tests compare newly written output with
the same rasterizer and verify CLI file creation, but nothing in
robot_navigation referenced the committed ``nav_grasp_room`` pair — a scene
edit could leave the shipped map assets stale while those tests pass. The
launches read the pair from this package's installed ``config/maps``, so both
the committed (source) pair and the installed one are gated here,
byte-for-byte, against a regeneration from the layout with the map CLI's own
defaults. robot_navigation already test-depends on sim_models (the rasterizer
and CLI live there); sim_models never depends on robot_navigation.
"""

from __future__ import annotations

from pathlib import Path

from sim_models.layout_map import DEFAULT_ORIGIN, DEFAULT_RESOLUTION, DEFAULT_SIZE
from sim_models.scene_compiler import get_scene_layout, write_layout_map

SCENE = "nav_grasp_room"


def _regenerated_pair(tmp_path: Path) -> tuple[bytes, str]:
    """The map pair regenerated from the layout with the CLI's defaults."""
    pgm_path, yaml_path = write_layout_map(
        get_scene_layout(SCENE),
        out_prefix=tmp_path / SCENE,
        resolution=DEFAULT_RESOLUTION,
        origin=DEFAULT_ORIGIN,
        size=DEFAULT_SIZE,
    )
    return pgm_path.read_bytes(), yaml_path.read_text(encoding="ascii")


def test_the_committed_map_pair_matches_the_layout(tmp_path: Path) -> None:
    committed = Path(__file__).resolve().parents[1] / "config" / "maps"
    pgm, yaml_text = _regenerated_pair(tmp_path)
    assert (committed / f"{SCENE}.pgm").read_bytes() == pgm, (
        f"{committed}/{SCENE}.pgm is stale: regenerate it with "
        f"`ros2 run sim_models layout_map {SCENE} <out_prefix>` from the scene's layout.yaml"
    )
    assert (committed / f"{SCENE}.yaml").read_text(encoding="ascii") == yaml_text, (
        f"{committed}/{SCENE}.yaml is stale: regenerate it with "
        f"`ros2 run sim_models layout_map {SCENE} <out_prefix>` from the scene's layout.yaml"
    )


def test_the_installed_map_pair_matches_the_layout(tmp_path: Path) -> None:
    """The pair the launches actually load (the installed share dir) cannot
    go stale while the source pair is fresh — a source edit without a
    rebuild would diverge the two and this gate fails instead."""
    from ament_index_python.packages import get_package_share_directory

    installed = Path(get_package_share_directory("robot_navigation")) / "config" / "maps"
    pgm, yaml_text = _regenerated_pair(tmp_path)
    assert (installed / f"{SCENE}.pgm").read_bytes() == pgm, (
        f"{installed}/{SCENE}.pgm is stale (the committed pair or the layout changed "
        "without a rebuild): rebuild robot_navigation so the installed map matches"
    )
    assert (installed / f"{SCENE}.yaml").read_text(encoding="ascii") == yaml_text, (
        f"{installed}/{SCENE}.yaml is stale (the committed pair or the layout changed "
        "without a rebuild): rebuild robot_navigation so the installed map matches"
    )
