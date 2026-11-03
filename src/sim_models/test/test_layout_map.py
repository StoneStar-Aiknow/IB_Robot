"""The layout-map rasteriser (#205 grasp stage D2).

Independent geometry fixtures pin :func:`rasterize_static_geometry`'s cell
rules — an asymmetric yawed box, a cylinder, an element crossing the map
border, and the row orientation (a single occupied cell at a known world
(x, y)) — plus the trinary PGM/YAML writer and the ``layout_map`` CLI.
The nav_room fidelity against the shipped ``sim_map.pgm`` lives in
``robot_navigation`` (the map is that package's artifact; a sim_models
test dependency would close a cycle).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_models.layout_map import main as layout_map_main
from sim_models.scene_compiler import get_scene_layout, rasterize_static_geometry, write_layout_map

RES = 0.05
ORIGIN = (-2.5, -2.5)
SIZE = (100, 100)


def _grid(static_geometry: list[dict]) -> np.ndarray:
    return rasterize_static_geometry({"static_geometry": static_geometry}, resolution=RES, origin=ORIGIN, size=SIZE)


def _cell(grid: np.ndarray, x: float, y: float) -> bool:
    """The grid value at a world point (row 0 = max y, the PGM order)."""
    height = grid.shape[0]
    column = int((x - ORIGIN[0]) / RES)
    row = height - 1 - int((y - ORIGIN[1]) / RES)
    return bool(grid[row, column])


def test_an_axis_aligned_box_occupies_its_exact_cell_intervals():
    """A box from x = 0.0 to 0.3 (yaw 0) occupies exactly the cells
    [0.0, 0.35) on x — the interval-overlap rule, no half-cell rounding."""
    grid = _grid([{"name": "b", "primitive": "box", "size": [0.3, 0.1, 0.1], "pose": [0.15, 0.0, 0.05, 0, 0, 0]}])
    assert _cell(grid, 0.01, 0.0) and _cell(grid, 0.29, 0.0), "inside the interval span"
    assert not _cell(grid, -0.01, 0.0), "the cell left of x = 0.0 stays free"
    assert not _cell(grid, 0.31, 0.0), "the cell that only touches x = 0.30 at a point stays free"


def test_an_asymmetric_yawed_box_occupies_its_bounding_box():
    """A yawed box rasterises as its axis-aligned bounding box: a 0.2 x 0.1
    box yawed 45 deg at (0.5, 0.5) spans |x|,|y| <= (0.1 + 0.05) * cos(45)
    per axis — the conservative envelope of the yawed footprint."""
    grid = _grid(
        [{"name": "y", "primitive": "box", "size": [0.2, 0.1, 0.1], "pose": [0.5, 0.5, 0.05, 0.0, 0.0, np.pi / 4]}]
    )
    extent = (0.1 + 0.05) * np.cos(np.pi / 4)
    assert _cell(grid, 0.5 + extent - 0.01, 0.5 + extent - 0.01), "the bounding-box corner cell is occupied"
    outside = extent + RES
    assert not _cell(grid, 0.5 + outside, 0.5), "beyond the bounding box stays free"
    assert not _cell(grid, 0.5, 0.5 + outside), "beyond the bounding box stays free (y)"


def test_a_cylinder_occupies_the_cells_its_disc_touches():
    """A radius-0.1 cylinder at (0, 0): the cell whose corner is 0.09 m
    away is occupied (the closest point of the cell rectangle is inside the
    disc); the cell whose closest corner is 0.11 m away is free."""
    grid = _grid([{"name": "c", "primitive": "cylinder", "size": [0.1, 0.6], "pose": [0.0, 0.0, 0.3, 0, 0, 0]}])
    assert _cell(grid, 0.09, 0.09), "the cell rectangle's closest point is within the radius"
    assert not _cell(grid, 0.11, 0.11), "the cell rectangle's closest point is beyond the radius"
    assert _cell(grid, 0.0, 0.0), "the centre cell is occupied"


def test_an_element_crossing_the_map_border_clips_to_the_grid():
    """A wall straddling the map border occupies only the in-grid part —
    no wrap-around, no shape error."""
    grid = _grid([{"name": "w", "primitive": "box", "size": [0.4, 4.0, 1.0], "pose": [1.95, 0.0, 0.5, 0, 0, 0]}])
    # x in [1.75, 2.15]; the grid ends at x = 2.5, so the whole wall is
    # inside — instead push it past the east border
    grid = _grid([{"name": "w", "primitive": "box", "size": [0.4, 4.0, 1.0], "pose": [2.45, 0.0, 0.5, 0, 0, 0]}])
    # x in [2.25, 2.65]: cells up to the border are occupied, none beyond;
    # the wall spans y in [-2, 2] -> the world rows 10..89 (the PGM rows
    # 10..89 too — the flip of [10, 90) in a 100-row grid is [10, 90))
    assert _cell(grid, 2.26, 0.0), "the in-grid part of the wall is occupied"
    assert grid[10:90, -1].all(), "the last column is occupied across the wall's rows"
    assert not grid[:10, -1].any() and not grid[90:, -1].any(), "outside the wall's y span stays free"
    assert grid.shape == (100, 100), "the grid keeps its declared size"


def test_the_row_orientation_is_pgm_order():
    """A single-cell box at a known world (x, y) lands on the PGM row for
    that y — row 0 = max y — so the array serialises directly into the map
    file ``map_server`` reads."""
    # the world cell x in [0.30, 0.35), y in [0.10, 0.15)
    grid = _grid([{"name": "s", "primitive": "box", "size": [0.04, 0.04, 0.04], "pose": [0.32, 0.12, 0.02, 0, 0, 0]}])
    assert grid.sum() == 1, "the fixture occupies exactly one cell"
    row, column = np.argwhere(grid)[0]
    # the KNOWN cell: the world cell x in [0.30, 0.35), y in [0.10, 0.15)
    # maps to column 56 and — row 0 = max y — to row 99 - 52 = 47
    assert (row, column) == (47, 56), f"expected the known cell (47, 56), got {(row, column)}"
    assert _cell(grid, 0.32, 0.12)


def test_invalid_raster_arguments_are_rejected():
    geometry = [{"name": "b", "primitive": "box", "size": [0.1, 0.1, 0.1], "pose": [0, 0, 0, 0, 0, 0]}]
    with pytest.raises(ValueError, match="resolution must be positive"):
        rasterize_static_geometry({"static_geometry": geometry}, resolution=0.0, origin=ORIGIN, size=SIZE)
    with pytest.raises(ValueError, match="size"):
        rasterize_static_geometry({"static_geometry": geometry}, resolution=RES, origin=ORIGIN, size=(0, 10))
    with pytest.raises(ValueError, match="unknown primitive"):
        _grid([{"name": "x", "primitive": "sphere", "size": [0.1], "pose": [0, 0, 0, 0, 0, 0]}])
    with pytest.raises(ValueError, match="pose needs 6 values"):
        _grid([{"name": "x", "primitive": "box", "size": [0.1, 0.1, 0.1], "pose": [0, 0, 0]}])


def test_write_layout_map_produces_the_trinary_pair(tmp_path):
    layout = get_scene_layout("nav_grasp_room")
    pgm, yaml_path = write_layout_map(layout, out_prefix=tmp_path / "m", resolution=RES, origin=ORIGIN, size=SIZE)
    data = pgm.read_bytes()
    assert data.startswith(b"P5\n100 100\n255\n"), "the PGM header is binary P5 with the declared size"
    raster = np.frombuffer(data[15:], dtype=np.uint8).reshape(100, 100)
    assert set(np.unique(raster)) <= {0, 254}, "trinary: 0 occupied, 254 free"
    expected = rasterize_static_geometry(layout, resolution=RES, origin=ORIGIN, size=SIZE)
    assert np.array_equal(raster == 0, expected), "the PGM raster equals the occupancy grid"
    text = yaml_path.read_text()
    assert "image: m.pgm" in text and "mode: trinary" in text
    assert "resolution: 0.05" in text and "origin: [-2.5, -2.5, 0]" in text
    assert "negate: 0" in text and "occupied_thresh: 0.65" in text and "free_thresh: 0.25" in text


def test_the_cli_writes_the_scene_map(tmp_path, capsys):
    out = tmp_path / "nav_grasp_room"
    assert layout_map_main(["nav_grasp_room", str(out)]) == 0
    assert (tmp_path / "nav_grasp_room.pgm").is_file()
    assert (tmp_path / "nav_grasp_room.yaml").is_file()
    assert "nav_grasp_room" in capsys.readouterr().out


def test_the_cli_rejects_unknown_scenes(tmp_path):
    assert layout_map_main(["no_such_scene", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x.pgm").exists()
