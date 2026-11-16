"""The layout-map CLI (#205 grasp stage D2).

Regenerates a scene's navigation map from its ``layout.yaml``
``static_geometry`` through the same rasteriser the tests pin
(:func:`sim_models.scene_compiler.rasterize_static_geometry`), so committed
maps can never drift from the layout data::

    ros2 run sim_models layout_map <scene> <out_prefix> \
        [--resolution 0.05] [--origin -2.5,-2.5] [--size 100]

The defaults mirror ``robot_navigation/config/maps/sim_map.yaml`` (the
nav_room map); ``nav_grasp_room``'s committed map is generated with them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_models.scene_compiler import get_scene_layout, write_layout_map

#: The map parameters of ``robot_navigation/config/maps/sim_map.yaml`` —
#: the defaults every layout map of the 4 m x 4 m rooms is generated with.
DEFAULT_RESOLUTION = 0.05
DEFAULT_ORIGIN = (-2.5, -2.5)
DEFAULT_SIZE = (100, 100)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="layout_map", description=__doc__.splitlines()[0])
    parser.add_argument("scene", help="the scene name (a directory under sim_models/scenes)")
    parser.add_argument("out_prefix", help="output path prefix: <prefix>.pgm and <prefix>.yaml")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="metres per cell")
    parser.add_argument(
        "--origin",
        type=lambda text: tuple(float(value) for value in text.split(",")),
        default=DEFAULT_ORIGIN,
        help="world x,y of the grid's lower-left corner",
    )
    parser.add_argument(
        "--size",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        default=DEFAULT_SIZE,
        help="grid width,height in cells",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    layout = get_scene_layout(args.scene)
    if not layout:
        print(f"layout_map: no layout.yaml for scene {args.scene!r}")
        return 2
    origin = (float(args.origin[0]), float(args.origin[1]))
    size = (int(args.size[0]), int(args.size[1]))
    pgm, yaml_path = write_layout_map(
        layout, out_prefix=Path(args.out_prefix), resolution=float(args.resolution), origin=origin, size=size
    )
    print(f"layout_map: {args.scene} -> {pgm} + {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
