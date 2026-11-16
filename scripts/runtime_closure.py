#!/usr/bin/env python3
"""Dependency closures of workspace packages, by dependency-tag class.

The independence gate (scripts/verify_runtime_independence.sh, sim-platform-registry
D10) needs three different closures of a robot runtime package, because the
question it asks each time is different:

- the **execution** closure answers "what is installed when this package runs":
  it follows exec_depend/depend/run_depend/build_export_depend edges transitively
  and is what the fixed allow-list audits;
- the **build** closure adds build_depend/buildtool_depend and populates the
  isolated workspace the gate builds;
- the **test** closure adds test_depend and is only used where package tests run.

catkin_pkg (the same parser colcon uses) normalises the manifest tags into
properties, which is where the tag classes merge:

- ``exec_depends`` absorbs ``<exec_depend>``, ``<depend>`` and format-1
  ``<run_depend>``;
- ``build_export_depends`` additionally holds ``<depend>`` and
  ``<build_export_depend>``;
- ``build_depends`` / ``buildtool_depends`` hold ``<depend>`` plus their own
  tags;
- ``test_depends`` holds only ``<test_depend>`` (a ``<depend>`` is already an
  execution edge, so it needs no extra test edge).

Dependencies that are not workspace packages (ROS release packages like rclpy,
python3-yaml, vendor overlays like aimdk_msgs) are dropped: the gate builds an
isolated workspace out of in-tree sources only, and the allow-list audit is
about in-tree membership.

Usage (stdout, one name per line, sorted, the root included):

    python3 scripts/runtime_closure.py [--kind execution|build|test] <package>...

Dependencies naming no workspace package are expected (system packages) and
dropped; an unknown root is an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from catkin_pkg.package import Package
from catkin_pkg.packages import find_packages

# catkin_pkg property names per closure kind. Each set is cumulative: the
# execution edges are the floor of every wider closure, so "build" adds to
# "execution" rather than redefining it, and "test" adds to "build".
CLOSURE_EDGE_PROPERTIES: dict[str, tuple[str, ...]] = {
    "execution": ("exec_depends", "build_export_depends"),
    "build": ("exec_depends", "build_export_depends", "build_depends", "buildtool_depends"),
    "test": ("exec_depends", "build_export_depends", "build_depends", "buildtool_depends", "test_depends"),
}


def load_workspace_packages(base_path: str | Path) -> dict[str, Package]:
    """Parse every package.xml under ``base_path`` into a name -> Package map.

    This is the same discovery colcon list uses (catkin_pkg.find_packages), so
    the gate's universe and colcon's agree by construction.
    """
    return {pkg.name: pkg for pkg in find_packages(str(base_path)).values()}


def closure_for(root: str, packages: dict[str, Package], kind: str = "execution") -> set[str]:
    """Transitive dependency closure of ``root`` following one kind's edges.

    Only edges naming workspace packages are followed; everything else (ROS
    release packages, pip names, vendor overlays) is dropped because it has no
    in-tree source to build.
    """
    if root not in packages:
        raise KeyError(f"unknown workspace package: {root}")
    properties = CLOSURE_EDGE_PROPERTIES[kind]
    seen: set[str] = set()
    frontier = [root]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        pkg = packages.get(current)
        if pkg is None:
            continue
        for prop in properties:
            for dep in getattr(pkg, prop):
                if dep.name in packages and dep.name not in seen:
                    frontier.append(dep.name)
    return seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kind",
        choices=sorted(CLOSURE_EDGE_PROPERTIES),
        default="execution",
        help="which dependency-tag class to follow (default: execution)",
    )
    parser.add_argument(
        "--base-path",
        default=str(Path(__file__).resolve().parent.parent / "src"),
        help="workspace source root whose package.xml files define the universe",
    )
    parser.add_argument("packages", nargs="+", help="root package name(s)")
    args = parser.parse_args(argv)

    packages = load_workspace_packages(args.base_path)
    merged: set[str] = set()
    try:
        for root in args.packages:
            merged |= closure_for(root, packages, args.kind)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name in sorted(merged):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
