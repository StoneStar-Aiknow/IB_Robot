"""The core never imports the simulation packages (sim-platform-registry D6).

AST plus string scan of every module under ``robot_runtime/robot_runtime`` and
``robot_runtime/launch``: the runtime contract layer discovers the physics
platforms through the ``ibrobot.sim_platforms`` registry only, so importing
(or launch-referencing) ``robot_runtime_sim`` or ``sim_models`` anywhere in
core is a boundary violation — a hardware-only deployment would pull the
simulator dependencies through the contract package. The registry's
``module:attr`` strings do not count: they are data, resolved lazily by
whichever package installed them.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Modules that must never be imported or launched from the core.
_FORBIDDEN_MODULES = ("robot_runtime_sim", "sim_models")

#: Every ``.py`` under the package module tree and the launch tree.
_SCANNED_FILES = sorted(
    [*(Path(__file__).resolve().parents[1] / "robot_runtime").glob("*.py")]
    + [*(Path(__file__).resolve().parents[1] / "launch").glob("*.launch.py")]
)


def _imported_names(tree: ast.AST) -> set[str]:
    """Top-level module names of every import form, import_module included."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def _share_lookups(tree: ast.AST) -> list[str]:
    """The string literals passed to get_package_share_directory calls."""
    calls: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_package_share_directory"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            calls.append(node.args[0].value)
    return calls


def test_every_core_module_is_scanned():
    assert _SCANNED_FILES, "no core modules found to scan"
    # the moved modules must not appear in the core tree at all
    names = {path.name for path in _SCANNED_FILES}
    assert "gazebo.py" not in names and "mujoco.py" not in names and "platforms.py" not in names


def test_no_core_module_imports_the_simulation_packages():
    for path in _SCANNED_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for module in _FORBIDDEN_MODULES:
            assert module not in _imported_names(tree), f"{path} imports {module}"
            # attribute-style and dotted references inside importlib strings
            assert f"import {module}" not in source, f"{path} references {module}"


def test_no_launch_reference_or_share_lookup_names_the_simulation_packages():
    for path in _SCANNED_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for module in _FORBIDDEN_MODULES:
            assert f'package="{module}"' not in source, f"{path} launches {module}"
            assert module not in _share_lookups(tree), f"{path} resolves the {module} share directory"
