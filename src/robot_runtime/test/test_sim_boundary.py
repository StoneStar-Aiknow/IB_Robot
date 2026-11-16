"""The core never imports the simulation packages (sim-platform-registry D6).

AST plus string scan of every module under the ``robot_runtime`` package tree
(``robot_runtime/robot_runtime`` including its ``testing/`` subpackage) and
the ``robot_runtime/launch`` tree, recursively: the runtime contract layer
discovers the physics platforms through the ``ibrobot.sim_platforms``
registry only, so importing (or launch-referencing) ``robot_runtime_sim`` or
``sim_models`` anywhere in core is a boundary violation — a hardware-only
deployment would pull the simulator dependencies through the contract
package. The registry's ``module:attr`` strings do not count: they are data,
resolved lazily by whichever package installed them.

The import check matches module ROOT names (``robot_runtime_sim.world`` is a
violation just like ``robot_runtime_sim``), reads ``import_module`` calls
through the AST (a dynamic reference is still a reference) and checks
``Node(package=...)`` keyword arguments structurally, so the spacing or
quoting style of the source cannot hide a violation. Small negative samples
prove the scanner itself rejects every forbidden form.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Modules that must never be imported or launched from the core; the match
#: is on the module ROOT name, so submodules (``robot_runtime_sim.world``)
#: are caught too.
_FORBIDDEN_MODULES = ("robot_runtime_sim", "sim_models")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Every ``.py`` under the package module tree (``robot_runtime/`` including
#: ``testing/``) and the launch tree, recursively.
_SCANNED_FILES = sorted(
    {
        *(_PACKAGE_ROOT / "robot_runtime").rglob("*.py"),
        *(_PACKAGE_ROOT / "launch").rglob("*.launch.py"),
    }
)


def _matches_forbidden(module: object) -> bool:
    """Whether a dotted module name is, or lives under, a forbidden root."""
    if not isinstance(module, str):
        return False
    return any(module == root or module.startswith(f"{root}.") for root in _FORBIDDEN_MODULES)


def _imported_names(tree: ast.AST) -> set[str]:
    """Dotted module names of every import form, ``import_module`` included.

    ``import_module`` counts however it was reached — ``importlib.import_module(...)``
    (attribute call) and a bare ``import_module(...)`` (the ``from importlib
    import import_module`` spelling) are both dynamic imports.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        elif (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                or (isinstance(node.func, ast.Name) and node.func.id == "import_module")
            )
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def _node_packages(tree: ast.AST) -> list[str]:
    """The constant ``package`` keyword values of ``Node(...)`` constructions."""
    packages: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Node"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "package" and isinstance(keyword.value, ast.Constant):
                packages.append(keyword.value.value)
    return packages


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
    # the recursion covers the nested trees the boundary must hold for
    scanned = {str(path) for path in _SCANNED_FILES}
    assert any("/testing/" in path for path in scanned), "the testing subpackage must be scanned"
    assert any(path.endswith(".launch.py") for path in scanned), "the launch tree must be scanned"


def test_no_core_module_imports_the_simulation_packages():
    for path in _SCANNED_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for module in _imported_names(tree):
            assert not _matches_forbidden(module), f"{path} imports {module}"
        for module in _FORBIDDEN_MODULES:
            # attribute-style and dotted references inside importlib strings
            assert f"import {module}" not in source, f"{path} references {module}"


def test_no_launch_reference_or_share_lookup_names_the_simulation_packages():
    for path in _SCANNED_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for package in _node_packages(tree):
            assert not _matches_forbidden(package), f"{path} launches {package}"
        for module in _FORBIDDEN_MODULES:
            assert module not in _share_lookups(tree), f"{path} resolves the {module} share directory"


# ---------------------------------------------------------------------------
# Scanner self-checks: negative samples proving each forbidden form is caught
# ---------------------------------------------------------------------------


def _scan_source(source: str) -> tuple[set[str], list[str], list[str]]:
    tree = ast.parse(source, filename="<negative sample>")
    return _imported_names(tree), _node_packages(tree), _share_lookups(tree)


def test_the_scanner_rejects_each_forbidden_import_form():
    for source in (
        "import robot_runtime_sim",
        "import robot_runtime_sim.world as world_hook",
        "from robot_runtime_sim import mujoco",
        "from sim_models.scene_compiler import compose",
        "from importlib import import_module\nimport_module('robot_runtime_sim.world')",
        "importlib.import_module(\n    'sim_models',\n)",
    ):
        imported, _, _ = _scan_source(source)
        assert any(_matches_forbidden(name) for name in imported), source


def test_the_scanner_rejects_forbidden_node_packages_regardless_of_spacing():
    for source in (
        'Node(package="robot_runtime_sim", executable="x")',
        'Node(package = "sim_models",\n    executable="x")',
        "Node(package='robot_runtime_sim', executable='x')",
        'Node(\n    executable="x",\n    package = "robot_runtime_sim.mujoco",\n)',
    ):
        _, packages, _ = _scan_source(source)
        assert any(_matches_forbidden(package) for package in packages), source


def test_the_scanner_rejects_a_forbidden_share_lookup():
    _, _, lookups = _scan_source("get_package_share_directory('robot_runtime_sim')")
    assert any(_matches_forbidden(lookup) for lookup in lookups)


def test_the_scanner_accepts_unrelated_modules_and_packages():
    imported, packages, lookups = _scan_source(
        "import robot_runtime\n"
        "from robot_runtime.sim_platforms import get_sim_platform\n"
        'Node(package="robot_runtime", executable="x")\n'
        "get_package_share_directory('robot_runtime')\n"
    )
    assert not any(_matches_forbidden(name) for name in imported)
    assert not any(_matches_forbidden(package) for package in packages)
    assert not any(_matches_forbidden(lookup) for lookup in lookups)
