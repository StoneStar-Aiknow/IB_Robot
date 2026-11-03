"""Package-layout regression tests for the manipulation_service split (issue #176).

Guards the module boundaries, the 1000-line budget, single helper
definitions, no cross-module private imports, ROS-free modules importing
without heavy optional dependencies, and the installed console scripts.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "manipulation_service"
SETUP_PY = Path(__file__).resolve().parents[1] / "setup.py"

ROS_FREE_MODULES = [
    "grasp_types",
    "point_clouds",
    "point_cloud_completion",
    "table_plane",
    "candidate_width",
    "params",
    "diagnostics",
    "detection_geometry",
    "debug_viz",
]

FORBIDDEN_AT_IMPORT = ["rclpy", "open3d", "matplotlib", "plotly", "torch", "cv2"]

# Each of these helpers must have exactly one definition in the package
# (with or without a leading underscore), per the spec.
SINGLE_DEFINITION_HELPERS = [
    "sample_execution_table_points",
    "save_ply",
    "make_gripper_mesh_visual",
    "build_gripper_lines",
    "save_gripper_meshes",
    "save_gripper_lines",
    "confidence_color",
    "parse_float_csv",
    "points_from_pixels",
]


def _package_modules():
    return sorted(PACKAGE_DIR.glob("*.py"))


def test_every_module_is_within_the_line_budget():
    for module in _package_modules():
        line_count = len(module.read_text(encoding="utf-8").splitlines())
        assert line_count <= 1000, f"{module.name} is {line_count} lines (budget 1000)"


def test_node_module_has_no_viz_definitions_or_imports():
    tree = ast.parse((PACKAGE_DIR / "grasp_planner_node.py").read_text(encoding="utf-8"))
    defined = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    viz_names = {
        "save_ply",
        "_save_ply",
        "make_gripper_mesh_visual",
        "_make_gripper_mesh_visual",
        "build_gripper_lines",
        "_build_gripper_lines",
        "save_gripper_meshes",
        "_save_gripper_meshes",
        "save_gripper_lines",
        "_save_gripper_lines",
        "render_debug_previews",
        "_render_debug_previews",
        "save_labeled_preview",
        "_save_labeled_preview",
        "save_interactive_preview",
        "_save_interactive_preview",
        "write_full_debug_artifacts",
        "_write_full_debug_artifacts",
    }
    assert not (defined & viz_names), f"grasp_planner_node defines viz helpers: {sorted(defined & viz_names)}"

    source = (PACKAGE_DIR / "grasp_planner_node.py").read_text(encoding="utf-8")
    for module in ("open3d", "matplotlib", "plotly"):
        assert f"import {module}" not in source, f"grasp_planner_node imports {module}"


def test_debug_script_lives_inside_the_package():
    assert not (Path(__file__).resolve().parents[1] / "debug_graspgen.py").exists()
    assert (PACKAGE_DIR / "debug_graspgen.py").is_file()


def test_each_helper_has_a_single_definition():
    definitions = {}
    for module in _package_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                name = node.name.lstrip("_")
                definitions.setdefault(name, []).append(module.name)
    for helper in SINGLE_DEFINITION_HELPERS:
        modules = definitions.get(helper, [])
        assert len(modules) == 1, f"{helper} defined in {modules}; expected exactly one module"


def _local_module_aliases(tree):
    """Map local names to the package modules they refer to.

    Follows Python's actual binding rules:

    - ``import manipulation_service.params as p`` binds ``p``;
      without an alias only ``manipulation_service`` is bound.
    - ``from manipulation_service import params`` / ``params as p`` binds the
      submodule ``manipulation_service.params``.
    - ``from . import params as p`` and ``from .params import x`` (level-1
      forms) resolve against the package.
    """
    aliases = {}

    def add(local, module_name):
        aliases[local] = module_name

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "manipulation_service" or alias.name.startswith("manipulation_service."):
                    # `import a.b.c` binds only `a` unless aliased.
                    add(alias.asname or alias.name.split(".")[0], alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            is_package_source = (node.module or "").startswith("manipulation_service") or node.level > 0
            if not is_package_source:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                # Dunder imports (__doc__, __all__, ...) are public protocol.
                if alias.name.startswith("_") and not alias.name.startswith("__"):
                    raise AssertionError(f"private import {alias.name} from {node.module}")
                bound = alias.asname or alias.name
                if (node.level > 0 and not node.module) or (node.module or "") == "manipulation_service":
                    # from . import params / from manipulation_service import
                    # params — binds the submodule itself.
                    add(bound, f"manipulation_service.{alias.name}")
    return aliases


def _package_submodule_names():
    return {module.stem for module in _package_modules() if module.name != "__init__"}


def _private_module_attribute_accesses(tree):
    """Yield chains for private attributes reached on package modules.

    Only accesses whose whole chain stays on the imported module object are
    module-level private access (e.g. ``params._secret``,
    ``manipulation_service.params._secret``). A public segment in the chain
    (``tp.TablePlane._private``) is an attribute of a class or instance the
    module exposes — private to that object, not to the module — and is out
    of scope, matching the spec's "attribute access on classes or instances
    is out of scope".
    """
    aliases = _local_module_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        # Collect the attribute chain from the outermost private attribute
        # down to its root name.
        segments = [node.attr]
        root = node.value
        while isinstance(root, ast.Attribute):
            segments.append(root.attr)
            root = root.value
        if not isinstance(root, ast.Name) or root.id not in aliases:
            continue
        module_name = aliases[root.id]
        if module_name != "manipulation_service" and not module_name.startswith("manipulation_service."):
            continue
        # segments is [private_tail, ..., first_attribute] on the bound module
        # `module_name`. A public segment targets an object the module
        # exposes — unless it names a package submodule (e.g. the `params` in
        # `manipulation_service.params._secret` after `import
        # manipulation_service.params`), which keeps the chain on module
        # objects. Everything else public (a class, an instance, ...) puts the
        # private tail out of scope per the spec; only chains that stay on
        # module objects all the way down are module-level private access.
        attribute_segments = list(reversed(segments))
        submodules = _package_submodule_names()

        submodules_set = set(submodules)

        def stays_on_module_objects(segment, *, submodule_names=submodules_set):
            return (segment.startswith("_") and not segment.startswith("__")) or segment in submodule_names

        if not all(stays_on_module_objects(segment) for segment in attribute_segments[:-1]):
            continue
        yield ".".join([root.id] + attribute_segments), node.attr


def _find_private_access(source: str) -> list[str]:
    tree = ast.parse(source)
    return [chain for chain, _attr in _private_module_attribute_accesses(tree)]


def test_no_cross_module_private_imports():
    for module in _package_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        accesses = list(_private_module_attribute_accesses(tree))
        assert not accesses, f"{module.name}: private attribute access on imported modules: {accesses}"


# One case per sample: a failure names exactly the shape that regressed,
# and no sample can mask another.

POSITIVE_SAMPLES = {
    "aliased_module_private": ("import manipulation_service.params as p\nx = p._secret\n", "p._secret"),
    "plain_import_binds_package_root": (
        "import manipulation_service.params\nx = manipulation_service.params._secret\n",
        "manipulation_service.params._secret",
    ),
    "from_package_import_module": (
        "from manipulation_service import params\nz = params._third\n",
        "params._third",
    ),
    "relative_alias": ("from . import table_plane as tp\nw = tp._hidden\n", "tp._hidden"),
    "private_chain_on_module": (
        "from . import table_plane as tp\nv = tp._plane_ref._nested_private\n",
        "tp._plane_ref._nested_private",
    ),
}

NEGATIVE_SAMPLES = {
    "dunder_access": "x = obj.__dict__\n",
    "instance_private": "y = self._state\n",
    "non_package_module": "import os\nx = os._sep_like\n",
    "third_party_module": "import numpy as np\nx = np._private\n",
    "public_member_use": "from manipulation_service.params import safe_name\nx = safe_name\n",
    "class_attribute_private": "from . import table_plane as tp\nv = tp.TablePlane._private\n",
    "dunder_import": "from manipulation_service.params import __doc__\nprint(__doc__)\n",
}


@pytest.mark.parametrize("sample_name", sorted(POSITIVE_SAMPLES))
def test_private_access_detector_flags_sample(sample_name):
    source, expected_chain = POSITIVE_SAMPLES[sample_name]
    flagged = _find_private_access(source)
    assert any(expected_chain in chain for chain in flagged), (sample_name, expected_chain, flagged)


@pytest.mark.parametrize("sample_name", sorted(NEGATIVE_SAMPLES))
def test_private_access_detector_clean_on_sample(sample_name):
    source = NEGATIVE_SAMPLES[sample_name]
    flagged = _find_private_access(source)
    assert flagged == [], (sample_name, flagged)


def _fresh_interpreter_imports(module_names, forbidden):
    code = (
        "import sys\n"
        + "".join(f"import manipulation_service.{name}\n" for name in module_names)
        + f"loaded = {{name for name in {forbidden!r} if name in sys.modules}}\n"
        + "print('LOADED:' + ','.join(sorted(loaded)) if loaded else 'CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PACKAGE_DIR.parents[1]),
    )
    return result


@pytest.mark.parametrize("module_name", ROS_FREE_MODULES)
def test_ros_free_module_imports_without_heavy_dependencies(module_name):
    result = _fresh_interpreter_imports([module_name], FORBIDDEN_AT_IMPORT)
    assert result.returncode == 0, f"import {module_name} failed: {result.stderr}"
    assert "CLEAN" in result.stdout, f"{module_name} loaded forbidden modules: {result.stdout}"


def test_debug_tool_import_is_cheap():
    result = _fresh_interpreter_imports(["debug_graspgen"], ["cv2", "open3d"])
    assert result.returncode == 0, f"import debug_graspgen failed: {result.stderr}"
    assert "CLEAN" in result.stdout, f"debug_graspgen loaded forbidden modules: {result.stdout}"


def test_console_scripts_set_and_targets():
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    console_scripts = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
            for keyword in node.keywords:
                if keyword.arg == "entry_points":
                    entries = keyword.value
                    for index, key in enumerate(entries.keys):
                        is_console_scripts = isinstance(key, ast.Constant) and key.value == "console_scripts"
                        if is_console_scripts:
                            console_scripts = entries.values[index]
    assert console_scripts is not None, "setup.py has no console_scripts"
    script_map = {}
    for item in console_scripts.elts:
        if isinstance(item, ast.Constant):
            name, _, target = item.value.partition("=")
            script_map[name.strip()] = target.strip()
    assert set(script_map) == {"grasp_planner_node", "grasp_verifier_node", "debug_graspgen"}, script_map

    for target in script_map.values():
        module_name, _, attr = target.partition(":")
        module_path = PACKAGE_DIR / f"{module_name.split('.')[-1]}.py"
        assert module_path.is_file(), f"console script module missing: {module_path}"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        top_level = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        assert attr in top_level, f"{target}: {attr} is not a top-level callable in {module_path.name}"
