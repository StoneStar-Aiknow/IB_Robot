"""Self-tests for the independence gate's closure computation (issue #191 D10).

scripts/runtime_closure.py computes the three closures the gate distinguishes.
The bar here is edge-class semantics, on synthetic manifests where every tag is
controlled (scripts/setup/tests/ is the shell-script precedent; this file
follows scripts/test_check_agent_skill.py, the existing python-script pytest
home — loaded by name, run from the repo root with ``pytest scripts``).

The three cases D10 names:
- a direct ``test_depend`` must NOT count (not even into the test closure of a
  narrower question — it counts only into the test closure, never into the
  execution closure the allow-list audits);
- a direct ``exec_depend`` must count;
- a transitive ``exec_depend`` through an intermediate package must count.

Plus: ``<depend>`` counts as execution; ``<build_export_depend>`` counts as
execution (both halves of that tag class, per the design's edge list); a
``build_depend`` does not leak into the execution closure; and the real
workspace's own so101 execution closure excludes the simulation packages —
the property the gate exists to enforce.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("runtime_closure.py")

MANIFEST_TEMPLATE = """<?xml version="1.0"?>
<package format="3">
  <name>{name}</name>
  <version>0.0.0</version>
  <description>closure probe</description>
  <maintainer email="probe@example.com">probe</maintainer>
  <license>MIT</license>
{edges}
</package>
"""

EDGE_TEMPLATE = "  <{tag}>{dep}</{tag}>"


def _load_module():
    spec = importlib.util.spec_from_file_location("runtime_closure", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_package(base: Path, name: str, edges: dict[str, list[str]]):
    lines = [EDGE_TEMPLATE.format(tag=tag, dep=dep) for tag, deps in edges.items() for dep in deps]
    manifest = MANIFEST_TEMPLATE.format(name=name, edges="\n".join(lines))
    package_dir = base / name
    package_dir.mkdir(parents=True)
    (package_dir / "package.xml").write_text(manifest, encoding="utf-8")


@pytest.fixture
def universe(tmp_path):
    """A synthetic workspace with one package per dependency-tag class.

    Layout (every name says which tag class reaches it from ``root``):

        root --exec_depend--> exec_leaf
        root --depend-------> depend_leaf      (exec+build)
        root --build_export--> export_leaf
        root --build_depend-> build_leaf
        root --buildtool----> buildtool_leaf
        root --test_depend--> test_leaf
        exec_leaf --exec_depend--> transitive_leaf
        build_leaf --exec_depend--> hidden_exec_leaf
        test_leaf --exec_depend--> test_only_chain_leaf
    """
    _write_package(
        tmp_path,
        "root",
        {
            "exec_depend": ["exec_leaf"],
            "depend": ["depend_leaf"],
            "build_export_depend": ["export_leaf"],
            "build_depend": ["build_leaf"],
            "buildtool_depend": ["buildtool_leaf"],
            "test_depend": ["test_leaf"],
        },
    )
    _write_package(tmp_path, "exec_leaf", {"exec_depend": ["transitive_leaf"]})
    _write_package(tmp_path, "depend_leaf", {})
    _write_package(tmp_path, "export_leaf", {})
    _write_package(tmp_path, "build_leaf", {"exec_depend": ["hidden_exec_leaf"]})
    _write_package(tmp_path, "buildtool_leaf", {})
    _write_package(tmp_path, "test_leaf", {"exec_depend": ["test_only_chain_leaf"]})
    _write_package(tmp_path, "transitive_leaf", {})
    _write_package(tmp_path, "hidden_exec_leaf", {})
    _write_package(tmp_path, "test_only_chain_leaf", {})
    return tmp_path


def test_direct_test_depend_does_not_count_in_execution_closure(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    closure = module.closure_for("root", packages, "execution")

    assert "test_leaf" not in closure, "a direct test_depend must not pull a package into the execution closure"
    assert "test_only_chain_leaf" not in closure, "nor may anything reachable only through a test_depend edge"


def test_direct_exec_depend_counts(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    closure = module.closure_for("root", packages, "execution")

    assert "exec_leaf" in closure


def test_transitive_exec_depend_through_intermediate_counts(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    closure = module.closure_for("root", packages, "execution")

    assert "transitive_leaf" in closure, (
        "an exec_depend of an exec_depend must reach the execution closure transitively"
    )


def test_depend_and_build_export_depend_are_execution_edges(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    closure = module.closure_for("root", packages, "execution")

    assert "depend_leaf" in closure, "<depend> is an execution edge"
    assert "export_leaf" in closure, "<build_export_depend> is an execution edge"


def test_build_edges_do_not_leak_into_execution_closure(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    closure = module.closure_for("root", packages, "execution")

    assert "build_leaf" not in closure, "build_depend must not widen the execution closure"
    assert "buildtool_leaf" not in closure, "buildtool_depend must not widen the execution closure"
    assert "hidden_exec_leaf" not in closure, "an exec_depend reachable only through a build edge must not count"


def test_build_closure_adds_build_edges(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    execution = module.closure_for("root", packages, "execution")
    build = module.closure_for("root", packages, "build")

    assert build > execution, "the build closure must be strictly wider here"
    assert {"build_leaf", "buildtool_leaf", "hidden_exec_leaf"} <= build
    assert "test_leaf" not in build, "test edges must not widen the build closure"


def test_test_closure_adds_test_edges(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    test = module.closure_for("root", packages, "test")

    assert {"test_leaf", "test_only_chain_leaf"} <= test, "test edges count only into the test closure"


def test_root_is_in_its_own_closure(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    assert "root" in module.closure_for("root", packages, "execution")


def test_unknown_root_is_rejected(universe):
    module = _load_module()
    packages = module.load_workspace_packages(universe)

    with pytest.raises(KeyError, match="unknown workspace package"):
        module.closure_for("no_such_package", packages, "execution")


def test_external_dependencies_are_dropped(universe):
    module = _load_module()
    _write_package(
        universe,
        "with_externals",
        {
            "exec_depend": ["rclpy", "python3-yaml", "aimdk_msgs"],
            "build_depend": ["nlohmann-json-dev"],
        },
    )
    packages = module.load_workspace_packages(universe)

    closure = module.closure_for("with_externals", packages, "test")

    assert closure == {"with_externals"}, "non-workspace dependency names must never appear in a closure"


def test_real_so101_execution_closure_excludes_simulation_packages():
    module = _load_module()
    repo_src = SCRIPT_PATH.parents[1] / "src"
    packages = module.load_workspace_packages(repo_src)

    closure = module.closure_for("so101_robot", packages, "execution")

    assert "sim_models" not in closure, "so101_robot's execution closure must not contain sim_models (#191)"
    assert "robot_runtime_sim" not in closure, (
        "so101_robot's execution closure must not contain robot_runtime_sim (#191)"
    )
    assert {"so101_robot", "robot_runtime", "ibrobot_msgs"} <= closure
