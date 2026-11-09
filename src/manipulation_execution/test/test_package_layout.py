"""Package-layout guard (design D6): subsystem boundaries and dependency direction.

Everything here is AST-only: no test imports a package module, so the guard
works without a ROS graph. The rule checkers are functions over source text
(plus the module context needed to resolve relative imports); each is
unit-tested on synthetic negatives so the guard is proven to fail, not only
to pass.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

PACKAGE = "manipulation_execution"
SETUP_DIR = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SETUP_DIR / PACKAGE
SETUP_PY = SETUP_DIR / "setup.py"

ROOT_ALLOWED_ENTRIES = {
    "__init__.py",
    "geometry.py",
    "launch_builders",
    "shared",
    "pick",
    "place",
    "hri",
}

SUBSYSTEMS = frozenset({"pick", "place", "hri"})

# The frozen set of console-script names on master (spec: external entry points are stable).
FROZEN_CONSOLE_SCRIPTS = frozenset(
    {
        "pick_action_client",
        "pick_executor_node",
        "grasp_pipeline_node",
        "place_executor_node",
        "imitate_human_motion_executor_node",
        "motion_imitation_stream_node",
        "motion_imitation_mock_node",
        "mirror_session_node",
        "mirror_session_release",
        "placement_replay",
    }
)

# The frozen launch-builder registrations: the robot_config launch registry resolves this
# group, so a renamed entry-point name or a moved target breaks the registry (D6 rule 4).
FROZEN_LAUNCH_BUILDERS = {
    "mirror_session": "manipulation_execution.launch_builders.mirror_session:MIRROR_SESSION_BUILDER",
    "motion_imitation_stream": (
        "manipulation_execution.launch_builders.motion_imitation_stream:MOTION_IMITATION_STREAM_BUILDER"
    ),
}


def _is_private(name: str) -> bool:
    """A leading-underscore name that is not a dunder."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_package_module(name: str) -> bool:
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def _resolve_relative(current_module: str, level: int, module: str | None, *, is_package: bool) -> str:
    """The absolute module a relative import names, given where it appears."""
    base = current_module if is_package else current_module.rsplit(".", 1)[0]
    for _ in range(level - 1):
        base = base.rsplit(".", 1)[0]
    return base + ("." + module if module else "")


def imported_package_modules(
    source: str,
    current_module: str,
    known_package_modules: frozenset[str] = frozenset(),
    *,
    is_package: bool = False,
) -> frozenset[str]:
    """Rule 3 base checker: every package module ``source`` imports, at any scope.

    A ``from <pkg module> import name`` counts as importing the named module
    whenever ``<pkg module>.name`` is a real module, and every ``from . import
    x`` member is conservatively recorded whether or not it is a module (the
    subsystem prefix is the same either way).
    """
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_package_module(alias.name):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                absolute = _resolve_relative(current_module, node.level, node.module, is_package=is_package)
            elif node.module and _is_package_module(node.module):
                absolute = node.module
            else:
                continue
            found.add(absolute)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if node.module is not None:
                    member = absolute + "." + alias.name
                    if member in known_package_modules:
                        found.add(member)
                else:  # ``from . import x``: the names may be modules
                    found.add(_resolve_relative(current_module, node.level, alias.name, is_package=is_package))
    return frozenset(found)


def _dotted_chain(expr: ast.AST) -> list[str] | None:
    """``a.b.c`` -> ``["a", "b", "c"]`` for pure name/attribute chains, else None."""
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        parts.reverse()
        return parts
    return None


def _bound_names(targets) -> set[str]:
    """Every name an assignment, loop, ``with`` or comprehension target binds."""
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Starred):
            visit(node.value)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    for target in targets:
        visit(target)
    return names


def private_import_violations(
    source: str,
    current_module: str,
    known_package_modules: frozenset[str],
    *,
    is_package: bool = False,
) -> list[str]:
    """Rule 2 checker: `_name` imported from a package module (absolute or
    relative, at any scope including inside functions), private modules, and
    private attribute reads through an imported package module.

    Names are resolved by walking scopes in statement order: an import binds a
    module alias (``import a.b.c`` binds the root ``a``; the attribute chain
    resolves the rest), while any later assignment, non-package import or
    re-import of a symbol replaces the binding, and function/lambda parameters
    shadow for their own scope only — a nested function's bindings never leak
    out. Class and instance attribute access (a name bound by
    ``from pkg import symbol``) and dunders are unflagged (design D6)."""
    tree = ast.parse(source)
    violations: list[str] = []

    def note(node: ast.AST, what: str) -> None:
        violations.append(f"line {getattr(node, 'lineno', '?')}: {what}")

    shadowed = None  # the binding of a name that is not a package-module alias

    def import_statement(node: ast.Import, bindings: dict[str, str | None]) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            if not _is_package_module(alias.name):
                bindings[local] = shadowed
                continue
            if _is_private(alias.name.rsplit(".", 1)[-1]):
                note(node, f"imports private module {alias.name!r}")
            bindings[local] = alias.name if alias.asname else alias.name.split(".", 1)[0]

    def import_from_statement(node: ast.ImportFrom, bindings: dict[str, str | None]) -> None:
        if node.level:
            absolute = _resolve_relative(current_module, node.level, node.module, is_package=is_package)
            in_package = True
        elif node.module and _is_package_module(node.module):
            absolute = node.module
            in_package = True
        else:
            in_package = False
        if in_package and node.module and _is_private(node.module.rsplit(".", 1)[-1]):
            note(node, f"imports private module {node.module!r}")
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            if in_package and _is_private(alias.name):
                note(node, f"imports private name {alias.name!r}")
            if not in_package:
                bindings[local] = shadowed
                continue
            member = (
                absolute + "." + alias.name
                if node.module is not None
                else _resolve_relative(current_module, node.level, alias.name, is_package=is_package)
            )
            bindings[local] = member if member in known_package_modules else shadowed

    def check_attribute(node: ast.Attribute, bindings: dict[str, str | None]) -> None:
        chain = _dotted_chain(node.value)
        if not chain:
            return
        root = chain[0]
        if root not in bindings or bindings[root] is shadowed:
            return
        extension = ".".join(chain[1:])
        path = bindings[root] + ("." + extension if extension else "")
        # The binding itself is an import-confirmed module; only an extended
        # chain (``alias.sub._x``) has to resolve to a real module.
        if not extension or path in known_package_modules:
            note(node, f"reads {'.'.join(chain)}.{node.attr} through an imported package module")

    def argument_names(arguments: ast.arguments) -> list[ast.arg]:
        args = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        if arguments.vararg:
            args.append(arguments.vararg)
        if arguments.kwarg:
            args.append(arguments.kwarg)
        return args

    def walk_expression(node: ast.AST | None, bindings: dict[str, str | None]) -> None:
        """Visit an expression tree; lambdas and comprehensions open sub-scopes."""
        if node is None:
            return
        if isinstance(node, ast.Attribute):
            if _is_private(node.attr):
                check_attribute(node, bindings)
            walk_expression(node.value, bindings)
            return
        if isinstance(node, ast.Lambda):
            inner = dict(bindings)
            for argument in argument_names(node.args):
                inner[argument.arg] = shadowed
            walk_expression(node.body, inner)
            return
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            # Conservative: the whole comprehension runs with its targets
            # shadowed, so an access in the first iterable is under-reported.
            inner = dict(bindings)
            for name in _bound_names([generator.target for generator in node.generators]):
                inner[name] = shadowed
            for child in ast.iter_child_nodes(node):
                walk_expression(child, inner)
            return
        if isinstance(node, ast.NamedExpr):
            walk_expression(node.value, bindings)
            bindings[node.target.id] = shadowed
            return
        for child in ast.iter_child_nodes(node):
            walk_expression(child, bindings)

    def function_scope(node: ast.FunctionDef | ast.AsyncFunctionDef, bindings: dict[str, str | None]) -> None:
        # Decorators, defaults and annotations evaluate in the enclosing scope.
        for decorator in node.decorator_list:
            walk_expression(decorator, bindings)
        for default in node.args.defaults:
            walk_expression(default, bindings)
        for default in node.args.kw_defaults:
            walk_expression(default, bindings)
        walk_expression(node.returns, bindings)
        for argument in argument_names(node.args):
            walk_expression(argument.annotation, bindings)
        inner = dict(bindings)
        for argument in argument_names(node.args):
            inner[argument.arg] = shadowed
        scan_statements(node.body, inner)

    def scan_statement(node: ast.AST, bindings: dict[str, str | None]) -> None:
        if isinstance(node, ast.Import):
            import_statement(node, bindings)
        elif isinstance(node, ast.ImportFrom):
            import_from_statement(node, bindings)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            bindings[node.name] = shadowed
            function_scope(node, bindings)
        elif isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                walk_expression(decorator, bindings)
            for base in node.bases:
                walk_expression(base, bindings)
            for keyword in node.keywords:
                walk_expression(keyword.value, bindings)
            bindings[node.name] = shadowed
            scan_statements(node.body, dict(bindings))
        elif isinstance(node, ast.Assign):
            walk_expression(node.value, bindings)
            for name in _bound_names(node.targets):
                bindings[name] = shadowed
        elif isinstance(node, ast.AnnAssign):
            walk_expression(node.value, bindings)
            walk_expression(node.annotation, bindings)
            if isinstance(node.target, ast.Name):
                bindings[node.target.id] = shadowed
        elif isinstance(node, ast.AugAssign):
            walk_expression(node.value, bindings)
            if isinstance(node.target, ast.Name):
                bindings[node.target.id] = shadowed
        elif isinstance(node, ast.For | ast.AsyncFor):
            walk_expression(node.iter, bindings)
            for name in _bound_names([node.target]):
                bindings[name] = shadowed
            scan_statements(node.body, bindings)
            scan_statements(node.orelse, bindings)
        elif isinstance(node, ast.While | ast.If):
            walk_expression(node.test, bindings)
            scan_statements(node.body, bindings)
            scan_statements(node.orelse, bindings)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                walk_expression(item.context_expr, bindings)
                if item.optional_vars is not None:
                    for name in _bound_names([item.optional_vars]):
                        bindings[name] = shadowed
            scan_statements(node.body, bindings)
        elif isinstance(node, ast.Try):
            scan_statements(node.body, bindings)
            for handler in node.handlers:
                if handler.name:
                    bindings[handler.name] = shadowed
                walk_expression(handler.type, bindings)
                scan_statements(handler.body, bindings)
            scan_statements(node.orelse, bindings)
            scan_statements(node.finalbody, bindings)
        elif isinstance(node, ast.Expr | ast.Return | ast.Assert | ast.Raise | ast.Delete):
            for child in ast.iter_child_nodes(node):
                walk_expression(child, bindings)
        else:
            # Conservative fallback (match and rarer statements): check every
            # expression with the current bindings and shadow every stored name.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr | ast.ExceptHandler):
                    walk_expression(child, bindings)
                elif isinstance(child, ast.stmt):
                    scan_statement(child, bindings)
            for name in _bound_names([node]):
                bindings[name] = shadowed

    def scan_statements(statements: list[ast.stmt], bindings: dict[str, str | None]) -> None:
        for statement in statements:
            scan_statement(statement, bindings)

    scan_statements(tree.body, {})
    violations.sort(key=lambda item: int(item.split(":", 1)[0].split()[-1]))
    return violations


def subsystem_of(module: str) -> str:
    """Which part of the layout a package module belongs to ("root" covers
    ``__init__``; ``geometry`` is a root module, not a subpackage)."""
    parts = module.split(".")
    if parts[0] != PACKAGE:
        return "external"
    if len(parts) == 1:
        return "root"
    second = parts[1]
    if second in SUBSYSTEMS | {"launch_builders", "shared"}:
        return second
    if len(parts) == 2 and second == "geometry":
        return "geometry"
    return "root"


def dependency_violations(current_module: str, imported: frozenset[str]) -> list[str]:
    """Rule 3 checker: one-way dependencies between the subsystems."""
    home = subsystem_of(current_module)
    problems = []
    for target in sorted(imported):
        away = subsystem_of(target)
        if home in SUBSYSTEMS and away in SUBSYSTEMS and away != home:
            problems.append(f"{home} imports {away} ({target})")
        elif home in {"geometry", "shared"} and away != "external":
            problems.append(f"{home} imports the package ({target})")
        elif home == "launch_builders" and away in SUBSYSTEMS | {"shared"}:
            problems.append(f"launch_builders imports {away} ({target})")
    return problems


def entry_points_from_setup_source(source: str) -> dict[str, dict[str, str]]:
    """Rule 4 base checker: the ``entry_points`` dict of ``setup()`` as
    ``group -> {name: "module:attr"}``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != "entry_points" or not isinstance(keyword.value, ast.Dict):
                continue
            groups: dict[str, dict[str, str]] = {}
            for group_key, group_value in zip(keyword.value.keys, keyword.value.values, strict=True):
                entries = {}
                for element in group_value.elts:
                    if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                        continue
                    name, _, target = element.value.partition("=")
                    entries[name.strip()] = target.strip()
                groups[group_key.value] = entries
            return groups
    return {}


def top_level_attributes(source: str) -> set[str]:
    """The names a module defines at its top level (functions, classes, assigns)."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def iter_package_sources():
    """(module path, is_package, source) for every ``.py`` under the package."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        parts = ("manipulation_execution", *relative.parts)
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1]) if is_package else ".".join(parts)
        yield module, is_package, path.read_text(encoding="utf-8"), str(path.relative_to(PACKAGE_ROOT))


def known_package_modules() -> frozenset[str]:
    modules = {module for module, _, _, _ in iter_package_sources()}
    return frozenset(modules)


_SOURCES = list(iter_package_sources())
_KNOWN_MODULES = known_package_modules()


# ---- rule 1: the root holds only the subpackages and geometry ----


def test_root_entries_are_exactly_the_subpackages_and_geometry():
    entries = {entry for entry in os.listdir(PACKAGE_ROOT) if entry != "__pycache__"}
    assert entries == ROOT_ALLOWED_ENTRIES


# ---- rule 5: every directory with .py files is a package ----


def test_every_directory_with_python_files_has_an_init():
    directories = {path.parent for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts}
    assert directories, "the package disappeared"
    missing = sorted(
        str(directory.relative_to(PACKAGE_ROOT))
        for directory in directories
        if not (directory / "__init__.py").is_file()
    )
    assert missing == []


# ---- rules 2 and 3 over the real tree ----


@pytest.mark.parametrize(
    ("module", "is_package", "source", "relative"),
    _SOURCES,
    ids=[relative for _, _, _, relative in _SOURCES],
)
def test_no_private_imports_or_cross_subsystem_dependencies(module, is_package, source, relative):
    problems = private_import_violations(source, module, _KNOWN_MODULES, is_package=is_package)
    problems += dependency_violations(
        module, imported_package_modules(source, module, _KNOWN_MODULES, is_package=is_package)
    )
    assert problems == [], f"{relative}: {problems}"


def test_retargeting_never_imports_the_imitation_executor():
    for module, is_package, source, relative in iter_package_sources():
        if not module.startswith(PACKAGE + ".hri.retargeting"):
            continue
        imported = imported_package_modules(source, module, _KNOWN_MODULES, is_package=is_package)
        assert not [target for target in imported if target.startswith(PACKAGE + ".hri.imitate")], relative


# ---- rule 4: entry points resolve and the console-script names are frozen ----


def test_entry_points_target_existing_modules_and_top_level_attributes():
    groups = entry_points_from_setup_source(SETUP_PY.read_text(encoding="utf-8"))
    assert set(groups) == {"console_scripts", "ibrobot.launch_builders"}

    assert set(groups["console_scripts"]) == FROZEN_CONSOLE_SCRIPTS
    assert groups["ibrobot.launch_builders"] == FROZEN_LAUNCH_BUILDERS

    for group, entries in groups.items():
        for name, target in entries.items():
            module_name, _, attribute = target.partition(":")
            assert module_name.startswith(PACKAGE + "."), f"{group}/{name}: {module_name} is outside the package"
            module_file = PACKAGE_ROOT / Path(*module_name.split(".")[1:]).with_suffix(".py")
            assert module_file.is_file(), f"{group}/{name}: {module_file} does not exist"
            assert attribute, f"{group}/{name}: target {target!r} names no attribute"
            attributes = top_level_attributes(module_file.read_text(encoding="utf-8"))
            assert attribute in attributes, f"{group}/{name}: {module_file.name} defines no {attribute}"


# ---- checker unit tests on synthetic negatives (proving the guard fails) ----


def test_checker_flags_relative_private_import():
    source = "from .base import _hidden\n"
    violations = private_import_violations(source, "manipulation_execution.hri.retargeting.plan_builder", frozenset())
    assert any("_hidden" in item for item in violations)

    source = "from ._private import thing\n"
    violations = private_import_violations(source, "manipulation_execution.hri.retargeting.plan_builder", frozenset())
    assert any("private module" in item for item in violations)


def test_checker_flags_private_access_through_a_module_alias():
    source = "import manipulation_execution.pick.node as node_mod\nnode_mod._reload_providers()\n"
    violations = private_import_violations(source, "manipulation_execution.place.node", frozenset())
    assert any("node_mod._reload_providers" in item for item in violations)

    source = "from manipulation_execution.pick import node\nnode._private_helper()\n"
    known = frozenset({"manipulation_execution.pick.node"})
    violations = private_import_violations(source, "manipulation_execution.place.node", known)
    assert any("node._private_helper" in item for item in violations)


def test_checker_flags_lazy_import_inside_a_function():
    source = "def late():\n    from manipulation_execution.hri.animation import _waypoints\n    return _waypoints\n"
    violations = private_import_violations(source, "manipulation_execution.pick.node", frozenset())
    assert any("_waypoints" in item for item in violations)


def test_checker_flags_cross_subsystem_import():
    source = "from manipulation_execution.hri.animation import waypoints\n"
    imported = imported_package_modules(source, "manipulation_execution.pick.node")
    assert imported == frozenset({"manipulation_execution.hri.animation"})
    assert dependency_violations("manipulation_execution.pick.node", imported)


def test_checker_flags_shared_or_geometry_importing_the_package():
    assert dependency_violations("manipulation_execution.geometry", frozenset({"manipulation_execution.pick.node"}))
    assert dependency_violations(
        "manipulation_execution.shared.runtime_ownership", frozenset({"manipulation_execution"})
    )


def test_checker_flags_launch_builders_reaching_into_subsystems():
    imported = frozenset({"manipulation_execution.pick.node"})
    assert dependency_violations("manipulation_execution.launch_builders.mirror_session", imported)
    # ... but its own sibling builder stays within the budget.
    assert not dependency_violations(
        "manipulation_execution.launch_builders.mirror_session",
        frozenset({"manipulation_execution.launch_builders.motion_imitation_stream"}),
    )


def test_checker_ignores_class_attribute_access_and_dunders():
    source = (
        "from manipulation_execution.pick.models import PickExecutorHelpers\n"
        "from manipulation_execution.pick.models import PipelineConfig\n"
        "PickExecutorHelpers._result_from_state(None)\n"
        "PipelineConfig.__dataclass_fields__\n"
    )
    known = frozenset({"manipulation_execution.pick.models"})
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert violations == []


def test_checker_flags_a_subpackage_imported_from_the_package_root():
    known = frozenset({"manipulation_execution.hri", "manipulation_execution.shared"})

    source = "from manipulation_execution import hri\n"
    imported = imported_package_modules(source, "manipulation_execution.pick.node", known)
    assert "manipulation_execution.hri" in imported
    assert dependency_violations("manipulation_execution.pick.node", imported)

    source = "from manipulation_execution import shared\n"
    imported = imported_package_modules(source, "manipulation_execution.launch_builders.x", known)
    assert "manipulation_execution.shared" in imported
    assert dependency_violations("manipulation_execution.launch_builders.x", imported)


def test_checker_flags_retargeting_importing_imitate_through_the_hri_package():
    # Must fail the retargeting scenario check in
    # test_retargeting_never_imports_the_imitation_executor.
    known = frozenset({"manipulation_execution.hri.imitate"})
    source = "from manipulation_execution.hri import imitate\n"
    imported = imported_package_modules(source, "manipulation_execution.hri.retargeting.plan_builder", known)
    assert [target for target in imported if target.startswith(PACKAGE + ".hri.imitate")]


def test_checker_flags_private_access_through_the_full_imported_chain():
    # ``import a.b.c`` binds ``a``; the chain a.b.c resolves the module.
    known = frozenset({"manipulation_execution.geometry"})
    source = "import manipulation_execution.geometry\nmanipulation_execution.geometry._hidden()\n"
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert any("_hidden" in item for item in violations)


def test_checker_flags_import_of_a_private_module():
    source = "import manipulation_execution.pick._hidden as hidden\n"
    violations = private_import_violations(source, "manipulation_execution.place.node", frozenset())
    assert any("private module" in item for item in violations)


def test_checker_ignores_a_module_alias_shadowed_by_a_parameter_or_local():
    known = frozenset({"manipulation_execution.geometry"})
    source = (
        "import manipulation_execution.geometry as mod\n"
        "mod._module_level()\n"
        "def by_parameter(mod):\n"
        "    return mod._instance_method()\n"
        "def by_local():\n"
        "    mod = object()\n"
        "    return mod._local_method()\n"
    )
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    joined = "\n".join(violations)
    assert "_module_level" in joined
    assert "_instance_method" not in joined
    assert "_local_method" not in joined


def test_checker_flags_a_read_shadowed_only_in_a_nested_function():
    # A nested function's own bindings never shadow the enclosing scope's.
    known = frozenset({"manipulation_execution.geometry"})
    source = (
        "import manipulation_execution.geometry as mod\n"
        "def outer():\n"
        "    def inner():\n"
        "        mod = object()\n"
        "    return mod._hidden()\n"
    )
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert any("_hidden" in item for item in violations)


def test_checker_tracks_statement_order_in_a_scope():
    known = frozenset({"manipulation_execution.geometry"})
    # An assignment after the import replaces the module alias: not flagged.
    source = (
        "def assigned_after_import(obj):\n"
        "    import manipulation_execution.geometry as mod\n"
        "    mod = obj\n"
        "    return mod._instance_method()\n"
    )
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert violations == []

    # A non-package from-import rebinds the name: not flagged.
    source = (
        "import manipulation_execution.geometry as mod\n"
        "from collections import Counter as mod\n"
        "mod._keep_positive(mod())\n"
    )
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert violations == []

    # The reverse order re-establishes the alias: flagged again.
    source = (
        "def reimported_after_assignment(obj):\n"
        "    mod = obj\n"
        "    import manipulation_execution.geometry as mod\n"
        "    return mod._hidden()\n"
    )
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert any("_hidden" in item for item in violations)


def test_checker_treats_lambda_parameters_as_shadowing():
    known = frozenset({"manipulation_execution.geometry"})
    source = "import manipulation_execution.geometry as mod\nhandler = lambda mod: mod._instance_method()\n"
    violations = private_import_violations(source, "manipulation_execution.pick.node", known)
    assert violations == []
