# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Packaging a GraspGen bundle is a ``perception_service`` command.

It used to be ``ros2 run model_utils package-graspgen-ascend-deployment``, which is where
the LeRobot policy assets came from: the packager lived beside the exporters and wrote what
the policy loader wanted to see. The bundle it writes is a perception bundle, read by the
adapter and the session in this package, so the command ships here alongside the other two
perception packagers.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_SETUP_PY = Path(__file__).resolve().parents[1] / "setup.py"

_PACKAGERS = {
    "package_perception_bundles": "perception_service.package_perception_bundles:main",
    "package_ascend_perception_bundles": "perception_service.package_ascend_perception_bundles:main",
    "package_graspgen_ascend_bundle": "perception_service.package_graspgen_ascend_bundle:main",
}


def _console_scripts() -> dict[str, str]:
    """Read the declared scripts from setup.py without installing the package."""
    tree = ast.parse(_SETUP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "entry_points":
                entry_points = ast.literal_eval(keyword.value)
                return dict(entry.split(" = ", 1) for entry in entry_points["console_scripts"])
    raise AssertionError(f"no setup(entry_points=...) call found in {_SETUP_PY}")


def test_every_bundle_packager_is_an_installed_command():
    declared = _console_scripts()

    assert {name: declared.get(name) for name in _PACKAGERS} == _PACKAGERS


def test_removed_grounded_sam2_commands_are_not_installed():
    declared = _console_scripts()

    assert "grounded_sam2_node" not in declared
    assert "grounded_sam2_snapshot" not in declared


@pytest.mark.parametrize("target", sorted(_PACKAGERS.values()))
def test_packager_command_targets_resolve(target):
    module_name, attribute = target.split(":")

    assert callable(getattr(importlib.import_module(module_name), attribute))
