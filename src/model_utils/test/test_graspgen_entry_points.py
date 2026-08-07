# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""The GraspGen export steps have to be reachable as installed commands.

Only the packager had a console script, so the two steps that produce its inputs had to
be run as ``python3 -m ...`` against a source checkout - which silently bypasses the
installed package and is not what the deployment documentation can ask an operator to do.
This pins them to entry points that actually resolve.

``model_utils`` owns the export: checkpoints to ONNX subgraphs to OMs. Packaging those OMs
into a bundle is the third step and it belongs to the package that owns the model, so it
is a ``perception_service`` command now - see that package's entry-point test.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_SETUP_PY = Path(__file__).resolve().parents[1] / "setup.py"

_GRASPGEN_COMMANDS = {
    "graspgen-export-onnx": "model_utils.graspgen_export.export_onnx:main",
    "graspgen-onnx-to-om": "model_utils.graspgen_export.onnx2om:main",
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


def test_every_graspgen_export_step_is_an_installed_command():
    declared = _console_scripts()

    assert {name: declared.get(name) for name in _GRASPGEN_COMMANDS} == _GRASPGEN_COMMANDS


def test_the_bundle_packager_is_no_longer_a_model_utils_command():
    """It writes a perception bundle, so it ships with the package that reads one."""
    declared = _console_scripts()

    assert not [name for name in declared if "package" in name and "graspgen" in name]


@pytest.mark.parametrize("target", sorted(_GRASPGEN_COMMANDS.values()))
def test_graspgen_command_targets_resolve(target):
    module_name, attribute = target.split(":")

    assert callable(getattr(importlib.import_module(module_name), attribute))
