"""A pose_composer skill takes exactly one argument: the pose in the user's words."""

import pytest
from test_runtime_named_motion_catalog import _compile, _context, _diagnostics, _write_catalog

from skill_catalog.models import SkillCompileError

IMPLEMENTATION = """schema_version: 1
kind: delegated_executor
robot: runtime_v1
executor: pose_composer
required_args: [target_name]
timeout_sec: 90.0
"""
FREEFORM = "{target_name: {type: string, freeform: true}}"


def test_a_description_skill_compiles(tmp_path):
    _write_catalog(tmp_path, implementation=IMPLEMENTATION, parameters=FREEFORM)
    snapshot = _compile(tmp_path, _context("pose_composer"))
    assert snapshot.templates["handshake"]["executor"] == "pose_composer"
    assert list(snapshot.templates["handshake"]["required_args"]) == ["target_name"]


@pytest.mark.parametrize(
    ("required", "parameters"),
    [
        ("[arm_side]", "{arm_side: {type: string, enum: [left, right]}}"),
        ("[target_name]", FREEFORM[:-1] + ", arm_side: {type: string, enum: [left, right]}}"),
    ],
)
def test_any_other_argument_shape_is_refused(tmp_path, required, parameters):
    implementation = IMPLEMENTATION.replace("[target_name]", required)
    _write_catalog(tmp_path, implementation=implementation, parameters=parameters)
    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path, _context("pose_composer"))
    assert (
        "required_args",
        "pose_composer skills take exactly target_name (the pose in the user's words)",
    ) in _diagnostics(raised)
