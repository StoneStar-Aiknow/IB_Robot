"""Catalog rules for runtime_play_audio skills, with and without a bound motion."""

from pathlib import Path

import pytest
from test_runtime_named_motion_catalog import _context, _diagnostics, _robot

from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import SkillCompileContext, SkillCompileError
from skill_catalog.source import DevelopmentStagingSkillSource

CLIP_ONLY = {"clip": "self_intro"}
WITH_WAVE = {"clip": "self_intro", "motion": "wave"}


def _both_executors() -> SkillCompileContext:
    delegated = {
        **_context("runtime_named_motion").delegated_executors,
        **_context("runtime_play_audio").delegated_executors,
    }
    base = _context()
    return SkillCompileContext(
        robot=_robot(),
        primitive_contracts=base.primitive_contracts,
        primitive_contract_digest=base.primitive_contract_digest,
        delegated_executors=delegated,
    )


def _write(
    root: Path,
    *,
    binding: dict,
    moves_robot: bool = True,
    control_mode: str = "named_motion",
    capabilities: str | None = "[motion.named]",
) -> None:
    package = root / "config" / "skills" / "self_introduction"
    (package / "implementations").mkdir(parents=True)
    (root / "config" / "profiles").mkdir(parents=True)
    (root / "config" / "profiles" / "motion_owning_robot.yaml").write_text(
        """schema_version: 1
name: motion_owning_robot
robot_name: motion_owning_robot
enabled_skills:
  - name: self_introduction
    implementation: runtime_v1
    planner_visible: true
""",
        encoding="utf-8",
    )
    scope = "[arm]" if moves_robot else "[]"
    required = f"  required_capabilities: {capabilities}\n" if capabilities is not None else ""
    (package / "manifest.yaml").write_text(
        f"""schema_version: 1
name: self_introduction
version: 1.0.0
semantic_level: skill
description:
  summary: Introduce the robot.
  category: social_interaction
  when_to_use: [introduce itself]
  motion_scope: {scope}
  intensity: moderate
capability:
  schema_version: 1
  summary: Introduce the robot.
  domain: social
  moves_robot: {str(moves_robot).lower()}
  required_control_mode: {control_mode}
  parameters:
    type: object
    properties: {{}}
    required: []
    additionalProperties: false
  recovery_policy: never_retry
{required}implementations:
  runtime_v1: implementations/runtime_v1.yaml
""",
        encoding="utf-8",
    )
    lines = "\n".join(f"  {key}: {value}" for key, value in binding.items())
    (package / "implementations" / "runtime_v1.yaml").write_text(
        f"""schema_version: 1
kind: delegated_executor
robot: runtime_v1
executor: runtime_play_audio
binding:
{lines}
required_args: []
timeout_sec: 60.0
""",
        encoding="utf-8",
    )


def _compile(root: Path, context: SkillCompileContext | None = None):
    return compile_skill_catalog(
        DevelopmentStagingSkillSource(root), profile_name="motion_owning_robot", context=context or _both_executors()
    )


def test_a_clip_with_a_motion_compiles_both_into_the_template(tmp_path):
    _write(tmp_path, binding=WITH_WAVE)

    template = _compile(tmp_path).templates["self_introduction"]

    assert template["executor"] == "runtime_play_audio"
    assert template["binding"] == WITH_WAVE


def test_a_clip_alone_still_compiles_as_a_non_moving_skill(tmp_path):
    _write(tmp_path, binding=CLIP_ONLY, moves_robot=False, capabilities=None)

    assert _compile(tmp_path).templates["self_introduction"]["binding"] == CLIP_ONLY


@pytest.mark.parametrize(
    ("overrides", "field", "message"),
    [
        ({"moves_robot": False}, "capability.moves_robot", "capability.moves_robot must be true"),
        ({"capabilities": None}, "capability.required_capabilities", "must require motion.named or motion.posture"),
        ({"binding": {"clip": "self_intro", "motion": "Wave-Hand"}}, "binding.motion", "runtime motion name"),
    ],
)
def test_a_clip_bound_with_a_motion_must_declare_what_it_moves(tmp_path, overrides, field, message):
    _write(tmp_path, **{"binding": WITH_WAVE, **overrides})

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path)

    assert any(path == field and message in text for path, text in _diagnostics(raised))


def test_a_clip_bound_with_a_motion_needs_the_named_motion_executor(tmp_path):
    _write(tmp_path, binding=WITH_WAVE)

    with pytest.raises(SkillCompileError) as raised:
        _compile(tmp_path, _context("runtime_play_audio"))

    assert ("binding.motion", "a clip played with a motion needs the runtime_named_motion executor") in _diagnostics(
        raised
    )
