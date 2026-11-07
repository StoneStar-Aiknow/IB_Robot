"""The opt-in catalog exposes deterministic preparation as part of pick semantics."""

from dataclasses import replace
from pathlib import Path

import pytest
from test_runtime_named_motion_catalog import _context

from skill_catalog.compiler import compile_skill_catalog
from skill_catalog.models import SkillCompileError
from skill_catalog.source import DevelopmentStagingSkillSource


def test_task_pose_profile_compiles_with_both_delegated_executors():
    context = _context("pose_composer")
    robot = replace(
        context.robot,
        robot_name="aimdk_x2_grasp",
        required_control_mode="moveit_planning",
        timeout_policy={"task_budget_sec": 180.0},
    )
    delegated = {**context.delegated_executors, **_context("grasp_pipeline").delegated_executors}
    delegated["grasp_pipeline"] = replace(
        delegated["grasp_pipeline"],
        model_deployment_name="mock",
        model_fingerprint="0" * 64,
        model_bundle_digest="1" * 64,
    )
    context = replace(context, robot=robot, delegated_executors=delegated)
    root = Path(__file__).resolve().parents[1]
    snapshot = compile_skill_catalog(
        DevelopmentStagingSkillSource(root), profile_name="aimdk_x2_grasp_task_poses", context=context
    )
    assert set(snapshot.enabled_skill_names) == {"pick_object", "return_to_task_pose"}
    assert snapshot.templates["return_to_task_pose"]["executor"] == "pose_composer"
    assert snapshot.templates["pick_object"]["implementation_identity"].endswith("#aimdk_x2_grasp_task_poses")
    assert "preparation posture" in str(snapshot.capability_view)


def _voice_mirror_grasp_context():
    context = _context()
    robot = replace(
        context.robot,
        robot_name="aimdk_x2_voice_mirror_grasp",
        supported_control_modes=("named_motion", "model_inference", "moveit_planning"),
        timeout_policy={"task_budget_sec": 180.0},
    )
    delegated = dict(context.delegated_executors)
    for executor in ("pose_composer", "grasp_pipeline", "mirror_session", "runtime_play_audio"):
        delegated.update(_context(executor).delegated_executors)
    delegated["grasp_pipeline"] = replace(
        delegated["grasp_pipeline"],
        model_deployment_name="mock",
        model_fingerprint="0" * 64,
        model_bundle_digest="1" * 64,
    )
    return replace(context, robot=robot, delegated_executors=delegated)


def test_voice_mirror_grasp_preserves_skills_and_compiles_preparation_budget():
    root = Path(__file__).resolve().parents[1]
    snapshot = compile_skill_catalog(
        DevelopmentStagingSkillSource(root),
        profile_name="aimdk_x2_voice_mirror_grasp",
        context=_voice_mirror_grasp_context(),
    )
    expected = {
        "wave_hand",
        "handshake",
        "raise_hand",
        "blow_kiss",
        "clap_hands",
        "mirror_start",
        "mirror_stop",
        "self_introduction",
        "pick_object",
        "free_pose",
        "return_to_task_pose",
    }
    assert set(snapshot.enabled_skill_names) == expected
    assert set(snapshot.planner_visible_skill_names) == expected
    assert snapshot.robot_name == "aimdk_x2_voice_mirror_grasp"
    assert snapshot.templates["return_to_task_pose"]["executor"] == "pose_composer"
    pick = snapshot.templates["pick_object"]
    assert pick["implementation_identity"].endswith("#aimdk_x2_grasp_task_poses")
    assert pick["executor"] == "grasp_pipeline"
    assert pick["timeout_sec"] == 150.0
    assert "preparation posture" in str(snapshot.capability_view["pick_object"])
    assert snapshot.templates["mirror_start"]["executor"] == "mirror_session"
    assert snapshot.templates["self_introduction"]["executor"] == "runtime_play_audio"
    assert snapshot.templates["self_introduction"]["binding"] == {"clip": "self_intro", "motion": "wave"}
    assert snapshot.capability_view["self_introduction"]["moves_robot"] is True


@pytest.mark.parametrize("missing", ["pose_composer", "grasp_pipeline"])
def test_voice_mirror_grasp_requires_both_task_pose_and_grasp_executors(missing):
    context = _voice_mirror_grasp_context()
    delegated = dict(context.delegated_executors)
    delegated.pop(missing)
    with pytest.raises(SkillCompileError) as raised:
        compile_skill_catalog(
            DevelopmentStagingSkillSource(Path(__file__).resolve().parents[1]),
            profile_name="aimdk_x2_voice_mirror_grasp",
            context=replace(context, delegated_executors=delegated),
        )
    assert any(item.message == "delegated executor is not registered" for item in raised.value.diagnostics)


def test_configured_pose_choices_bind_catalog_schema_and_roundtrip(tmp_path):
    import hashlib
    import json

    import jsonschema
    import yaml

    from robot_runtime.task_poses import normalize_task_poses, pose_catalog
    from skill_catalog.digest import derive_capability_view_from_registry

    root = Path(__file__).resolve().parents[1]
    task_file = root.parents[1] / "robot_config/config/task_poses/x2_example.yaml"
    tasks = normalize_task_poses(yaml.safe_load(task_file.read_text()))
    artifact = tmp_path / "fixed.json"
    artifact.write_text(
        json.dumps(
            {
                "poses": {
                    "arms_forward": {"user_message": "双臂向前伸直", "phrases": ["双手向前伸直", "双臂前举"]},
                    "arms_down": {"user_message": "回到准备位 HOME", "phrases": ["双臂放下"]},
                }
            }
        )
    )
    composer = {
        "mode": "fixed",
        "fixed_library": str(artifact),
        "fixed_library_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    choices = pose_catalog(tasks, composer)
    # Live startup serializes the exact same pure-data options as local compilation.
    live_choices = json.loads(json.dumps(choices))
    context = _voice_mirror_grasp_context()
    snapshots = [
        compile_skill_catalog(
            DevelopmentStagingSkillSource(root),
            profile_name="aimdk_x2_voice_mirror_grasp",
            context=replace(context, pose_catalog=options),
        )
        for options in (choices, live_choices)
    ]
    local, live = snapshots
    assert local.registry_digest == live.registry_digest
    assert local.capability_view == live.capability_view
    rebuilt = derive_capability_view_from_registry(json.loads(local.registry_preimage_json))
    for skill, valid, invalid in (
        ("return_to_task_pose", "grasp_ready", "准备位"),
        ("free_pose", "arms_forward", "双手向前伸直"),
    ):
        schema = live.parameter_schemas[skill]
        assert valid in schema["properties"]["target_name"]["enum"]
        assert rebuilt[skill]["target_options"] == choices[skill]
        # Existing Gateway schema admission uses this same JSON Schema contract.
        schema = json.loads(json.dumps(rebuilt[skill]["parameters"]))
        jsonschema.validate({"target_name": valid}, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"target_name": invalid}, schema)
    assert "HOME" in choices["return_to_task_pose"][0]["description"]
    assert "双臂前举" in choices["free_pose"][1]["aliases"]
    generated = compile_skill_catalog(
        DevelopmentStagingSkillSource(root),
        profile_name="aimdk_x2_voice_mirror_grasp",
        context=replace(context, pose_catalog=pose_catalog(tasks, {})),
    )
    assert generated.parameter_schemas["free_pose"]["properties"]["target_name"]["freeform"] is True
    assert "target_options" not in generated.capability_view["free_pose"]
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        pose_catalog(tasks, composer)
