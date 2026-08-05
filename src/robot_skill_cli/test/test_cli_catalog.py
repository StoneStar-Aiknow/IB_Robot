import argparse
import builtins
import importlib
import json
import math
from pathlib import Path

import pytest

from embodied_common.skill_request import canonical_skill_payload, skill_payload_hash
from robot_config.loader import load_robot_config_dict
from robot_config.timeout_policy import resolve_embodied_timeout_policy

CONFIG_PATH = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"


def test_catalog_import_does_not_load_rclpy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rclpy" or name.startswith("rclpy."):
            raise AssertionError("catalog-only commands must not import rclpy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("robot_skill_cli.catalog")
    importlib.reload(module)


def test_status_preflight_timeout_has_hardware_discovery_floor():
    from robot_skill_cli.cli import _status_preflight_timeout

    context = type("Context", (), {"view": {"timeout_policy": {"rpc_timeout_sec": 5.0}}})()
    assert _status_preflight_timeout(context) == 15.0


def test_load_catalog_uses_exported_config_resolver(monkeypatch):
    from robot_skill_cli import catalog

    calls = []
    monkeypatch.setattr(
        catalog,
        "resolve_robot_config_path",
        lambda *, config_name, config_path: calls.append((config_name, config_path)) or CONFIG_PATH,
    )

    view = catalog.load_capability_catalog(config_name="selected", config_path=None)

    assert calls == [("selected", None)]
    assert view["robot_name"] == "so101_single_arm"


def test_list_skills_uses_every_profile_enabled_entry_and_only_public_fields():
    from robot_skill_cli.catalog import list_skills, load_capability_catalog

    data = list_skills(load_capability_catalog(config_path=CONFIG_PATH))

    assert len(data["skills"]) == 16
    assert [skill["name"] for skill in data["skills"]] == sorted(skill["name"] for skill in data["skills"])
    assert all(
        set(skill)
        == {
            "name",
            "summary",
            "domain",
            "moves_robot",
            "required_control_mode",
        }
        for skill in data["skills"]
    )
    encoded = json.dumps(data)
    for forbidden in (
        "primitive_sequence",
        "primitive_name",
        "rule_entry",
        "target_pose_key",
        "place_name_from_request",
        "motion_direction_from_request",
        "motion_distance_from_request",
        "joint_positions",
        "description",
    ):
        assert forbidden not in encoded


def test_catalog_list_and_describe_project_explicit_capability_fields():
    from robot_skill_cli.catalog import describe_skill, list_skills, load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    listed = list_skills(view)
    described = describe_skill(view, "move_relative_ee")

    assert set(listed["skills"][0]) == {
        "name",
        "summary",
        "domain",
        "moves_robot",
        "required_control_mode",
    }
    assert described["summary"] == "Move the end effector in a requested direction by a requested distance."
    assert described["domain"] == "manipulation"
    assert described["moves_robot"] is True
    assert described["required_control_mode"] == "moveit_planning"
    assert described["parameters"]["properties"]["motion_distance"]["unit"] == "meters"
    assert described["recovery_policy"] == "ask_user"
    assert "description" not in described


def test_describe_exposes_derived_schema_timeout_policy_and_digest():
    from robot_skill_cli.catalog import describe_skill, load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    relative = describe_skill(view, "move_relative_ee")
    rotation = describe_skill(view, "rotate_gripper_cw")

    assert relative["parameters"]["required"] == ["motion_direction", "motion_distance"]
    assert relative["parameters"]["properties"]["motion_direction"]["enum"] == [
        "forward",
        "backward",
        "left",
        "right",
        "up",
        "down",
    ]
    assert relative["parameters"]["properties"]["motion_distance"] == {
        "exclusiveMinimum": 0,
        "type": "number",
        "unit": "meters",
    }
    assert rotation["parameters"]["properties"]["motion_distance"]["unit"] == "degrees"
    assert relative["timeout_policy"] == resolve_embodied_timeout_policy(
        load_robot_config_dict(CONFIG_PATH)["embodied"]
    )
    assert relative["config_digest"] == view["capability_digest"]


def test_catalog_uses_capability_metadata_without_legacy_descriptions():
    from embodied_common.capability_view import build_capability_view
    from robot_skill_cli.catalog import describe_skill, list_skills

    view = build_capability_view(
        {
            "name": "fallback_robot",
            "embodied": {
                "skill_templates": {
                    "fallback": {
                        "capability": {
                            "schema_version": 1,
                            "summary": "Open the fallback gripper.",
                            "domain": "manipulation",
                            "moves_robot": True,
                            "required_control_mode": "moveit_planning",
                            "parameters": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {},
                                "required": [],
                            },
                            "recovery_policy": "never_retry",
                        },
                        "description": {"summary": "Private fallback description."},
                        "primitive_sequence": [{"primitive_name": "open_gripper"}],
                    }
                }
            },
        },
        timeout_policy={"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
    )

    listed = list_skills(view)["skills"][0]
    described = describe_skill(view, "fallback")

    assert listed == {
        "name": "fallback",
        "summary": "Open the fallback gripper.",
        "domain": "manipulation",
        "moves_robot": True,
        "required_control_mode": "moveit_planning",
    }
    assert "description" not in described


def test_catalog_requires_an_explicit_profile(tmp_path):
    from robot_skill_cli.catalog import load_capability_catalog

    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: catalog_robot\n", encoding="utf-8")

    with pytest.raises(KeyError, match="embodied"):
        load_capability_catalog(config_path=config_path)


def test_unknown_skill_has_stable_error_and_exit_code(capsys):
    from robot_skill_cli.cli import main

    exit_code = main(["--config-path", str(CONFIG_PATH), "describe", "missing"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    assert payload["error"]["code"] == "UNKNOWN_SKILL"


def test_catalog_commands_emit_one_json_document(capsys):
    from robot_skill_cli.cli import main

    for arguments in (["list-skills"], ["describe", "move_relative_ee"], ["list-poses"]):
        exit_code = main(["--config-path", str(CONFIG_PATH), *arguments])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.err == ""
        assert len(captured.out.strip().splitlines()) == 1
        payload = json.loads(captured.out)
        assert payload["schema_version"] == 1
        assert payload["ok"] is True


def test_agent_plan_parser_uses_frozen_text_option_and_lifecycle_commands():
    from robot_skill_cli.cli import _build_parser

    parser = _build_parser()
    plan = parser.parse_args(["plan-text", "--request-id", "request-1", "--text", "打开夹爪"])
    confirm = parser.parse_args(
        [
            "confirm-plan",
            "--plan-token",
            "plan-token",
            "--plan-digest",
            "digest",
            "--task-id",
            "task-1",
        ]
    )

    assert plan.raw_command == "打开夹爪"
    assert plan.request_id == "request-1"
    assert confirm.command == "confirm-plan"


def test_runtime_capability_view_rejects_tampered_snapshot() -> None:
    from robot_skill_cli.catalog import capability_view_from_snapshot

    bridge = _FakeBridge(_ready_status("legacy-digest"))
    status = bridge.get_status()
    snapshot = bridge.get_skill_snapshot()
    snapshot["snapshot_json"] += " "

    with pytest.raises(ValueError, match="SKILL_SNAPSHOT_DIGEST_MISMATCH"):
        capability_view_from_snapshot(snapshot, status)


def test_skill_payload_normalization_and_default_timeout_identity():
    omitted = canonical_skill_payload(
        " move_relative_ee ",
        motion_direction=" FoRwArD ",
        motion_distance=-0.0,
        default_timeout_sec=30.0,
    )
    explicit = canonical_skill_payload(
        "move_relative_ee",
        motion_direction="forward",
        motion_distance=0.0,
        timeout_sec=30,
        default_timeout_sec=30.0,
    )

    assert omitted == explicit
    assert omitted["motion_distance"] == 0.0
    assert math.copysign(1.0, omitted["motion_distance"]) == 1.0
    assert skill_payload_hash(omitted) == skill_payload_hash(explicit)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skill_name": ""},
        {"skill_name": "demo", "motion_distance": float("nan")},
        {"skill_name": "demo", "timeout_sec": float("inf")},
    ],
)
def test_skill_payload_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        canonical_skill_payload(default_timeout_sec=30.0, **kwargs)


def test_schema_allows_omitted_optional_declared_parameter():
    from robot_skill_cli.cli import _validate_schema

    skill = {
        "name": "optional_relative_motion",
        "parameters": {
            "properties": {
                "motion_direction": {"type": "string", "enum": ["forward"]},
                "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            "required": ["motion_direction"],
        },
    }
    args = argparse.Namespace(
        target_name=None,
        place_name=None,
        motion_direction="forward",
        motion_distance=None,
    )

    _validate_schema(skill, args)


def test_schema_rejects_omitted_declared_required_parameter():
    from robot_skill_cli.cli import _CliArgumentError, _validate_schema

    skill = {
        "name": "optional_relative_motion",
        "parameters": {
            "properties": {
                "motion_direction": {"type": "string", "enum": ["forward"]},
                "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            "required": ["motion_direction"],
        },
    }
    args = argparse.Namespace(
        target_name=None,
        place_name=None,
        motion_direction=None,
        motion_distance=None,
    )

    with pytest.raises(_CliArgumentError, match="motion_direction is required for skill optional_relative_motion"):
        _validate_schema(skill, args)


def test_schema_validates_supplied_optional_parameter():
    from robot_skill_cli.cli import _CliArgumentError, _validate_schema

    skill = {
        "name": "optional_relative_motion",
        "parameters": {
            "properties": {
                "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            "required": [],
        },
    }
    args = argparse.Namespace(
        target_name=None,
        place_name=None,
        motion_direction=None,
        motion_distance=0.0,
    )

    with pytest.raises(_CliArgumentError, match="motion_distance must be a finite number greater than zero"):
        _validate_schema(skill, args)


class _FakeBridge:
    def __init__(self, status):
        self.status = status
        self.calls = []
        self.validate_payloads = []
        self.status_timeouts = []
        self.reload_requests = []
        self._snapshot = None

    def start(self):
        self.calls.append("start")
        return True

    def get_status(self, **kwargs):
        self.calls.append("status")
        self.status_timeouts.append(kwargs.get("timeout_sec"))
        if self._snapshot is None:
            self._snapshot = self._build_snapshot()
            self.status.update(
                registry_epoch="epoch-1",
                registry_generation=1,
                registry_digest=self._snapshot.registry_digest,
                capability_digest=self._snapshot.capability_digest,
            )
        return self.status

    def validate_skill(self, payload, **kwargs):
        self.calls.append("validate")
        self.validate_payloads.append(payload)
        return {"allowed": True, "reason": "allowed"}

    def reload_skill_catalog(self, **kwargs):
        self.reload_requests.append(kwargs)
        return {
            "success": True,
            "registry_epoch": "epoch-2",
            "old_generation": 1,
            "generation": 2,
            "registry_digest": "registry-2",
            "capability_digest": "capability-2",
            "source_release_digest": "source-2",
            "provenance_digest": "provenance-2",
            "error_code": "",
            "message": "reloaded",
            "changed_skills": ["nod_yes"],
            "diagnostics": [],
        }

    def get_skill_snapshot(self, **_kwargs):
        snapshot = self._snapshot or self._build_snapshot()
        return {
            "success": True,
            "registry_epoch": "epoch-1",
            "generation": 1,
            "registry_digest": snapshot.registry_digest,
            "capability_digest": snapshot.capability_digest,
            "provenance_digest": snapshot.provenance_digest,
            "snapshot_json": snapshot.snapshot_json,
        }

    @staticmethod
    def _build_snapshot():
        from robot_config.loader import load_robot_config_dict
        from robot_skill_cli.catalog import compile_local_snapshot

        config = load_robot_config_dict(CONFIG_PATH)
        return compile_local_snapshot(config, CONFIG_PATH)

    def close(self):
        self.calls.append("close")


def _ready_status(config_digest, *, ready=True, reason=""):
    return {
        "schema_version": 1,
        "robot_name": "so101_single_arm",
        "motion_authorized": ready,
        "active_control_mode": "moveit_planning",
        "busy": False,
        "active_task_id": "",
        "default_skill_timeout_sec": 30.0,
        "task_budget_sec": 180.0,
        "rpc_timeout_sec": 5.0,
        "config_digest": config_digest,
        "registry_epoch": "epoch-1",
        "registry_generation": 1,
        "registry_digest": "registry-digest",
        "request_state": "",
        "request_error_code": "",
        "capabilities": [
            {
                "name": "move_relative_ee",
                "ready": ready,
                "reason": reason,
                "required_control_mode": "moveit_planning",
            },
            {
                "name": "open_gripper_skill",
                "ready": ready,
                "reason": reason,
                "required_control_mode": "moveit_planning",
            },
        ],
    }


def test_status_command_uses_hardware_discovery_timeout_floor(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _FakeBridge(_ready_status("legacy-digest"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _transport: bridge)

    assert cli.main(["--config-path", str(CONFIG_PATH), "status"]) == 0

    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert bridge.status_timeouts == [15.0]


def test_reload_catalog_command_uses_bound_gateway_source(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _FakeBridge(_ready_status("legacy-digest"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _transport: bridge)

    assert (
        cli.main(
            [
                "--config-path",
                str(CONFIG_PATH),
                "reload-catalog",
                "--request-id",
                "reload-1",
                "--force",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["data"]["changed_skills"] == ["nod_yes"]
    assert bridge.reload_requests == [{"request_id": "reload-1", "force": True, "timeout_sec": 60.0}]


def test_validate_uses_verified_snapshot_instead_of_legacy_config_digest(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status("different-digest"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "1.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert view["capability_digest"] != bridge.status["config_digest"]
    assert exit_code == 0
    assert payload["ok"] is True
    assert bridge.calls == ["start", "status", "validate", "close"]


def test_validate_checks_schema_before_readiness_or_safety(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="MOTION_NOT_AUTHORIZED"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "open_gripper_skill",
            "--motion-distance",
            "1.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert bridge.calls == ["start", "status", "close"]


def test_validate_checks_runtime_readiness_before_safety(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="MOTION_NOT_AUTHORIZED"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"]["code"] == "MOTION_NOT_AUTHORIZED"
    assert payload["error"]["message"] == "MOTION_NOT_AUTHORIZED"
    assert bridge.calls == ["start", "status", "close"]


@pytest.mark.parametrize(
    "reason",
    [
        "MOTION_NOT_AUTHORIZED: operator authorization is disabled",
        "CONTROL_MODE_MISMATCH: requires moveit_planning, active mode is teleop",
        "SKILL_BUSY: another root execution is active",
        "CAPABILITY_NOT_READY: validate skill service unavailable",
        "CAPABILITY_NOT_READY: task executor action unavailable",
        "CAPABILITY_NOT_READY: arm trajectory action unavailable",
        "CAPABILITY_NOT_READY: ee pose unavailable or stale",
    ],
)
def test_validate_preserves_detailed_gateway_readiness_reason(monkeypatch, capsys, reason):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason=reason))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"] == {"code": reason.split(":", 1)[0], "message": reason}
    assert bridge.calls == ["start", "status", "close"]


def test_validate_preserves_legacy_bare_gateway_reason(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="CAPABILITY_NOT_READY"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"] == {"code": "CAPABILITY_NOT_READY", "message": "CAPABILITY_NOT_READY"}
    assert bridge.calls == ["start", "status", "close"]


def test_validate_splits_only_first_gateway_reason_colon(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    reason = "CAPABILITY_NOT_READY: ee pose unavailable or stale: gateway snapshot retained"
    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason=reason))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"] == {"code": "CAPABILITY_NOT_READY", "message": reason}


def test_validate_checks_timeout_schema_before_runtime_readiness(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="MOTION_NOT_AUTHORIZED"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
            "--timeout-sec",
            "999.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert bridge.calls == ["start", "status", "close"]


def test_validate_uses_status_default_timeout_and_shared_payload_hash(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"]))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    base_arguments = [
        "--config-path",
        str(CONFIG_PATH),
        "validate",
        "move_relative_ee",
        "--motion-direction",
        "forward",
        "--motion-distance",
        "0.03",
    ]

    assert cli.main(base_arguments) == 0
    omitted = json.loads(capsys.readouterr().out)
    assert cli.main([*base_arguments, "--timeout-sec", "30.0"]) == 0
    explicit = json.loads(capsys.readouterr().out)

    assert omitted["data"]["payload"] == explicit["data"]["payload"]
    assert omitted["data"]["payload_hash"] == explicit["data"]["payload_hash"]
    assert bridge.validate_payloads[0] == bridge.validate_payloads[1]


def test_validate_uses_action_wire_zero_for_omitted_motion_distance(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"]))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    assert cli.main(["--config-path", str(CONFIG_PATH), "validate", "open_gripper_skill"]) == 0

    response = json.loads(capsys.readouterr().out)
    assert response["data"]["payload"]["motion_distance"] == 0.0
    assert response["data"]["payload_hash"] == skill_payload_hash(response["data"]["payload"])
    assert bridge.validate_payloads == [response["data"]["payload"]]
