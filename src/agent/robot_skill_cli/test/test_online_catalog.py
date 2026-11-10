"""Online catalog inspection projects the bound Gateway snapshot.

The repository G1 preset has logical runtime interfaces and no local skill
source, so the old strict loader failed with ``description_required`` before
ROS was even reached. These tests pin the replacement contract: online
``list-skills`` / ``describe`` / ``list-poses`` bind only the transport
projection and read the exact snapshot the Gateway serves, while ``--offline``
keeps the strict local-development loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_skill_cli import catalog, cli
from robot_skill_cli.ros_bridge import BridgeError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "test" / "fixtures" / "so101_single_arm_offline.yaml"


def _snapshot():
    from robot_config.loader import load_robot_config_dict
    from robot_skill_cli.catalog import compile_local_snapshot

    return compile_local_snapshot(load_robot_config_dict(CONFIG_PATH), CONFIG_PATH)


def _status(snapshot, *, ready=True, epoch="epoch-1", generation=1, registry_digest=None):
    return {
        "schema_version": 1,
        "robot_name": "unitree_g1_arm_presets",
        "motion_authorized": False,
        "active_control_mode": "",
        "busy": False,
        "active_task_id": "",
        "default_skill_timeout_sec": 30.0,
        "task_budget_sec": 180.0,
        "rpc_timeout_sec": 5.0,
        "config_digest": snapshot.capability_digest,
        "capability_digest": snapshot.capability_digest,
        "registry_epoch": epoch,
        "registry_generation": generation,
        "registry_digest": registry_digest or snapshot.registry_digest,
        "primitive_contract_digest": snapshot.primitive_contract_digest,
        "source_release_digest": "",
        "provenance_digest": snapshot.provenance_digest,
        "control_plane_ready": ready,
        "control_plane_state": "READY" if ready else "FAILED",
        "control_plane_error_code": "" if ready else "SKILL_REGISTRY_NOT_READY",
        "request_state": "",
        "request_error_code": "",
        "capabilities": [],
    }


class _FakeBridge:
    def __init__(
        self,
        snapshot,
        *,
        status=None,
        status_error=None,
        snapshot_epoch="epoch-1",
        snapshot_generation=1,
        snapshot_error=None,
    ):
        self._snapshot = snapshot
        self._status = status or _status(snapshot)
        self._status_error = status_error
        self._snapshot_epoch = snapshot_epoch
        self._snapshot_generation = snapshot_generation
        self._snapshot_error = snapshot_error
        self.status_timeouts = []
        self.snapshot_requests = []
        self.closed = False

    def start(self):
        return True

    def get_status(self, **kwargs):
        self.status_timeouts.append(kwargs.get("timeout_sec"))
        if self._status_error is not None:
            raise self._status_error
        return self._status

    def get_skill_snapshot(self, **kwargs):
        self.snapshot_requests.append(kwargs)
        if self._snapshot_error is not None:
            code, message = self._snapshot_error
            return {"success": False, "error_code": code, "message": message}
        return {
            "success": True,
            "registry_epoch": self._snapshot_epoch,
            "generation": self._snapshot_generation,
            "registry_digest": self._snapshot.registry_digest,
            "capability_digest": self._snapshot.capability_digest,
            "provenance_digest": self._snapshot.provenance_digest,
            "snapshot_json": self._snapshot.snapshot_json,
        }

    def close(self):
        self.closed = True


@pytest.fixture
def strict_loader_forbidden(monkeypatch):
    """Online catalog commands must never compile the local catalog."""

    def forbidden(**_kwargs):
        raise AssertionError("online catalog command compiled the local catalog")

    monkeypatch.setattr(catalog, "load_catalog_context", forbidden)


def _bind_fake_bridge(monkeypatch, bridge):
    captured = {}

    def create(transport):
        captured["transport"] = transport
        return bridge

    monkeypatch.setattr(cli, "_create_bridge", create)
    return captured


def test_online_list_skills_uses_transport_only_and_the_gateway_snapshot(monkeypatch, capsys, strict_loader_forbidden):
    snapshot = _snapshot()
    bridge = _FakeBridge(snapshot)
    captured = _bind_fake_bridge(monkeypatch, bridge)
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["data"]["robot_name"] == snapshot.robot_name
    assert payload["data"]["config_digest"] == snapshot.capability_digest
    listed = {skill["name"] for skill in payload["data"]["skills"]}
    assert "move_relative_ee" in listed
    assert captured["transport"].status_service == "/embodied/get_skill_gateway_status"
    assert captured["transport"].snapshot_service == "/embodied/get_skill_snapshot"
    assert captured["transport"].robot_id == "unitree_g1_arm_presets"
    assert bridge.snapshot_requests[0]["registry_epoch"] == "epoch-1"
    assert bridge.snapshot_requests[0]["generation"] == 1
    assert bridge.closed is True


def test_online_describe_reads_parameters_from_the_gateway_snapshot(monkeypatch, capsys, strict_loader_forbidden):
    snapshot = _snapshot()
    bridge = _FakeBridge(snapshot)
    _bind_fake_bridge(monkeypatch, bridge)

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "describe", "move_relative_ee"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["data"]["name"] == "move_relative_ee"
    assert payload["data"]["moves_robot"] is True
    assert payload["data"]["parameters"]["properties"]["motion_distance"]["unit"] == "meters"
    assert payload["data"]["config_digest"] == snapshot.capability_digest


def test_online_list_skills_human_renders_the_gateway_snapshot(monkeypatch, capsys, strict_loader_forbidden):
    snapshot = _snapshot()
    _bind_fake_bridge(monkeypatch, _FakeBridge(snapshot))

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills", "--human"]) == 0

    output = capsys.readouterr().out
    assert snapshot.robot_name in output
    assert "move_relative_ee" in output


def test_online_list_poses_reads_the_gateway_snapshot(monkeypatch, capsys, strict_loader_forbidden):
    snapshot = _snapshot()
    _bind_fake_bridge(monkeypatch, _FakeBridge(snapshot))

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-poses"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload["data"]["poses"]) == set(snapshot.robot_context.named_poses)


def test_online_unknown_skill_keeps_the_stable_error_and_exit(monkeypatch, capsys, strict_loader_forbidden):
    _bind_fake_bridge(monkeypatch, _FakeBridge(_snapshot()))

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "describe", "missing"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "UNKNOWN_SKILL"


def test_online_catalog_rejects_a_snapshot_identity_mismatch(monkeypatch, capsys, strict_loader_forbidden):
    snapshot = _snapshot()
    bridge = _FakeBridge(snapshot, status=_status(snapshot, epoch="epoch-2"))
    _bind_fake_bridge(monkeypatch, bridge)

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills"]) == 3

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "SKILL_REGISTRY_VERSION_MISMATCH"
    assert bridge.closed is True


def test_online_catalog_reports_an_unready_control_plane(monkeypatch, capsys, strict_loader_forbidden):
    snapshot = _snapshot()
    _bind_fake_bridge(monkeypatch, _FakeBridge(snapshot, status=_status(snapshot, ready=False)))

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills"]) == 3

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "SKILL_REGISTRY_NOT_READY"


def test_online_catalog_reports_a_snapshot_failure_code(monkeypatch, capsys, strict_loader_forbidden):
    bridge = _FakeBridge(
        _snapshot(),
        snapshot_error=("SKILL_SNAPSHOT_NOT_FOUND", "registry snapshot is not published"),
    )
    _bind_fake_bridge(monkeypatch, bridge)

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills"]) == 3

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "SKILL_SNAPSHOT_NOT_FOUND"
    assert payload["error"]["message"] == "registry snapshot is not published"


def test_online_catalog_without_a_gateway_is_server_unavailable(monkeypatch, capsys, strict_loader_forbidden):
    bridge = _FakeBridge(
        _snapshot(),
        status_error=BridgeError(
            "SERVER_UNAVAILABLE",
            "service unavailable: /embodied/get_skill_gateway_status",
            exit_code=cli.EXIT_ROS_UNAVAILABLE,
        ),
    )
    _bind_fake_bridge(monkeypatch, bridge)

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills"]) == 4

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "SERVER_UNAVAILABLE"
    assert "description_required" not in payload["error"]["message"]
    assert bridge.closed is True


def test_offline_catalog_keeps_the_strict_local_loader(monkeypatch, capsys):
    def forbidden(_transport):
        raise AssertionError("offline catalog inspection must not start ROS")

    monkeypatch.setattr(cli, "_create_bridge", forbidden)

    assert cli.main(["--offline", "--config-name", "unitree_g1_arm_presets", "list-skills"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "description_required"
    assert "runtime.interface_description" in payload["error"]["message"]
