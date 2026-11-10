"""Real-ROS fake Gateway regression for the online catalog CLI surface.

The fixture serves the two public services the CLI actually calls and nothing
else. No robot, vendor SDK or Gateway process is involved: a catalog snapshot is
compiled from the checked-in offline fixture and published on the endpoints the
repository G1 preset binds by default.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor

from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot
from robot_config.loader import load_robot_config_dict
from robot_skill_cli import cli
from robot_skill_cli.catalog import compile_local_snapshot

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "test" / "fixtures" / "so101_single_arm_offline.yaml"

# The repository G1 preset declares no endpoint overrides, so these are the
# exact service names its transport projection binds.
STATUS_SERVICE = "/embodied/get_skill_gateway_status"
SNAPSHOT_SERVICE = "/embodied/get_skill_snapshot"


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def _gateway_snapshot():
    # The fixture's so101 profile compiles without logical interfaces; the CLI
    # must project this exact payload even though its configured robot is the
    # repository G1 preset, which has no local skill source at all.
    return compile_local_snapshot(load_robot_config_dict(CONFIG_PATH), CONFIG_PATH)


class _FakeCatalogGateway:
    def __init__(self, snapshot, *, suffix):
        self.snapshot = snapshot
        self.node = rclpy.create_node(f"online_catalog_gateway_{suffix}")
        self.status_requests = []
        self.snapshot_requests = []
        self.node.create_service(GetSkillGatewayStatus, STATUS_SERVICE, self._status)
        self.node.create_service(GetSkillSnapshot, SNAPSHOT_SERVICE, self._snapshot)

    def _status(self, request, response):
        self.status_requests.append(request)
        response.schema_version = 1
        response.robot_name = self.snapshot.robot_name
        response.motion_authorized = False
        response.active_control_mode = ""
        response.busy = False
        response.default_skill_timeout_sec = 30.0
        response.task_budget_sec = 180.0
        response.rpc_timeout_sec = 5.0
        response.config_digest = self.snapshot.capability_digest
        response.registry_epoch = "epoch-1"
        response.registry_generation = 1
        response.registry_digest = self.snapshot.registry_digest
        response.capability_digest = self.snapshot.capability_digest
        response.provenance_digest = self.snapshot.provenance_digest
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        return response

    def _snapshot(self, request, response):
        self.snapshot_requests.append(request)
        response.success = True
        response.registry_epoch = "epoch-1"
        response.generation = 1
        response.registry_digest = self.snapshot.registry_digest
        response.capability_digest = self.snapshot.capability_digest
        response.provenance_digest = self.snapshot.provenance_digest
        response.snapshot_json = self.snapshot.snapshot_json
        return response


def test_online_catalog_commands_read_a_real_gateway_snapshot(monkeypatch, capsys):
    _assert_isolated_ros_domain()
    rclpy.init()
    snapshot = _gateway_snapshot()
    suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    gateway = _FakeCatalogGateway(snapshot, suffix=suffix)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(gateway.node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills", "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert listed["ok"] is True
        assert listed["command"] == "list-skills"
        assert listed["data"]["robot_name"] == snapshot.robot_name
        assert listed["data"]["config_digest"] == snapshot.capability_digest
        skills = {skill["name"]: skill for skill in listed["data"]["skills"]}
        assert "move_relative_ee" in skills

        assert cli.main(["--config-name", "unitree_g1_arm_presets", "describe", "move_relative_ee"]) == 0
        described = json.loads(capsys.readouterr().out)
        assert described["ok"] is True
        assert described["data"]["name"] == "move_relative_ee"
        assert described["data"]["parameters"]["properties"]["motion_distance"]["unit"] == "meters"
        assert described["data"]["config_digest"] == snapshot.capability_digest

        assert len(gateway.status_requests) == 2
        assert len(gateway.snapshot_requests) == 2
        assert [request.registry_epoch for request in gateway.snapshot_requests] == ["epoch-1", "epoch-1"]
        assert [request.generation for request in gateway.snapshot_requests] == [1, 1]
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        gateway.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_online_catalog_without_a_gateway_reports_server_unavailable(monkeypatch, capsys):
    _assert_isolated_ros_domain()
    # Keep the real service wait short: the isolated domain has no Gateway, and
    # this test only cares which stable error the consumer reports.
    monkeypatch.setattr(cli, "_status_preflight_timeout", lambda _context: 0.25)

    assert cli.main(["--config-name", "unitree_g1_arm_presets", "list-skills"]) == 4

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "SERVER_UNAVAILABLE"
    assert "description_required" not in payload["error"]["message"]
    assert "interface_description" not in payload["error"]["message"]
