"""The bound client starts its transport on the first read, not at binding.

Regression: the config-binding commit dropped the ``bridge.start()`` call from
the public client. The CLI kept starting its own bridge, so ``robot-skill
status`` stayed green, but a process that only binds the config and then reads
(the Web workbench) was left with a transport that was never started. Its first
``get_status``/``list_skills`` raised ``BridgeError: ROS_UNAVAILABLE: ROS bridge
is not started`` from ``ResidentBridge._forward``, which the HTTP adapter could
only render as ``500 internal adapter error`` while the Gateway was online and
ready.

These tests use the real default transport factory (``ResidentBridge`` and the
in-process ``RosBridge``), not a stand-in: a real rclpy participant talks to the
minimal Gateway fixture below over an isolated ROS domain. They cover:

* daemon absent (in-process bridge): the first read starts it once and returns
  the Gateway's own status/catalog;
* daemon present: the same first read goes through the resident socket;
* start failure (in-process and daemon): ``ROS_UNAVAILABLE`` with the
  transport's ``start_error`` text, never the bare "not started" message, and
  no half-started bridge kept for later reads.

No robot is connected and nothing is commanded: the fixture serves only the two
read-only services, and every call is a status or catalog read.
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

from embodied_common.catalog_digest import (
    derive_capability_digest,
    derive_capability_preimage,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)
from robot_client.client import RobotClient
from robot_client.errors import ClientError
from robot_client.transport import gateway_transport

ROBOT_NAME = "pr590_g1"
SKILL_NAMES = ("arms_horizontal", "wave_hello")
REGISTRY_EPOCH = "epoch-pr590"
REGISTRY_GENERATION = 1


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and 1 <= int(allocated) <= 232
    assert os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def _catalog_documents(skill_names=SKILL_NAMES):
    """Build the canonical preimages and the digests the Gateway would publish."""
    registry_preimage = {
        "schema_version": 1,
        "robot_name": ROBOT_NAME,
        "skills": [
            {
                "name": name,
                "semantic_level": "skill",
                "template": {"schema_version": 1, "timeout_sec": 30.0},
            }
            for name in skill_names
        ],
    }
    capability_view = {
        name: {
            "name": name,
            "schema_version": 1,
            "summary": f"{name} summary",
            "domain": "gesture",
            "moves_robot": True,
            "required_control_mode": "named_motion",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }
        for name in skill_names
    }
    capability_preimage = derive_capability_preimage(
        robot_name=ROBOT_NAME,
        profile_name="pr590_profile",
        capability_view=capability_view,
        enabled_skill_names=skill_names,
        planner_visible_skill_names=skill_names,
        named_pose_names=(),
        timeout_policy={},
    )
    provenance_preimage = {"schema_version": 1, "robot_name": ROBOT_NAME, "source_release": "pr590-test"}
    payload = {
        "schema_version": 1,
        "registry_preimage": registry_preimage,
        "capability_preimage": capability_preimage,
        "provenance_preimage": provenance_preimage,
    }
    return payload, {
        "registry_digest": derive_registry_digest(registry_preimage),
        "capability_digest": derive_capability_digest(capability_preimage),
        "provenance_digest": derive_provenance_digest(provenance_preimage),
    }


class MockGateway:
    """Real ROS node serving only the two read-only Gateway services."""

    def __init__(self, *, suffix: str | None = None, robot_name: str = ROBOT_NAME) -> None:
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor

        from ibrobot_msgs.msg import SkillCapabilityStatus
        from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot

        self._rclpy = rclpy
        self._SkillCapabilityStatus = SkillCapabilityStatus
        self.robot_name = robot_name
        self.skill_names = list(SKILL_NAMES)
        self.status_calls = 0
        self.snapshot_calls = 0
        payload, digests = _catalog_documents(self.skill_names)
        self._snapshot_json = to_canonical_json(payload)
        self._digests = digests
        tag = suffix or uuid.uuid4().hex[:8]
        self.names = {
            "status": f"/pr590_gateway_{tag}/get_status",
            "snapshot": f"/pr590_gateway_{tag}/get_skill_snapshot",
        }
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            f"pr590_mock_gateway_{tag}_{uuid.uuid4().hex[:6]}",
            context=self._context,
            use_global_arguments=False,
        )
        group = ReentrantCallbackGroup()
        self._node.create_service(GetSkillGatewayStatus, self.names["status"], self._status, callback_group=group)
        self._node.create_service(GetSkillSnapshot, self.names["snapshot"], self._snapshot, callback_group=group)
        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name=f"pr590-gateway-{tag}", daemon=True)
        self._thread.start()

    def _status(self, request, response):
        self.status_calls += 1
        response.schema_version = 1
        response.robot_name = self.robot_name
        response.motion_authorized = False
        response.active_control_mode = "named_motion"
        response.busy = False
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        response.default_skill_timeout_sec = 30.0
        response.task_budget_sec = 60.0
        response.rpc_timeout_sec = 5.0
        response.config_digest = self._digests["capability_digest"]
        response.capability_digest = self._digests["capability_digest"]
        response.registry_epoch = REGISTRY_EPOCH
        response.registry_generation = REGISTRY_GENERATION
        response.registry_digest = self._digests["registry_digest"]
        response.source_release_digest = "source-release-pr590"
        response.provenance_digest = self._digests["provenance_digest"]
        response.retained_generations = [REGISTRY_GENERATION]
        response.capabilities = [
            self._SkillCapabilityStatus(
                schema_version=1,
                name=name,
                semantic_level="skill",
                planner_visible=True,
                ready=True,
                reason="",
                required_control_mode="named_motion",
            )
            for name in self.skill_names
        ]
        return response

    def _snapshot(self, request, response):
        self.snapshot_calls += 1
        response.success = True
        response.registry_epoch = REGISTRY_EPOCH
        response.generation = REGISTRY_GENERATION
        response.registry_digest = self._digests["registry_digest"]
        response.capability_digest = self._digests["capability_digest"]
        response.source_release_digest = "source-release-pr590"
        response.provenance_digest = self._digests["provenance_digest"]
        response.profile_name = "pr590_profile"
        response.snapshot_json = self._snapshot_json
        return response

    def stop(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
        self._context.try_shutdown()


def _transport(gateway):
    return gateway_transport(
        {
            "skill_gateway_status_service": gateway.names["status"],
            "skill_catalog_snapshot_service": gateway.names["snapshot"],
        },
        robot_id="g1",
        robot_name=ROBOT_NAME,
    )


@pytest.fixture
def gateway():
    _assert_isolated_ros_domain()
    instance = MockGateway()
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture
def in_process_bridge(monkeypatch):
    """Force the in-process bridge and count starts of the real transport."""
    monkeypatch.setenv("IBROBOT_ROBOT_SKILL_DAEMON", "0")
    from robot_client.transports.resident import ResidentBridge

    starts = []
    original = ResidentBridge.start

    def counted(self):
        starts.append(self)
        return original(self)

    monkeypatch.setattr(ResidentBridge, "start", counted)
    return starts


def test_the_first_status_read_starts_the_default_transport_once(gateway, in_process_bridge):
    client = RobotClient(_transport(gateway))
    try:
        # Binding resolves endpoints only; it must not create a ROS participant.
        assert client._bridge is None
        status = client.get_status()
        assert status["robot_name"] == ROBOT_NAME
        assert status["control_plane_ready"] is True
        assert status["motion_authorized"] is False
        assert status["registry_epoch"] == REGISTRY_EPOCH
        bridge = client._bridge
        assert type(bridge).__name__ == "ResidentBridge"
        # No daemon was used, so the started transport is the in-process bridge.
        assert bridge._local is not None
        assert bridge._connection is None
        assert in_process_bridge == [bridge]
        assert gateway.status_calls == 1

        second = client.get_status()
        assert second["robot_name"] == ROBOT_NAME
        assert in_process_bridge == [bridge]
    finally:
        client.close()


def test_the_first_catalog_read_starts_the_transport_and_lists_skills(gateway, in_process_bridge):
    client = RobotClient(_transport(gateway))
    try:
        listing = client.list_skills()
        assert [skill["name"] for skill in listing["skills"]] == list(SKILL_NAMES)
        assert all(skill["planner_visible"] for skill in listing["skills"])
        assert all(skill["ready"] for skill in listing["skills"])
        assert listing["robot_name"] == ROBOT_NAME
        assert listing["config_digest"] == gateway._digests["capability_digest"]
        assert len(in_process_bridge) == 1
        assert gateway.snapshot_calls == 1
    finally:
        client.close()


def test_a_start_failure_raises_ros_unavailable_with_the_transport_detail(gateway, monkeypatch):
    monkeypatch.setenv("IBROBOT_ROBOT_SKILL_DAEMON", "0")
    from robot_client.transports.resident import ResidentBridge

    starts = []
    original = ResidentBridge.start

    def counted(self):
        starts.append(self)
        return original(self)

    monkeypatch.setattr(ResidentBridge, "start", counted)

    def refuse():
        raise RuntimeError("pr590 injected wire-contract failure")

    # The real RosBridge.start() performs this import and validates the wire
    # contracts before initializing rclpy, so this failure is a real start
    # failure rather than a stubbed transport.
    monkeypatch.setattr("embodied_common.wire_contracts.validate_public_request_wire_contracts", refuse)

    client = RobotClient(_transport(gateway))
    try:
        with pytest.raises(ClientError) as raised:
            client.get_status()
        assert raised.value.code == "ROS_UNAVAILABLE"
        message = str(raised.value)
        assert "pr590 injected wire-contract failure" in message
        assert "ROS bridge is not started" not in message
        # The failed bridge is not kept: the next read is a fresh attempt.
        assert client._bridge is None
        with pytest.raises(ClientError):
            client.get_status()
        assert len(starts) == 2
        assert gateway.status_calls == 0
    finally:
        client.close()


def test_a_daemon_that_fails_to_start_reports_its_error(gateway, tmp_path, monkeypatch):
    from robot_client.transports import resident

    monkeypatch.setenv(resident.DIRECTORY_ENV, str(tmp_path / "daemon"))
    monkeypatch.setenv(resident.DISABLE_ENV, "1")
    path = resident.socket_path()
    daemon = resident.BridgeDaemon(path, idle_timeout_sec=30.0)
    daemon.bind()
    thread = threading.Thread(target=daemon.serve_forever, name="pr590-daemon-start-failure", daemon=True)
    thread.start()

    def refuse():
        raise RuntimeError("pr590 injected daemon start failure")

    monkeypatch.setattr("embodied_common.wire_contracts.validate_public_request_wire_contracts", refuse)
    client = RobotClient(_transport(gateway))
    try:
        with pytest.raises(ClientError) as raised:
            client.get_status()
        assert raised.value.code == "ROS_UNAVAILABLE"
        assert "pr590 injected daemon start failure" in str(raised.value)
        # Not the bare "_forward" symptom of a transport that was never started.
        assert "ROS bridge is not started" not in str(raised.value)
        assert client._bridge is None
        assert gateway.status_calls == 0
    finally:
        client.close()
        daemon.request_stop()
        thread.join(timeout=5.0)


def test_a_present_daemon_serves_the_first_read(gateway, tmp_path, monkeypatch):
    from robot_client.transports import resident

    monkeypatch.setenv(resident.DIRECTORY_ENV, str(tmp_path / "daemon"))
    monkeypatch.setenv(resident.DISABLE_ENV, "1")
    path = resident.socket_path()
    # Real daemon, real RosBridge inside it, real socket protocol; the shared
    # bridge is warmed against the fixture Gateway before the client connects.
    daemon = resident.BridgeDaemon(path, idle_timeout_sec=30.0)
    daemon.bind()
    thread = threading.Thread(target=daemon.serve_forever, name="pr590-daemon", daemon=True)
    thread.start()
    client = RobotClient(_transport(gateway))
    try:
        status = client.get_status()
        assert status["robot_name"] == ROBOT_NAME
        bridge = client._bridge
        assert bridge.resident is True
        assert bridge._connection is not None
        assert bridge._local is None
        listing = client.list_skills()
        assert [skill["name"] for skill in listing["skills"]] == list(SKILL_NAMES)
        assert gateway.status_calls >= 1
    finally:
        client.close()
        daemon.request_stop()
        thread.join(timeout=5.0)


def test_binding_does_not_initialize_ros_or_the_daemon(tmp_path, monkeypatch):
    """Binding (the Web startup path) resolves no socket and starts nothing."""
    from robot_client.transports import resident

    directory = tmp_path / "daemon"
    monkeypatch.setenv(resident.DIRECTORY_ENV, str(directory))
    monkeypatch.setenv(resident.DISABLE_ENV, "spawn")
    client = RobotClient(_transport(SimpleGatewayNames()))
    try:
        assert client._bridge is None
        assert client.session["robot_name"] == ROBOT_NAME
        # A spawn-enabled client would create the runtime directory while
        # starting; binding must not even look at the socket path.
        assert not directory.exists()
    finally:
        client.close()


class SimpleGatewayNames:
    names = {"status": "/pr590_bind_only/get_status", "snapshot": "/pr590_bind_only/get_skill_snapshot"}
