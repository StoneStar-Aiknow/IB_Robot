"""The Web adapter reads a bound Gateway through its real transport.

Regression: binding the robot configuration no longer started the transport, so
the first ``GET /api/status`` and ``GET /api/skills`` of a fresh server answered
``500 internal adapter error`` while the Gateway was online and READY. The CLI
kept starting its own bridge, which is why only the Web workbench was red.

These tests drive the real loopback HTTP adapter with the real config binding
and the real resident transport against a minimal Gateway fixture on an
isolated ROS domain. The fixture serves only the two read-only services the
first read needs, so no plan, action or motion is exercised. They cover:

* the homepage and the browser session never start ROS;
* ``GET /api/status`` and ``GET /api/skills`` genuinely read the fixture
  Gateway, once, and return its values;
* a start failure is a ``503`` with ``ROS_UNAVAILABLE`` and the transport's own
  diagnostic, not an internal error.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
import uuid
from http.cookies import SimpleCookie

import pytest

from embodied_common.catalog_digest import (
    derive_capability_digest,
    derive_capability_preimage,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)
from robot_skill_web.server import WebServer, bound_client

#: The repository deployment the Web server binds by name.
ROBOT_NAME = "unitree_g1_arm_presets"
SKILL_NAMES = ("arms_horizontal", "wave_hello")
REGISTRY_EPOCH = "epoch-pr590-web"
REGISTRY_GENERATION = 1


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and 1 <= int(allocated) <= 232
    assert os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def _catalog_documents():
    registry_preimage = {
        "schema_version": 1,
        "robot_name": ROBOT_NAME,
        "skills": [
            {"name": name, "semantic_level": "skill", "template": {"schema_version": 1, "timeout_sec": 30.0}}
            for name in SKILL_NAMES
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
        for name in SKILL_NAMES
    }
    capability_preimage = derive_capability_preimage(
        robot_name=ROBOT_NAME,
        profile_name="pr590_web_profile",
        capability_view=capability_view,
        enabled_skill_names=SKILL_NAMES,
        planner_visible_skill_names=SKILL_NAMES,
        named_pose_names=(),
        timeout_policy={},
    )
    provenance_preimage = {"schema_version": 1, "robot_name": ROBOT_NAME, "source_release": "pr590-web-test"}
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
    """Real ROS node serving the deployed endpoint names, read-only services only."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor

        from ibrobot_msgs.msg import SkillCapabilityStatus
        from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot

        self._SkillCapabilityStatus = SkillCapabilityStatus
        self.status_calls = 0
        self.snapshot_calls = 0
        payload, self.digests = _catalog_documents()
        self._snapshot_json = to_canonical_json(payload)
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            f"pr590_web_gateway_{uuid.uuid4().hex[:8]}",
            context=self._context,
            use_global_arguments=False,
        )
        group = ReentrantCallbackGroup()
        # The deployed default endpoints, the same ones bound_client resolves.
        self._node.create_service(
            GetSkillGatewayStatus, "/embodied/get_skill_gateway_status", self._status, callback_group=group
        )
        self._node.create_service(
            GetSkillSnapshot, "/embodied/get_skill_snapshot", self._snapshot, callback_group=group
        )
        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name="pr590-web-gateway", daemon=True)
        self._thread.start()

    def _status(self, request, response):
        self.status_calls += 1
        response.schema_version = 1
        response.robot_name = ROBOT_NAME
        response.motion_authorized = False
        response.active_control_mode = "named_motion"
        response.busy = False
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        response.default_skill_timeout_sec = 30.0
        response.task_budget_sec = 60.0
        response.rpc_timeout_sec = 5.0
        response.config_digest = self.digests["capability_digest"]
        response.capability_digest = self.digests["capability_digest"]
        response.registry_epoch = REGISTRY_EPOCH
        response.registry_generation = REGISTRY_GENERATION
        response.registry_digest = self.digests["registry_digest"]
        response.source_release_digest = "source-release-pr590"
        response.provenance_digest = self.digests["provenance_digest"]
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
            for name in SKILL_NAMES
        ]
        return response

    def _snapshot(self, request, response):
        self.snapshot_calls += 1
        response.success = True
        response.registry_epoch = REGISTRY_EPOCH
        response.generation = REGISTRY_GENERATION
        response.registry_digest = self.digests["registry_digest"]
        response.capability_digest = self.digests["capability_digest"]
        response.source_release_digest = "source-release-pr590"
        response.provenance_digest = self.digests["provenance_digest"]
        response.profile_name = "pr590_web_profile"
        response.snapshot_json = self._snapshot_json
        return response

    def stop(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
        self._context.try_shutdown()


class Browser:
    """One loopback browser session with cookies and the CSRF header."""

    def __init__(self, port) -> None:
        self.port = port
        self.cookies = {}

    def request(self, method, path, body=None):
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": f"http://127.0.0.1:{self.port}",
            "Cookie": "; ".join(f"{key}={value}" for key, value in self.cookies.items()),
            "X-CSRF-Token": self.cookies.get("_robot_skill_web_csrf", ""),
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        connection.request(method, path, None if body is None else json.dumps(body), headers)
        response = connection.getresponse()
        content = response.read()
        for key, value in response.getheaders():
            if key.lower() == "set-cookie":
                cookie = SimpleCookie()
                cookie.load(value)
                self.cookies.update({name: item.value for name, item in cookie.items()})
        status = response.status
        connection.close()
        try:
            return status, json.loads(content)
        except ValueError:
            return status, content.decode()

    def connect(self):
        assert self.request("GET", "/api/session")[0] == 200
        status, payload = self.request("POST", "/api/session", {})
        assert status == 201, payload
        return payload["connection"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(client, tmp_path):
    server = WebServer(client=client, port=_free_port(), submission_directory=tmp_path / "state")
    thread = threading.Thread(target=server.serve_forever, name="pr590-web", daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while server.httpd is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.httpd is not None
    return server, thread


@pytest.fixture
def live_web(tmp_path, monkeypatch):
    _assert_isolated_ros_domain()
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    monkeypatch.setenv("IBROBOT_ROBOT_SKILL_DAEMON", "0")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    gateway = MockGateway()
    client = bound_client(config_name=ROBOT_NAME)
    assert client._bridge is None  # binding is local only
    server, thread = _serve(client, tmp_path)
    try:
        yield Browser(server.httpd.server_port), client, gateway
    finally:
        server.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        gateway.stop()


def test_homepage_and_session_never_start_ros(live_web):
    browser, client, gateway = live_web
    status, body = browser.request("GET", "/")
    assert status == 200 and "IB-Robot" in body
    browser.connect()
    # Binding, serving the homepage and establishing the browser session must
    # not initialize ROS or contact the Gateway.
    assert client._bridge is None
    assert gateway.status_calls == 0
    assert gateway.snapshot_calls == 0


def test_status_and_skills_read_the_bound_gateway(live_web):
    browser, client, gateway = live_web
    browser.connect()

    status, payload = browser.request("GET", "/api/status")
    assert status == 200, payload
    assert payload["robot_name"] == ROBOT_NAME
    assert payload["control_plane_ready"] is True
    assert payload["control_plane_state"] == "READY"
    assert payload["motion_authorized"] is False
    assert payload["registry_epoch"] == REGISTRY_EPOCH
    bridge = client._bridge
    assert bridge is not None

    status, listing = browser.request("GET", "/api/skills")
    assert status == 200, listing
    assert listing["robot_name"] == ROBOT_NAME
    assert [skill["name"] for skill in listing["skills"]] == list(SKILL_NAMES)
    assert all(skill["ready"] for skill in listing["skills"])
    # The second read reuses the started transport instead of starting another.
    assert client._bridge is bridge
    assert gateway.status_calls >= 1
    assert gateway.snapshot_calls == 1


def test_a_start_failure_is_a_diagnosable_ros_unavailable(tmp_path, monkeypatch):
    """The reported symptom (500 internal adapter error) becomes a stable code."""
    _assert_isolated_ros_domain()
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    monkeypatch.setenv("IBROBOT_ROBOT_SKILL_DAEMON", "0")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))

    def refuse():
        raise RuntimeError("pr590 injected web start failure")

    monkeypatch.setattr("embodied_common.wire_contracts.validate_public_request_wire_contracts", refuse)
    client = bound_client(config_name=ROBOT_NAME)
    server, thread = _serve(client, tmp_path)
    browser = Browser(server.httpd.server_port)
    try:
        # The browser session is still established: it does not touch ROS.
        browser.connect()
        status, body = browser.request("GET", "/api/status")
        assert status == 503, body
        assert body["error"]["code"] == "ROS_UNAVAILABLE"
        assert "pr590 injected web start failure" in body["error"]["message"]
        assert body["error"]["message"] != "internal adapter error"
        status, body = browser.request("GET", "/api/skills")
        assert status == 503, body
        assert body["error"]["code"] == "ROS_UNAVAILABLE"
        assert "pr590 injected web start failure" in body["error"]["message"]
        assert client._bridge is None
    finally:
        server.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
