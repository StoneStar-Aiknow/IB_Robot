"""Exercise the real loopback HTTP adapter, not just its helper functions."""

import copy
import http.client
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.cookies import SimpleCookie
from urllib.parse import quote

import pytest

from robot_client.connection import ros_environment_scope
from robot_client.errors import ClientError
from robot_client.transports.ros_bridge import BridgeError
from robot_skill_web.server import WebServer

#: One deployment binding: endpoints plus ROS scope, fixed for the process.
TARGET_KEY = f"{ros_environment_scope()}:mock-binding"


class MockRobot:
    """Stand-in for the config-bound client: one target, no discovery."""

    def __init__(self):
        self.version = 1
        self.submissions = 0
        self.state = "ACTIVE"
        self.found = True
        self.cancel_count = 0
        self.reference = None
        self.calls = []

    @property
    def session(self):
        return {
            "connection_id": "connection-1",
            "connection_version": self.version,
            "robot_id": "mock_g1",
            "robot_name": "mock_g1",
            "caller_scope": "robot_skill_cli",
            "target_key": TARGET_KEY,
            "read_only": False,
        }

    def get_status(self):
        self.calls.append("get_status")
        return {
            "robot_name": "mock_g1",
            "motion_authorized": True,
            "control_plane_ready": True,
            "control_plane_state": "READY",
            "active_control_mode": "mock",
            "busy": self.submissions > 0,
        }

    def describe_skill(self, name):
        assert name == "wave_hand"
        return {
            "name": name,
            "display_name": "挥手",
            "tags": ["preset"],
            "summary": "挥手",
            "schema_version": 1,
            "planner_visible": True,
            "ready": True,
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    def list_skills(self):
        return {"robot_name": "mock_g1", "skills": [self.describe_skill("wave_hand")]}

    def prepare_workflow(self, request):
        self.preview = {
            "preview_id": "preview-1",
            "request_id": "request-1",
            "connection_id": self.session["connection_id"],
            "connection_version": self.version,
            "target": {"robot_name": "mock_g1", "robot_id": "mock_g1"},
            "plan": {
                "steps": request["workflow_steps"],
                "plan_id": "plan-1",
                "plan_digest": "digest-1",
                "registry_identity": {
                    "registry_epoch": "epoch-1",
                    "registry_generation": 1,
                    "registry_digest": "registry-1",
                },
                "proposed_task_budget_sec": 30,
            },
        }
        return copy.deepcopy(self.preview)

    def submit_prepared(self, preview_id, *, presenter, persist_reference):
        assert preview_id == self.preview["preview_id"]
        presenter(copy.deepcopy(self.preview))
        ref = {
            "target_key": TARGET_KEY,
            "robot_id": "mock_g1",
            "caller_scope": "robot_skill_cli",
            "request_id": "request-1",
            "execution_id": "",
            "legacy_task_id": "",
        }
        persist_reference(ref)
        self.submissions += 1
        self.reference = ref
        time.sleep(0.05)
        return ref

    def get_execution(self, reference):
        return {"found": self.found, "state": self.state, "execution_id": "execution-1", "message": "mock ledger"}

    def get_execution_events(self, reference, after_sequence=0):
        return {"events": [], "last_event_sequence": 0, "found": True}

    def cancel_execution(self, reference):
        self.cancel_count += 1
        self.state = "CANCELED"
        return {"accepted": True, "state": "STOPPING"}

    def close(self):
        pass


class Browser:
    def __init__(self, port):
        self.port = port
        self.cookies = {}

    def request(self, method, path, body=None, headers=None):
        defaults = {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": f"http://127.0.0.1:{self.port}",
            "Cookie": "; ".join(f"{key}={value}" for key, value in self.cookies.items()),
            "X-CSRF-Token": self.cookies.get("_robot_skill_web_csrf", ""),
        }
        defaults.update(headers or {})
        if body is not None:
            defaults["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request(method, path, None if body is None else json.dumps(body), defaults)
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

    def preview(self, version=1):
        status, result = self.request(
            "POST",
            "/api/previews",
            {"skill_name": "wave_hand", "parameters": {}, "raw_command": "mock wave", "connection_version": version},
        )
        assert status == 201, result
        return result


def wait_for_submission(server, preview_id):
    """Wait on the worker's own completion condition instead of a fixed sleep."""
    worker = server.state.submission_workers.get(preview_id)
    if worker is None:
        return
    with worker["ready"]:
        assert worker["ready"].wait_for(lambda: worker["done"], timeout=5), "submission worker did not finish"


@pytest.fixture
def web(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    robot = MockRobot()
    server = WebServer(client=robot, port=port, submission_directory=tmp_path / "state")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while server.httpd is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.httpd is not None
    yield Browser(port), robot, server
    server.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("path", ["/", "/index.html", "/app.js", "/styles.css"])
def test_static_assets_do_not_reuse_cached_ui(web, method, path):
    browser, robot, _server = web
    connection = http.client.HTTPConnection("127.0.0.1", browser.port, timeout=10)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        response.read()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
    finally:
        connection.close()
    assert robot.calls == []
    assert robot.submissions == 0


def test_http_passes_vendor_action_tags_through_unchanged(web):
    browser, robot, _server = web
    browser.connect()

    tagged = {
        "name": "wave_hand",
        "display_name": "挥手",
        "tags": ["preset", "control_ownership", "unitree_action_id_99"],
        "summary": "挥手",
        "planner_visible": True,
        "ready": True,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    robot.describe_skill = lambda name: dict(tagged, name=name)
    robot.list_skills = lambda: {"robot_name": "mock_g1", "skills": [dict(tagged)]}

    status, listing = browser.request("GET", "/api/skills")
    assert status == 200
    status, detail = browser.request("GET", "/api/skills/wave_hand")
    assert status == 200
    for skill in (listing["skills"][0], detail):
        assert skill["tags"] == ["preset", "control_ownership", "unitree_action_id_99"]
        assert "action_id" not in skill
        assert "vendor" not in skill
    assert robot.submissions == 0


def test_http_preserves_chinese_labels_without_changing_execution_id(web):
    browser, robot, _server = web
    browser.connect()
    status, listing = browser.request("GET", "/api/skills")
    assert status == 200
    status, detail = browser.request("GET", "/api/skills/wave_hand")
    assert status == 200
    for skill in (listing["skills"][0], detail):
        assert skill["name"] == "wave_hand"
        assert skill["display_name"] == "挥手"
        assert skill["summary"] == "挥手"
        assert skill["tags"] == ["preset"]
    preview = browser.preview()
    assert preview["plan"]["steps"][0]["skill_name"] == "wave_hand"
    assert robot.submissions == 0


def test_http_session_and_request_origin(web):
    browser, robot, _server = web
    assert browser.request("GET", "/api/session")[0] == 200
    assert browser.request("GET", "/api/session", headers={"Host": "evil.example"})[0] == 403
    assert browser.request("POST", "/api/session", {}, headers={"Origin": "https://evil.example"})[0] == 403
    assert browser.request("POST", "/api/session", {}, headers={"X-CSRF-Token": "wrong"})[0] == 403
    browser.cookies["_robot_skill_web_session"] = "forged"
    assert browser.request("GET", "/api/session")[1]["session_active"] is False
    assert browser.request("GET", "/api/status")[0] == 401
    assert robot.calls == []


@pytest.mark.parametrize("body", [{"startup_code": "code"}, {"deployment_key": "x"}, {"caller_token": "x"}])
def test_http_rejects_removed_login_contract(web, body):
    browser, robot, _server = web
    assert browser.request("GET", "/api/session")[0] == 200
    assert browser.request("POST", "/api/session", body)[0] == 400
    assert browser.request("GET", "/api/session")[1]["session_active"] is False
    assert robot.calls == []
    assert robot.submissions == 0


def test_session_exposes_only_the_bound_public_identity(web):
    browser, robot, _server = web
    before = browser.request("GET", "/api/session")[1]
    assert before == {"session_active": False, "connection": None, "unresolved_executions": []}
    connection = browser.connect()
    assert set(connection) == {
        "connection_id",
        "connection_version",
        "robot_id",
        "robot_name",
        "caller_scope",
        "target_key",
        "read_only",
    }
    assert "endpoints" not in connection and "transport" not in connection
    assert connection["target_key"] == TARGET_KEY
    # Establishing the session reads the local binding only; nothing is sent.
    assert robot.calls == []
    assert robot.submissions == 0


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/api/status"),
        ("GET", "/api/skills"),
        ("GET", "/api/executions/private"),
        ("POST", "/api/previews"),
        ("POST", "/api/executions"),
        ("POST", "/api/executions/private/cancel"),
    ],
)
def test_session_cookie_is_required_for_robot_and_task_access(web, method, path):
    browser, robot, _server = web
    browser.request("GET", "/api/session")
    assert browser.request(method, path, {} if method == "POST" else None)[0] in {401, 403, 404}
    assert robot.calls == []
    assert robot.submissions == robot.cancel_count == 0


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/api/robots"),
        ("POST", "/api/connections"),
        ("DELETE", "/api/connections/connection-1"),
    ],
)
def test_discovery_and_selection_endpoints_are_removed(web, method, path):
    browser, _robot, _server = web
    browser.connect()
    assert browser.request(method, path, {} if method != "GET" else None)[0] == 404


def test_unresolved_task_from_another_deployment_blocks_session(web, monkeypatch):
    browser, _robot, server = web
    monkeypatch.setattr(
        server.state.submissions,
        "unresolved",
        lambda: [{"reference": {"target_key": "other-scope:other-binding", "request_id": "old"}}],
    )
    browser.request("GET", "/api/session")
    status, error = browser.request("POST", "/api/session", {})
    assert status == 409
    assert "another deployment" in error["error"]["message"]
    assert browser.request("GET", "/api/session")[1]["connection"] is None
    for _ in range(4):
        assert browser.request("POST", "/api/session", {})[0] == 409
    assert browser.request("POST", "/api/session", {})[0] == 429


def test_reestablishing_a_session_revokes_the_previous_browser(web):
    browser, robot, _server = web
    original = browser.connect()
    replacement = Browser(browser.port)
    assert replacement.connect() == original
    assert browser.request("GET", "/api/status")[0] == 401
    assert replacement.request("GET", "/api/status")[0] == 200
    assert robot.submissions == 0


def test_session_expiry_allows_reestablishment_without_restart(web):
    browser, robot, server = web
    now = [0.0]
    server.state.sessions._now = lambda: now[0]
    original = browser.connect()
    token = browser.cookies["_robot_skill_web_session"]
    now[0] = server.state.sessions.lookup(token).expires_at
    assert browser.request("GET", "/api/status")[0] == 401
    assert browser.request("GET", "/api/session")[1]["session_active"] is False
    assert browser.connect() == original
    assert browser.cookies["_robot_skill_web_session"] != token
    assert robot.submissions == 0


def test_http_duplicate_submit_reuses_reference_and_keeps_session(web):
    browser, robot, server = web
    session = browser.connect()
    assert browser.request("GET", "/api/status")[1]["robot_name"] == "mock_g1"
    prepared = browser.preview()

    def submit(_index):
        isolated_browser = Browser(browser.port)
        isolated_browser.cookies = dict(browser.cookies)
        return isolated_browser.request(
            "POST",
            "/api/executions",
            {
                "preview_id": prepared["preview_id"],
                "connection_version": session["connection_version"],
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(submit, (1, 2)))
    assert [status for status, _body in replies] == [202, 202], replies
    assert replies[0][1]["execution"] == replies[1][1]["execution"]
    ref = replies[0][1]["execution"]
    assert ref["target_key"]
    wait_for_submission(server, prepared["preview_id"])
    restored = browser.request("GET", "/api/session")[1]
    assert restored["session_active"] is True
    assert restored["connection"] == session
    assert restored["unresolved_executions"][0]["reference"] == ref
    assert robot.submissions == 1
    assert server.state.submissions.get(prepared["preview_id"])["reference"] == ref
    assert browser.request("POST", "/api/session", {"candidate_id": "candidate-1"})[0] == 400
    execution_path = "/api/executions/" + quote(json.dumps(ref), safe="")
    assert browser.request("GET", execution_path)[0] == 200
    assert browser.request("POST", execution_path + "/cancel", {"connection_version": 99})[0] == 409
    assert robot.cancel_count == 0
    assert browser.request("POST", execution_path + "/cancel", {"connection_version": 1})[0] == 202
    assert browser.request("GET", execution_path)[1]["execution"]["state"] == "CANCELED"
    assert browser.request("GET", "/api/session")[1]["session_active"] is True


def test_http_stale_version_never_prepares_or_submits(web):
    browser, robot, _server = web
    browser.connect()
    status, _result = browser.request(
        "POST",
        "/api/previews",
        {"skill_name": "wave_hand", "parameters": {}, "raw_command": "mock", "connection_version": 99},
    )
    assert status == 409
    assert robot.submissions == 0
    assert browser.request("GET", "/api/status")[0] == 200


def test_invalid_session_cookie_does_not_restore_private_state(web):
    browser, _robot, _server = web
    browser.cookies["_robot_skill_web_session"] = "expired-or-forged"
    status, result = browser.request("GET", "/api/session")
    assert status == 200
    assert result == {"session_active": False, "connection": None, "unresolved_executions": []}


@pytest.mark.parametrize(
    "error",
    [ClientError("INVALID_RESPONSE", "invalid event response"), BridgeError("ROS_UNAVAILABLE", "offline", exit_code=4)],
)
def test_event_query_error_preserves_terminal_result_without_claiming_a_gap(web, monkeypatch, error):
    browser, robot, _server = web
    browser.connect()
    robot.state = "FAILED"
    reference = {
        "request_id": "request-1",
        "execution_id": "execution-1",
        "target_key": TARGET_KEY,
        "robot_id": "mock_g1",
        "caller_scope": "robot_skill_cli",
        "legacy_task_id": "",
    }

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(robot, "get_execution_events", unavailable)
    status, result = browser.request("GET", "/api/executions/" + quote(json.dumps(reference), safe=""))

    assert status == 200
    assert result["execution"]["state"] == "FAILED"
    assert result["events"]["unavailable"] is True
    assert result["events"]["error_code"] == error.code
    assert not result["events"].get("event_gap", False)
    assert robot.submissions == 0
    assert robot.cancel_count == 0


def test_restart_restores_original_target_without_resending(web):
    browser, robot, server = web
    browser.connect()
    prepared = browser.preview()
    status, result = browser.request(
        "POST", "/api/executions", {"preview_id": prepared["preview_id"], "connection_version": 1}
    )
    assert status == 202
    from robot_skill_web.serialization import SubmissionStore
    from robot_skill_web.server import WebState

    # Replace backend process state while preserving the same mock Gateway ledger.
    replacement = WebState(client=robot, submission_directory=server.state.submissions.directory)
    server.httpd.web_state = replacement
    server.state = replacement
    assert SubmissionStore(replacement.submissions.directory).unresolved()
    browser.cookies = {}
    restored = browser.request("GET", "/api/session")[1]
    assert restored["connection"] is None
    assert restored["unresolved_executions"] == []
    # The new process is bound to the same startup deployment; no robot is touched.
    session = browser.connect()
    assert robot.submissions == 1
    restored = browser.request("GET", "/api/session")[1]
    assert restored["connection"] == session
    assert restored["unresolved_executions"][0]["reference"] == result["execution"]
    path = "/api/executions/" + quote(json.dumps(result["execution"]), safe="")
    assert browser.request("GET", path)[1]["execution"]["state"] == "ACTIVE"
    assert robot.submissions == 1
    assert browser.request("POST", path + "/cancel", {"connection_version": session["connection_version"]})[0] == 202


def test_a_gateway_transport_error_keeps_its_code_and_is_retryable(web, monkeypatch):
    """A started bridge that loses the Gateway is a transport state, not a 500."""
    browser, robot, _server = web
    from robot_client.stop_contract import EXIT_ROS_UNAVAILABLE

    def offline():
        raise BridgeError(
            "SERVER_UNAVAILABLE",
            "service unavailable: /embodied/get_skill_gateway_status",
            exit_code=EXIT_ROS_UNAVAILABLE,
        )

    monkeypatch.setattr(robot, "get_status", offline)
    browser.connect()
    status, body = browser.request("GET", "/api/status")
    assert status == 503
    assert body["error"]["code"] == "SERVER_UNAVAILABLE"
    assert "service unavailable" in body["error"]["message"]
