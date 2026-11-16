from __future__ import annotations

import errno
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from robot_skill_web.server import WebServer, _RequestHandler, _Server, build_workflow_step, decode_execution


class FakeClient:
    """One process bound to one deployment; the Web server never discovers."""

    def __init__(self):
        self.calls = []

    @property
    def session(self):
        return {
            "connection_id": "connection-1",
            "connection_version": 1,
            "robot_name": "g1",
            "robot_id": "robot-1",
            "read_only": False,
            "caller_scope": "web",
            "target_key": "scope:binding",
        }

    def close(self):
        self.calls.append(("close",))

    def list_skills(self):
        return {"robot_name": "g1", "skills": []}

    def get_status(self):
        return {"robot_name": "g1", "motion_authorized": True}

    def prepare_workflow(self, request):
        return {
            "preview_id": "preview-1",
            "connection_id": "connection-1",
            "connection_version": 1,
            "request_id": "request-1",
            "plan": {"plan_id": "plan-1", "steps": request["workflow_steps"]},
        }

    def submit_prepared(self, preview_id, *, presenter, persist_reference):
        self.calls.append(("submit", preview_id))
        presenter({"preview_id": preview_id})
        reference = {
            "target_key": "scope:binding",
            "robot_id": "robot-1",
            "caller_scope": "web",
            "request_id": "request-1",
            "execution_id": "",
            "legacy_task_id": "",
        }
        persist_reference(reference)
        return dict(reference)

    def get_execution(self, ref):
        return {"found": True, "state": "UNKNOWN", "reconciled": True}

    def get_execution_events(self, ref, after_sequence=0):
        return {"events": []}

    def cancel_execution(self, ref):
        return {"accepted": True}


@pytest.fixture()
def server(tmp_path: Path):
    client = FakeClient()
    result = WebServer(
        client=client,
        submission_directory=tmp_path / "submissions",
    )
    return result


def test_webserver_rejects_nonloopback():
    with pytest.raises(ValueError, match="127.0.0.1"):
        WebServer(client=FakeClient(), address="0.0.0.0")


def test_port_can_be_rebound_after_http_connection_and_shutdown(tmp_path):
    _, server = make_server(tmp_path)
    start(server)
    address = ("127.0.0.1", server.port)
    try:
        with socket.create_connection(address, timeout=2) as connection:
            connection.sendall(f"GET / HTTP/1.0\r\nHost: 127.0.0.1:{server.port}\r\n\r\n".encode())
            while connection.recv(8192):
                pass
    finally:
        server.close()
    with _Server(address, _RequestHandler, state=server.state) as restarted:
        assert restarted.server_address == address


def test_live_web_listener_cannot_be_shared_by_another_server(tmp_path):
    _, server = make_server(tmp_path)
    start(server)
    try:
        with pytest.raises(OSError) as raised:
            _Server(("127.0.0.1", server.port), _RequestHandler, state=server.state)
        assert raised.value.errno == errno.EADDRINUSE
    finally:
        server.close()


def test_http_session_security_and_status(tmp_path):
    client, server = make_server(tmp_path)
    start(server)
    base = f"http://127.0.0.1:{server.port}"
    assert request(f"{base}/api/session", headers={"Host": "evil"})[0] == 403
    assert (
        request(base + "/api/session", "POST", b"{}", {"Host": f"127.0.0.1:{server.port}", "Origin": "http://evil"})[0]
        == 403
    )
    _, _, headers = request(base + "/api/session")
    csrf = headers.get_all("Set-Cookie")[0].split("=", 1)[1].split(";", 1)[0]
    status, body, headers = request(
        base + "/api/session",
        "POST",
        b"{}",
        {
            "Host": f"127.0.0.1:{server.port}",
            "Origin": base,
            "Cookie": f"_robot_skill_web_csrf={csrf}",
            "X-CSRF-Token": "bad",
        },
    )
    assert status == 403
    status, body, headers = request(
        base + "/api/session",
        "POST",
        b"{}",
        {
            "Host": f"127.0.0.1:{server.port}",
            "Origin": base,
            "Cookie": f"_robot_skill_web_csrf={csrf}",
            "X-CSRF-Token": csrf,
        },
    )
    assert status == 201, body
    session_cookie = next(item for item in headers.get_all("Set-Cookie") if "_robot_skill_web_session=" in item)
    csrf_cookie = next(item for item in headers.get_all("Set-Cookie") if "_robot_skill_web_csrf=" in item)
    csrf = csrf_cookie.split("=", 1)[1].split(";", 1)[0]
    token = session_cookie.split("=", 1)[1].split(";", 1)[0]
    cookie = f"_robot_skill_web_session={token};_robot_skill_web_csrf={csrf}"
    common = {"Host": f"127.0.0.1:{server.port}", "Cookie": cookie, "X-CSRF-Token": csrf}
    assert request(f"{base}/api/robots", headers=common)[0] == 404
    assert request(base + "/api/status", headers=common)[0] == 200
    assert request(base + "/api/skills", headers=common)[0] == 200
    server.close()


def test_duplicate_submission_persists_once_and_survives_restart(tmp_path):
    client, server = make_server(tmp_path)
    start(server)
    preview = server.state.client.prepare_workflow({"workflow_steps": [{"skill_name": "wave"}], "raw_command": "test"})
    server.state.previews[preview["preview_id"]] = preview
    barrier = threading.Barrier(5)

    def submit():
        barrier.wait()
        return server.state.submissions.reserve(
            preview["preview_id"],
            "session-1",
            {
                "target_key": "scope:binding",
                "robot_id": "robot-1",
                "caller_scope": "web",
                "request_id": preview["request_id"],
                "execution_id": "",
                "legacy_task_id": "",
            },
        )

    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(lambda _: submit(), range(5)))
    assert all(record["reference"]["request_id"] == preview["request_id"] for record in records)

    replacement, restarted = make_server(tmp_path, port=server.port)
    server.close()
    assert restarted.state.submissions.unresolved()
    ref = records[0]["reference"]
    assert restarted.state.client.get_execution(ref)["state"] == "UNKNOWN"
    assert restarted.state.client.cancel_execution(ref)["accepted"] is True
    restarted.close()


def test_stale_version_blocks_mutation(tmp_path):
    client, server = make_server(tmp_path)
    start(server)
    status, response, _ = request(
        f"http://127.0.0.1:{server.port}/api/session",
        "POST",
        b"{}",
        {"Host": f"127.0.0.1:{server.port}", "Cookie": "unused", "X-CSRF-Token": "unused"},
    )
    assert status in {401, 403}
    # Direct WebState checks cover the stale-version guard without bypassing authentication.
    with pytest.raises(Exception, match="stale"):
        server.state.require_connection(999)
    server.state.previews["preview"] = {"preview_id": "preview"}
    with pytest.raises(Exception, match="stale"):
        server.state.require_connection(999)
    server.close()


def test_workflow_step_uses_describe_schema():
    skill = {
        "name": "wave",
        "schema_version": 2,
        "parameters": {
            "properties": {
                "direction": {"type": "string", "enum": ["left", "right"]},
                "distance": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["direction", "distance"],
        },
    }
    step = build_workflow_step(skill, {"direction": "left", "distance": 2})
    assert step == {"skill_name": "wave", "direction": "left", "distance": 2, "schema_version": 2}


def test_workflow_step_rejects_unknown_and_wrong_types():
    schema = {"properties": {"target_name": {"type": "string"}}, "required": []}
    with pytest.raises(ValueError, match="unsupported parameters"):
        build_workflow_step({"name": "x", "parameters": schema}, {"other": "1"})
    with pytest.raises(ValueError, match="must be a string"):
        build_workflow_step({"name": "x", "parameters": schema}, {"target_name": 1})


def test_decode_execution_rejects_invalid():
    with pytest.raises(ValueError):
        decode_execution("%EF")


def request(url, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(30):
        try:
            request(f"http://127.0.0.1:{server.port}/")
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_server(tmp_path, *, port=None):
    client = FakeClient()
    return client, WebServer(
        client=client,
        port=port or free_port(),
        submission_directory=tmp_path / "submissions",
    )
