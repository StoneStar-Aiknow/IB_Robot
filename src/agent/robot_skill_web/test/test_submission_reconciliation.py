"""Reconcile a durable Web submission with the Gateway ledger, read only.

The Web adapter scenarios use a deterministic in-process Gateway/ledger double.
The admission-boundary scenarios additionally drive the real
:class:`robot_client.client.RobotClient` through a simulated transport bridge,
so the Web record and the client's own ``_unresolved`` lock are checked
together instead of reimplementing the client contract in a fake. No ROS
graph, robot, Gateway process, install tree or virtual environment is involved:
the private domain is pinned only to keep the environment deterministic.

The contract under test is the Web layer's own admission record:

* a goal is recorded before it can be sent, so a lost receipt is observable
  after the response was lost or the process restarted;
* only a read-only ``GET /api/executions`` may reconcile an association or a
  terminal result; a POST never queries, replays, or clears an unknown goal;
* accepted/running is still busy, only a terminal ledger state unlocks;
* conflicting references and late/stale observations never overwrite or
  unlock a conservative pending record.
"""

from __future__ import annotations

import copy
import http.client
import json
import socket
import threading
import time
import uuid
from http.cookies import SimpleCookie
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from robot_client.errors import ClientError
from robot_client.transports.ros_bridge import BridgeError
from robot_skill_web import server as server_module
from robot_skill_web.server import WebServer

TARGET_KEY = "private-domain-231:fake-binding"
REQUEST_ID = "request-reconcile-1"
EXECUTION_ID = "execution-reconcile-1"


class FakeGateway:
    """Deterministic Gateway and durable ledger double shared by restarts."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.submit_calls = 0
        self.cancel_calls = 0
        self.reference: dict[str, str] | None = None
        # submit behaviours: ``accepted``, ``lost``, ``pre_failure``, ``rejected``,
        # ``blocked`` (waits for ``release`` before its pre-goal failure).
        self.submit_behavior = "accepted"
        self.release = threading.Event()
        self.execution_found = True
        self.execution_state = "ACTIVE"
        self.execution_id = EXECUTION_ID
        self.execution_reconciled = False
        self.execution_error: Exception | None = None
        self.events: list[dict[str, object]] = []
        self.events_error: Exception | None = None
        self.events_found = True
        self.get_execution_calls = 0
        self.get_events_calls = 0
        self.request_ids: dict[str, str] = {}
        self.restored: list[dict[str, str]] = []

    @property
    def session(self) -> dict[str, object]:
        return {
            "connection_id": "connection-reconcile",
            "connection_version": 1,
            "robot_id": "mock_g1",
            "robot_name": "mock_g1",
            "caller_scope": "robot_skill_cli",
            "target_key": TARGET_KEY,
            "read_only": False,
        }

    def _reference(self, preview_id: str) -> dict[str, str]:
        return {
            "target_key": TARGET_KEY,
            "robot_id": "mock_g1",
            "caller_scope": "robot_skill_cli",
            "request_id": self.request_ids.get(preview_id, REQUEST_ID),
            "execution_id": "",
            "legacy_task_id": "",
        }

    def submit_prepared(self, preview_id, *, presenter, persist_reference):
        with self.lock:
            self.submit_calls += 1
            behavior = self.submit_behavior
        assert preview_id
        presenter({"preview_id": preview_id})
        ref = self._reference(preview_id)
        if behavior == "pre_failure":
            raise ClientError("SERVER_UNAVAILABLE", "Program action is not ready")
        if behavior == "blocked":
            # Simulate a pre-goal phase that outlives the HTTP response window.
            self.release.wait(timeout=5.0)
            raise ClientError("SERVER_UNAVAILABLE", "Program action is not ready")
        persist_reference(copy.deepcopy(ref))
        with self.lock:
            self.reference = copy.deepcopy(ref)
        if behavior == "rejected":
            raise ClientError("GOAL_REJECTED", "Gateway rejected the request", reference=ref)
        if behavior == "lost":
            # The ledger accepted the goal but the receipt response was lost.
            with self.lock:
                self.execution_found = False
                self.execution_state = "ACCEPTED"
            raise ClientError(
                "OBSERVATION_UNKNOWN",
                "goal receipt was not observed",
                reference=ref,
                observation_unknown=True,
            )
        return copy.deepcopy(ref)

    def restore_ref(self, reference):
        with self.lock:
            restored = copy.deepcopy(reference)
            self.restored.append(restored)
        return restored

    def get_execution(self, reference):
        with self.lock:
            self.get_execution_calls += 1
            error = self.execution_error
            found = self.execution_found
            state = self.execution_state
            execution_id = self.execution_id
            reconciled = self.execution_reconciled
        if error is not None:
            raise error
        if not found:
            return {"found": False, "error_code": "EXECUTION_NOT_FOUND", "observation_unknown": True}
        return {
            "found": True,
            "state": state,
            "execution_id": execution_id,
            "reconciled": reconciled,
            "message": "fake ledger",
        }

    def get_execution_events(self, reference, after_sequence=0):
        with self.lock:
            self.get_events_calls += 1
            error = self.events_error
            events = copy.deepcopy(self.events)
            found = self.events_found
            execution_id = self.execution_id
        if error is not None:
            raise error
        visible = [event for event in events if int(event.get("sequence", 0)) > after_sequence]
        return {
            "found": found,
            "execution_id": execution_id,
            "events": visible,
            "last_event_sequence": max((int(event["sequence"]) for event in events), default=0),
            "has_more": False,
        }

    def cancel_execution(self, reference):
        with self.lock:
            self.cancel_calls += 1
        return {"accepted": True, "state": "STOPPING"}

    def add_event(self, sequence: int, state: str, *, execution_id: str = EXECUTION_ID) -> None:
        with self.lock:
            self.events.append(
                {
                    "sequence": sequence,
                    "execution_id": execution_id,
                    "state": state,
                    "event_type": state,
                    "message": f"fake {state}",
                }
            )

    def close(self) -> None:
        pass


class ClientBridge:
    """Minimal transport bridge for the real RobotClient admission contract."""

    def __init__(self) -> None:
        self.started = 0
        self.goals: list[dict[str, object]] = []
        self.execution = {"found": True, "state": "RUNNING", "execution_id": "execution-real"}
        self.events = {"found": True, "events": [], "last_event_sequence": 0}
        self.execution_error = None
        self.events_error = None

    def start(self):
        self.started += 1
        return True

    def close(self):
        pass

    def get_status(self, **_kwargs):
        return {
            "robot_name": "g1",
            "registry_epoch": "epoch-real",
            "registry_generation": 1,
            "registry_digest": "registry-real",
            "capability_digest": "a" * 64,
            "control_plane_ready": True,
        }

    def wait_for_program_server(self, **_kwargs):
        return True

    def send_program_goal(self, **kwargs):
        self.goals.append(copy.deepcopy(kwargs))
        raise AssertionError("recovery must never send a second goal")

    def get_execution(self, *, request_id, execution_id, timeout_sec):
        assert request_id
        assert timeout_sec > 0
        if self.execution_error is not None:
            raise self.execution_error
        return copy.deepcopy(self.execution)

    def get_execution_events(self, *, request_id, execution_id="", after_sequence=0, limit=100, timeout_sec=5.0):
        assert request_id
        if self.events_error is not None:
            raise self.events_error
        visible = [event for event in self.events.get("events", []) if int(event.get("sequence", 0)) > after_sequence]
        return {**copy.deepcopy(self.events), "events": visible}

    def cancel_execution(self, **_kwargs):
        return {"accepted": True, "state": "STOPPING"}


class Browser:
    def __init__(self, port: int) -> None:
        self.port = port
        self.cookies: dict[str, str] = {}

    def request(self, method: str, path: str, body=None):
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": f"http://127.0.0.1:{self.port}",
            "Cookie": "; ".join(f"{key}={value}" for key, value in self.cookies.items()),
            "X-CSRF-Token": self.cookies.get("_robot_skill_web_csrf", ""),
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
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

    def connect(self) -> dict[str, object]:
        assert self.request("GET", "/api/session")[0] == 200
        status, payload = self.request("POST", "/api/session", {})
        assert status == 201, payload
        return payload["connection"]


class RunningWeb:
    """One Web process; ``restart`` shares only the durable directory and ledger."""

    def __init__(self, directory, gateway: FakeGateway) -> None:
        self.directory = directory
        self.gateway = gateway
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = WebServer(client=gateway, port=self.port, submission_directory=directory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while self.server.httpd is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert self.server.httpd is not None
        self.browser = Browser(self.port)
        self.browser.connect()

    def inject_preview(self, *, preview_id: str | None = None, request_id: str = REQUEST_ID) -> str:
        preview_id = preview_id or uuid.uuid4().hex
        # ``require_connection`` clears previews when the observed connection
        # changes.  The session was already established, so align the state
        # before adding the in-memory prepared plan.
        self.server.state.connection_id = self.gateway.session["connection_id"]
        self.gateway.request_ids[preview_id] = request_id
        self.server.state.previews[preview_id] = {
            "preview_id": preview_id,
            "request_id": request_id,
            "connection_id": self.gateway.session["connection_id"],
            "command": {"skill_name": "wave_hand", "display_name": "wave", "parameters": {}},
        }
        return preview_id

    def submit(self, preview_id: str):
        return self.browser.request(
            "POST",
            "/api/executions",
            {"preview_id": preview_id, "connection_version": 1},
        )

    def get_execution(self, reference: dict[str, object], after: int = 0):
        path = "/api/executions/" + quote(json.dumps(reference), safe="")
        if after:
            path += f"?after_sequence={after}"
        return self.browser.request("GET", path)

    def unresolved(self) -> list[dict[str, object]]:
        return self.browser.request("GET", "/api/session")[1]["unresolved_executions"]

    def close(self) -> None:
        self.server.close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()

    def restart(self) -> RunningWeb:
        self.close()
        return RunningWeb(self.directory, self.gateway)

    def wait_worker(self, preview_id: str) -> None:
        worker = self.server.state.submission_workers.get(preview_id)
        if worker is None:
            return
        with worker["ready"]:
            assert worker["ready"].wait_for(lambda: worker["done"], timeout=5), "worker did not settle"


@pytest.fixture
def running(tmp_path):
    gateway = FakeGateway()
    web = RunningWeb(tmp_path / "state", gateway)
    yield gateway, web
    web.close()


def _submitted_reference(body: dict[str, object]) -> dict[str, object]:
    assert "execution" in body, body
    return body["execution"]


def test_lost_receipt_is_reconciled_from_events_after_restart(running):
    gateway, web = running
    gateway.submit_behavior = "lost"
    preview_id = web.inject_preview()

    status, body = web.submit(preview_id)
    assert status == 503, body
    assert body["error"]["code"] == "SUBMISSION_PENDING"
    reference = _submitted_reference(body)
    assert reference["request_id"] == REQUEST_ID
    assert gateway.submit_calls == 1

    web.wait_worker(preview_id)
    record = web.server.state.submissions.get(preview_id)
    assert record["state"] == "unknown"
    assert web.unresolved() != []

    web = web.restart()
    try:
        unresolved = web.unresolved()
        assert [item["preview_id"] for item in unresolved] == [preview_id]
        assert unresolved[0]["reference"] == reference
        # Recovering a saved record is explicit at the client boundary.
        assert gateway.restored and gateway.restored[-1] == reference

        # An execution snapshot that has not linked the request yet still has
        # usable ledger events.  The GET may associate, but must not unlock.
        gateway.add_event(1, "ACCEPTED")
        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["found"] is False
        record = web.server.state.submissions.get(preview_id)
        assert record["association"]["execution_id"] == EXECUTION_ID
        assert record["phase"] == "active"
        assert web.unresolved()[0]["preview_id"] == preview_id

        # Accepted/running is known active work: no second goal is admitted.
        second = web.inject_preview(request_id="request-reconcile-2")
        status, blocked = web.submit(second)
        assert status == 409, blocked
        assert blocked["error"]["code"] == "EXECUTION_UNRESOLVED"
        assert blocked["execution"] == record["reference"]
        assert gateway.submit_calls == 1

        gateway.add_event(2, "RUNNING")
        status, observed = web.get_execution(reference, after=1)
        assert status == 200, observed
        assert web.unresolved()[0]["preview_id"] == preview_id

        # The terminal event is enough even when the snapshot is still blind.
        gateway.add_event(3, "SUCCEEDED")
        status, observed = web.get_execution(reference, after=2)
        assert status == 200, observed
        assert observed["execution"]["found"] is True
        assert observed["execution"]["state"] == "SUCCEEDED"
        assert web.unresolved() == []
        terminal_record = web.server.state.submissions.get(preview_id)
        assert terminal_record["observation"]["execution"]["found"] is True
        assert terminal_record["observation"]["execution"]["state"] == "SUCCEEDED"

        # Only now can a later submission start, and it starts exactly once.
        gateway.submit_behavior = "accepted"
        status, accepted = web.submit(second)
        assert status == 202, accepted
        assert accepted["execution"]["request_id"] == "request-reconcile-2"
        assert gateway.submit_calls == 2
    finally:
        web.close()


@pytest.mark.parametrize(("state", "expected_phase"), [("ACCEPTED", "active"), ("RUNNING", "active")])
def test_known_active_execution_keeps_the_admission_lock(running, state, expected_phase):
    gateway, web = running
    gateway.submit_behavior = "lost"
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 503, body
    reference = body["execution"]
    web.wait_worker(preview_id)

    gateway.execution_found = True
    gateway.execution_state = state
    status, observed = web.get_execution(reference)
    assert status == 200, observed
    assert observed["execution"]["state"] == state
    record = web.server.state.submissions.get(preview_id)
    assert record["phase"] == expected_phase
    assert record["state"] == "unknown"
    assert web.unresolved()[0]["preview_id"] == preview_id

    status, blocked = web.submit(web.inject_preview(request_id=f"request-{state.lower()}"))
    assert status == 409, blocked
    assert gateway.submit_calls == 1


def test_unavailable_ledger_never_unlocks_unknown_even_after_repeated_gets(running, monkeypatch):
    gateway, web = running
    gateway.submit_behavior = "lost"
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 503, body
    reference = body["execution"]
    web.wait_worker(preview_id)

    gateway.execution_error = BridgeError("ROS_UNAVAILABLE", "offline", exit_code=4)
    gateway.events_error = BridgeError("ROS_UNAVAILABLE", "offline", exit_code=4)
    for _ in range(3):
        status, body = web.get_execution(reference)
        assert status == 503, body
        assert web.server.state.submissions.unresolved()
        assert web.server.state.submissions.get(preview_id)["state"] == "unknown"

    status, blocked = web.submit(web.inject_preview(request_id="request-after-offline"))
    assert status == 409, blocked
    assert gateway.submit_calls == 1


def test_events_only_terminal_result_is_authoritative_when_the_snapshot_fails(running):
    gateway, web = running
    gateway.submit_behavior = "lost"
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 503, body
    reference = body["execution"]
    web.wait_worker(preview_id)

    gateway.execution_error = ClientError("ROS_UNAVAILABLE", "get execution timed out")
    gateway.add_event(1, "SUCCEEDED")
    status, observed = web.get_execution(reference)
    assert status == 200, observed
    assert observed["execution"]["found"] is True
    assert observed["execution"]["state"] == "SUCCEEDED"
    assert web.unresolved() == []


def test_pre_goal_failure_is_terminal_and_reports_the_real_error(running):
    gateway, web = running
    gateway.submit_behavior = "pre_failure"
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 503, body
    assert body["error"]["code"] == "SERVER_UNAVAILABLE"
    web.wait_worker(preview_id)
    record = web.server.state.submissions.get(preview_id)
    assert record["state"] == "REJECTED"
    assert record["failure"]["code"] == "SERVER_UNAVAILABLE"
    assert web.unresolved() == []

    # Repeating the rejected preview returns the same saved terminal and does
    # not send another goal.
    status, repeated = web.submit(preview_id)
    assert status == 200, repeated
    assert repeated["already_submitted"] is True
    assert repeated["submission_rejected"] is True
    assert repeated["error"]["code"] == "SERVER_UNAVAILABLE"
    assert gateway.submit_calls == 1

    gateway.submit_behavior = "accepted"
    status, accepted = web.submit(web.inject_preview(request_id="request-after-failure"))
    assert status == 202, accepted
    assert gateway.submit_calls == 2


def test_slow_pre_goal_phase_returns_a_queryable_reference(running, monkeypatch):
    monkeypatch.setattr(server_module, "SUBMISSION_RESPONSE_TIMEOUT_SEC", 0.1)
    gateway, web = running
    gateway.submit_behavior = "blocked"
    preview_id = web.inject_preview()

    status, body = web.submit(preview_id)
    assert status == 503, body
    assert body["error"]["code"] == "SUBMISSION_PENDING"
    reference = body["execution"]
    assert reference["request_id"] == REQUEST_ID
    assert gateway.submit_calls == 1

    # The worker settles later; the durable record still owns the decision and
    # a read-only query can report the rejection instead of an endless spinner.
    gateway.release.set()
    web.wait_worker(preview_id)
    status, observed = web.get_execution(reference)
    assert status == 200, observed
    assert observed["execution"]["submission_rejected"] is True
    assert observed["execution"]["error_code"] == "SERVER_UNAVAILABLE"
    assert web.unresolved() == []
    assert gateway.submit_calls == 1


def test_same_preview_and_request_are_idempotent_across_restart(running):
    gateway, web = running
    preview_id = web.inject_preview()
    status, first = web.submit(preview_id)
    assert status == 202, first
    status, second = web.submit(preview_id)
    assert status == 202, second
    assert first["execution"] == second["execution"]
    assert gateway.submit_calls == 1

    web = web.restart()
    try:
        # No in-memory preview survives, but the durable association is enough:
        # the saved reference is returned and the goal is never replayed.
        status, third = web.submit(preview_id)
        assert status == 202, third
        assert third["execution"] == first["execution"]
        assert gateway.submit_calls == 1
    finally:
        web.close()


def test_conflicting_reference_identity_is_rejected(running):
    gateway, web = running
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 202, body
    reference = body["execution"]

    conflicting = dict(reference)
    conflicting["execution_id"] = "execution-somebody-else"
    status, body = web.get_execution(conflicting)
    assert status == 409, body
    assert body["error"]["code"] == "EXECUTION_IDENTITY_CONFLICT"
    assert body["execution"] == reference
    record = web.server.state.submissions.get(preview_id)
    assert record["reference"]["execution_id"] == ""
    assert "association" not in record

    # The same preview id cannot be rebound to a different request identity.
    web.inject_preview(preview_id=preview_id, request_id="request-somebody-else")
    status, body = web.submit(preview_id)
    assert status == 409, body
    assert body["error"]["code"] == "EXECUTION_IDENTITY_CONFLICT"
    assert gateway.submit_calls == 1

    wrong_scope = dict(reference)
    wrong_scope["caller_scope"] = "another-caller"
    status, body = web.get_execution(wrong_scope)
    assert status == 409, body

    wrong_target = dict(reference)
    wrong_target["target_key"] = "another-binding"
    status, body = web.get_execution(wrong_target)
    assert status == 409, body


def test_late_nonterminal_ack_does_not_downgrade_a_terminal_record(running):
    gateway, web = running
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 202, body
    reference = body["execution"]

    gateway.execution_state = "SUCCEEDED"
    status, observed = web.get_execution(reference)
    assert observed["execution"]["state"] == "SUCCEEDED"
    assert web.unresolved() == []

    # A stale snapshot/ack after the terminal observation must not reopen work.
    gateway.execution_state = "RUNNING"
    status, observed = web.get_execution(reference)
    assert status == 200, observed
    assert observed["execution"]["state"] == "SUCCEEDED"
    assert web.unresolved() == []
    status, accepted = web.submit(web.inject_preview(request_id="request-after-terminal"))
    assert status == 202, accepted


def test_concurrent_gets_share_one_terminal_resolution(running):
    gateway, web = running
    preview_id = web.inject_preview()
    status, body = web.submit(preview_id)
    assert status == 202, body
    reference = body["execution"]
    gateway.execution_state = "SUCCEEDED"
    results: list[tuple[int, dict[str, object]]] = []

    def observe() -> None:
        results.append(web.get_execution(reference))

    threads = [threading.Thread(target=observe) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert [status for status, _body in results] == [200] * len(threads)
    assert all(body["execution"]["state"] == "SUCCEEDED" for _status, body in results)
    assert web.unresolved() == []
    record = web.server.state.submissions.get(preview_id)
    assert record["state"] == "SUCCEEDED"


# ---------------------------------------------------------------------------
# Store-level lifecycle tests
# ---------------------------------------------------------------------------


def stored_reference(request_id: str = REQUEST_ID) -> dict[str, str]:
    return {
        "target_key": TARGET_KEY,
        "robot_id": "mock_g1",
        "caller_scope": "robot_skill_cli",
        "request_id": request_id,
        "execution_id": "",
        "legacy_task_id": "",
    }


def test_store_intent_reload_and_events_only_association(tmp_path):
    from robot_skill_web.submission_store import PendingSubmissionStore

    store = PendingSubmissionStore(tmp_path)
    store.reserve_intent("preview", "connection", stored_reference())
    record = PendingSubmissionStore(tmp_path).get("preview")
    assert record["state"] == "unknown"
    assert record["phase"] == "submitting"
    assert record["goal_may_be_sent"] is False

    store.mark_goal_may_be_sent("preview", stored_reference())
    record = store.get("preview")
    assert record["phase"] == "pending"
    assert record["goal_may_be_sent"] is True
    assert "association" not in record
    assert store.unresolved()[0]["phase"] == "pending"

    result = store.reconcile(
        "preview",
        {"found": False, "observation_unknown": True},
        {
            "found": True,
            "execution_id": EXECUTION_ID,
            "events": [{"sequence": 1, "execution_id": EXECUTION_ID, "state": "ACCEPTED"}],
        },
        reference=stored_reference(),
    )
    assert result["terminal"] is False
    record = PendingSubmissionStore(tmp_path).get("preview")
    assert record["association"]["execution_id"] == EXECUTION_ID
    assert record["phase"] == "active"
    assert record["reference"]["execution_id"] == ""
    assert store.unresolved()

    result = store.reconcile(
        "preview",
        {"found": False},
        {"found": True, "events": [{"sequence": 2, "execution_id": EXECUTION_ID, "state": "SUCCEEDED"}]},
        reference=stored_reference(),
    )
    assert result["terminal"] is True
    assert result["state"] == "SUCCEEDED"
    assert store.unresolved() == []


def test_store_terminal_is_monotonic_and_ignores_late_ack(tmp_path):
    from robot_skill_web.submission_store import PendingSubmissionStore

    store = PendingSubmissionStore(tmp_path)
    store.reserve_intent("preview", "connection", stored_reference())
    store.mark_goal_may_be_sent("preview", stored_reference())
    store.reconcile("preview", {"found": True, "state": "SUCCEEDED"}, {"events": []}, reference=stored_reference())
    assert store.get("preview")["state"] == "SUCCEEDED"

    late = store.reconcile("preview", {"found": True, "state": "RUNNING"}, {"events": []}, reference=stored_reference())
    assert late["terminal"] is True
    assert late["state"] == "SUCCEEDED"
    other = store.reconcile("preview", {"found": True, "state": "FAILED"}, {"events": []}, reference=stored_reference())
    assert other["state"] == "SUCCEEDED"
    assert store.unresolved() == []


def test_store_conflicting_association_is_rejected_and_not_overwritten(tmp_path):
    from robot_skill_web.submission_store import PendingSubmissionStore

    store = PendingSubmissionStore(tmp_path)
    store.reserve_intent("preview", "connection", stored_reference())
    store.mark_goal_may_be_sent("preview", stored_reference())
    store.reconcile(
        "preview",
        {"found": True, "state": "RUNNING", "execution_id": EXECUTION_ID},
        {"events": []},
        reference=stored_reference(),
    )
    before = store.get("preview")

    conflicting = stored_reference()
    conflicting["execution_id"] = "execution-somebody-else"
    with pytest.raises(ClientError) as error:
        store.reconcile("preview", {"found": False}, {"events": []}, reference=conflicting)
    assert error.value.code == "EXECUTION_IDENTITY_CONFLICT"
    assert store.get("preview")["association"]["execution_id"] == EXECUTION_ID

    inconsistent = dict(stored_reference())
    inconsistent["execution_id"] = "execution-third"
    with pytest.raises(ClientError):
        store.mark_goal_may_be_sent("preview", inconsistent)
    assert store.get("preview") == before


def test_store_pre_goal_rejection_is_terminal_but_pending_is_not(tmp_path):
    from robot_skill_web.submission_store import PendingSubmissionStore

    rejected = PendingSubmissionStore(tmp_path / "rejected")
    rejected.reserve_intent("preview", "connection", stored_reference())
    rejected.mark_failure("preview", error_code="SERVER_UNAVAILABLE", message="not ready", rejected=True)
    record = rejected.get("preview")
    assert record["state"] == "REJECTED"
    assert record["failure"] == {"code": "SERVER_UNAVAILABLE", "message": "not ready"}
    assert rejected.unresolved() == []

    pending = PendingSubmissionStore(tmp_path / "pending")
    pending.reserve_intent("preview", "connection", stored_reference())
    pending.mark_goal_may_be_sent("preview", stored_reference())
    pending.mark_failure("preview", error_code="OBSERVATION_UNKNOWN", message="receipt lost", rejected=False)
    assert pending.get("preview")["state"] == "unknown"
    assert pending.unresolved()[0]["error_code"] == "OBSERVATION_UNKNOWN"


def test_store_legacy_unknown_record_loads_as_pending(tmp_path):
    from robot_skill_web.serialization import SubmissionStore
    from robot_skill_web.submission_store import PendingSubmissionStore

    SubmissionStore(tmp_path).reserve("legacy", "connection", stored_reference())
    store = PendingSubmissionStore(tmp_path)
    record = store.get("legacy")
    assert record["state"] == "unknown"
    unresolved = store.unresolved()
    assert unresolved[0]["preview_id"] == "legacy"
    assert unresolved[0]["phase"] == "pending"


def test_store_reads_message_like_event_objects(tmp_path):
    from robot_skill_web.submission_store import PendingSubmissionStore

    store = PendingSubmissionStore(tmp_path)
    store.reserve_intent("preview", "connection", stored_reference())
    store.mark_goal_may_be_sent("preview", stored_reference())
    events = {
        "found": True,
        "execution_id": EXECUTION_ID,
        "events": [
            SimpleNamespace(sequence=1, execution_id=EXECUTION_ID, state="RUNNING", event_type="RUNNING"),
            SimpleNamespace(sequence=2, execution_id=EXECUTION_ID, state="SUCCEEDED", event_type="SUCCEEDED"),
        ],
    }
    result = store.reconcile("preview", {"found": False}, events, reference=stored_reference())
    assert result["terminal"] is True
    assert result["state"] == "SUCCEEDED"
    assert store.get("preview")["association"]["execution_id"] == EXECUTION_ID


def test_real_client_restore_and_read_only_query_contract(tmp_path):
    from robot_client.client import RobotClient
    from robot_client.transport import gateway_transport
    from robot_skill_web.submission_store import PendingSubmissionStore

    bridge = ClientBridge()
    client = RobotClient(
        gateway_transport({}, robot_id="g1", robot_name="g1"),
        transport_factory=lambda **_kwargs: bridge,
    )
    session = client.session
    reference = {
        "target_key": session["target_key"],
        "robot_id": session["robot_id"],
        "caller_scope": session["caller_scope"],
        "request_id": "request-real-client",
        "execution_id": "",
        "legacy_task_id": "",
    }
    store = PendingSubmissionStore(tmp_path / "state")
    store.reserve_intent("preview", session["connection_id"], reference)
    store.mark_goal_may_be_sent("preview", reference)

    web = RunningWeb(tmp_path / "state", client)
    try:
        # The real client's read-only get never creates a lock; the Web layer
        # must adopt its durable record explicitly on session establishment.
        assert client._unresolved[reference["request_id"]] == reference

        bridge.execution = {"found": True, "state": "RUNNING", "execution_id": "execution-real"}
        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["state"] == "RUNNING"
        assert reference["request_id"] in client._unresolved
        assert web.server.state.submissions.get("preview")["phase"] == "active"

        bridge.execution = {"found": True, "state": "SUCCEEDED", "execution_id": "execution-real"}
        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["state"] == "SUCCEEDED"
        assert client._unresolved == {}
        assert web.server.state.submissions.unresolved() == []

        # A read-only query for a reference this deployment never submitted
        # must not create a client admission lock.
        foreign = {**reference, "request_id": "request-foreign"}
        bridge.execution = {"found": False, "state": "UNKNOWN", "execution_id": ""}
        status, observed = web.get_execution(foreign)
        assert status == 200, observed
        assert client._unresolved == {}
    finally:
        web.close()


def test_real_client_events_only_terminal_releases_the_adopted_lock(tmp_path):
    """A durable pending task must not leave two admission states separated.

    After a Web restart the session adopts the saved reference into the real
    client.  If the snapshot is unavailable but the ledger event page reports
    the terminal state, the Web record unlocks from the event evidence; the
    real client must release the same adopted reference, or a later submit
    stays rejected with ``EXECUTION_UNRESOLVED`` forever.
    """
    from robot_client.client import RobotClient
    from robot_client.transport import gateway_transport
    from robot_skill_web.submission_store import PendingSubmissionStore

    bridge = ClientBridge()
    bridge.execution = {"found": False, "state": "UNKNOWN", "execution_id": ""}
    bridge.events = {
        "found": True,
        "execution_id": "execution-real",
        "events": [
            {"sequence": 1, "state": "ACCEPTED", "execution_id": "execution-real"},
            {"sequence": 2, "state": "SUCCEEDED", "execution_id": "execution-real"},
        ],
        "last_event_sequence": 2,
        "has_more": False,
    }
    client = RobotClient(
        gateway_transport({}, robot_id="g1", robot_name="g1"),
        transport_factory=lambda **_kwargs: bridge,
    )
    session = client.session
    reference = {
        "target_key": session["target_key"],
        "robot_id": session["robot_id"],
        "caller_scope": session["caller_scope"],
        "request_id": "request-real-events-terminal",
        "execution_id": "",
        "legacy_task_id": "",
    }
    store = PendingSubmissionStore(tmp_path / "state")
    store.reserve_intent("preview", session["connection_id"], reference)
    store.mark_goal_may_be_sent("preview", reference)

    web = RunningWeb(tmp_path / "state", client)
    try:
        assert reference["request_id"] in client._unresolved
        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["state"] == "SUCCEEDED"
        assert web.server.state.submissions.unresolved() == []
        assert client._unresolved == {}
    finally:
        web.close()


def _real_client_pending(tmp_path, bridge, request_id):
    from robot_client.client import RobotClient
    from robot_client.transport import gateway_transport
    from robot_skill_web.submission_store import PendingSubmissionStore

    client = RobotClient(
        gateway_transport({}, robot_id="g1", robot_name="g1"),
        transport_factory=lambda **_kwargs: bridge,
    )
    session = client.session
    reference = {
        "target_key": session["target_key"],
        "robot_id": session["robot_id"],
        "caller_scope": session["caller_scope"],
        "request_id": request_id,
        "execution_id": "",
        "legacy_task_id": "",
    }
    store = PendingSubmissionStore(tmp_path / "state")
    store.reserve_intent("preview", session["connection_id"], reference)
    store.mark_goal_may_be_sent("preview", reference)
    return client, reference, RunningWeb(tmp_path / "state", client), store


def test_real_client_running_snapshot_keeps_the_recovered_task_busy(tmp_path):
    bridge = ClientBridge()
    bridge.execution = {"found": True, "state": "RUNNING", "execution_id": "execution-real"}
    bridge.events = {
        "found": True,
        "execution_id": "execution-real",
        "events": [],
        "last_event_sequence": 0,
        "has_more": False,
    }
    client, reference, web, _store = _real_client_pending(tmp_path, bridge, "request-real-running")
    try:
        assert reference["request_id"] in client._unresolved
        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["state"] == "RUNNING"
        assert web.server.state.submissions.unresolved() != []
        assert reference["request_id"] in client._unresolved
        assert bridge.goals == []
    finally:
        web.close()


def test_real_client_unavailable_ledger_keeps_unknown_locked(tmp_path):
    bridge = ClientBridge()
    bridge.execution_error = BridgeError("ROS_UNAVAILABLE", "snapshot service unavailable", exit_code=4)
    bridge.events_error = BridgeError("ROS_UNAVAILABLE", "event service unavailable", exit_code=4)
    client, reference, web, _store = _real_client_pending(tmp_path, bridge, "request-real-unknown")
    try:
        first_status, _ = web.get_execution(reference)
        second_status, _ = web.get_execution(reference)
        assert first_status == 503
        assert second_status == 503
        assert [record["reference"] for record in web.server.state.submissions.unresolved()] == [reference]
        assert reference["request_id"] in client._unresolved
        assert bridge.goals == []
    finally:
        web.close()


def test_real_client_rejected_record_is_not_revived_by_a_late_get(tmp_path):
    from robot_client.client import RobotClient
    from robot_client.transport import gateway_transport
    from robot_skill_web.submission_store import PendingSubmissionStore

    bridge = ClientBridge()
    client = RobotClient(
        gateway_transport({}, robot_id="g1", robot_name="g1"),
        transport_factory=lambda **_kwargs: bridge,
    )
    session = client.session
    reference = {
        "target_key": session["target_key"],
        "robot_id": session["robot_id"],
        "caller_scope": session["caller_scope"],
        "request_id": "request-real-rejected",
        "execution_id": "",
        "legacy_task_id": "",
    }
    store = PendingSubmissionStore(tmp_path / "state")
    store.reserve_intent("preview", session["connection_id"], reference)
    store.mark_goal_may_be_sent("preview", reference)
    store.mark_failure("preview", error_code="GOAL_REJECTED", message="Gateway rejected", rejected=True)

    web = RunningWeb(tmp_path / "state", client)
    try:
        assert client._unresolved == {}
        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["submission_rejected"] is True
        assert client._unresolved == {}
        assert web.server.state.submissions.unresolved() == []
        assert bridge.goals == []
    finally:
        web.close()


# ---------------------------------------------------------------------------
# Startup recovery: only a provably unsent intent may be released
# ---------------------------------------------------------------------------


def _seed_intent(directory, *, preview_id: str = "preview", reference=None, persisted: bool = False):
    from robot_skill_web.submission_store import PendingSubmissionStore

    reference = stored_reference() if reference is None else reference
    store = PendingSubmissionStore(directory)
    store.reserve_intent(preview_id, "connection-reconcile", reference)
    if persisted:
        store.mark_goal_may_be_sent(preview_id, reference)
    return reference


def test_restart_releases_an_intent_that_provably_never_sent_a_goal(tmp_path):
    gateway = FakeGateway()
    _seed_intent(tmp_path / "state")

    web = RunningWeb(tmp_path / "state", gateway)
    try:
        record = web.server.state.submissions.get("preview")
        assert record["state"] == "REJECTED"
        assert record["failure"]["code"] == "NOT_SENT"
        assert record["goal_may_be_sent"] is False
        assert web.unresolved() == []
        # The recovered intent is never replayed, and a new preview is admitted.
        assert gateway.submit_calls == 0
        status, body = web.submit(web.inject_preview(request_id="request-after-recovery"))
        assert status == 202, body
        assert gateway.submit_calls == 1

        status, detail = web.browser.request("GET", "/api/history/preview")
        assert status == 200, detail
        assert detail["record"]["state"] == "REJECTED"
        assert detail["record"]["failure"]["code"] == "NOT_SENT"
    finally:
        web.close()


def test_restart_keeps_a_persisted_intent_locked_without_replaying(tmp_path):
    gateway = FakeGateway()
    _seed_intent(tmp_path / "state", persisted=True)

    web = RunningWeb(tmp_path / "state", gateway)
    try:
        record = web.server.state.submissions.get("preview")
        assert record["phase"] == "pending"
        assert record["goal_may_be_sent"] is True
        assert [item["preview_id"] for item in web.unresolved()] == ["preview"]

        status, body = web.submit(web.inject_preview(request_id="request-another"))
        assert status == 409, body
        assert body["error"]["code"] == "EXECUTION_UNRESOLVED"

        status, observed = web.get_execution(stored_reference())
        assert status == 200, observed
        assert gateway.submit_calls == 0
        assert web.server.state.submissions.get("preview")["state"] != "SUCCEEDED"
    finally:
        web.close()


def test_startup_recovery_requires_explicit_boolean_not_sent_evidence(tmp_path):
    from robot_client.connection import write_private_json
    from robot_skill_web.submission_store import PendingSubmissionStore

    directory = tmp_path / "state"
    store = PendingSubmissionStore(directory)
    cases = {
        "legacy-missing": {},
        "legacy-non-bool": {"phase": "submitting", "goal_may_be_sent": 0},
        "other-phase": {"phase": "pending", "goal_may_be_sent": False},
        "other-target": {"phase": "submitting", "goal_may_be_sent": False},
    }
    for name, overrides in cases.items():
        reference = {**stored_reference(f"request-{name}")}
        if name == "other-target":
            reference["target_key"] = "another-binding"
        store.reserve_intent(name, "connection-reconcile", reference)
        record = store.get(name)
        if name == "legacy-missing":
            record.pop("phase")
            record.pop("goal_may_be_sent")
        else:
            record.update(overrides)
        write_private_json(directory / f"{name}.json", record)
    store.reserve_intent("unsent", "connection-reconcile", stored_reference("request-unsent"))

    fresh = PendingSubmissionStore(directory)
    assert fresh.recover_unsent(target_key=TARGET_KEY) == ["unsent"]
    for name in cases:
        assert fresh.get(name)["state"] == "unknown", name
        assert name in {record["preview_id"] for record in fresh.unresolved()}, name
    assert fresh.get("unsent")["state"] == "REJECTED"
    assert fresh.get("unsent")["failure"]["code"] == "NOT_SENT"


def test_a_live_worker_prevents_startup_recovery_from_marking_its_intent(running, monkeypatch):
    monkeypatch.setattr(server_module, "SUBMISSION_RESPONSE_TIMEOUT_SEC", 0.1)
    gateway, web = running
    gateway.submit_behavior = "blocked"
    preview_id = web.inject_preview()

    status, body = web.submit(preview_id)
    assert status == 503, body
    record = web.server.state.submissions.get(preview_id)
    assert record["phase"] == "submitting"
    assert record["goal_may_be_sent"] is False

    assert web.server.state.recover_unsent() == []
    assert web.server.state.submissions.get(preview_id)["phase"] == "submitting"

    gateway.release.set()
    web.wait_worker(preview_id)
    assert web.server.state.submissions.get(preview_id)["failure"]["code"] == "SERVER_UNAVAILABLE"


def test_real_client_startup_recovery_releases_an_unsent_intent(tmp_path):
    from robot_client.client import RobotClient
    from robot_client.transport import gateway_transport
    from robot_skill_web.submission_store import PendingSubmissionStore

    bridge = ClientBridge()
    client = RobotClient(
        gateway_transport({}, robot_id="g1", robot_name="g1"),
        transport_factory=lambda **_kwargs: bridge,
    )
    session = client.session
    reference = {
        "target_key": session["target_key"],
        "robot_id": session["robot_id"],
        "caller_scope": session["caller_scope"],
        "request_id": "request-unsent-real",
        "execution_id": "",
        "legacy_task_id": "",
    }
    store = PendingSubmissionStore(tmp_path / "state")
    store.reserve_intent("preview", session["connection_id"], reference)

    web = RunningWeb(tmp_path / "state", client)
    try:
        record = web.server.state.submissions.get("preview")
        assert record["state"] == "REJECTED"
        assert record["failure"]["code"] == "NOT_SENT"
        assert client._unresolved == {}

        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert observed["execution"]["submission_rejected"] is True
        assert observed["execution"]["found"] is False
        assert bridge.goals == []
    finally:
        web.close()


# ---------------------------------------------------------------------------
# Observation-only GETs never adopt a new client lock
# ---------------------------------------------------------------------------


def test_a_read_only_get_never_adopts_a_new_client_lock(tmp_path):
    bridge = ClientBridge()
    bridge.execution = {"found": True, "state": "RUNNING", "execution_id": "execution-real"}
    bridge.events = {
        "found": True,
        "execution_id": "execution-real",
        "events": [],
        "last_event_sequence": 0,
        "has_more": False,
    }
    client, reference, web, _store = _real_client_pending(tmp_path, bridge, "request-read-only")
    try:
        assert reference["request_id"] in client._unresolved  # session establishment adopted it
        with client._lock:
            client._unresolved.pop(reference["request_id"], None)

        status, observed = web.get_execution(reference)
        assert status == 200, observed
        assert client._unresolved == {}
    finally:
        web.close()


def test_a_get_between_read_and_rejection_cannot_revive_the_client_lock(tmp_path, monkeypatch):
    bridge = ClientBridge()
    bridge.execution = {"found": True, "state": "RUNNING", "execution_id": "execution-real"}
    bridge.events = {
        "found": True,
        "execution_id": "execution-real",
        "events": [],
        "last_event_sequence": 0,
        "has_more": False,
    }
    client, reference, web, _store = _real_client_pending(tmp_path, bridge, "request-race")
    state = web.server.state
    store = state.submissions
    read = threading.Event()
    release = threading.Event()
    original_find = store.find_for_reference

    def gated_find(candidate):
        record = original_find(candidate)
        read.set()
        assert release.wait(timeout=5), "GET did not resume"
        return record

    monkeypatch.setattr(store, "find_for_reference", gated_find)
    results = []
    thread = threading.Thread(target=lambda: results.append(web.get_execution(reference)), daemon=True)
    thread.start()
    try:
        assert read.wait(timeout=5), "GET did not reach the saved record"
        # The worker receives GOAL_REJECTED: the client pops both ledgers, then
        # the Web persists the terminal rejection.
        with client._lock:
            client._unresolved.pop(reference["request_id"], None)
            client._submitted.pop("preview", None)
        store.mark_failure("preview", error_code="GOAL_REJECTED", message="Gateway rejected the request", rejected=True)
    finally:
        release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    status, body = results[0]
    assert status == 200, body
    assert store.get("preview")["state"] == "REJECTED"
    assert client._unresolved == {}

    # A later history read observes the saved terminal and adds no lock.
    assert web.browser.request("GET", "/api/history")[0] == 200
    assert web.browser.request("GET", "/api/history/preview")[0] == 200
    assert client._unresolved == {}


def test_a_missing_or_empty_snapshot_is_not_a_cancellation(tmp_path):
    from robot_skill_web.submission_store import PendingSubmissionStore

    store = PendingSubmissionStore(tmp_path / "state")
    reference = stored_reference("request-empty-snapshot")
    store.reserve_intent("preview", "connection-reconcile", reference)
    store.mark_goal_may_be_sent("preview", reference)

    for execution in (None, {}, {"found": False, "state": "CANCELED"}):
        result = store.reconcile("preview", execution, {"events": []}, reference=reference)
        assert result["terminal"] is False, execution
        assert store.get("preview")["state"] == "unknown"
        assert "association" not in store.get("preview")

    result = store.reconcile("preview", {"found": True, "state": "CANCELED"}, {"events": []}, reference=reference)
    assert result["terminal"] is True
    assert store.get("preview")["state"] == "CANCELED"
