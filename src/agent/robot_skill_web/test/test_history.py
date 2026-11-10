"""History contract tests for the local Web execution archive."""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from urllib.parse import quote

import pytest
from test_http_contract import TARGET_KEY, Browser, MockRobot, web  # noqa: F401 - shared fake HTTP implementation

from robot_skill_web.serialization import SubmissionStore


def execution_path(reference: dict) -> str:
    return "/api/executions/" + quote(json.dumps(reference), safe="")


def history_path(preview_id: str) -> str:
    return "/api/history/" + quote(preview_id, safe="")


def reference(robot_id: str = "mock_g1", caller_scope: str = "robot_skill_cli", suffix: str = "instance-1") -> dict:
    return {
        "request_id": f"request-{robot_id}-{caller_scope}-{suffix}",
        "execution_id": "execution-1",
        "target_key": TARGET_KEY,
        "robot_id": robot_id,
        "caller_scope": caller_scope,
        "legacy_task_id": "",
    }


def archive_record(preview_id: str, **overrides) -> dict:
    value = {
        "preview_id": preview_id,
        "submitted_at": time.time(),
        "state": "SUCCEEDED",
        "reference": reference(),
    }
    value.update(overrides)
    return value


def write_record(directory, record: dict) -> None:
    store = SubmissionStore(directory)
    reserved = store.reserve(
        record["preview_id"],
        "private-session-id",
        record["reference"],
        command=record.get("command"),
    )
    if record.get("observation") is not None:
        store.observe(record["preview_id"], record["observation"]["execution"], record["observation"]["events"])
        reserved = store.get(record["preview_id"])
    if record["state"] != reserved["state"]:
        store.resolve(record["preview_id"], record["state"])
    stored = store.get(record["preview_id"])
    if "submitted_at" in record:
        # The request contract makes ordering durable across adapter restarts.
        stored["submitted_at"] = record["submitted_at"]
        (directory / f"{record['preview_id']}.json").write_text(
            json.dumps(stored, ensure_ascii=False), encoding="utf-8"
        )
    return stored


def _force_reload(live_store) -> None:
    """Reload an adapter store that was created before a fixture record was written."""
    live_store._loaded = False
    live_store._records.clear()
    live_store._unknown.clear()
    live_store.unresolved()


@pytest.fixture
def http_web(request):
    return request.getfixturevalue("web")


def test_history_requires_connected_http_session(http_web):
    browser, robot, _server = http_web
    browser.request("GET", "/api/session")
    assert browser.request("GET", "/api/history")[0] == HTTPStatus.UNAUTHORIZED
    assert browser.request("GET", history_path("preview"))[0] == HTTPStatus.UNAUTHORIZED
    assert robot.calls == []
    assert robot.submissions == robot.cancel_count == 0


def test_history_is_read_only_and_never_queries_live_gateway(http_web):
    browser, robot, server = http_web
    browser.connect()
    write_record(server.state.submissions.directory, archive_record("archived"))
    _force_reload(server.state.submissions)

    def fail(_name):
        raise AssertionError("history must not query the live Gateway")

    def fail_execution(_reference):
        raise AssertionError("history must not query execution state")

    def fail_events(*_args, **_kwargs):
        raise AssertionError("history must not query execution events")

    robot.get_status = fail
    robot.get_execution = fail_execution
    robot.get_execution_events = fail_events
    listing = browser.request("GET", "/api/history")[1]
    detail = browser.request("GET", history_path("archived"))[1]
    assert listing["total"] == 1
    assert detail["record"]["state"] == "SUCCEEDED"
    assert robot.submissions == robot.cancel_count == 0
    assert server.state.submissions.unresolved() == []


def test_history_is_scoped_by_robot_caller_and_bound_target(tmp_path):
    store = SubmissionStore(tmp_path)
    current = {
        "robot_id": "robot-current",
        "caller_scope": "caller-current",
        "target_key": "scope-current:binding-current",
    }
    records = [
        ("current", current),
        ("other-robot", current | {"robot_id": "robot-other"}),
        ("other-caller", current | {"caller_scope": "caller-other"}),
        ("other-scope", current | {"target_key": "other-scope:binding-current"}),
        ("other-binding", current | {"target_key": "scope-current:binding-other"}),
    ]
    for preview_id, ref in records:
        store.reserve(preview_id, "session", {**reference(), **ref})

    history = store.history(
        robot_id="robot-current", caller_scope="caller-current", target_key="scope-current:binding-current"
    )
    assert [record["preview_id"] for record in history["records"]] == ["current"]
    assert history["total"] == 1
    assert (
        store.history_record(
            "current",
            robot_id="robot-current",
            caller_scope="caller-current",
            target_key="scope-current:binding-current",
        )
        is not None
    )
    for preview_id in ("other-robot", "other-caller", "other-scope", "other-binding"):
        assert (
            store.history_record(
                preview_id,
                robot_id="robot-current",
                caller_scope="caller-current",
                target_key="scope-current:binding-current",
            )
            is None
        )


def test_history_hides_private_session_and_command_details(tmp_path):
    store = SubmissionStore(tmp_path)
    store.reserve(
        "preview",
        "private-session",
        reference(),
        command={
            "skill_name": "wave_hand",
            "display_name": "wave",
            "raw_command": "move left quickly",
            "parameters": {"target_name": "private-target"},
        },
    )
    store.observe(
        "preview",
        {"found": True, "state": "SUCCEEDED"},
        {"events": [{"sequence": 1, "message": "done"}], "event_gap": False},
    )
    store.resolve("preview", "SUCCEEDED")

    history = store.history(robot_id="mock_g1", caller_scope="robot_skill_cli", target_key=TARGET_KEY)
    full = store.history_record("preview", robot_id="mock_g1", caller_scope="robot_skill_cli", target_key=TARGET_KEY)

    assert history["records"][0]["command"] == {"skill_name": "wave_hand", "display_name": "wave"}
    assert "session_id" not in history["records"][0]
    assert "observation" not in history["records"][0]
    assert full["command"]["raw_command"] == "move left quickly"
    assert full["command"]["parameters"] == {"target_name": "private-target"}
    assert full["observation"]["execution"]["state"] == "SUCCEEDED"
    assert "session_id" not in full


def test_history_orders_latest_first_limits_and_reports_total(tmp_path):
    store = SubmissionStore(tmp_path)
    for index in range(105):
        preview_id = f"preview-{index:03d}"
        record = store.reserve(preview_id, "session", {**reference(), "request_id": f"request-{index}"})
        record["submitted_at"] = 1_700_000_000 + index
        path = tmp_path / f"{preview_id}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        path.chmod(0o600)

    store = SubmissionStore(tmp_path)
    history = store.history(robot_id="mock_g1", caller_scope="robot_skill_cli", target_key=TARGET_KEY, limit=7)

    assert history["limit"] == 7
    assert history["total"] == 105
    assert len(history["records"]) == 7
    assert [record["preview_id"] for record in history["records"]] == [
        f"preview-{index:03d}" for index in range(104, 97, -1)
    ]


def test_old_records_without_command_and_metadata_are_visible(tmp_path):
    old = {
        "preview_id": "old",
        "session_id": "legacy-session",
        "submitted_at": 1,
        "state": "SUCCEEDED",
        "reference": reference(),
    }
    record_path = tmp_path / "old.json"
    record_path.write_text(json.dumps(old), encoding="utf-8")
    record_path.chmod(0o600)

    reloaded = SubmissionStore(tmp_path)
    history = reloaded.history(robot_id="mock_g1", caller_scope="robot_skill_cli", target_key=TARGET_KEY)
    full = reloaded.history_record("old", robot_id="mock_g1", caller_scope="robot_skill_cli", target_key=TARGET_KEY)

    assert history["records"][0]["preview_id"] == "old"
    assert history["records"][0]["command"] == {}
    assert history["records"][0]["reference"]["request_id"] == "request-mock_g1-robot_skill_cli-instance-1"
    assert "session_id" not in full


def test_history_survives_backend_restart_and_remains_scoped_to_current_target(http_web):
    browser, robot, server = http_web
    browser.connect()
    prepared = browser.preview()
    submitted = browser.request(
        "POST", "/api/executions", {"preview_id": prepared["preview_id"], "connection_version": 1}
    )[1]["execution"]
    browser.request("GET", execution_path(submitted))

    from robot_skill_web.server import WebState

    replacement = WebState(client=robot, submission_directory=server.state.submissions.directory)
    server.httpd.web_state = replacement
    server.state = replacement
    browser.cookies = {}
    browser.connect()

    listing = browser.request("GET", "/api/history")[1]
    detail = browser.request("GET", history_path(prepared["preview_id"]))[1]["record"]
    assert listing["total"] == 1
    assert listing["records"][0]["reference"] == submitted
    assert detail["reference"] == submitted
    assert detail["command"]["skill_name"] == "wave_hand"
    assert detail["observation"]["execution"]["state"] == "ACTIVE"
    assert robot.submissions == 1
    assert browser.request("GET", "/api/session")[1]["connection"]["connection_id"] == "connection-1"


def test_web_submission_stores_command_labels_and_terminal_observation(http_web):
    browser, robot, server = http_web
    browser.connect()
    prepared = browser.preview()
    submitted = browser.request(
        "POST", "/api/executions", {"preview_id": prepared["preview_id"], "connection_version": 1}
    )[1]["execution"]
    assert browser.request("GET", execution_path(submitted))[1]["execution"]["state"] == "ACTIVE"

    robot.state = "SUCCEEDED"
    observed = browser.request("GET", execution_path(submitted))[1]
    assert observed["execution"]["state"] == "SUCCEEDED"

    history = browser.request("GET", "/api/history")[1]
    full = browser.request("GET", history_path(prepared["preview_id"]))[1]["record"]
    assert history["records"][0]["command"] == {"skill_name": "wave_hand", "display_name": "挥手"}
    assert full["command"] == {
        "skill_name": "wave_hand",
        "display_name": "挥手",
        "raw_command": "mock wave",
        "parameters": {},
    }
    assert full["state"] == "SUCCEEDED"
    assert full["observation"]["execution"] == observed["execution"]
    assert full["observation"]["events"]["events"] == []
    assert "session_id" not in full
    assert server.state.submissions.unresolved() == []


def test_terminal_archive_is_available_after_gateway_becomes_unavailable(http_web, monkeypatch):
    browser, robot, server = http_web
    browser.connect()
    prepared = browser.preview()
    submitted = browser.request(
        "POST", "/api/executions", {"preview_id": prepared["preview_id"], "connection_version": 1}
    )[1]["execution"]
    robot.state = "FAILED"
    assert browser.request("GET", execution_path(submitted))[1]["execution"]["state"] == "FAILED"
    terminal_record = server.state.submissions.get(prepared["preview_id"])

    def unavailable(*_args, **_kwargs):
        raise AssertionError("history must not access the unavailable Gateway")

    monkeypatch.setattr(robot, "get_execution", unavailable)
    monkeypatch.setattr(robot, "get_execution_events", unavailable)

    full = browser.request("GET", history_path(prepared["preview_id"]))[1]["record"]
    assert full["state"] == "FAILED"
    assert full["observation"]["execution"]["state"] == "FAILED"
    assert terminal_record["state"] == "FAILED"
    assert robot.submissions == 1


def test_history_reads_do_not_promote_unknown_records(http_web, monkeypatch):
    browser, robot, server = http_web
    browser.connect()
    write_record(server.state.submissions.directory, archive_record("unknown", state="unknown"))
    _force_reload(server.state.submissions)
    before = server.state.submissions.get("unknown")

    listing = browser.request("GET", "/api/history")[1]
    full = browser.request("GET", history_path("unknown"))[1]["record"]
    after = server.state.submissions.get("unknown")

    assert listing["records"][0]["state"] == "unknown"
    assert full["state"] == "unknown"
    assert "reconciled" not in before
    assert before == after
    assert server.state.submissions.unresolved()[0]["preview_id"] == "unknown"


def test_observe_merges_deduplicates_and_bounds_event_cache(tmp_path):
    store = SubmissionStore(tmp_path)
    store.reserve("preview", "session", reference())
    first_events = {
        "events": [{"sequence": 1, "message": "one"}, {"sequence": 2, "message": "old"}],
        "event_gap": False,
    }
    store.observe("preview", {"found": True, "state": "ACTIVE"}, first_events)
    # The next query returns only events after sequence 1, including a corrected duplicate.
    store.observe(
        "preview",
        {"found": True, "state": "SUCCEEDED"},
        {"events": [{"sequence": 2, "message": "new"}, {"sequence": 3, "message": "three"}], "truncated": True},
    )
    record = SubmissionStore(tmp_path).get("preview")

    assert record["observation"]["events"]["events"] == [
        {"sequence": 1, "message": "one"},
        {"sequence": 2, "message": "new"},
        {"sequence": 3, "message": "three"},
    ]
    assert record["observation"]["events"]["event_gap"] is False
    assert record["observation"]["events"]["truncated"] is False
    assert record["observed_at"] >= record["submitted_at"]

    terminal = {"found": True, "state": "FAILED"}
    terminal_events = {"events": [], "event_gap": True}
    store.observe("preview", terminal, terminal_events)
    store.resolve("preview", "FAILED")
    # A late nonterminal observation must not replace the terminal archive.
    store.observe("preview", {"found": True, "state": "ACTIVE"}, {"events": [{"sequence": 4}]})
    record = SubmissionStore(tmp_path).get("preview")
    assert record["observation"]["execution"] == terminal
    assert record["observation"]["events"]["event_gap"] is True
    assert record["observation"]["events"]["truncated"] is False
    assert record["observation"]["events"]["events"] == [
        {"sequence": 1, "message": "one"},
        {"sequence": 2, "message": "new"},
        {"sequence": 3, "message": "three"},
    ]

    for base in range(11):
        events = {
            "events": [{"sequence": index, "message": str(index)} for index in range(base * 10, base * 10 + 10)],
            "after_sequence": base * 10,
        }
        store.observe("preview", terminal, events)
    record = SubmissionStore(tmp_path).get("preview")
    assert [event["sequence"] for event in record["observation"]["events"]["events"]] == list(range(10, 110))
    assert record["observation"]["events"]["truncated"] is True


def test_execution_lookup_uses_exact_reference_across_deployments(http_web, monkeypatch):
    browser, robot, server = http_web
    browser.connect()
    prior = write_record(
        server.state.submissions.directory,
        archive_record(
            "prior-instance",
            reference=reference() | {"request_id": "request-1", "target_key": "prior-scope:prior-binding"},
            state="FAILED",
        ),
    )
    _force_reload(server.state.submissions)
    prepared = browser.preview()
    submitted = browser.request(
        "POST", "/api/executions", {"preview_id": prepared["preview_id"], "connection_version": 1}
    )[1]["execution"]
    current_path = execution_path(submitted)
    robot.state = "SUCCEEDED"
    assert browser.request("GET", current_path)[1]["execution"]["state"] == "SUCCEEDED"

    reloaded_prior = server.state.submissions.get("prior-instance")
    assert prior["reference"]["request_id"] == submitted["request_id"]
    assert prior["reference"]["target_key"] != submitted["target_key"]
    assert reloaded_prior["state"] == "FAILED"
    assert (
        server.state.submissions.by_request(submitted["request_id"], reference=submitted)["preview_id"]
        == prepared["preview_id"]
    )


def test_history_detail_returns_404_when_absent_or_out_of_scope(http_web):
    browser, _robot, server = http_web
    browser.connect()
    write_record(server.state.submissions.directory, archive_record("current"))
    out_of_scope = archive_record("out", reference=reference(robot_id="robot-other"))
    write_record(server.state.submissions.directory, out_of_scope)
    _force_reload(server.state.submissions)

    assert browser.request("GET", history_path("missing"))[0] == HTTPStatus.NOT_FOUND
    assert browser.request("GET", history_path("out"))[0] == HTTPStatus.NOT_FOUND
    listing = browser.request("GET", "/api/history")[1]
    assert [record["preview_id"] for record in listing["records"]] == ["current"]


def test_oversized_archives_remain_reloadable_and_keep_latest_observations(tmp_path):
    store = SubmissionStore(tmp_path)
    original_parameters = {"detail": "x" * (70 * 1024)}
    reserved = store.reserve(
        "oversized",
        "session",
        reference(),
        command={
            "skill_name": "wave_hand",
            "display_name": "wave",
            "raw_command": "run " + "x" * 5000,
            "parameters": original_parameters,
        },
    )
    assert reserved["command"]["parameters_omitted"] is True
    assert "parameters" not in reserved["command"]
    assert reserved["command"]["raw_command"] == ("run " + "x" * 5000)[:4000]

    execution = {"found": True, "state": "UNKNOWN", "reconciled": True, "message": "reconciled"}
    store.resolve("oversized", "UNKNOWN", reconciled=True)
    events = {
        "events": [
            {"sequence": index, "message": "latest-" + str(index) + "-" + "x" * 1200} for index in range(190, 291)
        ]
    }
    store.observe("oversized", execution, events)
    record = SubmissionStore(tmp_path).get("oversized")
    assert record["command"]["parameters_omitted"] is True
    assert record["observation"]["execution"] == {
        "found": True,
        "state": "UNKNOWN",
        "reconciled": True,
        "message": "reconciled",
    }
    cached_events = record["observation"]["events"]["events"]
    assert 0 < len(cached_events) < 100
    sequences = [event["sequence"] for event in cached_events]
    assert sequences == list(range(sequences[0], 291))
    assert sequences[-1] == 290
    assert record["observation"]["events"]["truncated"] is True
    assert all(len(event["message"]) <= 4000 for event in cached_events)
    assert len(json.dumps(record)) < 256 * 1024

    store.observe(
        "oversized",
        {"found": True, "state": "ACTIVE", "reconciled": False},
        {"events": [{"sequence": 999, "message": "delayed"}]},
    )
    delayed = SubmissionStore(tmp_path).get("oversized")
    assert delayed["observation"]["execution"] == record["observation"]["execution"]
    assert delayed["observation"]["events"]["events"] == record["observation"]["events"]["events"]
