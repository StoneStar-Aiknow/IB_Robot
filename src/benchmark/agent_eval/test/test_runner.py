"""EpisodeRunner + EventBook behaviour: session intake, admission, result attribution, stop rules."""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.runner import EventBook
from agent_eval_testkit import FakeWall, ScriptedPlatform, fixture_body, make_runner

STATE = {
    "tick": 4,
    "arms": {
        "left": {"xyz": [-0.3, -0.22, 0.92], "quat": [0.0, 0.0, 0.7071068, 0.7071068], "gripper": 1.0, "wrist": 0.0},
        "right": {"xyz": [0.3, -0.22, 0.92], "quat": [0.0, 0.0, 0.7071068, 0.7071068], "gripper": 1.0, "wrist": 0.0},
    },
}


def with_left(tick, **left):
    state = json.loads(json.dumps(STATE))
    state["tick"] = tick
    state["arms"]["left"].update(left)
    return state


def events(state, message="ok", steps=None):
    out = []
    if steps:
        out.append({"event_type": "presentation", "detail": {"presentation": {"summary": "", "steps": steps}}})
    out.append({"event_type": "terminal", "state": state, "user_message": message})
    return {"kind": "events", "events": out}


def skill(name, **kw):
    return [{"skill_name": name, "arm_side": "left", **kw}]


def start_event(wall, session_id="s0001", generation=1, **kw):
    return {"event": "session_start", "session_id": session_id, "generation": generation, "wall_time": wall.t, **kw}


def test_latched_sessions_from_before_the_runner_and_stale_generations_are_ignored():
    wall = FakeWall(1000.0)
    book = EventBook(fixture_body(), wall_clock=wall)
    book.on_session(start_event(FakeWall(900.0), "s0007", 7))  # latched from an earlier run
    book.on_session(start_event(wall, "s0003", 3))
    book.on_session(start_event(wall, "s0002", 2))  # older generation than one already seen
    book.on_session(start_event(wall, "s0003", 3))  # repeat
    book.on_session(start_event(wall, "s0004", 4))
    assert [book.next_episode(0.01)["session_id"] for _ in range(2)] == ["s0003", "s0004"]
    assert book.next_episode(0.01) is None
    assert [d["reason"] for d in book.dropped] == [
        "started before the runner (latched)",
        "stale generation",
        "repeated session_start",
    ]


def test_a_late_native_result_of_an_earlier_episode_or_generation_is_not_attributed(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=1, end_wait_s=0.2)
    first = ScriptedPlatform(book, body, [{"reply": events("ANSWERED", "DONE"), "after": None}])
    runner.io = first
    first.start(wall, "s0001", 1, STATE, instruction="Pick up the cup.", step_lim=50)
    # generation 1 is never judged here; the runtime resets to generation 2
    second = ScriptedPlatform(
        book, body, [{"reply": events("ANSWERED", "DONE"), "after": None}], native={"native_success": True}
    )
    runner.io = second
    second.start(wall, "s0002", 2, STATE, instruction="Pick up the cup.", step_lim=50)
    book.next_episode(0.01)  # generation 1 is superseded
    start = book.next_episode(0.01)
    assert start["generation"] == 2
    # results of the superseded generation and of an episode that never ran arrive late
    book.on_native_result({"session_id": "s0001", "generation": 1, "native_success": False, "failure_reason": "old"})
    book.on_native_result({"session_id": "s0000", "generation": 0, "native_success": False})
    second.decisions = iter([{"reply": events("ANSWERED", "DONE"), "after": None}])
    second.native = {"native_success": False, "failure_reason": "timeout", "take_action_cnt": 50}
    record = runner.run_episode(start)
    assert record["native_result"]["generation"] == 2
    assert record["native_result"]["failure_reason"] == "timeout"
    assert {d["reason"] for d in book.dropped} == {"native_result of a session not run here"}


def test_done_ends_the_episode_and_success_is_only_native(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=10)
    decisions = [
        {
            "reply": events("SUCCEEDED", "closed", skill("close_gripper_skill")),
            "after": with_left(20, xyz=[0.1, 0.0, 0.80], gripper=0.3),
        },
        {"reply": events("ANSWERED", "Done"), "after": None},
    ]
    platform = ScriptedPlatform(
        book, body, decisions, native={"native_success": False, "failure_reason": "object not lifted"}
    )
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="Pick up the cup.", step_lim=200)
    record = runner.run_episode(book.next_episode(0.01))
    assert record["finish_reason"] == "agent_done" and len(record["decisions"]) == 2
    assert platform.finish_calls == 1
    # the Agent's DONE ends the episode; RoboDojo's verdict is the result
    assert record["native_result"]["native_success"] is False


def test_a_few_answers_without_a_skill_in_a_row_end_the_episode(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=10)
    platform = ScriptedPlatform(book, body, [{"reply": events("ANSWERED", "thinking"), "after": None}] * 3)
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="x", step_lim=10)
    record = runner.run_episode(book.next_episode(0.01))
    assert record["finish_reason"] == "agent_stopped" and len(platform.texts) == 3  # recipe max_idle_replies


def test_unknown_outcome_stops_the_episode(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=10)
    platform = ScriptedPlatform(book, body, [{"reply": events("UNKNOWN", "lost"), "after": None}])
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="x", step_lim=10)
    record = runner.run_episode(book.next_episode(0.01))
    assert record["finish_reason"] == "agent_quarantined"
    assert len(platform.texts) == 1


def test_a_decision_timeout_stops_the_request_through_agent_control(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=1, decision_timeout_s=0.05)
    platform = ScriptedPlatform(book, body, [{"reply": {"kind": "timeout"}, "after": None}])
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="x", step_lim=10)
    record = runner.run_episode(book.next_episode(0.01))
    rid = record["agent_request_ids"][0]
    assert platform.controls == [{"operation": "stop", "request_id": rid}]
    assert record["decisions"][0]["terminal"]["state"] == "TIMEOUT"


def test_an_episode_without_native_result_has_no_verdict(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=1, end_wait_s=0.05)
    platform = ScriptedPlatform(book, body, [{"reply": events("ANSWERED", "DONE"), "after": None}])
    platform.end = lambda: None  # the platform never ends or judges
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="x", step_lim=10)
    record = runner.run_episode(book.next_episode(0.01))
    assert record["native_result"] is None and record["episode_end"] is None


def test_the_record_is_written_and_the_context_never_sees_the_native_result(tmp_path):
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=2)
    decisions = [
        {
            "reply": events("SUCCEEDED", "up", skill("move_relative_ee", motion_direction="up", motion_distance=0.1)),
            "after": with_left(8),
        },
        {
            "reply": events("SUCCEEDED", "up", skill("move_relative_ee", motion_direction="up", motion_distance=0.1)),
            "after": with_left(16),
        },
    ]
    native = {"native_success": True, "native_score": 1.0, "object_pose": "SECRET_POSE", "seed": 4}
    platform = ScriptedPlatform(book, body, decisions, native=native)
    runner.io = platform
    # a latched native result of an earlier run with the same session id arrives first
    book.on_native_result(
        {"session_id": "s0001", "generation": 1, "native_success": False, "object_pose": "SECRET_OLD"}
    )
    platform.start(wall, "s0001", 1, STATE, instruction="Pick up the cup.", step_lim=200)
    record = runner.run_episode(book.next_episode(0.01))
    assert all("SECRET" not in text for text in platform.texts)
    saved = json.loads(Path(record["record_path"]).read_text(encoding="utf-8"))
    assert saved["agent_request_ids"] == [p["request_id"] for p in platform.payloads]
    assert saved["finish_reason"] == "max_decisions"
    assert saved["native_result"]["native_success"] is True
    assert saved["native_result"]["object_pose"] == "SECRET_POSE"  # the record keeps it, the prompt never had it


def test_a_platform_that_exits_mid_decision_ends_the_episode_without_waiting_out_the_timeout(tmp_path):
    import time

    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=5, decision_timeout_s=60.0, end_wait_s=60.0)
    platform = ScriptedPlatform(book, body, [{"reply": {"kind": "timeout"}, "after": None}])
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="x", step_lim=10)
    gone = {"reason": None}
    book.set_abort_check(lambda: gone["reason"])
    import threading

    threading.Timer(0.3, lambda: gone.update(reason="client exited with code 1")).start()
    book.ABORT_GRACE_S = 0.2
    t0 = time.monotonic()
    record = runner.run_episode(book.next_episode(0.01))
    assert time.monotonic() - t0 < 5.0
    assert record["finish_reason"] == "platform_exited" and "client exited" in record["error"]
    rid = record["agent_request_ids"][0]
    assert platform.controls == [{"operation": "stop", "request_id": rid}]  # the Agent is stopped too
    assert platform.finish_calls == 0 and record["native_result"] is None


def test_waiting_for_an_episode_stops_when_the_platform_has_exited():
    import time

    import pytest
    from agent_eval.runner import PlatformExited

    book = EventBook(fixture_body())
    book.set_abort_check(lambda: "client exited with code 2")
    t0 = time.monotonic()
    with pytest.raises(PlatformExited, match="code 2"):
        book.next_episode(60.0)
    assert time.monotonic() - t0 < 2.0


def _planner_messages_per_decision(tmp_path):
    """Replay the runner's requests through the Agent's real conversation store and prompt builder."""
    from datetime import datetime, timezone

    from ibrobot_agent.contracts import AgentRequest
    from ibrobot_agent.conversation_store import SQLiteConversationStore
    from ibrobot_agent.planner import build_planner_messages

    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=2)
    platform = ScriptedPlatform(
        book,
        body,
        [
            {"reply": events("SUCCEEDED", "moved", skill("move_relative_ee", motion_direction="up")), "after": STATE},
            {"reply": events("SUCCEEDED", "moved", skill("move_relative_ee", motion_direction="up")), "after": STATE},
        ],
    )
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="Pick up the cup.", step_lim=50)
    runner.run_episode(book.next_episode(0.01))
    store = SQLiteConversationStore(tmp_path / "conversation.sqlite3", max_turns=6)
    prompts = []
    try:
        for n, payload in enumerate(platform.payloads):
            request = AgentRequest(
                schema_version=1,
                request_id=payload["request_id"],
                session_id=payload["session_id"],
                channel_id="robodojo_episode",
                principal_id="local_operator",
                robot_scope="robodojo_x5",
                text=payload["text"],
                received_at=datetime.now(timezone.utc),
            )
            store.append(request)  # as AgentService.submit does
            messages = build_planner_messages(
                request=request, context={"messages": store.context(request)}, catalog={"skills": []}
            )
            prompts.append(json.dumps(messages, ensure_ascii=False))
            store.append_assistant(request, f"ASSISTANT-REPLY-{n}")
    finally:
        store.close()
    return platform.payloads, prompts


def test_the_planner_sees_no_earlier_decision_of_the_episode(tmp_path):
    payloads, prompts = _planner_messages_per_decision(tmp_path)
    assert payloads[0]["session_id"] != payloads[1]["session_id"]
    assert "ASSISTANT-REPLY-0" not in prompts[1]
    assert payloads[0]["text"] not in prompts[1] or payloads[0]["text"] == payloads[1]["text"]
    assert prompts[1].count("Pick up the cup.") == prompts[0].count("Pick up the cup.")


def test_an_episode_without_its_own_observations_is_never_run(tmp_path):
    """F3: stale state / missing camera -> no admission, no decision, aborted (not drained)."""
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=5)
    platform = ScriptedPlatform(book, body, [])
    runner.io = platform
    platform.start(wall, "s0001", 1, None, instruction="x", step_lim=10)  # episode starts, no state arrives
    record = runner.run_episode(book.next_episode(0.01))
    assert record["finish_reason"] == "stale_observation" and record["decisions"] == []
    assert platform.admitted == [] and platform.texts == [] and platform.finish_calls == 0 and platform.aborts == 1
    assert {"ee_pose:left", "ee_pose:right", "joint_states", "clock"} <= set(record["stale_sources"])


def test_state_of_an_earlier_session_does_not_count_as_fresh(tmp_path):
    """Simulated stamps do not go back at a reset: older ones, or anything before the new clock, are stale."""
    body = fixture_body()
    runner, book, wall = make_runner(tmp_path, body, max_decisions=1)
    platform = ScriptedPlatform(book, body, [])
    runner.io = platform
    platform.start(wall, "s0002", 2, None)  # starts at simulated 200.0 s; nothing observed yet
    for side in body.sides:  # state of the earlier session (older stamp) arrives late
        book.on_ee(side, (0.0, 0.0, 1.0), (0, 0, 0, 1), stamp=150.0)
    book.on_joints(["x"], [0.0], stamp=150.0)
    book.on_clock(platform.clock(0, "RESETTING"))  # the reset itself is no observation
    assert set(book.stale_sources(platform.key)) >= {"ee_pose:left", "joint_states", "clock"}
    book.on_clock(platform.clock(0, "READY"))  # first observation of s0002
    for side in body.sides:
        book.on_ee(side, (0.0, 0.0, 1.0), (0, 0, 0, 1), stamp=150.0)  # reordered old state: still stale
    book.on_joints(["x"], [0.0], stamp=200.0)  # same stamp as the old session's last: this session's
    assert set(book.stale_sources(platform.key)) == {"ee_pose:left", "ee_pose:right"}
    record = runner.run_episode(book.next_episode(0.01))
    assert record["finish_reason"] == "stale_observation" and platform.admitted == []


def test_state_that_arrived_before_the_sessions_first_clock_is_not_fresh(tmp_path):
    body = fixture_body()
    _runner, book, wall = make_runner(tmp_path, body, max_decisions=1)
    platform = ScriptedPlatform(book, body, [])
    platform.start(wall, "s0003", 3, None)
    stamp = platform.time_base
    for side in body.sides:
        book.on_ee(side, (0.0, 0.0, 1.0), (0, 0, 0, 1), stamp=stamp)
    book.on_joints(["x"], [0.0], stamp=stamp)
    book.on_clock(platform.clock(0, "READY"))
    assert set(book.stale_sources(platform.key)) == {"ee_pose:left", "ee_pose:right", "joint_states"}
    platform.apply(STATE)
    assert book.stale_sources(platform.key) == []


def test_a_missing_planner_camera_blocks_admission(tmp_path):
    from agent_eval.runner import EpisodeRunner

    body = fixture_body()
    runner, _book, wall = make_runner(tmp_path, body, max_decisions=1)
    book = EventBook(body, wall_clock=wall, camera_topics=["/cam/head", "/cam/wrist"])
    runner = EpisodeRunner(book, None, runner.context, runner.settings, sleep=lambda _s: None, log=lambda _t: None)
    platform = ScriptedPlatform(book, body, [])
    runner.io = platform
    platform.start(wall, "s0001", 1, STATE, instruction="x", step_lim=10)
    book.on_camera("/cam/head", platform.time_base)  # wrist camera never delivers a frame of this session
    record = runner.run_episode(book.next_episode(0.01))
    assert record["finish_reason"] == "stale_observation"
    assert record["stale_sources"] == ["camera:/cam/wrist"] and platform.admitted == []
