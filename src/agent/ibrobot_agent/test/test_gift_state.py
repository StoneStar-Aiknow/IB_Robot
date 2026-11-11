"""Gift quota is explicit Agent state; a wake does not erase dialogue or execution facts."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import PlannerOutcome, request_hash_text
from ibrobot_agent.conversation_store import SQLiteConversationStore
from ibrobot_agent.gift_state import GiftStateError, SQLiteGiftStateStore
from ibrobot_agent.planner import build_planner_messages
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService
from ibrobot_agent.voice_bridge import PublishControl, PublishRequest

from .test_service import FakeCatalog, FakeExecution, FakePlanner, RenderedEvents, _request
from .test_voice_bridge import _core, _event, _of

SCOPE = ("so101_single_arm", "test", "p1")


def test_round_survives_restart_and_old_wakes_or_results_cannot_overwrite_it(tmp_path):
    path = tmp_path / "gifts.sqlite3"
    store = SQLiteGiftStateStore(path)
    first = store.reset(SCOPE, "wake-1")
    store.reserve(SCOPE, first["round_id"], "r1")
    store.finish(SCOPE, "wake-1", "r1", "delivered")
    assert store.reset(SCOPE, "wake-1")["status"] == "delivered"
    store.close()
    store = SQLiteGiftStateStore(path)
    try:
        assert store.get(SCOPE)["status"] == "delivered"
        second = store.reset(SCOPE, "wake-2")
        assert second["status"] == "available"
        with pytest.raises(GiftStateError, match="old wake"):
            store.reset(SCOPE, "wake-1")
        store.finish(SCOPE, "wake-1", "r1", "uncertain")
        assert store.get(SCOPE) == second
    finally:
        store.close()


def test_reservation_is_atomic_and_unfinished_delivery_cannot_be_reset(tmp_path):
    path = tmp_path / "gifts.sqlite3"
    one, two = SQLiteGiftStateStore(path), SQLiteGiftStateStore(path)
    one.reset(SCOPE, "wake-1")
    barrier = threading.Barrier(2)

    def reserve(store, request_id):
        barrier.wait()
        try:
            store.reserve(SCOPE, "wake-1", request_id)
            return True
        except GiftStateError:
            return False

    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(reserve, store, name) for store, name in [(one, "r1"), (two, "r2")]]
            assert sorted(future.result(3) for future in futures) == [False, True]
        with pytest.raises(GiftStateError):
            two.reset(SCOPE, "wake-2")
        two.recover_interrupted(SCOPE[0])
        assert one.get(SCOPE)["status"] == "uncertain"
        with pytest.raises(GiftStateError):
            one.reset(SCOPE, "wake-2")
    finally:
        one.close()
        two.close()


def test_wake_only_and_wake_with_request_reset_round_without_replacing_session():
    core = _core(wake_phrases=("欧拉欧拉",), persistent_session=True, gift_reset_on_wake=True)
    wake = _of(core.on_heard("欧拉欧拉", 0), PublishControl)[0].payload["wake_id"]
    first = _of(core.on_heard("拿苹果", 1), PublishRequest)[0].payload
    assert first["gift_wake_id"] == wake
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), 2)
    actions = core.on_heard("欧拉欧拉，拿橙子", 3)
    second = _of(actions, PublishRequest)[0].payload
    assert second["session_id"] == first["session_id"]
    assert second["gift_wake_id"] != wake
    assert _of(actions, PublishControl)[0].payload["wake_id"] == second["gift_wake_id"]


def test_busy_or_partial_or_unaddressed_speech_cannot_reset_a_gift():
    core = _core(wake_phrases=("欧拉欧拉",), persistent_session=True, gift_reset_on_wake=True)
    assert not _of(core.on_partial("欧拉欧拉", 0), PublishControl)
    assert not _of(core.on_heard("有人说欧拉欧拉", 0), PublishControl)
    first = _of(core.on_heard("欧拉欧拉拿苹果", 1), PublishRequest)[0].payload
    assert not _of(core.on_heard("欧拉欧拉", 2), PublishControl)
    assert core._gift_wake_id == first["gift_wake_id"]


def test_repeated_final_segment_reuses_wake_and_unresolved_execution_blocks_new_wake():
    core = _core(wake_phrases=("欧拉欧拉",), persistent_session=True, gift_reset_on_wake=True)
    first = core.on_heard("欧拉欧拉", 0, segment=("asr", 1))
    wake = _of(first, PublishControl)[0].payload["wake_id"]
    assert not _of(core.on_heard("欧拉欧拉", 1, segment=("asr", 1)), PublishControl)
    assert core._gift_wake_id == wake
    core._unsafe_request_id = "unknown-request"
    assert not _of(core.on_heard("欧拉欧拉", 2, segment=("asr", 2)), PublishControl)
    assert core._gift_wake_id == wake


def test_scope_isolation_and_failed_attempt_release(tmp_path):
    store = SQLiteGiftStateStore(tmp_path / "gift.sqlite3")
    try:
        store.reset(SCOPE, "wake-1")
        store.reserve(SCOPE, "wake-1", "r1")
        store.finish(SCOPE, "wake-1", "another-request", "delivered")
        assert store.get(SCOPE)["status"] == "in_progress"
        store.finish(SCOPE, "wake-1", "r1", "available")
        assert store.get(SCOPE)["status"] == "available"
        store.reserve(SCOPE, "wake-1", "r2")
        store.finish(SCOPE, "wake-1", "r2", "delivered")
        assert store.get((SCOPE[0], "other-channel", SCOPE[2]))["status"] == "available"
    finally:
        store.close()


def test_agent_api_reset_updates_voice_round_and_old_acknowledgement_cannot_revert_it():
    core = _core(wake_phrases=("欧拉欧拉",), persistent_session=True, gift_reset_on_wake=True)
    old_wake = _of(core.on_heard("欧拉欧拉", 0), PublishControl)[0].payload["wake_id"]

    def response(wake, revision):
        return {
            "operation": "reset_gift_round",
            "accepted": True,
            "wake_id": wake,
            "gift_state": {
                "round_id": wake,
                "revision": revision,
                "status": "available",
                **{name: getattr(core.config, name) for name in ("robot_scope", "channel_id", "principal_id")},
            },
        }

    core.on_response(response("manual-reset", 5), 1)
    core.on_response(response(old_wake, 1), 2)
    request = _of(core.on_heard("拿苹果", 3), PublishRequest)[0].payload
    assert request["gift_wake_id"] == "manual-reset"


class GiftCatalog(FakeCatalog):
    def get_catalog(self, status):
        catalog = super().get_catalog(status)
        catalog["skills"] += [
            {"name": name, "planner_visible": True, "semantic_level": "skill"}
            for name in ("pick_object", "place_in_container", "return_to_task_pose")
        ]
        return catalog


class Delivery(FakeExecution):
    def __init__(self, *, fail_return=False):
        super().__init__(status="failed" if fail_return else "succeeded")
        self.progress = lambda _: None
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def set_progress_callback(self, callback):
        self.progress = callback

    def execute(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(3)
        self.progress({"current_skill": "place_in_container", "state": "SUCCEEDED"})
        return super().execute(*args, **kwargs)


def wait_done(service, request):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        with service._queue_condition:
            if request.to_key() not in service._queued_or_active:
                return service.get_request(request.to_key())
        time.sleep(0.005)
    pytest.fail("Agent worker did not finish")


def make_service(tmp_path, execution=None):
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="好的",
        summary="送苹果",
        steps=(
            CanonicalWorkflowStep(schema_version=1, skill_name="pick_object", target_name="an apple"),
            CanonicalWorkflowStep(
                schema_version=1, skill_name="place_in_container", target_name="an apple", container_name="bowl"
            ),
        ),
    )
    events, execution = RenderedEvents(), execution or Delivery()
    service = AgentService(
        planner=FakePlanner(outcome),
        catalog=GiftCatalog(),
        store=SQLiteRequestStore(tmp_path / "requests.sqlite3"),
        conversation=SQLiteConversationStore(tmp_path / "conversation.sqlite3"),
        gift_state=SQLiteGiftStateStore(tmp_path / "gift.sqlite3"),
        execution=execution,
        presentation=events.gate,
        event_sink=events,
        execution_enabled=True,
        allowed_skills={"pick_object", "place_in_container"},
    )
    return service, execution


@pytest.mark.parametrize("fail_return", [False, True])
def test_same_conversation_can_get_another_gift_after_wake_even_if_return_failed(tmp_path, fail_return):
    service, execution = make_service(tmp_path, Delivery(fail_return=fail_return))
    try:
        first = replace(_request("r1", text="拿苹果"), gift_wake_id="wake-1")
        assert service.send_message(first).accepted
        wait_done(service, first)
        assert service.get_gift_state(SCOPE)["status"] == "delivered"
        repeated = replace(first, request_id="r2")
        assert service.send_message(repeated).accepted
        assert wait_done(service, repeated).state == "ANSWERED"
        assert execution.calls == 1
        second = replace(first, request_id="r3", gift_wake_id="wake-2")
        assert service.send_message(second).accepted
        wait_done(service, second)
        assert execution.calls == 2
        assert service.get_gift_state(SCOPE)["round_id"] == "wake-2"
        assert len(service._conversation.context(second)) >= 6
        assert request_hash_text(second) != request_hash_text(replace(second, gift_wake_id="wake-3"))
    finally:
        service.close()


def test_agent_reset_interface_rejects_reset_during_execution(tmp_path):
    service, execution = make_service(tmp_path)
    execution.release.clear()
    first = replace(_request("r1", text="拿苹果"), gift_wake_id="wake-1")
    try:
        assert service.send_message(first).accepted
        assert execution.entered.wait(2)
        with pytest.raises(GiftStateError, match="active"):
            service.reset_gift_round(SCOPE, "wake-2")
        assert service.get_gift_state(SCOPE)["round_id"] == "wake-1"
    finally:
        execution.release.set()
        wait_done(service, first)
        service.close()


def test_multiple_pick_pairs_cannot_bypass_one_gift_per_wake(tmp_path):
    service, execution = make_service(tmp_path)
    service._planner.outcome = replace(service._planner.outcome, steps=service._planner.outcome.steps * 2)
    try:
        request = replace(_request("r1", text="拿水果"), gift_wake_id="wake-1")
        assert service.send_message(request).accepted
        assert wait_done(service, request).state == "ANSWERED"
        assert execution.calls == 0
        assert service.get_gift_state(SCOPE)["status"] == "available"
    finally:
        service.close()


def test_unresolved_gift_blocks_delivery_but_allows_conversation(tmp_path):
    service, execution = make_service(tmp_path)
    try:
        state = service.reset_gift_round(SCOPE, "wake-1")
        service._gift_state.reserve(SCOPE, state["round_id"], "interrupted")
        service._gift_state.finish(SCOPE, state["round_id"], "interrupted", "uncertain")
        service._planner.outcome = PlannerOutcome(kind="conversation", user_message="我还在呢。")
        request = replace(_request("question", text="你还在吗"), gift_wake_id="wake-2")
        assert service.send_message(request).accepted
        assert wait_done(service, request).state == "ANSWERED"
        assert execution.calls == 0
        assert service.get_gift_state(SCOPE)["round_id"] == "wake-1"
        assert service.get_gift_state(SCOPE)["status"] == "uncertain"
    finally:
        service.close()


def test_new_round_reaches_planner_with_old_refusal_still_in_chat(tmp_path):
    service, execution = make_service(tmp_path)
    contexts = []
    original_plan = service._planner.plan

    def plan(request, context, catalog, cancel_token, *, on_preview=None):
        messages = build_planner_messages(request=request, context=context, catalog=catalog)
        contexts.append(json.loads(messages[-1]["content"])["conversation"])
        return original_plan(request, context, catalog, cancel_token, on_preview=on_preview)

    service._planner.plan = plan
    try:
        first = replace(_request("r1", text="拿苹果"), gift_wake_id="wake-1")
        service.send_message(first)
        wait_done(service, first)
        repeated = replace(first, request_id="r2", text="再给我一个苹果")
        service.send_message(repeated)
        assert wait_done(service, repeated).state == "ANSWERED"
        second = replace(repeated, request_id="r3", gift_wake_id="wake-2")
        service.send_message(second)
        assert wait_done(service, second).state == "SUCCEEDED"
        context = contexts[-1]
        assert context["gift_state"]["round_id"] == "wake-2"
        assert context["gift_state"]["status"] == "available"
        assert context["gift_state"]["already_given_message"] == "这轮已经送过礼物啦，重新叫欧拉欧拉可以开始新一轮。"
        assert any("重新叫欧拉欧拉" in row["content"] for row in context["messages"])
        assert execution.calls == 2
    finally:
        service.close()
