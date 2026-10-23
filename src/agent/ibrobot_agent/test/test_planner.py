from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from ibrobot_agent.contracts import AgentRequest, PlannerIdentity
from ibrobot_agent.planner import (
    PlannerAdapter,
    PlannerParseError,
    RulePlanner,
    build_planner_messages,
    parse_planner_outcome,
    read_only_outcome,
)


def _request(text: str) -> AgentRequest:
    return AgentRequest(
        schema_version=1,
        request_id="request-1",
        session_id="session-1",
        channel_id="test",
        principal_id="operator",
        robot_scope="so101_single_arm",
        text=text,
        received_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize("text", ["不要挥手", "你会挥手吗？", "如果让你点头会怎样？"])
def test_rule_planner_does_not_promote_negative_or_hypothetical_text(text):
    outcome = RulePlanner().plan(_request(text), {}, {}, threading.Event())

    assert outcome.kind == "conversation"
    assert not outcome.steps


def test_rule_planner_preserves_ordered_so101_workflow():
    outcome = RulePlanner().plan(_request("先回安全位，然后点头"), {}, {}, threading.Event())

    assert outcome.kind == "workflow"
    assert [step.skill_name for step in outcome.steps] == ["recover_safe_pose", "nod_yes"]


@pytest.mark.parametrize("text", ["please nod", "不同意", "再见"])
def test_rule_planner_does_not_match_aliases_inside_other_words(text):
    outcome = RulePlanner().plan(_request(text), {}, {}, threading.Event())

    if text == "please nod":
        assert [step.skill_name for step in outcome.steps] == ["nod_yes"]
    elif text == "不同意":
        assert [step.skill_name for step in outcome.steps] == ["shake_no"]
    else:
        assert outcome.kind == "rejected"


def test_rule_planner_asks_for_a_specific_skill():
    outcome = RulePlanner().plan(_request("帮我做一个动作"), {}, {}, threading.Event())

    assert outcome.kind == "needs_clarification"
    assert outcome.missing_fields == ("skill_name",)


@pytest.mark.parametrize(
    "text, query_kind",
    [("当前状态如何？", "status"), ("list skills", "list_skills"), ("有哪些姿态", "list_poses")],
)
def test_rule_planner_and_read_only_router_agree(text, query_kind):
    outcome = read_only_outcome(text)
    assert outcome.query_kind == query_kind
    assert RulePlanner().plan(_request(text), {}, {}, threading.Event()) == outcome


@pytest.mark.parametrize("text", ["不要查询当前状态", "如果查询当前状态会怎样？", "do not list skills"])
def test_read_only_router_preserves_negation_and_hypothetical_guards(text):
    assert read_only_outcome(text) is None
    assert RulePlanner().plan(_request(text), {}, {}, threading.Event()).kind == "conversation"


def test_parser_rejects_duplicate_json_keys():
    with pytest.raises(PlannerParseError, match="duplicate key"):
        parse_planner_outcome('{"kind":"conversation","kind":"workflow","user_message":"x"}')


@pytest.mark.parametrize(
    "raw",
    [
        '{"kind":"conversation","user_message":"x","endpoint":"/unsafe"}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","endpoint":"/unsafe"}]}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","timeout_sec":"1.0"}]}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","timeout_sec":true}]}',
        '{"kind":"workflow","user_message":"x","summary":"x","steps":[{"schema_version":1,"skill_name":"wave_hello","timeout_sec":NaN}]}',
    ],
)
def test_parser_rejects_unknown_fields_and_coercible_or_non_finite_numbers(raw):
    with pytest.raises(PlannerParseError):
        parse_planner_outcome(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"kind":"conversation","user_message":"x"}\n```',
        'prefix {"kind":"conversation","user_message":"x"}',
        '{"kind":"conversation","user_message":"x"} suffix',
    ],
)
def test_parser_rejects_wrappers_and_mixed_text(raw):
    with pytest.raises(PlannerParseError):
        parse_planner_outcome(raw)


class FakeModelClient:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.options = None
        self.stream_requested = None
        self.delta_observer = None

    def complete(
        self,
        messages,
        *,
        force_json,
        enable_thinking=True,
        extra_body=None,
        max_tokens=None,
        stream=False,
        on_content_delta=None,
    ):
        self.messages = messages
        self.options = {
            "enable_thinking": enable_thinking,
            "extra_body": extra_body,
            "max_tokens": max_tokens,
        }
        self.stream_requested = stream
        self.delta_observer = on_content_delta
        assert force_json is True
        return self.response


def _planner_adapter(response):
    client = FakeModelClient(response)
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kwargs: [{"role": "user", "content": kwargs["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
    )
    return planner, client


def test_planner_adapter_parses_strict_json_response():
    planner, client = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"你好"}',
            "tool_calls": [],
        }
    )

    outcome = planner.plan(_request("你好"), {}, {}, threading.Event())

    assert outcome.kind == "conversation"
    assert client.messages == [{"role": "user", "content": "你好"}]


def test_planner_adapter_rejects_tool_calls():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"你好"}',
            "tool_calls": [{"function": {"name": "unsafe"}}],
        }
    )

    with pytest.raises(PlannerParseError, match="must not return tool calls"):
        planner.plan(_request("你好"), {}, {}, threading.Event())


def test_planner_adapter_builds_prompt_with_conversation_and_catalog():
    client = FakeModelClient(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"普通回答"}',
            "tool_calls": [],
        }
    )
    planner = PlannerAdapter(
        client,
        prompt_builder=build_planner_messages,
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
    )
    planner.plan(
        _request("这是什么"),
        {"messages": [{"role": "user", "content": "上一轮"}]},
        {"skills": [{"name": "nod_yes"}]},
        threading.Event(),
    )

    assert "上一轮" in client.messages[0]["content"] or "上一轮" in client.messages[1]["content"]


def test_planner_adapter_supports_ordinary_knowledge_answer():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"我是小智。"}',
            "tool_calls": [],
        }
    )

    outcome = planner.plan(_request("你的名字是什么？"), {}, {}, threading.Event())

    assert outcome.kind == "conversation"
    assert outcome.user_message == "我是小智。"


def test_planner_adapter_rejects_execution_fields_on_conversation():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"普通回答","summary":"误用"}',
            "tool_calls": [],
        }
    )

    with pytest.raises(PlannerParseError, match="conversation outcomes cannot carry a summary"):
        planner.plan(_request("这是什么"), {}, {}, threading.Event())


def test_planner_adapter_forwards_the_configured_latency_options():
    """The knobs that decide how long one turn takes must reach the client.

    They used to be unreachable: ``plan()`` called ``complete`` with only
    ``force_json``, so configuration could not turn thinking off or bound the
    answer no matter what the YAML said.
    """
    client = FakeModelClient(
        {"status": "ok", "content": '{"kind":"conversation","user_message":"你好"}', "tool_calls": []}
    )
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kwargs: [{"role": "user", "content": kwargs["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
        enable_thinking=False,
        max_tokens=256,
        extra_body={"thinking": {"type": "disabled"}},
    )

    planner.plan(_request("你好"), {}, {}, threading.Event())

    assert client.options == {
        "enable_thinking": False,
        "extra_body": {"thinking": {"type": "disabled"}},
        "max_tokens": 256,
    }


def test_planner_adapter_defaults_leave_the_request_unchanged():
    """Without configuration the request must look exactly as it did before."""
    planner, client = _planner_adapter(
        {"status": "ok", "content": '{"kind":"conversation","user_message":"你好"}', "tool_calls": []}
    )

    planner.plan(_request("你好"), {}, {}, threading.Event())

    assert client.options == {"enable_thinking": True, "extra_body": None, "max_tokens": None}


def test_planner_adapter_reports_the_cost_of_a_turn():
    observed = []
    client = FakeModelClient(
        {
            "status": "ok",
            "content": '{"kind":"conversation","user_message":"你好"}',
            "tool_calls": [],
            "timing_ms": 2871,
            "usage": {"prompt_tokens": 4096, "completion_tokens": 37, "total_tokens": 4133},
            "reasoning": "x" * 900,
        }
    )
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kwargs: [{"role": "user", "content": kwargs["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
        observer=observed.append,
    )

    planner.plan(_request("你好"), {}, {}, threading.Event())

    assert observed == [
        {
            "status": "ok",
            "timing_ms": 2871,
            "prompt_tokens": 4096,
            "completion_tokens": 37,
            "reasoning_chars": 900,
        }
    ]


def test_planner_adapter_reports_a_failed_turn_too():
    """A timeout is the slowest outcome there is; it must still be recorded."""
    observed = []
    client = FakeModelClient({"status": "error", "error": "VLM API timeout after 20.0s", "timing_ms": 20001})
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kwargs: [{"role": "user", "content": kwargs["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
        observer=observed.append,
    )

    with pytest.raises(PlannerParseError):
        planner.plan(_request("你好"), {}, {}, threading.Event())

    assert observed[0]["status"] == "error"
    assert observed[0]["timing_ms"] == 20001


def test_planner_adapter_survives_a_broken_observer():
    """Observability must never be able to fail a plan."""

    def explode(_metrics):
        raise RuntimeError("logging backend down")

    client = FakeModelClient(
        {"status": "ok", "content": '{"kind":"conversation","user_message":"你好"}', "tool_calls": []}
    )
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kwargs: [{"role": "user", "content": kwargs["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
        observer=explode,
    )

    assert planner.plan(_request("你好"), {}, {}, threading.Event()).kind == "conversation"


def test_prompt_names_every_legal_reason_code():
    """A rejection with an invented reason_code loses the whole turn.

    ``REASON_CODE_VALUES`` is a closed set of three strings, and the prompt used
    to list ``reason_code`` only as a permitted field name. A model that
    correctly decided to refuse then had to guess the code, and any other
    value raised SKILL_SCHEMA_INVALID -- observed on the X2 as a refusal
    surfacing as FAILED. Keep the prompt and the contract tied together.
    """
    from ibrobot_agent.contracts import REASON_CODE_VALUES

    system = build_planner_messages(request=_request("点个头"), context={"messages": []}, catalog={"skills": []})[0][
        "content"
    ]

    for value in REASON_CODE_VALUES:
        assert value in system, f"prompt does not mention reason_code {value}"


def test_rejection_with_an_invented_reason_code_is_still_refused_locally():
    """The prompt is guidance; the contract stays the authority."""
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": '{"kind":"rejected","user_message":"不行","reason_code":"motion_not_authorized"}',
            "tool_calls": [],
        }
    )

    with pytest.raises(PlannerParseError, match="reason_code is invalid"):
        planner.plan(_request("点个头"), {}, {}, threading.Event())


@pytest.mark.parametrize("reason_code", sorted({"UNSUPPORTED_TASK", "OUT_OF_SCOPE", "SAFETY_RESTRICTED"}))
def test_each_documented_reason_code_parses(reason_code):
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": f'{{"kind":"rejected","user_message":"不行","reason_code":"{reason_code}"}}',
            "tool_calls": [],
        }
    )

    outcome = planner.plan(_request("点个头"), {}, {}, threading.Event())

    assert outcome.kind == "rejected"
    assert outcome.reason_code == reason_code


_NOD_CATALOG = {
    "robot_name": "so101",
    "skills": [
        {"name": "nod_yes", "planner_visible": True, "summary": "点头", "schema_version": 2},
        {"name": "internal_probe", "planner_visible": False, "summary": "内部"},
    ],
}
_COMMANDS = {"点头": "nod_yes", "点个头": "nod_yes", "内部探测": "internal_probe"}


def test_direct_command_skips_planning_for_an_exact_phrase():
    """A phrase the deployment declared unambiguous must not cost a model call."""
    from ibrobot_agent.planner import direct_command_outcome

    outcome = direct_command_outcome("点个头", _COMMANDS, _NOD_CATALOG)

    assert outcome is not None
    assert outcome.kind == "workflow"
    assert [step.skill_name for step in outcome.steps] == ["nod_yes"]
    # The step must carry the schema version the live catalog declares, not a guess.
    assert [step.schema_version for step in outcome.steps] == [2]


def test_direct_command_reply_repeats_the_phrase_not_the_skill_identifier():
    """A speech channel cannot say an identifier such as nod_yes; the phrase is the user's own words."""
    from ibrobot_agent.planner import direct_command_outcome

    outcome = direct_command_outcome("点个头", _COMMANDS, _NOD_CATALOG)

    assert outcome.user_message == "好的，点个头。"


def test_direct_command_ignores_surrounding_text():
    """Substring matching would fire the robot on a sentence that only mentions it."""
    from ibrobot_agent.planner import direct_command_outcome

    for text in ("先别点头", "如果我说点头你会怎么做", "点头之前先回安全姿态", "我刚才说的是点头吗"):
        assert direct_command_outcome(text, _COMMANDS, _NOD_CATALOG) is None, text


@pytest.mark.parametrize("text", ["不要点头", "别点头", "don't 点头", "假如点头"])
def test_direct_command_honours_the_negation_guards(text):
    from ibrobot_agent.planner import direct_command_outcome

    assert direct_command_outcome(text, _COMMANDS, _NOD_CATALOG) is None


def test_direct_command_requires_the_skill_to_be_in_the_live_catalog():
    """A stale mapping must degrade to planning, not name a skill the Gateway rejects."""
    from ibrobot_agent.planner import direct_command_outcome

    assert direct_command_outcome("点头", {"点头": "retired_skill"}, _NOD_CATALOG) is None


def test_direct_command_refuses_a_skill_the_planner_may_not_choose():
    from ibrobot_agent.planner import direct_command_outcome

    assert direct_command_outcome("内部探测", _COMMANDS, _NOD_CATALOG) is None


def test_direct_command_is_off_without_configuration():
    """No phrase reaches the robot without the planner unless opted in."""
    from ibrobot_agent.planner import direct_command_outcome

    assert direct_command_outcome("点个头", {}, _NOD_CATALOG) is None


def test_direct_command_tolerates_whitespace_and_case():
    from ibrobot_agent.planner import direct_command_outcome

    assert direct_command_outcome("  点个头  ", _COMMANDS, _NOD_CATALOG) is not None
    assert direct_command_outcome("NOD", {"nod": "nod_yes"}, _NOD_CATALOG) is not None


def test_direct_command_survives_a_catalog_without_skills():
    from ibrobot_agent.planner import direct_command_outcome

    assert direct_command_outcome("点头", _COMMANDS, {}) is None
    assert direct_command_outcome("点头", _COMMANDS, {"skills": "not a list"}) is None


def test_prompt_names_the_legal_workflow_step_fields():
    """A step carrying an invented field loses the whole turn.

    Observed against the live planner: it answered a two-skill request with a
    nested ``parameters`` object, which `validate_workflow_steps` rejects, so a
    correct plan surfaced as SKILL_SCHEMA_INVALID. The prompt said only "flat
    WorkflowStep objects", which does not tell the model what flat means.
    """
    from embodied_common.workflow_contracts import _WORKFLOW_STEP_COMMON_FIELDS

    system = build_planner_messages(
        request=_request("张开夹爪然后回到安全姿态"), context={"messages": []}, catalog={"skills": []}
    )[0]["content"]

    for field in _WORKFLOW_STEP_COMMON_FIELDS:
        assert field in system, f"prompt does not mention step field {field}"
    assert "parameters" in system


def test_prompt_lets_the_model_choose_every_skill_the_planner_view_lists():
    """The planner view drops the ``planner_visible`` flag along with the hidden skills.

    A prompt that asks for skills *marked* planner_visible then describes a field
    no entry carries, and a literal model refuses every request.
    """
    from robot_skill_cli.catalog import planner_catalog_view

    full = {
        "skills": [
            {"name": "nod_yes", "planner_visible": True, "summary": "点头", "schema_version": 1},
            {"name": "internal_probe", "planner_visible": False, "summary": "内部"},
        ]
    }
    view = planner_catalog_view(full)
    system = build_planner_messages(request=_request("点头"), context={"messages": []}, catalog=view)[0]["content"]

    assert [skill["name"] for skill in view["skills"]] == ["nod_yes"]
    assert all("planner_visible" not in skill for skill in view["skills"])
    assert "never one marked planner_visible=false" in system
    assert "use only planner_visible" not in system


def test_prompt_example_reply_is_speakable():
    """``user_message`` is said aloud, and a speech front end cannot say a skill identifier."""
    import json
    import re

    system = build_planner_messages(request=_request("点头"), context={"messages": []}, catalog={"skills": []})[0][
        "content"
    ]
    example = json.loads(re.search(r"Example workflow: (\{.*?\}\]\})", system).group(1))

    assert example["kind"] == "workflow"
    assert not re.search(r"[A-Za-z_]", example["user_message"])


def test_a_step_with_a_nested_parameters_object_is_still_refused_locally():
    planner, _ = _planner_adapter(
        {
            "status": "ok",
            "content": (
                '{"kind":"workflow","user_message":"好的。","summary":"nod",'
                '"steps":[{"schema_version":1,"skill_name":"nod_yes","parameters":{}}]}'
            ),
            "tool_calls": [],
        }
    )

    with pytest.raises(PlannerParseError, match="unknown fields: parameters"):
        planner.plan(_request("点头"), {}, {}, threading.Event())


class StreamingModelClient:
    """Model client that replays a canned answer as streamed deltas."""

    def __init__(self, content, *, chunk=8, status="ok"):
        self.content = content
        self.chunk = chunk
        self.status = status
        self.stream_requested = None
        self.deltas_sent = 0

    def complete(
        self,
        messages,
        *,
        force_json,
        enable_thinking=True,
        extra_body=None,
        max_tokens=None,
        stream=False,
        on_content_delta=None,
    ):
        self.stream_requested = stream
        if stream and on_content_delta is not None:
            for end in range(self.chunk, len(self.content) + self.chunk, self.chunk):
                self.deltas_sent += 1
                on_content_delta(self.content[:end])
        return {"status": self.status, "content": self.content, "tool_calls": []}


_WORKFLOW = (
    '{"kind":"workflow","user_message":"准备执行：挥手。","summary":"wave_hello",'
    '"steps":[{"schema_version":1,"skill_name":"wave_hello"}]}'
)


def _streaming_planner(content, **kwargs):
    client = StreamingModelClient(content)
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kw: [{"role": "user", "content": kw["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
        stream=True,
        **kwargs,
    )
    return planner, client


def test_streaming_announces_user_message_before_the_plan_is_complete():
    """The whole point: speak the short field while steps are still arriving."""
    planner, client = _streaming_planner(_WORKFLOW)
    previews = []

    outcome = planner.plan(_request("挥个手"), {}, {}, threading.Event(), on_preview=previews.append)

    assert client.stream_requested is True
    assert previews == ["准备执行：挥手。"]
    assert outcome.kind == "workflow"
    assert [step.skill_name for step in outcome.steps] == ["wave_hello"]


def test_streaming_announces_once_even_though_deltas_keep_arriving():
    planner, client = _streaming_planner(_WORKFLOW)
    previews = []

    planner.plan(_request("挥个手"), {}, {}, threading.Event(), on_preview=previews.append)

    assert client.deltas_sent > 1, "fixture did not actually stream"
    assert len(previews) == 1


def test_the_complete_plan_is_still_validated_after_speaking_early():
    """Speech moving early must not let an invalid plan through.

    This is the contract that makes early speech acceptable: only the
    announcement is early, never the execution.
    """
    bad = (
        '{"kind":"workflow","user_message":"准备执行：挥手。","summary":"wave",'
        '"steps":[{"schema_version":1,"skill_name":"wave_hello","parameters":{}}]}'
    )
    planner, _client = _streaming_planner(bad)
    previews = []

    with pytest.raises(PlannerParseError, match="unknown fields: parameters"):
        planner.plan(_request("挥个手"), {}, {}, threading.Event(), on_preview=previews.append)

    # It was previewed, and the plan was still refused -- the documented trade-off.
    assert previews == ["准备执行：挥手。"]


def test_a_cancelled_turn_does_not_speak():
    planner, _client = _streaming_planner(_WORKFLOW)
    cancelled = threading.Event()
    cancelled.set()
    previews = []

    with pytest.raises(PlannerParseError, match="cancelled"):
        planner.plan(_request("挥个手"), {}, {}, cancelled, on_preview=previews.append)

    assert previews == []


def test_an_empty_user_message_is_not_announced():
    planner, _client = _streaming_planner('{"kind":"conversation","user_message":"   "}')
    previews = []

    with pytest.raises(PlannerParseError):
        planner.plan(_request("你好"), {}, {}, threading.Event(), on_preview=previews.append)

    assert previews == []


def test_a_read_only_answer_is_not_previewed():
    """Its reply is rebuilt from the Gateway's facts after parsing; speaking
    the model's own text first could state a status the robot then
    contradicts."""
    planner, _client = _streaming_planner(
        '{"kind":"read_only","query_kind":"status","user_message":"运动已授权，可以执行。"}'
    )
    previews = []

    outcome = planner.plan(_request("现在能动吗"), {}, {}, threading.Event(), on_preview=previews.append)

    assert outcome.kind == "read_only"
    assert previews == []


def test_a_reply_waits_for_its_kind_before_it_is_previewed():
    """With user_message first, the reply is held until kind shows it is not
    read_only, then handed over once."""
    planner, client = _streaming_planner('{"user_message":"你好，我是小智。","kind":"conversation"}')
    previews = []

    outcome = planner.plan(_request("你是谁"), {}, {}, threading.Event(), on_preview=previews.append)

    assert client.deltas_sent > 1, "fixture did not actually stream"
    assert outcome.kind == "conversation"
    assert previews == ["你好，我是小智。"]


def test_a_broken_preview_callback_does_not_fail_the_plan():
    """A preview is an optimization; planning is the job."""
    planner, _client = _streaming_planner(_WORKFLOW)

    def broken(_text):
        raise RuntimeError("event transport down")

    outcome = planner.plan(_request("挥个手"), {}, {}, threading.Event(), on_preview=broken)

    assert outcome.kind == "workflow"


def test_without_a_preview_callback_no_delta_observer_is_passed():
    """A turn nobody listens to must not pay for incremental scanning."""
    planner, client = _streaming_planner(_WORKFLOW)

    planner.plan(_request("挥个手"), {}, {}, threading.Event())

    assert client.stream_requested is True
    assert client.deltas_sent == 0


def test_without_stream_no_delta_callback_is_passed():
    """The non-streaming path must stay exactly as it was."""
    client = StreamingModelClient(_WORKFLOW)
    planner = PlannerAdapter(
        client,
        prompt_builder=lambda **kw: [{"role": "user", "content": kw["request"].text}],
        planner_identity=PlannerIdentity("vlm", "fake-model", "json", "1", "1", "digest"),
    )

    planner.plan(_request("挥个手"), {}, {}, threading.Event())

    assert client.stream_requested is False
    assert client.deltas_sent == 0
