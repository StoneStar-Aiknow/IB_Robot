"""Strict planner-response parsing for ibrobot_agent.

The planner layer accepts only a single JSON object and turns it into the
typed ``PlannerOutcome`` contract. Anything else fails closed.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from embodied_common.incremental_json import first_complete_string_field
from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import PlannerIdentity, PlannerOutcome, planner_outcome_from_mapping

_NEGATION_MARKERS = ("不要", "别", "不想", "do not", "don't")
_HYPOTHETICAL_MARKERS = ("如果", "假如", "假设")
_READ_ONLY_QUERIES = (
    ("status", ("当前状态", "状态怎么样", "状态如何", "status"), "我来查看当前机器人状态。"),
    (
        "list_skills",
        ("有什么能力", "哪些能力", "有哪些技能", "技能列表", "能力列表", "list skills"),
        "我来查看当前机器人技能。",
    ),
    ("list_poses", ("有哪些姿态", "命名姿态", "姿态列表", "list poses"), "我来查看当前机器人姿态。"),
)


def read_only_outcome(text: str) -> PlannerOutcome | None:
    """Recognize factual queries before planning, excluding negated or hypothetical requests."""
    folded = text.casefold()
    if any(token in folded for token in (*_NEGATION_MARKERS, *_HYPOTHETICAL_MARKERS)):
        return None
    for query_kind, tokens, message in _READ_ONLY_QUERIES:
        if any(token in folded for token in tokens):
            return PlannerOutcome(kind="read_only", user_message=message, query_kind=query_kind)
    return None


def direct_command_outcome(text: str, commands: Mapping[str, str], catalog: Mapping[str, Any]) -> PlannerOutcome | None:
    """Turn an exactly-matching known phrase straight into a one-step workflow.

    A planning turn is one non-streaming model call, measured at a 2.7 s median
    and a tail past 10 s on the configured provider. For the handful of phrases
    a robot hears constantly ("点头", "停"), that entire wait buys a decision
    that was never in doubt, so the mapping is resolved locally instead.

    Deliberately narrow, because this is the one path that reaches the robot
    without the model seeing the request:

    * the phrase must match the whole utterance after normalization, never a
      substring -- "先别点头" must not fire the nod;
    * the same negation and hypothetical guards as the read-only router apply;
    * the skill must exist in the live catalog and be planner-visible, so a
      stale mapping degrades to normal planning instead of naming a skill the
      Gateway would reject; and
    * the mapping comes from configuration. There are no built-in phrases: which
      utterances are unambiguous is a property of a deployment, not of the code.
    """
    if not commands:
        return None
    folded = text.strip().casefold()
    if not folded:
        return None
    if any(token in folded for token in (*_NEGATION_MARKERS, *_HYPOTHETICAL_MARKERS)):
        return None
    phrase, skill_name = next(
        ((phrase, name) for phrase, name in commands.items() if phrase.strip().casefold() == folded),
        ("", None),
    )
    if skill_name is None:
        return None
    skills = catalog.get("skills")
    if not isinstance(skills, Sequence):
        return None
    entry = next(
        (
            skill
            for skill in skills
            if isinstance(skill, Mapping) and skill.get("name") == skill_name and skill.get("planner_visible", True)
        ),
        None,
    )
    if entry is None:
        return None
    schema_version = entry.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        schema_version = 1
    return planner_outcome_from_payload(
        {
            "kind": "workflow",
            # The configured phrase, not the skill name: it is the user's own
            # words, and a speech channel cannot say an identifier.
            "user_message": f"好的，{phrase.strip()}。",
            "summary": skill_name,
            "steps": [{"schema_version": schema_version, "skill_name": skill_name}],
        }
    )


class PlannerParseError(ValueError):
    """Raised when a model response cannot be parsed into a planner outcome."""

    code = "SKILL_SCHEMA_INVALID"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlannerParseError(f"planner response contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise PlannerParseError(f"planner response contains non-finite number: {value}")


def parse_planner_outcome(raw_text: str, *, steps: Sequence[Any] | None = None) -> PlannerOutcome:
    """Parse one JSON response blob into a validated PlannerOutcome."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise PlannerParseError("planner response must be a non-empty JSON object")
    if raw_text != raw_text.strip() or not raw_text.startswith("{") or not raw_text.endswith("}"):
        raise PlannerParseError("planner response must contain exactly one JSON object")
    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise PlannerParseError("planner response is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise PlannerParseError("planner response must be a JSON object")
    try:
        return planner_outcome_from_mapping(payload, steps=steps)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerParseError(str(exc)) from exc


def planner_outcome_from_payload(payload: Mapping[str, Any], *, steps: Sequence[Any] | None = None) -> PlannerOutcome:
    """Convert a pre-parsed payload mapping into a validated PlannerOutcome."""
    try:
        return planner_outcome_from_mapping(payload, steps=steps)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerParseError(str(exc)) from exc


class PlannerAdapter:
    """Adapt a model client returning ``complete()`` dictionaries to Planner."""

    def __init__(
        self,
        model_client: Any,
        *,
        prompt_builder: Any,
        planner_identity: Any,
        enable_thinking: bool = True,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        observer: Any = None,
        stream: bool = False,
    ) -> None:
        self._model_client = model_client
        self._prompt_builder = prompt_builder
        self.identity = planner_identity
        # Latency-relevant request options. They are configuration, not
        # defaults chosen here: which of them a given provider honours, and how
        # it spells "do not think", differs per provider.
        self._enable_thinking = enable_thinking
        self._max_tokens = max_tokens
        self._extra_body = dict(extra_body or {})
        # Called with the model's own accounting for one planning turn. The
        # whole user-visible planning delay is this one call, and without it
        # there is no way to tell a slow network from a model that is thinking.
        self._observer = observer
        # Streaming is only useful together with a per-turn on_preview: the
        # point is to hand over the one short field the user can already be
        # told while the rest of the plan is still being generated. The plan
        # itself is still validated in full, from the complete document.
        self._stream = stream

    def plan(
        self,
        request: Any,
        context: Mapping[str, Any],
        catalog: Mapping[str, Any],
        cancel_token: Any,
        *,
        on_preview: Callable[[str], None] | None = None,
    ) -> PlannerOutcome:
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        messages = self._prompt_builder(request=request, context=context, catalog=catalog)
        response = self._model_client.complete(
            messages,
            force_json=True,
            enable_thinking=self._enable_thinking,
            extra_body=self._extra_body or None,
            max_tokens=self._max_tokens,
            stream=self._stream,
            on_content_delta=(
                _preview_watcher(cancel_token, on_preview) if self._stream and on_preview is not None else None
            ),
        )
        self._report(response)
        if not isinstance(response, Mapping) or response.get("status") != "ok":
            detail = (
                response.get("error", "unknown model error")
                if isinstance(response, Mapping)
                else "invalid model response"
            )
            raise PlannerParseError(f"planner model call failed: {detail}")
        if response.get("tool_calls"):
            raise PlannerParseError("planner model must not return tool calls")
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        content = response.get("content")
        if not isinstance(content, str):
            raise PlannerParseError("planner model content must be a string")
        return parse_planner_outcome(content)

    def _report(self, response: Any) -> None:
        """Hand the turn's cost to the observer, never failing the plan for it."""
        if self._observer is None or not isinstance(response, Mapping):
            return
        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        reasoning = response.get("reasoning")
        try:
            self._observer(
                {
                    "status": response.get("status"),
                    "timing_ms": response.get("timing_ms"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    # A large reasoning payload behind a short answer is the
                    # signature of thinking that was never turned off.
                    "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                }
            )
        except Exception:
            return


#: Outcome kinds whose ``user_message`` is the reply as given. ``read_only`` is
#: not one: its reply is rebuilt from the Gateway after the plan is parsed.
_PREVIEWABLE_KINDS = frozenset({"conversation", "needs_clarification", "rejected", "workflow"})


def _preview_watcher(cancel_token: Any, on_preview: Callable[[str], None]) -> Callable[[str], None]:
    """Return a delta observer that hands ``user_message`` over once.

    The field is previewed as soon as it closes, which is ahead of the rest of
    the document -- that gap is the whole point of streaming here. What the
    caller does with it is the caller's business: the Agent publishes it as a
    ``reply_preview`` event and the channel decides whether to say it. The
    trade-off is real and deliberate: a channel that speaks it commits to the
    reply before the plan has been validated, so a plan that then fails
    validation leaves it having announced an action that will not happen.
    Nothing is executed early -- only the reply moves.

    The reply is only handed over once ``kind`` has arrived too, and never for
    ``read_only``: that answer is replaced by the Gateway's own facts, so the
    model's text could state a status ("运动已授权") the robot then contradicts.
    The other kinds reply with their ``user_message`` unchanged.
    """
    announced = False

    def watch(partial: str) -> None:
        nonlocal announced
        if announced or cancel_token.is_set():
            return
        kind = first_complete_string_field(partial, "kind")
        if kind is None:
            return
        if kind not in _PREVIEWABLE_KINDS:
            announced = True
            return
        value = first_complete_string_field(partial, "user_message")
        if not value or not value.strip():
            return
        announced = True
        # A preview is an optimization; planning is the job.
        with contextlib.suppress(Exception):
            on_preview(value)

    return watch


def build_planner_messages(
    *, request: Any, context: Mapping[str, Any], catalog: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build a channel-neutral prompt whose output is still locally validated.

    The prompt must name the legal ``reason_code`` values. They are a closed set
    of three strings, and a rejection carrying anything else fails the whole
    turn as SKILL_SCHEMA_INVALID -- so a model that correctly decided to refuse
    produced an error instead of a refusal.
    """
    system = (
        "You are the IB-Robot planner. Return exactly one JSON object and no markdown. "
        "Allowed kinds: conversation, read_only, needs_clarification, rejected, workflow. "
        "Never invent skills, parameters, endpoints, authorization, or execution tokens. "
        "For workflow, use only skills listed in CATALOG, never one marked planner_visible=false, "
        "and emit flat WorkflowStep objects. "
        "Questions, hypotheticals, quotations, and negated commands must not become workflows. "
        "For ordinary knowledge or identity questions, use kind=conversation and put the answer in user_message. "
        "The only permitted top-level fields are kind, user_message, query_kind, skill_name, missing_fields, "
        "reason_code, summary, and steps. Never return a field named text. "
        "For kind=rejected, reason_code must be exactly one of UNSUPPORTED_TASK, OUT_OF_SCOPE or "
        "SAFETY_RESTRICTED; any other value discards the whole answer. "
        "user_message is said aloud: write it as natural speech in the user's language, never a skill name "
        "or other identifier. Keep user_message <= 150 characters "
        "and summary <= 300 characters. For conversation/read_only/rejected, omit fields that do not belong "
        "to that kind (especially summary and steps). For every workflow step, schema_version is a JSON "
        "integer, never a string, float, or omitted: copy the schema_version of that skill's CATALOG entry, and "
        "use 1 when the entry has none. A step may contain only schema_version, skill_name, target_name, "
        "container_name, place_name, arm_side, motion_direction, motion_distance, imitation_duration_sec "
        "and timeout_sec; never nest a parameters object. Example conversation: "
        '{"kind":"conversation","user_message":"简短回答"}. Example workflow: '
        '{"kind":"workflow","user_message":"好的，点个头。","summary":"nod_yes",'
        '"steps":[{"schema_version":1,"skill_name":"nod_yes"}]}. '
        "If conversation.continuation is present, request.text is an interrupted task the user asked to continue: "
        "plan only its unfinished part, never repeating conversation.continuation.completed_steps and planning "
        "conversation.continuation.interrupted_step again, because it may have stopped half way."
    )
    payload = {
        "request": {"text": request.text, "reply_to_request_id": request.reply_to_request_id},
        "conversation": context,
        "catalog": catalog,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


class RulePlanner:
    """Deterministic SO-101 planner used for incubation and offline E2E tests.

    This is deliberately small and conservative.  It provides a usable natural
    language path without making a live model a prerequisite for validating the
    node and can later be replaced by :class:`PlannerAdapter` behind the same
    ``Planner`` protocol.
    """

    identity = PlannerIdentity(
        route="so101_rule_incubator",
        returned_model="deterministic-rule-planner",
        protocol="local-rule-v1",
        prompt_version="none",
        schema_version="1",
        config_digest="builtin-so101-v1",
    )

    _RULES = (
        (("挥手", "挥挥手", "打招呼", "wave", "hello"), "wave_hello"),
        (("点头", "同意", "nod", "yes"), "nod_yes"),
        (("摇头", "不同意", "shake", "no"), "shake_no"),
        (("打开夹爪", "张开夹爪", "开爪", "open gripper"), "open_gripper_skill"),
        (("关闭夹爪", "合上夹爪", "关爪", "close gripper"), "close_gripper_skill"),
        (("回安全位", "回到安全位", "回 home", "回home", "safe pose"), "recover_safe_pose"),
        (("回零位", "回到零位", "zero pose"), "recover_zero_pose"),
    )
    _ASCII_WORD = re.compile(r"[a-z]+")

    @classmethod
    def _alias_hit(cls, alias: str, text: str) -> bool:
        if alias.isascii():
            return alias.casefold() in cls._ASCII_WORD.findall(text.casefold())
        return alias.casefold() in text.casefold()

    @classmethod
    def _find_skills(cls, text: str) -> list[CanonicalWorkflowStep]:
        folded = text.casefold()
        matches = []
        all_aliases = [(alias.casefold(), skill_name) for aliases, skill_name in cls._RULES for alias in aliases]
        for aliases, skill_name in cls._RULES:
            if any(cls._alias_hit(alias, folded) for alias in aliases):
                # Prefer a longer Chinese alias, so "不同意" does not also mean "同意".
                if any(
                    longer_skill != skill_name
                    and len(longer) > len(alias.casefold())
                    and alias.casefold() in longer
                    and cls._alias_hit(longer, folded)
                    for longer, longer_skill in all_aliases
                    for alias in aliases
                    if not longer.isascii() and not alias.isascii()
                ):
                    continue
                matches.append(CanonicalWorkflowStep(1, skill_name))
        return matches

    def plan(self, request, context, catalog, cancel_token, *, on_preview=None):
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        text = request.text.strip()
        folded = text.casefold()
        if any(token in folded for token in _NEGATION_MARKERS):
            return PlannerOutcome(
                kind="conversation",
                user_message="我不会执行被明确否定的动作。请直接说明你希望机器人做什么。",
            )
        query = read_only_outcome(text)
        if query is not None:
            return query
        if any(token in folded for token in ("吗", "能不能", "会不会", *_HYPOTHETICAL_MARKERS, "?", "？")):
            return PlannerOutcome(
                kind="conversation",
                user_message="这是能力询问或假设，不会触发机器人动作。",
            )
        if any(token in folded for token in ("你好", "谢谢", "hello", "hi")) and not self._find_skills(folded):
            return PlannerOutcome(kind="conversation", user_message="你好，我可以帮助你控制 SO-101。")
        if any(token in folded for token in ("做一个动作", "做个动作", "动一下", "do something")):
            return PlannerOutcome(
                kind="needs_clarification",
                user_message="请明确选择当前目录中的一个动作，例如点头或挥手。",
                missing_fields=("skill_name",),
            )

        steps = self._find_skills(folded)
        if any(separator in text for separator in ("然后", "再", "之后", "接着")):
            # Rules above are checked against the complete text, preserving the
            # user's order by matching each clause rather than the alias table.
            clauses = [part.strip() for part in re.split(r"然后|再|之后|接着", text) if part.strip()]
            ordered = []
            for clause in clauses:
                clause_steps = self._find_skills(clause)
                if clause_steps:
                    ordered.extend(clause_steps)
            if ordered:
                steps = ordered
        if not steps:
            return PlannerOutcome(
                kind="rejected",
                user_message="我无法把这句话安全映射到当前 SO-101 的已验证技能。",
                reason_code="UNSUPPORTED_TASK",
            )
        if cancel_token.is_set():
            raise PlannerParseError("planning cancelled")
        continuation = context.get("continuation") if isinstance(context, Mapping) else None
        if isinstance(continuation, Mapping):
            # Same contract as the model prompt: drop fully completed steps, redo the interrupted one.
            steps = steps[int(continuation.get("completed_step_count", 0)) :]
            if not steps:
                return PlannerOutcome(kind="conversation", user_message="上一个任务的步骤已经全部完成。")
        summary = "、".join(step.skill_name for step in steps)
        return PlannerOutcome(
            kind="workflow", user_message=f"准备执行：{summary}。", summary=summary, steps=tuple(steps)
        )


__all__ = [
    "PlannerAdapter",
    "PlannerParseError",
    "RulePlanner",
    "build_planner_messages",
    "parse_planner_outcome",
    "planner_outcome_from_payload",
    "read_only_outcome",
]
