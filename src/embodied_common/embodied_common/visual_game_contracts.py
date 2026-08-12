"""Shared handler definitions and public capability views for visual games."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any

from embodied_common.canon import deep_freeze, sha256_text, to_canonical_json
from embodied_common.perception_contracts import (
    KNOWN_REQUIRED_INPUTS,
    SCENE_ANALYSIS_RESULT_FIELD_KINDS,
    validate_result_schema,
)

SORTING_HAT_HOUSES = ["斯莱特林", "格兰芬多", "拉文克劳", "赫奇帕奇"]
SORTING_HAT_UNDETERMINED = "无法判断"

_SORTING_HAT_PROMPT = (
    "这是一个纯娱乐性质的分院帽互动。你是霍格沃茨的分院帽。\n"
    "本次任务只关注画面中的这个人，请忽略桌面物品、电脑、设备、机械臂等一切背景物体，"
    "只根据这个人的外貌、表情、神态、发型、穿着、姿态和整体气质来判断他/她最适合哪个学院。\n\n"
    "四个学院的特点：\n"
    "- 斯莱特林：有野心，清楚自己想干什么，精明果断\n"
    "- 格兰芬多：勇敢，开朗，外向，富有冒险精神\n"
    "- 拉文克劳：聪明，富有智慧，喜欢思考，有创造力\n"
    "- 赫奇帕奇：忠诚，朴实，热爱自然和动物，勤劳肯干\n\n"
    "判断要求：\n"
    "1. 请根据这个人真实的外貌与气质独立判断，不要默认选择格兰芬多；四个学院都应有被选中的可能。\n"
    "2. 若画面中没有清晰可见的人，必须返回“无法判断”，不要根据背景物体或设备硬凑学院。\n\n"
    "请返回一个 JSON 对象，包含以下字段：\n"
    "- scene_summary：有人且清晰可见时，只能填写四个学院之一"
    "（斯莱特林、格兰芬多、拉文克劳、赫奇帕奇）；没有清晰可见的人时只能填写“无法判断”。"
    "不要添加任何解释或理由。\n"
    "- visible_objects：字符串数组，只填写你从这个人身上观察到的线索"
    "（如表情、神态、发型、穿着、姿态、气质等），不要填写桌面物品或设备。\n"
    "- robot_state_summary、ee_pose_interpretation：本互动不涉及机械臂状态，可填“本互动不涉及机械臂状态”。\n"
    "- risks：字符串数组，无风险时填空数组。\n"
    "- confidence：0.0 到 1.0 之间的置信度。"
)

_HANDLERS: Mapping[str, Mapping[str, Any]] = deep_freeze(
    {
        "sorting_hat_v1": {
            "game_name": "sorting_hat",
            "summary": "根据主相机中的人物形象判断其霍格沃茨学院",
            "prompt": _SORTING_HAT_PROMPT,
            "announcement": {
                "success_field": "scene_summary",
                "success_values": SORTING_HAT_HOUSES,
                "error_text": {
                    "NO_PERSON": "暂未识别到新生，请走入画面中央",
                    "CONFIG_MISMATCH": "视觉游戏配置不一致，请联系管理员",
                    "PERCEPTION_DISABLED": "视觉感知服务尚未启用",
                    "PERCEPTION_UNAVAILABLE": "视觉感知服务暂不可用，请稍后再试",
                    "PERCEPTION_FAILED": "视觉感知执行失败，请稍后再试",
                    "GAME_CAPACITY_EXHAUSTED": "视觉游戏结果空间已满，请稍后再试",
                },
            },
            "required_inputs": ["primary_image"],
            "result_schema": {
                "field": "scene_summary",
                "kind": "enum",
                "allowed_values": [*SORTING_HAT_HOUSES, SORTING_HAT_UNDETERMINED],
                "failure_values": {
                    SORTING_HAT_UNDETERMINED: {
                        "error_code": "NO_PERSON",
                        "message": "no clearly visible person",
                    }
                },
            },
        },
    }
)

_RUNTIME_ONLY_HANDLER_FIELDS = {"announcement", "prompt"}
_HANDLER_FIELDS = {"game_name", "summary", "prompt", "announcement", "required_inputs", "result_schema"}
_RESULT_SCHEMA_FIELDS = {"field", "kind", "allowed_values", "failure_values"}
_ANNOUNCEABLE_GATEWAY_ERROR_CODES = {
    "CONFIG_MISMATCH",
    "PERCEPTION_DISABLED",
    "PERCEPTION_UNAVAILABLE",
    "PERCEPTION_FAILED",
    "GAME_CAPACITY_EXHAUSTED",
    "GAME_RESULT_TIMEOUT",
    "INVALID_GAME_RESULT",
}


def _require_non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _validate_announcement(handler_name: str, announcement: Any, schema: Mapping[str, Any]) -> None:
    path = f"visual game handler '{handler_name}' announcement"
    if not isinstance(announcement, Mapping):
        raise ValueError(f"{path} must be a mapping")
    unknown_fields = sorted(set(announcement) - {"success_field", "success_values", "error_text"})
    if unknown_fields:
        raise ValueError(f"{path} has unsupported fields: {unknown_fields}")

    success_field = _require_non_empty_string(announcement.get("success_field"), path=f"{path}.success_field")
    if success_field != schema["field"]:
        raise ValueError(f"{path}.success_field must match result_schema.field")
    success_values = announcement.get("success_values")
    if not isinstance(success_values, list | tuple) or not success_values:
        raise ValueError(f"{path}.success_values must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in success_values):
        raise ValueError(f"{path}.success_values must contain non-empty strings")
    if len(set(success_values)) != len(success_values):
        raise ValueError(f"{path}.success_values must not contain duplicates")
    if schema["kind"] not in {"enum", "string"}:
        raise ValueError(f"{path}.success_field must use a string-valued result_schema")
    if schema["kind"] == "enum":
        unknown_values = sorted(set(success_values) - set(schema["allowed_values"]))
        if unknown_values:
            raise ValueError(f"{path}.success_values are outside result_schema.allowed_values: {unknown_values}")
    failure_values = schema.get("failure_values", {})
    overlap = sorted(set(success_values).intersection(failure_values))
    if overlap:
        raise ValueError(f"{path}.success_values must not include terminal failure values: {overlap}")

    error_text = announcement.get("error_text")
    if not isinstance(error_text, Mapping):
        raise ValueError(f"{path}.error_text must be a mapping")
    for error_code, text in error_text.items():
        _require_non_empty_string(error_code, path=f"{path}.error_text error code")
        _require_non_empty_string(text, path=f"{path}.error_text.{error_code}")
    failure_error_codes = {payload["error_code"] for payload in failure_values.values()}
    missing_failure_text = sorted(failure_error_codes - set(error_text))
    if missing_failure_text:
        raise ValueError(f"{path}.error_text is missing handler failure codes: {missing_failure_text}")
    unknown_error_codes = sorted(set(error_text) - failure_error_codes - _ANNOUNCEABLE_GATEWAY_ERROR_CODES)
    if unknown_error_codes:
        raise ValueError(f"{path}.error_text has unsupported error codes: {unknown_error_codes}")


def _validate_registered_handlers() -> None:
    """Fail fast at import time if a registered handler's contract is malformed.

    These are registry-definition bugs, not runtime result bugs: surfacing them
    only at runtime (as ``INVALID_GAME_RESULT_CONTRACT``) misleads callers. The
    runtime guard in :func:`get_visual_game_terminal_error` is kept as defense.
    """
    for handler_name, descriptor in _HANDLERS.items():
        _require_non_empty_string(handler_name, path="visual game handler name")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"visual game handler '{handler_name}' must be a mapping")
        unknown_fields = sorted(set(descriptor) - _HANDLER_FIELDS)
        missing_fields = sorted(_HANDLER_FIELDS - set(descriptor))
        if unknown_fields:
            raise ValueError(f"visual game handler '{handler_name}' has unsupported fields: {unknown_fields}")
        if missing_fields:
            raise ValueError(f"visual game handler '{handler_name}' is missing fields: {missing_fields}")
        game_name = _require_non_empty_string(
            descriptor.get("game_name"), path=f"visual game handler '{handler_name}' game_name"
        )
        if any(character.isspace() for character in game_name):
            raise ValueError(f"visual game handler '{handler_name}' game_name must not contain whitespace")
        summary = _require_non_empty_string(
            descriptor.get("summary"), path=f"visual game handler '{handler_name}' summary"
        )
        if len(summary) > 120:
            raise ValueError(f"visual game handler '{handler_name}' summary must be at most 120 characters")
        _require_non_empty_string(descriptor.get("prompt"), path=f"visual game handler '{handler_name}' prompt")

        required_inputs = descriptor.get("required_inputs")
        if not isinstance(required_inputs, list | tuple) or not required_inputs:
            raise ValueError(f"visual game handler '{handler_name}' required_inputs must be a non-empty list")
        if any(not isinstance(item, str) or not item.strip() for item in required_inputs):
            raise ValueError(f"visual game handler '{handler_name}' required_inputs must contain non-empty strings")
        if len(set(required_inputs)) != len(required_inputs):
            raise ValueError(f"visual game handler '{handler_name}' required_inputs must not contain duplicates")
        unknown_inputs = sorted(set(required_inputs) - KNOWN_REQUIRED_INPUTS)
        if unknown_inputs:
            raise ValueError(f"visual game handler '{handler_name}' has unsupported required_inputs: {unknown_inputs}")

        schema = descriptor.get("result_schema")
        if not isinstance(schema, Mapping):
            raise ValueError(f"visual game handler '{handler_name}' must declare a result_schema mapping")
        unknown_schema_fields = sorted(set(schema) - _RESULT_SCHEMA_FIELDS)
        if unknown_schema_fields:
            raise ValueError(
                f"visual game handler '{handler_name}' result_schema has unsupported fields: {unknown_schema_fields}"
            )
        field = _require_non_empty_string(
            schema.get("field"), path=f"visual game handler '{handler_name}' result_schema.field"
        )
        kind = _require_non_empty_string(
            schema.get("kind"), path=f"visual game handler '{handler_name}' result_schema.kind"
        )
        if field not in SCENE_ANALYSIS_RESULT_FIELD_KINDS:
            raise ValueError(f"visual game handler '{handler_name}' result_schema.field is unsupported: {field}")
        if kind not in SCENE_ANALYSIS_RESULT_FIELD_KINDS[field]:
            raise ValueError(
                f"visual game handler '{handler_name}' result_schema kind '{kind}' is invalid for field '{field}'"
            )
        allowed_values = schema.get("allowed_values")
        if kind == "enum":
            if not isinstance(allowed_values, list | tuple) or not allowed_values:
                raise ValueError(
                    f"visual game handler '{handler_name}' enum result_schema must declare non-empty allowed_values"
                )
            if any(not isinstance(value, str) or not value.strip() for value in allowed_values):
                raise ValueError(
                    f"visual game handler '{handler_name}' result_schema.allowed_values must contain non-empty strings"
                )
            if len(set(allowed_values)) != len(allowed_values):
                raise ValueError(
                    f"visual game handler '{handler_name}' result_schema.allowed_values must not contain duplicates"
                )
        elif "allowed_values" in schema:
            raise ValueError(
                f"visual game handler '{handler_name}' non-enum result_schema must not declare allowed_values"
            )
        failure_values = schema.get("failure_values", {})
        if not isinstance(failure_values, Mapping):
            raise ValueError(f"visual game handler '{handler_name}' failure_values must be a mapping")
        if failure_values and kind != "enum":
            raise ValueError(f"visual game handler '{handler_name}' failure_values require an enum result_schema")
        for value, descriptor_payload in failure_values.items():
            _require_non_empty_string(value, path=f"visual game handler '{handler_name}' failure value")
            if not isinstance(descriptor_payload, Mapping):
                raise ValueError(f"visual game handler '{handler_name}' failure_values.{value} must be a mapping")
            unknown_failure_fields = sorted(set(descriptor_payload) - {"error_code", "message"})
            if unknown_failure_fields:
                raise ValueError(
                    f"visual game handler '{handler_name}' failure_values.{value} has unsupported fields: "
                    f"{unknown_failure_fields}"
                )
            error_code = descriptor_payload.get("error_code")
            message = descriptor_payload.get("message")
            _require_non_empty_string(
                error_code, path=f"visual game handler '{handler_name}' failure_values.{value}.error_code"
            )
            _require_non_empty_string(
                message, path=f"visual game handler '{handler_name}' failure_values.{value}.message"
            )
            if value not in allowed_values:
                raise ValueError(
                    f"visual game handler '{handler_name}' failure_values.{value} "
                    "must be one of the result_schema allowed_values"
                )
        _validate_announcement(handler_name, descriptor.get("announcement"), schema)


_validate_registered_handlers()


def _unfreeze(value: Any) -> Any:
    """Deep-copy a frozen registry node into mutable dict/list leaves.

    ``copy.deepcopy`` cannot pickle ``MappingProxyType``; this walker restores
    plain ``dict``/``list`` so callers can freely mutate their private copy
    without touching the shared (frozen) registry.
    """
    if isinstance(value, Mapping):
        return {key: _unfreeze(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_unfreeze(item) for item in value]
    return value


def _get_visual_game_handler_definition(handler_name: str) -> dict[str, Any]:
    try:
        definition = _unfreeze(_HANDLERS[handler_name])
    except KeyError as exc:
        raise ValueError(f"unsupported visual game handler: {handler_name}") from exc
    prompt = definition.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"visual game handler '{handler_name}' must declare a non-empty runtime prompt")
    return definition


def get_visual_game_handler(handler_name: str) -> dict[str, Any]:
    """Return the public contract projection of one shared handler definition."""
    definition = _get_visual_game_handler_definition(handler_name)
    for field in _RUNTIME_ONLY_HANDLER_FIELDS:
        definition.pop(field, None)
    return definition


def get_visual_game_prompt(handler_name: str) -> str:
    """Return the runtime prompt from the same definition as the public contract."""
    return str(_get_visual_game_handler_definition(handler_name)["prompt"])


def get_visual_game_announcement(
    handler_name: str,
    *,
    state: str,
    success: bool,
    error_code: str,
    result: Mapping[str, Any],
) -> str | None:
    """Return the handler-owned announcement for one terminal result."""
    announcement = _get_visual_game_handler_definition(handler_name).get("announcement", {})
    if not isinstance(announcement, Mapping):
        return None
    if state == "succeeded" and success:
        field = announcement.get("success_field")
        values = announcement.get("success_values")
        if not isinstance(field, str) or not isinstance(values, list):
            return None
        value = result.get(field)
        return value if isinstance(value, str) and value in values else None
    if state == "failed" and not success:
        error_text = announcement.get("error_text")
        if not isinstance(error_text, Mapping):
            return None
        text = error_text.get(error_code)
        return text if isinstance(text, str) and text.strip() else None
    return None


def get_default_visual_game_handler(game_name: str) -> str:
    """Return the unique registered handler for a canonical game name."""
    matches = [name for name, descriptor in _HANDLERS.items() if descriptor["game_name"] == game_name]
    if not matches:
        raise ValueError(f"visual game '{game_name}' has no registered handler")
    if len(matches) > 1:
        raise ValueError(f"visual game '{game_name}' has multiple registered handlers: {sorted(matches)}")
    return matches[0]


def normalize_visual_game_policies(raw_games: Any) -> dict[str, dict[str, Any]]:
    """Validate deployment policies and attach their shared handler contracts."""
    if raw_games is None:
        return {}
    if not isinstance(raw_games, Mapping):
        raise ValueError("embodied.visual_games must be a mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_policy in raw_games.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("embodied.visual_games names must be non-empty strings")
        name = raw_name.strip()
        path = f"embodied.visual_games.{name or raw_name}"
        if not isinstance(raw_policy, Mapping):
            raise ValueError(f"{path} must be a mapping")
        removed_fields = sorted({"trigger_aliases", "trigger_mode"}.intersection(raw_policy))
        if removed_fields:
            raise ValueError(
                f"{path} no longer supports {removed_fields}; visual games are triggered through robot-skill"
            )

        enabled = raw_policy.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{path}.enabled must be a boolean")
        handler = raw_policy.get("handler")
        if handler is None:
            handler = get_default_visual_game_handler(name)
        if not isinstance(handler, str) or not handler.strip():
            raise ValueError(f"{path}.handler must be a non-empty string")
        handler = handler.strip()
        handler_contract = get_visual_game_handler(handler)
        if handler_contract["game_name"] != name:
            raise ValueError(f"{path}.handler is registered for game '{handler_contract['game_name']}'")

        summary = raw_policy.get("summary")
        if summary is None:
            summary = handler_contract["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"{path}.summary must be a non-empty string")
        summary = summary.strip()
        if len(summary) > 120:
            raise ValueError(f"{path}.summary must be at most 120 characters")

        announce = raw_policy.get("announce", False)
        if not isinstance(announce, bool):
            raise ValueError(f"{path}.announce must be a boolean")

        normalized[name] = {
            **handler_contract,
            "enabled": enabled,
            "handler": handler,
            "summary": summary,
            "announce": announce,
        }
    return normalized


def load_visual_game_policies_json(raw_json: str) -> dict[str, dict[str, Any]]:
    """Decode and normalize a visual-game policy parameter."""
    try:
        loaded = json.loads(raw_json) if raw_json else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"visual_games_json must be valid JSON: {exc}") from exc
    return normalize_visual_game_policies(loaded)


def validate_visual_game_result(handler_name: str, result: Mapping[str, Any]) -> str | None:
    """Validate one terminal result at the Agent-facing Gateway boundary."""
    return validate_result_schema(get_visual_game_handler(handler_name)["result_schema"], result)


def get_visual_game_terminal_error(handler_name: str, result: Mapping[str, Any]) -> tuple[str, str] | None:
    """Map an accepted handler result value to a declared terminal failure."""
    contract = get_visual_game_handler(handler_name)["result_schema"]
    field = contract.get("field")
    failure_values = contract.get("failure_values", {})
    if not isinstance(field, str) or not isinstance(failure_values, Mapping):
        return None
    descriptor = failure_values.get(result.get(field))
    if not isinstance(descriptor, Mapping):
        return None
    error_code = descriptor.get("error_code")
    message = descriptor.get("message")
    if not isinstance(error_code, str) or not error_code or not isinstance(message, str) or not message:
        return "INVALID_GAME_RESULT_CONTRACT", "visual game failure value has an invalid error descriptor"
    return error_code, message


def build_visual_game_capability_view(
    robot_name: str,
    visual_games: Any,
    *,
    timeout_sec: float,
    result_retention_sec: float,
    result_capacity: int,
    start_service: str,
    result_service: str,
    event_topic: str = "/embodied/visual_game_events",
) -> dict[str, Any]:
    """Build the stable public visual-game view and its configuration digest."""
    if not isinstance(robot_name, str) or not robot_name.strip():
        raise ValueError("robot_config.name must be a non-empty string")
    for field_name, value in (("timeout_sec", timeout_sec), ("result_retention_sec", result_retention_sec)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"visual game {field_name} must be a finite positive number")
    if isinstance(result_capacity, bool) or not isinstance(result_capacity, int) or result_capacity <= 0:
        raise ValueError("visual game result_capacity must be a positive integer")
    for field_name, value in (
        ("start_service", start_service),
        ("result_service", result_service),
        ("event_topic", event_topic),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"visual game {field_name} must be a non-empty string")
    if start_service == result_service:
        raise ValueError("visual game start_service and result_service must be different")
    normalized = normalize_visual_game_policies(visual_games)
    games = []
    for name, policy in sorted(normalized.items()):
        if not policy["enabled"]:
            continue
        games.append(
            {
                "name": name,
                "summary": policy["summary"],
                "handler": policy["handler"],
                "announce": policy["announce"],
                "required_inputs": copy.deepcopy(policy["required_inputs"]),
                "result_schema": copy.deepcopy(policy["result_schema"]),
            }
        )
    view = {
        "schema_version": 1,
        "robot_name": robot_name.strip(),
        "games": games,
        "timeout_sec": float(timeout_sec),
        "result_retention_sec": float(result_retention_sec),
        "result_capacity": result_capacity,
        "start_service": start_service,
        "result_service": result_service,
        "event_topic": event_topic,
    }
    view["config_digest"] = sha256_text(to_canonical_json(view))
    return view
