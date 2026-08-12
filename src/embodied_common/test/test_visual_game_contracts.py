import pytest

from embodied_common import visual_game_contracts as contracts
from embodied_common.visual_game_contracts import (
    SORTING_HAT_UNDETERMINED,
    build_visual_game_capability_view,
    get_default_visual_game_handler,
    get_visual_game_announcement,
    get_visual_game_handler,
    get_visual_game_prompt,
    get_visual_game_terminal_error,
    normalize_visual_game_policies,
)


def _games(*, enabled=True):
    return {
        "sorting_hat": {
            "enabled": enabled,
            "handler": "sorting_hat_v1",
            "summary": "Choose a Hogwarts house.",
        }
    }


def _view(games, *, start_service="/embodied/start_visual_game", event_topic="/embodied/visual_game_events"):
    return build_visual_game_capability_view(
        "test_robot",
        games,
        timeout_sec=130.0,
        result_retention_sec=300.0,
        result_capacity=128,
        start_service=start_service,
        result_service="/embodied/get_visual_game_result",
        event_topic=event_topic,
    )


def test_normalize_visual_game_policy_attaches_shared_handler_contract():
    policy = normalize_visual_game_policies(_games())["sorting_hat"]

    assert policy["handler"] == "sorting_hat_v1"
    assert policy["announce"] is False
    assert policy["required_inputs"] == ["primary_image"]
    assert policy["result_schema"]["field"] == "scene_summary"
    assert "prompt" not in policy
    assert "trigger_mode" not in policy
    assert "aliases" not in policy
    assert set(policy["result_schema"]["allowed_values"]) == {
        "斯莱特林",
        "格兰芬多",
        "拉文克劳",
        "赫奇帕奇",
        SORTING_HAT_UNDETERMINED,
    }


def test_handler_definition_single_sources_contract_and_runtime_prompt():
    contract = get_visual_game_handler("sorting_hat_v1")
    prompt = get_visual_game_prompt("sorting_hat_v1")

    assert "prompt" not in contract
    assert "分院帽" in prompt

    contract["result_schema"]["allowed_values"].clear()
    assert get_visual_game_handler("sorting_hat_v1")["result_schema"]["allowed_values"]


def test_game_digest_changes_when_announcement_changes():
    base = _view(_games())
    announced = _view({"sorting_hat": {**_games()["sorting_hat"], "announce": True}})

    assert base["config_digest"] != announced["config_digest"]


def test_handler_definition_without_runtime_prompt_fails_closed(monkeypatch):
    # The frozen registry (MappingProxyType) rejects setitem; replace the whole
    # registry reference so the prompt guard in _get_visual_game_handler_definition
    # is exercised against a malformed handler.
    monkeypatch.setattr(
        contracts,
        "_HANDLERS",
        contracts.deep_freeze(
            {
                "missing_prompt_v1": {
                    "game_name": "missing_prompt",
                    "summary": "Missing prompt.",
                    "required_inputs": ["primary_image"],
                    "result_schema": {"field": "scene_summary", "kind": "string"},
                },
            }
        ),
    )

    with pytest.raises(ValueError, match="missing fields.*prompt"):
        contracts._validate_registered_handlers()


def test_sorting_hat_undetermined_value_maps_to_no_person_terminal_error():
    assert get_visual_game_terminal_error("sorting_hat_v1", {"scene_summary": SORTING_HAT_UNDETERMINED}) == (
        "NO_PERSON",
        "no clearly visible person",
    )
    assert get_visual_game_terminal_error("sorting_hat_v1", {"scene_summary": "拉文克劳"}) is None


def test_sorting_hat_announcement_is_handler_owned():
    assert (
        get_visual_game_announcement(
            "sorting_hat_v1",
            state="succeeded",
            success=True,
            error_code="",
            result={"scene_summary": "拉文克劳"},
        )
        == "拉文克劳"
    )
    assert (
        get_visual_game_announcement(
            "sorting_hat_v1",
            state="failed",
            success=False,
            error_code="NO_PERSON",
            result={},
        )
        == "暂未识别到新生，请走入画面中央"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"handler": "missing_v1"}, "unsupported visual game handler"),
        ({"summary": ""}, "summary must be a non-empty string"),
        ({"trigger_mode": "agent"}, "no longer supports"),
        ({"trigger_aliases": ["分院帽"]}, "no longer supports"),
    ],
)
def test_invalid_or_removed_visual_game_policy_fails_closed(change, message):
    games = _games()
    games["sorting_hat"].update(change)

    with pytest.raises(ValueError, match=message):
        normalize_visual_game_policies(games)


def test_capability_view_exposes_only_enabled_games_and_digest_covers_transport():
    enabled = _view(_games())
    disabled = _view(_games(enabled=False))
    moved_service = _view(_games(), start_service="/other/start")
    moved_event_topic = _view(_games(), event_topic="/other/events")

    assert enabled["games"][0]["name"] == "sorting_hat"
    assert "trigger_mode" not in enabled["games"][0]
    assert "aliases" not in enabled["games"][0]
    assert disabled["games"] == []
    assert enabled["config_digest"] != disabled["config_digest"]
    assert enabled["config_digest"] != moved_service["config_digest"]
    assert enabled["config_digest"] != moved_event_topic["config_digest"]


def test_visual_game_policy_defaults_known_handler_and_summary():
    policy = normalize_visual_game_policies({"sorting_hat": {"enabled": False}})["sorting_hat"]

    assert policy["handler"] == "sorting_hat_v1"
    assert policy["summary"]


def test_handler_cannot_be_reused_under_a_different_game_name():
    with pytest.raises(ValueError, match="registered for game 'sorting_hat'"):
        normalize_visual_game_policies(
            {"other": {"enabled": True, "handler": "sorting_hat_v1", "summary": "Other game"}}
        )


def test_get_default_handler_distinguishes_zero_and_multiple_matches(monkeypatch):
    monkeypatch.setattr(contracts, "_HANDLERS", contracts.deep_freeze({}))
    with pytest.raises(ValueError, match="no registered handler"):
        get_default_visual_game_handler("sorting_hat")

    monkeypatch.setattr(
        contracts,
        "_HANDLERS",
        contracts.deep_freeze(
            {
                "a_v1": {"game_name": "dup", "result_schema": {"field": "x", "kind": "string"}},
                "a_v2": {"game_name": "dup", "result_schema": {"field": "x", "kind": "string"}},
            }
        ),
    )
    with pytest.raises(ValueError, match="multiple registered handlers"):
        get_default_visual_game_handler("dup")


def _valid_handler_descriptor():
    return {
        "game_name": "sorting_hat",
        "summary": "Choose one house.",
        "prompt": "Choose one house and return JSON.",
        "announcement": {
            "success_field": "scene_summary",
            "success_values": ["ok"],
            "error_text": {"NO_PERSON": "No person."},
        },
        "required_inputs": ["primary_image"],
        "result_schema": {
            "field": "scene_summary",
            "kind": "enum",
            "allowed_values": ["ok", "none"],
            "failure_values": {"none": {"error_code": "NO_PERSON", "message": "no person"}},
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda handler: handler.update(extra=True), "unsupported fields"),
        (lambda handler: handler.update(summary=""), "summary must be a non-empty string"),
        (lambda handler: handler.update(required_inputs=[]), "required_inputs must be a non-empty list"),
        (
            lambda handler: handler.update(required_inputs=["primary_image", "unknown"]),
            "unsupported required_inputs",
        ),
        (
            lambda handler: handler["result_schema"].update(kind="bogus"),
            "result_schema kind.*is invalid",
        ),
        (
            lambda handler: handler["result_schema"].update(field="unknown"),
            "result_schema.field is unsupported",
        ),
        (
            lambda handler: handler["result_schema"].update(allowed_values=["ok", "ok"]),
            "allowed_values must not contain duplicates",
        ),
        (
            lambda handler: handler["announcement"].update(success_field="other"),
            "success_field must match",
        ),
        (
            lambda handler: handler["announcement"].update(success_values=["none"]),
            "must not include terminal failure values",
        ),
        (
            lambda handler: handler["announcement"].update(error_text={}),
            "missing handler failure codes",
        ),
        (
            lambda handler: handler["announcement"]["error_text"].update(GAME_BUSY="busy"),
            "unsupported error codes",
        ),
    ],
)
def test_registered_handler_full_schema_validation(monkeypatch, mutate, message):
    handler = _valid_handler_descriptor()
    mutate(handler)
    monkeypatch.setattr(contracts, "_HANDLERS", contracts.deep_freeze({"sorting_hat_v1": handler}))

    with pytest.raises(ValueError, match=message):
        contracts._validate_registered_handlers()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "timeout_sec": 0,
                "result_retention_sec": 300.0,
                "result_capacity": 128,
                "start_service": "/s",
                "result_service": "/r",
            },
            "finite positive",
        ),
        (
            {
                "timeout_sec": 130.0,
                "result_retention_sec": 0,
                "result_capacity": 128,
                "start_service": "/s",
                "result_service": "/r",
            },
            "finite positive",
        ),
        (
            {
                "timeout_sec": 130.0,
                "result_retention_sec": 300.0,
                "result_capacity": 0,
                "start_service": "/s",
                "result_service": "/r",
            },
            "positive integer",
        ),
        (
            {
                "timeout_sec": 130.0,
                "result_retention_sec": 300.0,
                "result_capacity": 128,
                "start_service": "/s",
                "result_service": "/s",
            },
            "must be different",
        ),
        (
            {
                "timeout_sec": 130.0,
                "result_retention_sec": 300.0,
                "result_capacity": 128,
                "start_service": "",
                "result_service": "/r",
            },
            "non-empty string",
        ),
        (
            {
                "timeout_sec": 130.0,
                "result_retention_sec": 300.0,
                "result_capacity": 128,
                "start_service": "/s",
                "result_service": "/r",
            },
            None,
        ),
    ],
)
def test_capability_view_validation_errors(kwargs, message):
    if message is None:
        view = build_visual_game_capability_view("test_robot", _games(), **kwargs)
        assert "config_digest" in view
        return
    with pytest.raises(ValueError, match=message):
        build_visual_game_capability_view("test_robot", _games(), **kwargs)


def test_capability_view_rejects_empty_robot_name():
    with pytest.raises(ValueError, match="non-empty string"):
        build_visual_game_capability_view(
            "  ",
            _games(),
            timeout_sec=130.0,
            result_retention_sec=300.0,
            result_capacity=128,
            start_service="/s",
            result_service="/r",
        )


def test_malformed_failure_values_rejected_at_import(monkeypatch):
    handler = _valid_handler_descriptor()
    handler["result_schema"]["failure_values"]["none"]["error_code"] = ""
    monkeypatch.setattr(
        contracts,
        "_HANDLERS",
        contracts.deep_freeze({"bad_v1": handler}),
    )
    with pytest.raises(ValueError, match="error_code"):
        contracts._validate_registered_handlers()


def test_failure_value_outside_allowed_values_rejected_at_import(monkeypatch):
    handler = _valid_handler_descriptor()
    handler["result_schema"]["failure_values"] = {"not_allowed": {"error_code": "NO_PERSON", "message": "no person"}}
    monkeypatch.setattr(
        contracts,
        "_HANDLERS",
        contracts.deep_freeze({"bad_v1": handler}),
    )
    with pytest.raises(ValueError, match="allowed_values"):
        contracts._validate_registered_handlers()
