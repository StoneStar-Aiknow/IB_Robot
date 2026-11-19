"""The optional gesture service has bounded, atomic context and no runtime imports."""

import json
from dataclasses import replace

import pytest

from embodied_common.hand_gesture_contract import (
    JSON_BYTES,
    RequestMeta,
    ResultMeta,
    SourceState,
    dump_json,
    load_json,
    session_token,
)

CLIENT = "00000000-0000-4000-8000-000000000001"
SESSION = "00000000-0000-4000-8000-000000000002"
SERVER = "00000000-0000-4000-8000-000000000003"
ENGINE = "a" * 64


def request(**overrides):
    values = dict(
        requester_boot_id=CLIENT,
        producer_start_mono_ns=10,
        stream_generation=2,
        request_id=7,
        op="infer",
        session_token=SESSION,
        actor_id=3,
        stamp_ns=1_000_000_000,
        pear_stamp_ns=950_000_000,
        crop="both",
        roi=(-10, 20, 90, 120),
        source_size=(640, 480),
        budget_ms=400.0,
    )
    return RequestMeta(**(values | overrides))


def source(**overrides):
    values = dict(
        requester_boot_id=CLIENT,
        producer_start_mono_ns=10,
        stream_generation=2,
        state_seq=1,
        available=True,
        published_mono_ns=20,
        actor_id=3,
        session_token=SESSION,
        server_boot_id=SERVER,
        engine_sha256=ENGINE,
    )
    return SourceState(**(values | overrides))


def test_crop_context_round_trips_atomically_including_negative_roi():
    meta = request()
    assert RequestMeta.from_json(meta.to_json()) == meta
    assert meta.source_key == (10, CLIENT, 2)


def test_echoed_result_has_the_exact_request_and_backend_identity():
    result = ResultMeta(request(), SERVER, ENGINE, gesture="holy", score=0.91)
    decoded = ResultMeta.from_json(result.to_json())
    assert decoded == result
    assert decoded.request.actor_id == 3
    assert decoded.request.roi == (-10, 20, 90, 120)


def test_probe_and_retire_do_not_need_or_fabricate_image_context():
    probe = RequestMeta(CLIENT, 10, 2, 8)
    retire = replace(probe, op="retire", request_id=9, retire_through=7)
    assert RequestMeta.from_json(probe.to_json()) == probe
    assert RequestMeta.from_json(retire.to_json()) == retire
    with pytest.raises(ValueError):
        replace(retire, retire_through=10)


@pytest.mark.parametrize(
    "values",
    [
        {"request_id": True},
        {"stream_generation": 0},
        {"request_id": 7.0},
        {"actor_id": -1},
        {"session_token": "old"},
        {"budget_ms": float("nan")},
        {"budget_ms": float("inf")},
        {"budget_ms": 0.0},
        {"budget_ms": 801.0},
        {"crop": "body"},
        {"roi": (0, 0, 100, 90)},
        {"source_size": (0, 480)},
        {"stamp_ns": 0},
        {"pear_stamp_ns": 1_100_000_000},
    ],
)
def test_invalid_request_context_is_rejected(values):
    with pytest.raises(ValueError):
        request(**values)


def test_missing_or_unknown_fields_are_not_silently_defaulted():
    value = request().to_dict()
    for field in ("op", "actor_id", "request_id", "source_size"):
        with pytest.raises(ValueError):
            RequestMeta.from_dict({k: v for k, v in value.items() if k != field})
    with pytest.raises(ValueError):
        RequestMeta.from_dict(value | {"unknown": 1})


@pytest.mark.parametrize("text", ['{"schema":"x","v":NaN}', '{"schema":"x","schema":"x"}', "[]"])
def test_ambiguous_or_nonfinite_json_is_rejected(text):
    with pytest.raises(ValueError):
        load_json(text, "x")


def test_limits_are_utf8_bytes_and_apply_to_both_encoding_and_parsing():
    text = json.dumps({"schema": "x", "v": "手" * 1500}, ensure_ascii=False)
    assert len(text) < JSON_BYTES < len(text.encode())
    with pytest.raises(ValueError):
        load_json(text, "x")
    with pytest.raises(ValueError):
        dump_json({"schema": "x", "v": "手" * 1500})


def test_source_state_round_trip_and_strict_health_fields():
    state = source()
    assert SourceState.from_json(state.to_json()) == state
    with pytest.raises(ValueError):
        source(available="true")
    with pytest.raises(ValueError):
        source(published_mono_ns=9)
    with pytest.raises(ValueError):
        source(reason="手" * 43)
    with pytest.raises(ValueError):
        source(server_boot_id="")
    assert source(available=False, actor_id=-1, server_boot_id="", engine_sha256="", session_token="")


def test_invalid_result_class_score_or_identity_is_rejected():
    for kw in ({"gesture": "wave"}, {"score": True}, {"score": float("nan")}, {"idle": 1}):
        with pytest.raises(ValueError):
            ResultMeta(request(), SERVER, ENGINE, **kw)
    with pytest.raises(ValueError):
        ResultMeta(request(), "not-a-boot-id", ENGINE)


def test_session_token_uses_the_canonical_transition_not_subscriber_local_epoch():
    state = {
        "state": "MIRRORING",
        "phase": "running",
        "seq": 10,
        "last_transition": {"from": "MIRRORING/approach", "to": "MIRRORING/running", "t": 123.456},
    }
    assert session_token(state) == session_token(state | {"seq": 20})
    assert session_token(state) != session_token(state | {"last_transition": state["last_transition"] | {"t": 124.0}})
    assert session_token(state | {"phase": "wrap_up"}) == ""
    assert session_token(state | {"last_transition": None}) == ""
