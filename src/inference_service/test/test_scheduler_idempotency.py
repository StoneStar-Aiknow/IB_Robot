"""Unit tests for scheduler idempotency helpers.

Pure-Python, no ROS. Verifies canonical fingerprint stability + sensitivity,
UUID4/pipeline-id validation, deadline conversion rules, and rejection
of NaN/Infinity/non-UTF-8/undeclared types.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from inference_service.scheduler.idempotency import (
    CanonicalTime,
    IdempotencyError,
    canonical_fingerprint,
    deadline_to_monotonic_ns,
    resolve_entry_deadline_ns,
    validate_pipeline_id,
    validate_uuid4,
)


@dataclass
class _FakeTime:
    sec: int
    nanosec: int


# ---------------------------------------------------------------------------
# Canonical fingerprint.
# ---------------------------------------------------------------------------


def test_fingerprint_stable_across_calls():
    payload = {
        "request_id": "123e4567-e89b-42d3-a456-426614174000",
        "session_id": "00112233-4455-6677-8899-aabbccddeeff",
        "prompt": "pick banana",
        "priority": 0,
        "obs_timestamp": _FakeTime(1000, 5),
        "deadline": _FakeTime(2000, 0),
    }
    fp1 = canonical_fingerprint(payload)
    fp2 = canonical_fingerprint(dict(reversed(list(payload.items()))))  # different dict order
    assert fp1 == fp2
    assert len(fp1) == 64


def test_fingerprint_sensitive_to_payload_change():
    base = {"session_id": "00112233-4455-6677-8899-aabbccddeeff", "prompt": "a", "priority": 0}
    fp_base = canonical_fingerprint(base)
    assert canonical_fingerprint({**base, "prompt": "b"}) != fp_base
    assert canonical_fingerprint({**base, "priority": 1}) != fp_base
    # array order matters
    assert canonical_fingerprint({**base, "x": [1, 2]}) != canonical_fingerprint({**base, "x": [2, 1]})


def test_fingerprint_time_encoded_as_sec_nanosec():
    a = canonical_fingerprint({"t": _FakeTime(1, 2)})
    b = canonical_fingerprint({"t": {"sec": 1, "nanosec": 2}})
    assert a == b


def test_fingerprint_rejects_nan_and_infinity():
    with pytest.raises(IdempotencyError, match="NaN"):
        canonical_fingerprint({"x": float("nan")})
    with pytest.raises(IdempotencyError, match="Infinity"):
        canonical_fingerprint({"x": float("inf")})
    with pytest.raises(IdempotencyError, match="Infinity"):
        canonical_fingerprint({"x": float("-inf")})


def test_fingerprint_rejects_non_string_keys_and_unsupported_types():
    class Custom:
        pass

    with pytest.raises(IdempotencyError, match="non-string mapping key"):
        canonical_fingerprint({1: "x"})  # type: ignore[dict-item]
    with pytest.raises(IdempotencyError, match="unsupported idempotency payload type"):
        canonical_fingerprint({"x": Custom()})


# ---------------------------------------------------------------------------
# UUID4 and pipeline ID validation.
# ---------------------------------------------------------------------------


def test_validate_uuid4_accepts_canonical_rejects_non_canonical():
    good = "123e4567-e89b-42d3-a456-426614174000"
    assert validate_uuid4(good) == good
    # version != 4 (3rd group must start with '4')
    with pytest.raises(IdempotencyError):
        validate_uuid4("123e4567-e89b-32d3-a456-426614174000")
    # variant wrong (4th group must start with 8/9/a/b)
    with pytest.raises(IdempotencyError):
        validate_uuid4("123e4567-e89b-42d3-c456-426614174000")
    # uppercase rejected
    with pytest.raises(IdempotencyError):
        validate_uuid4("123E4567-E89B-42D3-A456-426614174000")
    # braces rejected
    with pytest.raises(IdempotencyError):
        validate_uuid4("{123e4567-e89b-42d3-a456-426614174000}")


def test_validate_pipeline_id():
    assert validate_pipeline_id("pi05_full") == "pi05_full"
    with pytest.raises(IdempotencyError):
        validate_pipeline_id("")


# ---------------------------------------------------------------------------
# Deadline conversion.
# ---------------------------------------------------------------------------


def test_resolve_entry_deadline_zero_uses_configured_default():
    # Open uses default_open_timeout; Dispatch/Close use default_request_timeout.
    open_deadline = resolve_entry_deadline_ns(
        0, is_open=True, default_open_timeout_ns=10_000_000_000, default_request_timeout_ns=5_000_000_000
    )
    # nonzero (now + open timeout); cannot assert exact now, just > timeout bound
    assert open_deadline > 10_000_000_000 - 1
    req_deadline = resolve_entry_deadline_ns(
        0, is_open=False, default_open_timeout_ns=10_000_000_000, default_request_timeout_ns=5_000_000_000
    )
    assert req_deadline > 5_000_000_000 - 1


def test_resolve_entry_deadline_nonzero_preserved():
    out = resolve_entry_deadline_ns(
        1_700_000_000_000_000_000, is_open=True, default_open_timeout_ns=10, default_request_timeout_ns=10
    )
    assert out == 1_700_000_000_000_000_000


def test_resolve_entry_deadline_rejects_negative():
    with pytest.raises(IdempotencyError):
        resolve_entry_deadline_ns(-1, is_open=False, default_open_timeout_ns=1, default_request_timeout_ns=1)


def test_deadline_to_monotonic_clamps_negative_remaining():
    # utc deadline in the past -> remaining clamped to 0 -> mono == now
    out = deadline_to_monotonic_ns(deadline_utc_ns=100, utc_now_ns=200, mono_now_ns=1000)
    assert out == 1000
    # utc deadline in the future -> mono = now + remaining
    out2 = deadline_to_monotonic_ns(deadline_utc_ns=300, utc_now_ns=100, mono_now_ns=1000)
    assert out2 == 1200


def test_canonical_time_from_msg_rejects_negative_and_out_of_range():
    assert CanonicalTime.from_msg(_FakeTime(1, 2)) == CanonicalTime(1, 2)
    with pytest.raises(IdempotencyError):
        CanonicalTime.from_msg(_FakeTime(-1, 0))
    with pytest.raises(IdempotencyError):
        CanonicalTime.from_msg(_FakeTime(0, 4_294_967_296))
