"""Owner-release proof uses actual fresh data-plane ownership, not only a seq."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("aimdk_msgs")

from aimdk_robot.runtime_node import AimdkRuntimeNode  # noqa: E402


def runtime(*, echoed_owner=None, sequence=1, at=None, new_claim=False):
    lock = threading.RLock()
    node = SimpleNamespace(
        _lock=lock,
        _dataplane_seen=threading.Condition(lock),
        _arm_owner=7,
        _arm_motion_epoch=11,
        _feedback_window_s=0.1,
        _dataplane=SimpleNamespace(control_seq=0, arm_owner=7, at=time.monotonic()),
    )
    node._control_word_locked = lambda: SimpleNamespace(seq=1, encode=lambda: "control")

    def publish(_message):
        node._dataplane = SimpleNamespace(
            control_seq=sequence,
            arm_owner=echoed_owner,
            at=time.monotonic() if at is None else at,
        )
        if new_claim:
            node._arm_owner = 99
            node._arm_motion_epoch += 1

    node._control_pub = SimpleNamespace(publish=publish)
    return node


def test_release_requires_applied_owner_and_fresh_sequence():
    node = runtime()
    assert AimdkRuntimeNode.release_arm_confirmed(node, 7, wait_s=0.001)
    assert node._arm_owner is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence": 0},
        {"echoed_owner": 7},
        {"at": float("nan")},
        {"at": float("inf")},
        {"at": -1.0},
        {"at": 1e30},
        {"new_claim": True},
    ],
)
def test_release_does_not_accept_stale_or_wrong_owner_evidence(kwargs):
    node = runtime(**kwargs)
    assert not AimdkRuntimeNode.release_arm_confirmed(node, 7, wait_s=0.001)
    if kwargs.get("new_claim"):
        assert node._arm_owner == 99


def test_old_owner_cannot_release_new_claim():
    node = runtime()
    node._arm_owner = 99
    assert not AimdkRuntimeNode.release_arm_confirmed(node, 7, wait_s=0.001)
    assert node._arm_owner == 99


def test_missing_guard_fails_closed():
    node = SimpleNamespace()
    accepted, reason = AimdkRuntimeNode.validate_named_hold_path(node, "overhead_heart", [0, 1], [[0] * 34] * 2)
    assert not accepted
    assert "requires" in reason


def test_guard_exception_or_nonboolean_answer_fails_closed():
    for answer in ((1, "unsafe truthy value"), None):
        node = SimpleNamespace(_named_hold_guard=SimpleNamespace(validate_path=lambda answer=answer, **kwargs: answer))
        assert not AimdkRuntimeNode.validate_named_hold_path(node, "overhead_heart", [0, 1], [[0] * 34] * 2)[0]

    def failed_guard(**kwargs):
        raise ValueError("broken artifact")

    node = SimpleNamespace(_named_hold_guard=SimpleNamespace(validate_path=failed_guard))
    assert not AimdkRuntimeNode.validate_named_hold_path(node, "overhead_heart", [0, 1], [[0] * 34] * 2)[0]


def test_cosine_peak_cannot_exceed_artifact_speed():
    node = SimpleNamespace(
        _named_hold_guard=SimpleNamespace(speed=[0.4] * 34, validate_path=lambda **kwargs: (True, "test geometry"))
    )
    ok, reason = AimdkRuntimeNode.validate_named_hold_path(
        node, "overhead_heart", [0, 3.141592653589793], [[0.0] * 34, [1.0] * 34]
    )
    assert not ok and "peak" in reason
