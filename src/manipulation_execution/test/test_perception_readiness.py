"""Availability checks do not issue detection, segmentation or grasp queries."""

import time
from types import SimpleNamespace

import pytest

from manipulation_execution.grasp_pipeline.contracts import GraspFailure
from manipulation_execution.grasp_pipeline.perception import PerceptionServices


def services(available=True, segment=True):
    checked = []

    def client(name):
        def wait(**kwargs):
            checked.append(name)
            return available

        return SimpleNamespace(srv_name=name, service_is_ready=lambda: False, wait_for_service=wait)

    value = object.__new__(PerceptionServices)
    value._detect, value._segment, value._grasps = (
        client("detect"),
        client("segment") if segment else None,
        client("grasps"),
    )
    return value, checked


@pytest.mark.parametrize("segment", [True, False])
def test_readiness_checks_only_the_configured_services(segment):
    value, checked = services(segment=segment)
    value.wait_ready(deadline=time.monotonic() + 5, cancelled=lambda: False)
    assert checked == (["detect", "segment", "grasps"] if segment else ["detect", "grasps"])


@pytest.mark.parametrize("failure", ["missing", "deadline", "cancelled"])
def test_unavailable_cancelled_or_expired_readiness_is_not_ready(failure):
    value, checked = services(available=False)
    with pytest.raises(GraspFailure) as error:
        value.wait_ready(
            deadline=time.monotonic() + (-1 if failure == "deadline" else 5),
            cancelled=lambda: failure == "cancelled",
        )
    assert error.value.code == {"missing": "RPC_FAILED", "deadline": "SKILL_TIMEOUT", "cancelled": "CANCELED"}[failure]
    assert checked == (["detect"] if failure == "missing" else [])
