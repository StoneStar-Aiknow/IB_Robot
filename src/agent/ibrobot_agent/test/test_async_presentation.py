from __future__ import annotations

import threading

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.async_presentation import AsyncPresentationPort, ModeRoutingPresentationPort
from ibrobot_agent.contracts import (
    ASYNC_PRESENTATION_MODE,
    GATED_PRESENTATION_MODE,
    Presentation,
    RequestKey,
    TaskRef,
)
from ibrobot_agent.presentation import presentation_digest

KEY = RequestKey("so101_single_arm", "channel", "principal", "request-1")


def _presentation(mode: str = ASYNC_PRESENTATION_MODE) -> Presentation:
    task_ref = TaskRef("task-1", "plan-1", "digest", "epoch", 1, "registry", 1)
    return Presentation(task_ref, 1, (CanonicalWorkflowStep(1, "nod_yes"),), mode, 30.0, "nod")


def test_async_port_publishes_receipt_free_payload_with_canonical_digest():
    published = []
    AsyncPresentationPort().present(KEY, _presentation(), publish=published.append, cancel_token=threading.Event())

    (detail,) = published
    assert detail["async_display"] is True
    assert detail["receipt_token"] == ""
    assert detail["presentation"] == _presentation().to_dict()
    assert detail["presentation_digest"] == presentation_digest(_presentation().to_dict())


def test_async_port_isolates_publish_failure_from_execution():
    port = AsyncPresentationPort()

    def broken(_detail):
        raise RuntimeError("display transport unavailable")

    port.present(KEY, _presentation(), publish=broken, cancel_token=threading.Event())
    assert port.dropped == 1


def test_async_port_ignores_cancel_token_and_drops_after_close():
    port = AsyncPresentationPort()
    cancelled = threading.Event()
    cancelled.set()
    published = []
    port.present(KEY, _presentation(), publish=published.append, cancel_token=cancelled)
    assert len(published) == 1  # display never doubles as a stop signal
    port.close()
    port.close()
    port.present(KEY, _presentation(), publish=published.append, cancel_token=threading.Event())
    assert len(published) == 1
    assert port.dropped == 1


class _RecordingPort:
    def __init__(self):
        self.modes = []

    def present(self, key, presentation, *, publish, cancel_token):
        self.modes.append(presentation.execution_mode)


def test_mode_router_keeps_receipt_gate_for_legacy_agent_plans():
    async_port, legacy_port = _RecordingPort(), _RecordingPort()
    router = ModeRoutingPresentationPort(async_port=async_port, legacy_port=legacy_port)
    for mode in (ASYNC_PRESENTATION_MODE, GATED_PRESENTATION_MODE, "unexpected_mode"):
        router.present(KEY, _presentation(mode), publish=lambda _: None, cancel_token=threading.Event())

    assert async_port.modes == [ASYNC_PRESENTATION_MODE]
    # Anything but the trusted Program mode fails closed to the receipt gate.
    assert legacy_port.modes == [GATED_PRESENTATION_MODE, "unexpected_mode"]
