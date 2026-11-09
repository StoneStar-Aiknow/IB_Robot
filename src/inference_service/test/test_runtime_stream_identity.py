"""Neutral stage identity survives request composition without a scheduler."""

from dataclasses import FrozenInstanceError

import pytest

from inference_service.unified_runtime import ModelRequest
from inference_service.unified_runtime.stage_ticket import StageTicket


def test_stage_ticket_round_trip_preserves_inputs_and_metadata():
    tensor = object()
    request = ModelRequest({"value": tensor}, {"role": "encoder", "stream": "first"})
    ticket = StageTicket(2, "snapshot", 7)
    attached = ticket.attach(request)
    assert StageTicket.from_request(attached) == ticket
    assert attached.inputs["value"] is tensor
    assert attached.metadata["role"] == "encoder" and attached.metadata["stream"] == "first"
    assert "stage_ticket" not in request.metadata
    with pytest.raises(FrozenInstanceError):
        ticket.generation = 3
    with pytest.raises(TypeError):
        attached.metadata["stage_ticket"]["generation"] = 3


@pytest.mark.parametrize("args", [(0,), (True,), (1, "", 1), (1, "snapshot", 0), (1, "snapshot", -1)])
def test_invalid_stage_identity_is_rejected_before_dispatch(args):
    with pytest.raises(ValueError):
        StageTicket(*args)
