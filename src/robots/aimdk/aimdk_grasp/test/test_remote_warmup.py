"""Startup checks infrastructure without detecting a default object."""

from types import SimpleNamespace

import pytest
from aimdk_grasp.backend import DEFAULTS
from aimdk_grasp.remote_head import _prepare_remote_models

from manipulation_execution.grasp_pipeline.contracts import GraspFailure


def backends(failure=None):
    calls = []

    def ready(**kwargs):
        assert not kwargs["cancelled"]()
        calls.append("services")
        if failure == "service":
            raise GraspFailure("RPC_FAILED", "model service unavailable")

    def capture(**kwargs):
        assert not kwargs["cancelled"]()
        calls.append("capture")
        if failure == "camera":
            raise GraspFailure("OBSERVATION_UNAVAILABLE", "no fresh image-time feedback")
        return object()

    return SimpleNamespace(capture=capture), SimpleNamespace(wait_ready=ready), calls


def test_startup_has_no_default_object_or_proposer_dependency():
    camera, services, calls = backends()
    _prepare_remote_models(camera, services)
    assert calls == ["services", "capture"]
    assert DEFAULTS["warmup_prompt"] == ""


@pytest.mark.parametrize("failure", ["service", "camera"])
def test_unavailable_services_or_stale_camera_still_block_startup(failure):
    camera, services, calls = backends(failure)
    with pytest.raises(RuntimeError, match="startup readiness"):
        _prepare_remote_models(camera, services)
    assert calls == (["services"] if failure == "service" else ["services", "capture"])
