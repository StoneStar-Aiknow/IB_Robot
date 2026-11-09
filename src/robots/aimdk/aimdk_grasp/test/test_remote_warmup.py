"""Startup checks infrastructure without detecting a default object."""

import json
from types import SimpleNamespace

import numpy as np
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


class ModelServices:
    """Records each model call with the per-call budget it ran under; ``fail`` names a model that refuses."""

    def __init__(self, *, detections=(), fail=None, grasp_rpc_s=(0.6,)):
        self.timeout_s, self.calls, self.seen = 4.0, [], []
        self._grasp_rpc_s = list(grasp_rpc_s)
        self._segment, self._detections, self._fail = "segment", list(detections), fail

    def wait_ready(self, **kwargs):
        pass

    def _record(self, name):
        self.seen.append((name, self.timeout_s))
        rpc_s = self._grasp_rpc_s.pop(0) if name == "generate_grasps" else 7.9
        self.calls.append(dict(service=name, rpc_s=rpc_s, inference_s=7.5, transport_s=0.4))
        if self._fail == name:
            raise GraspFailure("RPC_FAILED", f"/{name} timed out")

    def detect(self, rgb, frame_id, stamp_ns, prompt, *, box_threshold, text_threshold):
        self.prompt = prompt
        self._record("detect")
        return self._detections

    def masks(self, rgb, frame_id, stamp_ns, detections):
        self.boxes = [detection.box_xyxy for detection in detections]
        assert all(detection.mask is None for detection in detections)  # the segmenter itself must run
        self._record("segment")
        mask = np.zeros(np.shape(rgb)[:2], bool)
        mask[4:8, 4:8] = True
        return [mask]

    def generate_grasps(self, points, frame_id, stamp_ns, *, max_grasps, min_confidence):
        self.points = np.asarray(points)
        self._record("generate_grasps")
        return np.zeros((3, 4, 4)), np.zeros(3)


def observation(depth=1.0):
    return SimpleNamespace(
        rgb=np.zeros((10, 10, 3), np.uint8),
        depth_m=np.full((10, 10), depth),
        intrinsics=np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]]),
        frame_id="camera",
        stamp_ns=7,
    )


def warm(services, seen_observation, capsys):
    from aimdk_grasp.proposer import PerceptionSettings

    camera = SimpleNamespace(capture=lambda **kwargs: seen_observation)
    _prepare_remote_models(camera, services, PerceptionSettings(min_points=10))
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith('{"grasp_warmup"')]
    assert len(lines) == 1
    return json.loads(lines[0])["grasp_warmup"]


def test_each_model_runs_once_at_startup_under_a_cold_start_budget(capsys):
    """11-09: four of five generate_grasps timeouts (4 s) were each a stack's first pick."""
    services = ModelServices()
    report = warm(services, observation(), capsys)
    assert [name for name, _ in services.seen] == ["detect", "segment", "generate_grasps"]
    assert all(budget == 60.0 for _, budget in services.seen) and services.timeout_s == 4.0  # restored
    assert "an apple" in services.prompt
    # Nothing detected: the image centre's box still runs the segmenter; its masked depth feeds the grasps.
    np.testing.assert_allclose(services.boxes[0], [4.0, 4.0, 6.0, 6.0])
    assert report["generate_grasps"]["synthetic"] is False and len(services.points) == 16
    assert report["detect"] == dict(detections=0, rpc_s=7.9, inference_s=7.5, transport_s=0.4)
    assert report["segment"]["mask_pixels"] == 16 and report["generate_grasps"]["grasps"] == 3


def test_a_failed_or_empty_warmup_never_blocks_startup(capsys):
    services = ModelServices(fail="segment")
    report = warm(services, observation(depth=np.nan), capsys)
    assert report["segment"]["error"].startswith("RPC_FAILED")
    # No mask: generate_grasps still runs, on the synthetic sphere.
    assert report["generate_grasps"]["synthetic"] is True and len(services.points) == 2000
    assert services.timeout_s == 4.0


@pytest.mark.parametrize(
    "rpc_s,calls",
    [((7.5, 4.1, 0.6), 3), ((0.6,), 1), ((7.5, 7.5, 7.5, 7.5), 4)],
    ids=["until_fast", "already_fast", "at_most_four"],
)
def test_generate_grasps_is_warmed_until_it_fits_the_goal_budget(capsys, rpc_s, calls):
    """Live 2026-11-09: 7.5 s, ~4 s, then 0.6 s; one warmup call left the first pick over its 4 s budget."""
    services = ModelServices(grasp_rpc_s=rpc_s)
    report = warm(services, observation(), capsys)
    assert [name for name, _ in services.seen].count("generate_grasps") == calls
    assert report["generate_grasps"]["rpc_series_s"] == list(rpc_s[:calls])
    assert services.timeout_s == 4.0


def test_the_warmup_prompt_fits_the_310p_detector():
    from aimdk_grasp.remote_head import WARMUP_PROMPT

    assert len(WARMUP_PROMPT.split()) + 2 <= 8  # words plus the two special tokens
