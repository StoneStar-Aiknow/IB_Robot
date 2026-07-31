from concurrent.futures import Future
from types import SimpleNamespace

from sensor_msgs.msg import Image

from semantic_mapping.service_pipeline import ServiceFramePipeline, split_batches


class _Client:
    def __init__(self):
        self.requests = []
        self.futures = []

    def call_async(self, request):
        future = Future()
        self.requests.append(request)
        self.futures.append(future)
        return future


def _mask_response(count):
    detections = [SimpleNamespace(mask=Image()) for _ in range(count)]
    return SimpleNamespace(
        success=True, message="", detections=SimpleNamespace(detections=detections), model="sam-model"
    )


def test_split_batches_is_deterministic_and_bounded():
    assert [len(batch) for batch in split_batches(list(range(17)))] == [8, 8, 1]


def test_pipeline_fans_out_then_runs_bounded_siglip_batches():
    sam, ram, siglip = _Client(), _Client(), _Client()
    pipeline = ServiceFramePipeline(sam, ram, siglip)

    result = pipeline.process(Image())
    assert len(sam.requests) == 1
    assert len(ram.requests) == 1
    assert siglip.requests == []

    ram.futures[0].set_result(SimpleNamespace(success=True, message="", tags=["cup"], scores=[0.9], model="ram-model"))
    assert siglip.requests == []
    sam.futures[0].set_result(_mask_response(10))

    assert [len(request.masks) for request in siglip.requests] == [8, 2]
    first = [SimpleNamespace(mask_index=index) for index in range(8)]
    second = [SimpleNamespace(mask_index=index) for index in range(2)]
    siglip.futures[1].set_result(SimpleNamespace(success=True, message="", results=second, model="siglip-model-2"))
    assert not result.done()
    siglip.futures[0].set_result(SimpleNamespace(success=True, message="", results=first, model="siglip-model-1"))

    completed = result.result()
    assert [item.mask_index for item in completed.embeddings] == list(range(10))
    assert completed.tags == ("cup",)
    assert completed.model_diagnostics == {
        "sam2": ("sam-model",),
        "ram_plus": ("ram-model",),
        "siglip2_image": ("siglip-model-1", "siglip-model-2"),
    }


def test_pipeline_propagates_stage_failure_without_starting_siglip():
    sam, ram, siglip = _Client(), _Client(), _Client()
    result = ServiceFramePipeline(sam, ram, siglip).process(Image())
    sam.futures[0].set_result(SimpleNamespace(success=False, message="not ready"))

    assert "sam service failed: not ready" in str(result.exception())
    assert siglip.requests == []
