from concurrent.futures import Future
from types import SimpleNamespace

from sensor_msgs.msg import Image

from semantic_mapping.service_pipeline import ServiceFramePipeline, ram_mask_candidates, select_ram_label, split_batches


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

    ram.futures[0].set_result(
        SimpleNamespace(success=True, message="", tags=["office"], scores=[0.9], model="ram-model")
    )
    assert siglip.requests == []
    sam.futures[0].set_result(_mask_response(10))

    assert len(ram.requests) == 3
    assert not ram.requests[1].include_image
    assert [len(request.masks) for request in ram.requests[1:]] == [8, 2]
    assert ram.requests[1].max_mask_candidates == 5
    ram.futures[1].set_result(
        SimpleNamespace(
            success=True,
            message="",
            mask_tag_counts=[1] * 8,
            mask_tags=["cucumber"] * 8,
            mask_scores=[0.95] * 8,
            model="ram-model",
        )
    )
    assert siglip.requests == []
    ram.futures[2].set_result(
        SimpleNamespace(
            success=True,
            message="",
            mask_tag_counts=[1] * 2,
            mask_tags=["cucumber"] * 2,
            mask_scores=[0.95] * 2,
            model="ram-model",
        )
    )

    assert [len(request.masks) for request in siglip.requests] == [8, 2]
    assert siglip.requests[0].candidate_labels == []
    first = [SimpleNamespace(mask_index=index, matched_label="office", matched_score=0.1) for index in range(8)]
    second = [SimpleNamespace(mask_index=index, matched_label="office", matched_score=0.1) for index in range(2)]
    siglip.futures[1].set_result(SimpleNamespace(success=True, message="", results=second, model="siglip-model-2"))
    assert not result.done()
    siglip.futures[0].set_result(SimpleNamespace(success=True, message="", results=first, model="siglip-model-1"))

    completed = result.result()
    assert [item.mask_index for item in completed.embeddings] == list(range(10))
    assert completed.tags == ("office",)
    assert completed.mask_tag_counts == (1,) * 10
    assert completed.mask_tags == ("cucumber",) * 10
    assert [item.matched_label for item in completed.embeddings] == ["office"] * 10
    assert [item.matched_score for item in completed.embeddings] == [0.1] * 10
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


def test_pipeline_filters_sam_masks_before_local_inference():
    sam, ram, siglip = _Client(), _Client(), _Client()
    result = ServiceFramePipeline(sam, ram, siglip).process(Image(), mask_selector=lambda detections: [1, 3])
    ram.futures[0].set_result(SimpleNamespace(success=True, message="", tags=[], scores=[], model="ram-model"))
    sam.futures[0].set_result(_mask_response(4))

    assert len(ram.requests[1].masks) == 2
    ram.futures[1].set_result(
        SimpleNamespace(
            success=True,
            message="",
            mask_tag_counts=[1, 1],
            mask_tags=["banana", "bottle"],
            mask_scores=[0.9, 0.8],
            model="ram-model",
        )
    )
    assert len(siglip.requests[0].masks) == 2
    siglip.futures[0].set_result(SimpleNamespace(success=True, message="", results=[], model="siglip-model"))

    assert len(result.result().masks.detections) == 2


def test_pipeline_preserves_siglip_identity_when_all_masks_are_filtered():
    sam, ram, siglip = _Client(), _Client(), _Client()
    result = ServiceFramePipeline(sam, ram, siglip).process(Image(), mask_selector=lambda detections: [])
    ram.futures[0].set_result(SimpleNamespace(success=True, message="", tags=[], scores=[], model="ram-model"))
    sam.futures[0].set_result(_mask_response(2))

    assert len(siglip.requests) == 1
    assert siglip.requests[0].masks == []
    siglip.futures[0].set_result(SimpleNamespace(success=True, message="", results=[], model="siglip-model"))

    completed = result.result()
    assert completed.embeddings == ()
    assert completed.model_diagnostics["siglip2_image"] == ("siglip-model",)


def test_pipeline_encodes_embeddings_without_using_siglip_as_label_owner():
    sam, ram, siglip = _Client(), _Client(), _Client()
    result = ServiceFramePipeline(sam, ram, siglip).process(Image())
    ram.futures[0].set_result(SimpleNamespace(success=True, message="", tags=[], scores=[], model="ram-model"))
    sam.futures[0].set_result(_mask_response(2))
    ram.futures[1].set_result(
        SimpleNamespace(
            success=True,
            message="",
            mask_tag_counts=[0, 0],
            mask_tags=[],
            mask_scores=[],
            model="ram-model",
        )
    )
    assert siglip.requests[0].candidate_labels == []
    siglip.futures[0].set_result(
        SimpleNamespace(
            success=True,
            message="",
            results=[
                SimpleNamespace(mask_index=index, matched_label="cucumber", matched_score=0.06) for index in range(2)
            ],
            model="siglip-model",
        )
    )

    completed = result.result()
    assert completed.tags == ()
    assert [item.matched_label for item in completed.embeddings] == ["cucumber", "cucumber"]
    assert completed.mask_tags == ()


def test_pipeline_does_not_send_ram_labels_to_siglip():
    sam, ram, siglip = _Client(), _Client(), _Client()
    ServiceFramePipeline(sam, ram, siglip).process(Image())
    ram.futures[0].set_result(
        SimpleNamespace(
            success=True,
            message="",
            tags=["square", "projection screen", "white"],
            scores=[0.9, 0.8, 0.7],
            model="ram-model",
        )
    )
    sam.futures[0].set_result(_mask_response(1))
    ram.futures[1].set_result(
        SimpleNamespace(
            success=True,
            message="",
            mask_tag_counts=[1],
            mask_tags=["projection screen"],
            mask_scores=[0.8],
            model="ram-model",
        )
    )

    assert siglip.requests[0].candidate_labels == []


def test_pipeline_does_not_share_local_labels_between_masks():
    sam, ram, siglip = _Client(), _Client(), _Client()
    ServiceFramePipeline(sam, ram, siglip).process(Image())
    ram.futures[0].set_result(
        SimpleNamespace(success=True, message="", tags=["office"], scores=[0.9], model="ram-model")
    )
    sam.futures[0].set_result(_mask_response(2))
    ram.futures[1].set_result(
        SimpleNamespace(
            success=True,
            message="",
            mask_tag_counts=[2, 1],
            mask_tags=["flag", "banner", "plant"],
            mask_scores=[0.95, 0.8, 0.9],
            model="ram-model",
        )
    )

    assert len(siglip.requests) == 1
    assert siglip.requests[0].candidate_labels == []


def test_select_ram_label_prefers_local_entity_and_preserves_its_score():
    assert select_ram_label(1, [1, 2], ["chair", "banana", "fruit"], [0.8, 0.94, 0.7], 0.2) == (
        "banana",
        0.94,
    )


def test_select_ram_label_sorts_candidates_before_selection():
    assert select_ram_label(0, [3], ["food", "banana", "fruit"], [0.7, 0.95, 0.8], 0.2) == (
        "banana",
        0.95,
    )


def test_ram_mask_candidates_returns_only_the_requested_flattened_slice():
    assert ram_mask_candidates(1, [2, 1], ["banana", "fruit", "bin"], [0.9, 0.7, 0.8]) == (("bin", 0.8),)


def test_select_ram_label_rejects_low_confidence_and_empty_results():
    assert select_ram_label(0, [1], ["banana"], [0.1], 0.2) == ("unlabeled", 0.0)
    assert select_ram_label(1, [1], ["banana"], [0.9], 0.2) == ("unlabeled", 0.0)


def test_select_ram_label_skips_scene_exclusions():
    assert select_ram_label(
        0,
        [3],
        ["animal", "image", "banana"],
        [0.95, 0.9, 0.82],
        0.2,
        ["animal", "image"],
    ) == ("banana", 0.82)
    assert ram_mask_candidates(
        0,
        [3],
        ["animal", "image", "banana"],
        [0.95, 0.9, 0.82],
        ["animal", "image"],
    ) == (("banana", 0.82),)
