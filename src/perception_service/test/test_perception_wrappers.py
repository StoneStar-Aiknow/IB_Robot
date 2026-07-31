from types import SimpleNamespace

import numpy as np
import pytest

from perception_service.ram_plus_wrapper import RAMPlusWrapper
from perception_service.sam2_wrapper import SAM2Wrapper
from perception_service.siglip2_wrapper import SigLIP2Wrapper


class _AutomaticGenerator:
    @staticmethod
    def generate(image):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1:3, 2:5] = True
        return [
            {
                "segmentation": mask,
                "bbox": [2, 1, 3, 2],
                "area": 6,
                "predicted_iou": 0.9,
                "stability_score": 0.8,
            }
        ]


class _ImagePredictor:
    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, *, box, multimask_output):
        assert not multimask_output
        masks = np.ones((len(box), 1, *self.shape), dtype=bool)
        return masks, np.full((len(box), 1), 0.75), None


def test_sam2_wrapper_normalizes_automatic_and_box_results():
    wrapper = SAM2Wrapper(
        backend="cpu",
        automatic_generator=_AutomaticGenerator(),
        image_predictor=_ImagePredictor(),
    )
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    generated = wrapper.generate(image)
    segmented = wrapper.segment_boxes(image, np.asarray([[0, 0, 2, 3]], dtype=np.float32))

    assert generated[0].bbox_xyxy.tolist() == [2.0, 1.0, 5.0, 3.0]
    assert generated[0].mask.dtype == np.uint8
    assert segmented[0].mask.shape == (4, 6)
    assert segmented[0].score == pytest.approx(0.75)


def test_ram_plus_wrapper_filters_and_sorts_scored_tags():
    model = SimpleNamespace(
        class_threshold=np.asarray([0.4, 0.8, 0.4], dtype=np.float32),
        tag_list=np.asarray(["cup", "table", "bottle"]),
        delete_tag_index=[2],
    )
    wrapper = RAMPlusWrapper(
        backend="cpu",
        model=model,
        transform=lambda image: np.asarray(image),
        logits_inference=lambda image: np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    )

    results = wrapper.recognize(np.zeros((4, 4, 3), dtype=np.uint8))

    assert [(result.label, result.score) for result in results] == [
        ("table", pytest.approx(0.880797)),
        ("cup", pytest.approx(0.731059)),
    ]


def test_ram_plus_wrapper_honors_request_threshold():
    model = SimpleNamespace(
        class_threshold=np.asarray([0.1, 0.1], dtype=np.float32),
        tag_list=np.asarray(["cup", "table"]),
        delete_tag_index=[],
    )
    wrapper = RAMPlusWrapper(
        backend="cpu",
        model=model,
        transform=lambda image: np.asarray(image),
        logits_inference=lambda image: np.asarray([[1.0, 2.0]], dtype=np.float32),
    )

    assert wrapper.recognize(np.zeros((2, 2, 3), dtype=np.uint8), score_threshold=0.9) == []


def test_siglip2_wrapper_normalizes_and_matches_each_mask():
    wrapper = SigLIP2Wrapper(
        backend="cpu",
        image_encoder=lambda crops: np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        text_encoder=lambda prompts: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    masks = [np.eye(4, 5, dtype=np.uint8), np.ones((4, 5), dtype=np.uint8)]

    results = wrapper.encode(image, masks, ["cup", "table"])

    assert len(results) == 2
    assert np.linalg.norm(results[0].embedding) == pytest.approx(1.0)
    assert results[0].matched_label == "table"
    assert results[0].matched_score == pytest.approx(0.8)
    assert results[1].matched_label == "table"


def test_siglip2_text_encoding_is_normalized_bounded_and_uses_shared_prompts():
    prompts = []

    def encode_text(items):
        prompts.extend(items)
        return np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    wrapper = SigLIP2Wrapper(
        backend="cpu",
        image_encoder=lambda crops: np.ones((len(crops), 2), dtype=np.float32),
        text_encoder=encode_text,
    )

    features = wrapper.encode_text(["cup", "table"])

    assert prompts == ["This is a photo of cup.", "This is a photo of table."]
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    with pytest.raises(ValueError, match="limit 16"):
        wrapper.encode_text(["cup"] * 17)


def test_siglip2_wrapper_rejects_unbounded_or_empty_masks():
    wrapper = SigLIP2Wrapper(
        backend="cpu",
        image_encoder=lambda crops: np.ones((len(crops), 2), dtype=np.float32),
        text_encoder=lambda prompts: np.ones((len(prompts), 2), dtype=np.float32),
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="between 1 and 8"):
        wrapper.encode(image, [np.ones((2, 2), dtype=np.uint8)] * 9, [])
    with pytest.raises(ValueError, match="empty"):
        wrapper.encode(image, [np.zeros((2, 2), dtype=np.uint8)], [])


@pytest.mark.parametrize("wrapper", [SAM2Wrapper, SigLIP2Wrapper, RAMPlusWrapper])
def test_raw_wrappers_require_named_ascend_deployment(wrapper):
    with pytest.raises(RuntimeError, match="manifest named deployment"):
        wrapper(backend="ascend_om")
