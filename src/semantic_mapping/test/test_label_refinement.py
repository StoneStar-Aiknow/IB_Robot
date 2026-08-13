"""Tests for optional cloud label refinement."""

import json

import numpy as np
import pytest

from semantic_mapping.association import SemanticTrack
from semantic_mapping.label_refinement import (
    CloudLabelRefiner,
    LabelRefinementRejected,
    apply_refinement,
    parse_refinement_response,
    record_refinement_rejection,
    should_refine_label,
)
from semantic_mapping.representative_view import RepresentativeViewStore


def _track() -> SemanticTrack:
    return SemanticTrack(
        object_id="object-1",
        label="sky",
        confidence=0.72,
        position=np.zeros(3),
        size=np.ones(3),
        point_count=10,
        first_seen_ns=1,
        last_seen_ns=1,
    )


def test_refinement_trigger_is_scene_specific() -> None:
    assert should_refine_label("unlabeled", 0.0, [], 0.7)
    assert should_refine_label("Sky", 0.9, ["sky", "traffic light"], 0.7)
    assert should_refine_label("container", 0.69, [], 0.7)
    assert should_refine_label("bin", 0.85, [], 0.7, inconsistent=True)
    assert not should_refine_label("bin", 0.85, ["sky", "traffic light"], 0.7)


def test_refinement_response_is_strict_and_auditable() -> None:
    candidates = (("sky", 0.72), ("bin", 0.65))
    result = parse_refinement_response(
        '{"label":"bin","confidence":0.93}',
        model_identity="cloud-model-v1",
        candidates=candidates,
        min_confidence=0.8,
        created_ns=123,
    )
    track = _track()
    apply_refinement(track, result)

    assert track.label == "bin"
    assert track.confidence == pytest.approx(0.93)
    assert track.attributes["label_refinement"]["previous_label"] == "sky"
    assert track.attributes["label_refinement"]["model_identity"] == "cloud-model-v1"
    assert json.dumps(track.attributes)


def test_candidate_match_is_derived_locally() -> None:
    result = parse_refinement_response(
        '{"label":"banana","confidence":0.92}',
        model_identity="cloud-model-v1",
        candidates=(("food", 0.8),),
        min_confidence=0.8,
    )

    assert result.label == "banana"
    assert not result.candidate_match


@pytest.mark.parametrize(
    "content",
    [
        "bin",
        '{"label":"bin","confidence":0.9,"candidate_match":false}',
        '{"label":"bin","confidence":1.2}',
        '{"label":"unknown!","confidence":0.9}',
    ],
)
def test_refinement_response_rejects_invalid_output(content: str) -> None:
    with pytest.raises(ValueError):
        parse_refinement_response(
            content,
            model_identity="cloud-model-v1",
            candidates=(),
            min_confidence=0.8,
        )


def test_policy_rejection_explains_and_persists_cloud_result() -> None:
    track = _track()
    candidates = (("sky", 0.72),)
    with pytest.raises(LabelRefinementRejected, match="cloud label 'bin' confidence 0.700.*threshold 0.800") as caught:
        parse_refinement_response(
            '{"label":"bin","confidence":0.7}',
            model_identity="cloud-model-v1",
            candidates=candidates,
            min_confidence=0.8,
        )

    record_refinement_rejection(
        track,
        candidates=candidates,
        model_identity="cloud-model-v1",
        error=caught.value,
    )

    rejection = track.attributes["label_refinement_last_rejection"]
    assert rejection["ram_label"] == "sky"
    assert rejection["cloud_label"] == "bin"
    assert rejection["cloud_confidence"] == pytest.approx(0.7)
    assert rejection["failure_kind"] == "policy_rejected"


def test_cloud_refiner_sends_masked_crop_and_candidates() -> None:
    class Client:
        def chat(self, prompt, **kwargs):
            assert "sky (0.720)" in prompt
            assert "input_image" not in prompt
            assert kwargs["image"].startswith(b"\xff\xd8\xff")
            assert kwargs["model"] == "vision-model"
            return {"status": "ok", "content": '{"label":"bin","confidence":0.91}'}

    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    view = RepresentativeViewStore.create("object-1", 1, 0.7, image, mask, np.array([4, 4, 16, 16]))
    refiner = CloudLabelRefiner(
        Client(),
        model="vision-model",
        model_identity="vision-model-v1",
        prompt="Identify the object.",
        min_confidence=0.8,
    )

    assert refiner.refine(view, (("sky", 0.72),)).label == "bin"
