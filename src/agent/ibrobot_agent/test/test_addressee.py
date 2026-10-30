"""The addressee gate's checks on their own, without the voice bridge around them."""

from __future__ import annotations

import math

import pytest

from ibrobot_agent.addressee import SpeakerCheck, Voiceprint, cosine_similarity

#: A unit voiceprint.
ALICE = (1.0, 0.0, 0.0)
MODEL = "campplus@test"


def _vp(vector, model_id=MODEL):
    return Voiceprint(tuple(vector), model_id)


def test_cosine_similarity_of_the_same_voice_is_one():
    assert cosine_similarity(ALICE, ALICE) == pytest.approx(1.0)
    assert cosine_similarity((3.0, 4.0), (6.0, 8.0)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param((), (), id="both empty"),
        pytest.param(ALICE, (1.0, 0.0), id="different model dimension"),
        pytest.param(ALICE, (0.0, 0.0, 0.0), id="zero vector"),
    ],
)
def test_incomparable_vectors_are_not_evidence(left, right):
    """None, not a low score: the caller must treat these as a missing measurement."""
    assert cosine_similarity(left, right) is None


def test_the_speaker_check_accepts_what_it_cannot_compare():
    agrees, detail = SpeakerCheck(0.5, 2.5).verdict(_vp(ALICE), _vp((1.0, 0.0)))

    assert agrees
    assert detail == "not comparable"


def test_the_speaker_check_does_not_compare_across_models():
    """Same dimension, another model: a confident low score would be a false rejection."""
    agrees, detail = SpeakerCheck(0.5, 2.5).verdict(_vp(ALICE), _vp((0.0, 1.0, 0.0), "campplus@another"))

    assert agrees
    assert detail == "not comparable"


@pytest.mark.parametrize(
    ("measurement", "agrees"),
    [
        pytest.param((0.8, 0.6, 0.0), True, id="same voice, 0.80 similarity"),
        pytest.param((0.4, 0.92, 0.0), False, id="other voice, 0.40 similarity"),
    ],
)
def test_the_speaker_check_compares_against_its_threshold(measurement, agrees):
    assert SpeakerCheck(0.5, 2.5).verdict(_vp(ALICE), _vp(measurement))[0] is agrees


def test_merge_weights_every_accepted_utterance_equally():
    # No threshold or margin, so the arithmetic is tested on its own.
    check = SpeakerCheck(0.0, 6.0)

    merged = check.merge(_vp((1.0, 0.0)), _vp((0.0, 2.0)))

    assert merged.vector == pytest.approx((1.0, 1.0))
    assert merged.model_id == MODEL
    assert cosine_similarity(merged.vector, (1.0, 0.0)) == pytest.approx(math.sqrt(0.5))


@pytest.mark.parametrize(
    "measurement",
    [
        pytest.param(_vp((1.0, 0.0, 0.0)), id="another dimension"),
        pytest.param(_vp((1.0, 0.0), "campplus@another"), id="another model"),
    ],
)
def test_merge_ignores_a_measurement_it_cannot_compare(measurement):
    assert SpeakerCheck(0.0, 6.0).merge(_vp((1.0, 0.0)), measurement) == _vp((1.0, 0.0))
