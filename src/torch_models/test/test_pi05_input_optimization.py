from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torch_models.pi05_ascend_310p.input_optimization import (
    LANGUAGE_TOKEN_LIMIT_ENV,
    LanguageCapacityStep,
    configure_input_processors,
    language_token_limit,
    resize_host_images,
    trim_language_padding,
)


def test_language_capacity_is_explicit_and_bounded(monkeypatch):
    monkeypatch.delenv(LANGUAGE_TOKEN_LIMIT_ENV, raising=False)
    assert language_token_limit(200) is None
    for invalid in ("0", "-1", "201", "true"):
        monkeypatch.setenv(LANGUAGE_TOKEN_LIMIT_ENV, invalid)
        with pytest.raises(ValueError):
            language_token_limit(200)
    monkeypatch.setenv(LANGUAGE_TOKEN_LIMIT_ENV, "48")
    assert language_token_limit(200) == 48


def test_trimming_preserves_valid_tokens_and_holes():
    tokens = torch.tensor([[9, 0, 7, 6, 5, 4]])
    mask = torch.tensor([[True, True, False, True, False, False]])
    trimmed, trimmed_mask = trim_language_padding(tokens, mask, 4)
    torch.testing.assert_close(trimmed, tokens[:, :4])
    torch.testing.assert_close(trimmed_mask, mask[:, :4])
    assert trimmed.is_contiguous() and trimmed_mask.is_contiguous()
    assert tokens.shape == (1, 6)


def test_overflow_rejects_request_without_truncating_valid_tokens():
    tokens = torch.zeros(2, 6, dtype=torch.int64)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[1, 5] = True
    with pytest.raises(ValueError, match="exceeds.*4-token"):
        trim_language_padding(tokens, mask, 4)


def test_incompatible_input_shapes_rejected():
    with pytest.raises(ValueError, match="same.*shape"):
        trim_language_padding(torch.zeros(1, 6), torch.ones(6, dtype=torch.bool), 4)
    with pytest.raises(ValueError, match="shorter"):
        trim_language_padding(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool), 4)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("channels_first", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_cpu_resize_matches_bilinear_padding(batched, channels_first, contiguous):
    from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import resize_with_pad_torch

    image = torch.linspace(0, 1, 7 * 11 * 3).reshape(7, 11, 3)
    if channels_first:
        image = image.permute(2, 0, 1)
    if batched:
        image = image.unsqueeze(0)
    if contiguous:
        image = image.contiguous()
    config = SimpleNamespace(ascend_cpu_image_resize=True, image_resolution=(8, 8), image_features={"image": None})
    inputs = {"image": image, "task": "pick"}
    if channels_first and image.is_contiguous():
        expected = image
    else:
        expected = resize_with_pad_torch(image, 8, 8)
        if not batched:
            expected = expected[0]

    result = resize_host_images(inputs, config=config)

    torch.testing.assert_close(result["image"], expected, atol=1e-6, rtol=1e-5)
    assert result["image"].is_contiguous()
    assert inputs["image"] is image
    assert result["task"] == "pick"


def test_cpu_resize_disabled_does_not_transform_inputs():
    inputs = {"image": np.zeros((7, 11, 3), dtype=np.uint8)}
    assert resize_host_images(inputs, config=SimpleNamespace()) is inputs


def test_cpu_resize_rejects_unnormalized_integer_input():
    config = SimpleNamespace(ascend_cpu_image_resize=True, image_resolution=(8, 8), image_features={"image": None})
    with pytest.raises(ValueError, match="float32 host images"):
        resize_host_images({"image": np.zeros((7, 11, 3), dtype=np.uint8)}, config=config)


def test_language_step_is_before_upload_and_preserves_input_transition():
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.types import TransitionKey
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    marker = object()
    device = DeviceProcessorStep(device="cpu")
    processor = SimpleNamespace(steps=[marker, device])
    configure_input_processors(
        preprocessor=processor, postprocessor=None, config=SimpleNamespace(ascend_language_token_limit=4)
    )
    assert processor.steps[0] is marker and processor.steps[-1] is device
    step = processor.steps[1]
    assert isinstance(step, LanguageCapacityStep)
    source = {
        TransitionKey.OBSERVATION: {
            OBS_LANGUAGE_TOKENS: torch.arange(6)[None],
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True, False, False, False]]),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": "pick"},
    }
    result = step(source)
    assert result[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS].shape == (1, 4)
    assert source[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS].shape == (1, 6)
    assert result[TransitionKey.COMPLEMENTARY_DATA] == source[TransitionKey.COMPLEMENTARY_DATA]


def test_language_step_requires_known_upload_boundary():
    with pytest.raises(ValueError, match="exactly one device processor"):
        configure_input_processors(
            preprocessor=SimpleNamespace(steps=[]),
            postprocessor=None,
            config=SimpleNamespace(ascend_language_token_limit=48),
        )


def test_cpu_output_preserves_postprocessor_order_and_dtype():
    from lerobot.processor.device_processor import DeviceProcessorStep

    transform = object()
    original = DeviceProcessorStep(device="cpu", float_dtype="float32")
    postprocessor = SimpleNamespace(steps=[transform, original])
    configure_input_processors(
        preprocessor=SimpleNamespace(steps=[]),
        postprocessor=postprocessor,
        config=SimpleNamespace(ascend_cpu_output=True),
    )
    assert postprocessor.steps[0] is transform
    assert postprocessor.steps[1].tensor_device.type == "cpu"
    assert postprocessor.steps[1].float_dtype == "float32"
    assert postprocessor.steps[1] is not original
