from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

from torch_models.pi05_ascend_310p.input_optimization import (
    LANGUAGE_TOKEN_LIMIT_ENV,
    LanguageCapacityStep,
    configure_input_processors,
    language_token_limit,
    resize_host_images,
    trim_language_padding,
)


@pytest.fixture
def tokenizer_step():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "word": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    return TokenizerProcessorStep(tokenizer=tokenizer, max_length=200, padding="max_length", truncation=True)


def language_pipeline(tokenizer_step, capacity):
    pipeline = PolicyProcessorPipeline(steps=[tokenizer_step, DeviceProcessorStep(device="cpu")])
    configure_input_processors(
        preprocessor=pipeline, postprocessor=None, config=SimpleNamespace(ascend_language_token_limit=capacity)
    )
    return pipeline


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


def test_language_step_is_before_upload_and_preserves_input_transition(tokenizer_step):
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.types import TransitionKey
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    marker = object()
    device = DeviceProcessorStep(device="cpu")
    processor = SimpleNamespace(steps=[marker, tokenizer_step, device])
    configure_input_processors(
        preprocessor=processor, postprocessor=None, config=SimpleNamespace(ascend_language_token_limit=4)
    )
    assert processor.steps[0] is marker and processor.steps[-1] is device
    step = processor.steps[2]
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


@pytest.mark.parametrize("capacity", [48, 200])
@pytest.mark.parametrize("prompt_lengths", [(201,), (3, 201)])
def test_complete_tokenization_rejects_overflow_before_any_truncation(tokenizer_step, capacity, prompt_lengths):
    pipeline = language_pipeline(tokenizer_step, capacity)
    tasks = [" ".join(["word"] * length) for length in prompt_lengths]
    with pytest.raises(ValueError, match=f"exceeds the configured {capacity}-token capacity"):
        pipeline({"task": tasks})


@pytest.mark.parametrize("capacity", [48, 200])
def test_tokenization_pads_short_prompts_to_static_capacity(tokenizer_step, capacity):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    pipeline = language_pipeline(tokenizer_step, capacity)
    result = pipeline({"task": ["word", "word word word"]})
    expected_tokens = torch.ones((2, capacity), dtype=torch.int64)
    expected_tokens[0, :1] = 2
    expected_tokens[1, :3] = 2
    expected_mask = torch.zeros((2, capacity), dtype=torch.bool)
    expected_mask[0, :1] = True
    expected_mask[1, :3] = True
    torch.testing.assert_close(result[OBS_LANGUAGE_TOKENS], expected_tokens)
    torch.testing.assert_close(result[OBS_LANGUAGE_ATTENTION_MASK], expected_mask)


@pytest.mark.parametrize("capacity", [48, 200])
def test_tokenization_keeps_exact_capacity_prompts(tokenizer_step, capacity):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    result = language_pipeline(tokenizer_step, capacity)({"task": " ".join(["word"] * capacity)})
    torch.testing.assert_close(result[OBS_LANGUAGE_TOKENS], torch.full((1, capacity), 2, dtype=torch.int64))
    assert bool(result[OBS_LANGUAGE_ATTENTION_MASK].all())


def test_language_capacity_requires_tokenizer_before_upload(tokenizer_step):
    for steps in ([DeviceProcessorStep(device="cpu")], [DeviceProcessorStep(device="cpu"), tokenizer_step]):
        with pytest.raises(ValueError, match="exactly one tokenizer before the device"):
            configure_input_processors(
                preprocessor=SimpleNamespace(steps=steps),
                postprocessor=None,
                config=SimpleNamespace(ascend_language_token_limit=48),
            )


def test_language_capacity_rejects_left_padding(tokenizer_step):
    tokenizer_step.padding_side = "left"
    with pytest.raises(ValueError, match="right tokenizer padding"):
        language_pipeline(tokenizer_step, 48)


def test_cpu_resize_checks_loaded_normalizer_not_only_model_config():
    image = "observation.images.top"
    feature = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4))
    config = SimpleNamespace(
        ascend_cpu_image_resize=True,
        image_features={image: feature},
        normalization_mapping={FeatureType.VISUAL: NormalizationMode.IDENTITY},
    )
    normalizer = NormalizerProcessorStep(
        features={image: feature},
        norm_map={FeatureType.VISUAL: NormalizationMode.MEAN_STD},
        stats={image: {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
    )
    identity = NormalizerProcessorStep(
        features={image: feature}, norm_map={FeatureType.VISUAL: NormalizationMode.IDENTITY}
    )
    with pytest.raises(ValueError, match="IDENTITY normalization in the loaded preprocessor"):
        configure_input_processors(
            preprocessor=SimpleNamespace(steps=[identity, normalizer]), postprocessor=None, config=config
        )


@pytest.mark.parametrize("exclude_image", [False, True])
def test_cpu_resize_allows_identity_visual_processing_and_state_normalization(exclude_image):
    image = "observation.images.top"
    features = {
        image: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map={
            FeatureType.VISUAL: NormalizationMode.MEAN_STD if exclude_image else NormalizationMode.IDENTITY,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
        },
        normalize_observation_keys={"observation.state"} if exclude_image else None,
    )
    processor = SimpleNamespace(steps=[normalizer])
    configure_input_processors(
        preprocessor=processor,
        postprocessor=None,
        config=SimpleNamespace(ascend_cpu_image_resize=True, image_features={image: features[image]}),
    )
    assert processor.steps == [normalizer]


def test_native_image_path_does_not_require_identity_normalization():
    image = "observation.images.top"
    feature = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4))
    normalizer = NormalizerProcessorStep(
        features={image: feature}, norm_map={FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    )
    configure_input_processors(
        preprocessor=SimpleNamespace(steps=[normalizer]),
        postprocessor=None,
        config=SimpleNamespace(ascend_cpu_image_resize=False),
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
