from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_models.pi05_ascend_310p.inference_config import (
    DenoisingConfig,
    PI05AscendInferenceConfig,
    deployment_parameters,
)
from torch_models.pi05_ascend_310p.input_optimization import trim_language_padding
from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import (
    PI05Pytorch,
    _torchair_cache_dir,
    configure_pi05_ascend_310p_config,
)
from torch_models.pi05_ascend_310p.provider import create_provider


@pytest.fixture(autouse=True)
def clean_legacy_switches(monkeypatch):
    for name in (
        "IBROBOT_PI05_LANGUAGE_TOKEN_LIMIT",
        "IBROBOT_PI05_CPU_IMAGE_RESIZE",
        "IBROBOT_PI05_CPU_OUTPUT",
        "IBROBOT_PI05_VISION_FFN",
        "IBROBOT_PI05_GRAPH_COMPILE",
        "IBROBOT_PI05_NPU_FUSED_OPS",
        "LEROBOT_PI05_COMPILE_VISION_EMBED",
        "LEROBOT_PI05_ENABLE_VISION_NPU_FUSED_OPS",
    ):
        monkeypatch.delenv(name, raising=False)


def base_config():
    return SimpleNamespace(num_inference_steps=10, tokenizer_max_length=200, dtype="float32", rtc_config=None)


def test_deployments_use_independent_instance_settings_with_mode_specific_capacity():
    provider = create_provider()
    slow, fast = base_config(), base_config()
    for config, mode, steps, solver, capacity in (
        (slow, "fidelity", 10, "euler", 200),
        (fast, "fast", 3, "ab2", 48),
    ):
        options = provider.resolve_load_options(config=config, extra_parameters=deployment_parameters(mode))
        provider.configure_config(config, model_dtype="fp16")
        assert options["inference_config"] is config.ascend_inference_config
        assert config.num_inference_steps == steps
        assert config.ascend_inference_config.denoising.solver == solver
        assert config.ascend_language_token_limit == capacity
        assert config.ascend_cpu_image_resize and config.ascend_cpu_output
    assert slow.num_inference_steps == 10
    with pytest.raises(FrozenInstanceError):
        fast.ascend_inference_config.language_capacity = 46


@pytest.mark.parametrize("mode", ["fidelity", "fast"])
def test_fidelity_preserves_valid_tokens_beyond_fast_capacity(mode):
    settings = PI05AscendInferenceConfig.parse(deployment_parameters(mode))
    config = configure_pi05_ascend_310p_config(base_config(), inference_config=settings)
    tokens = torch.arange(200).reshape(1, 200)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    if mode == "fast":
        with pytest.raises(ValueError, match="exceeds the configured 48-token capacity"):
            trim_language_padding(tokens, mask, config.ascend_language_token_limit)
    else:
        actual_tokens, actual_mask = trim_language_padding(tokens, mask, config.ascend_language_token_limit)
        torch.testing.assert_close(actual_tokens, tokens)
        torch.testing.assert_close(actual_mask, mask)


@pytest.mark.parametrize(
    "parameters",
    [
        {"denoising": {"solver": "unknown"}},
        {"denoising": {"steps": True}},
        {"denoising": {"steps": 0}},
        {"denoising": {"solver": "ab2", "steps": 1}},
        {"denoising": {"schedule": "curvature"}},
        {"denoising": {"timesteps": [1, 0]}},
        {"denoising": None},
        {"language_capacity": 0},
        {"language_capacity": True},
        {"vision_ffn": "true"},
        {"action_output_device": "cuda"},
        {"image_preprocessing": "unknown"},
        {"unknown_model_option": 1},
    ],
)
def test_provider_owns_parameter_validation(parameters):
    with pytest.raises(ValueError):
        PI05AscendInferenceConfig.parse(parameters)


def test_explicit_settings_reject_conflicting_environment(monkeypatch):
    settings = PI05AscendInferenceConfig.parse(deployment_parameters("fast"))
    monkeypatch.setenv("IBROBOT_PI05_CPU_OUTPUT", "0")
    with pytest.raises(ValueError, match="conflicts"):
        configure_pi05_ascend_310p_config(base_config(), inference_config=settings)


def test_explicit_settings_reject_unsupported_precision_and_capacity():
    settings = PI05AscendInferenceConfig.parse(deployment_parameters("fast"))
    with pytest.raises(ValueError, match="model_dtype=fp16"):
        configure_pi05_ascend_310p_config(base_config(), model_dtype="fp32", inference_config=settings)
    config = base_config()
    config.tokenizer_max_length = 32
    with pytest.raises(ValueError, match="tokenizer_max_length"):
        configure_pi05_ascend_310p_config(config, inference_config=settings)


def test_legacy_settings_are_resolved_once(monkeypatch):
    monkeypatch.setenv("IBROBOT_PI05_LANGUAGE_TOKEN_LIMIT", "48")
    config = configure_pi05_ascend_310p_config(base_config())
    settings = config.ascend_inference_config
    monkeypatch.setenv("IBROBOT_PI05_LANGUAGE_TOKEN_LIMIT", "46")
    configure_pi05_ascend_310p_config(config)
    assert config.ascend_inference_config is settings
    assert config.ascend_language_token_limit == 48


def test_settings_separate_graph_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("IBROBOT_PI05_TORCHAIR_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p._cann_build_identity", lambda: "test-cann"
    )
    module = SimpleNamespace(__version__="2.5.1")
    common = dict(
        deployment_fingerprint="same-checkpoint",
        torch_module=module,
        torch_npu_module=module,
        device_name="Ascend310P1",
        model_dtype="float16",
        prefix_graph_mode="compiled_vision_split",
        npu_fused_ops=True,
        vision_npu_fused_ops=True,
        vision_ffn=True,
    )
    configs = [PI05AscendInferenceConfig.parse(deployment_parameters(mode)) for mode in ("fidelity", "fast")]
    paths = [
        _torchair_cache_dir(
            **common, language_capacity=config.language_capacity, inference_settings_key=config.cache_key
        )
        for config in configs
    ]
    assert paths[0] != paths[1]


def denoise_model(solver, steps):
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.inference_config = PI05AscendInferenceConfig(denoising=DenoisingConfig(solver=solver, steps=steps))
    model._action_denoise_dt = torch.tensor(-1.0 / steps)
    model._action_denoise_adarms_cond_table = 1.0 - torch.arange(steps).float() / steps
    model._action_denoise_adarms_modulation_steps = tuple(range(steps))
    model._prepare_denoise_mask_context_for_compile = lambda pads, x: (None, None)
    model.seen = []

    def velocity(*, x_t, adarms_cond, adarms_modulation, **kwargs):
        model.seen.append(adarms_modulation)
        return torch.ones_like(x_t) * adarms_cond

    model._denoise_step_with_prepared_mask_context = velocity
    return model


@pytest.mark.parametrize(
    ("solver", "steps", "expected"), [("euler", 10, -0.55), ("euler", 3, -2 / 3), ("ab2", 3, -5 / 9)]
)
def test_compiled_loop_math_call_count_and_request_local_history(solver, steps, expected):
    # Integrate v(t)=t from 1 to 0. Analytic integral is -0.5; AB2's Euler
    # startup gives -5/9 with three evaluations instead of Euler's -2/3.
    model = denoise_model(solver, steps)
    for initial in (0.0, 2.0, 0.0):
        model.seen.clear()
        noise = torch.full((1, 50, 32), initial)
        output = model._action_denoise_steps_for_compile(None, None, noise)
        torch.testing.assert_close(output, torch.full_like(noise, initial + expected))
        torch.testing.assert_close(noise, torch.full_like(noise, initial))
        assert model.seen == list(range(steps))


def test_eager_and_graph_denoising_use_the_same_ab2_rule():
    model = denoise_model("ab2", 3)
    model.config = SimpleNamespace(chunk_size=50, rtc_config=None)
    model._can_use_compiled_action_inference = lambda kwargs: False
    model._cast_action_tensor = lambda tensor: tensor
    model._action_fused_stage_timing_enabled = False
    model._start_action_fused_stage_timing = lambda **kwargs: None
    model._finish_action_fused_stage_timing = lambda record: None
    model._rtc_enabled = lambda: False
    model.rtc_processor = None
    model.embed_prefix = lambda *args: (
        torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    model._prepare_attention_masks_4d = lambda masks: masks
    model.paligemma_with_expert = SimpleNamespace(
        _npu_fused_inference_enabled=False,
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))),
        forward=lambda **kwargs: (None, None),
    )
    model.denoise_step = lambda *, x_t, timestep, **kwargs: torch.ones_like(x_t) * timestep
    noise = torch.zeros(1, 50, 32)
    eager = model.sample_actions(
        [], [], torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 2, dtype=torch.bool), noise=noise
    )
    graph = model._action_denoise_steps_for_compile(None, None, noise)
    torch.testing.assert_close(eager, graph)
