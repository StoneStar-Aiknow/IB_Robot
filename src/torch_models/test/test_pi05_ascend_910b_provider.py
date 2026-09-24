import json
from types import SimpleNamespace

import pytest
from lerobot.configs import PreTrainedConfig

from torch_models.pi05_ascend_910b import provider
from torch_models.pi05_ascend_910b.quantization import QuantizationConfig


@pytest.fixture
def local_bundle(tmp_path):
    (tmp_path / "model.safetensors").touch()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    return {
        "config": SimpleNamespace(quantization=None),
        "bundle_root": tmp_path,
        "tokenizer_path": str(tokenizer),
        "device_name": "Ascend910B3",
    }


def _runtime_version(name: str) -> str:
    return {
        "transformers": "5.5.4",
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0",
    }[name]


@pytest.mark.parametrize("device_name", ["Ascend910B3", "Ascend910_9362"])
def test_provider_accepts_verified_runtime(monkeypatch, local_bundle, device_name) -> None:
    monkeypatch.setattr(provider, "package_version", _runtime_version)
    local_bundle["device_name"] = device_name
    selected = provider.create_provider()

    selected.validate(**local_bundle)

    assert selected.load_config is provider.load_pi05_ascend_910b_config
    assert selected.load_options == {}


@pytest.mark.parametrize(
    ("distribution", "version", "message"),
    [
        ("transformers", "5.3.0", "Transformers >=5.4,<5.6"),
        ("torch", "2.11.0", "PyTorch 2.10.0"),
        ("torch-npu", "2.9.0", "Torch-NPU 2.10.0"),
    ],
)
def test_provider_rejects_unverified_runtime(monkeypatch, local_bundle, distribution, version, message) -> None:
    monkeypatch.setattr(
        provider,
        "package_version",
        lambda name: version if name == distribution else _runtime_version(name),
    )

    with pytest.raises(ValueError, match=message):
        provider.create_provider().validate(**local_bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device_name", "Ascend910A", "requires an Ascend910B/Ascend910_93"),
        ("device_name", "Ascend310P1", "requires an Ascend910B/Ascend910_93"),
        ("tokenizer_path", None, "bundled tokenizer"),
    ],
)
def test_provider_rejects_incompatible_bundle(local_bundle, field, value, message) -> None:
    local_bundle[field] = value
    with pytest.raises(ValueError, match=message):
        provider.create_provider().validate(**local_bundle)


def test_provider_rejects_absent_weights(local_bundle) -> None:
    (local_bundle["bundle_root"] / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="bundled model.safetensors"):
        provider.create_provider().validate(**local_bundle)


def test_config_loader_preserves_pi05_fields_and_attaches_910b_fields(tmp_path) -> None:
    payload = {
        "type": "pi05",
        "chunk_size": 12,
        "n_action_steps": 12,
        "dtype": "float32",
        "compile_model": True,
        "denoise_solver": "ab2",
        "compile_inference_graph": False,
        "quantization": {
            "quant_method": "int8_w8a8",
            "w_bits": 8,
            "a_bits": 8,
            "smooth": False,
            "group_size": 0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    config = provider.load_pi05_ascend_910b_config(
        bundle_path=tmp_path,
        config_type=PreTrainedConfig,
    )

    assert config.chunk_size == 12
    assert config.n_action_steps == 12
    assert config.device == "cpu"
    assert config.dtype == "bfloat16"
    assert config.compile_model is False
    assert config.denoise_solver == "ab2"
    assert config.compile_inference_graph is False
    assert isinstance(config.quantization, QuantizationConfig)

    configured = provider.configure_pi05_ascend_910b_config(config)

    assert configured is config
    assert configured.compile_model is False
    assert isinstance(configured.quantization, QuantizationConfig)
    assert configured.quantization.quant_method == "int8_w8a8"


@pytest.mark.parametrize("model_dtype", ["fp16", "fp32"])
def test_config_rejects_unverified_model_dtypes(model_dtype) -> None:
    with pytest.raises(ValueError, match="requires model_dtype"):
        provider.configure_pi05_ascend_910b_config(SimpleNamespace(quantization=None), model_dtype=model_dtype)


@pytest.mark.parametrize("quantized", [False, True])
def test_provider_enables_full_optimization_profile(quantized) -> None:
    calls = []
    policy = SimpleNamespace(
        config=SimpleNamespace(quantization=object() if quantized else None),
        prepare_inference_optimizations=lambda **kwargs: calls.append(kwargs) or {"ready": True},
    )

    result = provider.create_provider().prepare(
        policy=policy,
        deployment_fingerprint="fixture",
        torch_module=object(),
        torch_npu_module=object(),
        device_name="Ascend910B3",
    )

    assert result == {"ready": True}
    assert calls == [
        {
            "enable_npu_fused_ops": True,
            "enable_graph_compile": True,
            "enable_qkv_fusion": True,
            "enable_mlp_fusion": quantized,
            "mlp_fusion_scope": "prefix",
            "enable_shared_prefix_fias": True,
            "enable_adarms_bias_fusion": True,
            "adarms_bias_fusion_stage": "all",
        }
    ]
