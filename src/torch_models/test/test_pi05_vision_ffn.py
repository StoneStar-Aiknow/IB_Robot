import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_models.pi05_ascend_310p.inference_config import PI05AscendInferenceConfig
from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import PI05Ascend310PPolicy
from torch_models.pi05_ascend_310p.vision_siglip_npu import PI05SiglipMLP


def make_mlp():
    return PI05SiglipMLP(SimpleNamespace(hidden_size=4, intermediate_size=8, hidden_act="gelu_pytorch_tanh"))


def add_test_weights(mlp):
    mlp.ffn_weight1 = nn.Parameter(mlp.fc1.weight.detach().t().contiguous(), requires_grad=False)
    mlp.ffn_weight2 = nn.Parameter(mlp.fc2.weight.detach().t().contiguous(), requires_grad=False)


@pytest.mark.parametrize(("training", "compiling"), [(False, False), (True, False), (True, True)])
def test_prepared_mlp_preserves_eager_and_training_math(monkeypatch, training, compiling):
    mlp = make_mlp().train(training)
    add_test_weights(mlp)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    inputs = torch.randn(2, 8, 4)

    expected = mlp.fc2(mlp.activation_fn(mlp.fc1(inputs)))
    torch.testing.assert_close(mlp(inputs), expected)


def test_compiled_mlp_preserves_batch_sequence_shape_and_bias(monkeypatch):
    mlp = make_mlp().eval()
    add_test_weights(mlp)
    inputs = torch.randn(2, 8, 4)
    expected = mlp.fc2(mlp.activation_fn(mlp.fc1(inputs)))
    calls = []

    def fake_ffn(x, weight1, weight2, activation, *, bias1, bias2, inner_precise):
        calls.append((x.shape, activation, inner_precise))
        return torch.nn.functional.gelu(x @ weight1 + bias1, approximate="tanh") @ weight2 + bias2

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_ffn=fake_ffn))
    torch.testing.assert_close(mlp(inputs), expected)
    assert calls == [(torch.Size([16, 4]), "gelu", 1)]


def test_unprepared_mlp_keeps_checkpoint_keys_and_rejects_cpu_packing():
    mlp = make_mlp()
    assert set(mlp.state_dict()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    with pytest.raises(ValueError, match="FP16 NPU"):
        mlp.prepare_npu_ffn()
    mlp.config.hidden_act = "relu"
    with pytest.raises(ValueError, match="gelu_pytorch_tanh"):
        mlp.prepare_npu_ffn()


@pytest.mark.parametrize(
    ("setting", "value", "dtype", "mode"),
    [
        ("IBROBOT_PI05_GRAPH_COMPILE", "0", "float16", "compiled_vision_split"),
        ("LEROBOT_PI05_ENABLE_VISION_NPU_FUSED_OPS", "0", "float16", "compiled_vision_split"),
        ("IBROBOT_PI05_GRAPH_COMPILE", "1", "float32", "compiled_vision_split"),
        ("IBROBOT_PI05_GRAPH_COMPILE", "1", "float16", "eager_vision_split"),
    ],
)
def test_ffn_option_rejects_incompatible_runtime(monkeypatch, setting, value, dtype, mode):
    monkeypatch.setenv("IBROBOT_PI05_VISION_FFN", "1")
    monkeypatch.setenv(setting, value)
    policy = PI05Ascend310PPolicy.__new__(PI05Ascend310PPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(dtype=dtype)
    policy.inference_config = PI05AscendInferenceConfig(vision_ffn=True)
    policy.model = SimpleNamespace(_prefix_graph_mode=lambda: mode)
    with pytest.raises(ValueError, match="requires FP16 with the compiled local vision tower"):
        policy.prepare_for_inference(
            deployment_fingerprint="test", torch_module=torch, torch_npu_module=None, device_name="Ascend310P1"
        )
