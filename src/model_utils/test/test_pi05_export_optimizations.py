import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import TypeVar

import pytest
import torch

from model_utils.pi05_export.ascend_export_patches import (
    _build_patch_registry,
    _exact_fused_geglu,
    _fused_up_gate_weight,
    _patch_gemma_fast_gelu_npu,
    _patch_gemma_geglu_donor,
    _patch_gemma_geglu_npu,
    _patch_pytorch_gelu_tanh_npu,
    _patch_siglip_fast_gelu_npu,
    ascend_onnx_export_patches,
)


def _import_vlm_class():
    class _PreTrainedConfig:
        pass

    class _PI05Config:
        pass

    class _PreTrainedPolicy(torch.nn.Module):
        pass

    def _module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        module.__path__ = []
        return module

    stub_modules = {
        "lerobot": _module("lerobot"),
        "lerobot.configs": _module("lerobot.configs"),
        "lerobot.configs.policies": _module("lerobot.configs.policies", PreTrainedConfig=_PreTrainedConfig),
        "lerobot.policies": _module("lerobot.policies"),
        "lerobot.policies.pi05": _module("lerobot.policies.pi05"),
        "lerobot.policies.pi05.configuration_pi05": _module(
            "lerobot.policies.pi05.configuration_pi05", PI05Config=_PI05Config
        ),
        "lerobot.policies.pretrained": _module(
            "lerobot.policies.pretrained", PreTrainedPolicy=_PreTrainedPolicy, T=TypeVar("T")
        ),
        "lerobot.utils": _module("lerobot.utils"),
        "lerobot.utils.constants": _module(
            "lerobot.utils.constants",
            OBS_LANGUAGE_ATTENTION_MASK="observation.language.attention_mask",
            OBS_LANGUAGE_TOKENS="observation.language.tokens",
            OPENPI_ATTENTION_MASK_VALUE=-2.3819763e38,
        ),
        "lerobot.utils.import_utils": _module("lerobot.utils.import_utils", _transformers_available=False),
        "model_utils.pi05_export.pi_gemma": _module(
            "model_utils.pi05_export.pi_gemma",
            PaliGemmaForConditionalGenerationWithPiGemma=object,
            PiGemmaForCausalLM=object,
        ),
    }
    original_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        from model_utils.pi05_export.modeling_pi05_vlm import PI05VLMPytorch

        return PI05VLMPytorch
    finally:
        sys.modules.pop("model_utils.pi05_export.modeling_pi05_vlm", None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


PI05VLMPytorch = _import_vlm_class()


class _FakePaliGemma:
    def __init__(self):
        self.image_batch_shapes = []

    def embed_image(self, images):
        self.image_batch_shapes.append(tuple(images.shape))
        values = images[:, 0, 0, 0].to(torch.float16)
        return torch.stack((values, values + 0.25), dim=1).unsqueeze(-1).expand(-1, -1, 3)

    def embed_language_tokens(self, tokens):
        return tokens.to(torch.float16).unsqueeze(-1).expand(-1, -1, 3)


def _model_stub():
    paligemma = _FakePaliGemma()
    model = SimpleNamespace(
        paligemma_with_expert=paligemma,
        _apply_checkpoint=lambda func, *args, **kwargs: func(*args, **kwargs),
    )
    return model, paligemma


def _embed_prefix(model, images, img_masks):
    tokens = torch.tensor([[1, 2], [3, 4]])
    masks = torch.tensor([[True, False], [True, True]])
    return PI05VLMPytorch.embed_prefix(model, images, img_masks, tokens, masks)


def test_embed_prefix_batches_cameras_and_restores_camera_major_order():
    model, paligemma = _model_stub()
    images = [
        torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1),
        torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1),
    ]
    img_masks = [torch.tensor([True, False]), torch.tensor([False, True])]

    embs, pad_masks, att_masks = _embed_prefix(model, images, img_masks)

    assert paligemma.image_batch_shapes == [(4, 1, 1, 1)]
    torch.testing.assert_close(
        embs[:, :4, 0],
        torch.tensor([[1.0, 1.25, 10.0, 10.25], [2.0, 2.25, 20.0, 20.25]], dtype=torch.float16),
    )
    assert pad_masks.tolist() == [
        [True, True, False, False, True, False],
        [False, False, True, True, True, True],
    ]
    assert not att_masks.any()


def test_embed_prefix_rejects_empty_or_unpaired_images():
    model, _ = _model_stub()
    tokens = torch.ones((2, 2), dtype=torch.int64)
    masks = torch.ones((2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="must not be empty"):
        PI05VLMPytorch.embed_prefix(model, [], [], tokens, masks)
    with pytest.raises(ValueError, match="length mismatch"):
        _embed_prefix(model, [torch.ones(2, 1, 1, 1)], [])


@pytest.mark.parametrize(
    ("other_image", "message"),
    [
        (torch.ones(2, 1, 1, 2), "shape"),
        (torch.ones(3, 1, 1, 1), "batch size"),
        (torch.ones(2, 1, 1, 1, dtype=torch.float64), "dtype"),
        (torch.ones(2, 1, 1, 1, device="meta"), "device"),
    ],
)
def test_embed_prefix_rejects_incompatible_camera_images(other_image, message):
    model, _ = _model_stub()
    images = [torch.ones(2, 1, 1, 1), other_image]
    img_masks = [torch.ones(2, dtype=torch.bool), torch.ones(other_image.shape[0], dtype=torch.bool)]

    with pytest.raises(ValueError, match=message):
        _embed_prefix(model, images, img_masks)


@pytest.mark.parametrize(
    ("use_npu_ops", "fast_gelu", "expected", "unexpected"),
    [
        (True, False, _patch_gemma_geglu_npu, _patch_pytorch_gelu_tanh_npu),
        (True, True, _patch_pytorch_gelu_tanh_npu, _patch_gemma_geglu_npu),
    ],
)
def test_gelu_patch_registry_precedence(use_npu_ops, fast_gelu, expected, unexpected):
    patch_fns = {patch_fn for _, patch_fn in _build_patch_registry(use_npu_ops=use_npu_ops, fast_gelu=fast_gelu)}

    assert expected in patch_fns
    assert unexpected not in patch_fns


@pytest.mark.parametrize(
    ("scope", "expected", "unexpected"),
    [
        ("none", {_patch_gemma_geglu_npu}, {_patch_pytorch_gelu_tanh_npu, _patch_siglip_fast_gelu_npu}),
        ("all", {_patch_pytorch_gelu_tanh_npu}, {_patch_gemma_geglu_npu, _patch_siglip_fast_gelu_npu}),
        ("vision", {_patch_siglip_fast_gelu_npu, _patch_gemma_geglu_npu}, {_patch_pytorch_gelu_tanh_npu}),
        ("gemma", {_patch_gemma_fast_gelu_npu}, {_patch_gemma_geglu_npu, _patch_siglip_fast_gelu_npu}),
    ],
)
def test_scoped_gelu_patch_registry(scope, expected, unexpected):
    patch_fns = {patch_fn for _, patch_fn in _build_patch_registry(use_npu_ops=True, fast_gelu_scope=scope)}

    assert expected <= patch_fns
    assert not (unexpected & patch_fns)


def test_scoped_gelu_patches_only_target_requested_model_family(monkeypatch):
    class SiglipMLP:
        def forward(self, hidden_states):
            return hidden_states

    class GemmaMLP:
        def forward(self, hidden_states):
            return hidden_states

    modules = {
        "transformers.models.siglip.modeling_siglip": SimpleNamespace(SiglipMLP=SiglipMLP),
        "transformers.models.gemma.modeling_gemma": SimpleNamespace(GemmaMLP=GemmaMLP),
    }
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    def fake_import_module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    original_siglip = SiglipMLP.forward
    original_gemma = GemmaMLP.forward

    vision_undo = _patch_siglip_fast_gelu_npu()
    assert SiglipMLP.forward is not original_siglip
    assert GemmaMLP.forward is original_gemma
    for cls, attr, original in reversed(vision_undo):
        setattr(cls, attr, original)

    gemma_undo = _patch_gemma_fast_gelu_npu()
    assert SiglipMLP.forward is original_siglip
    assert GemmaMLP.forward is not original_gemma
    for cls, attr, original in reversed(gemma_undo):
        setattr(cls, attr, original)


def test_gelu_patch_registry_has_no_npu_gelu_off_npu():
    patch_fns = {patch_fn for _, patch_fn in _build_patch_registry(use_npu_ops=False)}

    assert _patch_gemma_geglu_npu not in patch_fns
    assert _patch_pytorch_gelu_tanh_npu not in patch_fns
    assert _patch_siglip_fast_gelu_npu not in patch_fns
    assert _patch_gemma_fast_gelu_npu not in patch_fns


def test_fused_geglu_donor_registry_is_ort_only():
    patch_fns = {patch_fn for _, patch_fn in _build_patch_registry(use_npu_ops=False, fused_geglu_donor=True)}

    assert _patch_gemma_geglu_donor in patch_fns
    assert _patch_gemma_geglu_npu not in patch_fns
    assert _patch_pytorch_gelu_tanh_npu not in patch_fns
    assert _patch_siglip_fast_gelu_npu not in patch_fns
    assert _patch_gemma_fast_gelu_npu not in patch_fns


@pytest.mark.parametrize("scope", ["all", "vision", "gemma"])
def test_fused_geglu_donor_rejects_fast_gelu_scope(scope):
    with pytest.raises(ValueError, match="FastGELU disabled"):
        _build_patch_registry(use_npu_ops=False, fast_gelu_scope=scope, fused_geglu_donor=True)


def test_fused_geglu_donor_rejects_npu_export():
    with pytest.raises(ValueError, match="non-NPU"):
        _build_patch_registry(use_npu_ops=True, fused_geglu_donor=True)


def test_required_npu_activation_patch_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    with (
        pytest.raises(RuntimeError, match="required NPU activation patch"),
        ascend_onnx_export_patches(use_npu_ops=True),
    ):
        pass


def test_exact_fused_geglu_uses_up_then_gate_order():
    up = torch.tensor([[1.0, 2.0]])
    gate = torch.tensor([[0.5, -1.0]])

    actual = _exact_fused_geglu(torch.cat([up, gate], dim=-1))
    expected = up * torch.nn.functional.gelu(gate, approximate="tanh")

    torch.testing.assert_close(actual, expected)


def test_fused_up_gate_weight_concatenates_distinct_projections():
    module = SimpleNamespace(
        up_proj=SimpleNamespace(weight=torch.tensor([[1.0], [2.0]])),
        gate_proj=SimpleNamespace(weight=torch.tensor([[3.0], [4.0]])),
    )

    fused = _fused_up_gate_weight(module)

    torch.testing.assert_close(fused, torch.tensor([[1.0], [2.0], [3.0], [4.0]]))


def test_fused_up_gate_weight_falls_back_for_wrapped_projection():
    module = SimpleNamespace(up_proj=SimpleNamespace(linear=object()), gate_proj=SimpleNamespace(weight=torch.ones(1)))

    assert _fused_up_gate_weight(module) is None
