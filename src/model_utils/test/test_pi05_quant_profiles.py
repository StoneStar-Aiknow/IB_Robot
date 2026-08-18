from __future__ import annotations

from pathlib import Path

import pytest

from model_utils.pi05_export import _cli
from model_utils.pi05_export.quant import w8a8_common
from model_utils.pi05_export.quant.profiles import (
    bundled_quantization_profiles,
    metadata_path,
    parse_quantization_profile,
    validate_quantization_metadata,
    write_quantization_metadata,
)


def _q40_node_names() -> list[str]:
    names = []
    names.extend(f"/layers.{layer}/self_attn/q_proj/MatMul" for layer in range(17))
    names.extend(f"/layers.{layer}/self_attn/k_proj/MatMul" for layer in range(18))
    names.extend(f"/layers.{layer}/self_attn/v_proj/MatMul" for layer in range(18))
    names.extend(f"/layers.{layer}/self_attn/o_proj/MatMul" for layer in range(17))
    names.extend(f"/layers.{layer}/mlp/down_proj/MatMul" for layer in (0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16))
    for projection in ("q_proj", "v_proj"):
        names.extend(
            f"/vision_tower/vision_model/encoder/layers.{layer}/self_attn/{projection}/MatMul" for layer in range(27)
        )
    return names


def _ae108_node_names() -> list[str]:
    names = []
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        names.extend(f"/layers.{layer}/self_attn/{projection}/MatMul" for layer in range(18))
    names.extend(f"/layers.{layer}/mlp/MatMul" for layer in range(18))
    names.extend(f"/layers.{layer}/mlp/down_proj/MatMul" for layer in range(18))
    return names


def _write_policy(policy: Path, model: bytes = b"weights") -> None:
    policy.mkdir()
    (policy / "config.json").write_text('{"type":"pi05"}', encoding="utf-8")
    (policy / "model.safetensors").write_bytes(model)
    (policy / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (policy / "policy_postprocessor.json").write_text("{}", encoding="utf-8")


def test_q40_profile_has_exact_validated_scope():
    profile = bundled_quantization_profiles()["pi05-q40-v1"]

    assert profile.digest == "8afca52b7fd7e5774f8049c81caa98bba9f6980af949f742c498797095daae36"
    assert profile.vlm.enabled
    assert not profile.action_expert.enabled
    assert sum(selector.expected for selector in profile.vlm.selectors) == 136
    assert profile.vlm.expected_selected_nodes == 136
    assert profile.vlm.expected_quantized_nodes == 136
    assert profile.vlm.fused_geglu_donor is False
    assert profile.vlm.expected_npu_geglu_nodes == 17


def test_ae_attention_profile_requires_complete_trajectory():
    profile = bundled_quantization_profiles()["pi05-ae-attn-v1"]

    assert profile.status == "validated"
    assert not profile.vlm.enabled
    assert profile.action_expert.enabled
    assert sum(selector.expected for selector in profile.action_expert.selectors) == 72
    assert profile.action_expert.expected_selected_nodes == 72
    assert profile.action_expert.expected_quantized_nodes == 72
    assert profile.action_expert.expected_calibration_steps == 10
    assert profile.action_expert.donor_dtype == "fp32"


def test_ae_attention_mlp_smoothquant_profile_has_exact_validated_scope():
    profile = bundled_quantization_profiles()["pi05-ae-attn-mlp-sq-v1"]
    role = profile.action_expert

    assert profile.digest == "cfe53326b0bbf64ae0bf87f5ee829c8f6c1b9f984eb7435c40a593b28c390c3b"
    assert profile.status == "checkpoint-validated"
    assert profile.target_soc == "Ascend310P3"
    assert profile.npu_geglu is True
    assert profile.fast_gelu is False
    assert not profile.vlm.enabled
    assert role.enabled
    assert role.donor_dtype == "fp16"
    assert role.fused_geglu_donor is True
    assert role.expected_npu_geglu_nodes == 18
    assert role.expected_calibration_steps == 10
    assert role.smoothquant_alpha == 0.5
    assert role.smoothquant_epsilon == 1e-5
    assert sum(selector.expected for selector in role.selectors) == 108
    assert role.expected_selected_nodes == 108
    assert role.expected_quantized_nodes == 108

    names = _ae108_node_names()
    selection = w8a8_common.select_quantizable_nodes(
        [(name, "MatMul") for name in names],
        list(role.disable_regex),
        [selector.regex for selector in role.selectors],
        expected_regex_matches=[selector.expected for selector in role.selectors],
        expected_selected_nodes=108,
    )
    assert list(selection.selected_names) == names


def test_q40_profile_selects_exact_nodes_and_rejects_graph_drift():
    profile = bundled_quantization_profiles()["pi05-q40-v1"]
    names = _q40_node_names()
    quantizable = [(name, "MatMul") for name in names]
    patterns = [selector.regex for selector in profile.vlm.selectors]
    expected = [selector.expected for selector in profile.vlm.selectors]

    selection = w8a8_common.select_quantizable_nodes(
        quantizable,
        [],
        patterns,
        expected_regex_matches=expected,
        expected_selected_nodes=136,
    )

    assert list(selection.selected_names) == names
    with pytest.raises(ValueError, match="regex match mismatch"):
        w8a8_common.select_quantizable_nodes(
            quantizable[:-1],
            [],
            patterns,
            expected_regex_matches=expected,
            expected_selected_nodes=136,
        )


def test_export_profile_can_save_quant_profile_reference(tmp_path):
    policy = tmp_path / "bundle"
    policy.mkdir()
    config_path = tmp_path / "pi05-export.yaml"

    resolved = _cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy-path",
            str(policy),
            "--device",
            "npu",
            "--donor-device",
            "cpu",
            "--quant-profile",
            "pi05-q40-v1",
            "--batch-path",
            str(tmp_path / "observations.safetensors"),
            "--steps",
            "vlm_quant",
            "--save-as",
            "q40-run",
        ]
    )

    saved = _cli.load_config(str(config_path))["profiles"]["q40-run"]
    assert resolved.quantization_profile is bundled_quantization_profiles()["pi05-q40-v1"]
    assert saved["quant_profile"] == "pi05-q40-v1"
    assert "steps" not in saved


def test_config_can_define_custom_quantization_profile(tmp_path):
    policy = tmp_path / "bundle"
    policy.mkdir()
    config_path = tmp_path / "pi05-export.yaml"
    config_path.write_text(
        """
quantization_profiles:
  custom-v1:
    format: pi05-quant-profile-v1
    status: experimental
    vlm:
      enabled: true
      selectors:
        - name: one-layer
          regex: '^/layers\\.0/self_attn/q_proj/MatMul$'
          expected: 1
      disable_regex: []
      expected_selected_nodes: 1
      expected_quantized_nodes: 1
    action_expert:
      enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    resolved = _cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy-path",
            str(policy),
            "--quant-profile",
            "custom-v1",
            "--batch-path",
            str(tmp_path / "observations.safetensors"),
            "--steps",
            "vlm_quant",
        ]
    )

    assert resolved.quantization_profile is not None
    assert resolved.quantization_profile.name == "custom-v1"
    assert resolved.quantization_profile.vlm.selectors[0].expected == 1


def test_custom_profile_rejects_contradictory_geglu_modes():
    with pytest.raises(ValueError, match="cannot enable both"):
        parse_quantization_profile(
            "invalid",
            {
                "format": "pi05-quant-profile-v1",
                "npu_geglu": True,
                "fast_gelu": True,
                "vlm": {"enabled": False},
                "action_expert": {"enabled": False},
            },
        )


def test_custom_profile_requires_complete_smoothquant_parameters():
    with pytest.raises(ValueError, match="declare smoothquant_alpha and smoothquant_epsilon together"):
        parse_quantization_profile(
            "invalid-smoothquant",
            {
                "format": "pi05-quant-profile-v1",
                "vlm": {"enabled": False},
                "action_expert": {
                    "enabled": True,
                    "selectors": [{"name": "one", "regex": "MatMul", "expected": 1}],
                    "expected_selected_nodes": 1,
                    "expected_quantized_nodes": 1,
                    "smoothquant_alpha": 0.5,
                },
            },
        )


def test_custom_profile_allows_ae_exact_fused_geglu_donor():
    profile = parse_quantization_profile(
        "ae-fused",
        {
            "format": "pi05-quant-profile-v1",
            "npu_geglu": True,
            "fast_gelu": False,
            "vlm": {"enabled": False},
            "action_expert": {
                "enabled": True,
                "selectors": [{"name": "fused-mlp", "regex": r"/mlp/MatMul$", "expected": 18}],
                "expected_selected_nodes": 18,
                "expected_quantized_nodes": 18,
                "fused_geglu_donor": True,
                "expected_npu_geglu_nodes": 18,
            },
        },
    )

    assert profile.action_expert.fused_geglu_donor is True
    assert profile.action_expert.expected_npu_geglu_nodes == 18


def test_custom_ae_fused_geglu_donor_requires_exact_npu_geglu():
    with pytest.raises(ValueError, match="requires exact NPU GeGLU"):
        parse_quantization_profile(
            "ae-fused",
            {
                "format": "pi05-quant-profile-v1",
                "vlm": {"enabled": False},
                "action_expert": {
                    "enabled": True,
                    "selectors": [{"name": "fused-mlp", "regex": r"/mlp/MatMul$", "expected": 18}],
                    "expected_selected_nodes": 18,
                    "expected_quantized_nodes": 18,
                    "fused_geglu_donor": True,
                },
            },
        )


def test_effective_cli_rejects_fast_gelu_with_npu_geglu_profile(tmp_path):
    policy = tmp_path / "bundle"
    policy.mkdir()
    config_path = tmp_path / "pi05-export.yaml"
    config_path.write_text(
        """
quantization_profiles:
  npu-geglu-v1:
    format: pi05-quant-profile-v1
    npu_geglu: true
    vlm:
      enabled: true
      selectors:
        - name: one-layer
          regex: '^/layers\\.0/self_attn/q_proj/MatMul$'
          expected: 1
      expected_selected_nodes: 1
      expected_quantized_nodes: 1
    action_expert:
      enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="requires fast_gelu=False"):
        _cli.resolve(
            [
                "--config",
                str(config_path),
                "--policy-path",
                str(policy),
                "--quant-profile",
                "npu-geglu-v1",
                "--fast-gelu",
                "--batch-path",
                str(tmp_path / "observations.safetensors"),
                "--steps",
                "vlm_quant",
            ]
        )


def test_q40_profile_rejects_action_expert_quantization(tmp_path):
    policy = tmp_path / "bundle"
    policy.mkdir()

    with pytest.raises(SystemExit, match="disables ae quantization"):
        _cli.resolve(
            [
                "--policy-path",
                str(policy),
                "--device",
                "npu",
                "--quant-profile",
                "pi05-q40-v1",
                "--steps",
                "ae_quant",
            ]
        )


def test_q40_profile_rejects_non_npu_export(tmp_path):
    policy = tmp_path / "bundle"
    policy.mkdir()

    with pytest.raises(SystemExit, match="requires device=npu"):
        _cli.resolve(
            [
                "--policy-path",
                str(policy),
                "--device",
                "cpu",
                "--quant-profile",
                "pi05-q40-v1",
                "--batch-path",
                str(tmp_path / "observations.safetensors"),
                "--steps",
                "vlm_quant",
            ]
        )


def test_list_quant_profiles_includes_q40(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli.resolve(["--list-quant-profiles"])

    assert exc_info.value.code == 0
    assert "pi05-q40-v1" in capsys.readouterr().out


def test_quant_metadata_rejects_changed_policy(tmp_path):
    profile = bundled_quantization_profiles()["pi05-q40-v1"]
    policy = tmp_path / "bundle"
    _write_policy(policy)
    donor = tmp_path / "donor.onnx"
    npu = tmp_path / "npu.onnx"
    output = tmp_path / "output.onnx"
    donor.write_bytes(b"donor")
    npu.write_bytes(b"npu")
    output.write_bytes(b"quantized")
    path = metadata_path(output)

    write_quantization_metadata(
        path=path,
        profile_name=profile.name,
        profile_hash=profile.digest,
        role="vlm",
        policy_path=policy,
        donor_onnx=donor,
        npu_onnx=npu,
        output_onnx=output,
        selected_nodes=_q40_node_names(),
        actual_quantized_nodes=136,
    )
    validate_quantization_metadata(
        path=path,
        profile=profile,
        role="vlm",
        policy_path=policy,
        donor_onnx=donor,
        npu_onnx=npu,
        output_onnx=output,
    )

    (policy / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="current profile/source"):
        validate_quantization_metadata(
            path=path,
            profile=profile,
            role="vlm",
            policy_path=policy,
            donor_onnx=donor,
            npu_onnx=npu,
            output_onnx=output,
        )

    write_quantization_metadata(
        path=path,
        profile_name=profile.name,
        profile_hash=profile.digest,
        role="vlm",
        policy_path=policy,
        donor_onnx=donor,
        npu_onnx=npu,
        output_onnx=output,
        selected_nodes=_q40_node_names(),
        actual_quantized_nodes=136,
    )
    donor.write_bytes(b"changed donor")
    with pytest.raises(ValueError, match="donor_onnx"):
        validate_quantization_metadata(
            path=path,
            profile=profile,
            role="vlm",
            policy_path=policy,
            donor_onnx=donor,
            npu_onnx=npu,
            output_onnx=output,
        )
