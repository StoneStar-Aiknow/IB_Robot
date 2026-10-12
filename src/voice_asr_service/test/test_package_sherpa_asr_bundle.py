"""Contract tests for schema-v3 sherpa ASR bundles."""

from pathlib import Path

import pytest

from inference_manifest import load_inference_manifest
from inference_manifest.errors import ManifestValidationError
from voice_asr_service.package_sherpa_asr_bundle import package_sherpa_asr_bundle


def _artifact_dir(tmp_path: Path, names: tuple[str, ...]) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir()
    result = {}
    for name in names:
        path = source / f"{name}.onnx" if name != "tokens" else source / "tokens.txt"
        path.write_bytes(name.encode())
        result[name] = path
    return result


def test_package_streaming_transducer_roles(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "encoder", "decoder", "joiner"))
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=True)

    validated = load_inference_manifest(bundle, "torch_cuda")
    assert validated.identity.model_type == "sherpa_onnx"
    assert set(validated.resolved_artifacts) == {"tokens", "encoder", "decoder", "joiner"}


def test_package_streaming_paraformer_roles(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "encoder", "decoder"))
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=True, streaming_kind="paraformer")

    validated = load_inference_manifest(bundle, "torch_cpu")
    assert set(validated.resolved_artifacts) == {"tokens", "encoder", "decoder"}


def test_package_offline_roles_are_explicit(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "model"))
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=False, include_cuda=False)

    validated = load_inference_manifest(bundle, "torch_cpu")
    assert set(validated.resolved_artifacts) == {"tokens", "model"}
    with pytest.raises(ManifestValidationError):
        load_inference_manifest(bundle, "torch_cuda")


def _static_encoder(path: Path) -> None:
    """Tiny streaming-encoder stand-in: x + one cached state in, encoder_out + new state out."""
    from onnx import TensorProto, helper, save

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["x"], ["encoder_out"], axes=[1], keepdims=1),
            helper.make_node("Identity", ["cached_key_0"], ["new_cached_key_0"]),
        ],
        "encoder",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 61, 80]),
            helper.make_tensor_value_info("cached_key_0", TensorProto.FLOAT, [2, 1, 4]),
        ],
        [
            helper.make_tensor_value_info("encoder_out", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("new_cached_key_0", TensorProto.FLOAT, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    helper.set_model_props(model, {"decode_chunk_len": "48"})
    save(model, str(path))


def test_package_ascend_deployment_runs_encoder_om_on_npu_and_search_on_cpu(tmp_path: Path) -> None:
    artifacts = _artifact_dir(tmp_path, ("tokens", "encoder", "decoder", "joiner"))
    om = tmp_path / "source" / "encoder.om"
    om.write_bytes(b"om")
    static = tmp_path / "source" / "encoder.static.onnx"
    _static_encoder(static)
    bundle = tmp_path / "bundle"
    package_sherpa_asr_bundle(
        bundle, artifacts=artifacts, streaming=True, ascend_encoder_om=om, ascend_encoder_onnx=static
    )

    deployment = load_inference_manifest(bundle, "ascend_310p").deployment
    profiles = deployment.role_runtime_profiles
    assert (profiles["encoder"].backend, profiles["decoder"].backend, profiles["joiner"].backend) == (
        "ascend",
        "onnx",
        "onnx",
    )
    assert deployment.artifacts["encoder"].format == "om"
    assert deployment.metadata["decode_chunk_len"] == 48
    encoder = deployment.bindings["encoder"]
    assert [b.semantic for b in encoder.inputs] == [
        "host.asr.encoder.features",
        "host.asr.encoder.state_in.cached_key_0",
    ]
    assert [b.semantic for b in encoder.outputs] == ["host.asr.encoder.out", "host.asr.encoder.state_out.cached_key_0"]
    assert tuple(encoder.outputs[0].shape) == (1, 1, 80)
    # The recurrent state lives in the encoder OM, not in the sherpa decoder.
    (state_link,) = deployment.execution_contract.state_links
    assert (state_link.role, state_link.state_bank) == ("encoder", "asr.encoder.bank")
    assert deployment.execution_contract.state_bank_mode == "runtime_exclusive"
    # CPU deployments keep using the int8 sherpa artifacts unchanged.
    cpu = load_inference_manifest(bundle, "torch_cpu").deployment
    assert cpu.artifacts["encoder"].format == "onnx"
    assert cpu.execution_contract.state_links[0].role == "decoder"
