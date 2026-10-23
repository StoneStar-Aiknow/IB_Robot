"""Audio contract validation for voice_asr bundles, from the launch-builder side.

``validate_voice_asr_model_config`` is the voice_asr launch builder's
bundle gate, so its tests live with the builder in its owning package
(openspec change ``migrate-business-launch-builders`` moved both here from
robot_config; the file's old docstring explained the reverse placement with
the pre-migration dependency direction, which no longer applies).

The bundle packagers used to build the fixtures are this package's own.
"""

from __future__ import annotations

from pathlib import Path

from voice_asr_service.launch_builders.voice_asr import validate_voice_asr_model_config
from voice_asr_service.package_sherpa_asr_bundle import package_sherpa_asr_bundle
from voice_asr_service.package_silero_vad_bundle import package_silero_vad_bundle


def _asr_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    artifacts = {}
    for name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        path = source / name
        path.write_bytes(name.encode())
        artifacts[name.removesuffix(".txt") if name == "tokens.txt" else name.removesuffix(".onnx")] = path
    bundle = tmp_path / "asr-bundle"
    package_sherpa_asr_bundle(bundle, artifacts=artifacts, streaming=True, include_cuda=False)
    return bundle


def _silero_bundle(tmp_path: Path, *, include_ascend: bool = False) -> Path:
    onnx = tmp_path / "silero.onnx"
    onnx.write_bytes(b"fake-silero-onnx")
    om = tmp_path / "silero.om"
    om.write_bytes(b"fake-silero-om")
    om_310b = tmp_path / "silero_310b.om"
    om_310b.write_bytes(b"fake-silero-310b-om")
    bundle = tmp_path / "silero-vad"
    package_silero_vad_bundle(
        bundle,
        onnx_source=onnx,
        om_source=om if include_ascend else None,
        om_310b_source=om_310b if include_ascend else None,
        include_ascend=include_ascend,
    )
    return bundle


def test_asr_audio_contract_matching_config_passes(tmp_path):
    bundle = _asr_bundle(tmp_path)
    errors = validate_voice_asr_model_config(str(bundle), "torch_cpu", sample_rate=16000, chunk_size=512)
    assert errors == []


def test_asr_audio_contract_sample_rate_mismatch_is_rejected(tmp_path):
    bundle = _asr_bundle(tmp_path)
    errors = validate_voice_asr_model_config(str(bundle), "torch_cpu", sample_rate=8000)
    assert any("sample_rate_hz=16000" in error and "sample_rate=8000" in error for error in errors)


def test_vad_deployment_contract_mismatch_is_rejected(tmp_path):
    asr_bundle = _asr_bundle(tmp_path)
    vad_bundle = _silero_bundle(tmp_path)
    errors = validate_voice_asr_model_config(
        str(asr_bundle),
        "torch_cpu",
        sample_rate=16000,
        chunk_size=1024,
        vad_bundle_path=str(vad_bundle),
        vad_deployment="torch_cpu",
    )
    assert any("frame_size=512" in error and "chunk_size=1024" in error for error in errors)

    errors = validate_voice_asr_model_config(
        str(asr_bundle),
        "torch_cpu",
        sample_rate=16000,
        chunk_size=512,
        vad_bundle_path=str(vad_bundle),
        vad_deployment="torch_cpu",
    )
    assert errors == []


def test_vad_bundle_load_failure_is_reported(tmp_path):
    asr_bundle = _asr_bundle(tmp_path)
    errors = validate_voice_asr_model_config(
        str(asr_bundle),
        "torch_cpu",
        vad_bundle_path=str(tmp_path / "absent-vad"),
        vad_deployment="torch_cpu",
    )
    assert any("VAD bundle/deployment is invalid" in error for error in errors)
