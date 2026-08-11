from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

from model_utils.pi05_export import _cli

pipeline = import_module("model_utils.pi05_export.__main__")


def test_verify_step_requires_only_batch_and_runs_before_om():
    verify = _cli.STEPS_BY_NAME["verify"]

    assert verify.param_deps == ("batch_path",)
    assert _cli.PARAMS_BY_DEST["task"].default == ""
    assert _cli.STEP_NAMES.index("verify") < _cli.STEP_NAMES.index("vlm_om")


def test_verify_runner_forwards_canonical_batch(tmp_path, monkeypatch):
    batch_path = tmp_path / "observations.safetensors"
    args = SimpleNamespace(batch_path=str(batch_path), task="pick", device="cpu", schedule_file=None, log_level="INFO")
    ctx = SimpleNamespace(
        args=args,
        policy_path=tmp_path / "bundle",
        vlm_onnx=tmp_path / "vlm.onnx",
        ae_onnx=tmp_path / "ae.onnx",
    )
    calls = []
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_verify(ctx)

    module, argv = calls[0]
    assert module == "model_utils.pi05_export.verify_pi05_split_equivalence"
    assert argv[argv.index("--batch-path") + 1] == str(batch_path)
    assert argv[argv.index("--task") + 1] == "pick"
    assert "--verify-max-abs-max" not in argv


def test_verify_runner_allows_batch_task(tmp_path, monkeypatch):
    args = SimpleNamespace(
        batch_path=str(tmp_path / "observations.safetensors"),
        task="",
        device="cpu",
        schedule_file=None,
        log_level="INFO",
    )
    ctx = SimpleNamespace(
        args=args,
        policy_path=tmp_path / "bundle",
        vlm_onnx=tmp_path / "vlm.onnx",
        ae_onnx=tmp_path / "ae.onnx",
    )
    calls = []
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_verify(ctx)

    _, argv = calls[0]
    assert argv[argv.index("--task") + 1] == ""
