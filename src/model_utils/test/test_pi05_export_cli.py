from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

from model_utils.pi05_export import _cli
from model_utils.pi05_export.quant.profiles import bundled_quantization_profiles, parse_quantization_profile

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


def test_vlm_quant_runner_forwards_amp_ranking_options(tmp_path, monkeypatch):
    args = SimpleNamespace(
        batch_path=str(tmp_path / "observations.safetensors"),
        num_calib=16,
        amp_num=12,
        amp_rank_samples=8,
        amp_scratch_dir=str(tmp_path / "scratch"),
        device="npu",
        task="pick",
        log_level="INFO",
    )
    ctx = SimpleNamespace(args=args, policy_path=tmp_path / "bundle", vlm_w8a8=tmp_path / "vlm-w8a8.onnx")
    calls = []
    monkeypatch.setattr(pipeline, "_quant_inputs", lambda *_args, **_kwargs: (tmp_path / "donor.onnx", None))
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_vlm_quant(ctx)

    module, argv = calls[0]
    assert module == "model_utils.pi05_export.quant.quantize_vlm"
    assert argv[argv.index("--amp-rank-samples") + 1] == "8"
    assert argv[argv.index("--amp-scratch-dir") + 1] == str(tmp_path / "scratch")
    assert "--fused-geglu-donor" not in argv


def test_vlm_donor_export_keeps_geglu_unfused_by_default(tmp_path, monkeypatch):
    args = SimpleNamespace(dtype="fp16", donor_device="cpu", fast_gelu=False, npu_geglu=True, log_level="INFO")
    ctx = SimpleNamespace(
        args=args,
        policy_path=tmp_path / "bundle",
        output_dir=tmp_path / "onnx",
        runtime_save_dir=tmp_path / "runtime",
    )
    calls = []
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_vlm_donor_onnx(ctx)

    assert calls[0][0] == "model_utils.pi05_export.convert_onnx_vlm"
    assert "--fused-geglu-donor" not in calls[0][1]
    assert "--skip-runtime-save" in calls[0][1]
    assert "pi05-vlm.onnx" in calls[0][1][calls[0][1].index("--output") + 1]


def test_vlm_unfused_export_and_quant_use_exact_route(tmp_path, monkeypatch):
    args = SimpleNamespace(
        dtype="fp16",
        donor_device="cpu",
        device="npu",
        fast_gelu=False,
        npu_geglu=False,
        batch_path=str(tmp_path / "observations.safetensors"),
        num_calib=16,
        amp_num=0,
        amp_rank_samples=1,
        amp_scratch_dir=None,
        task="pick",
        log_level="INFO",
    )
    ctx = SimpleNamespace(
        args=args,
        policy_path=tmp_path / "bundle",
        output_dir=tmp_path / "onnx",
        runtime_save_dir=tmp_path / "runtime",
        vlm_w8a8=tmp_path / "vlm-w8a8.onnx",
    )
    calls = []
    monkeypatch.setattr(
        pipeline,
        "_quant_inputs",
        lambda *_args, **_kwargs: (tmp_path / "donor.onnx", tmp_path / "npu.onnx"),
    )
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_vlm_onnx(ctx)
    pipeline._run_vlm_donor_onnx(ctx)
    pipeline._run_vlm_quant(ctx)

    assert "--no-npu-geglu" in calls[0][1]
    assert "--fast-gelu" not in calls[0][1]
    assert "--fused-geglu-donor" not in calls[1][1]
    assert "pi05-vlm.onnx" in calls[1][1][calls[1][1].index("--output") + 1]
    assert "--unfused-geglu-deployment" in calls[2][1]
    assert "--fused-geglu-donor" not in calls[2][1]


def test_q40_profile_forwards_strategy_without_fusing_donor(tmp_path, monkeypatch):
    profile = bundled_quantization_profiles()["pi05-q40-v1"]
    args = SimpleNamespace(
        dtype="fp16",
        donor_device="cpu",
        device="npu",
        fast_gelu=False,
        npu_geglu=True,
        batch_path=str(tmp_path / "observations.safetensors"),
        num_calib=16,
        amp_num=0,
        amp_rank_samples=1,
        amp_scratch_dir=None,
        task="pick",
        log_level="INFO",
    )
    ctx = SimpleNamespace(
        args=args,
        policy_path=tmp_path / "bundle",
        output_dir=tmp_path / "onnx",
        runtime_save_dir=tmp_path / "runtime",
        vlm_w8a8=tmp_path / "vlm-w8a8.onnx",
        quantization_profile=profile,
    )
    calls = []
    monkeypatch.setattr(
        pipeline,
        "_quant_inputs",
        lambda *_args, **_kwargs: (tmp_path / "donor.onnx", tmp_path / "npu.onnx"),
    )
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_vlm_donor_onnx(ctx)
    pipeline._run_vlm_quant(ctx)

    donor_args = calls[0][1]
    quant_args = calls[1][1]
    assert "--fused-geglu-donor" not in donor_args
    assert "pi05-vlm.onnx" in donor_args[donor_args.index("--output") + 1]
    assert "--fused-geglu-donor" not in quant_args
    assert "--require-npu-geglu" in quant_args
    assert quant_args[quant_args.index("--expected-npu-geglu-nodes") + 1] == "17"
    assert quant_args[quant_args.index("--expected-selected-nodes") + 1] == "136"
    assert quant_args[quant_args.index("--expected-quantized-nodes") + 1] == "136"
    assert quant_args[quant_args.index("--quant-profile-name") + 1] == "pi05-q40-v1"
    assert len(profile.vlm.selectors) == 7


def test_ae_attention_profile_exports_fp32_calibration_donor(tmp_path, monkeypatch):
    profile = bundled_quantization_profiles()["pi05-ae-attn-v1"]
    ctx = SimpleNamespace(
        args=SimpleNamespace(dtype="fp16", donor_device="cpu", log_level="INFO"),
        policy_path=tmp_path / "bundle",
        output_dir=tmp_path / "onnx",
        runtime_save_dir=tmp_path / "runtime",
        quantization_profile=profile,
    )
    calls = []
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_ae_donor_onnx(ctx)

    module, argv = calls[0]
    assert module == "model_utils.pi05_export.convert_onnx_action_expert"
    assert argv[argv.index("--dtype") + 1] == "fp32"


def test_ae_fused_geglu_profile_exports_matching_donor_and_validates_npu_graph(tmp_path, monkeypatch):
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
    ctx = SimpleNamespace(
        args=SimpleNamespace(
            dtype="fp16",
            donor_device="cpu",
            num_calib=16,
            amp_num=0,
            amp_rank_samples=1,
            amp_scratch_dir=None,
            log_level="INFO",
        ),
        policy_path=tmp_path / "bundle",
        output_dir=tmp_path / "onnx",
        runtime_save_dir=tmp_path / "runtime",
        calib_dir=tmp_path / "calib",
        ae_w8a8=tmp_path / "ae-w8a8.onnx",
        quantization_profile=profile,
    )
    calls = []
    monkeypatch.setattr(
        pipeline,
        "_quant_inputs",
        lambda *_args, **_kwargs: (tmp_path / "donor.onnx", tmp_path / "npu.onnx"),
    )
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_ae_donor_onnx(ctx)
    pipeline._run_ae_quant(ctx)

    donor_args = calls[0][1]
    quant_args = calls[1][1]
    assert "--fused-geglu-donor" in donor_args
    assert "pi05-action_expert_fused-geglu.onnx" in donor_args[donor_args.index("--output") + 1]
    assert "--fused-geglu-donor" in quant_args
    assert "--require-npu-geglu" in quant_args
    assert quant_args[quant_args.index("--expected-npu-geglu-nodes") + 1] == "18"


def test_ae_attention_mlp_profile_forwards_smoothquant_parameters(tmp_path):
    profile = bundled_quantization_profiles()["pi05-ae-attn-mlp-sq-v1"]
    ctx = SimpleNamespace(quantization_profile=profile)

    argv = pipeline._quant_profile_args(ctx, role="ae", output_onnx=tmp_path / "ae-w8a8.onnx")

    assert argv[argv.index("--smoothquant-alpha") + 1] == "0.5"
    assert argv[argv.index("--smoothquant-epsilon") + 1] == "1e-05"
    assert argv[argv.index("--expected-selected-nodes") + 1] == "108"
    assert argv[argv.index("--expected-quantized-nodes") + 1] == "108"
    assert argv[argv.index("--expected-calibration-steps") + 1] == "10"


def test_quant_om_revalidates_profile_metadata_immediately_before_compile(monkeypatch):
    events = []
    ctx = SimpleNamespace(vlm_w8a8="vlm.onnx", vlm_quant_om="vlm.om")
    monkeypatch.setattr(
        pipeline,
        "_validate_quantized_profile_dependencies",
        lambda value, steps: events.append(("validate", value, steps)),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_om",
        lambda value, **kwargs: events.append(("compile", value, kwargs)),
    )

    pipeline._run_vlm_quant_om(ctx)

    assert [event[0] for event in events] == ["validate", "compile"]
