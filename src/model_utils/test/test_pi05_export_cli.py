from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace

import pytest

from inference_manifest import load_policy_metadata
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
    args = SimpleNamespace(dtype="fp16", donor_device="cpu", log_level="INFO")
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


def test_legacy_fast_gelu_resolves_to_all_scope(tmp_path):
    resolved = _cli.resolve(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "--policy-path",
            str(tmp_path / "bundle"),
            "--steps",
            "vlm_onnx",
            "--fast-gelu",
        ]
    )

    assert resolved.args.fast_gelu is True
    assert resolved.args.fast_gelu_scope == "all"


def test_legacy_fast_gelu_rejects_narrow_scope(tmp_path):
    with pytest.raises(SystemExit, match="alias for --fast-gelu-scope all"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--steps",
                "vlm_onnx",
                "--fast-gelu",
                "--fast-gelu-scope",
                "vision",
            ]
        )


def test_cli_scope_overrides_legacy_fast_gelu_from_defaults(tmp_path):
    config_path = tmp_path / "pi05-export.yaml"
    config_path.write_text("defaults:\n  fast_gelu: true\n", encoding="utf-8")

    resolved = _cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy-path",
            str(tmp_path / "bundle"),
            "--steps",
            "vlm_onnx",
            "--fast-gelu-scope",
            "vision",
        ]
    )

    assert resolved.args.fast_gelu is False
    assert resolved.args.fast_gelu_scope == "vision"


def test_cli_legacy_fast_gelu_overrides_scope_from_defaults(tmp_path):
    config_path = tmp_path / "pi05-export.yaml"
    config_path.write_text("defaults:\n  fast_gelu_scope: vision\n", encoding="utf-8")

    resolved = _cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy-path",
            str(tmp_path / "bundle"),
            "--steps",
            "vlm_onnx",
            "--fast-gelu",
        ]
    )

    assert resolved.args.fast_gelu is True
    assert resolved.args.fast_gelu_scope == "all"


def test_cli_no_fast_gelu_overrides_scope_from_last_run(tmp_path):
    config_path = tmp_path / "pi05-export.yaml"
    config_path.write_text("_last:\n  fast_gelu_scope: vision\n", encoding="utf-8")

    resolved = _cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy-path",
            str(tmp_path / "bundle"),
            "--steps",
            "vlm_onnx",
            "--no-fast-gelu",
        ]
    )

    assert resolved.args.fast_gelu is False
    assert resolved.args.fast_gelu_scope == "none"


def test_cli_no_fast_gelu_rejects_explicit_scope(tmp_path):
    with pytest.raises(SystemExit, match="cannot be combined"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--steps",
                "vlm_onnx",
                "--no-fast-gelu",
                "--fast-gelu-scope",
                "vision",
            ]
        )


def test_removed_npu_geglu_option_is_rejected():
    with pytest.raises(SystemExit):
        _cli.build_parser().parse_args(["--no-npu-geglu"])


@pytest.mark.parametrize(
    "option",
    [
        "--quant-deployment",
        "--calib-dir",
        "--exp-dir",
        "--output-dir",
        "--runtime-save-dir",
        "--om-dir",
    ],
)
def test_removed_top_level_options_are_rejected(option):
    with pytest.raises(SystemExit):
        _cli.build_parser().parse_args([option, "value"])


def test_profiled_publication_requires_fresh_fp_role_export(tmp_path):
    with pytest.raises(SystemExit, match="publication requires steps ae_onnx,vlm_onnx"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--device",
                "npu",
                "--soc-version",
                "Ascend310P3",
                "--quant-profile",
                "pi05-vlm-text-ae-attn-mlp-sq-v1",
                "--steps",
                "vlm_om,ae_quant_om",
            ]
        )


def test_profiled_fp_publication_requires_matching_onnx_export(tmp_path):
    with pytest.raises(SystemExit, match="requires vlm_onnx with vlm_om"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--device",
                "npu",
                "--soc-version",
                "Ascend310P3",
                "--quant-profile",
                "pi05-vlm-text-ae-attn-mlp-sq-v1",
                "--steps",
                "vlm_om,ae_om",
            ]
        )


def test_profiled_ae_quant_requires_fresh_onnx_and_fp_om_steps(tmp_path):
    with pytest.raises(SystemExit, match="automatic calibration requires steps ae_om,vlm_om,vlm_onnx"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--device",
                "npu",
                "--soc-version",
                "Ascend310P3",
                "--batch-path",
                str(tmp_path / "observations.safetensors"),
                "--quant-profile",
                "pi05-vlm-text-ae-attn-mlp-sq-v1",
                "--steps",
                "ae_onnx,ae_quant",
            ]
        )


def test_unprofiled_ae_quant_requires_same_run_calibration_artifacts(tmp_path):
    with pytest.raises(SystemExit, match="automatic calibration requires steps ae_om,vlm_om,vlm_onnx"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--soc-version",
                "Ascend310P3",
                "--batch-path",
                str(tmp_path / "observations.safetensors"),
                "--steps",
                "ae_onnx,ae_quant",
            ]
        )


def test_unprofiled_vlm_quant_requires_same_run_onnx(tmp_path):
    with pytest.raises(SystemExit, match="vlm_quant requires vlm_onnx"):
        _cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy-path",
                str(tmp_path / "bundle"),
                "--batch-path",
                str(tmp_path / "observations.safetensors"),
                "--steps",
                "vlm_quant",
            ]
        )


@pytest.mark.parametrize(
    ("scope", "vlm_scope", "ae_scope"),
    [
        ("none", "none", "none"),
        ("all", "all", "all"),
        ("vision", "vision", "none"),
        ("vlm-text", "text", "none"),
        ("ae", "none", "all"),
    ],
)
def test_export_runners_forward_scoped_fast_gelu(tmp_path, monkeypatch, scope, vlm_scope, ae_scope):
    args = SimpleNamespace(
        dtype="fp16",
        device="npu",
        fast_gelu=False,
        fast_gelu_scope=scope,
        log_level="INFO",
    )
    ctx = SimpleNamespace(
        args=args,
        policy_path=tmp_path / "bundle",
        output_dir=tmp_path / "onnx",
        runtime_save_dir=tmp_path / "runtime",
    )
    calls = []
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_vlm_onnx(ctx)
    pipeline._run_ae_onnx(ctx)

    vlm_args = calls[0][1]
    ae_args = calls[1][1]
    assert vlm_args[vlm_args.index("--fast-gelu-scope") + 1] == vlm_scope
    assert ae_args[ae_args.index("--fast-gelu-scope") + 1] == ae_scope
    assert "--no-npu-geglu" not in vlm_args


def test_q40_profile_forwards_strategy_without_fusing_donor(tmp_path, monkeypatch):
    profile = bundled_quantization_profiles()["pi05-q40-v1"]
    args = SimpleNamespace(
        dtype="fp16",
        donor_device="cpu",
        device="npu",
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
    monkeypatch.setattr(pipeline, "_capture_ae_calibration", lambda _ctx: None)
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
            "fast_gelu_scope": "none",
            "vlm": {"enabled": False},
            "action_expert": {
                "enabled": True,
                "selectors": [{"name": "fused-mlp", "regex": r"/mlp/MatMul$", "expected": 18}],
                "expected_selected_nodes": 18,
                "expected_quantized_nodes": 18,
                "fused_geglu_donor": True,
                "expected_npu_geglu_nodes": 18,
                "expected_calibration_steps": 10,
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
        calibration_dir=tmp_path / "calibration" / "ae",
        ae_w8a8=tmp_path / "ae-w8a8.onnx",
        quantization_profile=profile,
    )
    calls = []
    monkeypatch.setattr(
        pipeline,
        "_quant_inputs",
        lambda *_args, **_kwargs: (tmp_path / "donor.onnx", tmp_path / "npu.onnx"),
    )
    monkeypatch.setattr(pipeline, "_capture_ae_calibration", lambda _ctx: None)
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
    profile = bundled_quantization_profiles()["pi05-vlm-text-ae-attn-mlp-sq-v1"]
    ctx = SimpleNamespace(quantization_profile=profile)

    argv = pipeline._quant_profile_args(ctx, role="ae", output_onnx=tmp_path / "ae-w8a8.onnx")

    assert argv[argv.index("--smoothquant-alpha") + 1] == "0.5"
    assert argv[argv.index("--smoothquant-epsilon") + 1] == "1e-05"
    assert argv[argv.index("--expected-selected-nodes") + 1] == "108"
    assert argv[argv.index("--expected-quantized-nodes") + 1] == "108"
    assert argv[argv.index("--expected-calibration-steps") + 1] == "10"


def test_profile_forwards_explicit_smoothquant_verify_tolerances(tmp_path):
    profile = parse_quantization_profile(
        "ae-tolerance",
        {
            "format": "pi05-quant-profile-v1",
            "vlm": {"enabled": False},
            "action_expert": {
                "enabled": True,
                "selectors": [{"name": "one", "regex": "one", "expected": 1}],
                "expected_selected_nodes": 1,
                "expected_quantized_nodes": 1,
                "smoothquant_alpha": 0.5,
                "smoothquant_epsilon": 1e-5,
                "smoothquant_verify_rtol": 0.005,
                "smoothquant_verify_atol": 0.005,
            },
        },
    )
    argv = pipeline._quant_profile_args(
        SimpleNamespace(quantization_profile=profile),
        role="ae",
        output_onnx=tmp_path / "ae-w8a8.onnx",
    )

    assert argv[argv.index("--smoothquant-verify-rtol") + 1] == "0.005"
    assert argv[argv.index("--smoothquant-verify-atol") + 1] == "0.005"


def test_unprofiled_ae_quant_forwards_configured_trajectory_steps(tmp_path, monkeypatch):
    policy = tmp_path / "bundle"
    policy.mkdir()
    (policy / "config.json").write_text('{"num_inference_steps": 7}', encoding="utf-8")
    args = SimpleNamespace(
        num_calib=2,
        amp_num=0,
        amp_rank_samples=1,
        amp_scratch_dir=None,
        log_level="INFO",
    )
    ctx = SimpleNamespace(
        args=args,
        policy_path=policy,
        calibration_dir=tmp_path / "calibration" / "ae",
        ae_w8a8=tmp_path / "ae-w8a8.onnx",
        quantization_profile=None,
    )
    calls = []
    monkeypatch.setattr(pipeline, "_capture_ae_calibration", lambda _ctx: None)
    monkeypatch.setattr(pipeline, "_quant_inputs", lambda *_args, **_kwargs: (tmp_path / "donor.onnx", None))
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._run_ae_quant(ctx)

    argv = calls[0][1]
    assert argv[argv.index("--expected-calibration-steps") + 1] == "7"


def test_ae_calibration_is_generated_from_fresh_fp_oms(tmp_path, monkeypatch):
    profile = bundled_quantization_profiles()["pi05-vlm-text-ae-attn-mlp-sq-v1"]
    calibration_dir = tmp_path / "calibration" / "ae"
    ctx = SimpleNamespace(
        args=SimpleNamespace(
            soc_version="Ascend310P3",
            batch_path=str(tmp_path / "observations.safetensors"),
            num_calib=2,
            task="pick",
        ),
        policy_path=tmp_path / "bundle",
        vlm_om=tmp_path / "om" / "vlm.om",
        ae_om=tmp_path / "om" / "ae.om",
        calibration_dir=calibration_dir,
        quantization_profile=profile,
    )
    calls = []
    deployments = []

    monkeypatch.setattr(pipeline, "load_observation_batch", lambda _path: [object(), object(), object()])
    monkeypatch.setattr(pipeline, "_link_policy_metadata", lambda _source, destination: destination.mkdir())
    monkeypatch.setattr(
        pipeline,
        "write_pi05_ascend_deployment",
        lambda *args, **kwargs: deployments.append((args, kwargs)),
    )
    monkeypatch.setattr(pipeline, "_run_module", lambda module, argv: calls.append((module, argv)))

    pipeline._capture_ae_calibration(ctx)

    assert len(deployments) == 1
    deployment_args, deployment_kwargs = deployments[0]
    assert deployment_args[1] == "pi05-ae-calibration"
    assert deployment_args[4] == ctx.vlm_om
    assert deployment_args[6] == ctx.ae_om
    assert deployment_kwargs == {"prefer_hardlink": True}
    module, argv = calls[0]
    assert module == "model_utils.pi05_om_dump"
    assert argv[argv.index("--batch-count") + 1] == "2"
    assert argv[argv.index("--out-dir") + 1] == str(calibration_dir)
    assert argv[argv.index("--task") + 1] == "pick"
    assert argv[argv.index("--seed") + 1] == "42"
    assert not deployment_args[0].exists()


def test_temporary_calibration_bundle_rebases_absolute_local_references(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    tokenizer = source / "tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                "chunk_size": 50,
                "max_action_dim": 32,
            }
        ),
        encoding="utf-8",
    )
    (source / "policy_preprocessor.json").write_text(
        json.dumps({"tokenizer_name": str(tokenizer)}),
        encoding="utf-8",
    )
    (source / "policy_postprocessor.json").write_text("{}", encoding="utf-8")

    pipeline._link_policy_metadata(source, destination)

    copied = json.loads((destination / "policy_preprocessor.json").read_text(encoding="utf-8"))
    original = json.loads((source / "policy_preprocessor.json").read_text(encoding="utf-8"))
    assert copied["tokenizer_name"] == "tokenizer"
    assert original["tokenizer_name"] == str(tokenizer)
    assert load_policy_metadata(destination).external_dependencies == ()


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
