"""Tests for the GraspGen-X export pipeline's path and shape handling.

Everything here runs on a host with neither a GPU nor an Ascend device: the
graph build and the ATC compile need Torch-NPU and the SoC, but where the
intermediates land and which shapes reach ATC are decided by plain logic.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler

from model_utils.export_paths import export_work_dir
from model_utils.graspgenx_export import export_onnx, onnx2om
from model_utils.graspgenx_export.binding import generator_binding, validate_binding
from model_utils.graspgenx_export.export_onnx import resolve_grasps
from model_utils.graspgenx_export.onnx2om import format_input_shape, resolve_output_dir


def _manifest(bundle_root=None):
    manifest = {
        "model_type": "graspgenx",
        "grasps": 400,
        "steps": 20,
        "onnx": "diffusion_loop_g400.onnx",
        "metadata": "diffusion_loop_g400.json",
        "sha256": "0" * 64,
        "input_shape": {"observation": [400, 1024], "initial_noise": [400, 12]},
    }
    if bundle_root is not None:
        manifest["bundle_root"] = str(bundle_root)
    return manifest


def test_the_work_dir_sits_beside_models_not_inside_the_bundle(tmp_path):
    """The exporter must use the shared layout, not its own _work path.

    A hand-rolled ``bundle.parent / "_work"`` puts the directory inside the
    bundle family under models/ for any nested bundle, which is not the
    documented models/_work/<bundle>/<exporter> layout.
    """
    bundle = tmp_path / "models" / "graspgenx" / "omnihand_left_x2"
    bundle.mkdir(parents=True)

    work_dir = export_work_dir(bundle, "graspgenx_export")

    assert work_dir == tmp_path / "models" / "_work" / "graspgenx" / "omnihand_left_x2" / "graspgenx_export"
    assert not any(child.name == "_work" for child in (tmp_path / "models" / "graspgenx").iterdir())


def test_an_explicit_work_dir_inside_the_bundle_is_refused(tmp_path):
    bundle = tmp_path / "models" / "policy"
    bundle.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside the policy bundle"):
        export_work_dir(bundle, "graspgenx_export", bundle / "assets" / "om")


def test_the_om_output_defaults_beside_the_manifest(tmp_path):
    work_dir = tmp_path / "models" / "_work" / "policy" / "graspgenx_export"
    work_dir.mkdir(parents=True)
    manifest_path = work_dir / "graspgenx_export.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    assert resolve_output_dir(_manifest(), manifest_path, None) == work_dir


def test_the_om_cannot_be_compiled_into_the_bundle_it_came_from(tmp_path):
    bundle = tmp_path / "models" / "policy"
    (bundle / "assets").mkdir(parents=True)
    manifest_path = tmp_path / "models" / "_work" / "policy" / "graspgenx_export.json"
    manifest_path.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside the policy bundle"):
        resolve_output_dir(_manifest(bundle), manifest_path, bundle / "assets")


def test_a_bundle_that_moved_away_does_not_block_the_compile(tmp_path):
    """The recorded bundle only bounds the destination while it is still there."""
    manifest_path = tmp_path / "work" / "graspgenx_export.json"
    manifest_path.parent.mkdir(parents=True)
    elsewhere = tmp_path / "out"

    assert resolve_output_dir(_manifest(tmp_path / "gone"), manifest_path, elsewhere) == elsewhere


def test_input_shapes_reach_atc_in_its_own_syntax():
    assert (
        format_input_shape({"observation": [400, 1024], "position_noise": [19, 400, 6]})
        == "observation:400,1024;position_noise:19,400,6"
    )


@pytest.mark.parametrize(
    "shapes",
    [
        {"observation": []},
        {"observation": [400, -1]},
        {"observation": [400, 0]},
        {"observation": ["400"]},
        {"observation": [True]},
    ],
)
def test_a_shape_atc_would_read_as_dynamic_is_refused(shapes):
    """The manifest's sha256 covers the ONNX, not the shapes written beside it.

    ATC accepts a -1 as a dynamic axis, which would produce a graph whose
    shape the loader's check can no longer speak for.
    """
    with pytest.raises(ValueError):
        format_input_shape(shapes)


@pytest.mark.parametrize("declared", [200, 400])
def test_the_bundle_decides_the_row_count(declared):
    """No default of the exporter's own, which could only disagree with a bundle.

    Defaulting to a fixed number meant that running the documented flow with
    defaults against a default-packaged bundle always failed: the packager
    writes num_grasps=200 and this exporter asked for 400.
    """
    assert resolve_grasps(declared, None) == declared


def test_an_explicit_row_count_matching_the_bundle_is_honoured():
    assert resolve_grasps(400, 400) == 400


def test_an_explicit_row_count_the_generator_would_decline_is_refused():
    with pytest.raises(SystemExit, match="declares num_grasps=200"):
        resolve_grasps(200, 400)


def test_dumped_constants_carry_generator_binding_and_use_the_inference_schedule(tmp_path, monkeypatch):
    class Head(torch.nn.Linear):
        def _step_embedding(self, timestep, device):
            return torch.zeros((1, 4), device=device)

    gen = tmp_path / "assets/gen"
    gen.mkdir(parents=True)
    checkpoint, config = gen / "generator.pth", gen / "config.yaml"
    checkpoint.write_bytes(b"weights")
    config.write_text("diffusion:\n  num_diffusion_iters: 20\n  num_diffusion_iters_eval: 10\n")
    assets = {"generator_checkpoint": "assets/gen/generator.pth", "checkpoints_dir": "assets"}
    (tmp_path / "assets/adapter.json").write_text(json.dumps(assets))
    scheduler = DDPMScheduler(num_train_timesteps=20)
    generator = SimpleNamespace(
        diffusion_head=Head(1, 1),
        num_diffusion_iters_eval=10,
        noise_scheduler_pos=scheduler,
        noise_scheduler_rot=scheduler,
        _install_inference_head=lambda: None,
        _prepare_likelihood_scales=lambda betas, device: [(-2 * b, 0.0) for b in betas],
    )
    monkeypatch.setattr(
        export_onnx, "_assembly", lambda: lambda *args, **kwargs: SimpleNamespace(grasp_generator=generator)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _: "test"), raising=False)
    constants = tmp_path / "constants.npz"
    export_onnx.dump_constants(tmp_path, constants, 4)
    stored = np.load(constants)
    binding = generator_binding(checkpoint, config)
    validate_binding(
        {
            "generator_binding": json.loads(str(stored["generator_binding"])),
            "grasps": int(stored["grasps"]),
            "steps": int(stored["steps"]),
        },
        binding,
        4,
    )
    # With zero model output and zero variance noise this isolates both mean
    # coefficients, compared with diffusers rather than the shared helper.
    sample = torch.tensor([[0.2, -0.3, 0.4]])
    from unittest.mock import patch

    for index, timestep in enumerate(stored["timesteps"]):
        beta, alpha, original_coeff, sample_coeff, _ = stored["coefficients_pos"][index]
        original = (sample / alpha).clamp(-1, 1)
        actual = original_coeff * original + sample_coeff * sample
        with patch("diffusers.schedulers.scheduling_ddpm.randn_tensor", return_value=torch.zeros_like(sample)):
            expected = scheduler.step(torch.zeros_like(sample), int(timestep), sample)
        torch.testing.assert_close(actual, expected.prev_sample)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="binding"):
        validate_binding(
            {"generator_binding": binding, "grasps": 4, "steps": 10}, generator_binding(checkpoint, config), 4
        )


@pytest.mark.parametrize("change", ["none", "metadata", "missing_binding", "steps"])
def test_converter_checks_metadata_provenance_before_invoking_atc(tmp_path, monkeypatch, change):
    source = tmp_path / "diffusion_loop_g400.onnx"
    source.write_bytes(b"exported graph")
    binding = {"version": 1, "steps": 20, "generator_checkpoint_sha256": "a" * 64, "generator_config_sha256": "b" * 64}
    metadata = tmp_path / "diffusion_loop_g400.json"
    metadata.write_text(json.dumps({"generator_binding": binding, "grasps": 400, "steps": 20}))
    manifest = {
        **_manifest(),
        "sha256": onnx2om._sha256(source),
        "metadata_sha256": onnx2om._sha256(metadata),
        "generator_binding": binding,
    }
    if change == "metadata":
        metadata.write_text(metadata.read_text() + "\n")
    elif change == "missing_binding":
        manifest.pop("generator_binding")
    elif change == "steps":
        metadata.write_text(json.dumps({"generator_binding": binding, "grasps": 400, "steps": 10}))
        manifest["metadata_sha256"] = onnx2om._sha256(metadata)
    path = tmp_path / "export.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["onnx2om", "--manifest", str(path), "--dry-run", "--atc", "/fake/atc"])
    if change == "none":
        assert onnx2om.main() == 0
    else:
        with pytest.raises((SystemExit, ValueError), match="metadata|binding|must cover"):
            onnx2om.main()
