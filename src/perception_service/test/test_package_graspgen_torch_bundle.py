import hashlib
import json

from inference_manifest import load_inference_manifest
from perception_service.graspgen_adapter import GraspGenAdapter
from perception_service.package_graspgen_torch_bundle import package_graspgen_torch_bundle


def _bundle(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "graspgen_robotiq_2f_140"
    checkpoints = source / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "graspgen_robotiq_2f_140.yml").write_text(
        "data:\n  num_points: 2048\n  num_grasps_per_object: 1000\n"
        "diffusion:\n  kappa: 1.75\n  num_diffusion_iters_eval: 7\n"
        "  num_grasps_per_object: 500\n",
        encoding="utf-8",
    )
    (checkpoints / "graspgen_robotiq_2f_140_gen.pth").write_bytes(b"generator")
    (checkpoints / "graspgen_robotiq_2f_140_dis.pth").write_bytes(b"discriminator")
    return source, root


def test_torch_packager_writes_strict_cuda_identity_bundle(tmp_path):
    source, root = _bundle(tmp_path)

    package_graspgen_torch_bundle(root, source_root=source)

    validated = load_inference_manifest(root, "torch_cuda")
    assert validated.manifest.model.kind == "perception"
    assert validated.manifest.model.family == "graspgen"
    assert validated.manifest.model.outputs[0].shape == (-1, 4, 4)
    assert validated.manifest.model.outputs[1].shape == (-1,)
    assert validated.deployment.device == "cuda"
    assert len(validated.fingerprint) == 64
    adapter = root / "assets" / "adapter.json"
    assert adapter.is_file()
    assert "assets/adapter.json" in {item.path for item in validated.manifest.bundle.files}
    adapter_document = json.loads(adapter.read_text(encoding="utf-8"))
    assert adapter_document["torch_module_loader"] == ("perception_service.torch_model_loaders:load_graspgen")
    assert adapter_document["source_sha256"] == {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in (
            "assets/graspgen_config.yml",
            "assets/discriminator_checkpoint.pth",
            "assets/generator_checkpoint.pth",
        )
    }
    graspgen_adapter = GraspGenAdapter.from_bundle(root)
    assert graspgen_adapter.config.kappa == 1.75
    assert graspgen_adapter.config.diffusion_steps == 7
    assert graspgen_adapter.config.grasp_batch_size == 1000
    assert graspgen_adapter.config.point_count == 2048

    package_graspgen_torch_bundle(root)
    unchanged = load_inference_manifest(root, "torch_cuda")
    assert unchanged.manifest.bundle.revision == validated.manifest.bundle.revision

    (source / "checkpoints" / "graspgen_robotiq_2f_140_gen.pth").write_bytes(b"replacement")
    package_graspgen_torch_bundle(root, source_root=source)
    changed = load_inference_manifest(root, "torch_cuda")
    assert changed.manifest.bundle.revision == validated.manifest.bundle.revision + 1


def test_torch_packager_rejects_missing_checkpoint(tmp_path):
    source, root = _bundle(tmp_path)
    (source / "checkpoints" / "graspgen_robotiq_2f_140_dis.pth").unlink()

    try:
        package_graspgen_torch_bundle(root, source_root=source)
    except FileNotFoundError as exc:
        assert "graspgen_robotiq_2f_140_dis.pth" in str(exc)
    else:
        raise AssertionError("missing discriminator checkpoint must fail closed")
