# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""A GraspGenX bundle binds the official release checkpoints to one gripper description."""

from __future__ import annotations

import json

import pytest

from inference_manifest import load_inference_manifest
from perception_service.graspgenx_adapter import GRASPGENX_SAMPLE_COUNT_SEMANTIC, GraspGenXAdapter
from perception_service.package_graspgenx_torch_bundle import package_graspgenx_torch_bundle


def _sources(tmp_path):
    release = tmp_path / "release"
    for network, epochs in (("gen", (90, 736)), ("dis", (1056,))):
        (release / network).mkdir(parents=True)
        (release / network / "config.yaml").write_text(f"network: {network}\n", encoding="utf-8")
        for epoch in epochs:
            (release / network / f"epoch_{epoch}.pth").write_bytes(f"{network}-{epoch}".encode())
    gripper = tmp_path / "omnihand_left_x2"
    (gripper / "meshes").mkdir(parents=True)
    (gripper / "config.json").write_text("{}", encoding="utf-8")
    (gripper / "gripper.urdf").write_text("<robot/>", encoding="utf-8")
    (gripper / "meshes" / "palm.obj").write_text("v 0 0 0\n", encoding="utf-8")
    return release, gripper


def test_the_packager_writes_a_cuda_bundle_with_the_newest_checkpoints_and_the_gripper(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "graspgenx_omnihand_left_x2"

    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)

    validated = load_inference_manifest(root, "torch_cuda")
    model = validated.manifest.model
    assert (model.interface, model.model_type, model.operation) == ("tensor_model", "graspgenx", "generate_grasps")
    assert [tensor.semantic for tensor in model.inputs][1] == GRASPGENX_SAMPLE_COUNT_SEMANTIC
    assert validated.deployment.device == "cuda"
    assert (root / "assets/graspgenx/gen/generator.pth").read_bytes() == b"gen-736"
    assert (root / "assets/graspgenx/dis/discriminator.pth").read_bytes() == b"dis-1056"
    files = {item.path for item in validated.manifest.bundle.files}
    assert {
        "assets/adapter.json",
        "assets/x_grippers/omnihand_left_x2/config.json",
        "assets/x_grippers/omnihand_left_x2/meshes/palm.obj",
    } <= files
    adapter = json.loads((root / "assets/adapter.json").read_text(encoding="utf-8"))
    assert adapter["torch_module_loader"] == "perception_service.torch_model_loaders:load_graspgenx"
    assert GraspGenXAdapter.from_bundle(root).config.gripper_name == "omnihand_left_x2"


def test_repackaging_in_place_keeps_the_revision_and_a_new_checkpoint_bumps_it(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    first = load_inference_manifest(root, "torch_cuda").manifest.bundle

    package_graspgenx_torch_bundle(root)
    assert load_inference_manifest(root, "torch_cuda").manifest.bundle.revision == first.revision

    (release / "gen" / "epoch_736.pth").write_bytes(b"retrained")
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    changed = load_inference_manifest(root, "torch_cuda").manifest.bundle
    assert (changed.uuid, changed.revision) == (first.uuid, first.revision + 1)

    # bundle.files and the digest record paths only; a changed mesh must still bump it.
    (gripper / "meshes" / "palm.obj").write_text("v 1 0 0\n", encoding="utf-8")
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    assert load_inference_manifest(root, "torch_cuda").manifest.bundle.revision == first.revision + 2


def test_the_packager_fails_closed_on_missing_sources(tmp_path):
    release, gripper = _sources(tmp_path)
    (release / "dis" / "config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="dis config is missing"):
        package_graspgenx_torch_bundle(tmp_path / "a", checkpoint_root=release, gripper_dir=gripper)

    release, gripper = _sources(tmp_path / "b")
    (gripper / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="no config.json"):
        package_graspgenx_torch_bundle(tmp_path / "c", checkpoint_root=release, gripper_dir=gripper)

    with pytest.raises(FileNotFoundError, match="provide --checkpoint-root"):
        package_graspgenx_torch_bundle(tmp_path / "empty")
    with pytest.raises(ValueError, match="pass both"):
        package_graspgenx_torch_bundle(tmp_path / "d", checkpoint_root=release)


def test_repackaging_under_another_gripper_name_keeps_one_gripper(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper, gripper_name="omnihand_right_x2")

    files = {item.path for item in load_inference_manifest(root, "torch_cuda").manifest.bundle.files}
    assert not [path for path in files if "omnihand_left_x2" in path]
    assert "assets/x_grippers/omnihand_right_x2/config.json" in files
