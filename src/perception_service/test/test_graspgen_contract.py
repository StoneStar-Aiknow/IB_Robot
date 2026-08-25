# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""The exporter and the packager must describe the same GraspGen pipeline.

The two tools live in different packages - ``model_utils`` exports the ONNX subgraphs and
compiles them, ``perception_service`` packages the OMs into a perception bundle - and each
used to carry its own copy of the eight-role order and of the PointNet++ sampling
constants. Nothing compared the copies, so an edit to one would produce OMs the other
packaged happily and the session then drove with the wrong grouping. They read
``inference_manifest.graspgen`` now, and these tests hold them to it - including the
contract version, which is what stops a stale export from being packaged at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_utils.graspgen_contract import (
    GRASPGEN_CONTRACT_VERSION,
    GRASPGEN_EXECUTION,
    GRASPGEN_NPOINTS,
    GRASPGEN_NSAMPLES,
    GRASPGEN_RADII,
    graspgen_geometry,
)
from model_utils.graspgen_export import ONNX_ARTIFACT_ORDER
from model_utils.graspgen_export.export_onnx import ARTIFACT_ORDER, MANIFEST_NAME, _write_manifest
from perception_service import package_graspgen_ascend_bundle as packager

_CONFIG = {
    "diffusion": {
        "grasp_repr": "r3_so3",
        "kappa": 2.02217,
        "num_diffusion_iters_eval": 10,
        "compositional_schedular": True,
    },
    "data": {"num_points": 2048},
}


def _export_manifest(tmp_path: Path) -> dict:
    path = _write_manifest(
        output_dir=tmp_path,
        config_path=tmp_path / "gripper.yaml",
        config=_CONFIG,
        generator_checkpoint=tmp_path / "generator.pth",
        discriminator_checkpoint=tmp_path / "discriminator.pth",
        artifacts={role: {"onnx": f"{role}.onnx"} for role in GRASPGEN_EXECUTION},
        grasp_batch_size=1000,
        opset=14,
        constant_folding=True,
        simplify=True,
    )
    assert path == tmp_path / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_exporter_and_packager_share_one_execution_order():
    assert list(GRASPGEN_EXECUTION) == ARTIFACT_ORDER
    assert list(GRASPGEN_EXECUTION) == ONNX_ARTIFACT_ORDER
    assert packager.GRASPGEN_EXECUTION is GRASPGEN_EXECUTION


def test_exported_manifest_carries_the_shared_execution_and_geometry(tmp_path):
    manifest = _export_manifest(tmp_path)

    assert manifest["execution"] == list(GRASPGEN_EXECUTION)
    assert manifest["backend_config"]["geometry"] == graspgen_geometry(include_head_stage=True)


def test_exported_manifest_stamps_the_contract_the_packager_requires(tmp_path):
    manifest = _export_manifest(tmp_path)

    assert manifest["contract_version"] == GRASPGEN_CONTRACT_VERSION
    packager.require_contract_version(manifest)


@pytest.mark.parametrize("declared", [None, GRASPGEN_CONTRACT_VERSION + 1, "1"])
def test_packager_refuses_an_export_from_another_contract(declared):
    manifest = {"model_type": "graspgen"}
    if declared is not None:
        manifest["contract_version"] = declared

    with pytest.raises(ValueError, match="contract_version"):
        packager.require_contract_version(manifest)


def test_the_encoder_head_is_the_only_difference_between_the_two_geometry_forms():
    """The head groups whatever stage two produced, so it samples nothing of its own."""
    exported = graspgen_geometry(include_head_stage=True)
    packaged = graspgen_geometry()

    for key, sampled in (
        ("npoints", list(GRASPGEN_NPOINTS)),
        ("radii", list(GRASPGEN_RADII)),
        ("nsamples", list(GRASPGEN_NSAMPLES)),
    ):
        assert packaged[key] == sampled
        assert exported[key] == [*sampled, None]
    assert exported["fps_start_index"] == packaged["fps_start_index"]
    assert exported["ball_query_order"] == packaged["ball_query_order"]
