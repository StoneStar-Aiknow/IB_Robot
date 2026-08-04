# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""GraspGen ONNX/OM export tooling for Ascend deployment.

This package ports the upstream ModelZoo-PyTorch/ACL_PyTorch GraspGen toolchain
into IB-Robot so that the eight compiled Ascend OM sub-graphs can be produced
from a single command and packaged into a unified ``inference_manifest.json``
that the standard ``inference_service.backends.ascend.AscendBackend`` consumes.

The host-side geometry (FPS / ball-query / grouping), the dual DDPM schedulers
and the SO(3) helpers live in
``inference_service.backends.ascend.graspgen_runtime`` and are shared between
the backend runtime and the export-time verification tools.
"""

from model_utils.graspgen_export.export_onnx import ARTIFACT_ORDER as ONNX_ARTIFACT_ORDER

__all__ = ["ONNX_ARTIFACT_ORDER"]
