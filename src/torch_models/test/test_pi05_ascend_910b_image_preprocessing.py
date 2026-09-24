#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU parity tests for the PI0.5 Ascend910B image preprocessing hot path."""

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from torch import nn

from torch_models.pi05_ascend_910b.modeling_pi05_ascend_910b import (
    PI05Ascend910BPolicy,
    resize_with_pad_torch,
)

IMAGE_KEYS = (
    "observation.images.base",
    "observation.images.wrist",
    "observation.images.empty_camera_0",
)


class _ImageHarness(nn.Module):
    preprocess_raw_image_for_inference = PI05Ascend910BPolicy.preprocess_raw_image_for_inference
    _preprocess_images = PI05Ascend910BPolicy._preprocess_images

    def __init__(self, config: PI05Config) -> None:
        super().__init__()
        self.config = config
        self.register_parameter("_device_anchor", nn.Parameter(torch.empty(0), requires_grad=False))


def _config() -> PI05Config:
    config = PI05Config(device="cpu")
    config.input_features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for key in IMAGE_KEYS}
    return config


def _legacy_preprocess_image(image: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    is_channels_first = image.shape[1] == 3
    if is_channels_first:
        image = image.permute(0, 2, 3, 1)
    if tuple(image.shape[1:3]) != resolution:
        image = resize_with_pad_torch(image, *resolution)
    image = image * 2.0 - 1.0
    if is_channels_first:
        image = image.permute(0, 3, 1, 2)
    return image


def test_raw_image_hook_converts_bhwc_uint8_without_value_drift() -> None:
    harness = _ImageHarness(_config())
    raw = torch.randint(0, 256, (2, 19, 31, 3), dtype=torch.uint8)

    actual = harness.preprocess_raw_image_for_inference(raw)
    expected = raw.permute(0, 3, 1, 2).contiguous().float() / 255.0

    assert actual.shape == (2, 3, 19, 31)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "raw",
    [
        torch.zeros(1, 3, 8, 8, dtype=torch.uint8),
        torch.zeros(1, 8, 8, 3, dtype=torch.float32),
    ],
)
def test_raw_image_hook_rejects_invalid_contract(raw) -> None:
    harness = _ImageHarness(_config())
    with pytest.raises((TypeError, ValueError)):
        harness.preprocess_raw_image_for_inference(raw)


@pytest.mark.parametrize("layout", ["bchw", "bhwc"])
def test_model_image_preprocessing_matches_legacy_and_preserves_input(layout) -> None:
    config = _config()
    harness = _ImageHarness(config)
    bchw = torch.rand(2, 3, 180, 320, dtype=torch.float32)
    source = bchw if layout == "bchw" else bchw.permute(0, 2, 3, 1).contiguous()
    source_before = source.clone()

    images, masks = harness._preprocess_images({IMAGE_KEYS[0]: source})
    expected = _legacy_preprocess_image(source, tuple(config.image_resolution))
    if layout == "bhwc":
        expected = expected.permute(0, 3, 1, 2)

    assert len(images) == len(masks) == 3
    assert images[0].shape == (2, 3, 224, 224)
    torch.testing.assert_close(images[0], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(source, source_before, rtol=0.0, atol=0.0)
    assert masks[0].tolist() == [True, True]
    for image, mask in zip(images[1:], masks[1:], strict=True):
        assert torch.count_nonzero(image.add(1.0)) == 0
        assert mask.tolist() == [False, False]
