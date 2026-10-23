# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
"""Compare the optimized step to diffusers with the same variance noise."""

import pytest
import torch
from diffusers import DDPMScheduler

from torch_models.graspgenx_ascend_310p.diffusion_scheduler import AscendDDPMStep


@pytest.mark.parametrize("schedule", ["linear", "squaredcos_cap_v2"])
@pytest.mark.parametrize("steps", [20, 10, 4])
@pytest.mark.parametrize("clip", [True, False])
def test_every_step_matches_diffusers_with_identical_noise(schedule, steps, clip):
    scheduler = DDPMScheduler(num_train_timesteps=20, beta_schedule=schedule, clip_sample=clip)
    scheduler.set_timesteps(steps)
    fast = AscendDDPMStep(scheduler)
    rng = torch.Generator().manual_seed(17)
    reference = torch.randn((7, 3), generator=rng)
    actual = reference.clone()
    for timestep in scheduler.timesteps:
        prediction = torch.randn(reference.shape, generator=rng)
        expected = scheduler.step(prediction, timestep, reference, generator=torch.Generator().manual_seed(29))
        result = fast(prediction, timestep, actual, generator=torch.Generator().manual_seed(29))
        torch.testing.assert_close(result.prev_sample, expected.prev_sample, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(result.pred_original_sample, expected.pred_original_sample, rtol=2e-5, atol=2e-6)
        reference, actual = expected.prev_sample, result.prev_sample


def test_changing_schedule_invalidates_a_cached_timestep_edge():
    scheduler = DDPMScheduler(num_train_timesteps=20)
    fast = AscendDDPMStep(scheduler)
    scheduler.set_timesteps(20)
    fast.coefficients(18)
    scheduler.set_timesteps(10)
    prediction, sample = torch.ones((2, 3)), torch.full((2, 3), 0.25)
    expected = scheduler.step(prediction, 18, sample, generator=torch.Generator().manual_seed(31))
    actual = fast(prediction, 18, sample, generator=torch.Generator().manual_seed(31))
    torch.testing.assert_close(actual.prev_sample, expected.prev_sample)


@pytest.mark.parametrize("variance", ["fixed_small_log", "fixed_large_log", "learned", "learned_range"])
def test_variances_the_scalar_shortcut_cannot_represent_are_declined(variance):
    assert not AscendDDPMStep(DDPMScheduler(variance_type=variance)).supported()
