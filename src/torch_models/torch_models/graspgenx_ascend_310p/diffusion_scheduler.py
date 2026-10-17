# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
"""DDPM inference arithmetic shared by the eager loop and the graph exporter."""

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


def host_noise(shape, dtype, device, generator=None):
    """The 310P device RNG is unavailable; draw on the host and copy."""
    return torch.randn(shape, generator=generator, dtype=dtype).to(device)


def ddpm_coefficients(scheduler, timestep: int):
    """Use the actual inference schedule, including skipped training steps."""
    previous = int(scheduler.previous_timestep(timestep))
    alpha = scheduler.alphas_cumprod[timestep]
    previous_alpha = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.one
    beta = 1 - alpha
    current_alpha = alpha / previous_alpha
    current_beta = 1 - current_alpha
    return (
        float(beta**0.5),
        float(alpha**0.5),
        float(previous_alpha**0.5 * current_beta / beta),
        float(current_alpha**0.5 * (1 - previous_alpha) / beta),
        float(scheduler._get_variance(timestep) ** 0.5) if timestep > 0 else 0.0,
    )


class AscendDDPMStep:
    """Cache host scalars per schedule edge instead of dispatching scalar ops."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._coefficients: dict = {}

    def supported(self) -> bool:
        scheduler = self.scheduler
        return (
            getattr(scheduler.config, "prediction_type", "epsilon") == "epsilon"
            and getattr(scheduler, "variance_type", "fixed_small") in {"fixed_small", "fixed_large"}
            and not getattr(scheduler.config, "thresholding", False)
            and getattr(scheduler.config, "clip_sample_range", 1.0) == 1.0
            and hasattr(scheduler, "alphas_cumprod")
            and hasattr(scheduler, "previous_timestep")
        )

    def coefficients(self, timestep: int):
        # set_timesteps can change the edge even when timestep itself recurs.
        key = (timestep, int(self.scheduler.previous_timestep(timestep)))
        if key not in self._coefficients:
            self._coefficients[key] = ddpm_coefficients(self.scheduler, timestep)
        return self._coefficients[key]

    def __call__(self, model_output, timestep, sample, generator=None, return_dict=True, **kwargs):
        if kwargs or not self.supported():
            return type(self.scheduler).step(
                self.scheduler,
                model_output,
                timestep,
                sample,
                generator=generator,
                return_dict=return_dict,
                **kwargs,
            )
        step = int(timestep)
        beta_sqrt, alpha_sqrt, original_coeff, sample_coeff, variance_sqrt = self.coefficients(step)
        original = (sample - beta_sqrt * model_output) / alpha_sqrt
        if self.scheduler.config.clip_sample:
            original = torch.clamp(original, -1, 1)
        previous = original_coeff * original + sample_coeff * sample
        if step > 0:
            previous = previous + variance_sqrt * host_noise(
                model_output.shape,
                model_output.dtype,
                model_output.device,
                generator,
            )
        if not return_dict:
            return (previous,)
        return DDPMSchedulerOutput(prev_sample=previous, pred_original_sample=original)
