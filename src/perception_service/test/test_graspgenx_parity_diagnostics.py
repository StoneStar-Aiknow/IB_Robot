# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
"""The diagnostics must not report unrelated samples as device parity."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler
from verify_graspgenx_loop_parity import eager_loop
from verify_graspgenx_pose_parity import compare, geometric_matches


def _poses(count):
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, 0, 3] = np.arange(count) * 0.02
    return poses


def test_unrelated_sorted_grasps_have_no_geometric_overlap(tmp_path, capsys):
    left = _poses(400)
    right = left.copy()
    right[:, 1, 3] = 1.0
    paths = [tmp_path / name for name in ("left.npz", "right.npz")]
    for path, poses in zip(paths, (left, right), strict=True):
        np.savez(path, pose=poses, confidence=np.linspace(1, 0, 400))
    assert compare(*paths) == 0
    output = capsys.readouterr().out
    for count in (10, 50, 200, 400):
        assert f"geometric matches 0/{count}" in output


def test_permuted_poses_are_matched_by_geometry_and_each_pose_is_used_once():
    poses = _poses(5)
    permutation = [3, 0, 4, 1, 2]
    i, j, translation, rotation = geometric_matches(poses, poses[permutation], 5, 10)
    assert len(i) == 5
    np.testing.assert_allclose(poses[i], poses[permutation][j])
    assert not translation.any() and not rotation.any()
    assert len(geometric_matches(poses[:1], np.repeat(poses[:1], 5, axis=0), 5, 10)[0]) == 1


def test_rotation_tolerance_prevents_a_translation_only_match():
    pose = _poses(1)
    rotated = pose.copy()
    rotated[0, :3, :3] = np.diag([-1, -1, 1])
    assert len(geometric_matches(pose, rotated, 5, 10)[0]) == 0


@pytest.mark.parametrize("steps", [20, 10])
def test_reference_records_post_step_latents_and_is_repeatable(steps):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, observation, timestep, sample):
            return sample * self.weight

    generator = SimpleNamespace(
        diffusion_head=Head(),
        noise_scheduler_pos=DDPMScheduler(num_train_timesteps=20),
        noise_scheduler_rot=DDPMScheduler(num_train_timesteps=20),
    )
    generator.noise_scheduler_pos.set_timesteps(steps)
    timesteps = generator.noise_scheduler_pos.timesteps.tolist()
    rng = np.random.default_rng(13)
    inputs = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in {
            "observation": (4, 1024),
            "initial_noise": (4, 6),
            "position_noise": (steps - 1, 4, 3),
            "rotation_noise": (steps - 1, 4, 3),
        }.items()
    }
    first = eager_loop(generator, inputs, timesteps)
    second = eager_loop(generator, inputs, timesteps)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(first["latents"][-1], first["grasps"])
    assert not np.array_equal(first["latents"][0], inputs["initial_noise"])
    assert np.isfinite(first["likelihood"]).all()
