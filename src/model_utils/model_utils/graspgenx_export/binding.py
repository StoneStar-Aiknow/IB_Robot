# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
"""Bind compiled loops and reusable constants to the generator they replace."""

import hashlib
from pathlib import Path

import yaml


def generator_binding(checkpoint: Path, config: Path) -> dict:
    def digest(path):
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                sha.update(block)
        return sha.hexdigest()

    diffusion = yaml.safe_load(config.read_text(encoding="utf-8"))["diffusion"]
    steps = diffusion.get("num_diffusion_iters_eval", diffusion.get("num_diffusion_iters"))
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2:
        raise ValueError("compiled diffusion loops require at least two inference steps")
    return {
        "version": 1,
        "generator_checkpoint_sha256": digest(checkpoint),
        "generator_config_sha256": digest(config),
        "steps": steps,
    }


def validate_binding(metadata: dict, expected: dict, grasps: int) -> None:
    if metadata.get("generator_binding") != expected:
        raise ValueError("diffusion loop generator binding is missing or incompatible; re-export and compile the loop")
    if metadata.get("grasps") != grasps or metadata.get("steps") != expected["steps"]:
        raise ValueError(f"diffusion loop must cover {grasps} grasps at {expected['steps']} steps")
