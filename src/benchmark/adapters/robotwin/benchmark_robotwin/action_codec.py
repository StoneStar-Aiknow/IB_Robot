"""RoboTwin action codec: policy action dict -> canonical dual-arm flat action.

The generic ``StepBenchmark`` service decodes the wire payload via
``benchmark_runtime.action_codec.decode_step_action_generic`` into a
``dict[str, np.ndarray]``. This module validates that dict against the
RoboTwin dual-arm action contract and returns a canonical ``float32``
flat array of shape ``(arm_dim + 1) * 2``:

    ``[left_arm(arm_dim), left_gripper(1), right_arm(arm_dim),
    right_gripper(1)]``

For the default ``aloha-agilex`` embodiment this is 14 floats
(arm_dim=6 per arm); ``franka-panda`` uses arm_dim=7 (16 floats).

Pre-native validation failures raise
:class:`benchmark_runtime.adapter.PreNativeValidationError` so the
environment node can retry the step without poisoning the episode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark_runtime.adapter import PreNativeValidationError

_ACTION_KEY = "action"
_DEFAULT_ARM_DIM = 6


def validate_robotwin_action_dict(action: dict[str, Any], arm_dim: int = _DEFAULT_ARM_DIM) -> np.ndarray:
    """Validate the decoded action dict and return canonical float32 flat action.

    Raises :class:`PreNativeValidationError` on any contract violation.
    """
    if not isinstance(action, dict):
        raise PreNativeValidationError(f"action must be a dict, got {type(action).__name__}")
    if len(action) != 1 or _ACTION_KEY not in action:
        raise PreNativeValidationError(
            f"action dict must contain exactly one key '{_ACTION_KEY}', got keys {sorted(action.keys())}"
        )
    raw = action[_ACTION_KEY]
    if not isinstance(raw, np.ndarray):
        raise PreNativeValidationError(f"action['{_ACTION_KEY}'] must be a numpy array, got {type(raw).__name__}")

    expected_dim = (arm_dim + 1) * 2
    flat = np.asarray(raw, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise PreNativeValidationError("action array is empty")
    if flat.size != expected_dim:
        raise PreNativeValidationError(f"action dim must be {expected_dim} (arm_dim={arm_dim}), got {flat.size}")
    if not np.all(np.isfinite(flat)):
        raise PreNativeValidationError("action array contains NaN or Inf")

    left_arm = flat[:arm_dim]
    left_gripper = flat[arm_dim : arm_dim + 1]
    right_arm = flat[arm_dim + 1 : 2 * arm_dim + 1]
    right_gripper = flat[2 * arm_dim + 1 : 2 * arm_dim + 2]

    # qpos actions are absolute joint targets in radians: embodiment joints
    # legitimately command beyond +/-1 rad, so the arm channels must pass
    # through unclipped. Only the gripper channels are normalized [0, 1].
    left_gripper = np.clip(left_gripper, 0.0, 1.0)
    right_gripper = np.clip(right_gripper, 0.0, 1.0)

    canonical = np.concatenate([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)
    return np.ascontiguousarray(canonical)
