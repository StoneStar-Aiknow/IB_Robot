"""RoboTwin observation codec: raw Sapien obs → policy contract dict.

RoboTwin ``Base_Task.get_obs()`` returns a nested dict with three cameras
(``head_camera``, ``left_camera``, ``right_camera``) and a ``joint_action``
dict carrying dual-arm joint targets + normalized gripper values.

This codec flattens that into the flat ``dict[str, np.ndarray]`` the
benchmark runtime expects, with keys matching the robot YAML contract:

- ``observation.images.head``         — third-person static camera RGB.
- ``observation.images.left_wrist``   — left eye-in-hand camera RGB.
- ``observation.images.right_wrist``  — right eye-in-hand camera RGB.
- ``observation.state``               — dual-arm state vector
  ``[left_arm_qpos(arm_dim), left_gripper(1), right_arm_qpos(arm_dim),
  right_gripper(1)]`` as ``float32``.

Images are resized to ``TARGET_IMAGE_SIZE`` (480, 640) via nearest-neighbor
to match the RobotWin PI05 contract and avoid hard PIL/cv2 dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

POLICY_IMAGE_HEAD_KEY = "observation.images.cam_high"
POLICY_IMAGE_LEFT_KEY = "observation.images.cam_left_wrist"
POLICY_IMAGE_RIGHT_KEY = "observation.images.cam_right_wrist"
POLICY_STATE_KEY = "observation.state"

TARGET_IMAGE_SIZE: tuple[int, int] = (480, 640)
_DEFAULT_ARM_DIM = 6

_REQUIRED_OBS_KEYS = ("head_camera", "left_camera", "right_camera")
_REQUIRED_JOINT_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")


class ObservationCodecError(ValueError):
    """Raised when the RoboTwin raw observation cannot be converted."""


def _resize_image_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    row_idx = np.linspace(0, h - 1, target_h).round().astype(np.intp)
    col_idx = np.linspace(0, w - 1, target_w).round().astype(np.intp)
    return img[row_idx][:, col_idx]


def _extract_camera_rgb(raw_obs: Mapping[str, Any], camera_name: str) -> np.ndarray:
    camera_block = raw_obs.get("observation", {})
    if not isinstance(camera_block, Mapping):
        raise ObservationCodecError("raw observation.observation must be a Mapping")
    camera = camera_block.get(camera_name)
    if not isinstance(camera, Mapping):
        raise ObservationCodecError(f"camera '{camera_name}' missing from raw observation")
    rgb = camera.get("rgb")
    if not isinstance(rgb, np.ndarray):
        raise ObservationCodecError(f"camera '{camera_name}' rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ObservationCodecError(f"camera '{camera_name}' rgb must be HWC uint8, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def _extract_state(raw_obs: Mapping[str, Any], arm_dim: int) -> np.ndarray:
    joint_action = raw_obs.get("joint_action")
    if not isinstance(joint_action, Mapping):
        raise ObservationCodecError("raw observation.joint_action must be a Mapping")
    left_arm = joint_action.get("left_arm")
    right_arm = joint_action.get("right_arm")
    left_gripper = joint_action.get("left_gripper")
    right_gripper = joint_action.get("right_gripper")
    for key, val in (
        ("left_arm", left_arm),
        ("right_arm", right_arm),
        ("left_gripper", left_gripper),
        ("right_gripper", right_gripper),
    ):
        if key not in joint_action:
            raise ObservationCodecError(f"joint_action.{key} missing from raw observation")
        if val is None:
            raise ObservationCodecError(f"joint_action.{key} is None")
    left_arm = np.asarray(left_arm, dtype=np.float32).reshape(-1)
    right_arm = np.asarray(right_arm, dtype=np.float32).reshape(-1)
    left_gripper = np.asarray(left_gripper, dtype=np.float32).reshape(-1)
    right_gripper = np.asarray(right_gripper, dtype=np.float32).reshape(-1)
    if left_arm.size != arm_dim:
        raise ObservationCodecError(f"left_arm must have {arm_dim} entries, got {left_arm.size}")
    if right_arm.size != arm_dim:
        raise ObservationCodecError(f"right_arm must have {arm_dim} entries, got {right_arm.size}")
    if left_gripper.size != 1:
        raise ObservationCodecError(f"left_gripper must be a scalar, got shape {left_gripper.shape}")
    if right_gripper.size != 1:
        raise ObservationCodecError(f"right_gripper must be a scalar, got shape {right_gripper.shape}")
    if not np.all(np.isfinite(left_arm)) or not np.all(np.isfinite(right_arm)):
        raise ObservationCodecError("joint_action arm arrays contain NaN or Inf")
    if not np.all(np.isfinite(left_gripper)) or not np.all(np.isfinite(right_gripper)):
        raise ObservationCodecError("joint_action gripper values contain NaN or Inf")
    state = np.concatenate([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)
    if state.size != (arm_dim + 1) * 2:
        raise ObservationCodecError(f"state must have {(arm_dim + 1) * 2} entries, got {state.size}")
    return state


def convert_observation(raw_obs: Mapping[str, Any], arm_dim: int = _DEFAULT_ARM_DIM) -> dict[str, np.ndarray]:
    if not isinstance(raw_obs, Mapping):
        raise ObservationCodecError(f"raw_obs must be a Mapping, got {type(raw_obs).__name__}")

    head_rgb = _extract_camera_rgb(raw_obs, "head_camera")
    left_rgb = _extract_camera_rgb(raw_obs, "left_camera")
    right_rgb = _extract_camera_rgb(raw_obs, "right_camera")

    target_h, target_w = TARGET_IMAGE_SIZE
    head_resized = _resize_image_nearest(head_rgb, target_h, target_w)
    left_resized = _resize_image_nearest(left_rgb, target_h, target_w)
    right_resized = _resize_image_nearest(right_rgb, target_h, target_w)

    state = _extract_state(raw_obs, arm_dim)

    return {
        POLICY_IMAGE_HEAD_KEY: np.ascontiguousarray(head_resized, dtype=np.uint8),
        POLICY_IMAGE_LEFT_KEY: np.ascontiguousarray(left_resized, dtype=np.uint8),
        POLICY_IMAGE_RIGHT_KEY: np.ascontiguousarray(right_resized, dtype=np.uint8),
        POLICY_STATE_KEY: np.ascontiguousarray(state, dtype=np.float32),
    }
