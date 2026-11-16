"""Calibrated head-camera overlay: projection convention and drawing (no ROS)."""

from __future__ import annotations

import numpy as np
import pytest
from robodojo_robot.overlay import CameraModel, draw_overlay

# RoboDojo arx_x5 cam_head at episode start: behind the table, 30 deg about x (looking +y and down).
HEAD = {
    "K": [[288.1325, 0.0, 320.0], [0.0, 288.1325, 240.0], [0.0, 0.0, 1.0]],
    "resolution": [640, 480],
    "cam_to_world": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.8660253, -0.5000001, -0.41],
        [0.0, 0.5000001, 0.8660253, 1.308],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


def test_usd_camera_projection_matches_the_rendered_scene():
    camera = CameraModel.from_calibration(HEAD, (640, 480))
    # The scissors of layout arx_x5/0/general_pickup_0 render at about (495, 215) in the head image.
    u, v, depth = camera.project([(0.397, -0.037, 0.772)])[0]
    assert depth > 0 and u == pytest.approx(496, abs=3) and v == pytest.approx(216, abs=3)
    # x grows to the right, y grows up the image (away from the camera)
    centre, right, far = camera.project([(0.0, 0.0, 0.765), (0.2, 0.0, 0.765), (0.0, 0.2, 0.765)])
    assert right[0] > centre[0] and far[1] < centre[1]


def test_intrinsics_scale_with_the_image_size():
    full = CameraModel.from_calibration(HEAD, (640, 480)).project([(0.1, 0.1, 0.765)])[0]
    half = CameraModel.from_calibration(HEAD, (320, 240)).project([(0.1, 0.1, 0.765)])[0]
    assert half[0] == pytest.approx(full[0] / 2) and half[1] == pytest.approx(full[1] / 2)


def test_draw_overlay_marks_grid_and_grippers_without_touching_the_input():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = CameraModel.from_calibration(HEAD, (640, 480))
    out = draw_overlay(rgb, camera, table_z=0.765, grippers={"left": [-0.3, -0.1, 0.9], "right": [0.3, -0.1, 0.9]})
    assert out.shape == rgb.shape and rgb.max() == 0
    assert (out[..., 0] > 200).sum() > 500  # yellow grid and labels drawn
    u, v, _ = camera.project([(0.3, -0.1, 0.765)])[0]  # red circle around the right foot point
    patch = out[int(v) - 6 : int(v) + 7, int(u) - 6 : int(u) + 7]
    assert ((patch[..., 0] > 200) & (patch[..., 2] < 100)).any()
