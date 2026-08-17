from collections import deque

import cv2
import numpy as np

from robot_calibration.live_overlay import LiveOverlay, _overlay


class CameraInfo:
    k = [100.0, 0.0, 5.0, 0.0, 100.0, 5.0, 0.0, 0.0, 1.0]


def test_live_overlay_draws_visible_lidar_points_with_two_pixel_radius(monkeypatch):
    radii = []
    monkeypatch.setattr(cv2, "circle", lambda _image, _center, radius, *_args: radii.append(radius))

    _overlay(np.zeros((10, 10, 3), dtype=np.uint8), np.array([[0.0, 0.0, 1.0]]), np.eye(4), CameraInfo())

    assert radii == [2]


def test_live_overlay_colors_points_by_camera_depth(monkeypatch):
    colors = []
    monkeypatch.setattr(
        cv2,
        "circle",
        lambda _image, _center, _radius, color, *_args: colors.append(color),
    )

    _overlay(
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 4.0]]),
        np.eye(4),
        CameraInfo(),
    )

    assert len(colors) == 2
    assert colors[0] != colors[1]


def test_live_overlay_accumulates_three_recent_clouds(monkeypatch):
    node = LiveOverlay.__new__(LiveOverlay)
    node._cloud_history = deque(maxlen=3)
    monkeypatch.setattr(
        "robot_calibration.live_overlay.point_cloud2.read_points_numpy",
        lambda message, **_kwargs: message,
    )

    for value in range(1, 5):
        node._cloud_callback(np.full((2, 3), value, dtype=float))

    assert node._cloud.shape == (6, 3)
    assert np.array_equal(node._cloud[:, 0], [2, 2, 3, 3, 4, 4])
