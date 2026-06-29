from pathlib import Path

import cv2
import numpy as np

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "realsense_rgbd_frame"


def test_realsense_rgbd_fixture_is_offline_readable():
    color = cv2.imread(str(FIXTURE_DIR / "color.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(FIXTURE_DIR / "depth.png"), cv2.IMREAD_UNCHANGED)
    depth_meters = np.load(FIXTURE_DIR / "depth_meters.npy")

    assert color is not None
    assert depth is not None
    assert color.shape[:2] == depth.shape[:2] == depth_meters.shape
    assert color.dtype == np.uint8
    assert depth.dtype == np.uint16
    assert depth_meters.dtype == np.float32
    assert np.count_nonzero(np.isfinite(depth_meters) & (depth_meters > 0.0)) > 0
