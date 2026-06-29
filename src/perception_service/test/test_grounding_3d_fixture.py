import json
from pathlib import Path

import numpy as np

from perception_service.grounding_3d import bbox_to_3d_pose

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "realsense_rgbd_frame"


def test_bbox_to_3d_pose_uses_offline_realsense_fixture():
    depth_meters = np.load(FIXTURE_DIR / "depth_meters.npy")
    camera_info = json.loads((FIXTURE_DIR / "camera_info.json").read_text(encoding="utf-8"))
    frame_id = camera_info["header"]["frame_id"]

    pose = bbox_to_3d_pose(
        bbox_xyxy=(200, 120, 320, 260),
        depth_meters=depth_meters,
        camera_info=camera_info,
        frame_id=frame_id,
        max_depth_m=15.0,
    )

    assert pose is not None
    assert pose.header.frame_id == "camera_color_optical_frame"
    assert pose.pose.position.z > 0.0
    assert pose.pose.orientation.w == 1.0


def test_bbox_to_3d_pose_returns_none_for_invalid_bbox():
    depth_meters = np.load(FIXTURE_DIR / "depth_meters.npy")
    camera_info = json.loads((FIXTURE_DIR / "camera_info.json").read_text(encoding="utf-8"))

    assert bbox_to_3d_pose((10, 10, 10, 20), depth_meters, camera_info, "camera") is None
