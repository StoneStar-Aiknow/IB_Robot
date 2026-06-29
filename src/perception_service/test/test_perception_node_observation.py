import json
from unittest.mock import Mock

import rclpy
from rclpy.parameter import Parameter

from ibrobot_msgs.msg import SceneAnalysisRequest
from perception_service.perception_service_node import PerceptionServiceNode
from perception_service.response_parser import parse_scene_analysis_response


def test_perception_node_publishes_grounded_observation_from_vlm_bbox():
    rclpy.init()
    try:
        node = PerceptionServiceNode(
            parameter_overrides=[
                Parameter("primary_camera_topic", Parameter.Type.STRING, ""),
                Parameter("wrist_camera_topic", Parameter.Type.STRING, ""),
                Parameter("ee_pose_topic", Parameter.Type.STRING, "/unused/ee_pose"),
                Parameter("joint_state_topic", Parameter.Type.STRING, "/unused/joint_states"),
            ]
        )
        published = []
        node._observation_publisher = Mock()  # noqa: SLF001
        node._observation_publisher.publish.side_effect = published.append  # noqa: SLF001
        scene_snapshot = {
            "camera_topic": "/camera/camera/color/image_raw",
            "image_data_url": "data:image/jpeg;base64,abc",
            "images": [],
            "camera_views": {
                "primary": {
                    "camera_info": {
                        "available": True,
                        "width": 640,
                        "height": 480,
                    }
                }
            },
            "rgbd_context": {},
            "ee_pose": {"frame_id": "base"},
            "joint_state": {"name": ["1"], "position": [0.0]},
            "errors": [],
        }
        raw_response = json.dumps(
            {
                "scene_summary": "桌面上有一个香蕉",
                "visible_objects": ["香蕉"],
                "objects": [
                    {
                        "label": "banana",
                        "bbox_2d": [120, 80, 260, 220],
                        "confidence": 0.91,
                        "attributes": {"color": "yellow", "graspability": "good"},
                    }
                ],
                "robot_state_summary": "机械臂处于观察位姿",
                "ee_pose_interpretation": "末端位于桌面上方",
                "risks": [],
                "confidence": 0.88,
            },
            ensure_ascii=False,
        )
        request = SceneAnalysisRequest()
        request.request_id = "req-1"
        request.source = "test"
        request.session_id = "session-1"

        analysis = parse_scene_analysis_response(raw_response)
        node._publish_result(  # noqa: SLF001
            request=request,
            success=True,
            message="scene analysis completed",
            analysis=analysis,
            raw_response=raw_response,
            scene_snapshot=scene_snapshot,
        )

        assert len(published) == 1
        observation = published[0]
        assert observation.success is True
        assert observation.scene_summary == "桌面上有一个香蕉"
        assert len(observation.objects) == 1
        assert observation.objects[0].label == "banana"
        assert observation.objects[0].bbox_x == 120
        assert observation.objects[0].bbox_y == 80
        assert observation.objects[0].bbox_width == 140
        assert observation.objects[0].bbox_height == 140
        assert observation.objects[0].confidence == 0.91
    finally:
        node.destroy_node()
        rclpy.shutdown()
