from perception_service.response_parser import parse_scene_analysis_response


def test_parse_scene_analysis_response():
    analysis = parse_scene_analysis_response(
        """
        {
          "scene_summary": "桌面上有一个红色方块",
          "visible_objects": ["红色方块", "机械臂"],
          "robot_state_summary": "机械臂处于观察位姿",
          "ee_pose_interpretation": "末端位于桌面上方",
          "risks": ["无明显风险"],
          "confidence": 0.92
        }
        """
    )
    assert analysis.scene_summary == "桌面上有一个红色方块"
    assert analysis.visible_objects == ["红色方块", "机械臂"]
    assert analysis.risks == ["无明显风险"]
    assert analysis.confidence == 0.92


def test_parse_scene_analysis_response_accepts_code_fence():
    analysis = parse_scene_analysis_response(
        """```json
        {
          "scene_summary": "ok",
          "visible_objects": "机械臂",
          "robot_state_summary": "ready",
          "ee_pose_interpretation": "base front",
          "risks": [],
          "confidence": 0.5
        }
        ```"""
    )
    assert analysis.visible_objects == ["机械臂"]


def test_parse_scene_analysis_response_accepts_string_confidence_label():
    analysis = parse_scene_analysis_response(
        """
        {
          "scene_summary": "桌面上可能有水果",
          "visible_objects": ["香蕉"],
          "robot_state_summary": "机械臂在观察位",
          "ee_pose_interpretation": "末端在桌面上方",
          "risks": ["目标存在部分遮挡"],
          "confidence": "medium"
        }
        """
    )
    assert analysis.confidence == 0.6
