from perception_service.prompt_builder import build_scene_analysis_messages


def test_build_messages_includes_context_history_and_image():
    messages = build_scene_analysis_messages(
        user_text="看看桌面",
        user_context={"focus_object": "red_block"},
        scene_snapshot={
            "camera_topic": "/camera/top/image_raw",
            "image_data_url": "data:image/jpeg;base64,abc",
            "images": [
                {
                    "view_name": "primary",
                    "camera_topic": "/camera/top/image_raw",
                    "image_data_url": "data:image/jpeg;base64,abc",
                },
                {
                    "view_name": "wrist",
                    "camera_topic": "/camera/wrist/image_raw",
                    "image_data_url": "data:image/jpeg;base64,def",
                },
            ],
            "camera_views": {
                "primary": {"camera_topic": "/camera/top/image_raw"},
                "wrist": {"camera_topic": "/camera/wrist/image_raw"},
            },
            "rgbd_context": {"multi_view_enabled": True, "depth_available_views": ["primary"]},
            "ee_pose": {"frame_id": "base"},
            "joint_state": {"name": ["1"], "position": [0.1]},
        },
        conversation_history=[
            {"role": "user", "text": "上一轮问了桌面"},
            {"role": "assistant", "text": "上一轮回复了摘要"},
        ],
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert messages[-1]["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "focus_object" in messages[-1]["content"][0]["text"]
    assert "analysis_preferences" in messages[-1]["content"][0]["text"]
    assert "rgbd_context" in messages[-1]["content"][0]["text"]
    assert "The user question is the primary semantic objective" in messages[0]["content"][0]["text"]


def test_build_messages_infers_manipulation_preferences_from_user_text():
    messages = build_scene_analysis_messages(
        user_text="把桌面上的香蕉夹起来",
        user_context={},
        scene_snapshot={
            "camera_topic": "/camera/top/image_raw",
            "image_data_url": "",
            "images": [],
            "rgbd_context": {"depth_available_views": ["primary"]},
            "ee_pose": {"frame_id": "base"},
            "joint_state": {"name": ["1"], "position": [0.1]},
        },
        conversation_history=[],
    )

    payload_text = messages[-1]["content"][0]["text"]
    assert "manipulation_assessment" in payload_text
    assert "graspability and reachability" in payload_text
    assert "raw_user_intent" in payload_text
    assert "RGB-D" in payload_text
