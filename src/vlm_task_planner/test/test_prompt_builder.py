from vlm_task_planner.prompt_builder import build_chat_messages


def test_prompt_builder_includes_task_and_image():
    messages = build_chat_messages(
        task_text="抓取目标物并放到右侧托盘",
        scene_snapshot={
            "camera_topic": "/camera/top/image_raw",
            "image_data_url": "data:image/jpeg;base64,AAAA",
            "images": [
                {
                    "view_name": "primary",
                    "camera_topic": "/camera/top/image_raw",
                    "image_data_url": "data:image/jpeg;base64,AAAA",
                },
                {
                    "view_name": "wrist",
                    "camera_topic": "/camera/wrist/image_raw",
                    "image_data_url": "data:image/jpeg;base64,BBBB",
                },
            ],
            "camera_views": {
                "primary": {"camera_topic": "/camera/top/image_raw"},
                "wrist": {"camera_topic": "/camera/wrist/image_raw"},
            },
            "rgbd_context": {"depth_available_views": ["primary"]},
            "ee_pose": {"frame_id": "base"},
            "joint_state": {"name": ["1"], "position": [0.0]},
        },
        scene_analysis={
            "scene_summary": "桌面上能看到香蕉",
            "visible_objects": ["香蕉", "机械臂"],
        },
        allowed_skills=["inspect_scene", "recover_safe_pose"],
        named_poses={"tray_right": {}},
        named_targets={"demo_object": {}},
        workspace={"x": [0.0, 0.5]},
        relative_motion_reference_frame="base",
        relative_motion_direction_mapping={"forward": [1.0, 0.0, 0.0]},
    )
    assert messages[1]["content"][0]["type"] == "text"
    assert "抓取目标物并放到右侧托盘" in messages[1]["content"][0]["text"]
    assert "scene_understanding" in messages[1]["content"][0]["text"]
    assert "required_missing_skills" in messages[1]["content"][0]["text"]
    assert "Object grounding, picking, placing" in messages[0]["content"][0]["text"]
    assert messages[1]["content"][1]["type"] == "image_url"
    assert messages[1]["content"][2]["type"] == "image_url"
    assert "rgbd_context" in messages[1]["content"][0]["text"]


def test_prompt_builder_omits_image_when_missing():
    messages = build_chat_messages(
        task_text="回到home",
        scene_snapshot={
            "camera_topic": "/camera/top/image_raw",
            "image_data_url": "",
            "images": [],
            "rgbd_context": {},
            "ee_pose": None,
            "joint_state": None,
        },
        scene_analysis=None,
        allowed_skills=["recover_safe_pose"],
        named_poses={"home": {}},
        named_targets={},
        workspace={},
        relative_motion_reference_frame="base",
        relative_motion_direction_mapping={},
    )
    assert len(messages[1]["content"]) == 1
    assert "multi-view image context" in messages[0]["content"][0]["text"]
