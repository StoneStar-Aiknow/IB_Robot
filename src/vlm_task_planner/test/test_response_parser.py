from vlm_task_planner.response_parser import parse_planner_response


def test_parse_response_with_pick_and_place():
    plan = parse_planner_response(
        """
        {
          "intent": "pick_and_place",
          "skill_sequence": [
            {"skill_name": "pick_named_target", "args": {"target_name": "demo_object"}},
            {"skill_name": "place_named_pose", "args": {"place_name": "tray_right"}}
          ],
          "confidence": 0.91,
          "scene_summary": "target visible on the table"
        }
        """,
        allowed_skills=["pick_named_target", "place_named_pose"],
        default_target_name="demo_object",
        default_place_name="tray_right",
        default_relative_motion_step_m=0.03,
    )
    assert plan.task_type == "pick_and_place"
    assert plan.target_name == "demo_object"
    assert plan.place_name == "tray_right"
    assert plan.skill_sequence == ["pick_named_target", "place_named_pose"]
    assert plan.required_missing_skills == []
    assert plan.confidence == 0.91


def test_parse_response_rejects_unknown_skill():
    try:
        parse_planner_response(
            '{"skill_sequence": [{"skill_name": "freeform_pose", "args": {}}]}',
            allowed_skills=["inspect_scene"],
            default_target_name="demo_object",
            default_place_name="tray_right",
            default_relative_motion_step_m=0.03,
        )
    except ValueError as exc:
        assert "unsupported skill" in str(exc)
    else:
        raise AssertionError("expected unsupported skill validation error")


def test_parse_response_supports_missing_required_skills():
    plan = parse_planner_response(
        """
        {
          "intent": "pick_only",
          "required_missing_skills": ["locate_object_precisely"],
          "planner_reason": "需要先定位香蕉的抓取位姿",
          "skill_sequence": [],
          "confidence": "medium",
          "scene_summary": "看到了香蕉但缺少精确抓取定位"
        }
        """,
        allowed_skills=["pick_named_target", "place_named_pose"],
        default_target_name="demo_object",
        default_place_name="tray_right",
        default_relative_motion_step_m=0.03,
    )
    assert plan.required_missing_skills == ["locate_object_precisely"]
    assert plan.skill_sequence == []
    assert plan.planner_reason == "需要先定位香蕉的抓取位姿"
    assert plan.confidence == 0.6
