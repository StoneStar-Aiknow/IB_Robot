from vlm_task_planner.planner_fallback import fallback_plan_from_text


def test_fallback_plan_reuses_rule_parser():
    plan = fallback_plan_from_text(
        "夹爪往前一点",
        default_target_name="demo_object",
        default_place_name="tray_right",
        default_relative_motion_step_m=0.03,
    )
    assert plan.planner_source == "rule_fallback"
    assert plan.skill_sequence == ["move_relative_ee"]
    assert plan.motion_direction == "forward"
    assert plan.motion_distance == 0.03
