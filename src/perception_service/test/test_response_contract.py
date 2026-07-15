"""Tests for response_contract enforcement in perception_service_node.

The check is a pure staticmethod, so these exercise it directly without spinning
up a ROS node. Keep this package-level test generic: it must not import entry
business packages such as ``embodied_agent``.
"""

from embodied_common.scene_analysis import SceneAnalysis
from perception_service.perception_service_node import PerceptionServiceNode

HOUSES = ["斯莱特林", "格兰芬多", "拉文克劳", "赫奇帕奇"]


def _context(**overrides):
    ctx = {
        "response_contract": {
            "field": "scene_summary",
            "kind": "enum",
            "allowed_values": list(HOUSES),
        }
    }
    ctx.update(overrides)
    return ctx


def _analysis(scene_summary: str) -> SceneAnalysis:
    return SceneAnalysis(
        scene_summary=scene_summary,
        visible_objects=[],
        robot_state_summary="",
        ee_pose_interpretation="",
        risks=[],
        confidence=1.0,
    )


def _check(context: dict, scene_summary: str):
    return PerceptionServiceNode._check_response_contract(context, _analysis(scene_summary))


def test_valid_house_passes():
    assert _check(_context(), "格兰芬多") is None


def test_empty_value_is_rejected():
    assert _check(_context(), "") is not None


def test_non_house_value_is_rejected():
    assert _check(_context(), "麻瓜") is not None


def test_house_with_trailing_annotation_is_rejected():
    # A well-formed JSON with a chatty scene_summary must not slip through.
    assert _check(_context(), "格兰芬多，因为他很勇敢") is not None


def test_no_contract_passes_through():
    assert _check({}, "任意内容") is None


def test_malformed_contract_is_rejected():
    assert _check({"response_contract": "not-a-dict"}, "任意内容") is not None


def test_missing_kind_is_rejected():
    ctx = _context()
    del ctx["response_contract"]["kind"]
    assert _check(ctx, "格兰芬多") is not None


def test_unknown_kind_is_rejected():
    ctx = _context()
    ctx["response_contract"]["kind"] = "regex"
    assert _check(ctx, "任意内容") is not None


def test_unknown_field_is_rejected():
    ctx = _context()
    ctx["response_contract"]["field"] = "no_such_field"
    assert _check(ctx, "格兰芬多") is not None


def test_malformed_allowed_values_is_rejected():
    ctx = _context()
    ctx["response_contract"]["allowed_values"] = []
    assert _check(ctx, "格兰芬多") is not None
