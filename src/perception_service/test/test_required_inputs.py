import rclpy
from rclpy.parameter import Parameter

from perception_service.perception_service_node import PerceptionServiceNode

_MALFORMED_CONTEXT = {"required_inputs": [{"bad": "dict"}, ["bad-list"], 123]}


def _make_node(debug: bool) -> PerceptionServiceNode:
    return PerceptionServiceNode(
        parameter_overrides=[
            Parameter("primary_camera_topic", Parameter.Type.STRING, ""),
            Parameter("wrist_camera_topic", Parameter.Type.STRING, ""),
            Parameter("ee_pose_topic", Parameter.Type.STRING, "/unused/ee_pose"),
            Parameter("joint_state_topic", Parameter.Type.STRING, "/unused/joint_states"),
            Parameter("debug_tracing", Parameter.Type.BOOL, debug),
        ]
    )


def test_malformed_required_inputs_do_not_depend_on_debug_tracing():
    """Malformed list items are ignored without changing behavior under debug."""
    rclpy.init()
    try:
        debug_off = _make_node(debug=False)
        debug_on = _make_node(debug=True)
        try:
            assert debug_off._resolve_required_inputs(_MALFORMED_CONTEXT) is None  # noqa: SLF001
            assert debug_on._resolve_required_inputs(_MALFORMED_CONTEXT) is None  # noqa: SLF001
        finally:
            debug_off.destroy_node()
            debug_on.destroy_node()
    finally:
        rclpy.shutdown()


def test_malformed_required_inputs_with_known_key_ignore_bad_items_under_debug():
    rclpy.init()
    try:
        node = _make_node(debug=True)
        try:
            result = node._resolve_required_inputs(  # noqa: SLF001
                {"required_inputs": ["primary_image", {"bad": "dict"}, ["bad-list"], 123]}
            )
            assert result == {"primary_image"}
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
