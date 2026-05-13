"""Safety validation node for the embodied minimal closure."""

import rclpy
from rclpy.node import Node

from ibrobot_msgs.srv import ValidatePrimitive, ValidateSkill
from safety_guard.rules import (
    load_json_mapping,
    validate_primitive_request,
    validate_skill_request,
)


class SafetyGuardNode(Node):
    """Provide explicit safety validation services for skills and primitives."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("safety_guard_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("validate_primitive_service", "/embodied/validate_primitive")
        self.declare_parameter("named_poses_json", "{}")
        self.declare_parameter("named_targets_json", "{}")
        self.declare_parameter("skill_templates_json", "{}")
        self.declare_parameter("workspace_json", "{}")
        self.declare_parameter("debug_tracing", True)

        self._validate_skill_service = self.get_parameter("validate_skill_service").get_parameter_value().string_value
        self._validate_primitive_service = (
            self.get_parameter("validate_primitive_service").get_parameter_value().string_value
        )
        self._named_poses = load_json_mapping(self.get_parameter("named_poses_json").get_parameter_value().string_value)
        self._named_targets = load_json_mapping(
            self.get_parameter("named_targets_json").get_parameter_value().string_value
        )
        self._skill_templates = load_json_mapping(
            self.get_parameter("skill_templates_json").get_parameter_value().string_value
        )
        self._workspace = load_json_mapping(self.get_parameter("workspace_json").get_parameter_value().string_value)
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value

        self.create_service(ValidateSkill, self._validate_skill_service, self._handle_validate_skill)
        self.create_service(ValidatePrimitive, self._validate_primitive_service, self._handle_validate_primitive)

        self.get_logger().info(
            "[embodied-debug] safety_guard ready: "
            f"validate_skill_service={self._validate_skill_service}, "
            f"validate_primitive_service={self._validate_primitive_service}"
        )

    def _handle_validate_skill(self, request, response):
        try:
            allowed, reason = validate_skill_request(
                request.skill_name,
                request.target_name,
                request.place_name,
                request.motion_direction,
                request.motion_distance,
                self._named_poses,
                self._named_targets,
                self._skill_templates,
            )
        except Exception as exc:
            self.get_logger().error(f"[safety_guard] uncaught exception in skill validation: {exc}")
            allowed, reason = False, f"internal error: {exc}"
        response.allowed = allowed
        response.reason = reason
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] safety_guard skill_check "
                f"skill={request.skill_name} target={request.target_name or '-'} "
                f"place={request.place_name or '-'} motion_direction={request.motion_direction or '-'} "
                f"motion_distance={request.motion_distance:.3f} allowed={allowed} reason='{reason}'"
            )
        return response

    def _handle_validate_primitive(self, request, response):
        try:
            allowed, reason = validate_primitive_request(
                request.primitive_name,
                request.pose_name,
                request.relative_dx,
                request.relative_dy,
                request.relative_dz,
                request.target_x,
                request.target_y,
                request.target_z,
                request.gripper_position,
                self._named_poses,
                self._workspace,
            )
        except Exception as exc:
            self.get_logger().error(f"[safety_guard] uncaught exception in primitive validation: {exc}")
            allowed, reason = False, f"internal error: {exc}"
        response.allowed = allowed
        response.reason = reason
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] safety_guard primitive_check "
                f"primitive={request.primitive_name} pose={request.pose_name or '-'} "
                f"delta=({request.relative_dx:.3f}, {request.relative_dy:.3f}, {request.relative_dz:.3f}) "
                f"target=({request.target_x:.3f}, {request.target_y:.3f}, {request.target_z:.3f}) "
                f"gripper={request.gripper_position:.3f} allowed={allowed} reason='{reason}'"
            )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyGuardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
