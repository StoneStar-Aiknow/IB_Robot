"""Rule planner node for the embodied minimal closure."""

import rclpy

from embodied_agent.base_node import BaseTaskNode
from embodied_agent.command_parser import parse_text_command
from embodied_agent.task_context import dump_task_context, ensure_timeout_context
from ibrobot_msgs.msg import TaskCommand, TaskStatus


class TaskPlannerNode(BaseTaskNode):
    """Plan a deterministic skill sequence from a TaskCommand."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("task_planner_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("input_topic", "/embodied/task_command")
        self.declare_parameter("output_topic", "/embodied/planned_task")
        self.declare_parameter("status_topic", "/embodied/task_status")
        self.declare_parameter("default_target_name", "demo_object")
        self.declare_parameter("default_place_name", "tray_right")
        self.declare_parameter("default_relative_motion_step_m", 0.03)
        self.declare_parameter("debug_tracing", False)

        self._input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self._output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._status_topic = self.get_parameter("status_topic").get_parameter_value().string_value
        self._default_target = self.get_parameter("default_target_name").get_parameter_value().string_value
        self._default_place = self.get_parameter("default_place_name").get_parameter_value().string_value
        self._default_relative_motion_step = (
            self.get_parameter("default_relative_motion_step_m").get_parameter_value().double_value
        )
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value

        self._planned_publisher = self.create_publisher(TaskCommand, self._output_topic, 10)
        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self.create_subscription(TaskCommand, self._input_topic, self._handle_task_command, 10)

        self.get_logger().info(
            f"[embodied-debug] task_planner ready: input_topic={self._input_topic}, output_topic={self._output_topic}"
        )

    def _handle_task_command(self, msg: TaskCommand) -> None:
        plan = parse_text_command(
            msg.raw_command,
            default_target_name=self._default_target,
            default_place_name=self._default_place,
            default_relative_motion_step_m=self._default_relative_motion_step,
        )
        if not plan.skill_sequence:
            self._publish_status(
                task_id=msg.task_id,
                state="rejected",
                success=False,
                message=plan.message,
                error_code="UNSUPPORTED_COMMAND",
                recoverable=True,
                replan_requested=True,
            )
            self.get_logger().warning(f"[embodied-debug] task_planner rejected task_id={msg.task_id}: {plan.message}")
            return

        planned = TaskCommand()
        planned.task_id = msg.task_id
        planned.source = msg.source
        planned.raw_command = msg.raw_command
        planned.task_type = plan.task_type
        planned.target_name = plan.target_name
        planned.place_name = plan.place_name
        planned.motion_direction = plan.motion_direction
        planned.motion_distance = float(plan.motion_distance)
        planned.priority = msg.priority
        planned.timeout_sec = msg.timeout_sec
        context = ensure_timeout_context(msg.context_json, msg.timeout_sec)
        context["skill_sequence"] = plan.skill_sequence
        planned.context_json = dump_task_context(context)
        self._planned_publisher.publish(planned)
        self._publish_status(
            task_id=msg.task_id,
            state="planned",
            success=True,
            message=f"planned skills: {plan.skill_sequence}",
        )

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] task_planner planned "
                f"task_id={msg.task_id} task_type={plan.task_type} "
                f"target={plan.target_name or '-'} place={plan.place_name or '-'} "
                f"motion_direction={plan.motion_direction or '-'} motion_distance={plan.motion_distance:.3f} "
                f"skills={plan.skill_sequence}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
