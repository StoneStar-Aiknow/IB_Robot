"""Task entry node for the embodied minimal closure."""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from embodied_agent.command_parser import PlannedTask, parse_text_command
from embodied_agent.task_context import (
    TIMEOUT_CONTEXT_KEY,
    build_timeout_context,
    dump_task_context,
    load_task_context,
)
from ibrobot_msgs.msg import TaskCommand, TaskStatus


def build_direct_planned_task(
    *,
    task_id: str,
    source: str,
    raw_command: str,
    plan: PlannedTask,
    priority: int = 1,
    timeout_sec: float = 0.0,
    existing_context_json: str = "",
) -> TaskCommand:
    task = TaskCommand()
    task.task_id = task_id
    task.source = source
    task.raw_command = raw_command
    task.task_type = plan.task_type
    task.target_name = plan.target_name
    task.place_name = plan.place_name
    task.motion_direction = plan.motion_direction
    task.motion_distance = float(plan.motion_distance)
    task.priority = priority
    task.timeout_sec = timeout_sec
    context = load_task_context(existing_context_json)
    if timeout_sec > 0.0 and TIMEOUT_CONTEXT_KEY not in context:
        context[TIMEOUT_CONTEXT_KEY] = build_timeout_context(timeout_sec)
    context["skill_sequence"] = plan.skill_sequence
    task.context_json = dump_task_context(context)
    return task


class TaskEntryNode(Node):
    """Convert ASR text into TaskCommand envelopes or direct planned tasks."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("task_entry_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("input_topic", "/voice_command")
        self.declare_parameter("output_topic", "/embodied/task_command")
        self.declare_parameter("planned_output_topic", "/embodied/planned_task")
        self.declare_parameter("status_topic", "/embodied/task_status")
        self.declare_parameter("default_target_name", "demo_object")
        self.declare_parameter("default_place_name", "tray_right")
        self.declare_parameter("default_relative_motion_step_m", 0.03)
        self.declare_parameter("default_task_timeout_sec", 180.0)
        self.declare_parameter("debug_tracing", True)

        self._input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self._output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._planned_output_topic = self.get_parameter("planned_output_topic").get_parameter_value().string_value
        self._status_topic = self.get_parameter("status_topic").get_parameter_value().string_value
        self._default_target = self.get_parameter("default_target_name").get_parameter_value().string_value
        self._default_place = self.get_parameter("default_place_name").get_parameter_value().string_value
        self._default_relative_motion_step = (
            self.get_parameter("default_relative_motion_step_m").get_parameter_value().double_value
        )
        self._default_task_timeout = self.get_parameter("default_task_timeout_sec").get_parameter_value().double_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value

        self._publisher = self.create_publisher(TaskCommand, self._output_topic, 10)
        self._planned_publisher = self.create_publisher(TaskCommand, self._planned_output_topic, 10)
        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self.create_subscription(String, self._input_topic, self._handle_text_command, 10)

        self.get_logger().info(
            "[embodied-debug] task_entry ready: "
            f"input_topic={self._input_topic}, output_topic={self._output_topic}, "
            f"planned_output_topic={self._planned_output_topic}, status_topic={self._status_topic}"
        )

    def _publish_planned_status(self, task_id: str, skill_sequence: list[str]) -> None:
        status = TaskStatus()
        status.task_id = task_id
        status.state = "planned"
        status.success = True
        status.current_skill = ""
        status.completed_skills = []
        status.error_code = ""
        status.message = f"planned skills: {skill_sequence}"
        status.recoverable = False
        status.replan_requested = False
        self._status_publisher.publish(status)

    def _handle_text_command(self, msg: String) -> None:
        command = (msg.data or "").strip()
        if not command:
            return

        task_id = f"task-{time.time_ns()}"
        plan = parse_text_command(
            command,
            default_target_name=self._default_target,
            default_place_name=self._default_place,
            default_relative_motion_step_m=self._default_relative_motion_step,
        )
        if plan.skill_sequence:
            planned_task = build_direct_planned_task(
                task_id=task_id,
                source="voice_asr",
                raw_command=command,
                plan=plan,
                timeout_sec=self._default_task_timeout,
            )
            self._planned_publisher.publish(planned_task)
            self._publish_planned_status(task_id, plan.skill_sequence)
            if self._debug:
                self.get_logger().info(
                    "[embodied-debug] task_entry direct planned "
                    f"task_id={task_id} task_type={plan.task_type} "
                    f"skills={plan.skill_sequence}"
                )
            return

        task = TaskCommand()
        task.task_id = task_id
        task.source = "voice_asr"
        task.raw_command = command
        task.task_type = "unplanned"
        task.target_name = ""
        task.place_name = ""
        task.motion_direction = ""
        task.motion_distance = 0.0
        task.priority = 1
        task.timeout_sec = self._default_task_timeout
        timeout_context = build_timeout_context(self._default_task_timeout)
        task.context_json = dump_task_context({TIMEOUT_CONTEXT_KEY: timeout_context} if timeout_context else {})
        self._publisher.publish(task)

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] task_entry forwarded raw command task_id={task.task_id} text='{command}'"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskEntryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
