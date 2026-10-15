"""Task entry node for the embodied minimal closure."""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from embodied_agent.task_context import TIMEOUT_CONTEXT_KEY, build_timeout_context, dump_task_context
from embodied_common.dispatch_binding import new_binding
from ibrobot_msgs.msg import TaskCommand, TaskStatus


def _set_task_budget(binding, timeout_context: dict) -> None:
    started = float(timeout_context["created_at_unix_sec"])
    deadline = float(timeout_context["deadline_unix_sec"])
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = int(started)
    binding.task_budget.started_at.nanosec = int((started - int(started)) * 1_000_000_000)
    binding.task_budget.deadline.sec = int(deadline)
    binding.task_budget.deadline.nanosec = int((deadline - int(deadline)) * 1_000_000_000)


class TaskEntryNode(Node):
    """Convert ASR text into TaskCommand envelopes."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("task_entry_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("input_topic", "/voice_command")
        self.declare_parameter("output_topic", "/embodied/task_command")
        self.declare_parameter("status_topic", "/embodied/task_status")
        self.declare_parameter("default_task_timeout_sec", 180.0)
        self.declare_parameter("debug_tracing", False)

        self._input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self._output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._status_topic = self.get_parameter("status_topic").get_parameter_value().string_value
        self._default_task_timeout = self.get_parameter("default_task_timeout_sec").get_parameter_value().double_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value

        self._publisher = self.create_publisher(TaskCommand, self._output_topic, 10)
        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self.create_subscription(String, self._input_topic, self._handle_text_command, 10)

        self.get_logger().info(
            "[embodied-debug] task_entry ready: "
            f"input_topic={self._input_topic}, output_topic={self._output_topic}, "
            f"status_topic={self._status_topic}"
        )

    def _handle_text_command(self, msg: String) -> None:
        command = (msg.data or "").strip()
        if not command:
            return

        task_id = f"task-{time.time_ns()}"
        task = TaskCommand()
        task.dispatch_binding = new_binding(task_id=task_id)
        task.source = "voice_asr"
        task.raw_command = command
        task.task_type = "unplanned"
        task.workflow_steps = []
        task.target_name = ""
        task.container_name = ""
        task.place_name = ""
        task.motion_direction = ""
        task.motion_distance = 0.0
        task.priority = 1
        task.timeout_sec = self._default_task_timeout
        timeout_context = build_timeout_context(self._default_task_timeout)
        if timeout_context:
            _set_task_budget(task.dispatch_binding, timeout_context)
        task.context_json = dump_task_context({TIMEOUT_CONTEXT_KEY: timeout_context} if timeout_context else {})
        self._publisher.publish(task)

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] task_entry forwarded raw command task_id={task.dispatch_binding.task_id} text='{command}'"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskEntryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
