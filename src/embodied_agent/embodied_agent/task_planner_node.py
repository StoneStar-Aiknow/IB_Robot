"""Rule planner node for the embodied minimal closure."""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from embodied_agent.base_node import BaseTaskNode
from embodied_agent.command_parser import parse_text_workflow
from embodied_agent.task_context import dump_task_context, ensure_timeout_context
from embodied_common.dispatch_binding import copy_binding, workflow_step
from embodied_common.workflow_contracts import compute_workflow_digest
from ibrobot_msgs.msg import TaskCommand, TaskStatus
from skill_catalog.ros_consumer import CatalogViewSynchronizer


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
        self.declare_parameter("skill_aliases_json", "")
        self.declare_parameter("skill_gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot")
        self.declare_parameter("skill_registry_event_topic", "/embodied/skill_registry_events")
        self.declare_parameter("snapshot_sync_period_sec", 0.5)
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
        self._catalog = CatalogViewSynchronizer(
            self,
            status_service=self.get_parameter("skill_gateway_status_service").get_parameter_value().string_value,
            snapshot_service=self.get_parameter("skill_catalog_snapshot_service").get_parameter_value().string_value,
            event_topic=self.get_parameter("skill_registry_event_topic").get_parameter_value().string_value,
            sync_period_sec=self.get_parameter("snapshot_sync_period_sec").get_parameter_value().double_value,
        )

        self._planned_publisher = self.create_publisher(TaskCommand, self._output_topic, 10)
        self._status_publisher = self.create_publisher(TaskStatus, self._status_topic, 10)
        self.create_subscription(TaskCommand, self._input_topic, self._handle_task_command, 10)

        self.get_logger().info(
            f"[embodied-debug] task_planner ready: input_topic={self._input_topic}, output_topic={self._output_topic}"
        )

    def _handle_task_command(self, msg: TaskCommand) -> None:
        catalog = self._catalog.current
        if catalog is None:
            self._publish_status(
                task_id=msg.dispatch_binding.task_id,
                state="rejected",
                success=False,
                message="catalog snapshot is not ready",
                error_code="SKILL_REGISTRY_NOT_READY",
                recoverable=True,
                replan_requested=True,
            )
            return
        parsed_steps, parse_error = parse_text_workflow(
            msg.raw_command,
            default_target_name=self._default_target,
            default_place_name=self._default_place,
            default_relative_motion_step_m=self._default_relative_motion_step,
            skill_aliases={name: list(values) for name, values in catalog.aliases.items()} or None,
        )
        if not parsed_steps:
            self._publish_status(
                task_id=msg.dispatch_binding.task_id,
                state="rejected",
                success=False,
                message=parse_error,
                error_code="UNSUPPORTED_COMMAND",
                recoverable=True,
                replan_requested=True,
            )
            self.get_logger().warning(
                f"[embodied-debug] task_planner rejected task_id={msg.dispatch_binding.task_id}: {parse_error}"
            )
            return
        skill_sequence = [step.skill_sequence[0] for step in parsed_steps]
        if any(skill_name not in catalog.planner_visible_names for skill_name in skill_sequence):
            self._publish_status(
                task_id=msg.dispatch_binding.task_id,
                state="rejected",
                success=False,
                message="planned skill is not planner-visible in the captured catalog",
                error_code="SKILL_SCHEMA_INVALID",
                recoverable=True,
                replan_requested=True,
            )
            return
        current_catalog = self._catalog.current
        if current_catalog is None or current_catalog.identity != catalog.identity:
            self._publish_status(
                task_id=msg.dispatch_binding.task_id,
                state="rejected",
                success=False,
                message="catalog changed while planning; replan against the current snapshot",
                error_code="SKILL_REGISTRY_VERSION_MISMATCH",
                recoverable=True,
                replan_requested=True,
            )
            return

        planned = TaskCommand()
        planned.dispatch_binding = copy_binding(msg.dispatch_binding)
        planned.dispatch_binding.expected_registry_epoch = catalog.identity.registry_epoch
        planned.dispatch_binding.expected_registry_generation = catalog.identity.generation
        planned.dispatch_binding.expected_registry_digest = catalog.identity.registry_digest
        planned.source = msg.source
        planned.raw_command = msg.raw_command
        planned.task_type = parsed_steps[0].task_type if len(parsed_steps) == 1 else "workflow"
        planned.target_name = parsed_steps[0].target_name if len(parsed_steps) == 1 else ""
        planned.place_name = parsed_steps[0].place_name if len(parsed_steps) == 1 else ""
        planned.motion_direction = parsed_steps[0].motion_direction if len(parsed_steps) == 1 else ""
        planned.motion_distance = float(parsed_steps[0].motion_distance) if len(parsed_steps) == 1 else 0.0
        planned.priority = msg.priority
        planned.timeout_sec = msg.timeout_sec
        planned.workflow_steps = [
            workflow_step(
                skill_name=parsed.skill_sequence[0],
                target_name=parsed.target_name,
                place_name=parsed.place_name,
                motion_direction=parsed.motion_direction,
                motion_distance=parsed.motion_distance,
                timeout_sec=float(catalog.timeout_policy.get("default_skill_timeout_sec", msg.timeout_sec)),
            )
            for parsed in parsed_steps
        ]
        if planned.dispatch_binding.task_budget.schema_version == 1:
            planned.dispatch_binding.workflow_digest = compute_workflow_digest(
                root_task_id=planned.dispatch_binding.root_task_id or planned.dispatch_binding.task_id,
                task_budget=planned.dispatch_binding.task_budget,
                expected_registry_epoch=catalog.identity.registry_epoch,
                expected_registry_generation=catalog.identity.generation,
                expected_registry_digest=catalog.identity.registry_digest,
                workflow_steps=planned.workflow_steps,
            )
        context = ensure_timeout_context(msg.context_json, msg.timeout_sec)
        planned.context_json = dump_task_context(context)
        self._planned_publisher.publish(planned)
        self._publish_status(
            task_id=msg.dispatch_binding.task_id,
            state="planned",
            success=True,
            message=f"planned skills: {skill_sequence}",
        )

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] task_planner planned "
                f"task_id={msg.dispatch_binding.task_id} task_type={planned.task_type} "
                f"target={planned.target_name or '-'} place={planned.place_name or '-'} "
                f"motion_direction={planned.motion_direction or '-'} motion_distance={planned.motion_distance:.3f} "
                f"skills={skill_sequence}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskPlannerNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
