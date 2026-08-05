"""ROS lifecycle tests for the Agent plan boundary."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from embodied_agent.agent_plan_node import AgentPlanNode
from ibrobot_msgs.action import ExecuteAgentPlan, SkillCommand
from ibrobot_msgs.msg import SkillCapabilityStatus
from ibrobot_msgs.srv import (
    ConfirmAgentPlan,
    GetSkillGatewayStatus,
    GetSkillSnapshot,
    PlanAgentCommand,
    ValidateAgentPlan,
    ValidateSkill,
)


def _future_result(future, timeout_sec: float = 3.0):
    deadline = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.done()
    return future.result()


@pytest.fixture
def plan_rig():
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init()
    suffix = uuid.uuid4().hex
    names = SimpleNamespace(
        action=f"/agent_{suffix}/skill",
        confirm=f"/agent_{suffix}/confirm",
        plan=f"/agent_{suffix}/plan",
        status=f"/agent_{suffix}/status",
        snapshot=f"/agent_{suffix}/snapshot",
        validate=f"/agent_{suffix}/validate",
        validate_skill=f"/agent_{suffix}/validate_skill",
    )
    mock_node = rclpy.create_node(f"agent_mock_{suffix}")
    client_node = rclpy.create_node(f"agent_client_{suffix}")
    from robot_config.loader import load_robot_config_dict
    from robot_skill_cli.catalog import compile_local_snapshot

    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"
    snapshot = compile_local_snapshot(load_robot_config_dict(config_path), config_path)
    registry = ("epoch-test", 7, snapshot.registry_digest)

    def status_callback(_request, response):
        response.schema_version = 1
        response.registry_epoch, response.registry_generation, response.registry_digest = registry
        response.default_skill_timeout_sec = 2.0
        response.task_budget_sec = 10.0
        response.rpc_timeout_sec = 1.0
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        capability = SkillCapabilityStatus()
        capability.schema_version = 1
        capability.name = "open_gripper_skill"
        capability.semantic_level = "skill"
        capability.planner_visible = True
        capability.ready = True
        response.capabilities = [capability]
        return response

    def validate_callback(_request, response):
        response.allowed = True
        response.actual_registry_epoch, response.actual_registry_generation, response.actual_registry_digest = registry
        return response

    def snapshot_callback(_request, response):
        response.success = True
        response.registry_epoch, response.generation, response.registry_digest = registry
        response.capability_digest = snapshot.capability_digest
        response.provenance_digest = snapshot.provenance_digest
        response.source_release_digest = str(snapshot.provenance["source_release_digest"])
        response.profile_name = snapshot.profile_name
        response.snapshot_json = snapshot.snapshot_json
        return response

    def execute_callback(goal_handle):
        result = SkillCommand.Result()
        result.success = True
        result.message = "skill completed"
        result.actual_registry_epoch, result.actual_registry_generation, result.actual_registry_digest = registry
        result.executed_primitives = []
        goal_handle.succeed()
        return result

    mock_node.create_service(GetSkillGatewayStatus, names.status, status_callback)
    mock_node.create_service(GetSkillSnapshot, names.snapshot, snapshot_callback)
    mock_node.create_service(ValidateSkill, names.validate_skill, validate_callback)
    skill_server = ActionServer(mock_node, SkillCommand, names.action, execute_callback)
    plan_node = AgentPlanNode(
        parameter_overrides=[
            Parameter(name, value=value)
            for name, value in {
                "gateway_status_service": names.status,
                "validate_skill_service": names.validate_skill,
                "skill_catalog_snapshot_service": names.snapshot,
                "skill_action_name": names.action,
                "plan_service": names.plan,
                "validate_plan_service": names.validate,
                "confirm_plan_service": names.confirm,
                "execute_plan_action": names.action + "_agent",
                "rpc_timeout_sec": 1.0,
                "default_target_name": "demo_object",
                "default_place_name": "home",
            }.items()
        ]
    )
    executor = MultiThreadedExecutor(num_threads=6)
    for node in (mock_node, client_node, plan_node):
        executor.add_node(node)
    import threading

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    plan_client = client_node.create_client(PlanAgentCommand, names.plan)
    validate_client = client_node.create_client(ValidateAgentPlan, names.validate)
    confirm_client = client_node.create_client(ConfirmAgentPlan, names.confirm)
    execute_client = ActionClient(
        client_node,
        ExecuteAgentPlan,
        names.action + "_agent",
    )
    for client in (plan_client, validate_client, confirm_client):
        assert client.wait_for_service(timeout_sec=2.0)
    assert execute_client.wait_for_server(timeout_sec=2.0)
    try:
        yield SimpleNamespace(
            confirm_client=confirm_client,
            execute_client=execute_client,
            plan_client=plan_client,
            registry=registry,
            validate_client=validate_client,
        )
    finally:
        executor.shutdown(timeout_sec=1.0)
        thread.join(timeout=1.0)
        skill_server.destroy()
        for node in (plan_node, client_node, mock_node):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_plan_validate_confirm_execute_and_terminal_replay(plan_rig):
    plan_request = PlanAgentCommand.Request()
    plan_request.schema_version = 1
    plan_request.request_id = "request-1"
    plan_request.raw_command = "打开夹爪"
    planned = _future_result(plan_rig.plan_client.call_async(plan_request))
    assert planned.success is True
    assert planned.plan.plan_token
    assert planned.plan.workflow_steps[0].skill_name == "open_gripper_skill"

    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    validated = _future_result(plan_rig.validate_client.call_async(validate_request))
    assert validated.allowed is True

    confirm_request = ConfirmAgentPlan.Request()
    confirm_request.schema_version = 1
    confirm_request.plan_token = planned.plan.plan_token
    confirm_request.plan_digest = planned.plan.plan_digest
    confirm_request.task_id = "agent-task-1"
    confirm_request.registry_epoch, confirm_request.registry_generation, confirm_request.registry_digest = (
        plan_rig.registry
    )
    confirmed = _future_result(plan_rig.confirm_client.call_async(confirm_request))
    assert confirmed.confirmed is True

    def execute():
        goal = ExecuteAgentPlan.Goal()
        goal.schema_version = 1
        goal.plan_token = planned.plan.plan_token
        goal.confirmation_token = confirmed.confirmation_token
        goal.task_id = "agent-task-1"
        goal.timeout_sec = 5.0
        handle = _future_result(plan_rig.execute_client.send_goal_async(goal))
        assert handle.accepted
        return _future_result(handle.get_result_async()).result

    first = execute()
    replay = execute()
    assert first.success is True
    assert replay.success is True
    assert replay.completed_step_count == 1


def test_plan_service_returns_one_typed_ordered_multi_skill_workflow(plan_rig):
    request = PlanAgentCommand.Request()
    request.schema_version = 1
    request.request_id = "request-workflow"
    request.raw_command = "先点头，然后挥手"

    planned = _future_result(plan_rig.plan_client.call_async(request))

    assert planned.success is True
    assert [step.skill_name for step in planned.plan.workflow_steps] == ["nod_yes", "wave_hello"]
    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    validated = _future_result(plan_rig.validate_client.call_async(validate_request))
    assert validated.allowed is True
