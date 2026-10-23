from __future__ import annotations

import os
import threading
import time
import uuid

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST
from ibrobot_msgs.msg import SkillRegistryEvent
from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot, ValidatePrimitive, ValidateSkill
from safety_guard.safety_guard_node import SafetyGuardNode
from skill_catalog.models import SkillRobotContext, SkillSnapshot


def _snapshot(name: str) -> SkillSnapshot:
    robot = SkillRobotContext(
        robot_name="test_robot",
        context_schema_version=1,
        robot_config_digest="robot-digest",
        named_poses={},
        named_targets={},
        arm_joint_names=("1",),
        joint_limits={},
        workspace_limits={},
        required_control_mode="moveit_planning",
        timeout_policy={"default_skill_timeout_sec": 30.0, "task_budget_sec": 180.0},
        relative_motion_reference_frame="base",
        relative_motion_step_m=0.03,
        relative_motion_direction_mapping={},
        gripper_open_position=1.0,
        gripper_closed_position=0.0,
        execution_endpoints={
            "skill_action": "/skill",
            "primitive_action": "/primitive",
            "validate_skill_service": "/validate",
            "validate_primitive_service": "/validate-primitive",
            "gateway_status_service": "/status",
            "begin_workflow_service": "/begin",
            "finalize_workflow_service": "/finalize",
            "task_executor_action": "/task",
            "arm_trajectory_action": "/trajectory",
            "move_configuration_service": "/move",
        },
    )
    template = {
        "capability": {
            "name": name,
            "summary": "Gripper skill.",
            "domain": "manipulation",
            "semantic_level": "skill",
            "planner_visible": True,
            "moves_robot": True,
            "required_control_mode": "moveit_planning",
            "parameters": {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
            "recovery_policy": "never_retry",
        },
        "primitive_sequence": [{"primitive_name": "open_gripper"}],
    }
    return SkillSnapshot(
        robot_name="test_robot",
        profile_name="test",
        primitive_contract_digest=PRIMITIVE_CONTRACT_DIGEST,
        robot_context=robot,
        delegated_executors={},
        templates={name: template},
        semantic_levels={name: "skill"},
        aliases={name: ()},
        parameter_schemas={name: template["capability"]["parameters"]},
        requirements={name: frozenset()},
        provenance={"schema_version": 1, "source_release_digest": f"source-{name}"},
        enabled_skill_names=(name,),
        planner_visible_skill_names=(name,),
        capability_view={name: template["capability"]},
    )


def _wait_for(predicate, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def _future_result(future, timeout_sec: float = 3.0):
    _wait_for(future.done, timeout_sec)
    return future.result()


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    expected = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert expected.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == expected
    rclpy.init()
    try:
        yield
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_safety_guard_asynchronously_switches_and_retains_verified_snapshots():
    suffix = f"safety_{uuid.uuid4().hex}"
    names = {
        "event": f"/{suffix}/event",
        "snapshot": f"/{suffix}/snapshot",
        "status": f"/{suffix}/status",
        "validate": f"/{suffix}/validate",
        "validate_primitive": f"/{suffix}/validate_primitive",
    }
    snapshots = {1: _snapshot("open_gripper_skill"), 2: _snapshot("close_gripper_skill")}
    state = {"generation": 1, "retained": [1]}
    gateway = rclpy.create_node(f"gateway_{suffix}")
    client_node = rclpy.create_node(f"client_{suffix}")

    def status_callback(_request, response):
        snapshot = snapshots[state["generation"]]
        response.schema_version = 1
        response.registry_epoch = "epoch-1"
        response.registry_generation = state["generation"]
        response.registry_digest = snapshot.registry_digest
        response.retained_generations = state["retained"]
        return response

    def snapshot_callback(request, response):
        snapshot = snapshots[int(request.generation)]
        response.success = True
        response.registry_epoch = "epoch-1"
        response.generation = request.generation
        response.registry_digest = snapshot.registry_digest
        response.capability_digest = snapshot.capability_digest
        response.provenance_digest = snapshot.provenance_digest
        response.snapshot_json = snapshot.snapshot_json
        return response

    gateway.create_service(GetSkillGatewayStatus, names["status"], status_callback)
    gateway.create_service(GetSkillSnapshot, names["snapshot"], snapshot_callback)
    event_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    event_publisher = gateway.create_publisher(SkillRegistryEvent, names["event"], event_qos)
    safety = SafetyGuardNode(
        parameter_overrides=[
            Parameter(name, value=value)
            for name, value in {
                "validate_skill_service": names["validate"],
                "validate_primitive_service": names["validate_primitive"],
                "skill_gateway_status_service": names["status"],
                "skill_catalog_snapshot_service": names["snapshot"],
                "skill_registry_event_topic": names["event"],
                "snapshot_sync_period_sec": 0.05,
            }.items()
        ]
    )
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (gateway, client_node, safety):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    validate_client = client_node.create_client(ValidateSkill, names["validate"])
    validate_primitive_client = client_node.create_client(ValidatePrimitive, names["validate_primitive"])
    try:
        _wait_for(lambda: safety._snapshot_cache.current_identity is not None)
        assert safety._snapshot_cache.current_identity.generation == 1

        state["generation"] = 2
        state["retained"] = [1, 2]
        event = SkillRegistryEvent()
        event.schema_version = 1
        event.registry_epoch = "epoch-1"
        event.old_generation = 1
        event.new_generation = 2
        event.registry_digest = snapshots[2].registry_digest
        event.capability_digest = snapshots[2].capability_digest
        event.provenance_digest = snapshots[2].provenance_digest
        event_publisher.publish(event)
        _wait_for(lambda: safety._snapshot_cache.current_identity.generation == 2)

        stale_event = SkillRegistryEvent()
        stale_event.schema_version = 1
        stale_event.registry_epoch = "epoch-1"
        stale_event.old_generation = 0
        stale_event.new_generation = 1
        stale_event.registry_digest = snapshots[1].registry_digest
        event_publisher.publish(stale_event)
        time.sleep(0.1)
        assert safety._snapshot_cache.current_identity.generation == 2

        request = ValidateSkill.Request()
        request.schema_version = 1
        request.dispatch_binding.schema_version = 1
        request.dispatch_binding.expected_registry_epoch = "epoch-1"
        request.dispatch_binding.expected_registry_generation = 1
        request.dispatch_binding.expected_registry_digest = snapshots[1].registry_digest
        request.skill_name = "open_gripper_skill"
        old_generation = _future_result(validate_client.call_async(request))
        assert old_generation.allowed is True
        assert old_generation.actual_registry_generation == 1

        request.dispatch_binding.expected_registry_generation = 2
        request.dispatch_binding.expected_registry_digest = "wrong"
        wrong_digest = _future_result(validate_client.call_async(request))
        assert wrong_digest.allowed is False
        assert wrong_digest.error_code == "SKILL_REGISTRY_VERSION_MISMATCH"
        assert wrong_digest.actual_registry_generation == 2

        primitive_request = ValidatePrimitive.Request()
        missing_identity = _future_result(validate_primitive_client.call_async(primitive_request))
        assert missing_identity.allowed is False
        assert missing_identity.error_code == "SKILL_SCHEMA_INVALID"

        primitive_request.dispatch_binding.schema_version = 1
        primitive_request.schema_version = 1
        primitive_request.dispatch_binding.task_id = "task-1"
        primitive_request.dispatch_binding.root_task_id = "task-1"
        primitive_request.dispatch_binding.dispatch_nonce = "nonce-1"
        primitive_request.dispatch_binding.expected_registry_epoch = "epoch-1"
        primitive_request.dispatch_binding.expected_registry_generation = 1
        primitive_request.dispatch_binding.expected_registry_digest = snapshots[1].registry_digest
        primitive_request.primitive_name = "open_gripper"
        exact_primitive = _future_result(validate_primitive_client.call_async(primitive_request))
        assert exact_primitive.allowed is True
        assert exact_primitive.actual_registry_generation == 1
    finally:
        executor.shutdown(timeout_sec=1.0)
        spin_thread.join(timeout=1.0)
        for node in (safety, client_node, gateway):
            node.destroy_node()
