import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from unique_identifier_msgs.msg import UUID

from embodied_common.dispatch_binding import copy_binding, new_binding, workflow_step
from embodied_common.skill_request import derive_skill_task_id
from embodied_common.workflow_contracts import compute_workflow_digest
from ibrobot_msgs.action import ExecuteTaskPlan, PrimitiveCommand, SkillCommand
from ibrobot_msgs.srv import (
    BeginWorkflowExecution,
    FinalizeWorkflowExecution,
    GetSkillGatewayStatus,
    GetSkillSnapshot,
    ReloadSkillCatalog,
    ValidatePrimitive,
    ValidateSkill,
)
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from skill_library.gateway_policy import GatewayPolicy, RuntimeSnapshot, SkillRequirements
from skill_library.skill_executor_node import SkillExecutorNode

_TIMEOUT_POLICY_FIELDS = (
    "default_skill_timeout_sec",
    "task_budget_sec",
    "robot_state_freshness_sec",
    "scene_freshness_sec",
    "model_idle_timeout_sec",
    "rpc_timeout_sec",
    "gripper_settle_sec",
)


def _expected_timeout_policy(overrides):
    normalized_config = {
        "name": overrides["robot_name"],
        "embodied": {
            "named_poses": json.loads(overrides["named_poses_json"]),
            "named_targets": json.loads(overrides.get("named_targets_json", "{}")),
            "timeouts": {field: overrides[field] for field in _TIMEOUT_POLICY_FIELDS},
        },
    }
    return resolve_embodied_timeout_policy(normalized_config["embodied"])


_TEST_CONTROL_MODE = "moveit_planning"


def _constructor_overrides(suffix: str, catalog_root, profile_name: str):
    return {
        "active_control_mode": _TEST_CONTROL_MODE,
        "cmd_pose_topic": f"/{suffix}/cmd_pose",
        "default_skill_timeout_sec": 2.5,
        "ee_pose_topic": f"/{suffix}/ee_pose",
        "gripper_settle_sec": 0.4,
        "joint_state_topic": f"/{suffix}/joint_state",
        "model_idle_timeout_sec": 8.0,
        "named_poses_json": json.dumps({"home": {}}),
        "named_targets_json": "{}",
        "primitive_action_name": f"/{suffix}/primitive",
        "robot_name": profile_name,
        "robot_state_freshness_sec": 0.1,
        "rpc_timeout_sec": 0.2,
        "scene_freshness_sec": 0.3,
        "skill_action_name": f"/{suffix}/skill",
        "skill_catalog_profile": profile_name,
        "skill_catalog_source_mode": "development",
        "skill_catalog_source_root": str(catalog_root),
        "skill_gateway_status_service": f"/{suffix}/status",
        "skill_required_control_mode": _TEST_CONTROL_MODE,
        "task_budget_sec": 10.0,
    }


def _write_yaml(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _skill_manifest(name, summary, *, parameters, recovery_policy, anchor_pose, profile_name, intensity="moderate"):
    return {
        "schema_version": 1,
        "name": name,
        "version": "1.0.0",
        "semantic_level": "atomic_operator",
        "description": {
            "summary": summary,
            "category": "manipulation",
            "when_to_use": [summary],
            "motion_scope": ["arm"],
            "intensity": intensity,
            "anchor_pose": anchor_pose,
        },
        "capability": {
            "schema_version": 1,
            "summary": summary,
            "domain": "manipulation",
            "moves_robot": True,
            "required_control_mode": _TEST_CONTROL_MODE,
            "parameters": parameters,
            "recovery_policy": recovery_policy,
        },
        "implementations": {profile_name: f"implementations/{profile_name}.yaml"},
    }


def _skill_implementation(profile_name, primitive_sequence):
    return {
        "schema_version": 1,
        "kind": "primitive_sequence",
        "robot": profile_name,
        "initial_gripper_state": "none",
        "timeout_sec": 9.0,
        "primitive_sequence": primitive_sequence,
    }


def _write_test_catalog(root, profile_name, *, include_relative=True, bad_primitive=False) -> None:
    """Stage a minimal DevelopmentStagingSkillSource catalog (move [+ relative] [+ bad])."""
    skills_dir = root / "config" / "skills"
    profiles_dir = root / "config" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    enabled_skills = [{"name": "move", "implementation": profile_name, "planner_visible": True}]
    if include_relative:
        enabled_skills.append({"name": "relative", "implementation": profile_name, "planner_visible": True})
    if bad_primitive:
        enabled_skills.append({"name": "bad", "implementation": profile_name, "planner_visible": True})

    _write_yaml(
        profiles_dir / f"{profile_name}.yaml",
        {
            "schema_version": 1,
            "name": profile_name,
            "robot_name": profile_name,
            "enabled_skills": enabled_skills,
        },
    )

    _write_yaml(
        skills_dir / "move" / "manifest.yaml",
        _skill_manifest(
            "move",
            "Move to a named pose.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            recovery_policy="never_retry",
            anchor_pose="home",
            profile_name=profile_name,
        ),
    )
    _write_yaml(
        skills_dir / "move" / "implementations" / f"{profile_name}.yaml",
        _skill_implementation(profile_name, [{"primitive_name": "move_to_named_pose", "pose_name": "home"}]),
    )

    if include_relative:
        _write_yaml(
            skills_dir / "relative" / "manifest.yaml",
            _skill_manifest(
                "relative",
                "Move the end effector by a requested offset.",
                parameters={
                    "type": "object",
                    "properties": {
                        "motion_direction": {"type": "string", "enum": ["forward"]},
                        "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
                    },
                    "required": ["motion_direction", "motion_distance"],
                    "additionalProperties": False,
                },
                recovery_policy="ask_user",
                anchor_pose="none",
                profile_name=profile_name,
                intensity="subtle",
            ),
        )
        _write_yaml(
            skills_dir / "relative" / "implementations" / f"{profile_name}.yaml",
            _skill_implementation(
                profile_name,
                [
                    {
                        "primitive_name": "move_relative_ee",
                        "motion_direction_from_request": True,
                        "motion_distance_from_request": True,
                    }
                ],
            ),
        )

    if bad_primitive:
        _write_yaml(
            skills_dir / "bad" / "manifest.yaml",
            _skill_manifest(
                "bad",
                "Exercise the unknown primitive rejection path.",
                parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                recovery_policy="never_retry",
                anchor_pose="none",
                profile_name=profile_name,
            ),
        )
        _write_yaml(
            skills_dir / "bad" / "implementations" / f"{profile_name}.yaml",
            _skill_implementation(profile_name, [{"primitive_name": "definitely_not_a_real_primitive"}]),
        )


def _assert_test_ros_environment() -> None:
    expected_domain = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert expected_domain is not None
    assert expected_domain.isdigit()
    assert 1 <= int(expected_domain) <= 232
    assert os.environ.get("ROS_DOMAIN_ID") == expected_domain
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def _wait_for(predicate, timeout_sec: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for ROS test condition")


def _future_result(future, timeout_sec: float = 2.0):
    _wait_for(future.done, timeout_sec)
    return future.result()


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    _assert_test_ros_environment()
    rclpy.init()
    try:
        yield
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture
def gateway_rig(request, tmp_path):
    _assert_test_ros_environment()
    suffix = f"gateway_{os.getpid()}_{uuid.uuid4().hex}"
    profile_name = f"robot_{suffix}"
    names = {
        "arm": f"/{suffix}/arm_trajectory",
        "begin_workflow": f"/{suffix}/begin_workflow",
        "cmd_pose": f"/{suffix}/cmd_pose",
        "ee_pose": f"/{suffix}/ee_pose",
        "finalize_workflow": f"/{suffix}/finalize_workflow",
        "joint_state": f"/{suffix}/joint_state",
        "primitive": f"/{suffix}/primitive",
        "reload": f"/{suffix}/reload",
        "skill": f"/{suffix}/skill",
        "snapshot": f"/{suffix}/snapshot",
        "status": f"/{suffix}/status",
        "task_executor": f"/{suffix}/task_executor",
        "validate_primitive": f"/{suffix}/validate_primitive",
        "validate_skill": f"/{suffix}/validate_skill",
    }
    test_config = dict(getattr(request, "param", {}))
    hold_task_executor = test_config.pop("_hold_task_executor", False)
    hold_validate_primitive = test_config.pop("_hold_validate_primitive", False)
    serve_task_executor = hold_task_executor or test_config.pop("_task_executor", False)
    mock_node = rclpy.create_node(f"mock_{suffix}")
    client_node = rclpy.create_node(f"client_{suffix}")

    def validate_skill(request, response):
        response.allowed = True
        response.reason = ""
        response.actual_registry_epoch = request.dispatch_binding.expected_registry_epoch
        response.actual_registry_generation = request.dispatch_binding.expected_registry_generation
        response.actual_registry_digest = request.dispatch_binding.expected_registry_digest
        return response

    validate_primitive_calls = []
    validate_primitive_started = threading.Event()
    validate_primitive_release = threading.Event()

    def validate_primitive(_request, response):
        validate_primitive_calls.append(True)
        if hold_validate_primitive:
            validate_primitive_started.set()
            assert validate_primitive_release.wait(timeout=1.0)
        response.allowed = True
        response.reason = ""
        response.actual_registry_epoch = _request.dispatch_binding.expected_registry_epoch
        response.actual_registry_generation = _request.dispatch_binding.expected_registry_generation
        response.actual_registry_digest = _request.dispatch_binding.expected_registry_digest
        return response

    mock_node.create_service(ValidateSkill, names["validate_skill"], validate_skill)
    mock_node.create_service(ValidatePrimitive, names["validate_primitive"], validate_primitive)
    _write_test_catalog(tmp_path, profile_name)
    overrides = {
        "active_control_mode": _TEST_CONTROL_MODE,
        "arm_trajectory_action_name": names["arm"],
        "begin_workflow_service": names["begin_workflow"],
        "cmd_pose_topic": names["cmd_pose"],
        "default_skill_timeout_sec": 2.5,
        "ee_pose_topic": names["ee_pose"],
        "finalize_workflow_service": names["finalize_workflow"],
        "joint_state_topic": names["joint_state"],
        "ledger_terminal_capacity": 8,
        "gripper_settle_sec": 0.4,
        "model_idle_timeout_sec": 8.0,
        "motion_authorized": True,
        "named_poses_json": json.dumps({"home": {}}),
        "primitive_action_name": names["primitive"],
        "robot_name": profile_name,
        "robot_state_freshness_sec": 0.1,
        "rpc_timeout_sec": 0.2,
        "scene_freshness_sec": 0.3,
        "skill_action_name": names["skill"],
        "skill_catalog_profile": profile_name,
        "skill_catalog_reload_service": names["reload"],
        "skill_catalog_source_mode": "development",
        "skill_catalog_source_root": str(tmp_path),
        "skill_catalog_snapshot_service": names["snapshot"],
        "skill_gateway_status_service": names["status"],
        "skill_required_control_mode": _TEST_CONTROL_MODE,
        "task_budget_sec": 10.0,
        "task_executor_action_name": names["task_executor"],
        "validate_primitive_service": names["validate_primitive"],
        "validate_skill_service": names["validate_skill"],
    }
    overrides.update(test_config)
    _assert_test_ros_environment()
    executor_node = SkillExecutorNode(
        parameter_overrides=[Parameter(name, value=value) for name, value in overrides.items()],
        node_name=f"executor_{suffix}",
    )
    executor = MultiThreadedExecutor(num_threads=6)
    for node in (mock_node, client_node, executor_node):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    skill_client = ActionClient(client_node, SkillCommand, names["skill"])
    primitive_client = ActionClient(client_node, PrimitiveCommand, names["primitive"])
    status_client = client_node.create_client(GetSkillGatewayStatus, names["status"])
    snapshot_client = client_node.create_client(GetSkillSnapshot, names["snapshot"])
    reload_client = client_node.create_client(ReloadSkillCatalog, names["reload"])
    begin_workflow_client = client_node.create_client(BeginWorkflowExecution, names["begin_workflow"])
    finalize_workflow_client = client_node.create_client(FinalizeWorkflowExecution, names["finalize_workflow"])
    _wait_for(skill_client.server_is_ready)
    _wait_for(primitive_client.server_is_ready)
    _wait_for(status_client.service_is_ready)
    _wait_for(snapshot_client.service_is_ready)
    _wait_for(reload_client.service_is_ready)
    _wait_for(begin_workflow_client.service_is_ready)
    _wait_for(finalize_workflow_client.service_is_ready)

    task_started = threading.Event()
    task_release = threading.Event()
    downstream_goals = []
    task_ids = []
    task_server = None
    if serve_task_executor:
        task_calls = []

        def execute_task(goal_handle):
            task_calls.append(True)
            downstream_goals.append(goal_handle.request)
            task_ids.append(goal_handle.request.task_id)
            if hold_task_executor and len(task_calls) == 1:
                task_started.set()
                assert task_release.wait(timeout=2.0)
            result = ExecuteTaskPlan.Result()
            result.success = True
            result.message = ""
            result.steps_completed = len(goal_handle.request.steps)
            result.total_duration_s = 0.0
            goal_handle.succeed()
            return result

        task_server = ActionServer(mock_node, ExecuteTaskPlan, names["task_executor"], execute_task)

    rig = SimpleNamespace(
        begin_workflow_client=begin_workflow_client,
        client_node=client_node,
        downstream_goals=downstream_goals,
        executor=executor,
        executor_node=executor_node,
        finalize_workflow_client=finalize_workflow_client,
        names=names,
        overrides=overrides,
        primitive_client=primitive_client,
        reload_client=reload_client,
        skill_client=skill_client,
        snapshot_client=snapshot_client,
        status_client=status_client,
        task_release=task_release,
        task_server=task_server,
        task_ids=task_ids,
        task_started=task_started,
        validate_primitive_calls=validate_primitive_calls,
        validate_primitive_release=validate_primitive_release,
        validate_primitive_started=validate_primitive_started,
    )
    try:
        yield rig
    finally:
        executor.shutdown(timeout_sec=1.0)
        spin_thread.join(timeout=1.0)
        for node in (executor_node, client_node, mock_node):
            node.destroy_node()


def _send_skill(rig, *, task_id: str = "task-1", skill_name: str = "move"):
    goal = SkillCommand.Goal()
    goal.dispatch_binding = new_binding(task_id=task_id)
    status = _get_status(rig)
    goal.dispatch_binding.expected_registry_epoch = status.registry_epoch
    goal.dispatch_binding.expected_registry_generation = status.registry_generation
    goal.dispatch_binding.expected_registry_digest = status.registry_digest
    goal.skill_name = skill_name
    goal.target_name = ""
    goal.place_name = ""
    goal.motion_direction = ""
    goal.motion_distance = 0.0
    goal.timeout_sec = 0.0
    goal_handle = _future_result(rig.skill_client.send_goal_async(goal))
    assert goal_handle.accepted
    return _future_result(goal_handle.get_result_async()).result


def _get_status(rig, *, task_id: str = "", payload_hash: str = ""):
    request = GetSkillGatewayStatus.Request()
    request.task_id = task_id
    request.payload_hash = payload_hash
    return _future_result(rig.status_client.call_async(request))


def _set_root_identity(rig, binding) -> None:
    status = _get_status(rig)
    binding.expected_registry_epoch = status.registry_epoch
    binding.expected_registry_generation = status.registry_generation
    binding.expected_registry_digest = status.registry_digest


def test_snapshot_service_response_matches_generated_contract(gateway_rig):
    request = GetSkillSnapshot.Request()
    request.schema_version = 1
    response = _future_result(gateway_rig.snapshot_client.call_async(request))
    assert response.success is True
    assert response.generation > 0
    assert response.snapshot_json


def test_reload_service_response_matches_generated_contract_and_keeps_gateway_alive(gateway_rig):
    before = _get_status(gateway_rig)
    request = ReloadSkillCatalog.Request()
    request.schema_version = 1
    request.request_id = f"reload-{uuid.uuid4().hex}"
    request.force = True

    response = _future_result(gateway_rig.reload_client.call_async(request))

    assert response.success is True
    assert response.old_generation == before.registry_generation
    assert response.generation == before.registry_generation
    assert response.changed_skills == []
    after = _get_status(gateway_rig)
    assert after.registry_epoch == before.registry_epoch
    assert after.registry_generation == before.registry_generation

    replay = _future_result(gateway_rig.reload_client.call_async(request))
    assert replay.success is True
    assert replay.generation == response.generation
    assert replay.registry_digest == response.registry_digest


def test_reload_service_activates_changed_implementation_and_increments_generation(gateway_rig):
    before = _get_status(gateway_rig)
    implementation_path = (
        Path(gateway_rig.overrides["skill_catalog_source_root"])
        / "config"
        / "skills"
        / "move"
        / "implementations"
        / f"{gateway_rig.overrides['skill_catalog_profile']}.yaml"
    )
    changed_implementation = _skill_implementation(
        gateway_rig.overrides["skill_catalog_profile"],
        [{"primitive_name": "move_to_named_pose", "pose_name": "home"}],
    )
    changed_implementation["initial_gripper_state"] = "open"
    _write_yaml(implementation_path, changed_implementation)
    request = ReloadSkillCatalog.Request()
    request.schema_version = 1
    request.request_id = f"reload-{uuid.uuid4().hex}"
    request.force = True

    response = _future_result(gateway_rig.reload_client.call_async(request))

    assert response.success is True
    assert response.old_generation == before.registry_generation
    assert response.generation == before.registry_generation + 1
    assert response.changed_skills == ["move"]
    after = _get_status(gateway_rig)
    assert after.registry_generation == response.generation
    snapshot_request = GetSkillSnapshot.Request()
    snapshot_request.schema_version = 1
    snapshot_request.registry_epoch = response.registry_epoch
    snapshot_request.generation = response.generation
    snapshot = _future_result(gateway_rig.snapshot_client.call_async(snapshot_request))
    move = next(
        item for item in json.loads(snapshot.snapshot_json)["registry_preimage"]["skills"] if item["name"] == "move"
    )
    assert move["template"]["initial_gripper_state"] == "open"


def _capability(response, name: str):
    return next(capability for capability in response.capabilities if capability.name == name)


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_typed_workflow_begin_ordered_children_and_finalize_are_idempotent(gateway_rig):
    _wait_for(gateway_rig.executor_node._task_executor_client.server_is_ready)
    status = _get_status(gateway_rig)
    root_task_id = "workflow-root"
    root_binding = new_binding(task_id=root_task_id)
    root_binding.expected_registry_epoch = status.registry_epoch
    root_binding.expected_registry_generation = status.registry_generation
    root_binding.expected_registry_digest = status.registry_digest
    started = time.time()
    deadline = started + 8.0
    root_binding.task_budget.schema_version = 1
    root_binding.task_budget.started_at.sec = int(started)
    root_binding.task_budget.started_at.nanosec = int((started - int(started)) * 1_000_000_000)
    root_binding.task_budget.deadline.sec = int(deadline)
    root_binding.task_budget.deadline.nanosec = int((deadline - int(deadline)) * 1_000_000_000)
    steps = [workflow_step(skill_name="move", timeout_sec=2.5) for _ in range(2)]
    root_binding.workflow_digest = compute_workflow_digest(
        root_task_id=root_task_id,
        task_budget=root_binding.task_budget,
        expected_registry_epoch=status.registry_epoch,
        expected_registry_generation=status.registry_generation,
        expected_registry_digest=status.registry_digest,
        workflow_steps=steps,
    )
    begin_request = BeginWorkflowExecution.Request()
    begin_request.dispatch_binding = root_binding
    begin_request.workflow_steps = steps

    began = _future_result(gateway_rig.begin_workflow_client.call_async(begin_request))
    repeated_begin = _future_result(gateway_rig.begin_workflow_client.call_async(begin_request))

    assert began.success is True
    assert len(began.root_lease_nonce) >= 32
    assert repeated_begin.success is True
    assert repeated_begin.root_lease_nonce == began.root_lease_nonce
    active_status = _get_status(gateway_rig, task_id=root_task_id, payload_hash=root_binding.workflow_digest)
    assert active_status.active_owner_kind == "workflow"
    assert active_status.active_workflow_digest == root_binding.workflow_digest
    assert active_status.active_workflow_step_index == 0

    def send_child(index: int):
        goal = SkillCommand.Goal()
        goal.dispatch_binding = copy_binding(root_binding)
        goal.dispatch_binding.task_id = derive_skill_task_id(root_task_id, index)
        goal.dispatch_binding.workflow_step_index = index
        goal.dispatch_binding.root_lease_nonce = began.root_lease_nonce
        goal.skill_name = "move"
        goal.timeout_sec = 2.5
        goal_handle = _future_result(gateway_rig.skill_client.send_goal_async(goal))
        assert goal_handle.accepted
        return _future_result(goal_handle.get_result_async()).result

    out_of_order = send_child(1)
    assert out_of_order.success is False
    assert out_of_order.error_code == "SKILL_WORKFLOW_DIGEST_MISMATCH"
    assert send_child(0).success is True
    assert send_child(1).success is True

    finalize_request = FinalizeWorkflowExecution.Request()
    finalize_request.dispatch_binding = copy_binding(root_binding)
    finalize_request.dispatch_binding.root_lease_nonce = began.root_lease_nonce
    finalize_request.terminal_state = FinalizeWorkflowExecution.Request.SUCCEEDED
    finalize_request.completed_step_count = 2
    finalized = _future_result(gateway_rig.finalize_workflow_client.call_async(finalize_request))
    repeated_finalize = _future_result(gateway_rig.finalize_workflow_client.call_async(finalize_request))

    assert finalized.success is True
    assert finalized.actual_completed_step_count == 2
    assert repeated_finalize.success is True
    assert _get_status(gateway_rig, task_id=root_task_id).request_state == "terminal"


def _begin_single_step_workflow(rig, root_task_id: str):
    status = _get_status(rig)
    binding = new_binding(task_id=root_task_id)
    binding.expected_registry_epoch = status.registry_epoch
    binding.expected_registry_generation = status.registry_generation
    binding.expected_registry_digest = status.registry_digest
    started = time.time()
    deadline = started + 8.0
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = int(started)
    binding.task_budget.started_at.nanosec = int((started - int(started)) * 1_000_000_000)
    binding.task_budget.deadline.sec = int(deadline)
    binding.task_budget.deadline.nanosec = int((deadline - int(deadline)) * 1_000_000_000)
    steps = [workflow_step(skill_name="move", timeout_sec=2.5)]
    binding.workflow_digest = compute_workflow_digest(
        root_task_id=root_task_id,
        task_budget=binding.task_budget,
        expected_registry_epoch=status.registry_epoch,
        expected_registry_generation=status.registry_generation,
        expected_registry_digest=status.registry_digest,
        workflow_steps=steps,
    )
    request = BeginWorkflowExecution.Request()
    request.dispatch_binding = binding
    request.workflow_steps = steps
    began = _future_result(rig.begin_workflow_client.call_async(request))
    assert began.success is True
    return binding, began


def _send_workflow_child(rig, binding, root_lease_nonce: str):
    goal = SkillCommand.Goal()
    goal.dispatch_binding = copy_binding(binding)
    goal.dispatch_binding.task_id = derive_skill_task_id(binding.root_task_id, 0)
    goal.dispatch_binding.workflow_step_index = 0
    goal.dispatch_binding.root_lease_nonce = root_lease_nonce
    goal.skill_name = "move"
    goal.timeout_sec = 2.5
    goal_handle = _future_result(rig.skill_client.send_goal_async(goal))
    assert goal_handle.accepted
    return _future_result(goal_handle.get_result_async()).result


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_failed_workflow_child_cannot_be_replayed(gateway_rig, monkeypatch):
    binding, began = _begin_single_step_workflow(gateway_rig, "workflow-failed-child")
    monkeypatch.setattr(gateway_rig.executor_node, "_validate_skill", lambda *_args, **_kwargs: (False, "denied"))

    failed = _send_workflow_child(gateway_rig, binding, began.root_lease_nonce)
    replay = _send_workflow_child(gateway_rig, binding, began.root_lease_nonce)

    assert failed.success is False
    assert failed.error_code == "SKILL_REJECTED"
    assert replay.success is False
    assert replay.error_code == failed.error_code

    finalize = FinalizeWorkflowExecution.Request()
    finalize.dispatch_binding = copy_binding(binding)
    finalize.dispatch_binding.root_lease_nonce = began.root_lease_nonce
    finalize.terminal_state = FinalizeWorkflowExecution.Request.FAILED
    finalize.completed_step_count = 0
    finalized = _future_result(gateway_rig.finalize_workflow_client.call_async(finalize))
    assert finalized.success is True


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_workflow_can_fail_before_child_reaches_gateway_admission(gateway_rig):
    binding, began = _begin_single_step_workflow(gateway_rig, "workflow-pre-dispatch-failure")
    finalize = FinalizeWorkflowExecution.Request()
    finalize.dispatch_binding = copy_binding(binding)
    finalize.dispatch_binding.root_lease_nonce = began.root_lease_nonce
    finalize.terminal_state = FinalizeWorkflowExecution.Request.FAILED
    finalize.completed_step_count = 0

    finalized = _future_result(gateway_rig.finalize_workflow_client.call_async(finalize))

    assert finalized.success is True
    assert finalized.actual_completed_step_count == 0


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_finalize_release_failure_keeps_terminal_workflow_busy_and_is_retryable(gateway_rig, monkeypatch):
    binding, began = _begin_single_step_workflow(gateway_rig, "workflow-release-failure")
    assert _send_workflow_child(gateway_rig, binding, began.root_lease_nonce).success is True
    workflow = gateway_rig.executor_node._active_workflow
    generation = workflow.bundle.generation
    assert generation in gateway_rig.executor_node._runtime_coordinator.retained_generations
    original_release = workflow.policy.release_workflow
    release_attempts = 0

    def release_workflow(owner, token):
        nonlocal release_attempts
        release_attempts += 1
        return False if release_attempts == 1 else original_release(owner, token)

    monkeypatch.setattr(workflow.policy, "release_workflow", release_workflow)
    request = FinalizeWorkflowExecution.Request()
    request.dispatch_binding = copy_binding(binding)
    request.dispatch_binding.root_lease_nonce = began.root_lease_nonce
    request.terminal_state = FinalizeWorkflowExecution.Request.SUCCEEDED
    request.completed_step_count = 1

    failed = _future_result(gateway_rig.finalize_workflow_client.call_async(request))
    status = _get_status(gateway_rig, task_id=binding.root_task_id)

    assert failed.success is False
    assert failed.error_code == "GATEWAY_FINALIZATION_FAILED"
    assert status.busy is True
    assert status.active_owner_kind == "workflow"
    assert status.request_state == "terminal"
    assert workflow.runtime_generation_released is True
    assert generation in gateway_rig.executor_node._runtime_coordinator.retained_generations
    retried = _future_result(gateway_rig.finalize_workflow_client.call_async(request))
    assert retried.success is True
    assert release_attempts == 2


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_expired_workflow_is_reaped_when_executor_does_not_finalize(gateway_rig):
    status = _get_status(gateway_rig)
    root_task_id = "workflow-expired"
    binding = new_binding(task_id=root_task_id)
    binding.expected_registry_epoch = status.registry_epoch
    binding.expected_registry_generation = status.registry_generation
    binding.expected_registry_digest = status.registry_digest
    started = time.time()
    deadline = started + 0.5
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = int(started)
    binding.task_budget.started_at.nanosec = int((started - int(started)) * 1_000_000_000)
    binding.task_budget.deadline.sec = int(deadline)
    binding.task_budget.deadline.nanosec = int((deadline - int(deadline)) * 1_000_000_000)
    steps = [workflow_step(skill_name="move", timeout_sec=0.5)]
    binding.workflow_digest = compute_workflow_digest(
        root_task_id=root_task_id,
        task_budget=binding.task_budget,
        expected_registry_epoch=binding.expected_registry_epoch,
        expected_registry_generation=binding.expected_registry_generation,
        expected_registry_digest=binding.expected_registry_digest,
        workflow_steps=steps,
    )
    request = BeginWorkflowExecution.Request()
    request.dispatch_binding = binding
    request.workflow_steps = steps

    began = _future_result(gateway_rig.begin_workflow_client.call_async(request))
    assert began.success is True
    _wait_for(lambda: _get_status(gateway_rig, task_id=root_task_id).request_state == "terminal", timeout_sec=2.0)
    status = _get_status(gateway_rig, task_id=root_task_id)
    assert status.active_owner_kind == ""
    assert status.request_state == "terminal"


def _status_snapshot(**overrides) -> RuntimeSnapshot:
    values = {
        "motion_authorized": True,
        "active_control_mode": "moveit_planning",
        "required_control_mode": "moveit_planning",
        "busy": False,
        "active_task_id": "",
        "validate_ready": True,
        "task_executor_ready": True,
        "arm_trajectory_ready": True,
        "ee_pose_fresh": True,
    }
    values.update(overrides)
    return RuntimeSnapshot(**values)


def _primitive_goal(task_id: str, rig=None) -> PrimitiveCommand.Goal:
    goal = PrimitiveCommand.Goal()
    goal.dispatch_binding = new_binding(task_id=task_id)
    if rig is not None:
        _set_root_identity(rig, goal.dispatch_binding)
    goal.primitive_name = "move_to_named_pose"
    goal.pose_name = "home"
    return goal


def _relative_primitive_goal(task_id: str, rig=None) -> PrimitiveCommand.Goal:
    goal = _primitive_goal(task_id, rig)
    goal.primitive_name = "move_relative_ee"
    goal.pose_name = ""
    goal.relative_dx = 0.01
    return goal


def _record_fresh_ee_pose(rig, *, x: float = 0.0) -> None:
    pose = PoseStamped()
    pose.pose.position.x = x
    pose.pose.orientation.w = 1.0
    rig.executor_node._handle_ee_pose(pose)


@pytest.mark.parametrize("gateway_rig", [{"motion_authorized": False}], indirect=True)
def test_unauthorized_skill_does_not_dispatch_primitive(gateway_rig, monkeypatch):
    dispatched_primitives = []
    monkeypatch.setattr(
        gateway_rig.executor_node._primitive_client,
        "send_goal_async",
        lambda goal: dispatched_primitives.append(goal),
    )

    result = _send_skill(gateway_rig)
    status = _get_status(gateway_rig)
    capability = _capability(status, "move")

    assert result.success is False
    assert result.error_code == "MOTION_NOT_AUTHORIZED"
    assert dispatched_primitives == []
    assert status.motion_authorized is False
    assert capability.ready is False
    assert capability.reason == "MOTION_NOT_AUTHORIZED: operator authorization is disabled"


def test_direct_skill_rejects_incomplete_registry_binding(gateway_rig):
    goal = SkillCommand.Goal()
    goal.dispatch_binding = new_binding(task_id="missing-identity")
    goal.skill_name = "move"

    handle = _future_result(gateway_rig.skill_client.send_goal_async(goal))
    result = _future_result(handle.get_result_async()).result

    assert result.success is False
    assert result.error_code == "SKILL_SCHEMA_INVALID"


def test_zero_budget_root_rejects_timeout_over_catalog_entry_cap(gateway_rig):
    goal = SkillCommand.Goal()
    goal.dispatch_binding = new_binding(task_id="full-root-budget")
    _set_root_identity(gateway_rig, goal.dispatch_binding)
    goal.skill_name = "move"
    goal.timeout_sec = gateway_rig.overrides["task_budget_sec"]

    handle = _future_result(gateway_rig.skill_client.send_goal_async(goal))
    result = _future_result(handle.get_result_async()).result

    assert result.error_code == "TIMEOUT_EXCEEDS_POLICY"


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_external_primitive_reports_exact_registry_identity(gateway_rig):
    goal = _primitive_goal("primitive-identity", gateway_rig)

    handle = _future_result(gateway_rig.primitive_client.send_goal_async(goal))
    result = _future_result(handle.get_result_async()).result
    status = _get_status(gateway_rig)

    assert result.actual_registry_epoch == status.registry_epoch
    assert result.actual_registry_generation == status.registry_generation
    assert result.actual_registry_digest == status.registry_digest


def test_status_ledger_query_matrix(gateway_rig):
    ledger = gateway_rig.executor_node._gateway_ledger
    ledger.record("active", "active-hash", "active")
    ledger.record("terminal", "terminal-hash", "terminal", error_code="MOTION_NOT_AUTHORIZED")

    empty = _get_status(gateway_rig)
    active = _get_status(gateway_rig, task_id="active")
    terminal = _get_status(gateway_rig, task_id="terminal")
    unknown = _get_status(gateway_rig, task_id="unknown")
    duplicate = _get_status(gateway_rig, task_id="active", payload_hash="active-hash")
    conflict = _get_status(gateway_rig, task_id="active", payload_hash="different-hash")
    invalid = _get_status(gateway_rig, payload_hash="hash-without-task")

    for response in (empty, active, terminal, unknown, duplicate, conflict, invalid):
        assert response.schema_version == 1
        assert response.robot_name == gateway_rig.overrides["robot_name"]
        assert response.motion_authorized is True
        assert response.active_control_mode == _TEST_CONTROL_MODE
        assert response.busy is False
        assert response.active_task_id == ""
        assert response.default_skill_timeout_sec == pytest.approx(2.5)
        assert response.task_budget_sec == pytest.approx(10.0)
        assert response.rpc_timeout_sec == pytest.approx(0.2)
        assert response.config_digest

    assert (empty.request_state, empty.request_error_code) == ("", "")
    assert (active.request_state, active.request_error_code) == ("active", "")
    assert (terminal.request_state, terminal.request_error_code) == ("terminal", "")
    assert (unknown.request_state, unknown.request_error_code) == ("", "")
    assert (duplicate.request_state, duplicate.request_error_code) == ("active", "DUPLICATE_TASK_ID")
    assert (conflict.request_state, conflict.request_error_code) == ("active", "TASK_ID_CONFLICT")
    assert (invalid.request_state, invalid.request_error_code) == ("", "INVALID_ARGUMENT")


def test_gateway_config_digest_equals_capability_digest(gateway_rig):
    status = _get_status(gateway_rig)

    assert status.config_digest == status.capability_digest
    assert status.config_digest
    assert gateway_rig.executor_node._gateway_timeout_policy == _expected_timeout_policy(gateway_rig.overrides)
    assert not gateway_rig.executor_node.has_parameter("skill_templates_json")


def test_constructor_declares_catalog_parameters_and_drops_legacy_template_parameter(tmp_path):
    _assert_test_ros_environment()
    suffix = f"params_{os.getpid()}_{uuid.uuid4().hex}"
    profile_name = f"robot_{suffix}"
    _write_test_catalog(tmp_path, profile_name)

    node = SkillExecutorNode(
        parameter_overrides=[
            Parameter(name, value=value)
            for name, value in _constructor_overrides(suffix, tmp_path, profile_name).items()
        ],
        node_name=f"executor_{suffix}",
    )
    try:
        assert not node.has_parameter("skill_templates_json")
        assert node.has_parameter("skill_catalog_source_mode")
        assert node.has_parameter("skill_catalog_source_root")
        assert node.has_parameter("skill_catalog_profile")
    finally:
        node.destroy_node()


@pytest.mark.parametrize("gateway_rig", [{"active_control_mode": "startup_override_mode"}], indirect=True)
def test_active_mode_override_is_reported_and_rejects_without_primitive_dispatch(gateway_rig, monkeypatch):
    dispatched_primitives = []
    monkeypatch.setattr(
        gateway_rig.executor_node._primitive_client,
        "send_goal_async",
        lambda goal: dispatched_primitives.append(goal),
    )

    result = _send_skill(gateway_rig)
    status = _get_status(gateway_rig)

    assert result.success is False
    assert result.error_code == "CONTROL_MODE_MISMATCH"
    assert dispatched_primitives == []
    assert status.active_control_mode == "startup_override_mode"
    assert _capability(status, "move").reason == (
        f"CONTROL_MODE_MISMATCH: requires {_TEST_CONTROL_MODE}, active mode is startup_override_mode"
    )


def test_stale_ee_and_unavailable_dependencies_keep_status_nonblocking(gateway_rig):
    started = time.monotonic()
    status = _get_status(gateway_rig)
    elapsed = time.monotonic() - started
    relative = _capability(status, "relative")

    assert elapsed < 0.5
    assert relative.ready is False
    assert relative.reason == "CAPABILITY_NOT_READY: task executor action unavailable"
    assert relative.required_control_mode == _TEST_CONTROL_MODE


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_internal_relative_primitive_rejects_pose_stale_after_root_admission(gateway_rig, monkeypatch):
    _record_fresh_ee_pose(gateway_rig)
    validation_started = threading.Event()
    release_validation = threading.Event()

    def validate_skill(*_args, **_kwargs):
        validation_started.set()
        assert release_validation.wait(timeout=1.0)
        return True, ""

    monkeypatch.setattr(gateway_rig.executor_node, "_validate_skill", validate_skill)
    goal = SkillCommand.Goal()
    goal.dispatch_binding = new_binding(task_id="stale-internal")
    _set_root_identity(gateway_rig, goal.dispatch_binding)
    goal.skill_name = "relative"
    goal.motion_direction = "forward"
    goal.motion_distance = 0.01
    goal_handle = _future_result(gateway_rig.skill_client.send_goal_async(goal))
    assert goal_handle.accepted
    result_future = goal_handle.get_result_async()

    try:
        _wait_for(validation_started.is_set)
        time.sleep(gateway_rig.overrides["robot_state_freshness_sec"] + 0.05)
        release_validation.set()
        result = _future_result(result_future).result
    finally:
        release_validation.set()

    assert result.success is False
    assert result.error_code == "CAPABILITY_NOT_READY"
    assert "ee pose unavailable or stale" in result.message
    assert gateway_rig.validate_primitive_calls == []
    assert gateway_rig.downstream_goals == []


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_external_relative_primitive_rejects_stale_pose_before_validation_or_dispatch(gateway_rig):
    _record_fresh_ee_pose(gateway_rig)
    time.sleep(gateway_rig.overrides["robot_state_freshness_sec"] + 0.05)

    goal_handle = _future_result(
        gateway_rig.primitive_client.send_goal_async(_relative_primitive_goal("stale-external", gateway_rig))
    )
    assert goal_handle.accepted
    result = _future_result(goal_handle.get_result_async()).result

    assert result.success is False
    assert result.error_code == "CAPABILITY_NOT_READY"
    assert "ee pose unavailable or stale" in result.message
    assert gateway_rig.validate_primitive_calls == []
    assert gateway_rig.downstream_goals == []


@pytest.mark.parametrize(
    "gateway_rig",
    [{"_hold_validate_primitive": True, "_task_executor": True, "rpc_timeout_sec": 1.0}],
    indirect=True,
)
def test_external_relative_primitive_rechecks_validated_snapshot_after_delayed_validation(gateway_rig):
    _record_fresh_ee_pose(gateway_rig, x=0.2)
    goal_handle = _future_result(
        gateway_rig.primitive_client.send_goal_async(_relative_primitive_goal("stale-after-validation", gateway_rig))
    )
    assert goal_handle.accepted
    result_future = goal_handle.get_result_async()

    try:
        _wait_for(gateway_rig.validate_primitive_started.is_set)
        time.sleep(gateway_rig.overrides["robot_state_freshness_sec"] + 0.05)
        _record_fresh_ee_pose(gateway_rig, x=0.4)
        gateway_rig.validate_primitive_release.set()
        result = _future_result(result_future).result
    finally:
        gateway_rig.validate_primitive_release.set()

    assert result.success is False
    assert result.error_code == "CAPABILITY_NOT_READY"
    assert result.message == "ee pose unavailable or stale"
    assert gateway_rig.validate_primitive_calls == [True]
    assert gateway_rig.downstream_goals == []


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_external_relative_primitive_rechecks_validated_snapshot_after_delayed_server_readiness(
    gateway_rig, monkeypatch
):
    _record_fresh_ee_pose(gateway_rig, x=0.2)
    readiness_started = threading.Event()
    release_readiness = threading.Event()

    def wait_for_server(**_kwargs):
        readiness_started.set()
        assert release_readiness.wait(timeout=1.0)
        return True

    monkeypatch.setattr(gateway_rig.executor_node._task_executor_client, "wait_for_server", wait_for_server)
    goal_handle = _future_result(
        gateway_rig.primitive_client.send_goal_async(_relative_primitive_goal("stale-after-readiness", gateway_rig))
    )
    assert goal_handle.accepted
    result_future = goal_handle.get_result_async()

    try:
        _wait_for(readiness_started.is_set)
        time.sleep(gateway_rig.overrides["robot_state_freshness_sec"] + 0.05)
        _record_fresh_ee_pose(gateway_rig, x=0.4)
        release_readiness.set()
        result = _future_result(result_future).result
    finally:
        release_readiness.set()

    assert result.success is False
    assert result.error_code == "CAPABILITY_NOT_READY"
    assert result.message == "ee pose unavailable or stale"
    assert gateway_rig.validate_primitive_calls == [True]
    assert gateway_rig.downstream_goals == []


@pytest.mark.parametrize("gateway_rig", [{"_hold_task_executor": True}], indirect=True)
def test_external_primitive_with_active_task_id_and_random_uuid_is_busy(gateway_rig):
    root_goal = SkillCommand.Goal()
    root_goal.dispatch_binding = new_binding(task_id="active-task")
    _set_root_identity(gateway_rig, root_goal.dispatch_binding)
    root_goal.skill_name = "move"
    root_handle = _future_result(gateway_rig.skill_client.send_goal_async(root_goal))
    assert root_handle.accepted
    root_result_future = root_handle.get_result_async()
    _wait_for(gateway_rig.task_started.is_set)

    try:
        external_goal = _primitive_goal("active-task", gateway_rig)
        external_uuid = UUID(uuid=list(uuid.uuid4().bytes))
        external_handle = _future_result(
            gateway_rig.primitive_client.send_goal_async(external_goal, goal_uuid=external_uuid)
        )
        assert external_handle.accepted
        external_result = _future_result(external_handle.get_result_async()).result
        status = _get_status(gateway_rig)

        assert external_result.error_code == "SKILL_BUSY"
        assert len(gateway_rig.validate_primitive_calls) == 1
        assert status.busy is True
        assert status.active_task_id == "active-task"
        assert _capability(status, "move").reason == "SKILL_BUSY: another root execution is active"
    finally:
        gateway_rig.task_release.set()
        _future_result(root_result_future)


@pytest.mark.parametrize("gateway_rig", [{"_hold_task_executor": True}], indirect=True)
def test_root_skill_busy_rejection_preserves_admission_reason(gateway_rig, monkeypatch):
    root_goal = SkillCommand.Goal()
    root_goal.dispatch_binding = new_binding(task_id="active-task")
    _set_root_identity(gateway_rig, root_goal.dispatch_binding)
    root_goal.skill_name = "move"
    root_handle = _future_result(gateway_rig.skill_client.send_goal_async(root_goal))
    assert root_handle.accepted
    root_result_future = root_handle.get_result_async()
    _wait_for(gateway_rig.task_started.is_set)
    dispatched_primitives = []
    monkeypatch.setattr(
        gateway_rig.executor_node._primitive_client,
        "send_goal_async",
        lambda goal: dispatched_primitives.append(goal),
    )

    try:
        result = _send_skill(gateway_rig, task_id="blocked-task")

        assert result.success is False
        assert result.error_code == "SKILL_BUSY"
        assert result.message == "another root execution is active"
        assert dispatched_primitives == []
    finally:
        gateway_rig.task_release.set()
        _future_result(root_result_future)


@pytest.mark.parametrize(
    ("requirements", "snapshot", "expected_reason"),
    [
        (
            SkillRequirements(),
            _status_snapshot(motion_authorized=False),
            "MOTION_NOT_AUTHORIZED: operator authorization is disabled",
        ),
        (
            SkillRequirements(),
            _status_snapshot(active_control_mode="teleop"),
            "CONTROL_MODE_MISMATCH: requires moveit_planning, active mode is teleop",
        ),
        (
            SkillRequirements(),
            _status_snapshot(busy=True, active_task_id="other-task"),
            "SKILL_BUSY: another root execution is active",
        ),
        (
            SkillRequirements(validate_skill=True),
            _status_snapshot(validate_ready=False),
            "CAPABILITY_NOT_READY: validate skill service unavailable",
        ),
        (
            SkillRequirements(task_executor=True),
            _status_snapshot(task_executor_ready=False),
            "CAPABILITY_NOT_READY: task executor action unavailable",
        ),
        (
            SkillRequirements(arm_trajectory=True),
            _status_snapshot(arm_trajectory_ready=False),
            "CAPABILITY_NOT_READY: arm trajectory action unavailable",
        ),
        (
            SkillRequirements(fresh_ee_pose=True),
            _status_snapshot(ee_pose_fresh=False),
            "CAPABILITY_NOT_READY: ee pose unavailable or stale",
        ),
    ],
)
def test_capability_status_preserves_gateway_readiness_reason(requirements, snapshot, expected_reason):
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = GatewayPolicy(
        {"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
        {"skill": requirements},
        parameter_schemas={
            "skill": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            }
        },
    )
    node._skill_required_control_mode = "moveit_planning"

    capability = node._capability_status("skill", snapshot)

    assert capability.ready is False
    assert capability.reason == expected_reason


@pytest.mark.parametrize("gateway_rig", [{"_task_executor": True}], indirect=True)
def test_gateway_uses_canonical_task_id_for_internal_primitive(gateway_rig):
    result = _send_skill(gateway_rig, task_id="  canonical-task  ")

    assert result.success is True
    assert gateway_rig.task_ids == ["canonical-task"]


def test_audit_logs_only_public_gateway_fields():
    messages = []
    node = object.__new__(SkillExecutorNode)
    node.get_logger = lambda: SimpleNamespace(info=messages.append)

    node._audit("requested", task_id="task-1", payload_hash="digest", skill="move")

    assert len(messages) == 1
    assert json.loads(messages[0]) == {
        "event": "requested",
        "payload_hash": "digest",
        "skill": "move",
        "task_id": "task-1",
    }


def test_cancel_request_and_propagation_audits_are_distinct_and_public_only():
    messages = []
    node = object.__new__(SkillExecutorNode)
    node._state_lock = threading.RLock()
    node._active_audit_context = {
        "payload_hash": "digest",
        "skill": "move",
        "task_id": "task-1",
    }
    node.get_logger = lambda: SimpleNamespace(info=messages.append)

    assert node._handle_cancel(object()) == CancelResponse.ACCEPT
    node._audit_cancel_propagated()
    with node._state_lock:
        node._active_audit_context = None
    assert node._handle_cancel(object()) == CancelResponse.ACCEPT

    assert [json.loads(message) for message in messages] == [
        {
            "event": "cancel_requested",
            "payload_hash": "digest",
            "skill": "move",
            "task_id": "task-1",
        },
        {
            "event": "cancel_propagated",
            "payload_hash": "digest",
            "skill": "move",
            "task_id": "task-1",
        },
        {"event": "cancel_requested"},
    ]


def test_constructor_rejects_unknown_gateway_primitive(tmp_path):
    _assert_test_ros_environment()
    suffix = f"invalid_{os.getpid()}_{uuid.uuid4().hex}"
    profile_name = f"robot_{suffix}"
    _write_test_catalog(tmp_path, profile_name, include_relative=False, bad_primitive=True)

    with pytest.raises(ValueError, match="unsupported primitive"):
        SkillExecutorNode(
            parameter_overrides=[
                Parameter(name, value=value)
                for name, value in _constructor_overrides(suffix, tmp_path, profile_name).items()
            ],
            node_name=f"executor_{suffix}",
        )
