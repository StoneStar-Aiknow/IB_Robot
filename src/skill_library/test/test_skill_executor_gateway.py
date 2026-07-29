import json
import os
import threading
import time
import uuid
from contextlib import suppress
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from unique_identifier_msgs.msg import UUID

from embodied_common.capability_view import build_capability_view
from ibrobot_msgs.action import ExecuteTaskPlan, PrimitiveCommand, SkillCommand
from ibrobot_msgs.srv import GetSkillGatewayStatus, ValidatePrimitive, ValidateSkill
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


def _local_capability_view(overrides):
    raw_skill_templates = overrides.get("skill_templates_json", "")
    normalized_config = {
        "name": overrides["robot_name"],
        "embodied": {
            "named_poses": json.loads(overrides["named_poses_json"]),
            "named_targets": json.loads(overrides.get("named_targets_json", "{}")),
            "timeouts": {field: overrides[field] for field in _TIMEOUT_POLICY_FIELDS},
        },
    }
    if raw_skill_templates.strip():
        normalized_config["embodied"]["skill_templates"] = json.loads(raw_skill_templates)
    timeout_policy = resolve_embodied_timeout_policy(normalized_config["embodied"])
    return build_capability_view(normalized_config, timeout_policy=timeout_policy)


def _constructor_overrides(suffix: str):
    return {
        "cmd_pose_topic": f"/{suffix}/cmd_pose",
        "default_skill_timeout_sec": 2.5,
        "ee_pose_topic": f"/{suffix}/ee_pose",
        "gripper_settle_sec": 0.4,
        "joint_state_topic": f"/{suffix}/joint_state",
        "model_idle_timeout_sec": 8.0,
        "named_poses_json": "{}",
        "named_targets_json": "{}",
        "primitive_action_name": f"/{suffix}/primitive",
        "robot_name": f"robot_{suffix}",
        "robot_state_freshness_sec": 0.1,
        "rpc_timeout_sec": 0.2,
        "scene_freshness_sec": 0.3,
        "skill_action_name": f"/{suffix}/skill",
        "skill_gateway_status_service": f"/{suffix}/status",
        "skill_templates_json": "{}",
        "task_budget_sec": 10.0,
    }


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
def gateway_rig(request):
    _assert_test_ros_environment()
    suffix = f"gateway_{os.getpid()}_{uuid.uuid4().hex}"
    names = {
        "arm": f"/{suffix}/arm_trajectory",
        "cmd_pose": f"/{suffix}/cmd_pose",
        "ee_pose": f"/{suffix}/ee_pose",
        "joint_state": f"/{suffix}/joint_state",
        "primitive": f"/{suffix}/primitive",
        "skill": f"/{suffix}/skill",
        "status": f"/{suffix}/status",
        "task_executor": f"/{suffix}/task_executor",
        "validate_primitive": f"/{suffix}/validate_primitive",
        "validate_skill": f"/{suffix}/validate_skill",
    }
    test_config = dict(getattr(request, "param", {}))
    omit_skill_templates_json = test_config.pop("_omit_skill_templates_json", False)
    hold_task_executor = test_config.pop("_hold_task_executor", False)
    hold_validate_primitive = test_config.pop("_hold_validate_primitive", False)
    serve_task_executor = hold_task_executor or test_config.pop("_task_executor", False)
    mock_node = rclpy.create_node(f"mock_{suffix}")
    client_node = rclpy.create_node(f"client_{suffix}")

    def validate_skill(_request, response):
        response.allowed = True
        response.reason = ""
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
        return response

    mock_node.create_service(ValidateSkill, names["validate_skill"], validate_skill)
    mock_node.create_service(ValidatePrimitive, names["validate_primitive"], validate_primitive)
    templates = {
        "move": {
            "capability": {
                "schema_version": 1,
                "summary": "Move to a named pose.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "gateway_mode",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
                "recovery_policy": "never_retry",
            },
            "primitive_sequence": [
                {"primitive_name": "move_to_named_pose", "pose_name": "home"},
            ],
        },
        "relative": {
            "capability": {
                "schema_version": 1,
                "summary": "Move the end effector by a requested offset.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "gateway_mode",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "motion_direction": {"type": "string", "enum": ["forward"]},
                        "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
                    },
                    "required": ["motion_direction", "motion_distance"],
                },
                "recovery_policy": "ask_user",
            },
            "primitive_sequence": [
                {
                    "primitive_name": "move_relative_ee",
                    "motion_direction_from_request": True,
                    "motion_distance_from_request": True,
                }
            ],
        },
    }
    overrides = {
        "active_control_mode": "gateway_mode",
        "arm_trajectory_action_name": names["arm"],
        "cmd_pose_topic": names["cmd_pose"],
        "default_skill_timeout_sec": 2.5,
        "ee_pose_topic": names["ee_pose"],
        "joint_state_topic": names["joint_state"],
        "ledger_terminal_capacity": 8,
        "gripper_settle_sec": 0.4,
        "model_idle_timeout_sec": 8.0,
        "motion_authorized": True,
        "named_poses_json": json.dumps({"home": {}}),
        "primitive_action_name": names["primitive"],
        "robot_name": f"robot_{suffix}",
        "robot_state_freshness_sec": 0.1,
        "rpc_timeout_sec": 0.2,
        "scene_freshness_sec": 0.3,
        "skill_action_name": names["skill"],
        "skill_gateway_status_service": names["status"],
        "skill_required_control_mode": "gateway_mode",
        "skill_templates_json": json.dumps(templates),
        "task_budget_sec": 10.0,
        "task_executor_action_name": names["task_executor"],
        "validate_primitive_service": names["validate_primitive"],
        "validate_skill_service": names["validate_skill"],
    }
    overrides.update(test_config)
    if omit_skill_templates_json:
        del overrides["skill_templates_json"]
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
    _wait_for(skill_client.server_is_ready)
    _wait_for(primitive_client.server_is_ready)
    _wait_for(status_client.service_is_ready)

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
        client_node=client_node,
        downstream_goals=downstream_goals,
        executor=executor,
        executor_node=executor_node,
        names=names,
        overrides=overrides,
        primitive_client=primitive_client,
        skill_client=skill_client,
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
    goal.task_id = task_id
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


def _capability(response, name: str):
    return next(capability for capability in response.capabilities if capability.name == name)


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


def _primitive_goal(task_id: str) -> PrimitiveCommand.Goal:
    goal = PrimitiveCommand.Goal()
    goal.task_id = task_id
    goal.primitive_name = "move_to_named_pose"
    goal.pose_name = "home"
    return goal


def _relative_primitive_goal(task_id: str) -> PrimitiveCommand.Goal:
    goal = _primitive_goal(task_id)
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
        assert response.active_control_mode == "gateway_mode"
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


def test_gateway_config_digest_uses_full_resolved_timeout_policy(gateway_rig):
    local_view = _local_capability_view(gateway_rig.overrides)
    status = _get_status(gateway_rig)

    assert gateway_rig.executor_node._gateway_timeout_policy == local_view["timeout_policy"]
    assert status.config_digest == local_view["capability_digest"]


@pytest.mark.parametrize(
    "gateway_rig",
    [
        {"_omit_skill_templates_json": True},
        {"skill_templates_json": ""},
        {"skill_templates_json": " \t\n"},
    ],
    indirect=True,
)
def test_gateway_blank_skill_templates_construct_an_empty_catalog(gateway_rig):
    status = _get_status(gateway_rig)
    fallback_result = _send_skill(gateway_rig, skill_name="inspect_scene")

    assert gateway_rig.executor_node._skill_templates == {}
    assert gateway_rig.executor_node._skill_requirements == {}
    assert gateway_rig.executor_node._capability_view["skills"] == []
    assert list(status.capabilities) == []
    assert fallback_result.success is False
    assert fallback_result.error_code == "SKILL_REJECTED"
    assert gateway_rig.downstream_goals == []


@pytest.mark.parametrize(
    "gateway_rig",
    [
        {"skill_templates_json": "{}"},
        {
            "skill_templates_json": json.dumps(
                {
                    "disabled_skill": {
                        "disabled": True,
                        "primitive_sequence": [{"primitive_name": "open_gripper"}],
                    }
                }
            )
        },
    ],
    indirect=True,
)
def test_gateway_explicitly_empty_skill_templates_expose_no_capabilities(gateway_rig):
    status = _get_status(gateway_rig)
    local_view = _local_capability_view(gateway_rig.overrides)

    assert gateway_rig.executor_node._skill_templates == {}
    assert gateway_rig.executor_node._skill_requirements == {}
    assert list(status.capabilities) == []
    assert local_view["skills"] == []
    assert status.config_digest == local_view["capability_digest"]


def test_constructor_rejects_incomplete_or_wrong_config_digest():
    _assert_test_ros_environment()
    suffix = f"digest_{os.getpid()}_{uuid.uuid4().hex}"
    overrides = _constructor_overrides(suffix)
    local_view = _local_capability_view(overrides)
    complete = local_view["capability_digest"]
    incomplete = build_capability_view(
        {
            "name": overrides["robot_name"],
            "embodied": {
                "named_poses": {},
                "named_targets": {},
                "skill_templates": {},
            },
        },
        timeout_policy={
            "default_skill_timeout_sec": overrides["default_skill_timeout_sec"],
            "task_budget_sec": overrides["task_budget_sec"],
        },
    )["capability_digest"]

    node = None
    with suppress(ValueError):
        node = SkillExecutorNode(
            parameter_overrides=[
                Parameter(name, value=value) for name, value in {**overrides, "config_digest": complete}.items()
            ],
            node_name=f"executor_{suffix}",
        )
    try:
        assert node is not None
        assert node._config_digest == complete
    finally:
        if node is not None:
            node.destroy_node()

    for index, configured_digest in enumerate((incomplete, "wrong-digest"), start=1):
        with pytest.raises(ValueError, match="config_digest must match"):
            SkillExecutorNode(
                parameter_overrides=[
                    Parameter(name, value=value)
                    for name, value in {**overrides, "config_digest": configured_digest}.items()
                ],
                node_name=f"executor_{suffix}_{index}",
            )


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
        "CONTROL_MODE_MISMATCH: requires gateway_mode, active mode is startup_override_mode"
    )


def test_stale_ee_and_unavailable_dependencies_keep_status_nonblocking(gateway_rig):
    started = time.monotonic()
    status = _get_status(gateway_rig)
    elapsed = time.monotonic() - started
    relative = _capability(status, "relative")

    assert elapsed < 0.5
    assert relative.ready is False
    assert relative.reason == "CAPABILITY_NOT_READY: task executor action unavailable"
    assert relative.required_control_mode == "gateway_mode"


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
    goal.task_id = "stale-internal"
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
        gateway_rig.primitive_client.send_goal_async(_relative_primitive_goal("stale-external"))
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
    [{"_hold_validate_primitive": True, "_task_executor": True}],
    indirect=True,
)
def test_external_relative_primitive_rechecks_validated_snapshot_after_delayed_validation(gateway_rig):
    _record_fresh_ee_pose(gateway_rig, x=0.2)
    goal_handle = _future_result(
        gateway_rig.primitive_client.send_goal_async(_relative_primitive_goal("stale-after-validation"))
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
        gateway_rig.primitive_client.send_goal_async(_relative_primitive_goal("stale-after-readiness"))
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
    root_goal.task_id = "active-task"
    root_goal.skill_name = "move"
    root_handle = _future_result(gateway_rig.skill_client.send_goal_async(root_goal))
    assert root_handle.accepted
    root_result_future = root_handle.get_result_async()
    _wait_for(gateway_rig.task_started.is_set)

    try:
        external_goal = _primitive_goal("active-task")
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
    root_goal.task_id = "active-task"
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


def test_constructor_rejects_unknown_gateway_primitive():
    _assert_test_ros_environment()
    suffix = f"invalid_{os.getpid()}_{uuid.uuid4().hex}"

    with pytest.raises(ValueError, match="unknown primitive"):
        SkillExecutorNode(
            parameter_overrides=[
                Parameter(
                    "skill_templates_json",
                    value=json.dumps({"bad": {"primitive_sequence": [{"primitive_name": "bad"}]}}),
                ),
                Parameter("skill_action_name", value=f"/{suffix}/skill"),
                Parameter("primitive_action_name", value=f"/{suffix}/primitive"),
                Parameter("skill_gateway_status_service", value=f"/{suffix}/status"),
                Parameter("ee_pose_topic", value=f"/{suffix}/ee_pose"),
                Parameter("joint_state_topic", value=f"/{suffix}/joint_state"),
                Parameter("cmd_pose_topic", value=f"/{suffix}/cmd_pose"),
            ],
            node_name=f"executor_{suffix}",
        )
