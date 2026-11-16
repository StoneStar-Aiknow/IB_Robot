"""Public motion/feedback binding and provider-less legacy routing."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rclpy.qos import ReliabilityPolicy

from robot_config.interface_binding import InterfaceBindingError
from robot_runtime import contract as RUNTIME
from task_dispatch.task_executor_node import TaskExecutorNode, _load_robot_yaml


@pytest.fixture
def config():
    return {
        "runtime": {
            "provider": "unit_robot",
            "interface_description": {
                "interfaces": {
                    "motion.move_to_pose": {
                        "kind": "service",
                        "direction": "serve",
                        "message_type": "ibrobot_msgs/srv/MoveToPose",
                        "endpoint": "/unit/move",
                    },
                    "gripper.trajectory": {
                        "kind": "action",
                        "direction": "serve",
                        "message_type": "control_msgs/action/FollowJointTrajectory",
                        "endpoint": "/unit/gripper",
                    },
                    "joint.state": {
                        "kind": "topic",
                        "direction": "publish",
                        "message_type": "sensor_msgs/msg/JointState",
                        "endpoint": "/unit/joints",
                        "joint_names": ["arm", "finger"],
                        "qos": {"depth": 7, "reliability": "best_effort"},
                    },
                }
            },
        },
        "robot_model": {
            "joint_groups": {"arm": ["arm"], "gripper": ["finger"]},
            "joint_limits": {"finger": {"min": -0.2, "max": 0.8}},
        },
        "task_dispatch": {"gripper_trajectory_interface": "gripper.trajectory"},
    }


def node_for(config):
    node = object.__new__(TaskExecutorNode)
    node._robot_cfg = config
    return node


def test_public_bindings_select_endpoints_qos_and_model_limits(config):
    node = node_for(config)
    node._configure_interfaces()
    assert node._move_service_name == "/unit/move"
    assert node._gripper_action_name == "/unit/gripper"
    assert node._joint_state_topic == "/unit/joints"
    assert node._joint_state_qos.depth == 7
    assert node._joint_state_qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert node._gripper_joint == "finger"
    assert node._gripper_limits == {"min": -0.2, "max": 0.8}


@pytest.mark.parametrize(
    "field,value",
    [
        ("gripper_trajectory_interface", ""),
        ("gripper_trajectory_interface", "unknown"),
        ("gripper_trajectory_interface", "joint.state"),
        ("move_to_pose_interface", "gripper.trajectory"),
        ("joint_state_interface", "motion.move_to_pose"),
    ],
)
def test_public_binding_failure_never_uses_legacy_fallback(config, field, value):
    config["task_dispatch"][field] = value
    node = node_for(config)
    with pytest.raises(InterfaceBindingError):
        node._configure_interfaces()


@pytest.mark.parametrize("mutation", ["missing_model", "two_grippers", "missing_feedback"])
def test_public_model_and_feedback_required(config, mutation):
    if mutation == "missing_model":
        config.pop("robot_model")
    elif mutation == "two_grippers":
        config["robot_model"]["joint_groups"]["gripper"] = ["finger", "other"]
    else:
        config["runtime"]["interface_description"]["interfaces"]["joint.state"]["joint_names"] = ["arm"]
    with pytest.raises(InterfaceBindingError):
        node_for(config)._configure_interfaces()


@pytest.mark.parametrize("position", [-0.3, 0.9, float("nan"), float("inf")])
def test_gripper_limits_reject_before_noop_or_motion(config, position):
    node = node_for(config)
    node._configure_interfaces()
    node._is_redundant_gripper_open = Mock()
    node._gripper_action_client = Mock()
    success, message = node._exec_gripper(SimpleNamespace(gripper_position=position))
    assert not success and "public joint limits" in message
    node._is_redundant_gripper_open.assert_not_called()
    assert node._gripper_action_client.mock_calls == []


def test_providerless_config_uses_neutral_motion_endpoint():
    node = node_for({"joints": {"gripper": ["6"]}})
    node._configure_interfaces()
    assert node._move_service_name == RUNTIME.MOVE_TO_POSE_SERVICE
    assert node._joint_state_topic == "/joint_states"
    assert node._gripper_joint == "6"
    assert node._gripper_limits is None


def test_loader_requires_public_model_and_preserves_resolved_config_path(monkeypatch, config):
    original = deepcopy(config)
    monkeypatch.setattr(
        "task_dispatch.task_executor_node.load_robot_section", lambda _: ("/resolved/robot.yaml", config)
    )
    resolver = Mock(return_value={"bound": True})
    monkeypatch.setattr("task_dispatch.task_executor_node.resolve_robot_interfaces", resolver)
    assert _load_robot_yaml("robot.yaml") == {"bound": True}
    supplied = resolver.call_args.args[0]
    assert supplied["_config_path"] == "/resolved/robot.yaml"
    assert supplied["runtime"]["require_model"] is True
    assert config == original


# --- contract v2: multi-arm runtime without a public model (task_dispatch.groups) -------------


@pytest.fixture
def dual_config(config):
    interfaces = config["runtime"]["interface_description"]["interfaces"]
    interfaces["joint.state"]["joint_names"] = ["l1", "r1", "left_gripper", "right_gripper"]
    for side in ("left", "right"):
        interfaces[f"joint.trajectory_{side}_gripper"] = {
            "kind": "action",
            "direction": "serve",
            "message_type": "control_msgs/action/FollowJointTrajectory",
            "endpoint": f"/{side}_gripper_trajectory_controller/follow_joint_trajectory",
        }
    config.pop("robot_model")
    config["task_dispatch"] = {
        "default_group": "left_arm",
        "groups": {
            side + "_arm": {
                "gripper_joint": f"{side}_gripper",
                "gripper_trajectory_interface": f"joint.trajectory_{side}_gripper",
            }
            for side in ("left", "right")
        },
    }
    return config


def test_groups_bind_one_gripper_per_arm_without_a_public_model(dual_config):
    node = node_for(dual_config)
    node._configure_interfaces()
    assert node._move_service_name == "/unit/move"
    assert node._default_group == "left_arm" and node._gripper_joint == "left_gripper"
    assert {g: b[:2] for g, b in node._gripper_bindings.items()} == {
        "left_arm": ("left_gripper", "/left_gripper_trajectory_controller/follow_joint_trajectory"),
        "right_arm": ("right_gripper", "/right_gripper_trajectory_controller/follow_joint_trajectory"),
    }


def test_gripper_step_goes_to_its_groups_action_and_unknown_groups_are_refused(dual_config):
    node = node_for(dual_config)
    node._configure_interfaces()
    node._gripper_action_client = Mock()
    node._gripper_clients = {"left_arm": node._gripper_action_client, "right_arm": Mock()}
    assert node._gripper_binding("right_arm") == (
        "right_gripper",
        {"min": 0.0, "max": 1.0},
        node._gripper_clients["right_arm"],
    )
    assert node._gripper_binding("")[0] == "left_gripper"
    success, message = node._exec_gripper(SimpleNamespace(gripper_position=0.0, target_group="middle_arm"))
    assert not success and "unknown target_group" in message


def test_single_gripper_robot_refuses_a_named_group(config):
    node = node_for(config)
    node._configure_interfaces()
    node._gripper_action_client = Mock()
    success, message = node._exec_gripper(SimpleNamespace(gripper_position=0.5, target_group="right_arm"))
    assert not success and "unknown target_group" in message
    assert node._gripper_action_client.mock_calls == []


def test_groups_need_a_default_group(dual_config):
    dual_config["task_dispatch"]["default_group"] = "torso"
    with pytest.raises(InterfaceBindingError):
        node_for(dual_config)._configure_interfaces()


# ---- withdrawing a motion the executor stops waiting for (cancel / deadline) ----

CANCEL_INTERFACE = {
    "kind": "service",
    "direction": "serve",
    "message_type": "ibrobot_msgs/srv/CancelSimulationCommand",
    "endpoint": "/unit/cancel_command",
}


def test_the_cancel_service_binding_is_optional_and_typed(dual_config):
    node = node_for(dual_config)
    node._configure_interfaces()
    assert node._cancel_service_name is None  # unbound: the legacy local-only cancel
    dual_config["runtime"]["interface_description"]["interfaces"]["motion.cancel_command"] = CANCEL_INTERFACE
    dual_config["task_dispatch"]["cancel_command_interface"] = "motion.cancel_command"
    node._configure_interfaces()
    assert node._cancel_service_name == "/unit/cancel_command"
    dual_config["task_dispatch"]["cancel_command_interface"] = "motion.move_to_pose"
    with pytest.raises(InterfaceBindingError):
        node._configure_interfaces()


class _Future:
    def __init__(self, result=None, done=True):
        self._result, self._done, self.cancelled = result, done, False

    def done(self):
        return self._done

    def result(self):
        return self._result

    def cancel(self):
        self.cancelled = True


class _CancelClient:
    def __init__(self, reply):
        self.reply, self.ids = reply, []

    def service_is_ready(self):
        return True

    def wait_for_service(self, timeout_sec=None):
        return True

    def call_async(self, request):
        self.ids.append(request.command_id)
        return self.reply


def _executor(cancel_client):
    node = object.__new__(TaskExecutorNode)
    node._cancel_client = cancel_client
    node.get_logger = Mock
    return node


def test_without_a_cancel_service_only_the_local_wait_is_dropped():
    move = _Future(done=False)
    ok, message = _executor(None)._withdraw_motion(move, "task-1", "Cancelled during motion")
    assert not ok and message == "Cancelled during motion" and move.cancelled


def test_a_withdrawn_motion_is_reported_only_after_the_runtime_confirmed_it():
    client = _CancelClient(_Future(SimpleNamespace(outcome="CANCELLED", ticks_executed=4)))
    move = _Future(SimpleNamespace(outcome="CANCELLED"))
    ok, message = _executor(client)._withdraw_motion(move, "task-1", "Cancelled during motion")
    assert not ok and client.ids == ["task-1"]
    assert "runtime withdrew motion task-1 (CANCELLED, 4 ticks)" in message and not move.cancelled


def test_an_unconfirmed_withdrawal_says_so(monkeypatch):
    import task_dispatch.task_executor_node as module

    monkeypatch.setattr(module, "_WITHDRAW_CONFIRM_SEC", 0.05)
    move = _Future(done=False)
    ok, message = _executor(_CancelClient(_Future(done=False)))._withdraw_motion(move, "task-1", "Motion timed out")
    assert not ok and "did not confirm withdrawing motion task-1" in message and move.cancelled
