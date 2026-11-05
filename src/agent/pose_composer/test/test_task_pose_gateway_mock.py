"""Real Gateway and pose handler/planner, with Mock transport, arm and grasp."""

from concurrent.futures import Future
from threading import Lock
from types import SimpleNamespace

import pytest
from test_task_pose_hands import MockHands, hand_snapshot
from test_task_poses import MockArm, snapshot

from ibrobot_msgs.action import SkillCommand
from pose_composer.node import PoseComposerNode
from pose_composer.task_poses import TaskPoseRunner
from robot_runtime.task_poses import RETURN_TASK_POSE
from skill_library import skill_executor_node as gateway_module
from skill_library.skill_executor_node import SkillExecutorNode


def future(value):
    result = Future()
    result.set_result(value)
    return result


class Goal:
    def __init__(self, request, arm):
        self.request, self.arm, self.terminal = request, arm, None
        self.feedback = []

    @property
    def is_cancel_requested(self):
        return self.arm.cancelled

    def succeed(self):
        self.terminal = "succeeded"

    def abort(self):
        self.terminal = "aborted"

    def canceled(self):
        self.terminal = "canceled"

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)


def system(monkeypatch, *, away=False, fault="", hands=False, hand_fault=""):
    arm = MockArm(away=away)
    config, events = hand_snapshot() if hands else snapshot(), []
    monkeypatch.setattr(gateway_module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(gateway_module.time, "monotonic", lambda: arm.now)
    logs = []
    server = object.__new__(PoseComposerNode)
    server.get_logger = lambda: SimpleNamespace(info=logs.append)
    server._busy, server._joint_lock = Lock(), Lock()
    server._motion_limits = (0.0, 0.0)
    server._trajectory_modes = {"policy_stream"}
    server._runtime_mode = lambda: "policy_stream"

    def publish():
        server._positions = dict(arm.q)
        server._joint_received = dict.fromkeys(arm.q, arm.now)

    def pause(seconds):
        arm.sleep(seconds)
        publish()

    def motion(_goal, plan, *, deadline):
        events.append("trajectory")
        result = arm.move(plan, deadline)
        publish()
        return result

    publish()
    server._run_trajectory = motion
    server._task_pose_runner = TaskPoseRunner(config, simulation=True, clock=lambda: arm.now, pause=pause)
    restorer = MockHands(arm, events, advance=pause, fault=hand_fault)
    server._hand_restorer = lambda: restorer

    class Client:
        def wait_for_server(self, timeout_sec):
            return True

        def send_goal_async(self, request, feedback_callback):
            assert request.named_pose == "grasp_ready"
            assert not request.text and not request.composition_id
            if fault == "mixed_request":
                request.text = "a different free-text target"
            events.append("prepare")
            result = server._on_execute(Goal(request, arm))
            events.append("verified" if result.success else "preparation_failed")
            if fault == "cancel_between":
                arm.cancelled = True
            return future(
                SimpleNamespace(accepted=True, get_result_async=lambda: future(SimpleNamespace(result=result)))
            )

    gateway = object.__new__(SkillExecutorNode)
    gateway._preparation_logs = logs
    gateway._pose_composer_client = Client()
    gateway._pose_composer_action = "/mock_pose/execute"
    gateway._task_poses = config
    gateway._rpc_timeout = 0.1
    gateway._active_skill_admission = object()
    gateway._register_delegated_dispatch = lambda *_: b"mock_registered"
    gateway._confirm_delegated_terminal = lambda *_: events.append("preparation_terminal_confirmed")
    gateway._wait_for_future = lambda f, **_: f.done()
    gateway._set_result_catalog_identity = lambda *_: None
    gateway.get_logger = lambda: SimpleNamespace(error=lambda message: pytest.fail(message))
    gateway._audit_cancel_propagated = lambda: None
    gateway._validate_skill = lambda *_args, **_kwargs: (True, "")
    gateway._active_mirror_session_command = lambda _: ""
    gateway._ensure_skill_control_mode = lambda *_: (events.append("mode_admitted") or True, "")
    gateway._ensure_skill_capabilities = lambda _: (True, "")
    gateway._runtime_enabled = True
    gateway._runtime_status_for_admission = lambda: (
        SimpleNamespace(active_mode="head" if fault == "mode_between" else "policy_stream"),
        "",
    )
    gateway._required_control_mode_for_skill = lambda _: "moveit_planning"
    gateway._runtime_mode_for = lambda _: "policy_stream"
    gateway._skill_required_control_mode = "moveit_planning"
    gateway._skill_templates = {
        "pick_object": {"executor": "grasp_pipeline", "timeout_sec": 20.0},
        RETURN_TASK_POSE: {"executor": "pose_composer", "timeout_sec": 5.0},
    }

    def grasp(handle, _template, *, effective_timeout_sec=None):
        assert handle.terminal is None
        assert arm.q == pytest.approx(server._task_pose_runner.snapshot["poses"]["grasp_ready"]["joint_positions"])
        assert effective_timeout_sec > 0
        events.extend(["capture_current_scene", "grasp"])
        handle.succeed()
        result = SkillCommand.Result()
        result.success, result.executed_primitives = True, ["mock_grasp"]
        return result

    gateway._execute_pick_skill = grasp
    request = SkillCommand.Goal()
    request.skill_name, request.target_name = "pick_object", "cup"
    request.dispatch_binding.root_task_id = "mock_task"
    return gateway, arm, Goal(request, arm), events


@pytest.mark.parametrize("away", [False, True])
@pytest.mark.parametrize("hands", [False, True])
def test_mock_preparation_then_fresh_grasp(monkeypatch, away, hands):
    gateway, arm, goal, events = system(monkeypatch, away=away, hands=hands)
    result = gateway._execute_skill_child(goal, effective_timeout_sec=20.0)
    assert result.success and goal.terminal == "succeeded"
    assert events.index("verified") < events.index("capture_current_scene") < events.index("grasp")
    assert len(arm.moves) == int(away)
    assert result.executed_primitives == ["task_pose:grasp_ready", "mock_grasp"]
    assert goal.request.target_name == "cup"
    if hands:
        assert events.index("takeover") < events.index("opening_left") < events.index("opening_right")
        assert events.index("opening_right") < events.index("confirming") < events.index("verified")
        assert "opened_hands=left,right" in result.message and len(result.message) <= 300
        assert any('"confirming"' in line for line in gateway._preparation_logs)
        if away:
            assert events.index("takeover") < events.index("trajectory") < events.index("opening_left")


@pytest.mark.parametrize("fault", ["cancel_between", "mode_between", "mixed_request"])
def test_mock_preparation_failure_or_stop_blocks_grasp(monkeypatch, fault):
    gateway, arm, goal, events = system(monkeypatch, away=True, fault=fault)
    result = gateway._execute_skill_child(goal, effective_timeout_sec=20.0)
    assert not result.success and goal.terminal in {"aborted", "canceled"}
    assert "capture_current_scene" not in events and "grasp" not in events
    if fault == "mixed_request":
        assert not arm.moves


@pytest.mark.parametrize("hands", [False, True])
def test_standalone_named_skill_uses_same_handler_without_grasp(monkeypatch, hands):
    gateway, arm, goal, events = system(monkeypatch, away=True, hands=hands)
    goal.request.skill_name, goal.request.target_name = RETURN_TASK_POSE, "grasp_ready"
    result = gateway._execute_skill_child(goal, effective_timeout_sec=20.0)
    assert result.success and goal.terminal == "succeeded" and len(arm.moves) == 1
    assert "grasp" not in events
    if hands:
        assert "confirming" in events


@pytest.mark.parametrize("fault", ["preflight", "opening_right", "confirming", "cancel"])
def test_each_hand_phase_failure_blocks_capture_and_grasp(monkeypatch, fault):
    gateway, _arm, goal, events = system(monkeypatch, hands=True, hand_fault=fault)
    result = gateway._execute_skill_child(goal, effective_timeout_sec=20.0)
    assert not result.success and goal.terminal in {"aborted", "canceled"}
    assert "capture_current_scene" not in events and "grasp" not in events
    if fault in {"opening_right", "cancel"}:
        assert "opened_hands=left" in result.message
        assert "opened_hands=left,right" not in result.message
