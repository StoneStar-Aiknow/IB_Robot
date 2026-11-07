"""Task poses are always dispatched with the gravity-sag command shaper."""

from threading import Lock
from types import SimpleNamespace

from aimdk_grasp.sag_compensation import SagCompensation
from test_task_poses import MockArm, snapshot

from ibrobot_msgs.action import ExecuteArmPose
from pose_composer.node import PoseComposerNode
from pose_composer.task_poses import TaskPoseRunner


def test_task_poses_send_their_trajectory_with_the_sag_shaper():
    arm = MockArm(away=True)
    calls = []
    server = object.__new__(PoseComposerNode)
    server.get_logger = lambda: SimpleNamespace(info=lambda *_: None)
    server._joint_lock = Lock()
    server._motion_limits = (0.0, 0.0)
    server._runtime_mode = lambda: "policy_stream"
    server._feedback = lambda *_: None
    server._hand_restorer = None
    server._task_pose_runner = TaskPoseRunner(snapshot(), simulation=True, clock=lambda: arm.now, pause=arm.sleep)
    server._task_pose_commands = commands = SagCompensation(list(arm.q))

    def publish():
        server._positions = dict(arm.q)
        server._joint_received = dict.fromkeys(arm.q, arm.now)

    def run_trajectory(_goal, plan, *, deadline, **shaping):
        calls.append(shaping)
        result = arm.move(plan, deadline)
        publish()
        return result

    server._run_trajectory = run_trajectory
    publish()
    goal = SimpleNamespace(
        request=SimpleNamespace(timeout_sec=5.0, gateway_task_id="task"),
        is_cancel_requested=False,
        succeed=lambda: None,
    )
    result = server._execute_task_pose(
        goal, ExecuteArmPose.Result(), "grasp_ready", snapshot()["digest"], "policy_stream"
    )
    assert result.success
    assert calls == [{"commands": commands}]
