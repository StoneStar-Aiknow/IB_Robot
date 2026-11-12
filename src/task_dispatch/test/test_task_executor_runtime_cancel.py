"""A cancelled MOVE_TO_POSE step withdraws its command in the runtime before the step ends.

Cancelling the local service future does not stop the server's callback. With
``task_dispatch.cancel_command_interface`` bound, the executor withdraws the motion in the
runtime by the ``command_id`` it put in the request, and reports the step only once the
runtime confirmed it and the motion's own reply came back.

Here the executor runs against the real direct-control runtime node on robot_runtime_sim's
CPU toy platform (its test doubles, as in robot_runtime_sim's node tests). The test lives in
task_dispatch because the behaviour is the executor's, and robot_runtime_sim cannot
test_depend on task_dispatch (robot_config, which task_dispatch depends on, depends on it).
"""

from __future__ import annotations

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("robot_runtime_sim.direct_control.node")

from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from ibrobot_msgs.msg import TaskStep  # noqa: E402
from ibrobot_msgs.srv import CancelSimulationCommand, MoveToPose, SetRuntimeMode  # noqa: E402
from robot_runtime_sim.direct_control.node import SimulationRuntime  # noqa: E402
from robot_runtime_sim.testing.direct_control_fakes import TOY_PROFILE, ToyAdapter, wait_until  # noqa: E402
from task_dispatch.task_executor_node import TaskExecutorNode  # noqa: E402


@pytest.fixture
def rig():
    rclpy.init()
    adapter = ToyAdapter()
    overrides = [Parameter("profile", value=str(TOY_PROFILE)), Parameter("platform", value="toy")]
    runtime = SimulationRuntime(adapter=adapter, node_name="toy_runtime", parameter_overrides=overrides)
    client = Node("task_executor_client")
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(runtime)
    executor.add_node(client)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    yield runtime, client, adapter
    executor.shutdown()
    spin.join(timeout=5.0)
    runtime.destroy_node()
    client.destroy_node()
    rclpy.shutdown()


def _bound_executor(client: Node) -> TaskExecutorNode:
    """The executor as task_dispatch binds it for this runtime: only what a MOVE_TO_POSE step uses."""
    node = object.__new__(TaskExecutorNode)
    node.get_logger = client.get_logger
    node._move_service_name = "/motion/move_to_pose"
    node._move_client = client.create_client(MoveToPose, "/motion/move_to_pose")
    node._cancel_client = client.create_client(CancelSimulationCommand, "/simulation/cancel_command")
    node._cancel_requested = False
    return node


def _far_pose_step() -> TaskStep:
    step = TaskStep()
    step.type = TaskStep.MOVE_TO_POSE
    step.target_pose.position.x, step.target_pose.position.y, step.target_pose.position.z = 2.0, 0.1, 0.1
    step.target_pose.orientation.w = 1.0
    step.velocity_scaling = 1.0  # 1.9 rad at 0.06 rad/tick: 32 ticks of 20 ms wall each
    step.target_group = "a"
    return step


def test_a_cancelled_move_step_ends_only_after_the_runtime_withdrew_it(rig):
    runtime, client, adapter = rig
    core = runtime.core
    mode = client.create_client(SetRuntimeMode, "/runtime/set_mode")
    assert mode.wait_for_service(timeout_sec=10.0)
    future = mode.call_async(SetRuntimeMode.Request(mode="trajectory"))
    assert wait_until(future.done) and future.result().success
    real_submit, seen = core.submit, {}

    def recording_submit(cmd, timeout_s):
        seen["cmd"] = cmd
        return real_submit(cmd, timeout_s)

    core.submit = recording_submit
    executor = _bound_executor(client)
    box = {}

    def run_step():
        box["result"] = executor._exec_move_to_pose(_far_pose_step())
        box["withdrawn_at_return"] = seen["cmd"].done.is_set()
        box["tick_at_return"] = adapter.env.tick

    thread = threading.Thread(target=run_step, daemon=True)
    thread.start()
    assert wait_until(lambda: adapter.env.tick >= 3)
    executor._cancel_requested = True  # what the task's cancel callback sets
    thread.join(timeout=15.0)
    assert not thread.is_alive()
    ok, message = box["result"]
    command_id = seen["cmd"].command_id
    assert not ok and command_id.startswith("task-") and f"withdrew motion {command_id} (CANCELLED" in message
    assert box["withdrawn_at_return"] and seen["cmd"].result["outcome"] == "CANCELLED"
    time.sleep(0.3)
    assert adapter.env.tick <= box["tick_at_return"] + 1  # nothing moves once the step has ended
