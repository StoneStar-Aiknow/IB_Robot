"""agent_eval's evaluation layer on the direct-control runtime: admission gate and verdict channel."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from agent_eval.runtime_extension import NOT_ADMITTED, EvaluationLayer


class FakeCore:
    def __init__(self, session_id="s0002", generation=2):
        self.session = (session_id, generation)

    def check_session(self, session_id, generation):
        if (session_id, generation) != self.session:
            raise RuntimeError(f"{session_id} g{generation} is not the active session")


def test_motion_is_held_until_the_active_session_is_admitted_and_only_for_its_generation():
    published = []
    layer = EvaluationLayer(FakeCore(), published.append)
    assert layer.gate(2) == NOT_ADMITTED
    ok, message = layer.admit("s0002", 3)
    assert not ok and "not admitted" in message and layer.gate(3) == NOT_ADMITTED
    ok, _ = layer.admit("s0001", 1)  # an earlier session is never opened again
    assert not ok and layer.gate(1) == NOT_ADMITTED
    ok, _ = layer.admit("s0002", 2)
    assert ok and layer.gate(2) == "" and layer.gate(3) == NOT_ADMITTED


def test_each_platform_result_goes_out_on_the_verdict_channel():
    published = []
    layer = EvaluationLayer(FakeCore(), published.append)
    layer.on_result({"session_id": "s0002", "generation": 2, "native_success": None, "interrupted": True})
    assert published[0]["interrupted"] is True and published[0]["session_id"] == "s0002"
    assert isinstance(published[0]["wall_time"], float)


# ---------------------------------------------------------------- on the toy direct-control runtime


@pytest.fixture
def rig(tmp_path):
    rclpy = pytest.importorskip("rclpy")
    fakes = pytest.importorskip("robot_runtime_sim.testing.direct_control_fakes")
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter

    from robot_runtime_sim.direct_control.node import SimulationRuntime

    # As run_suite does: the runtime's own profile copy lists the extension.
    document = yaml.safe_load(Path(fakes.TOY_PROFILE).read_text(encoding="utf-8"))
    document["simulation"]["toy"]["extensions"] = ["agent_eval.runtime_extension:attach"]
    profile = tmp_path / "toy_with_evaluation.yaml"
    profile.write_text(yaml.safe_dump(document), encoding="utf-8")
    rclpy.init()
    adapter = fakes.ToyAdapter()
    runtime = SimulationRuntime(
        adapter=adapter,
        node_name="toy_runtime",
        parameter_overrides=[Parameter("profile", value=str(profile)), Parameter("platform", value="toy")],
    )
    client = rclpy.create_node("agent_eval_extension_client")
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(runtime)
    executor.add_node(client)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    yield runtime, client, fakes
    executor.shutdown()
    spin.join(timeout=5.0)
    adapter.stop()
    runtime.destroy_node()
    client.destroy_node()
    rclpy.shutdown()


def _call(node, srv, name, request, timeout=10.0):
    cli = node.create_client(srv, name)
    assert cli.wait_for_service(timeout_sec=10.0)
    future = cli.call_async(request)
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    return future.result()


def test_the_extension_admits_episodes_and_publishes_verdicts_on_the_runtime(rig):
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from ibrobot_msgs.srv import AdmitSimulationEpisode, MoveToConfiguration, SetRuntimeMode

    runtime, client, fakes = rig
    core = runtime.core
    assert fakes.wait_until(lambda: core.clock()["state"] == "READY", 10.0)
    session_id, generation = core.clock()["session_id"], core.clock()["generation"]
    assert _call(client, SetRuntimeMode, "/runtime/set_mode", SetRuntimeMode.Request(mode="trajectory")).success
    move = MoveToConfiguration.Request()
    move.target_joint_state.name, move.target_joint_state.position = ["a_grip"], [0.5]
    refused = _call(client, MoveToConfiguration, "/motion/move_to_joint", move)
    assert refused.outcome == "REJECTED" and "not admitted" in refused.message

    admit = "/evaluation/admit_episode"
    wrong = AdmitSimulationEpisode.Request(session_id=session_id, generation=generation + 1)
    assert not _call(client, AdmitSimulationEpisode, admit, wrong).success
    right = AdmitSimulationEpisode.Request(session_id=session_id, generation=generation)
    assert _call(client, AdmitSimulationEpisode, admit, right).success
    assert _call(client, MoveToConfiguration, "/motion/move_to_joint", move).outcome == "SUCCEEDED"

    # The caller gives the session up (the toy's loop stopped first, as a real platform waits for
    # its reset): the interrupted end reaches a late (latched) subscriber, on this channel only.
    runtime.adapter.stop()
    assert _call(client, Trigger, "/simulation/abort_session", Trigger.Request()).success
    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    verdicts = []
    client.create_subscription(
        String, "/evaluation/native_result", lambda m: verdicts.append(json.loads(m.data)), latched
    )
    assert fakes.wait_until(lambda: bool(verdicts), 10.0)
    assert verdicts[0]["session_id"] == session_id and verdicts[0]["interrupted"] is True
    assert verdicts[0]["native_success"] is None
