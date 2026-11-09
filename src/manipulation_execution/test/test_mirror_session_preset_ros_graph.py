"""Real local ROS protocol smoke; the peer never commands any robot hardware.

Each case owns two test nodes, a private context and UUID-named interfaces.
The only data topics are the named action's feedback/status and String
lease/ack topics. No runtime, vendor, arm, hand or joint node is constructed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass

import pytest
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from ibrobot_msgs.action import ExecuteNamedMotion
from manipulation_execution.mirror_session.hooks import UpperBodyCommand
from manipulation_execution.mirror_session.preset import PresetEpoch
from manipulation_execution.mirror_session.preset_ros import RosPresetPort

_MOTION = "overhead_heart"


def _description(prefix):
    qos = {"depth": 10, "reliability": "reliable", "durability": "volatile"}
    return {
        "interfaces": {
            "motion.named": {
                "kind": "action",
                "direction": "serve",
                "message_type": "ibrobot_msgs/action/ExecuteNamedMotion",
                "endpoint": f"{prefix}/named",
            },
            "motion.named_hold_lease": {
                "kind": "topic",
                "direction": "subscribe",
                "message_type": "std_msgs/msg/String",
                "endpoint": f"{prefix}/lease",
                "qos": qos,
            },
            "motion.named_hold_lease_ack": {
                "kind": "topic",
                "direction": "publish",
                "message_type": "std_msgs/msg/String",
                "endpoint": f"{prefix}/ack",
                "qos": qos,
            },
        }
    }


class _ProtocolPeer:
    """A bounded protocol peer, with no output beyond action results/feedback."""

    def __init__(self, node, prefix, closing, deadline):
        self.node = node
        self.closing = closing
        self.deadline = deadline
        self.lock = threading.RLock()
        self.leases = {}
        self.acked = set()
        self.expected_id = ""
        self.owner = None
        self.events = []
        self.payloads = []
        self.errors = []
        self.group = ReentrantCallbackGroup()
        self.ack = node.create_publisher(String, f"{prefix}/ack", 10)
        self.subscription = node.create_subscription(
            String, f"{prefix}/lease", self._lease, 10, callback_group=self.group
        )
        self.server = ActionServer(
            node,
            ExecuteNamedMotion,
            f"{prefix}/named",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
            handle_accepted_callback=self._accepted,
            callback_group=self.group,
        )

    def _lease(self, message):
        try:
            payload = json.loads(message.data)
            goal_id, motion = payload["goal_id"], payload["motion"]
            assert len(bytes.fromhex(goal_id)) == 16 and motion == _MOTION
            with self.lock:
                self.leases[goal_id] = payload
                self.payloads.append(payload)
                self.events.append(("lease", goal_id))
        except Exception as exc:
            with self.lock:
                self.errors.append(f"invalid lease: {exc!r}")

    def acknowledge(self, goal_id):
        with self.lock:
            lease = self.leases[goal_id]
            self.acked.add(goal_id)
            self.events.append(("ack", goal_id))
            self.ack.publish(String(data=json.dumps({"goal_id": goal_id, "motion": lease["motion"]})))

    def _goal(self, request):
        # The public goal_callback receives Goal only, without its wire UUID.
        # Check the expected UUID's received lease here; _accepted checks the
        # actual ServerGoalHandle UUID before starting the normal executor.
        with self.lock:
            goal_id = self.expected_id
            self.events.append(("goal_request", goal_id))
            valid = (
                goal_id in self.leases
                and goal_id in self.acked
                and self.leases[goal_id]["motion"] == request.name == _MOTION
                and request.target == "both"
                and request.interrupt is False
                and request.resume_token == ""
                and self.owner is None
            )
            if not valid:
                self.errors.append("goal reached admission without its received and acknowledged lease")
            return GoalResponse.ACCEPT if valid else GoalResponse.REJECT

    def _accepted(self, handle):
        goal_id = bytes(handle.goal_id.uuid).hex()
        with self.lock:
            valid = goal_id == self.expected_id and goal_id in self.leases and goal_id in self.acked
            self.events.append(("accepted_uuid" if valid else "wrong_uuid", goal_id))
            if not valid:
                self.errors.append("accepted wire UUID did not match the received UUID-bound lease")
        handle.execute(execute_callback=self._execute if valid else self._reject_uuid)

    @staticmethod
    def _reject_uuid(handle):
        handle.abort()
        return ExecuteNamedMotion.Result(
            success=False, error_code=ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM, message="test UUID mismatch"
        )

    def _cancel(self, handle):
        with self.lock:
            self.events.append(("cancel_request", bytes(handle.goal_id.uuid).hex()))
        return CancelResponse.ACCEPT

    def _release(self, goal_id):
        with self.lock:
            assert self.owner == goal_id
            self.owner = None
            self.events.append(("released", goal_id))

    def _execute(self, handle):
        goal_id = bytes(handle.goal_id.uuid).hex()
        with self.lock:
            self.owner = goal_id
            self.events.append(("executing", goal_id))
        next_feedback = 0.0
        try:
            while not self.closing.is_set() and self.node.context.ok() and time.monotonic() < self.deadline:
                if handle.is_cancel_requested:
                    self._release(goal_id)
                    handle.canceled()
                    return ExecuteNamedMotion.Result(
                        success=False, error_code=ExecuteNamedMotion.Result.CANCELLED, message="test owner released"
                    )
                with self.lock:
                    payload = self.leases[goal_id]
                    field_name = "stop_target" if "stop_target" in payload else "exit_target"
                    target = payload.get(field_name)
                if target is not None:
                    assert len(target) == 34
                    encoded = json.dumps(target, separators=(",", ":"), allow_nan=False).encode("utf-8")
                    result = json.dumps(
                        {"motion": _MOTION, f"{field_name}_sha256": hashlib.sha256(encoded).hexdigest()}
                    )
                    with self.lock:
                        self.events.append((f"{field_name}_applied", goal_id))
                    self._release(goal_id)
                    handle.succeed()
                    return ExecuteNamedMotion.Result(
                        success=True, error_code=ExecuteNamedMotion.Result.NONE, message=result
                    )
                now = time.monotonic()
                if now >= next_feedback:
                    handle.publish_feedback(ExecuteNamedMotion.Feedback(state="holding", progress=1.0))
                    next_feedback = now + 0.03
                self.closing.wait(0.005)
            self._release(goal_id)
            handle.abort()
            return ExecuteNamedMotion.Result(
                success=False, error_code=ExecuteNamedMotion.Result.TIMEOUT, message="bounded test peer stopped"
            )
        except Exception as exc:
            with self.lock:
                self.errors.append(f"protocol executor failed: {exc!r}")
                self.owner = None
            if handle.is_active:
                handle.abort()
            return ExecuteNamedMotion.Result(
                success=False, error_code=ExecuteNamedMotion.Result.INTERNAL_ERROR, message="test peer exception"
            )


@dataclass
class _Graph:
    port: RosPresetPort
    peer: _ProtocolPeer
    deadline: float
    closing: threading.Event
    spin_errors: list

    def wait(self, predicate):
        while not predicate():
            self.healthy()
            assert time.monotonic() < self.deadline, "isolated named-action protocol did not finish within 5 s"
            self.closing.wait(0.005)
        self.healthy()

    def healthy(self):
        assert not self.spin_errors, f"ROS executor failed: {self.spin_errors}"
        with self.peer.lock:
            assert not self.peer.errors, f"protocol peer failed: {self.peer.errors}"

    def begin_with_ack_barrier(self):
        self.port.begin(_MOTION, PresetEpoch(7, 1))
        goal_id = self.port.snapshot().goal_id
        with self.peer.lock:
            self.peer.expected_id = goal_id
        self.wait(lambda: goal_id in self.peer.leases)
        # Receipt of the lease is insufficient. Withhold ACK on the real wire
        # long enough for an erroneous immediate dispatch to reach admission.
        until = min(time.monotonic() + 0.1, self.deadline)
        while time.monotonic() < until:
            self.healthy()
            self.closing.wait(0.005)
        state = self.port.snapshot()
        assert state.pending and not state.accepted and state.stage == "waiting_lease_ack"
        with self.peer.lock:
            assert not any(event == "goal_request" for event, _ in self.peer.events)
        self.peer.acknowledge(goal_id)
        self.wait(lambda: self.port.snapshot().accepted and self.port.snapshot().stage == "holding")
        with self.peer.lock:
            order = [event for event, identifier in self.peer.events if identifier == goal_id]
            assert order.index("lease") < order.index("ack") < order.index("goal_request")
            assert order.index("goal_request") < order.index("accepted_uuid") < order.index("executing")
            assert self.peer.owner == goal_id
        return goal_id


@pytest.fixture
def graph():
    domain = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert domain, "the repository's per-process ROS domain isolation must be loaded"
    assert os.environ.get("ROS_DOMAIN_ID") == domain
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    suffix = uuid.uuid4().hex
    prefix = f"/test_mirror_preset_{suffix}"
    context = Context()
    closing = threading.Event()
    deadline = time.monotonic() + 5.0
    spin_errors = []
    server_node = client_node = peer = executor = spin_thread = None
    stopped = True
    rclpy.init(args=[], context=context, domain_id=int(domain))
    try:
        server_node = Node(f"preset_protocol_server_{suffix}", context=context)
        client_node = Node(f"preset_protocol_client_{suffix}", context=context)
        peer = _ProtocolPeer(server_node, prefix, closing, deadline)
        port = RosPresetPort(client_node, _description(prefix), callback_group=ReentrantCallbackGroup())
        executor = MultiThreadedExecutor(num_threads=4, context=context)
        executor.add_node(server_node)
        executor.add_node(client_node)

        def spin():
            try:
                while not closing.is_set() and context.ok():
                    executor.spin_once(timeout_sec=0.02)
            except Exception as exc:
                if not closing.is_set():
                    spin_errors.append(repr(exc))

        spin_thread = threading.Thread(target=spin, name=f"preset_graph_{suffix}", daemon=True)
        spin_thread.start()
        result = _Graph(port, peer, deadline, closing, spin_errors)
        result.wait(
            lambda: client_node.count_subscribers(f"{prefix}/lease") == 1 and peer.ack.get_subscription_count() == 1
        )
        yield result
        result.healthy()
    finally:
        # Even an assertion failure terminates every execute loop, so a green
        # test cannot leave an action worker blocking Python's atexit join.
        closing.set()
        if executor is not None:
            executor.wake()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        if executor is not None:
            stopped = executor.shutdown(timeout_sec=2.0)
        thread_alive = spin_thread is not None and spin_thread.is_alive()
        if not thread_alive and stopped:
            if peer is not None:
                peer.server.destroy()
            if client_node is not None:
                client_node.destroy_node()
            if server_node is not None:
                server_node.destroy_node()
            context.try_shutdown()
        assert not thread_alive, "test ROS spin thread failed to terminate"
        assert stopped, "test ROS executor retained an unfinished callback"


def test_real_uuid_lease_ack_goal_exit_hash_confirms_release(graph):
    goal_id = graph.begin_with_ack_barrier()
    target = UpperBodyCommand((0.1,) * 14, (0.2,) * 20)
    graph.port.refresh(target)
    graph.wait(lambda: graph.port.snapshot().terminal)
    state = graph.port.snapshot()
    assert state.goal_id == goal_id and state.epoch == PresetEpoch(7, 1)
    assert state.accepted and not state.pending and state.success and state.release_confirmed
    assert state.final_target_verified and state.final_target == target
    assert state.final_target_at == state.observed_at
    assert not state.fault
    with graph.peer.lock:
        assert graph.peer.owner is None
        assert ("released", goal_id) in graph.peer.events
        assert graph.peer.leases[goal_id]["exit_target"] == list(target.arm + target.hand)


def test_real_cancelled_result_confirms_release_without_a_resume_target(graph):
    goal_id = graph.begin_with_ack_barrier()
    graph.port.cancel("test_stop")
    graph.wait(lambda: graph.port.snapshot().terminal)
    state = graph.port.snapshot()
    assert state.goal_id == goal_id and state.accepted and not state.pending
    assert state.release_confirmed and not state.success and state.stage == "cancelled"
    assert state.final_target is None and state.final_target_at is None and not state.final_target_verified
    assert not state.fault
    with graph.peer.lock:
        assert graph.peer.owner is None
        order = [event for event, identifier in graph.peer.events if identifier == goal_id]
        assert order.index("executing") < order.index("cancel_request") < order.index("released")


def test_real_stop_target_success_proves_release_without_a_resume_final_target(graph):
    goal_id = graph.begin_with_ack_barrier()
    stop = UpperBodyCommand((0.15,) * 14, (0.25,) * 20)
    graph.port.stop_to(stop)
    # The coordinator can keep passing an ordinary mirror exit while it renews
    # a stopping flight. Only the fixed stop target is allowed onto the wire.
    graph.port.refresh(UpperBodyCommand((0.7,) * 14, (0.8,) * 20))
    graph.wait(lambda: graph.port.snapshot().terminal)
    state = graph.port.snapshot()
    assert state.goal_id == goal_id and state.epoch == PresetEpoch(7, 1)
    assert state.accepted and not state.pending and state.success and state.release_confirmed
    assert state.stop_target_verified and state.stop_target == stop
    assert state.stop_target_at == state.observed_at
    assert state.final_target is None and state.final_target_at is None and not state.final_target_verified
    assert not state.fault
    with graph.peer.lock:
        assert graph.peer.owner is None
        order = [event for event, identifier in graph.peer.events if identifier == goal_id]
        assert order.index("executing") < order.index("stop_target_applied") < order.index("released")
        assert "exit_target_applied" not in order and "cancel_request" not in order
        targets = [payload for payload in graph.peer.payloads if "stop_target" in payload]
        assert targets and all(payload["stop_target"] == list(stop.arm + stop.hand) for payload in targets)
        assert not any("exit_target" in payload for payload in graph.peer.payloads)
