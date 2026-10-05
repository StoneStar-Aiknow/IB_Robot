"""Business execution over injected public endpoints, with no robot-specific dependency."""

import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.executors import MultiThreadedExecutor

from ibrobot_msgs.action import ExecuteNamedMotion
from ibrobot_msgs.srv import ExecuteInteractionDemo, SpeakText
from robot_interaction_demo.node import InteractionDemoNode
from robot_interaction_demo.runtime_demo import execute_named_motion, execute_speech

SPEECH_ENDPOINT = "/test/runtime/speech"
MOTION_ENDPOINT = "/test/runtime/motion"


@pytest.fixture
def rpc():
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "ROS graph tests require root domain isolation"
    rclpy.init()
    server = rclpy.create_node("interaction_demo_test_runtime")
    client = rclpy.create_node("interaction_demo_test_client")
    state = SimpleNamespace(calls=[], speech_ok=True, motion_ok=True, accept=True, delay=0.0)

    def speak(request, response):
        state.calls.append(("speak", request))
        response.success = state.speech_ok
        response.error_code = "" if state.speech_ok else "BUSY"
        response.message = "queued" if state.speech_ok else "busy"
        response.utterance_id = request.trace_id
        return response

    def motion(handle):
        state.calls.append(("motion", handle.request))
        time.sleep(state.delay)
        result = ExecuteNamedMotion.Result()
        result.success = state.motion_ok
        result.error_code = result.NONE if state.motion_ok else result.UNSAFE_POSTURE
        result.message = "completed" if state.motion_ok else "stable stand required"
        if state.motion_ok:
            handle.succeed()
        else:
            handle.abort()
        return result

    server.create_service(SpeakText, SPEECH_ENDPOINT, speak)
    action = ActionServer(
        server,
        ExecuteNamedMotion,
        MOTION_ENDPOINT,
        execute_callback=motion,
        goal_callback=lambda _: GoalResponse.ACCEPT if state.accept else GoalResponse.REJECT,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(server)
    state.executor = executor
    thread = threading.Thread(target=executor.spin)
    thread.start()
    try:
        yield client, state
    finally:
        assert executor.shutdown(timeout_sec=5.0)
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        action.destroy()
        server.destroy_node()
        client.destroy_node()
        rclpy.shutdown()


def test_speech_preserves_call_parameters_and_reports_acceptance_only(rpc):
    client, state = rpc
    result = execute_speech(
        client,
        SPEECH_ENDPOINT,
        text="你好",
        language="zh-CN",
        priority=60,
        interrupt=True,
        timeout_sec=5.0,
    )
    assert result["success"] and result["phase"] == "accepted"
    assert "does not confirm" in result["note"]
    [(kind, request)] = state.calls
    assert kind == "speak"
    assert (request.text, request.language, request.priority, request.interrupt) == ("你好", "zh-CN", 60, True)
    assert result["utterance_id"] == request.trace_id


def test_speech_accepts_empty_language_for_runtime_default(rpc):
    client, state = rpc
    assert execute_speech(
        client, SPEECH_ENDPOINT, text="hello", language="", priority=50, interrupt=False, timeout_sec=5.0
    )["success"]
    assert state.calls[0][1].language == ""


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"text": "", "language": "", "priority": 50, "interrupt": False}, "text"),
        ({"text": "hi", "language": "", "priority": 101, "interrupt": False}, "priority"),
        ({"text": "hi", "language": "", "priority": 50, "interrupt": 1}, "interrupt"),
    ],
)
def test_invalid_speech_is_rejected_before_ros(rpc, kwargs, match):
    client, state = rpc
    with pytest.raises(ValueError, match=match):
        execute_speech(client, SPEECH_ENDPOINT, timeout_sec=5.0, **kwargs)
    assert not state.calls


def test_speech_runtime_rejection_is_not_success(rpc):
    client, state = rpc
    state.speech_ok = False
    result = execute_speech(
        client, SPEECH_ENDPOINT, text="hi", language="", priority=50, interrupt=False, timeout_sec=5.0
    )
    assert not result["success"] and result["error_code"] == "BUSY"


@pytest.mark.parametrize("name,target", [("wave", "right"), ("raise_hand", "right"), ("clap", "")])
def test_motion_uses_parameterized_semantic_request(rpc, name, target):
    client, state = rpc
    result = execute_named_motion(
        client,
        MOTION_ENDPOINT,
        name=name,
        target=target,
        interrupt=False,
        allow_motion=True,
        timeout_sec=5.0,
    )
    assert result["success"] and result["phase"] == "terminal"
    [(kind, request)] = state.calls
    assert kind == "motion"
    assert (request.name, request.target, request.interrupt) == (name, target, False)


def test_motion_requires_explicit_request_authorization(rpc):
    client, state = rpc
    with pytest.raises(ValueError, match="allow_motion"):
        execute_named_motion(
            client,
            MOTION_ENDPOINT,
            name="wave",
            target="right",
            interrupt=False,
            allow_motion=False,
            timeout_sec=5.0,
        )
    assert not state.calls


@pytest.mark.parametrize("name,target", [("", "right"), ("wave", "body")])
def test_invalid_motion_is_rejected_before_ros(rpc, name, target):
    client, state = rpc
    with pytest.raises(ValueError):
        execute_named_motion(
            client,
            MOTION_ENDPOINT,
            name=name,
            target=target,
            interrupt=False,
            allow_motion=True,
            timeout_sec=5.0,
        )
    assert not state.calls


@pytest.mark.parametrize("accept", [True, False])
def test_motion_runtime_failure_and_goal_rejection(rpc, accept):
    client, state = rpc
    state.accept = accept
    state.motion_ok = False
    result = execute_named_motion(
        client,
        MOTION_ENDPOINT,
        name="wave",
        target="right",
        interrupt=False,
        allow_motion=True,
        timeout_sec=5.0,
    )
    assert not result["success"]
    if accept:
        assert result["error_code"] == ExecuteNamedMotion.Result.UNSAFE_POSTURE
    else:
        assert not state.calls


def test_timed_out_motion_is_not_retried(rpc):
    client, state = rpc
    state.delay = 2.0
    with pytest.raises(TimeoutError, match="may still execute"):
        execute_named_motion(
            client,
            MOTION_ENDPOINT,
            name="wave",
            target="right",
            interrupt=False,
            allow_motion=True,
            timeout_sec=1.0,
        )
    assert len(state.calls) == 1


@pytest.fixture
def resident(rpc):
    client, state = rpc
    demo = InteractionDemoNode(
        parameter_overrides=[
            rclpy.parameter.Parameter("speech_service", value=SPEECH_ENDPOINT),
            rclpy.parameter.Parameter("named_motion_action", value=MOTION_ENDPOINT),
            rclpy.parameter.Parameter("timeout_sec", value=5.0),
        ]
    )
    state.executor.add_node(demo)
    transport = client.create_client(ExecuteInteractionDemo, "/interaction_demo/execute")
    assert transport.wait_for_service(timeout_sec=3.0)

    def request(message):
        future = transport.call_async(message)
        rclpy.spin_until_future_complete(client, future, timeout_sec=7.0)
        assert future.done(), "resident service did not finish"
        response = future.result()
        result = json.loads(response.result_json)
        assert response.success == result["success"]
        return result

    try:
        yield request, state, transport, client
    finally:
        state.executor.remove_node(demo)
        demo.destroy_node()


def speech_request(text="hello", language="", priority=50, interrupt=False):
    return ExecuteInteractionDemo.Request(
        operation=ExecuteInteractionDemo.Request.SPEAK,
        text=text,
        language=language,
        priority=priority,
        interrupt=interrupt,
    )


def motion_request(name="wave", target="right", allow=True):
    return ExecuteInteractionDemo.Request(
        operation=ExecuteInteractionDemo.Request.NAMED_MOTION,
        motion_name=name,
        target=target,
        allow_motion=allow,
    )


def test_resident_startup_does_not_command_robot(resident):
    _, state, _, _ = resident
    assert not state.calls


def test_resident_dispatches_parameterized_speech_and_motion(resident):
    request, state, _, _ = resident
    assert request(speech_request("自定义文本", "zh-CN", 70, True))["phase"] == "accepted"
    assert request(motion_request("raise_hand", "left"))["phase"] == "terminal"
    assert state.calls[0][1].text == "自定义文本"
    assert state.calls[1][1].name == "raise_hand"


def test_resident_rejects_unknown_operation_and_unapproved_motion(resident):
    request, state, _, _ = resident
    assert not request(ExecuteInteractionDemo.Request(operation=99))["success"]
    assert not request(motion_request(allow=False))["success"]
    assert not state.calls


def test_resident_rejects_busy_request_without_queueing(resident):
    request, state, transport, client = resident
    state.delay = 0.8
    running = transport.call_async(motion_request())
    deadline = time.monotonic() + 3.0
    while not state.calls and time.monotonic() < deadline:
        rclpy.spin_once(client, timeout_sec=0.02)
    assert state.calls
    result = request(speech_request())
    assert not result["success"] and "busy" in result["message"]
    rclpy.spin_until_future_complete(client, running, timeout_sec=3.0)
    assert running.done() and running.result().success
    assert len(state.calls) == 1


def test_installed_cli_sends_call_values_to_resident(resident):
    _, state, _, _ = resident
    invocations = [
        (["speak", "--text", "CLI text", "--language", "en-US", "--priority", "40"], 0),
        (["motion", "--name", "wave", "--target", "right"], 1),
        (["motion", "--name", "wave", "--target", "right", "--allow-motion"], 0),
    ]
    for args, code in invocations:
        result = subprocess.run(
            ["ros2", "run", "robot_interaction_demo", "runtime-demo", *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == code, result.stderr
        payload, _ = json.JSONDecoder().raw_decode(result.stdout)
        assert payload["success"] is (code == 0)
    assert state.calls[0][1].text == "CLI text"
    assert state.calls[0][1].language == "en-US"
    assert state.calls[1][1].name == "wave"
