"""Launch lifecycle tests for the production benchmark graph."""

from __future__ import annotations

from types import SimpleNamespace

from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.launch_builders.benchmark import (
    _benchmark_process_exit_handler,
    _with_benchmark_lifecycle,
)


def test_benchmark_lifecycle_wraps_each_required_node_with_process_exit_handler() -> None:
    node = Node(package="benchmark_runtime", executable="benchmark_evaluator_node", name="benchmark_evaluator")

    actions = _with_benchmark_lifecycle(node, "evaluator")

    assert actions[0] is node
    assert isinstance(actions[1], RegisterEventHandler)
    event_handler = actions[1].event_handler
    assert isinstance(event_handler, OnProcessExit)
    assert event_handler._OnActionEventBase__action_matcher is node


def test_uncoordinated_evaluator_exit_requests_global_shutdown() -> None:
    on_exit = _benchmark_process_exit_handler("benchmark:evaluator")

    result = on_exit(SimpleNamespace(returncode=0), SimpleNamespace(is_shutdown=False))

    assert isinstance(result, EmitEvent)
    assert isinstance(result.event, Shutdown)
    assert result.event.reason == "benchmark evaluation completed"


def test_coordinated_lane_evaluator_exit_waits_for_coordinator() -> None:
    on_exit = _benchmark_process_exit_handler("benchmark:evaluator", coordinated=True)

    assert on_exit(SimpleNamespace(returncode=0), SimpleNamespace(is_shutdown=False)) is None


def test_coordinator_exit_requests_global_shutdown() -> None:
    on_exit = _benchmark_process_exit_handler("coordinator")

    result = on_exit(SimpleNamespace(returncode=0), SimpleNamespace(is_shutdown=False))

    assert isinstance(result, EmitEvent)
    assert isinstance(result.event, Shutdown)
    assert result.event.reason == "benchmark evaluation completed"


def test_required_benchmark_failure_requests_global_shutdown() -> None:
    on_exit = _benchmark_process_exit_handler("benchmark:environment")

    result = on_exit(SimpleNamespace(returncode=17), SimpleNamespace(is_shutdown=False))

    assert isinstance(result, EmitEvent)
    assert isinstance(result.event, Shutdown)
    assert result.event.reason == "benchmark required process 'benchmark:environment' exited with return code 17"


def test_exit_handler_does_not_reenter_shutdown_callback() -> None:
    on_exit = _benchmark_process_exit_handler("evaluator")

    assert on_exit(SimpleNamespace(returncode=0), SimpleNamespace(is_shutdown=True)) is None
