"""Thin supervised client for the canonical PickObject action pipeline."""

from __future__ import annotations

import argparse
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from unique_identifier_msgs.msg import UUID

from ibrobot_msgs.action import PickObject

_MODE_BY_NAME = {
    "execute": PickObject.Goal.MODE_EXECUTE,
    "plan_only": PickObject.Goal.MODE_PLAN_ONLY,
    "observe_only": PickObject.Goal.MODE_OBSERVE_ONLY,
}


class PickActionClient(Node):
    """Send one supervised goal to the production pick executor."""

    def __init__(self, action_name: str) -> None:
        super().__init__("pick_action_client")
        self._action_name = str(action_name)
        self._client = ActionClient(self, PickObject, self._action_name)

    @staticmethod
    def _print_feedback(message) -> None:
        feedback = message.feedback
        print(
            f"PICK_FEEDBACK phase={feedback.phase} progress={float(feedback.progress):.3f} "
            f"attempt={int(feedback.attempt)} detail={feedback.detail}",
            flush=True,
        )

    def execute(
        self,
        *,
        task_id: str,
        target_query: str,
        timeout_sec: float,
        mode: str,
        release_after_success: bool,
        release_drop_height_m: float,
        ready_timeout_sec: float,
        goal_response_timeout_sec: float,
    ):
        if mode not in _MODE_BY_NAME:
            raise ValueError(f"unknown pick mode: {mode}")
        if not self._client.wait_for_server(timeout_sec=max(0.1, ready_timeout_sec)):
            raise RuntimeError(f"PickObject action server is unavailable: {self._action_name}")

        goal = PickObject.Goal()
        goal.task_id = str(task_id)
        goal.target_query = str(target_query)
        goal.timeout_sec = float(timeout_sec)
        goal.mode = int(_MODE_BY_NAME[mode])
        goal.release_after_success = bool(release_after_success)
        goal.release_drop_height_m = float(release_drop_height_m)
        print(
            f"PICK_ACTION_SEND action={self._action_name} task_id={goal.task_id} target={goal.target_query!r} "
            f"mode={mode} timeout_s={goal.timeout_sec:.1f} release_after_success={goal.release_after_success} "
            f"release_drop_height_m={goal.release_drop_height_m:.4f}",
            flush=True,
        )

        feedback_observed = False

        def feedback_callback(message) -> None:
            nonlocal feedback_observed
            feedback_observed = True
            self._print_feedback(message)

        goal_uuid = UUID(uuid=list(uuid.uuid4().bytes))
        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=feedback_callback,
            goal_uuid=goal_uuid,
        )
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=max(0.1, goal_response_timeout_sec))
        if not send_future.done():
            # The request and feedback/result channels are separate DDS endpoints. Under discovery
            # pressure a server may accept and execute a goal even when its SendGoal response is
            # dropped. Never resend the motion goal: query the original UUID instead.
            send_future.cancel()
            recovered_response = PickObject.Impl.SendGoalService.Response()
            recovered_response.accepted = True
            goal_handle = ClientGoalHandle(self._client, goal_uuid, recovered_response)
            print(
                f"PICK_ACTION_GOAL_RESPONSE_RECOVERY action={self._action_name} "
                f"timeout_s={max(0.1, goal_response_timeout_sec):.1f} feedback_observed={feedback_observed}",
                flush=True,
            )
        else:
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError(f"PickObject goal was rejected by {self._action_name}")

        result_future = goal_handle.get_result_async()
        wait_started = time.monotonic()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=max(1.0, timeout_sec + 10.0))
        if not result_future.done():
            goal_handle.cancel_goal_async()
            raise RuntimeError(f"PickObject result timed out after {time.monotonic() - wait_started:.1f}s")
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError("PickObject action returned no result")
        result = wrapped_result.result
        print(
            f"PICK_ACTION_RESULT success={bool(result.success)} error_code={result.error_code or '-'} "
            f"candidate={int(result.candidate_index)} attempts={int(result.attempts)} "
            f"verification_status={int(result.verification_status)} "
            f"verification_confidence={float(result.verification_confidence):.3f} "
            f"released_after_success={bool(result.released_after_success)} "
            f"debug_output_dir={result.debug_output_dir or '-'} "
            f"pipeline_timings_json={result.pipeline_timings_json or '{}'} message={result.message}",
            flush=True,
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Runtime text target passed to PickObject")
    parser.add_argument("--task-id", default="", help="Optional task identifier")
    parser.add_argument("--action-name", default="/manipulation/execute_pick")
    parser.add_argument("--mode", choices=tuple(_MODE_BY_NAME), default="execute")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--goal-response-timeout-s", type=float, default=10.0)
    parser.add_argument("--release-after-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--release-drop-height-m", type=float, default=0.015)
    return parser


def main(args=None) -> None:
    parsed = build_parser().parse_args(args)
    rclpy.init(args=None)
    node = PickActionClient(parsed.action_name)
    try:
        action_started = time.perf_counter()
        result = node.execute(
            task_id=parsed.task_id,
            target_query=parsed.prompt,
            timeout_sec=parsed.timeout_s,
            mode=parsed.mode,
            release_after_success=parsed.release_after_success,
            release_drop_height_m=parsed.release_drop_height_m,
            ready_timeout_sec=parsed.ready_timeout_s,
            goal_response_timeout_sec=parsed.goal_response_timeout_s,
        )
        action_elapsed = time.perf_counter() - action_started
        print(
            f"PICK_ACTION_TIMING wall_time_s={action_elapsed:.6f} "
            f"pipeline_timings_json={result.pipeline_timings_json or '{}'}",
            flush=True,
        )
        print(f"FLOW_RESULT success={bool(result.success)} error={result.error_code or '-'}", flush=True)
        if not result.success:
            raise RuntimeError(result.message or result.error_code or "pick failed")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
