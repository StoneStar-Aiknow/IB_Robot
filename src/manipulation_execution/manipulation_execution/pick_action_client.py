"""Supervised client for the production PickObject pipeline.

This is a hardware bring-up client, not a second grasp implementation.  It
reads the live Gateway registry and sends one explicitly supervised goal to
the production executor.  Hermes continues to use the delegated SkillCommand
path and never sets ``supervised_direct``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from unique_identifier_msgs.msg import UUID

from ibrobot_msgs.action import PickObject
from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot
from skill_catalog.consumer import CatalogIdentity, verify_snapshot_response

_READINESS_REQUEST_TIMEOUT_S = 5.0


def resolve_supervised_timeouts(requested_timeout_sec: float, task_budget_sec: float) -> tuple[float, float]:
    """Keep the pick timeout below the root budget reserved by the Gateway."""
    requested_timeout_sec = float(requested_timeout_sec)
    task_budget_sec = float(task_budget_sec)
    if not math.isfinite(requested_timeout_sec) or requested_timeout_sec <= 0.0:
        raise RuntimeError("timeout must be positive and finite")
    if not math.isfinite(task_budget_sec) or task_budget_sec <= 0.0:
        raise RuntimeError("Gateway task budget must be positive and finite")
    if requested_timeout_sec >= task_budget_sec:
        raise RuntimeError(
            f"timeout must be less than the Gateway task budget ({task_budget_sec:.1f}s) to leave dispatch headroom"
        )
    return requested_timeout_sec, task_budget_sec


def resolve_supervised_task_id(requested_task_id: str, *, exact: bool = False, unique_suffix: str = "") -> str:
    """Give each supervised hardware attempt a fresh Gateway identity."""
    prefix = str(requested_task_id).strip()
    if exact:
        if not prefix:
            raise RuntimeError("--exact-task-id requires a non-empty --task-id")
        return prefix
    suffix = unique_suffix or uuid.uuid4().hex
    return f"{prefix or 'supervised-pick'}-{suffix}"


class PickActionClient(Node):
    """Send one supervised goal after loading the live registry identity."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pick_action_client")
        self._args = args
        self._pick_client = ActionClient(self, PickObject, args.action_name)
        self._status_client = self.create_client(GetSkillGatewayStatus, args.status_service)
        self._snapshot_client = self.create_client(GetSkillSnapshot, args.snapshot_service)

    @staticmethod
    def _wait_future(node: Node, future, timeout_sec: float, label: str):
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(0.1, timeout_sec))
        if not future.done():
            future.cancel()
            raise RuntimeError(f"{label} timed out or returned no response")
        try:
            response = future.result()
        except Exception as exc:
            raise RuntimeError(f"{label} request failed: {exc}") from exc
        if response is None:
            raise RuntimeError(f"{label} timed out or returned no response")
        return response

    def _call_readiness_service(self, client, request_factory, timeout_sec: float, label: str):
        """Retry an idempotent startup query within one bounded readiness window."""
        timeout_sec = max(0.1, float(timeout_sec))
        deadline = time.monotonic() + timeout_sec
        max_attempts = max(1, math.ceil(timeout_sec / _READINESS_REQUEST_TIMEOUT_S))
        attempts = 0
        service_seen = False
        last_error = ""
        while attempts < max_attempts:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if not client.wait_for_service(timeout_sec=min(1.0, remaining)):
                continue
            service_seen = True
            attempts += 1
            request_timeout = min(_READINESS_REQUEST_TIMEOUT_S, remaining)
            try:
                return self._wait_future(
                    self,
                    client.call_async(request_factory()),
                    request_timeout,
                    label,
                )
            except RuntimeError as exc:
                last_error = str(exc)
                if attempts < max_attempts and time.monotonic() < deadline:
                    self.get_logger().warning(
                        f"Readiness query failed for {label} (attempt {attempts}/{max_attempts}); retrying: {exc}"
                    )
        if not service_seen:
            raise RuntimeError(f"Gateway service is unavailable after {timeout_sec:.1f}s: {label}")
        detail = f": {last_error}" if last_error else ""
        raise RuntimeError(f"{label} failed after {attempts} readiness attempts{detail}")

    def _load_runtime_identity(self, task_id: str) -> tuple[dict, dict, float]:
        args = self._args

        def status_request():
            request = GetSkillGatewayStatus.Request()
            request.schema_version = 1
            request.task_id = task_id
            return request

        status = self._call_readiness_service(
            self._status_client,
            status_request,
            args.ready_timeout_s,
            args.status_service,
        )
        if not status.control_plane_ready:
            raise RuntimeError(
                f"Gateway control plane is not ready: {status.control_plane_state} "
                f"{status.control_plane_error_code or ''}".strip()
            )
        if not status.motion_authorized:
            raise RuntimeError("Gateway motion is not authorized; launch the pipeline with authorize_motion:=true")
        capability = next((item for item in status.capabilities if item.name == "pick_object"), None)
        if capability is None or not capability.ready:
            reason = capability.reason if capability is not None else "pick_object capability is unavailable"
            raise RuntimeError(f"Gateway pick_object capability is not ready: {reason}")

        def snapshot_request():
            request = GetSkillSnapshot.Request()
            request.schema_version = 1
            request.registry_epoch = status.registry_epoch
            request.generation = status.registry_generation
            return request

        snapshot = self._call_readiness_service(
            self._snapshot_client,
            snapshot_request,
            args.ready_timeout_s,
            args.snapshot_service,
        )
        verify_snapshot_response(
            snapshot,
            CatalogIdentity(status.registry_epoch, int(status.registry_generation), status.registry_digest),
        )
        payload = json.loads(snapshot.snapshot_json)
        executors = payload["registry_preimage"]["delegated_executors"]
        executor = next((item for item in executors if item.get("name") == "grasp_pipeline"), None)
        if executor is None:
            raise RuntimeError("Gateway registry has no grasp_pipeline executor")
        identity = {
            key: executor.get(key, "")
            for key in (
                "name",
                "contract_version",
                "endpoint_kind",
                "endpoint_name",
                "configuration_digest",
                "model_deployment_name",
                "model_fingerprint",
                "model_bundle_digest",
            )
        }
        return (
            {
                "epoch": status.registry_epoch,
                "generation": int(status.registry_generation),
                "digest": status.registry_digest,
                "budget": float(status.task_budget_sec),
            },
            identity,
            float(status.rpc_timeout_sec),
        )

    @staticmethod
    def _fill_identity(message, identity: dict) -> None:
        message.schema_version = 1
        for field_name, value in identity.items():
            setattr(message, field_name, value)

    def execute(self) -> object:
        args = self._args
        task_id = resolve_supervised_task_id(args.task_id, exact=args.exact_task_id)
        registry, identity, _rpc_timeout = self._load_runtime_identity(task_id)
        timeout_sec, task_budget_sec = resolve_supervised_timeouts(args.timeout_s, registry["budget"])
        if not self._pick_client.wait_for_server(timeout_sec=args.ready_timeout_s):
            raise RuntimeError(f"PickObject action server is unavailable: {args.action_name}")

        goal = PickObject.Goal()
        goal.dispatch_binding.schema_version = 1
        goal.dispatch_binding.task_id = task_id
        goal.dispatch_binding.root_task_id = task_id
        goal.dispatch_binding.expected_registry_epoch = registry["epoch"]
        goal.dispatch_binding.expected_registry_generation = registry["generation"]
        goal.dispatch_binding.expected_registry_digest = registry["digest"]
        now = time.time()
        goal.dispatch_binding.task_budget.schema_version = 1
        goal.dispatch_binding.task_budget.started_at.sec = int(now)
        goal.dispatch_binding.task_budget.started_at.nanosec = int((now - int(now)) * 1_000_000_000)
        deadline = now + task_budget_sec
        goal.dispatch_binding.task_budget.deadline.sec = int(deadline)
        goal.dispatch_binding.task_budget.deadline.nanosec = int((deadline - int(deadline)) * 1_000_000_000)
        self._fill_identity(goal.expected_executor, identity)
        goal.target_query = args.prompt
        goal.timeout_sec = timeout_sec
        goal.supervised_direct = True
        goal.mode = PickObject.Goal.MODE_EXECUTE
        goal.release_after_success = bool(args.release_after_success)
        print(
            f"PICK_ACTION_SEND action={args.action_name} task_id={task_id} target={args.prompt!r} "
            f"timeout_s={timeout_sec:.1f} supervised_direct=true",
            flush=True,
        )

        def feedback_callback(message) -> None:
            feedback = message.feedback
            print(
                f"PICK_FEEDBACK phase={feedback.phase} progress={float(feedback.progress):.3f} "
                f"attempt={int(feedback.attempt)} detail={feedback.detail}",
                flush=True,
            )

        goal_uuid = UUID(uuid=list(uuid.uuid4().bytes))
        send_future = self._pick_client.send_goal_async(goal, feedback_callback=feedback_callback, goal_uuid=goal_uuid)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=max(0.1, args.goal_response_timeout_s))
        if not send_future.done():
            send_future.cancel()
            recovered_response = PickObject.Impl.SendGoalService.Response()
            recovered_response.accepted = True
            goal_handle = ClientGoalHandle(self._pick_client, goal_uuid, recovered_response)
            print("PICK_ACTION_GOAL_RESPONSE_RECOVERY feedback may still be active", flush=True)
        else:
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError(f"PickObject goal was rejected by {args.action_name}")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=max(1.0, timeout_sec + 10.0))
        if not result_future.done():
            goal_handle.cancel_goal_async()
            raise RuntimeError(f"PickObject result timed out after {timeout_sec:.1f}s")
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError("PickObject action returned no result")
        result = wrapped_result.result
        print(
            f"PICK_ACTION_RESULT success={bool(result.success)} error_code={result.error_code or '-'} "
            f"candidate={int(result.candidate_index)} attempts={int(result.attempts)} "
            f"verification_status={int(result.verification_status)} "
            f"pipeline_timings_json={result.pipeline_timings_json or '{}'} message={result.message}",
            flush=True,
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", "--target-name", required=True)
    parser.add_argument(
        "--task-id",
        default="",
        help="human-readable task ID prefix; a unique suffix is appended for each supervised attempt",
    )
    parser.add_argument(
        "--exact-task-id",
        action="store_true",
        help="send --task-id verbatim; repeated executions may be rejected by the Gateway ledger",
    )
    parser.add_argument("--action-name", default="/manipulation/execute_pick")
    parser.add_argument("--status-service", default="/embodied/get_skill_gateway_status")
    parser.add_argument("--snapshot-service", default="/embodied/get_skill_snapshot")
    parser.add_argument("--timeout-s", type=float, default=230.0)
    parser.add_argument("--ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--goal-response-timeout-s", type=float, default=10.0)
    parser.add_argument("--release-after-success", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(args=None) -> None:
    parsed = build_parser().parse_args(args)
    rclpy.init(args=None)
    node = PickActionClient(parsed)
    try:
        result = node.execute()
        print(f"FLOW_RESULT success={bool(result.success)} error={result.error_code or '-'}", flush=True)
        if not result.success:
            raise RuntimeError(result.message or result.error_code or "pick failed")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
