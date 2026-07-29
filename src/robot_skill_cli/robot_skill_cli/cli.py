"""Argument parsing and command dispatch for robot-skill."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from collections.abc import Sequence
from typing import Any

from robot_skill_cli import __version__
from robot_skill_cli.output import (
    EXIT_GATEWAY_REJECTED,
    EXIT_INVALID_INPUT,
    EXIT_ROS_UNAVAILABLE,
    EXIT_SIGINT,
    EXIT_SIGTERM,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    error_envelope,
    feedback_event,
    json_dumps,
    result_event,
    success_envelope,
)


class _CliArgumentError(ValueError):
    pass


class _CommandError(RuntimeError):
    def __init__(self, code: str, message: str, *, exit_code: int) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(message)


class _CommandExit:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CliArgumentError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="robot-skill", description="Inspect and execute IB-Robot capabilities.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config-name", help="robot_config name")
    config_group.add_argument("--config-path", help="explicit robot_config YAML path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-skills", help="list enabled high-level skills")
    describe_parser = subparsers.add_parser("describe", help="describe one enabled skill")
    describe_parser.add_argument("skill")
    subparsers.add_parser("list-poses", help="list configured named poses")
    subparsers.add_parser("status", help="query Capability Gateway status")
    validate_parser = subparsers.add_parser("validate", help="validate one skill without executing it")
    validate_parser.add_argument("skill")
    _add_skill_parameters(validate_parser)
    execute_parser = subparsers.add_parser("execute", help="execute one high-level skill")
    execute_parser.add_argument("skill")
    execute_parser.add_argument("--task-id", required=True)
    _add_skill_parameters(execute_parser)
    cancel_parser = subparsers.add_parser("cancel", help="cancel one active task by task ID")
    cancel_parser.add_argument("--task-id", required=True)
    return parser


def _add_skill_parameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-name")
    parser.add_argument("--place-name")
    parser.add_argument("--motion-direction")
    parser.add_argument("--motion-distance", type=float)
    parser.add_argument("--timeout-sec", type=float)


def _run_catalog_command(args: argparse.Namespace, view: dict[str, Any]) -> dict:
    from robot_skill_cli.catalog import describe_skill, list_poses, list_skills

    if args.command == "list-skills":
        return list_skills(view)
    if args.command == "describe":
        return describe_skill(view, args.skill)
    if args.command == "list-poses":
        return list_poses(view)
    raise _CliArgumentError(f"unsupported command: {args.command}")


def _create_bridge(transport):
    from robot_skill_cli.ros_bridge import RosBridge

    return RosBridge(
        status_service=transport.status_service,
        validate_skill_service=transport.validate_skill_service,
        skill_action=transport.skill_action_name,
    )


def _validate_schema(skill: dict[str, Any], args: argparse.Namespace) -> None:
    parameters = skill["parameters"]
    properties = parameters["properties"]
    required = set(parameters["required"])
    values = {
        "target_name": args.target_name,
        "place_name": args.place_name,
        "motion_direction": args.motion_direction,
        "motion_distance": args.motion_distance,
    }
    for name, value in values.items():
        if name not in properties:
            if value not in (None, "", 0, 0.0):
                raise _CliArgumentError(f"{name} is not accepted by skill {skill['name']}")
            continue
        schema = properties[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            if name in required:
                raise _CliArgumentError(f"{name} is required for skill {skill['name']}")
            if value is None:
                continue
        if schema["type"] == "string" and "enum" in schema and value.strip().lower() not in schema["enum"]:
            raise _CliArgumentError(f"{name} must be one of: {', '.join(schema['enum'])}")
        if schema["type"] == "number" and (
            isinstance(value, bool) or not math.isfinite(value) or value <= schema.get("exclusiveMinimum", -math.inf)
        ):
            raise _CliArgumentError(f"{name} must be a finite number greater than zero")


def _validate_timeout(status: dict[str, Any], timeout_sec: float | None) -> float:
    effective_timeout = status["default_skill_timeout_sec"] if timeout_sec is None else timeout_sec
    if not math.isfinite(effective_timeout) or effective_timeout <= 0.0:
        raise _CliArgumentError("timeout_sec must be a finite number greater than zero")
    if effective_timeout > status["task_budget_sec"]:
        raise _CliArgumentError("timeout_sec must not exceed the Gateway task budget")
    return effective_timeout


def _prepare_request(
    args: argparse.Namespace, context, bridge
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    from embodied_common.skill_request import canonical_skill_payload, skill_payload_hash
    from robot_skill_cli.catalog import describe_skill

    status = bridge.get_status(
        task_id="",
        payload_hash="",
        timeout_sec=context.view["timeout_policy"]["rpc_timeout_sec"],
    )
    if status["config_digest"] != context.view["capability_digest"]:
        raise _CommandError(
            "CONFIG_MISMATCH",
            "local capability digest does not match the Gateway",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    skill = describe_skill(context.view, args.skill)
    _validate_schema(skill, args)
    effective_timeout = _validate_timeout(status, args.timeout_sec)
    capability = next((item for item in status["capabilities"] if item["name"] == skill["name"]), None)
    if capability is None or not capability["ready"]:
        reason = capability["reason"] if capability is not None and capability["reason"] else "CAPABILITY_NOT_READY"
        reason = str(reason)
        error_code, separator, _message = reason.partition(":")
        error_code = error_code.strip()
        if not error_code:
            error_code = "CAPABILITY_NOT_READY"
            reason = error_code
        elif not separator:
            reason = error_code
        raise _CommandError(error_code, reason, exit_code=EXIT_GATEWAY_REJECTED)
    payload = canonical_skill_payload(
        skill["name"],
        target_name=args.target_name,
        place_name=args.place_name,
        motion_direction=args.motion_direction,
        motion_distance=0.0 if args.motion_distance is None else args.motion_distance,
        timeout_sec=effective_timeout,
        default_timeout_sec=status["default_skill_timeout_sec"],
    )
    validation = bridge.validate_skill(payload, timeout_sec=status["rpc_timeout_sec"])
    if not validation["allowed"]:
        raise _CommandError("SAFETY_REJECTED", validation["reason"], exit_code=EXIT_GATEWAY_REJECTED)
    return payload, skill_payload_hash(payload), status, validation


def _run_validate(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    payload, payload_hash, _status, validation = _prepare_request(args, context, bridge)
    return {
        "allowed": True,
        "reason": validation["reason"],
        "payload": payload,
        "payload_hash": payload_hash,
    }


def _identity_status(bridge, task_id: str, payload_hash: str, timeout_sec: float) -> dict[str, Any]:
    return bridge.get_status(task_id=task_id, payload_hash=payload_hash, timeout_sec=timeout_sec)


def _public_result(result) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "error_code": str(result.error_code),
        "message": str(result.message),
        "executed_step_count": len(result.executed_primitives),
    }


def _print_result(task_id: str, payload_hash: str, data: dict[str, Any]) -> None:
    print(json_dumps(result_event(task_id, payload_hash, **data)), flush=True)


def _result_exit_code(data: dict[str, Any]) -> int:
    if data["success"]:
        return EXIT_SUCCESS
    if "TIMEOUT" in data["error_code"]:
        return EXIT_TIMEOUT
    return EXIT_GATEWAY_REJECTED


def _cancel_pending_task(bridge, task_id: str, rpc_timeout: float) -> bool:
    deadline = time.monotonic() + rpc_timeout
    cancel_confirmed = False
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            status = bridge.get_status(task_id=task_id, payload_hash="", timeout_sec=remaining)
            if status["request_error_code"]:
                return False
            if status["request_state"] == "terminal":
                return True
            if status["request_state"] == "active" or not cancel_confirmed:
                cancel = bridge.cancel_task(task_id, timeout_sec=remaining)
                cancel_confirmed = bool(cancel["accepted"])
        except Exception as exc:
            import sys

            sys.stderr.write(f"cancel polling failed: {exc}\n")
            return False
        time.sleep(0.02)
    return False


def _run_execute(args: argparse.Namespace, context, bridge) -> _CommandExit:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    payload, payload_hash, status, _validation = _prepare_request(args, context, bridge)
    rpc_timeout = status["rpc_timeout_sec"]
    output_lock = threading.Lock()
    terminal_output = {"written": False}

    def emit_feedback(feedback: dict[str, str]) -> None:
        with output_lock:
            if terminal_output["written"]:
                return
            print(
                json_dumps(
                    feedback_event(
                        task_id,
                        payload_hash,
                        feedback["state"],
                        feedback["detail"],
                    )
                ),
                flush=True,
            )

    def emit_result(data: dict[str, Any]) -> None:
        with output_lock:
            if terminal_output["written"]:
                return
            terminal_output["written"] = True
            _print_result(task_id, payload_hash, data)

    def transport_failure(message: str) -> _CommandExit:
        if _cancel_pending_task(bridge, task_id, rpc_timeout):
            data = {
                "success": False,
                "error_code": "ROS_UNAVAILABLE",
                "message": message,
                "executed_step_count": 0,
            }
            exit_code = EXIT_ROS_UNAVAILABLE
        else:
            data = {
                "success": False,
                "error_code": "SKILL_CANCEL_TIMEOUT",
                "message": "robot stop state is unknown",
                "executed_step_count": 0,
            }
            exit_code = EXIT_TIMEOUT
        emit_result(data)
        return _CommandExit(exit_code)

    def read_terminal_result(result_future) -> dict[str, Any] | None:
        try:
            wrapped_result = result_future.result()
            if wrapped_result is None or wrapped_result.result is None:
                return None
            return _public_result(wrapped_result.result)
        except Exception:
            return None

    identity_status = _identity_status(bridge, task_id, payload_hash, rpc_timeout)
    if identity_status["request_error_code"]:
        error_code = identity_status["request_error_code"]
        emit_result(
            {
                "success": False,
                "error_code": error_code,
                "message": error_code,
                "executed_step_count": 0,
            },
        )
        return _CommandExit(EXIT_GATEWAY_REJECTED)
    if not bridge.wait_for_skill_server(timeout_sec=rpc_timeout):
        raise _CommandError("SERVER_UNAVAILABLE", "skill action server unavailable", exit_code=EXIT_ROS_UNAVAILABLE)

    interrupted = {"signal": None}

    def handle_signal(signum, _frame) -> None:
        interrupted["signal"] = signum

    previous_handlers = {
        signal.SIGINT: signal.signal(signal.SIGINT, handle_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, handle_signal),
    }
    try:
        goal_future = bridge.send_skill_goal(
            payload,
            task_id=task_id,
            feedback_callback=emit_feedback,
        )
        if not bridge.wait_future(goal_future, timeout_sec=rpc_timeout):
            converged = _cancel_pending_task(bridge, task_id, rpc_timeout)
            if converged and interrupted["signal"] is not None:
                data = {
                    "success": False,
                    "error_code": "SKILL_CANCELLED",
                    "message": "task reached terminal after signal cancellation",
                    "executed_step_count": 0,
                }
                exit_code = EXIT_SIGINT if interrupted["signal"] == signal.SIGINT else EXIT_SIGTERM
            elif converged:
                data = {
                    "success": False,
                    "error_code": "RESULT_TIMEOUT",
                    "message": "task reached terminal after goal response timeout",
                    "executed_step_count": 0,
                }
                exit_code = EXIT_TIMEOUT
            else:
                data = {
                    "success": False,
                    "error_code": "SKILL_CANCEL_TIMEOUT",
                    "message": "robot stop state is unknown",
                    "executed_step_count": 0,
                }
                exit_code = EXIT_TIMEOUT
            emit_result(data)
            return _CommandExit(exit_code)
        try:
            goal_handle = goal_future.result()
        except Exception:
            return transport_failure("goal response unavailable")
        if goal_handle is None or not goal_handle.accepted:
            rejected_status = bridge.get_status(
                task_id=task_id,
                payload_hash=payload_hash,
                timeout_sec=rpc_timeout,
            )
            error_code = rejected_status["request_error_code"] or "GOAL_REJECTED"
            emit_result(
                {
                    "success": False,
                    "error_code": error_code,
                    "message": error_code,
                    "executed_step_count": 0,
                },
            )
            return _CommandExit(EXIT_GATEWAY_REJECTED)

        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            return transport_failure("result request unavailable")
        deadline = time.monotonic() + payload["timeout_sec"] + rpc_timeout
        while not result_future.done():
            if interrupted["signal"] is not None:
                signal_number = interrupted["signal"]
                if not bridge.cancel_goal(goal_handle, result_future, timeout_sec=rpc_timeout):
                    data = {
                        "success": False,
                        "error_code": "SKILL_CANCEL_TIMEOUT",
                        "message": "robot stop state is unknown",
                        "executed_step_count": 0,
                    }
                    emit_result(data)
                    return _CommandExit(EXIT_TIMEOUT)
                data = read_terminal_result(result_future)
                if data is None:
                    return transport_failure("terminal result unavailable")
                emit_result(data)
                exit_code = EXIT_SIGINT if signal_number == signal.SIGINT else EXIT_SIGTERM
                return _CommandExit(exit_code)
            if time.monotonic() >= deadline:
                if bridge.cancel_goal(goal_handle, result_future, timeout_sec=rpc_timeout):
                    data = read_terminal_result(result_future)
                    if data is None:
                        return transport_failure("terminal result unavailable")
                else:
                    data = {
                        "success": False,
                        "error_code": "SKILL_CANCEL_TIMEOUT",
                        "message": "robot stop state is unknown",
                        "executed_step_count": 0,
                    }
                emit_result(data)
                return _CommandExit(EXIT_TIMEOUT)
            time.sleep(0.02)

        data = read_terminal_result(result_future)
        if data is None:
            return transport_failure("terminal result unavailable")
        emit_result(data)
        return _CommandExit(_result_exit_code(data))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _task_status(bridge, task_id: str, timeout_sec: float) -> dict[str, Any]:
    status = bridge.get_status(task_id=task_id, payload_hash="", timeout_sec=timeout_sec)
    if status["request_error_code"] in {"DUPLICATE_TASK_ID", "TASK_ID_CONFLICT"}:
        raise _CommandError(
            status["request_error_code"],
            "task-ID-only lookup returned an identity error",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    return status


def _run_cancel(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    initial_timeout = context.view["timeout_policy"]["rpc_timeout_sec"]
    status = _task_status(bridge, task_id, initial_timeout)
    if status["request_state"] == "terminal":
        return {"task_id": task_id, "already_terminal": True, "status": status}
    if status["request_state"] != "active":
        raise _CommandError("GOAL_NOT_FOUND", f"active task not found: {task_id}", exit_code=EXIT_GATEWAY_REJECTED)

    rpc_timeout = status["rpc_timeout_sec"]
    cancel_result = bridge.cancel_task(task_id, timeout_sec=rpc_timeout)
    deadline = time.monotonic() + rpc_timeout
    while time.monotonic() < deadline:
        status = _task_status(bridge, task_id, rpc_timeout)
        if status["request_state"] == "terminal":
            return {
                "task_id": task_id,
                "already_terminal": False,
                "cancel": cancel_result,
                "status": status,
            }
        if status["request_state"] != "active":
            raise _CommandError(
                "GOAL_NOT_FOUND", f"task record disappeared: {task_id}", exit_code=EXIT_GATEWAY_REJECTED
            )
        time.sleep(0.02)
    raise _CommandError(
        "SKILL_CANCEL_TIMEOUT",
        "robot stop state is unknown",
        exit_code=EXIT_TIMEOUT,
    )


def _run_runtime_command(args: argparse.Namespace, context, transport) -> dict[str, Any] | _CommandExit:
    from robot_skill_cli.ros_bridge import BridgeError

    bridge = _create_bridge(transport)
    if not bridge.start():
        raise BridgeError("ROS_UNAVAILABLE", "failed to initialize ROS bridge", exit_code=EXIT_ROS_UNAVAILABLE)
    try:
        if args.command == "status":
            return bridge.get_status(timeout_sec=context.view["timeout_policy"]["rpc_timeout_sec"])
        if args.command == "validate":
            return _run_validate(args, context, bridge)
        if args.command == "execute":
            return _run_execute(args, context, bridge)
        if args.command == "cancel":
            return _run_cancel(args, context, bridge)
        raise _CliArgumentError(f"unsupported command: {args.command}")
    finally:
        bridge.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    command = "unknown"
    try:
        args = parser.parse_args(argv)
        command = args.command
        from robot_skill_cli.catalog import load_catalog_context, load_runtime_context

        if command in {"list-skills", "describe", "list-poses"}:
            context = load_catalog_context(config_name=args.config_name, config_path=args.config_path)
            data = _run_catalog_command(args, context.view)
        else:
            context, transport = load_runtime_context(config_name=args.config_name, config_path=args.config_path)
            data = _run_runtime_command(args, context, transport)
    except _CommandError as exc:
        print(json_dumps(error_envelope(command, exc.code, str(exc))))
        return exc.exit_code
    except (_CliArgumentError, FileNotFoundError, ValueError) as exc:
        print(json_dumps(error_envelope(command, getattr(exc, "code", "INVALID_ARGUMENT"), str(exc))))
        return EXIT_INVALID_INPUT
    except Exception as exc:
        from robot_skill_cli.ros_bridge import BridgeError

        if isinstance(exc, BridgeError):
            print(json_dumps(error_envelope(command, exc.code, str(exc))))
            return exc.exit_code
        raise

    if isinstance(data, _CommandExit):
        return data.exit_code
    print(json_dumps(success_envelope(command, data)))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
