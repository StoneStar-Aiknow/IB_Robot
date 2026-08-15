"""Functional-verification driver for the interactive closed loop.

Drives ``InteractiveController`` against the live Capability Gateway (real
``RosBridge``) to prove the closed loop end to end: discover,
reject-out-of-catalog, present then execute immediately, ``别动`` stop to a
definite terminal, and ``继续`` fresh-state continuation.

Install/run inside the built workspace::

    source .shrc_local && export ROS_DOMAIN_ID=56 && source install/setup.sh
    robot-skill-closed-loop --config-name so101_single_arm \
        --raw-command '点个头' --skill nod_yes \
        --stop-after-sec 1.5 \
        --continue-command '继续挥挥手' --continue-skill wave_hand

Breakpoint resume is intentionally rejected on this baseline because the
Gateway has no server-owned continuation admission contract. A fresh
continuation may still be requested after a definite cancellation::

    robot-skill-closed-loop --config-name so101_single_arm \
        --raw-command '点头两次再摇头最后撒娇' --steps nod_yes,nod_yes,shake_no,act_cute \
        --stop-after-sec 3.0 \
        --continue-command '继续挥挥手' --continue-skill wave_hello

``--stop-after-sec`` spawns a daemon thread that calls ``request_stop()`` after
the delay, simulating the operator saying ``别动`` mid-execution. The driver
then requires a definite terminal before issuing the continuation.
"""

from __future__ import annotations

import argparse
import threading
from typing import Any

from robot_skill_cli.catalog import load_runtime_context
from robot_skill_cli.interactive_control import (
    InteractiveControlError,
    InteractiveController,
)
from robot_skill_cli.output import json_dumps
from robot_skill_cli.ros_bridge import RosBridge

_EXIT_EXECUTION_FAILED = 13
_EXIT_MOTION_UNKNOWN = 15


def _workflow_step(args: argparse.Namespace, skill: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_name": skill,
        "target_name": args.target_name or "",
        "place_name": args.place_name or "",
        "motion_direction": args.motion_direction or "",
        "motion_distance": args.motion_distance or 0.0,
        "timeout_sec": args.timeout_sec or 0.0,
    }


def _emit(label: str, payload: Any) -> None:
    if isinstance(payload, dict | list):
        print(json_dumps({"event": label, "data": payload}), flush=True)
    else:
        print(f"[{label}] {payload}", flush=True)


def _terminal_exit_code(terminal: dict[str, Any]) -> int:
    state = str(terminal.get("state", ""))
    if state == "succeeded":
        return 0
    if state == "unknown":
        return _EXIT_MOTION_UNKNOWN
    return _EXIT_EXECUTION_FAILED


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-skill-closed-loop", description="Drive the interactive closed loop against the live Gateway."
    )
    parser.add_argument("--config-name", help="robot_config name (bound by hermes-robot normally)")
    parser.add_argument("--config-path", help="explicit robot_config YAML path")
    parser.add_argument("--raw-command", required=True, help="audit text for the first workflow")
    parser.add_argument("--skill", help="first workflow skill name (single-step; use --steps for multi-step)")
    parser.add_argument("--steps", help="comma-separated skill names for a multi-step first workflow")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--place-name", default="")
    parser.add_argument("--motion-direction", default="")
    parser.add_argument("--motion-distance", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument(
        "--stop-after-sec",
        type=float,
        default=0.0,
        help="if >0, call request_stop() after this many seconds (simulates 别动)",
    )
    parser.add_argument("--continue-command", default="", help="audit text for the continuation workflow")
    parser.add_argument("--continue-skill", default="", help="continuation skill name; enables the 继续 phase (fresh)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify that unsupported breakpoint resume fails closed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    context, transport = load_runtime_context(config_name=args.config_name, config_path=args.config_path)
    bridge = RosBridge(
        status_service=transport.status_service,
        snapshot_service=transport.snapshot_service,
        reload_service=transport.reload_service,
        validate_skill_service=transport.validate_skill_service,
        skill_action=transport.skill_action_name,
        plan_service=transport.plan_service,
        validate_plan_service=transport.validate_plan_service,
        confirm_plan_service=transport.confirm_plan_service,
        execute_plan_action=transport.execute_plan_action,
    )
    if not bridge.start():
        print(
            json_dumps(
                {"event": "error", "error_code": "ROS_UNAVAILABLE", "message": "failed to initialize ROS bridge"}
            ),
            flush=True,
        )
        return 4
    controller = InteractiveController(bridge, timeout_policy=context.view["timeout_policy"])
    stop_thread = None
    stop_timer = threading.Event()
    output_lock = threading.Lock()
    feedback_phase = {"active": None, "next": 0}

    def emit(label: str, payload: Any) -> None:
        with output_lock:
            _emit(label, payload)

    def begin_feedback_phase():
        with output_lock:
            feedback_phase["next"] += 1
            phase = feedback_phase["next"]
            feedback_phase["active"] = phase

        def feedback(payload: dict[str, Any]) -> None:
            with output_lock:
                if feedback_phase["active"] == phase:
                    _emit("feedback", payload)

        return phase, feedback

    def emit_terminal(label: str, payload: dict[str, Any], phase: int) -> None:
        with output_lock:
            if feedback_phase["active"] == phase:
                feedback_phase["active"] = None
            _emit(label, payload)

    try:
        emit("discover", controller.discover())
        if args.steps:
            first_steps = [_workflow_step(args, s.strip()) for s in args.steps.split(",") if s.strip()]
        elif args.skill:
            first_steps = [_workflow_step(args, args.skill)]
        else:
            print(
                json_dumps(
                    {"event": "error", "error_code": "CLI_ARGUMENT", "message": "either --skill or --steps is required"}
                ),
                flush=True,
            )
            return 4
        emit("prepare", controller.prepare_workflow(args.raw_command, first_steps))

        if args.stop_after_sec > 0.0:

            def _stop_after():
                if stop_timer.wait(args.stop_after_sec):
                    return
                emit("stop_phrase", "别动")
                controller.request_stop()

            stop_thread = threading.Thread(target=_stop_after, daemon=True)
            stop_thread.start()

        # Presentation was flushed above. Internal validation/confirmation is a
        # Gateway admission step, not a user-facing confirmation gate.
        emit("auto_confirm", controller.confirm_plan())

        first_phase, first_feedback = begin_feedback_phase()
        first_terminal = controller.execute(feedback_callback=first_feedback)
        emit_terminal("execute_terminal", first_terminal, first_phase)
        stop_timer.set()
        if stop_thread is not None:
            stop_thread.join(timeout=1.0)
            stop_thread = None
        exit_code = _terminal_exit_code(first_terminal)

        if args.resume:
            controller.continue_workflow(args.continue_command or "继续", resume=True)
            raise RuntimeError("unsupported breakpoint resume was unexpectedly admitted")
        elif args.continue_skill and first_terminal["state"] == "stopped":
            presentation = controller.continue_workflow(
                args.continue_command or "继续", [_workflow_step(args, args.continue_skill)]
            )
            emit("continue_prepare", presentation)
            emit("continue_confirm", controller.confirm_plan())
            continue_phase, continue_feedback = begin_feedback_phase()
            continue_terminal = controller.execute(feedback_callback=continue_feedback)
            emit_terminal("continue_terminal", continue_terminal, continue_phase)
            exit_code = _terminal_exit_code(continue_terminal)
        elif args.continue_skill:
            emit(
                "continue_skipped",
                {"reason": "only a definite canceled terminal permits continuation", "state": first_terminal["state"]},
            )
        return exit_code
    except InteractiveControlError as exc:
        print(json_dumps({"event": "error", "error_code": exc.code, "message": str(exc)}), flush=True)
        return 13
    except Exception as exc:
        print(json_dumps({"event": "error", "error_code": "ROS_UNAVAILABLE", "message": str(exc)}), flush=True)
        return 4
    finally:
        stop_timer.set()
        if stop_thread is not None:
            stop_thread.join(timeout=1.0)
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
