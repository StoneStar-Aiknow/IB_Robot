"""Functional-verification driver for the interactive closed loop.

Drives ``InteractiveController`` against the live Capability Gateway (real
``RosBridge``) to prove the five-feature closed loop end to end: discover,
reject-out-of-catalog, prepare/confirm, ``别动`` stop to a definite terminal,
and ``继续`` fresh-state continuation.

Install/run inside the built workspace::

    source .shrc_local && export ROS_DOMAIN_ID=56 && source install/setup.sh
    robot-skill-closed-loop --config-name so101_single_arm \
        --raw-command '点个头' --skill nod_yes \
        --stop-after-sec 1.5 \
        --continue-command '继续挥挥手' --continue-skill wave_hand

Breakpoint resume (continue the prior plan's remaining steps after a stop)::

    robot-skill-closed-loop --config-name so101_single_arm \
        --raw-command '点头两次再摇头最后撒娇' --steps nod_yes,nod_yes,shake_no,act_cute \
        --stop-after-sec 3.0 \
        --resume --continue-command '继续'

``--stop-after-sec`` spawns a daemon thread that calls ``request_stop()`` after
the delay, simulating the operator saying ``别动`` mid-execution. The driver
then requires a definite terminal before issuing the continuation.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

from robot_skill_cli.catalog import load_runtime_context
from robot_skill_cli.interactive_control import (
    UNKNOWN,
    InteractiveControlError,
    InteractiveController,
)
from robot_skill_cli.output import json_dumps
from robot_skill_cli.ros_bridge import RosBridge


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
        help="continue by resuming the prior plan's remaining steps (breakpoint resume)",
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
    try:
        _emit("discover", controller.discover())
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
        _emit("prepare", controller.prepare_workflow(args.raw_command, first_steps))
        _emit("auto_confirm", controller.confirm_plan())

        if args.stop_after_sec > 0.0:

            def _stop_after():
                time.sleep(args.stop_after_sec)
                _emit("stop_phrase", "别动")
                controller.request_stop()

            stop_thread = threading.Thread(target=_stop_after, daemon=True)
            stop_thread.start()

        first_terminal = controller.execute(feedback_callback=lambda fb: _emit("feedback", fb))
        _emit("execute_terminal", first_terminal)

        if (args.resume or args.continue_skill) and first_terminal["state"] != UNKNOWN:
            if args.resume:
                presentation = controller.continue_workflow(args.continue_command or "继续", resume=True)
            else:
                presentation = controller.continue_workflow(
                    args.continue_command or "继续", [_workflow_step(args, args.continue_skill)]
                )
            _emit("continue_prepare", presentation)
            _emit("continue_confirm", controller.confirm_plan())
            _emit("continue_terminal", controller.execute())
        elif args.resume or args.continue_skill:
            _emit(
                "continue_skipped",
                {"reason": "prior terminal is unknown; refuse to continue", "state": first_terminal["state"]},
            )
        return 0
    except InteractiveControlError as exc:
        print(json_dumps({"event": "error", "error_code": exc.code, "message": str(exc)}), flush=True)
        return 13
    finally:
        if stop_thread is not None:
            stop_thread.join(timeout=1.0)
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
