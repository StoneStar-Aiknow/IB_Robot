#!/usr/bin/env python3

"""Prepare ros2_control controllers and switch the requested group atomically."""

import argparse
import inspect
import sys
from collections.abc import Iterable
from contextlib import suppress

import rclpy
from controller_manager import configure_controller, list_controllers, load_controller, switch_controllers

try:
    from controller_manager.controller_manager_services import ServiceNotFoundError
except ImportError:
    # Older ROS 2 controller_manager releases expose RuntimeError instead.
    class ServiceNotFoundError(RuntimeError):
        pass


from rclpy.node import Node


def _controller_state(controllers: Iterable[object], controller_name: str) -> str | None:
    return next((controller.state for controller in controllers if controller.name == controller_name), None)


def _supports_argument_count(function: object, count: int) -> bool:
    """Return whether a controller_manager helper accepts the modern call shape."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return True
    parameters = tuple(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return len(positional) >= count


def _call_controller_manager(
    function: object,
    modern_args: tuple[object, ...],
    legacy_args: tuple[object, ...],
):
    """Call current timeout-aware helpers or the older ROS 2 helper signatures."""
    args = modern_args if _supports_argument_count(function, len(modern_args)) else legacy_args
    return function(*args)


def _list_controller_state(
    node: Node,
    controller_manager: str,
    controller_name: str,
    service_timeout: float,
    call_timeout: float,
) -> str | None:
    response = _call_controller_manager(
        list_controllers,
        (node, controller_manager, service_timeout, call_timeout),
        (node, controller_manager),
    )
    return _controller_state(response.controller, controller_name)


def _prepare_controller(
    node: Node,
    controller_manager: str,
    controller_name: str,
    controller_manager_timeout: float,
    service_call_timeout: float,
) -> str:
    """Ensure one controller is loaded and configured without changing activation."""
    state = _list_controller_state(
        node,
        controller_manager,
        controller_name,
        controller_manager_timeout,
        service_call_timeout,
    )

    if state is None:
        try:
            response = _call_controller_manager(
                load_controller,
                (node, controller_manager, controller_name, controller_manager_timeout, service_call_timeout),
                (node, controller_manager, controller_name),
            )
        except RuntimeError as exc:
            node.get_logger().warning(
                f"load_controller did not return successfully for '{controller_name}'; checking manager state"
            )
            state = _list_controller_state(
                node,
                controller_manager,
                controller_name,
                controller_manager_timeout,
                service_call_timeout,
            )
            if state is None:
                raise RuntimeError(f"Failed to load controller '{controller_name}'") from exc
        else:
            if response.ok:
                node.get_logger().info(f"Loaded controller '{controller_name}'")
                state = "unconfigured"
            else:
                state = _list_controller_state(
                    node,
                    controller_manager,
                    controller_name,
                    controller_manager_timeout,
                    service_call_timeout,
                )
                if state is None:
                    raise RuntimeError(f"Failed to load controller '{controller_name}'")

    if state == "unconfigured":
        try:
            response = _call_controller_manager(
                configure_controller,
                (node, controller_manager, controller_name, controller_manager_timeout, service_call_timeout),
                (node, controller_manager, controller_name),
            )
        except RuntimeError as exc:
            node.get_logger().warning(
                f"configure_controller did not return successfully for '{controller_name}'; checking manager state"
            )
            state = _list_controller_state(
                node,
                controller_manager,
                controller_name,
                controller_manager_timeout,
                service_call_timeout,
            )
            if state not in {"inactive", "active"}:
                raise RuntimeError(f"Failed to configure controller '{controller_name}'") from exc
        else:
            if response.ok:
                state = "inactive"
            else:
                state = _list_controller_state(
                    node,
                    controller_manager,
                    controller_name,
                    controller_manager_timeout,
                    service_call_timeout,
                )
                if state not in {"inactive", "active"}:
                    raise RuntimeError(f"Failed to configure controller '{controller_name}'")

    if state not in {"inactive", "active"}:
        raise RuntimeError(f"Controller '{controller_name}' has unsupported lifecycle state '{state}'")

    node.get_logger().info(f"Controller '{controller_name}' is prepared in '{state}' state")
    return state


def _group_has_requested_states(
    node: Node,
    controller_manager: str,
    active_controller_names: tuple[str, ...],
    inactive_controller_names: tuple[str, ...],
    controller_manager_timeout: float,
    service_call_timeout: float,
) -> bool:
    response = _call_controller_manager(
        list_controllers,
        (node, controller_manager, controller_manager_timeout, service_call_timeout),
        (node, controller_manager),
    )
    states = {controller.name: controller.state for controller in response.controller}
    return all(states.get(name) == "active" for name in active_controller_names) and all(
        states.get(name) == "inactive" for name in inactive_controller_names
    )


def _normalized_controller_names(controller_names: Iterable[str], group_name: str) -> tuple[str, ...]:
    names = tuple(controller_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{group_name} controller names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{group_name} controller names must be unique")
    return names


def spawn_controllers(
    node: Node,
    controller_manager: str,
    controller_names: Iterable[str],
    inactive_controller_names: Iterable[str],
    controller_manager_timeout: float,
    service_call_timeout: float,
    switch_timeout: float,
) -> None:
    """Prepare all controllers, then apply the requested states in one switch call."""
    active_names = _normalized_controller_names(controller_names, "active")
    inactive_names = _normalized_controller_names(inactive_controller_names, "inactive")
    overlap = set(active_names) & set(inactive_names)
    if overlap:
        raise ValueError(f"Controllers cannot be both active and inactive: {sorted(overlap)}")
    if not active_names and not inactive_names:
        raise ValueError("At least one controller name is required")

    prepared_states = {
        controller_name: _prepare_controller(
            node,
            controller_manager,
            controller_name,
            controller_manager_timeout,
            service_call_timeout,
        )
        for controller_name in (*active_names, *inactive_names)
    }
    activate_names = [name for name in active_names if prepared_states[name] != "active"]
    deactivate_names = [name for name in inactive_names if prepared_states[name] == "active"]

    if not activate_names and not deactivate_names:
        node.get_logger().info("Controller group already has the requested activation states")
        return

    if activate_names and deactivate_names:
        failure_message = "Failed to switch controller group"
    elif activate_names:
        failure_message = "Failed to activate controller group"
    else:
        failure_message = "Failed to deactivate controller group"

    try:
        response = _call_controller_manager(
            switch_controllers,
            (
                node,
                controller_manager,
                deactivate_names,
                activate_names,
                True,
                True,
                switch_timeout,
                max(service_call_timeout, switch_timeout),
            ),
            (node, controller_manager, deactivate_names, activate_names, True, True, switch_timeout),
        )
    except RuntimeError as exc:
        node.get_logger().warning("switch_controller did not return successfully for the group; checking manager state")
        if not _group_has_requested_states(
            node,
            controller_manager,
            active_names,
            inactive_names,
            controller_manager_timeout,
            service_call_timeout,
        ):
            raise RuntimeError(failure_message) from exc
    else:
        if not response.ok and not _group_has_requested_states(
            node,
            controller_manager,
            active_names,
            inactive_names,
            controller_manager_timeout,
            service_call_timeout,
        ):
            raise RuntimeError(failure_message)
    node.get_logger().info(
        f"Applied controller group state: active={list(active_names)}, inactive={list(inactive_names)}"
    )


def spawn_controller(
    node: Node,
    controller_manager: str,
    controller_name: str,
    controller_manager_timeout: float,
    service_call_timeout: float,
    switch_timeout: float,
    *,
    activate: bool = True,
) -> None:
    """Compatibility wrapper for callers that manage one controller."""
    spawn_controllers(
        node,
        controller_manager,
        [controller_name] if activate else [],
        [] if activate else [controller_name],
        controller_manager_timeout,
        service_call_timeout,
        switch_timeout,
    )


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return timeout


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller_names", nargs="*", help="Controllers to load, configure, and activate together.")
    parser.add_argument(
        "--controller-manager",
        default="controller_manager",
        help="Controller manager node name.",
    )
    parser.add_argument(
        "--controller-manager-timeout",
        type=_positive_timeout,
        default=120.0,
        help="Timeout for controller manager service discovery.",
    )
    parser.add_argument(
        "--service-call-timeout",
        type=_positive_timeout,
        default=120.0,
        help="Timeout for each controller manager service response.",
    )
    parser.add_argument(
        "--switch-timeout",
        type=_positive_timeout,
        default=120.0,
        help="Controller-manager timeout for activation completion.",
    )
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Load and configure all positional controllers without activating them.",
    )
    parser.add_argument(
        "--inactive-controller",
        dest="inactive_controller_names",
        action="append",
        default=[],
        help="Controller to load/configure but leave inactive. Repeat for multiple controllers.",
    )
    args = parser.parse_args(rclpy.utilities.remove_ros_args(args=argv)[1:])
    if args.inactive:
        args.inactive_controller_names = [*args.controller_names, *args.inactive_controller_names]
        args.controller_names = []
    if not args.controller_names and not args.inactive_controller_names:
        parser.error("at least one controller name is required")
    return args


def _resolve_controller_manager(node: Node, controller_manager: str) -> str:
    if controller_manager.startswith("/"):
        return controller_manager
    namespace = node.get_namespace().rstrip("/")
    return f"{namespace}/{controller_manager}" if namespace else f"/{controller_manager}"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    args = _parse_args(argv)

    rclpy.init(args=argv)
    node = Node("controller_spawner_group")
    controller_manager = _resolve_controller_manager(node, args.controller_manager)
    try:
        spawn_controllers(
            node,
            controller_manager,
            args.controller_names,
            args.inactive_controller_names,
            args.controller_manager_timeout,
            args.service_call_timeout,
            args.switch_timeout,
        )
        return 0
    except (RuntimeError, ValueError, ServiceNotFoundError) as exc:
        node.get_logger().fatal(str(exc))
        return 1
    finally:
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
