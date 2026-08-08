#!/usr/bin/env python3

"""Load and activate one ros2_control controller with explicit service timeouts."""

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


def spawn_controller(
    node: Node,
    controller_manager: str,
    controller_name: str,
    controller_manager_timeout: float,
    service_call_timeout: float,
    switch_timeout: float,
) -> None:
    """Ensure one controller is loaded, configured, and active."""
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

    if state == "active":
        node.get_logger().info(f"Controller '{controller_name}' is already active")
        return

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
            if state != "inactive":
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
                if state != "inactive":
                    raise RuntimeError(f"Failed to configure controller '{controller_name}'")

    if state != "inactive":
        raise RuntimeError(f"Controller '{controller_name}' has unsupported lifecycle state '{state}'")

    try:
        response = _call_controller_manager(
            switch_controllers,
            (node, controller_manager, [], [controller_name], True, True, switch_timeout, service_call_timeout),
            (node, controller_manager, [], [controller_name], True, True, switch_timeout),
        )
    except RuntimeError as exc:
        node.get_logger().warning(
            f"switch_controller did not return successfully for '{controller_name}'; checking manager state"
        )
        state = _list_controller_state(
            node,
            controller_manager,
            controller_name,
            controller_manager_timeout,
            service_call_timeout,
        )
        if state != "active":
            raise RuntimeError(f"Failed to activate controller '{controller_name}'") from exc
    else:
        if not response.ok:
            state = _list_controller_state(
                node,
                controller_manager,
                controller_name,
                controller_manager_timeout,
                service_call_timeout,
            )
            if state != "active":
                raise RuntimeError(f"Failed to activate controller '{controller_name}'")
    node.get_logger().info(f"Configured and activated controller '{controller_name}'")


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return timeout


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller_name", help="Controller to load, configure, and activate.")
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
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=argv)[1:])


def _resolve_controller_manager(node: Node, controller_manager: str) -> str:
    if controller_manager.startswith("/"):
        return controller_manager
    namespace = node.get_namespace().rstrip("/")
    return f"{namespace}/{controller_manager}" if namespace else f"/{controller_manager}"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    args = _parse_args(argv)

    rclpy.init(args=argv)
    node = Node(f"controller_spawner_{args.controller_name}")
    controller_manager = _resolve_controller_manager(node, args.controller_manager)
    try:
        spawn_controller(
            node,
            controller_manager,
            args.controller_name,
            args.controller_manager_timeout,
            args.service_call_timeout,
            args.switch_timeout,
        )
        return 0
    except (RuntimeError, ServiceNotFoundError) as exc:
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
