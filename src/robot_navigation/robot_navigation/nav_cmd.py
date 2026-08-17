import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger

from ibrobot_msgs.action import ExecuteNavigation

COMMAND_TYPES = {
    "forward": ExecuteNavigation.Goal.FORWARD,
    "backward": ExecuteNavigation.Goal.BACKWARD,
    "leftward": ExecuteNavigation.Goal.STRAFE_LEFT,
    "rightward": ExecuteNavigation.Goal.STRAFE_RIGHT,
    "turn-left": ExecuteNavigation.Goal.TURN_LEFT,
    "turn-right": ExecuteNavigation.Goal.TURN_RIGHT,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a single navigation command for manual testing.")
    parser.add_argument("--action-name", required=True, help="Absolute ExecuteNavigation action name.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("cancel")
    for name in COMMAND_TYPES:
        command = subparsers.add_parser(name)
        command.add_argument("value", type=float)
    absolute = subparsers.add_parser("absolute")
    absolute.add_argument("x", type=float)
    absolute.add_argument("y", type=float)
    absolute.add_argument("yaw_deg", type=float)
    return parser


def _pose(x: float, y: float, yaw_deg: float) -> PoseStamped:
    if not all(math.isfinite(value) for value in (x, y, yaw_deg)):
        raise ValueError("absolute target values must be finite")
    yaw = math.radians(yaw_deg)
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def _send(node: Node, goal: ExecuteNavigation.Goal, action_name: str) -> int:
    client = ActionClient(node, ExecuteNavigation, action_name)
    if not client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error(f"{action_name} is unavailable")
        return 1
    future = client.send_goal_async(goal, feedback_callback=lambda message: print(message.feedback.state))
    rclpy.spin_until_future_complete(node, future)
    goal_handle = future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().error("Navigation command was rejected")
        return 1
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    print(result.message)
    return 0 if result.success else 1


def main(args=None):
    parsed = _parser().parse_args(args)
    rclpy.init(args=args)
    node = Node("nav_cmd")
    try:
        if parsed.command == "status":
            action_client = ActionClient(node, ExecuteNavigation, parsed.action_name)
            nav2_client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
            print(
                f"{parsed.action_name}: {'ready' if action_client.wait_for_server(timeout_sec=2.0) else 'unavailable'}"
            )
            print(f"/navigate_to_pose: {'ready' if nav2_client.wait_for_server(timeout_sec=2.0) else 'unavailable'}")
            return 0
        if parsed.command == "cancel":
            client = node.create_client(Trigger, "/navigation/cancel_current")
            if not client.wait_for_service(timeout_sec=5.0):
                node.get_logger().error("/navigation/cancel_current is unavailable")
                return 1
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future)
            response = future.result()
            print(response.message)
            return 0 if response.success else 1

        goal = ExecuteNavigation.Goal()
        if parsed.command == "absolute":
            goal.command_type = ExecuteNavigation.Goal.ABSOLUTE_POSE
            goal.target_pose = _pose(parsed.x, parsed.y, parsed.yaw_deg)
        else:
            if not math.isfinite(parsed.value) or parsed.value <= 0.0:
                raise ValueError("relative command value must be positive and finite")
            goal.command_type = COMMAND_TYPES[parsed.command]
            goal.value = parsed.value if not parsed.command.startswith("turn-") else math.radians(parsed.value)
        return _send(node, goal, parsed.action_name)
    except ValueError as exc:
        print(f"nav_cmd: {exc}", file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
