"""Query a semantic object and print a compact map-frame stand-off pose."""

import argparse
import json
import math
import sys

import numpy as np
import rclpy

from ibrobot_msgs.srv import GetSemanticObjects

from .target_resolution import generate_staging_candidates


def object_position_from_msg(object_msg) -> np.ndarray:
    """Extract the map-frame position from a SemanticObject3D message.

    ``SemanticObject3D.pose`` is a PoseWithCovarianceStamped, so the position
    lives three ``.pose`` levels deep.
    """
    return np.asarray(
        [
            object_msg.pose.pose.pose.position.x,
            object_msg.pose.pose.pose.position.y,
            object_msg.pose.pose.pose.position.z,
        ],
        dtype=np.float64,
    )


def compute_standoff_pose(
    object_position: np.ndarray,
    reference_position: np.ndarray,
    stand_off_distance_m: float,
) -> tuple[float, float, float]:
    """Return ``(x, y, yaw_degrees)`` for the preferred stand-off candidate."""
    candidate = generate_staging_candidates(
        object_position,
        reference_position,
        stand_off_distance_m,
        candidate_count=1,
    )[0]
    return (
        float(candidate.position[0]),
        float(candidate.position[1]),
        float(math.degrees(candidate.yaw)),
    )


def _query_object(node, label: str, timeout_sec: float):
    client = node.create_client(GetSemanticObjects, "/semantic_mapping/get_objects")
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return None
    request = GetSemanticObjects.Request()
    request.label = label
    request.include_inactive = True
    request.max_age_sec = 0.0
    request.max_results = 1
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.exception() is not None:
        return None
    response = future.result()
    if not response.success or not response.semantic_map.objects:
        return None
    return response.semantic_map.objects[0]


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="manual semantic label to query")
    parser.add_argument("stand_off_distance_m", nargs="?", type=float, default=0.2)
    parser.add_argument("--reference-x", type=float, default=0.0)
    parser.add_argument("--reference-y", type=float, default=0.0)
    parser.add_argument("--reference-z", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    cli_args, _unknown = parser.parse_known_args(args)

    node = None
    try:
        if cli_args.stand_off_distance_m <= 0.0 or cli_args.timeout_sec <= 0.0:
            print("[]")
            return 1
        # Application arguments are consumed by argparse; do not pass them to
        # rclpy's global ROS argument parser.
        rclpy.init(args=[])
        node = rclpy.create_node("semantic_map_standoff")
        object_msg = _query_object(node, cli_args.label, cli_args.timeout_sec)
        if object_msg is None:
            print("[]")
            return 1
        object_position = object_position_from_msg(object_msg)
        pose = compute_standoff_pose(
            object_position,
            np.asarray([cli_args.reference_x, cli_args.reference_y, cli_args.reference_z], dtype=np.float64),
            cli_args.stand_off_distance_m,
        )
        print(json.dumps(list(pose), separators=(",", ":")))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI contract reports failures as an empty array
        print(f"semantic_map_standoff: {exc}", file=sys.stderr)
        print("[]")
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
