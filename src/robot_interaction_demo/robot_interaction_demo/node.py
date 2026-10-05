"""Resident parameterized interaction demo with endpoints injected by robot_config."""

import json
import math
import threading

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import ExecuteInteractionDemo
from robot_interaction_demo.runtime_demo import execute_named_motion, execute_speech

DEFAULT_SERVICE = "/interaction_demo/execute"


class InteractionDemoNode(Node):
    def __init__(self, **kwargs):
        super().__init__("interaction_demo", **kwargs)
        readonly = ParameterDescriptor(read_only=True)
        self._speech_service = self.declare_parameter("speech_service", "", readonly).value
        self._motion_action = self.declare_parameter("named_motion_action", "", readonly).value
        self._timeout = self.declare_parameter("timeout_sec", 40.0, readonly).value
        service = self.declare_parameter("service_name", DEFAULT_SERVICE, readonly).value
        for name, endpoint in (("speech_service", self._speech_service), ("named_motion_action", self._motion_action)):
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise ValueError(f"{name} is required")
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("timeout_sec must be positive and finite")
        if not isinstance(service, str) or not service.strip():
            raise ValueError("service_name must be non-empty")
        self._busy = threading.Lock()
        self.create_service(ExecuteInteractionDemo, service, self._handle, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f"Interaction demo ready at {service}; waiting for parameterized requests")

    def _handle(self, request, response):
        if not self._busy.acquire(blocking=False):
            result = {"success": False, "phase": "rejected", "message": "Demo busy; request was not queued"}
        else:
            try:
                if request.operation == request.SPEAK:
                    result = execute_speech(
                        self,
                        self._speech_service,
                        text=request.text,
                        language=request.language,
                        priority=request.priority,
                        interrupt=request.interrupt,
                        timeout_sec=self._timeout,
                        externally_spun=True,
                    )
                elif request.operation == request.NAMED_MOTION:
                    result = execute_named_motion(
                        self,
                        self._motion_action,
                        name=request.motion_name,
                        target=request.target,
                        interrupt=request.interrupt,
                        allow_motion=request.allow_motion,
                        timeout_sec=self._timeout,
                        externally_spun=True,
                    )
                else:
                    result = {
                        "success": False,
                        "phase": "rejected",
                        "message": f"Unknown operation {request.operation}",
                    }
            except (ValueError, TimeoutError) as exc:
                result = {"success": False, "phase": "rejected", "message": str(exc)}
            except Exception as exc:
                self.get_logger().error(f"Demo request failed: {exc}")
                result = {"success": False, "phase": "failed", "message": str(exc)}
            finally:
                self._busy.release()
        response.success = bool(result["success"])
        response.result_json = json.dumps(result, ensure_ascii=False)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = InteractionDemoNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
