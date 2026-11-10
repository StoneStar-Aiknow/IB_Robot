"""Generic model resource scheduling, independent of Policy product sessions."""

import json

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger

from inference_service.scheduler.node_lifecycle import spin_until_hosts_close
from inference_service.scheduler.runtime_profiles import DEFAULT_MIN_SAMPLES
from inference_service.scheduler.runtime_protocol import (
    DEFAULT_CONTROL_RPC_TIMEOUT_NS,
    DEFAULT_RUNTIME_ENDPOINT,
    DEFAULT_STATUS_STALE_TIMEOUT_NS,
    RuntimeExecutionTarget,
)
from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint


class GlobalRuntimeSchedulerNode(Node):
    """Sole owner of the host-wide runtime resource ledger."""

    def __init__(self, *, parameter_overrides=None):
        super().__init__("global_runtime_scheduler", parameter_overrides=parameter_overrides)
        try:
            if "global_policy" in self._parameter_overrides:
                raise ValueError("global_policy has been renamed to priority_zero_policy")
            for name, default in (
                ("runtime_targets_json", "[]"),
                ("runtime_resources_json", "{}"),
                ("runtime_scheduler_endpoint", DEFAULT_RUNTIME_ENDPOINT),
                ("priority_zero_policy", "fifo"),
                ("status_stale_timeout_ns", DEFAULT_STATUS_STALE_TIMEOUT_NS),
                ("control_rpc_timeout_ns", DEFAULT_CONTROL_RPC_TIMEOUT_NS),
                ("profile_min_samples", DEFAULT_MIN_SAMPLES),
                ("profile_max_age_days", 30),
            ):
                self.declare_parameter(name, default)

            def value(name):
                return self.get_parameter(name).value

            targets = [RuntimeExecutionTarget(**item) for item in json.loads(value("runtime_targets_json"))]
            resources = json.loads(value("runtime_resources_json"))
            if not isinstance(resources, dict):
                raise ValueError("runtime_resources_json must be an object")
            endpoint = value("runtime_scheduler_endpoint")
            self._runtime_endpoint = GlobalRuntimeEndpoint(
                self,
                targets,
                endpoint=endpoint,
                resources=resources or None,
                policy=value("priority_zero_policy"),
                stale_ns=value("status_stale_timeout_ns"),
                control_rpc_timeout_ns=value("control_rpc_timeout_ns"),
                profile_min_samples=value("profile_min_samples"),
                profile_max_age_days=value("profile_max_age_days"),
            )
            self._readiness_service = self.create_service(Trigger, endpoint + "/ready", self._readiness_callback)
        except BaseException:
            self.destroy_node()
            raise

    def _readiness_callback(self, request, response):
        core = self._runtime_endpoint.core
        unavailable = core.unavailable_resources()
        response.success = core.ready
        response.message = json.dumps(list(unavailable)) if unavailable else ""
        return response

    def collect_retired_transport(self, executor):
        self._runtime_endpoint.collect_retired_clients(executor)

    def begin_shutdown(self):
        self._runtime_endpoint.core.begin_shutdown()

    def shutdown_pending(self):
        return self._runtime_endpoint.core.shutdown_pending()


def main(argv=None):
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    node = executor = None
    try:
        node = GlobalRuntimeSchedulerNode()
        executor = MultiThreadedExecutor(num_threads=8)
        executor.add_node(node)
        spin_until_hosts_close(node, executor)
    except ExternalShutdownException:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
