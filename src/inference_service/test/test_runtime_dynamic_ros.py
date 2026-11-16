"""Public scheduler accepts a separately started model host through ROS."""

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from inference_service.global_runtime_scheduler_node import GlobalRuntimeSchedulerNode
from inference_service.scheduler.runtime_controller import freeze_inputs
from inference_service.scheduler.runtime_profiles import RuntimeProfileRegistry
from inference_service.scheduler.runtime_protocol import RuntimeExecutionTarget
from inference_service.scheduler.runtime_ros import SchedulerRuntimeAdapter
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelRuntimeHandle

pytestmark = pytest.mark.usefixtures("simulated_runtime_construction")


@pytest.mark.parametrize("predictive", [False, True])
def test_public_scheduler_starts_empty_then_serves_late_models_and_retires_one(tmp_path, predictive):
    resources = {"cpu": {"physical_device": "local/cpu", "mode": "serial"}}
    if predictive:
        resources["cpu"].update(
            predictive_admission=True,
            managed_coverage=True,
            hardware_profile_fingerprint="test_hardware",
        )
    rclpy.init()
    global_node = GlobalRuntimeSchedulerNode(
        parameter_overrides=[
            Parameter(
                "runtime_resources_json",
                value=json.dumps(resources),
            ),
            Parameter("runtime_scheduler_endpoint", value="/test/dynamic/global"),
            Parameter("status_stale_timeout_ns", value=7_000_000_000),
        ],
    )
    runtime = global_node._runtime_endpoint
    host_node = Node("late_dynamic_host")
    peer_node = Node("second_dynamic_host")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(global_node)
    executor.add_node(host_node)
    executor.add_node(peer_node)
    stopped, errors = threading.Event(), []

    def spin_control():
        try:
            while not stopped.is_set():
                executor.spin_once(timeout_sec=0.02)
                global_node.collect_retired_transport(executor)
        except Exception as exc:
            errors.append(exc)

    spin = threading.Thread(target=spin_control, daemon=True)
    spin.start()

    class Model:
        def execute(self, request, _context):
            return {"answer": request.inputs["value"] * 2}

    handle = ModelRuntimeHandle(executor=Model())
    handle.load()
    adapter = None
    peer_adapter = None
    peer_handle = ModelRuntimeHandle(executor=Model())
    peer_handle.load()
    try:
        assert runtime.core.targets == {}
        selected = RuntimeExecutionTarget(
            target_id="perception/late",
            endpoint="/test/dynamic/late",
            hardware_resource_id="cpu",
            physical_device="local/cpu",
            deployment_fingerprint="test",
            composition_fingerprint="test",
            lifecycle_group="late",
            configuration_generation="test",
            interface="tensor_model",
            model_type="late",
            operation="infer",
            profile_observation_inputs=("value",),
        )
        if predictive:
            selected = replace(
                selected,
                predictive_admission=True,
                hardware_profile_fingerprint="test_hardware",
                runtime_version="torch-test-version",
                profile_path=str(tmp_path / "runtime.json"),
            )
            (tmp_path / "runtime.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "scope": "runtime_execution",
                        "identity": RuntimeProfileRegistry.identity(selected),
                        "measured_at": time.time(),
                        "buckets": [
                            {
                                "work": "execute",
                                "input_bucket": freeze_inputs(
                                    ModelRequest({"value": np.asarray(value)}),
                                    4096,
                                    selected.profile_observation_inputs,
                                )[3],
                                "samples": 10000,
                                "upper_bound_ns": 100_000_000,
                            }
                            for value in (3,)
                        ],
                    }
                )
            )
            handle.assembly.session = SimpleNamespace(runtime_version="torch-test-version")
            peer_handle.assembly.session = SimpleNamespace(runtime_version="onnx-test-version")
        adapter = SchedulerRuntimeAdapter(host_node, selected, handle, endpoint="/test/dynamic/global")
        expires = time.monotonic() + 5
        while not adapter.ready or not runtime.core.resource_ready("cpu"):
            assert time.monotonic() < expires
            time.sleep(0.01)
        assert runtime.core.stale_ns == 7_000_000_000
        assert adapter._stale_seconds == 7.0
        result = adapter.execute(ModelRequest({"value": np.asarray(3)}), ExecutionContext("late", Deadline.after(3)))
        assert result.outputs["answer"] == 6
        peer_target = replace(
            selected, target_id="perception/peer", endpoint="/test/dynamic/peer", lifecycle_group="peer"
        )
        if predictive:
            peer_target = replace(
                peer_target, runtime_version="onnx-test-version", profile_path=str(tmp_path / "peer.json")
            )
            document = json.loads((tmp_path / "runtime.json").read_text())
            document["identity"] = RuntimeProfileRegistry.identity(peer_target)
            (tmp_path / "peer.json").write_text(json.dumps(document))
        peer_adapter = SchedulerRuntimeAdapter(peer_node, peer_target, peer_handle, endpoint="/test/dynamic/global")
        expires = time.monotonic() + 5
        while not peer_adapter.ready or not runtime.core.resource_ready("cpu"):
            assert time.monotonic() < expires
            time.sleep(0.01)
        assert len(runtime.core.targets) == 2
        result = peer_adapter.execute(
            ModelRequest({"value": np.asarray(4)}), ExecutionContext("peer", Deadline.after(3))
        )
        assert result.outputs["answer"] == 8
        adapter.close()
        assert selected.target_id in runtime.core._retired
        assert runtime.core.resource_ready("cpu")
        result = peer_adapter.execute(
            ModelRequest({"value": np.asarray(5)}), ExecutionContext("peer_remains", Deadline.after(3))
        )
        assert result.outputs["answer"] == 10
        # Advance only the ledger's collection clock after proven retirement;
        # the real ROS wait set remains active while the client is removed.
        until = runtime.core._retired[selected.target_id] + 1
        runtime.core.now_ns = lambda: until
        expires = time.monotonic() + 2
        while selected.target_id in runtime._clients or runtime._retired_clients:
            assert time.monotonic() < expires
            time.sleep(0.01)
        runtime.core.now_ns = time.monotonic_ns
        assert spin.is_alive() and not errors
    finally:
        if peer_adapter is not None:
            peer_adapter.close()
        if adapter is not None:
            adapter.close()
        stopped.set()
        executor.wake()
        executor.shutdown(timeout_sec=2)
        spin.join(timeout=2)
        handle.close()
        peer_handle.close()
        host_node.destroy_node()
        peer_node.destroy_node()
        global_node.destroy_node()
        rclpy.shutdown()
