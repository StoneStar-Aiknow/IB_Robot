"""Fixed-target ROS transport closure using deterministic CPU executors.

These tests retain fixed-catalogue coverage through the real V1 service.
Dynamic membership, recovery and receipts have dedicated protocol suites.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import ExitStack
from dataclasses import replace

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from inference_service.scheduler.runtime_protocol import ControlError
from inference_service.scheduler.runtime_ros import GlobalRuntimeEndpoint, SchedulerRuntimeAdapter
from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelRuntimeHandle

from .runtime_scheduler_helpers import attested_resources, target


def wait_for(predicate, timeout=5):
    expires = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < expires, "runtime control did not reach the expected state"
        time.sleep(0.01)


@pytest.fixture
def transport(request):
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "load the shared ROS domain isolation hooks"
    events = []
    handles, adapters, targets = [], [], []

    class Model:
        def __init__(self, name):
            self.name = name
            self.resets = 0

        def execute(self, request, context):
            events.append((self.name, "start"))
            assert request.metadata["priority"] == 0
            time.sleep(0.03)
            events.append((self.name, "end"))
            return {"answer": request.inputs["value"] * 2}

        def reset(self, context):
            self.resets += 1

    with ExitStack() as infrastructure:
        rclpy.init()
        infrastructure.callback(rclpy.shutdown)
        nodes = []
        for name in ("fixed_global", "fixed_first", "fixed_second"):
            node = Node(name)
            infrastructure.callback(node.destroy_node)
            nodes.append(node)
        executor = MultiThreadedExecutor(num_threads=6)
        spin = None
        try:
            for node in nodes:
                executor.add_node(node)
            for index, name in enumerate(("first", "second")):
                t = replace(target(name, resource="cpu"), physical_device="local/cpu")
                targets.append(t)
                handle = ModelRuntimeHandle(executor=Model(name), resettable=True)
                handle.load()
                handles.append(handle)
                adapters.append(SchedulerRuntimeAdapter(nodes[index + 1], t, handle, endpoint="/test/fixed/global"))
            endpoint = GlobalRuntimeEndpoint(
                nodes[0],
                targets,
                endpoint="/test/fixed/global",
                resources=None if getattr(request, "param", None) == "fixed_only" else attested_resources(*targets),
            )
            spin = threading.Thread(target=executor.spin, daemon=True)
            spin.start()
            try:
                wait_for(lambda: all(adapter.ready for adapter in adapters))
                yield endpoint, adapters, handles, events
            finally:
                for handle, adapter in zip(handles, adapters, strict=True):
                    handle.close(ExecutionContext("close", Deadline.after(5)))
                    # Local Close may return before Global receives the ACK reply.
                    # Only sequential, confirmed shutdown is supported here.
                    wait_for(lambda a=adapter: a.target.target_id not in endpoint.core.shutdown_pending())
                wait_for(lambda: not endpoint.core.shutdown_pending())
        finally:
            try:
                assert executor.shutdown(timeout_sec=5)
            finally:
                if spin is not None and spin.ident is not None:
                    spin.join(timeout=5)
                    assert not spin.is_alive()
        # Destroy protocol entities only after successful Global confirmation.
        # ExitStack still attempts every node/context cleanup on assertion failure,
        # preserving the original exception in any subsequent exception chain.
        for node in nodes:
            node.executor = None
        for adapter in adapters:
            adapter.destroy_transport()


def test_ros_shared_queue_routes_public_handles_and_preserves_results(transport):
    endpoint, adapters, handles, events = transport
    with ThreadPoolExecutor(2) as workers:
        calls = [
            workers.submit(
                handle.execute, ModelRequest({"value": index}), ExecutionContext(f"business-{index}", Deadline.after(5))
            )
            for index, handle in enumerate(handles)
        ]
        assert [future.result(timeout=6).outputs["answer"] for future in calls] == [0, 2]
    assert [event for _, event in events] == ["start", "end", "start", "end"]
    assert all(record.state == "completed" for record in endpoint.core._records.values())
    handles[0].reset(ExecutionContext("reset", Deadline.after(5)))
    assert handles[0].assembly.runtime_executor.resets == 1
    assert handles[1].assembly.runtime_executor.resets == 0
    assert handles[0].execute(ModelRequest({"value": 4}), ExecutionContext("again", Deadline.after(5))).outputs == {
        "answer": 8
    }


def test_lost_submit_reply_reuses_the_same_operation(transport, monkeypatch):
    endpoint, adapters, handles, events = transport
    call = adapters[0]._call
    lost = []

    def lose_once(verb, submission, timeout, **kwargs):
        result = call(verb, submission, timeout, **kwargs)
        if verb == "submit" and not lost:
            lost.append(submission.operation_id)
            raise TimeoutError("discarded submit response")
        return result

    monkeypatch.setattr(adapters[0], "_call", lose_once)
    result = handles[0].execute(ModelRequest({"value": 3}), ExecutionContext("lost-reply", Deadline.after(5)))
    assert result.outputs == {"answer": 6}
    assert events == [("first", "start"), ("first", "end")]
    assert list(endpoint.core._records) == lost


def test_terminal_notifications_dispatch_short_deadline_peers_without_polling(transport, monkeypatch):
    endpoint, adapters, handles, events = transport
    # Remove periodic Query/ACK reconciliation as a source of progress. Real
    # ROS guard callbacks must publish completion and dispatch the queued peer.
    with monkeypatch.context() as scoped:
        scoped.setattr(endpoint.core, "pending_reports", lambda: [])
        for _ in range(3):
            with ThreadPoolExecutor(2) as workers:
                calls = [
                    workers.submit(
                        handle.execute,
                        ModelRequest({"value": index}),
                        ExecutionContext(f"short-{index}", Deadline.after(0.2)),
                    )
                    for index, handle in enumerate(handles)
                ]
                assert [future.result(timeout=2).outputs["answer"] for future in calls] == [0, 2]
        wait_for(lambda: endpoint.core.resource_settled("cpu"))
    assert [event for _, event in events] == ["start", "end"] * 6


def test_profile_collector_cli_reads_real_ros_measurements(transport, tmp_path):
    import json
    import subprocess
    import sys

    from inference_service.scheduler.runtime_profiles import read_runtime_samples

    endpoint, adapters, handles, _ = transport
    target_file = tmp_path / "target.json"
    target_file.write_text(json.dumps(adapters[0].target.to_dict()))
    output = tmp_path / "models" / "_work" / "first" / "profiling" / "samples.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from inference_service.scheduler.runtime_profiles import collect_main; collect_main()",
            "--target",
            str(target_file),
            "--output",
            str(output),
            "--endpoint",
            endpoint.endpoint,
            "--count",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for(lambda: output.exists() or process.poll() is not None, timeout=15)
        assert output.exists(), process.communicate(timeout=1)
        for index in (0, 1, 0):
            handles[index].execute(ModelRequest({"value": 1}), ExecutionContext("sample", Deadline.after(5)))
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        samples = list(read_runtime_samples(output))
        assert len(samples) == 2
        assert all(s["target_id"] == adapters[0].target.target_id and s["occupancy_ns"] > 0 for s in samples)
        assert samples[1]["sequence"] - samples[0]["sequence"] == 2
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


@pytest.mark.parametrize("verb", ["unsupported", "REGISTER"])
def test_unknown_control_verbs_are_not_exposed(transport, verb):
    _, adapters, _, _ = transport
    with pytest.raises(ControlError, match="INVALID_VERB"):
        adapters[0]._call_control({"verb": verb, "epoch": adapters[0].controller.epoch}, 2)


@pytest.mark.parametrize("transport", ["fixed_only"], indirect=True)
def test_fixed_catalogue_rejects_unknown_target_over_ros(transport):
    endpoint, adapters, _, _ = transport
    status = adapters[0].controller.status()
    status["target"] = {**status["target"], "target_id": "perception/unconfigured"}
    with pytest.raises(ControlError, match="TARGET_CONFIGURATION_MISMATCH"):
        adapters[0]._call_control({"verb": "register", "status": status}, 2)
    assert len(endpoint.core.targets) == 2


@pytest.mark.parametrize("real_ort", [False, True])
def test_factory_role_session_public_ros_path_rejects_bad_role_then_executes(tmp_path, real_ort):
    from types import SimpleNamespace

    import numpy as np

    from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, runtime_context_defaults
    from inference_service.model_sessions import build_onnx_model_session
    from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeFactory, RuntimeProviders
    from inference_service.unified_runtime.registry import SessionBuilderRegistry

    from .test_onnx_model_session import _context, _fake_ort, _FakeGraph, _stateless_manifest

    context = _context(tmp_path, _stateless_manifest())
    graph = _FakeGraph(("input",), ("prob",), lambda feed: [np.asarray([sum(feed["input"])], np.float32)])
    ort = _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
    if real_ort:
        import onnx
        import onnxruntime as ort

        model_graph = onnx.helper.make_graph(
            [onnx.helper.make_node("ReduceSum", ["input"], ["prob"], keepdims=1)],
            "managed-output",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4])],
            [onnx.helper.make_tensor_value_info("prob", onnx.TensorProto.FLOAT, [1])],
        )
        model = onnx.helper.make_model(model_graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=9)
        onnx.save(model, tmp_path / "artifacts/model.onnx")
    with ExitStack() as infrastructure:
        providers = RuntimeProviders.create(object(), object())
        infrastructure.callback(providers.close)
        registries = SimpleNamespace(
            session_builder_registry=SessionBuilderRegistry(),
            backend_registry=BackendRegistry(STATIC_BACKEND_DESCRIPTORS),
        )
        with runtime_context_defaults(priority_scheduling=False, managed_execution=True):
            handle = ModelRuntimeFactory.load_session(
                context.validated_manifest,
                registry_set=registries,
                providers=providers,
                role_execution=True,
                override=build_onnx_model_session,
                builder_options={"ort_loader": lambda: ort},
            )
        unbound = infrastructure.enter_context(ExitStack())
        unbound.callback(handle.close)
        t = replace(target("factory", resource="cpu"), deployment_fingerprint=context.validated_manifest.fingerprint)
        rclpy.init()
        infrastructure.callback(rclpy.shutdown)
        nodes = []
        for name in ("factory_global", "factory_host"):
            node = Node(name)
            infrastructure.callback(node.destroy_node)
            nodes.append(node)
        executor = MultiThreadedExecutor(num_threads=4)
        spin = None
        try:
            for node in nodes:
                executor.add_node(node)
            endpoint = GlobalRuntimeEndpoint(nodes[0], [t], endpoint="/test/factory/global")
            adapter = SchedulerRuntimeAdapter(nodes[1], t, handle, endpoint="/test/factory/global")
            unbound.pop_all()  # The adapter now owns the public Close path.
            spin = threading.Thread(target=executor.spin, daemon=True)
            spin.start()
            try:
                wait_for(lambda: adapter.ready)
                with pytest.raises(ExecutionFailure) as failure:
                    handle.execute(
                        ModelRequest({"host.silero.audio": np.ones(4, np.float32)}, {"role": "missing_role"}),
                        ExecutionContext("invalid-role", Deadline.after(5)),
                    )
                assert failure.value.evidence.state.value == "not_started"
                assert not graph.feeds and handle.local_ready
                assert handle.health.failure_count == 1
                result = handle.execute(
                    ModelRequest({"host.silero.audio": np.ones(4, np.float32)}, {"role": "model"}),
                    ExecutionContext("valid-role", Deadline.after(5)),
                )
                assert result.outputs["host.silero.prob"].tolist() == [4]
                assert real_ort or len(graph.feeds) == 1
                assert handle.health.failure_count == 1
            finally:
                handle.close(ExecutionContext("factory-close", Deadline.after(5)))
                wait_for(lambda: not endpoint.core.shutdown_pending())
        finally:
            try:
                assert executor.shutdown(timeout_sec=5)
            finally:
                if spin is not None and spin.ident is not None:
                    spin.join(timeout=5)
                    assert not spin.is_alive()
        for node in nodes:
            node.executor = None
        adapter.destroy_transport()


@pytest.mark.parametrize("path", ["queue", "factory"])
@pytest.mark.parametrize("failure_at", ["construction", "close", "confirmation", "executor"])
def test_ros_harness_cleans_infrastructure_after_failure(monkeypatch, tmp_path, path, failure_at):
    import sys
    from types import SimpleNamespace

    from inference_service.unified_runtime import RuntimeProviders

    module = sys.modules[__name__]
    events, nodes = [], []

    def create_node(name):
        if failure_at == "construction" and nodes:
            raise RuntimeError("construction failed")
        nodes.append(name)

        def destroy():
            events.append(name)
            if failure_at == "close" and name == nodes[0]:
                raise RuntimeError("node cleanup also failed")

        return SimpleNamespace(destroy_node=destroy)

    close = ModelRuntimeHandle.close

    def close_handle(self, *args, **kwargs):
        close(self, *args, **kwargs)
        if failure_at == "close":
            raise RuntimeError("close failed")

    provider_close = RuntimeProviders.close

    def close_providers(self):
        events.append("providers")
        provider_close(self)

    waits = []

    def wait(predicate, timeout=5):
        waits.append(None)
        if failure_at == "confirmation" and len(waits) > 1:
            raise RuntimeError("confirmation failed")
        assert predicate()

    monkeypatch.setattr(module.rclpy, "init", lambda: None)
    monkeypatch.setattr(module.rclpy, "shutdown", lambda: events.append("context"))
    monkeypatch.setattr(module, "Node", create_node)
    monkeypatch.setattr(
        module,
        "MultiThreadedExecutor",
        lambda **_: SimpleNamespace(
            add_node=lambda _: None, spin=lambda: None, shutdown=lambda **_: failure_at != "executor"
        ),
    )
    monkeypatch.setattr(
        module,
        "GlobalRuntimeEndpoint",
        lambda *_args, **_kwargs: SimpleNamespace(core=SimpleNamespace(shutdown_pending=lambda: [])),
    )
    monkeypatch.setattr(
        module,
        "SchedulerRuntimeAdapter",
        lambda _node, target, _handle, **_kwargs: SimpleNamespace(
            target=target, ready=True, destroy_transport=lambda: None
        ),
    )
    monkeypatch.setattr(module, "wait_for", wait)
    monkeypatch.setattr(ModelRuntimeHandle, "close", close_handle)
    monkeypatch.setattr(RuntimeProviders, "close", close_providers)
    with pytest.raises((RuntimeError, AssertionError)) as failed:
        if path == "queue":
            fixture = transport.__wrapped__(SimpleNamespace())
            next(fixture)
            fixture.close()
        else:
            test_factory_role_session_public_ros_path_rejects_bad_role_then_executes(tmp_path, False)
    errors = []
    error = failed.value
    while error is not None:
        errors.append(error)
        error = error.__context__
    assert any(
        isinstance(error, AssertionError) if failure_at == "executor" else f"{failure_at} failed" in str(error)
        for error in errors
    )
    assert all(name in events for name in nodes)
    assert "context" in events
    assert ("providers" in events) == (path == "factory")


def test_invalid_global_service_name_leaves_original_handle_retryable():
    from .runtime_scheduler_helpers import Executor

    rclpy.init()
    node = Node("invalid_endpoint_host")
    handle = ModelRuntimeHandle(executor=Executor())
    handle.load()
    adapter = None
    try:
        with pytest.raises(ValueError, match="invalid runtime endpoint"):
            SchedulerRuntimeAdapter(node, target(resource="cpu"), handle, endpoint="/invalid endpoint")
        assert not handle.execution_policy_bound
        assert handle.execute(ModelRequest(), ExecutionContext("direct")).outputs["answer"].tolist() == [1]
        adapter = SchedulerRuntimeAdapter(node, target(resource="cpu"), handle, endpoint="/valid_endpoint")
        assert handle.execution_policy_bound
    finally:
        if adapter is not None:
            adapter.controller.shutdown()
            handle._close_local(ExecutionContext("cleanup"), _dispatch_key=adapter.controller._authority)
            adapter.destroy_transport()
        else:
            handle.close()
        node.destroy_node()
        rclpy.shutdown()
