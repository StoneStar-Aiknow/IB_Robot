"""The real Global process must serve host Close after a launch-wide stop."""

import json
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from inference_service.scheduler.runtime_binding import bind_runtime
from inference_service.scheduler.runtime_protocol import RuntimeExecutionTarget
from inference_service.unified_runtime import (
    Deadline,
    ExecutionContext,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    OutcomeState,
)

pytestmark = [
    pytest.mark.usefixtures("simulated_runtime_construction"),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_signal_during_ros_callback_load_prevents_publication(tmp_path, stop_signal):
    import textwrap

    events, release = tmp_path / "events", tmp_path / "release"
    script = tmp_path / "loading_host.py"
    script.write_text(
        textwrap.dedent("""
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace
        import rclpy
        from rclpy.node import Node
        import inference_service.host_lifecycle as lifecycle
        from inference_service.runtime_creation import runtime_creation
        from inference_service.unified_runtime import ModelRuntimeHandle, RuntimeProviders
        from inference_service.unified_runtime.lifecycle import lifecycle_remaining

        events, release = map(Path, sys.argv[1:])
        def record(event):
            with events.open("a") as output:
                output.write(event + "\\n")
        class Stop(lifecycle.LifecycleStop):
            def set(self):
                super().set()
                record("stopping")
        lifecycle.LifecycleStop = Stop
        class Model:
            def load(self):
                record("loading")
                until = time.monotonic() + 10
                while not release.exists():
                    assert time.monotonic() < until
                    time.sleep(0.01)
            def execute(self, *_):
                return {}
            def close(self):
                assert 0 < lifecycle_remaining(60) <= 3
                record("model_closed")
        class Host(Node):
            def __init__(self, deps, retain):
                super().__init__("loading_signal_host")
                retain(self)
                self.deps = deps
                self.timer = self.create_timer(0.01, self.load)
            def load(self):
                self.timer.cancel()
                try:
                    with runtime_creation(providers=self.deps.providers, node=self, timeout=60):
                        model = ModelRuntimeHandle(Model())
                        model.load()
                    record("published")
                except RuntimeError as exc:
                    assert "stopped during initialization" in str(exc)
                    record("startup_stopped")
        lifecycle.HOST_SHUTDOWN_TIMEOUT_SEC = 3
        lifecycle.run_model_node(Host, lambda: SimpleNamespace(providers=RuntimeProviders.create(object(), object())), args=[])
        assert not rclpy.ok()
        record("ros_shutdown")
    """)
    )
    with (tmp_path / "host.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), str(events), str(release)], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            until = time.monotonic() + 15
            while not events.exists() or "loading" not in events.read_text():
                assert process.poll() is None and time.monotonic() < until
                time.sleep(0.01)
            process.send_signal(stop_signal)
            while "stopping" not in events.read_text().splitlines():
                assert process.poll() is None and time.monotonic() < until
                time.sleep(0.01)
            release.touch()
            assert process.wait(timeout=10) == 0
            assert events.read_text().splitlines() == [
                "loading",
                "stopping",
                "model_closed",
                "startup_stopped",
                "ros_shutdown",
            ]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            log.seek(0)
            print(log.read())


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("host_count", [1, 2])
@pytest.mark.parametrize("lost_unregister", [False, True])
def test_global_process_keeps_transport_until_host_close(
    tmp_path, stop_signal, host_count, lost_unregister, monkeypatch
):
    from inference_service.scheduler import runtime_ros

    reply = runtime_ros._reply
    receipts = []

    def receive(response):
        value = reply(response)
        if value.get("state") == "unregistered":
            receipts.append(value)
            if lost_unregister and len(receipts) == 1:
                raise TimeoutError("lost final unregister response")
        return value

    monkeypatch.setattr(runtime_ros, "_reply", receive)
    rclpy.init()
    hosts = [Node(f"shutdown_host_{index}") for index in range(host_count)]
    executor = MultiThreadedExecutor(num_threads=4)
    for host in hosts:
        executor.add_node(host)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    closed = []

    class Model:
        def execute(self, request, context):
            return {"answer": 1}

        def close(self):
            closed.append(True)

    handles, bindings = [], []
    for index, host in enumerate(hosts):
        handle = ModelRuntimeHandle(executor=Model())
        handle.load()
        target = RuntimeExecutionTarget(
            target_id=f"perception/shutdown_{index}",
            endpoint=f"/runtime/shutdown_host_{index}",
            hardware_resource_id="cpu",
            physical_device="local/cpu",
            deployment_fingerprint="test",
            composition_fingerprint="test",
            lifecycle_group=f"shutdown_{index}",
            configuration_generation="test",
            interface="tensor_model",
            model_type="test",
            operation="infer",
        )
        handles.append(handle)
        bindings.append(bind_runtime(host, handle, target))
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text(
        json.dumps(
            {
                "global_runtime_scheduler": {
                    "ros__parameters": {
                        "runtime_targets_json": "[]",
                        "runtime_resources_json": json.dumps({"cpu": {"physical_device": "local/cpu"}}),
                    }
                }
            }
        )
    )
    with (tmp_path / "global.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "inference_service.global_runtime_scheduler_node",
                "--ros-args",
                "--params-file",
                str(parameters),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            expires = time.monotonic() + 15
            while not all(binding.ready for binding in bindings):
                assert process.poll() is None
                assert time.monotonic() < expires
                time.sleep(0.02)
            for handle in handles:
                # Registration can beat reverse DDS discovery. Establish the
                # working channel before testing signal/retirement behavior;
                # only a proven unsent request may be retried here.
                while True:
                    try:
                        result = handle.execute(ModelRequest(), ExecutionContext("first", Deadline.after(3)))
                    except ExecutionFailure as exc:
                        if (
                            exc.code != "HOST_UNAVAILABLE"
                            or exc.evidence.state is not OutcomeState.NOT_STARTED
                            or time.monotonic() >= expires
                        ):
                            raise
                        time.sleep(0.02)
                    else:
                        assert result.outputs == {"answer": 1}
                        break
            process.send_signal(stop_signal)
            time.sleep(0.2)
            assert process.poll() is None, "Global exited before the registered host closed"
            with ThreadPoolExecutor(host_count) as pool:
                futures = [
                    pool.submit(handle.close, ExecutionContext("shutdown", Deadline.after(5))) for handle in handles
                ]
                for future in futures:
                    future.result(timeout=6)
            assert len(closed) == host_count
            assert len(receipts) >= host_count + int(lost_unregister)
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            executor.shutdown(timeout_sec=5)
            spin.join(timeout=5)
            assert not spin.is_alive()
            for host in hosts:
                host.destroy_node()
            rclpy.shutdown()
            log.seek(0)
            print(log.read())


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("deferred_load", [False, True])
def test_public_host_process_completes_signal_to_ros_shutdown(tmp_path, stop_signal, deferred_load):
    """Exercise the public host entry, detached executor Close and providers."""
    import textwrap

    events = tmp_path / "events"
    host_script = tmp_path / "host.py"
    host_script.write_text(
        textwrap.dedent("""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        import rclpy
        from rclpy.node import Node
        from inference_service.host_lifecycle import HostRequests, close_node_runtimes, run_model_node
        from inference_service.runtime_creation import runtime_creation
        from inference_service.scheduler.runtime_environment import RuntimeEnvironment
        from inference_service.scheduler.runtime_protocol import RuntimeExecutionTarget
        from inference_service.unified_runtime import Deadline, ExecutionContext, ModelRequest, ModelRuntimeHandle, RuntimeProviders
        from inference_service.unified_runtime.errors import CancellationRequested

        events = Path(sys.argv[1])
        def record(event):
            with events.open("a") as output:
                output.write(event + "\\n")
        def dependencies():
            return SimpleNamespace(providers=RuntimeProviders.create(
                SimpleNamespace(close=lambda: record("acl_closed")),
                SimpleNamespace(close=lambda: record("providers_closed")),
                runtime_environment=RuntimeEnvironment(scheduler_enabled=True),
            ))
        class Host(Node):
            def __init__(self, deps, retain):
                super().__init__("public_signal_host")
                self.requests = HostRequests()
                retain(self)
                self.deps = deps
                self.model = None
                if sys.argv[2] == "immediate":
                    self.load_model()
                self.timer = self.create_timer(0.02, self.ready)
            def load_model(self):
                target = RuntimeExecutionTarget(
                    target_id="perception/signal", endpoint="/runtime/public_signal_host",
                    hardware_resource_id="cpu", physical_device="local/cpu",
                    deployment_fingerprint="test", composition_fingerprint="test",
                    lifecycle_group="signal", configuration_generation="test",
                    interface="tensor_model", model_type="test", operation="infer",
                )
                with runtime_creation(providers=self.deps.providers, node=self, targets=[target.to_dict()], timeout=10):
                    self.model = ModelRuntimeHandle(SimpleNamespace(execute=lambda *_: {"answer": 1}, reset=lambda: None, close=lambda: record("model_closed")), resettable=True)
                    self.model.load()
            def ready(self):
                if self.model is None:
                    self.load_model()
                if self.model.ready:
                    with self.requests.request(ExecutionContext("callback", Deadline.after(3))) as context:
                        assert self.model.execute(ModelRequest(), context).outputs == {"answer": 1}
                        self.model.reset(context)
                    record("ready")
                    self.timer.cancel()
            def destroy_node(self):
                assert self.executor is None or self not in self.executor.get_nodes()
                try:
                    with self.requests.request(ExecutionContext("after-stop")):
                        raise AssertionError("host still admits requests")
                except CancellationRequested:
                    record("requests_stopped")
                close_node_runtimes(self)
                assert self.model.closed
                record("host_closed")
                return super().destroy_node()
        run_model_node(Host, dependencies, args=[], drain_callbacks=True, request_tracker=lambda node: node.requests)
        assert not rclpy.ok()
        record("ros_shutdown")
    """)
    )
    # This Global has no required startup host; dynamic registration supplies it.
    parameters = tmp_path / "public-host-parameters.yaml"
    parameters.write_text(
        json.dumps(
            {
                "global_runtime_scheduler": {
                    "ros__parameters": {
                        "runtime_targets_json": "[]",
                        "runtime_resources_json": json.dumps({"cpu": {"physical_device": "local/cpu"}}),
                    }
                }
            }
        )
    )
    with (tmp_path / "global.log").open("w+") as global_log, (tmp_path / "host.log").open("w+") as host_log:
        global_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "inference_service.global_runtime_scheduler_node",
                "--ros-args",
                "--params-file",
                str(parameters),
            ],
            stdout=global_log,
            stderr=subprocess.STDOUT,
        )
        host_process = subprocess.Popen(
            [sys.executable, str(host_script), str(events), "deferred" if deferred_load else "immediate"],
            stdout=host_log,
            stderr=subprocess.STDOUT,
        )
        try:
            expires = time.monotonic() + 15
            while not events.exists() or "ready" not in events.read_text().splitlines():
                assert global_process.poll() is None and host_process.poll() is None
                assert time.monotonic() < expires
                time.sleep(0.02)
            global_process.send_signal(stop_signal)
            host_process.send_signal(stop_signal)
            assert host_process.wait(timeout=20) == 0
            assert global_process.wait(timeout=20) == 0
            assert events.read_text().splitlines() == [
                "ready",
                "requests_stopped",
                "model_closed",
                "host_closed",
                "providers_closed",
                "acl_closed",
                "ros_shutdown",
            ]
        finally:
            for process in (host_process, global_process):
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            for log in (host_log, global_log):
                log.seek(0)
                print(log.read())
