"""Real native lifecycle and typed protocol with an explicitly fake engine.

These tests never import TensorRT/CUDA, publish robot labels or run a robot.
The fake proves wiring and failure behavior, not hardware performance.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from conftest import FakeAclRuntimeManager
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image

from embodied_common.hand_gesture_contract import RequestMeta, ResultMeta
from ibrobot_msgs.srv import ClassifyHandGesture
from inference_manifest import load_inference_manifest
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, ResourceDomainAdmissions
from inference_service.model_service_node import ModelServiceNode, _runtime_info
from inference_service.model_service_plugin import ModelServicePlugin
from inference_service.unified_runtime import (
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeProviders,
    SessionBuilderRegistry,
)
from perception_service.hand_gesture_adapter import INPUT_SHAPE, OUTPUT_SHAPE, VERIFIED_ENGINE_SHA256
from perception_service.hand_gesture_plugin import HandGestureClassifyPlugin
from perception_service.hand_gesture_session import TensorRtHandGestureSession
from perception_service.package_hand_gesture_bundle import DEPLOYMENT, package_hand_gesture_bundle


class Clock:
    value = 100.0

    def __call__(self):
        return self.value

    def step(self, seconds=0.2):
        self.value += seconds


class Runner:
    runtime_version = "fake-tensorrt-10.7.0"

    def __init__(self, blob, **_options):
        self.blob = blob
        self.calls, self.closes = [], 0
        self.quiescent = True
        self.output = np.zeros(OUTPUT_SHAPE, np.float32)
        self.output[0, 4 + 8, 0] = 0.9
        self.error = None
        self.entered, self.release = threading.Event(), None

    def infer(self, x):
        self.calls.append(x.copy())
        self.quiescent = False
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5), "test failed to release fake execution"
        if self.error is not None:
            raise self.error
        self.quiescent = True
        return self.output.copy()

    def close(self):
        self.closes += 1
        self.quiescent = True


@pytest.fixture
def setup_plugin(tmp_path, monkeypatch):
    monkeypatch.setattr("perception_service.hand_gesture_session.Path.home", lambda: tmp_path)
    monkeypatch.setattr("perception_service.hand_gesture_session.resource_headroom_reason", lambda: "")
    engine = tmp_path / "fake.engine"
    engine.write_bytes(b"explicitly-fake-engine")
    root = package_hand_gesture_bundle(
        tmp_path / "bundle", engine, expected_sha256=hashlib.sha256(engine.read_bytes()).hexdigest()
    )
    validated = load_inference_manifest(root, DEPLOYMENT)
    registry = RegistrySet(
        BackendRegistry(STATIC_BACKEND_DESCRIPTORS), SessionBuilderRegistry(), RuntimeAssemblerRegistry()
    ).freeze()
    providers = RuntimeProviders.create(FakeAclRuntimeManager(), ResourceDomainAdmissions())
    clock, runners, plugins = Clock(), [], []

    def factory(blob, **options):
        runner = Runner(blob, **options)
        runners.append(runner)
        return runner

    class Plugin(HandGestureClassifyPlugin):
        _session_factory = staticmethod(lambda: TensorRtHandGestureSession(runner_factory=factory))
        _clock = staticmethod(clock)

    def create(options=None):
        plugin = Plugin(None, validated, options or {}, registry_set=registry, providers=providers)
        plugins.append(plugin)
        return plugin

    plugin = create()
    yield SimpleNamespace(
        plugin=plugin, runner=runners[0], clock=clock, create=create, runners=runners, validated=validated, root=root
    )
    for item in plugins:
        item.close()
    providers.close()


def request(*, op="infer", request_id=1, source=None, **changes):
    meta = source or RequestMeta(
        requester_boot_id=str(uuid.uuid4()),
        producer_start_mono_ns=900000000000,
        stream_generation=1,
        request_id=1,
        op="infer",
        session_token=str(uuid.uuid4()),
        actor_id=7,
        stamp_ns=1_234_567_890,
        pear_stamp_ns=1_234_000_000,
        crop="both",
        roi=(-10, 20, 310, 340),
        source_size=(1344, 972),
        budget_ms=400.0,
    )
    meta = replace(meta, op=op, request_id=request_id, **changes)
    image = Image()
    if op == "infer":
        image.width, image.height, image.step, image.encoding = 320, 320, 960, "bgr8"
        image.header.stamp.sec, image.header.stamp.nanosec = divmod(meta.stamp_ns, 1_000_000_000)
        image.header.frame_id = "original_optical_frame"
        image.data = np.tile(np.array([0, 127, 255], np.uint8), (320, 320, 1)).tobytes()
    return ClassifyHandGesture.Request(image=image, metadata_json=meta.to_json()), meta


def dispatch(plugin, value):
    host = SimpleNamespace(plugin=plugin, failure_reason="", get_parameter=lambda _n: SimpleNamespace(value="test"))
    host.runtime_info = lambda: _runtime_info("gesture-test", plugin.validated, plugin.runtime_status())
    return ModelServiceNode._dispatch(host, value, ClassifyHandGesture.Response())


def result(response):
    return ResultMeta.from_json(response.result_json)


def test_infer_echoes_entire_context_and_standard_diagnostics(setup_plugin):
    setup = setup_plugin
    req, meta = request()
    response = dispatch(setup.plugin, req)
    record = result(response)
    assert response.success and response.model.ready
    assert response.model.backend == "tensorrt"
    assert response.model.runtime_version == Runner.runtime_version
    assert response.model.manifest_fingerprint == setup.validated.manifest.bundle.digest.value
    assert response.inference_time_ms >= 0
    assert record.request == meta and record.gesture == "heart" and record.score == pytest.approx(0.9)
    assert record.server_boot_id == setup.plugin.server_boot_id
    assert record.engine_sha256 == hashlib.sha256(setup.runner.blob).hexdigest()
    assert record.idle and len(setup.runner.calls) == 1
    x = setup.runner.calls[0]
    assert x.shape == INPUT_SHAPE and x.dtype == np.float32 and x.flags.c_contiguous
    assert x[0, :, 0, 0].tolist() == pytest.approx([1.0, 127 / 255.0, 0.0])


@pytest.mark.parametrize("op", ["probe", "retire"])
def test_control_never_infers_and_boot_identity_stays_fixed(setup_plugin, op):
    setup = setup_plugin
    req, meta = request(op=op, request_id=2, **({"retire_through": 1} if op == "retire" else {}))
    response = dispatch(setup.plugin, req)
    assert response.success and not setup.runner.calls
    out = result(response)
    assert out.request == meta and out.idle and out.server_boot_id == setup.plugin.server_boot_id
    assert out.status == ("retired" if op == "retire" else "ok")


@pytest.mark.parametrize(
    "field,value",
    [("encoding", "rgb8"), ("width", 319), ("height", 319), ("step", 961), ("data", b"short"), ("is_bigendian", 1)],
)
def test_malformed_crop_is_failure_not_valid_none(setup_plugin, field, value):
    req, _ = request()
    setattr(req.image, field, value)
    response = dispatch(setup_plugin.plugin, req)
    assert not response.success and not setup_plugin.runner.calls
    assert result(response).status == "rejected"


def test_stamp_mismatch_is_rejected(setup_plugin):
    req, _ = request()
    req.image.header.stamp.nanosec += 1
    response = dispatch(setup_plugin.plugin, req)
    assert not response.success and not setup_plugin.runner.calls
    assert result(response).status == "rejected"


@pytest.mark.parametrize(
    "metadata",
    [
        "{}",
        '{"schema":"wrong"}',
        '{"schema":"x2_hand_gesture_request/v1","schema":"duplicate"}',
        '{"budget_ms":NaN}',
        "x" * 4097,
    ],
)
def test_bad_protocol_has_no_fabricated_result(setup_plugin, metadata):
    req, _ = request()
    req = SimpleNamespace(image=req.image, metadata_json=metadata)
    response = dispatch(setup_plugin.plugin, req)
    assert not response.success and response.result_json == "" and not setup_plugin.runner.calls


def test_valid_none_remains_distinct_from_model_error(setup_plugin):
    setup = setup_plugin
    setup.runner.output.fill(0)
    req, _ = request()
    response = dispatch(setup.plugin, req)
    assert response.success and result(response).gesture == "none" and result(response).status == "ok"
    setup.clock.step()
    setup.runner.error = RuntimeError("fake CUDA failure")
    req, _ = request(request_id=2)
    response = dispatch(setup.plugin, req)
    assert not response.success and result(response).status == "error"
    assert not response.model.ready and not result(response).idle


@pytest.mark.parametrize("fault", ["nan", "inf", "shape", "dtype", "range"])
def test_bad_model_output_fails(setup_plugin, fault):
    setup = setup_plugin
    if fault in ("nan", "inf"):
        setup.runner.output[0, 4, 0] = float(fault)
    elif fault == "shape":
        setup.runner.output = np.zeros((1, 38, 2), np.float32)
    elif fault == "dtype":
        setup.runner.output = setup.runner.output.astype(np.float64)
    else:
        setup.runner.output[0, 4, 0] = 1.1
    req, _ = request()
    response = dispatch(setup.plugin, req)
    assert not response.success
    assert result(response).status == "error"


def test_retire_during_blocked_execution_never_claims_idle_and_suppresses_old_result(setup_plugin):
    setup = setup_plugin
    setup.runner.release = threading.Event()
    req, meta = request()
    output = []
    thread = threading.Thread(target=lambda: output.append(dispatch(setup.plugin, req)))
    thread.start()
    try:
        assert setup.runner.entered.wait(2)
        retire, retired_meta = request(op="retire", request_id=2, source=meta, retire_through=1)
        response = dispatch(setup.plugin, retire)
        assert not response.success and result(response).status == "busy" and not result(response).idle
        assert result(response).request == retired_meta and result(response).retired_through == 1
        new, _ = request(request_id=3, source=meta)
        assert result(dispatch(setup.plugin, new)).status == "busy"
        assert len(setup.runner.calls) == 1
    finally:
        setup.runner.release.set()
        thread.join(3)
    assert not thread.is_alive() and len(output) == 1 and not output[0].success
    assert result(output[0]).status == "rejected"
    response = dispatch(setup.plugin, retire)
    assert response.success and result(response).idle and result(response).retired_through == 1
    assert not dispatch(setup.plugin, req).success
    assert len(setup.runner.calls) == 1


def test_error_controls_bypass_unready_gate_without_running_model(setup_plugin):
    setup = setup_plugin
    setup.runner.error = RuntimeError("fake stream synchronization failed")
    req, meta = request()
    assert not dispatch(setup.plugin, req).success
    probe, _ = request(op="probe", request_id=2, source=meta)
    response = dispatch(setup.plugin, probe)
    assert not response.success and result(response).status == "not_ready"
    assert result(response).server_boot_id == setup.plugin.server_boot_id
    retire, _ = request(op="retire", request_id=3, source=meta, retire_through=1)
    response = dispatch(setup.plugin, retire)
    assert not response.success and not result(response).idle
    # Only an actual completion proof, never client cancellation, permits idle.
    setup.runner.quiescent = True
    response = dispatch(setup.plugin, retire)
    assert response.success and result(response).status == "retired" and result(response).idle
    assert len(setup.runner.calls) == 1


@pytest.mark.parametrize("when", ["before", "after"])
def test_budget_uses_local_elapsed_time(setup_plugin, monkeypatch, when):
    setup = setup_plugin
    if when == "before":
        old = setup.plugin.adapter.preprocess

        def preprocess(image):
            setup.clock.step(0.5)
            return old(image)

        monkeypatch.setattr(setup.plugin.adapter, "preprocess", preprocess)
    else:
        old = setup.runner.infer

        def infer(x):
            setup.clock.step(0.5)
            return old(x)

        monkeypatch.setattr(setup.runner, "infer", infer)
    req, _ = request(producer_start_mono_ns=8000000000000000)
    response = dispatch(setup.plugin, req)
    assert not response.success and result(response).status == "rejected"
    assert len(setup.runner.calls) == (0 if when == "before" else 1)


def test_duplicate_retirement_rate_and_restart_identity(setup_plugin):
    setup = setup_plugin
    req, meta = request()
    assert dispatch(setup.plugin, req).success
    assert not dispatch(setup.plugin, req).success
    fast, _ = request(request_id=2, source=meta)
    assert result(dispatch(setup.plugin, fast)).status == "busy"
    assert len(setup.runner.calls) == 1
    setup.clock.step()
    next_req, _ = request(request_id=3, source=meta)
    assert dispatch(setup.plugin, next_req).success
    boot = setup.plugin.server_boot_id
    setup.plugin.close()
    setup.plugin.close()
    assert setup.runner.closes == 1
    restarted = setup.create()
    assert restarted.server_boot_id != boot
    assert len(setup.runners) == 2


def test_source_ledger_is_bounded_and_does_not_evict_tombstones(setup_plugin):
    setup = setup_plugin
    for _ in range(64):
        req, _ = request(op="retire", request_id=2, retire_through=1)
        assert dispatch(setup.plugin, req).success
    req, _ = request(op="probe")
    response = dispatch(setup.plugin, req)
    assert not response.success and result(response).status == "rejected"
    assert len(setup.plugin._sources) == 64 and not setup.runner.calls


def test_stream_generations_are_compacted_without_reaccepting_old_frames(setup_plugin):
    setup = setup_plugin
    old_request, original = request()
    for generation in range(1, 129):
        retire, _ = request(op="retire", request_id=2, source=original, stream_generation=generation, retire_through=1)
        response = dispatch(setup.plugin, retire)
        assert response.success and result(response).idle
    assert len(setup.plugin._sources) == 1
    assert not dispatch(setup.plugin, old_request).success
    late_retire, old_meta = request(op="retire", request_id=3, source=original, retire_through=2)
    response = dispatch(setup.plugin, late_retire)
    assert response.success and result(response).retired_through == 2 and result(response).request == old_meta
    current, _ = request(request_id=4, source=original, stream_generation=128)
    assert dispatch(setup.plugin, current).success
    assert len(setup.runner.calls) == 1


def test_native_session_loads_once_and_actual_digest_is_checked(setup_plugin):
    setup = setup_plugin
    for i in range(3):
        req, _ = request(request_id=i + 1)
        assert dispatch(setup.plugin, req).success
        setup.clock.step()
    assert len(setup.runners) == 1
    artifact = setup.validated.deployment.artifacts["model"]
    (setup.root / artifact.path).write_bytes(b"tampered")
    setup.plugin.close()
    with pytest.raises(Exception, match="SHA-256"):
        setup.create()
    assert len(setup.runners) == 1


def test_second_engine_cannot_overlap_existing_device_lease(setup_plugin):
    setup = setup_plugin
    with pytest.raises(Exception, match="multiple live instances"):
        setup.create()
    registry = RegistrySet(
        BackendRegistry(STATIC_BACKEND_DESCRIPTORS), SessionBuilderRegistry(), RuntimeAssemblerRegistry()
    ).freeze()
    providers = RuntimeProviders.create(FakeAclRuntimeManager(), ResourceDomainAdmissions())
    try:
        # A separate provider's admission pool cannot bypass the per-device OS lease.
        with pytest.raises(Exception, match="lease"):
            type(setup.plugin)(None, setup.validated, {}, registry_set=registry, providers=providers)
    finally:
        providers.close()
    assert len(setup.runners) == 1
    probe, _ = request(op="probe")
    assert dispatch(setup.plugin, probe).success


def test_default_plugins_do_not_opt_into_unready_requests():
    assert ModelServicePlugin.accepts_unready_request(None, object()) is False


@pytest.mark.parametrize(
    "option,value",
    [
        ("confidence", float("nan")),
        ("confidence", 0),
        ("max_infer_rate_hz", 11),
        ("max_infer_rate_hz", True),
        ("backend", "torch"),
        ("fallback", True),
    ],
)
def test_invalid_runtime_options_fail_before_loading(setup_plugin, option, value):
    setup = setup_plugin
    setup.plugin.close()
    with pytest.raises(ValueError):
        setup.create({option: value})
    assert len(setup.runners) == 1


def test_packager_never_overwrites_or_accepts_wrong_engine(tmp_path):
    source = tmp_path / "engine"
    source.write_bytes(b"not-the-reviewed-engine")
    with pytest.raises(ValueError, match="SHA-256"):
        package_hand_gesture_bundle(tmp_path / "bundle", source)
    assert not (tmp_path / "bundle").exists()
    (tmp_path / "bundle").mkdir()
    with pytest.raises(FileExistsError):
        package_hand_gesture_bundle(tmp_path / "bundle", source)


def test_review_manifest_matches_packager_and_pins_verified_artifact(setup_plugin):
    root = Path(__file__).resolve().parents[1] / "config" / "hand_gesture_bundle"
    template = json.loads((root / "inference_manifest.json").read_text())
    generated = json.loads((setup_plugin.root / "inference_manifest.json").read_text())
    generated["deployments"][DEPLOYMENT]["artifacts"]["model"]["sha256"] = VERIFIED_ENGINE_SHA256
    assert template == generated
    assert (root / "assets" / "adapter.json").read_bytes() == (
        setup_plugin.root / "assets" / "adapter.json"
    ).read_bytes()
    assert not list(root.rglob("*.engine"))


@pytest.mark.parametrize(
    "class_id,name", [(8, "heart"), (9, "heart"), (2, "holy"), (7, "xsign"), (33, "none"), (1, "none")]
)
def test_postprocessing_mapping_and_none(setup_plugin, class_id, name):
    setup = setup_plugin
    setup.runner.output.fill(0)
    setup.runner.output[0, 4 + class_id, 0] = 0.8
    req, _ = request()
    response = dispatch(setup.plugin, req)
    assert response.success and result(response).gesture == name


def test_control_image_must_be_empty(setup_plugin):
    req, _ = request(op="retire", request_id=2, retire_through=1)
    req.image.width = 320
    response = dispatch(setup_plugin.plugin, req)
    assert not response.success and result(response).status == "rejected"
    assert not setup_plugin.runner.calls


def test_resource_pressure_pauses_admission_and_recovers_without_inference(setup_plugin, monkeypatch):
    setup = setup_plugin
    monkeypatch.setattr("perception_service.hand_gesture_session.resource_headroom_reason", lambda: "low memory")
    req, _ = request()
    response = dispatch(setup.plugin, req)
    assert not response.success and not response.model.ready and not setup.runner.calls
    probe, _ = request(op="probe")
    response = dispatch(setup.plugin, probe)
    assert not response.success and result(response).status == "not_ready"
    monkeypatch.setattr("perception_service.hand_gesture_session.resource_headroom_reason", lambda: "")
    assert dispatch(setup.plugin, req).success
    assert len(setup.runner.calls) == 1 and len(setup.runners) == 1


def test_real_ros_host_keeps_retirement_behind_actual_execution(setup_plugin, monkeypatch):
    """Exercise generated ROS messages, the real host and its exclusive group."""
    setup = setup_plugin
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    setup.plugin.close()
    registry = RegistrySet(
        BackendRegistry(STATIC_BACKEND_DESCRIPTORS), SessionBuilderRegistry(), RuntimeAssemblerRegistry()
    ).freeze()
    providers = RuntimeProviders.create(FakeAclRuntimeManager(), ResourceDomainAdmissions())
    monkeypatch.setattr("inference_service.model_service_node._load_class", lambda _name: type(setup.plugin))
    endpoint = f"/test/hand_gesture_{uuid.uuid4().hex}"
    parameters = {
        "bundle_path": str(setup.root),
        "deployment": DEPLOYMENT,
        "adapter_class": "test:InjectedFakeEnginePlugin",
        "service_type": "ibrobot_msgs/srv/ClassifyHandGesture",
        "service_endpoint": endpoint,
        "required": "true",
        "require_semantic_identity": "true",
    }
    args = ["--ros-args"]
    for name, value in parameters.items():
        args += ["-p", f"{name}:={value}"]
    rclpy.init(args=args)
    server = client_node = executor = thread = None
    active_runner = None
    try:
        server = ModelServiceNode(registry_set=registry, providers=providers)
        client_node = rclpy.create_node(f"gesture_client_{uuid.uuid4().hex}")
        client = client_node.create_client(ClassifyHandGesture, endpoint)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(server)
        executor.add_node(client_node)
        thread = threading.Thread(target=executor.spin)
        thread.start()
        assert client.wait_for_service(timeout_sec=5)

        def wait(future):
            done = threading.Event()
            future.add_done_callback(lambda _future: done.set())
            assert done.wait(5), "ROS test service response timed out"
            return future.result()

        probe, _ = request(op="probe")
        response = wait(client.call_async(probe))
        assert (
            response.success and response.model.ready and result(response).engine_sha256 == server.plugin.engine_sha256
        )
        active_runner = setup.runners[-1]
        assert not active_runner.calls
        active_runner.release = threading.Event()
        infer, meta = request(request_id=2)
        inference = client.call_async(infer)
        assert active_runner.entered.wait(3)
        # Cancelling the client's wait does not cancel native model work.
        inference.cancel()
        assert not active_runner.quiescent
        retire, _ = request(op="retire", request_id=3, source=meta, retire_through=2)
        retirement = client.call_async(retire)
        assert not threading.Event().wait(0.05)
        assert not retirement.done() and len(active_runner.calls) == 1
        active_runner.release.set()
        response = wait(retirement)
        assert response.success and result(response).status == "retired" and result(response).idle
        assert result(response).retired_through == 2
        assert len(active_runner.calls) == 1
        response = wait(client.call_async(infer))
        assert not response.success and result(response).status == "rejected"
        assert len(active_runner.calls) == 1
    finally:
        if active_runner is not None and active_runner.release is not None:
            active_runner.release.set()
        if executor is not None:
            executor.shutdown(timeout_sec=5)
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive()
        if client_node is not None:
            client_node.destroy_node()
        if server is not None:
            server.destroy_node()
        providers.close()
        rclpy.try_shutdown()
