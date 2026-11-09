"""Focused unified-runtime ownership tests for Speech Direction streaming."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from inference_service.unified_runtime import (  # noqa: E402
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    LifecycleState,
    ModelRequest,
    ModelRuntimeHandle,
    OwnedComponent,
    RuntimeAssembly,
)
from voice_asr_service.speech_direction.model_sessions import (  # noqa: E402
    SpeechDirectionRoleRunner,
    SpeechDirectionSessionResources,
)
from voice_asr_service.speech_direction.runtime import SpeechDirectionRuntime  # noqa: E402
from voice_asr_service.speech_direction.speech_gate import SileroVadEngine  # noqa: E402
from voice_asr_service.speech_direction.streaming_runtime import (  # noqa: E402
    SpeechDirectionStreamingRuntime,
)


class _Pipeline:
    hop_size = 2
    frame_size = 4
    sr = 16000

    def __init__(self) -> None:
        self.vad_state = SimpleNamespace(get=lambda: (0.8, True))
        self.doa_state = SimpleNamespace()
        self.reset_calls = 0
        self.processed: list[tuple[np.ndarray, int | None]] = []
        self.close_calls: list[bool] = []
        self.processed_event = threading.Event()

    def reset(self) -> None:
        self.reset_calls += 1

    def process_block(self, audio: np.ndarray, *, capture_start_sample: int | None = None):
        self.processed.append((audio.copy(), capture_start_sample))
        self.processed_event.set()
        return SimpleNamespace(seg_output=90, seg_seq=len(self.processed))

    def close(self, *, close_backends: bool = True) -> None:
        self.close_calls.append(close_backends)


def test_native_stream_history_is_bounded_and_old_owned_handles_close_idempotently():
    runtime = SpeechDirectionStreamingRuntime(_Pipeline())
    context = ExecutionContext("history")
    runtime.load()
    first = None
    for _ in range(260):
        stream = runtime.open_stream(context)
        if first is None:
            first = stream
        runtime.close_stream(stream, context)
    assert len(runtime._closed_stream_ids) == 128
    runtime.close_stream(first, context)
    current = runtime.open_stream(context)
    assert current.stream_id != first.stream_id
    runtime.close_stream(current, context)
    runtime.close()


class _Resource:
    def __init__(self) -> None:
        self.load_calls = 0
        self.load_contexts: list[object] = []
        self.close_calls = 0

    def load(self, _context: object) -> None:
        self.load_calls += 1
        self.load_contexts.append(_context)

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("recoverable", [False, True])
def test_hop_failure_preserves_native_reload_requirement(recoverable):
    from inference_service.unified_runtime import OutcomeEvidence, RecoveryAction, RecoveryRequirement

    handle, streaming, pipeline, _ = _handle()
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))
    original = ExecutionFailure(
        "native_reload",
        "native runtime must be replaced",
        recoverable=recoverable,
        recovery=RecoveryRequirement("runtime", "reload"),
        evidence=OutcomeEvidence.completed("native"),
    )

    def fail(*_, **__):
        raise original

    pipeline.process_block = fail
    try:
        request = ModelRequest({"audio": np.ones((6, 2), np.float32)})
        with pytest.raises(ExecutionFailure) as adapted:
            streaming.step(streaming.active_stream, request, ExecutionContext("adapter"))
        assert adapted.value.recovery is original.recovery
        assert adapted.value.recoverable is recoverable
        with pytest.raises(ExecutionFailure) as caught:
            handle.step(stream, request, ExecutionContext("hop"))
        assert caught.value.recovery.action is RecoveryAction.RELOAD
        # This fixture has no reload factory. Handle cannot advertise recovery
        # it cannot perform, even when the native adapter retains that flag.
        assert not caught.value.recoverable
        assert caught.value.evidence.outcome_known and caught.value.evidence.state_mutated
        assert handle.state is LifecycleState.FAILED
    finally:
        handle.close()


class _VadBackend:
    def __init__(self) -> None:
        self.close_calls = 0

    def infer(self, _audio: np.ndarray) -> float:
        return 0.5

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1


def _handle() -> tuple[ModelRuntimeHandle, SpeechDirectionStreamingRuntime, _Pipeline, _Resource]:
    pipeline = _Pipeline()
    streaming = SpeechDirectionStreamingRuntime(pipeline, close_backends=False)
    session_resource = _Resource()
    contract = ExecutionContract(
        state_scope="stream",
        execution_structure="direct",
        cancellation_granularity="checkpoint",
        state_bank_mode="runtime_exclusive",
        max_open_streams=1,
    )
    role_context = object()
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=streaming,
            streaming_runtime=streaming,
            session=session_resource,
            owned_components=(
                OwnedComponent(session_resource, "session", load_context=role_context),
                OwnedComponent(streaming, "streaming-runtime"),
            ),
            stateful=True,
            resettable=True,
            state_scope="stream",
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
            execution_contract=contract,
        )
    )
    return handle, streaming, pipeline, session_resource


def test_speech_direction_stream_lifecycle_and_exclusive_admission() -> None:
    handle, _streaming, pipeline, session_resource = _handle()
    handle.load()
    assert session_resource.load_calls == 1
    session_entry = next(entry for entry in handle.owned_components if entry.name == "session")
    assert session_resource.load_contexts == [session_entry.load_context]

    first = handle.open_stream(ExecutionContext("open-1"))
    assert first.stream_id
    with pytest.raises(ExecutionFailure) as capacity:
        handle.open_stream(ExecutionContext("open-2"))
    assert capacity.value.code == "stream_capacity_exhausted"

    audio = np.arange(8, dtype=np.float32).reshape(4, 2)
    result = handle.step(
        first,
        ModelRequest({"audio": audio, "capture_start_sample": 128}),
        ExecutionContext("step-1"),
    )
    assert result.outputs["segment_angle"] == 90
    assert result.to_dict()["outputs"]["segment_seq"] == 1
    np.testing.assert_array_equal(pipeline.processed[0][0], audio)
    assert pipeline.processed[0][1] == 128

    handle.reset_stream(first, ExecutionContext("reset-1"))
    assert pipeline.reset_calls >= 3  # load, open, and stream reset
    handle.close_stream(first, ExecutionContext("close-1"))
    handle.close_stream(first, ExecutionContext("close-1-again"))
    assert first.closed
    with pytest.raises(ExecutionFailure) as closed:
        handle.step(first, ModelRequest({"audio": audio}), ExecutionContext("step-closed"))
    assert closed.value.code == "stream_closed"

    second = handle.open_stream(ExecutionContext("open-3"))
    assert second.stream_id != first.stream_id
    handle.close()
    assert pipeline.close_calls == [False]
    assert session_resource.close_calls == 1


def test_session_resources_and_role_runner_do_not_close_sessions_from_host_layer() -> None:
    session = _Resource()
    resources = SpeechDirectionSessionResources({"fullsubnet": (session, object())})
    resources.load()
    resources.close()
    resources.close()
    assert session.load_calls == 1
    assert session.close_calls == 1

    runner = SpeechDirectionRoleRunner(session, context=None)
    runner.close()
    assert session.close_calls == 1


@pytest.mark.parametrize("final_close", [False, True])
def test_stream_result_uses_public_identity_and_close_does_not_require_reset(monkeypatch, final_close):
    handle, streaming, pipeline, resource = _handle()
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))
    native_id = streaming._stream.stream_id
    result = handle.step(stream, ModelRequest({"audio": np.zeros((4, 2), dtype=np.float32)}), ExecutionContext("step"))
    assert result.metadata["stream_id"] == stream.stream_id == result.evidence.details["stream_id"]
    assert result.metadata["native_stream_id"] == native_id

    def reset_failed():
        raise RuntimeError("permanent device reset failure")

    monkeypatch.setattr(pipeline, "reset", reset_failed)
    if final_close:
        handle.close()
    else:
        handle.close_stream(stream)
        # The bank cannot be reused until open_stream successfully resets it.
        with pytest.raises(ExecutionFailure, match="permanent device reset failure"):
            handle.open_stream(ExecutionContext("reopen"))
    assert stream.closed and streaming._stream is None
    handle.close()
    assert resource.close_calls == 1 and pipeline.close_calls == [False]


def test_invalid_stream_input_is_rejected_before_pipeline_execution() -> None:
    handle, _streaming, pipeline, session_resource = _handle()
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))
    with pytest.raises(ExecutionFailure) as invalid:
        handle.step(stream, ModelRequest({"other": np.zeros((2, 2), dtype=np.float32)}), ExecutionContext("bad"))
    assert invalid.value.code == "invalid_stream_input"
    assert not pipeline.processed
    with pytest.raises(ExecutionFailure) as invalid_shape:
        handle.step(
            stream,
            ModelRequest({"audio": np.zeros((2, 3), dtype=np.float32)}),
            ExecutionContext("bad-shape"),
        )
    assert invalid_shape.value.code == "invalid_stream_input"
    assert not pipeline.processed
    handle.close()
    assert session_resource.close_calls == 1


def test_capture_runtime_routes_worker_blocks_through_handle() -> None:
    handle, _streaming, pipeline, session_resource = _handle()
    config = SimpleNamespace(
        audio=SimpleNamespace(sample_rate=16000, channels=2),
        speech_direction_max_age_ms=1000,
    )
    runtime = SpeechDirectionRuntime(
        config,
        pipeline,
        enable_capture=False,
        model_runtime_handle=handle,
    )
    runtime.start()
    runtime.feed_audio(np.ones((2, 2), dtype=np.float32))
    assert pipeline.processed_event.wait(1.0)
    assert runtime.stream_handle is not None
    runtime.stop()
    assert runtime.stream_handle is None
    assert session_resource.close_calls == 1
    assert pipeline.close_calls == [False]
    # Give the worker a scheduling opportunity before the test exits; a live
    # daemon worker would otherwise hide an ownership/race regression.
    time.sleep(0.01)


def test_node_shutdown_retains_worker_owner_until_thread_and_handle_close(monkeypatch):
    import rclpy
    from rclpy.node import Node

    from inference_service.unified_runtime import RuntimeProviders
    from voice_asr_service.speech_direction.node import SpeechDirectionNode

    handle, _, pipeline, resource = _handle()
    entered, release = threading.Event(), threading.Event()
    process = pipeline.process_block

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return process(*args, **kwargs)

    monkeypatch.setattr(pipeline, "process_block", blocked)
    runtime = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=2), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
        model_runtime_handle=handle,
    )
    providers = RuntimeProviders(object(), object())

    class ShutdownNode(SpeechDirectionNode):
        def __init__(self):
            Node.__init__(self, "speech_shutdown_test")
            self._runtime = runtime
            self._providers = providers
            self._diagnostics_recorder = None

    rclpy.init()
    node = ShutdownNode()
    try:
        runtime.start()
        runtime.feed_audio(np.ones((2, 2), np.float32))
        assert entered.wait(1)
        destroyed = node.destroy_node()
        assert not handle.closed and resource.close_calls == 0
        release.set()
        providers.close()
        assert handle.closed and resource.close_calls == 1 and not runtime._threads
        assert node.destroy_node() is destroyed
    finally:
        release.set()
        runtime.stop()
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_provider_cleanup_observes_budget_while_joining_speech_worker(stop):
    from inference_service.unified_runtime import Deadline, RuntimeProviders
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    handle, _, pipeline, resource = _handle()
    handle.load()
    runtime = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=2), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
        model_runtime_handle=handle,
    )
    entered, release = threading.Event(), threading.Event()
    worker = threading.Thread(target=lambda: (entered.set(), release.wait(2)))
    worker.start()
    runtime._threads = [worker]
    providers = RuntimeProviders(object(), object())
    providers.retain_cleanup(runtime)
    context = ExecutionContext("shutdown", Deadline.after(0.03))
    if stop == "cancel":
        context.token.cancel()
    try:
        assert entered.wait(1)
        started = time.monotonic()
        with pytest.raises(RuntimeError), lifecycle_scope(context):
            providers.close()
        assert time.monotonic() - started < 0.5
        assert worker in runtime._threads and resource.close_calls == 0
    finally:
        release.set()
        worker.join(1)
        providers.close()
    assert handle.closed and resource.close_calls == 1


@pytest.mark.parametrize("torch_fullnet", [False, True])
@pytest.mark.parametrize("construction_failure", [False, True])
def test_node_assembly_uses_the_injected_shared_resource_domain(monkeypatch, torch_fullnet, construction_failure):
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.backends import BackendAdmissionEvidence, BackendCapabilities
    from inference_service.backends.admission import ResourceDomainAdmissions
    from inference_service.unified_runtime import Deadline, RuntimeProviders
    from voice_asr_service.speech_direction import node as module
    from voice_asr_service.speech_direction.config import SpeechDirectionConfig

    providers = RuntimeProviders(object(), ResourceDomainAdmissions())
    cleanup_blocked = [construction_failure]

    def close_resource(resource):
        from inference_service.backends.errors import BackendLifecycleError

        if cleanup_blocked[0]:
            raise BackendLifecycleError("device cleanup pending", close_pending=True)
        resource.close_calls += 1

    monkeypatch.setattr(_Resource, "close", close_resource)
    if construction_failure:
        from voice_asr_service.speech_direction import runtime as runtime_module

        def fail_buffer(**_):
            raise MemoryError("ring allocation failed")

        monkeypatch.setattr(runtime_module, "MultiChannelRingBuffer", fail_buffer)
        monkeypatch.setattr(module.Node, "destroy_node", lambda _: True)
    # This test exercises production assembly and admission, not ROS transport.
    monkeypatch.setitem(sys.modules, "audio_common_msgs.msg", SimpleNamespace(AudioDataStamped=object))
    capabilities = BackendCapabilities(
        stateful=True,
        resettable=True,
        supports_multiple_instances=True,
        resource_domain="ascend:0",
        admission_evidence=BackendAdmissionEvidence(
            sdk_initialization=True,
            multi_instance_execution=True,
            failure_isolation=True,
            independent_close=True,
        ),
    )

    def session(context, **kwargs):
        from dataclasses import replace

        result = _Resource()
        result.name, result.capabilities = "speech-model", capabilities
        if context.backend == "torch":
            result.name = "fullsubnet-torch"
            result.capabilities = replace(capabilities, resource_domain=None)
        return result

    monkeypatch.setattr(
        module,
        "load_inference_manifest",
        lambda bundle, *_: SimpleNamespace(
            deployment=SimpleNamespace(backend="torch" if torch_fullnet and "fullsubnet" in str(bundle) else "ascend")
        ),
    )
    monkeypatch.setattr(module, "_validate_silero_deployment_contract", lambda *_: None)
    monkeypatch.setattr(module, "build_stateful_fullsubnet", lambda **_: _VadBackend())
    monkeypatch.setattr(module, "SileroVadEngine", lambda **_: _VadBackend())
    monkeypatch.setattr(module, "SpeechGate", lambda **kwargs: SimpleNamespace(silero=kwargs["silero_engine"]))
    monkeypatch.setattr(module, "StftSrpPhat", lambda **_: object())
    monkeypatch.setattr(module, "StreamingSpeechDirectionPipeline", lambda *args, **kwargs: _Pipeline())
    monkeypatch.setattr(module.SpeechDirectionRuntime, "start", lambda self: self._model_runtime_handle.load())
    nodes = []
    for _ in range(2):
        node = module.SpeechDirectionNode.__new__(module.SpeechDirectionNode)
        node._config = SpeechDirectionConfig()
        if torch_fullnet:
            node._config.fullnet.backend = "stateful_torch_cuda"
        node._providers = providers
        node._registry_set = SimpleNamespace(
            session_builder_registry=SimpleNamespace(create=session), backend_registry=object()
        )
        node._session_resources = None
        node._runtime = None
        node._diagnostics_recorder = None
        node.get_logger = lambda: SimpleNamespace(error=lambda *_: None)
        node._enhanced_audio_topic, node._enhanced_audio_pub = "", None
        node.create_subscription = lambda *_: None
        if construction_failure:
            with pytest.raises(MemoryError, match="ring allocation"):
                node._build_and_start()
            owner = node._runtime_handle
            assert node._runtime is None and node._session_resources is None
            assert node.destroy_node()
            assert not owner.closed
            cleanup_blocked[0] = False
            providers.close()
            assert owner.closed and not providers.resource_admission_provider._domains
            return
        node._build_and_start()
        nodes.append(node)
    first, second = (node._runtime_handle for node in nodes)
    a, b = (owner.open_stream(ExecutionContext("open")) for owner in (first, second))
    entered, release = threading.Event(), threading.Event()
    original = nodes[0]._runtime.pipeline.process_block

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(nodes[0]._runtime.pipeline, "process_block", blocked)
    request = ModelRequest({"audio": np.ones((6, 2), np.float32)})
    try:
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(first.step, a, request, ExecutionContext("first"))
            try:
                assert entered.wait(1)
                with pytest.raises(ExecutionFailure, match="deadline"):
                    second.step(b, request, ExecutionContext("contender", Deadline.after(0.02)))
                assert not nodes[1]._runtime.pipeline.processed
            finally:
                release.set()
            running.result(1)
        assert second.step(b, request, ExecutionContext("next")).evidence.outcome_known
    finally:
        release.set()
        first.close()
        second.close()
        providers.close()


def test_stop_keeps_pending_handle_after_worker_wait_error(monkeypatch):
    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.unified_runtime import RuntimeProviders
    from inference_service.unified_runtime.errors import DeadlineExceeded
    from voice_asr_service.speech_direction import runtime as module

    handle, _, pipeline, resource = _handle()
    handle.load()
    runtime = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=2), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
        model_runtime_handle=handle,
    )
    alive = iter((True, False))
    runtime._threads = [SimpleNamespace(is_alive=lambda: next(alive))]
    blocked = [True]

    def close():
        if blocked[0]:
            raise BackendLifecycleError("device still active", close_pending=True)
        resource.close_calls += 1

    def expire():
        raise DeadlineExceeded(phase="control_drain")

    monkeypatch.setattr(resource, "close", close)
    monkeypatch.setattr(module, "check_lifecycle", expire)
    with pytest.raises(BackendLifecycleError) as caught:
        runtime.stop()
    assert caught.value.close_pending and isinstance(caught.value.__cause__, DeadlineExceeded)
    assert not handle.closed and not runtime._threads
    providers = RuntimeProviders(object(), object())
    providers.retain_cleanup(runtime)
    blocked[0] = False
    providers.close()
    assert handle.closed and runtime._handle_closed and resource.close_calls == 1


@pytest.mark.parametrize("cancel_between_roles", [False, True])
def test_real_pipeline_passes_outer_cancellation_between_fb_sb_and_vad(tmp_path, cancel_between_roles):
    from inference_service.backends import BackendCapabilities, RuntimeContext
    from inference_service.model_sessions import ModelSession
    from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful import StatefulFullSubNetEnhancer
    from voice_asr_service.speech_direction.pipeline import DoaState, VadState
    from voice_asr_service.speech_direction.pipeline_streaming import (
        StreamingPipelineParams,
        StreamingSpeechDirectionPipeline,
    )

    contexts = []
    calls = []
    execution = ExecutionContext("outer-step")

    class Session(ModelSession):
        def __init__(self):
            super().__init__("speech-checkpoint", BackendCapabilities(stateful=True, resettable=True))

        def _load(self, context, rollback):
            pass

        def _execute(self, request, context):
            return {}

        def _execute_role(self, role, inputs, request, context):
            calls.append(role)
            contexts.append(context)
            if not cancel_between_roles:
                execution.token.cancel("stop after FB")
            return {"host.fullsubnet.fb_features": np.ones((4, 2, 257), np.float32)}

        def _check_completed_context(self, context, phase):
            super()._check_completed_context(context, phase)
            if cancel_between_roles and phase == "model.fullsubnet_fb":
                execution.token.cancel("cancel after FB result check, before SB")

        def _validate_role_values(self, *args, **kwargs):
            pass

        def _reset(self):
            pass

        def _close(self):
            pass

    session = Session()
    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    session.load(context)
    fullnet = StatefulFullSubNetEnhancer(SpeechDirectionRoleRunner(session, context))
    path = tmp_path / "vad.om"
    path.write_bytes(b"test")
    vad = SileroVadEngine(str(path), inference_runner=SpeechDirectionRoleRunner(session, context))
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        vad,
        SimpleNamespace(angles=np.array([90.0])),
        StreamingPipelineParams(raw_activity_gate_enabled=False),
        VadState(),
        DoaState(),
    )
    native = SpeechDirectionStreamingRuntime(pipeline)
    owner = ModelRuntimeHandle(
        native,
        streaming_runtime=native,
        state_scope="stream",
        state_bank_mode="runtime_exclusive",
        stateful=True,
        resettable=True,
    )
    capture = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=6), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
        model_runtime_handle=owner,
    )
    capture._ensure_stream_runtime()
    stream = capture.stream_handle
    request = ModelRequest({"audio": np.ones((6, 256), np.float32)})
    try:
        owner.step(stream, request, ExecutionContext("warmup"))
        with pytest.raises(ExecutionFailure) as caught:
            owner.step(stream, request, execution)
        assert calls == ["fullsubnet_fb"] and contexts == [execution]
        assert contexts[0].token is execution.token
        assert caught.value.code == "request_canceled" and caught.value.evidence.outcome_known
        assert caught.value.evidence.state.value != "not_started"
        assert caught.value.evidence.state_mutated
        assert owner.state is LifecycleState.RESET_REQUIRED
        with pytest.raises(ExecutionFailure, match="recovery_required"):
            owner.step(stream, request, ExecutionContext("blocked"))
        assert calls == ["fullsubnet_fb"]
        capture.reset()
        assert owner.ready and capture.stream_handle is stream
        assert owner.step(stream, request, ExecutionContext("after-reset")).evidence.outcome_known
    finally:
        capture.stop()
        session.close()


def test_vad_role_runner_inherits_outer_deadline_and_cancellation():
    from inference_service.backends import BackendCapabilities, RuntimeContext
    from inference_service.model_sessions import ModelSession
    from inference_service.unified_runtime import Deadline
    from inference_service.unified_runtime.lifecycle import lifecycle_scope

    calls = []

    class Session(ModelSession):
        def __init__(self):
            super().__init__("vad-checkpoint", BackendCapabilities())

        def _load(self, context, rollback):
            pass

        def _execute(self, request, context):
            return {}

        def _execute_role(self, role, inputs, request, context):
            calls.append((role, context))
            return {"host.silero.prob": np.asarray([0.75], dtype=np.float32)}

        def _validate_role_values(self, *args, **kwargs):
            pass

        def _close(self):
            pass

    session = Session()
    deployment = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    session.load(deployment)
    runner = SpeechDirectionRoleRunner(session, deployment)
    outer = ExecutionContext("outer-vad", Deadline.after(10))
    try:
        with lifecycle_scope(outer):
            assert runner.infer(np.zeros(512, np.float32)) == 0.75
            outer.token.cancel("stop VAD")
            with pytest.raises(ExecutionFailure, match="request_canceled"):
                runner.infer(np.zeros(512, np.float32))
        assert calls == [("silero_vad", outer)]
        assert calls[0][1].deadline is outer.deadline
    finally:
        session.close()


@pytest.mark.parametrize("compatibility", [False, True])
def test_capture_stop_retries_pending_native_owner(monkeypatch, compatibility):
    from inference_service.backends.errors import BackendLifecycleError

    handle, _, pipeline, resource = _handle()
    calls = []

    def close(*_args, **_kwargs):
        calls.append("close")
        if len(calls) == 1:
            raise BackendLifecycleError("release pending", close_pending=True)

    monkeypatch.setattr(pipeline if compatibility else resource, "close", close)
    runtime = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=2), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
        model_runtime_handle=None if compatibility else handle,
    )
    runtime.start()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            runtime.stop()
        assert failure.value.close_pending
        assert not runtime._handle_closed and not runtime._pipeline_closed
        runtime.stop()
        runtime.stop()
        assert runtime._model_runtime_handle.closed and calls == ["close", "close"]
    finally:
        runtime.stop()


def test_silero_host_close_leaves_backend_close_for_the_owner(tmp_path) -> None:
    model_path = tmp_path / "silero.om"
    model_path.write_bytes(b"mock")
    backend = _VadBackend()
    engine = SileroVadEngine(str(model_path), inference_runner=backend)
    engine.close(close_runner=False)
    assert backend.close_calls == 0
    engine.close()
    assert backend.close_calls == 1


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_open_stream_reset_lock_inherits_request_controls(stop):
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.unified_runtime import Deadline
    from inference_service.unified_runtime.lifecycle import lifecycle_lock

    lock = threading.Lock()
    entered = threading.Event()
    pipeline = _Pipeline()
    runtime = SpeechDirectionStreamingRuntime(pipeline)
    runtime.load()
    resets = []

    def reset():
        entered.set()
        with lifecycle_lock(lock):
            resets.append("reset")

    pipeline.reset = reset
    owner = ModelRuntimeHandle(
        runtime,
        streaming_runtime=runtime,
        state_scope="stream",
        state_bank_mode="runtime_exclusive",
        stateful=True,
        resettable=True,
    )
    owner.load()
    resets.clear()
    entered.clear()
    context = ExecutionContext("open", Deadline.after(0.1) if stop == "deadline" else None)
    lock.acquire()
    try:
        with ThreadPoolExecutor(1) as pool:
            opening = pool.submit(owner.open_stream, context)
            try:
                assert entered.wait(1)
                if stop == "cancel":
                    context.token.cancel("cancel while reset waits for lock")
                with pytest.raises(ExecutionFailure) as caught:
                    opening.result(1)
                assert caught.value.code == ("request_canceled" if stop == "cancel" else "deadline_exceeded")
                assert resets == [] and runtime.active_stream is None
            finally:
                lock.release()
    finally:
        owner.close()


@pytest.mark.parametrize("compatibility", [False, True])
def test_runtime_reset_resets_each_native_session_once_and_clears_host_state(tmp_path, compatibility):
    from inference_service.backends import BackendCapabilities, RuntimeContext
    from inference_service.model_sessions import ModelSession
    from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful import StatefulFullSubNetEnhancer
    from voice_asr_service.speech_direction.pipeline import DoaState, VadState
    from voice_asr_service.speech_direction.pipeline_streaming import (
        StreamingPipelineParams,
        StreamingSpeechDirectionPipeline,
    )

    counts = {"fullsubnet": 0, "vad": 0}

    class Session(ModelSession):
        def __init__(self, name):
            super().__init__(name, BackendCapabilities(stateful=True, resettable=True))

        def _load(self, context, rollback):
            pass

        def _execute(self, request, context):
            return {}

        def _reset(self):
            counts[self.name] += 1

        def _close(self):
            pass

    sessions = {name: Session(name) for name in counts}
    runtime_context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace(backend="ascend")))
    # The compatibility constructor owns the pipeline rather than Sessions.
    if compatibility:
        for session in sessions.values():
            session.load(runtime_context)
    fullnet = StatefulFullSubNetEnhancer(
        SpeechDirectionRoleRunner(sessions["fullsubnet"], runtime_context), initialize_backend=False
    )
    path = tmp_path / "vad.om"
    path.write_bytes(b"test")
    vad = SileroVadEngine(str(path), inference_runner=SpeechDirectionRoleRunner(sessions["vad"], runtime_context))
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        vad,
        SimpleNamespace(angles=np.array([90.0])),
        StreamingPipelineParams(),
        VadState(),
        DoaState(),
        initialize_backend=False,
    )
    owner = None
    if not compatibility:
        streaming = SpeechDirectionStreamingRuntime(pipeline, close_backends=False)
        owner = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=streaming,
                streaming_runtime=streaming,
                session=sessions["fullsubnet"],
                role_assemblies=sessions,
                owned_components=tuple(
                    OwnedComponent(session, name, load_context=runtime_context) for name, session in sessions.items()
                )
                + (OwnedComponent(streaming),),
                stateful=True,
                resettable=True,
                state_scope="stream",
                state_bank_mode="runtime_exclusive",
                max_open_streams=1,
            )
        )
    capture = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=6), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
        model_runtime_handle=owner,
    )
    try:
        capture._ensure_stream_runtime()
        stream = capture.stream_handle
        counts.update({name: 0 for name in counts})
        fullnet._fb_sum.fill(1)
        vad._context.fill(1)
        pipeline.gate._state = "ACTIVE"
        pipeline._samples_processed = 1024
        capture.reset()
        assert counts == {"fullsubnet": 1, "vad": 1}
        assert capture.stream_handle is stream and capture.runtime_handle.ready
        assert not np.any(fullnet._fb_sum) and not np.any(vad._context)
        assert pipeline.gate._state == "IDLE" and pipeline._samples_processed == 0
        # A public stream-only reset still has to reset its native state.
        counts.update({name: 0 for name in counts})
        capture.runtime_handle.reset_stream(stream, ExecutionContext("stream-only"))
        assert counts == {"fullsubnet": 1, "vad": 1}
    finally:
        capture.stop()
        for session in sessions.values():
            session.close()


@pytest.mark.parametrize("invalid", ["hop", "channels", "closed"])
def test_invalid_block_is_rejected_before_host_state_advances(invalid):
    from inference_service.unified_runtime import OutcomeState

    pipeline = _Pipeline()
    pipeline.input_channels = (0, 1, 2, 3)
    adapter = SpeechDirectionStreamingRuntime(pipeline)
    adapter.load()
    context = ExecutionContext("invalid")
    stream = adapter.open_stream(context)
    audio = np.ones((4, 2), np.float32)
    if invalid == "hop":
        audio = audio[:, :1]
    elif invalid == "channels":
        audio = audio[:2]
    else:
        pipeline._closed = True
    try:
        with pytest.raises(ExecutionFailure) as caught:
            adapter.step(stream, ModelRequest({"audio": audio}), context)
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert not caught.value.evidence.state_mutated
        assert not pipeline.processed
    finally:
        adapter.close()
