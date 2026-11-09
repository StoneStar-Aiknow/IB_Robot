"""Voice owners keep ACL models and their leases until native release succeeds."""

from types import SimpleNamespace

import pytest

from inference_service.backends import BackendCapabilities, RuntimeContext
from inference_service.backends.ascend.model import AclDeviceBuffer, AclModel
from inference_service.backends.errors import BackendLifecycleError
from inference_service.model_sessions import ModelSession
from inference_service.unified_runtime import ExecutionFailure, ModelRuntimeHandle, RuntimeAssembly
from voice_asr_service.speech_direction.model_sessions import SpeechDirectionSessionResources


def test_composite_load_failure_does_not_inherit_a_childs_clean_rollback():
    calls = []
    blocked = [True]

    class Session(ModelSession):
        def __init__(self, name):
            super().__init__(name, BackendCapabilities())

        def _load(self, context, rollback):
            if self.name == "B":
                raise ValueError("B load failed")

        def _execute(self, request, context):
            return {}

        def _close(self):
            calls.append(self.name)
            if self.name == "A" and blocked[0]:
                raise BackendLifecycleError("A retained", close_pending=True)

    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace()))
    resources = SpeechDirectionSessionResources({name: (Session(name), context) for name in ("A", "B")})
    owner = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=SimpleNamespace(), owned_components=(resources,)))
    with pytest.raises(ExecutionFailure):
        owner.load()
    assert id(resources) not in owner._released_ids
    assert not resources._closed
    blocked[0] = False
    owner.close()
    assert owner.closed and resources._closed and calls[-1] == "A"


@pytest.mark.parametrize("kind", ["silero", "fullsubnet", "zipformer"])
def test_voice_runner_retries_host_release_before_closing_lease(tmp_path, monkeypatch, kind):
    from voice_asr_service.speech_direction import silero_acl
    from voice_asr_service.speech_direction.enhancement import fullsubnet_stateful_acl
    from voice_asr_service.transducer_ascend import ZipformerEncoderAclRunner

    events = []
    failures = [True]

    def free_host(pointer):
        events.append(pointer)
        if failures[0]:
            failures[0] = False
            return 1
        return 0

    lease = SimpleNamespace(
        acl=SimpleNamespace(rt=SimpleNamespace(free_host=free_host)),
        bind_current_thread=lambda: None,
        close=lambda: events.append("lease"),
    )
    provider = SimpleNamespace(acquire=lambda _: lease)

    class Model(AclModel):
        def __init__(self, lease, role, path, bindings):
            super().__init__(lease, role, path, bindings)
            self.output_host_buffers = [role]

        def load_descriptor(self):
            self.input_descriptors = tuple(SimpleNamespace(size=4, shape=b.shape) for b in self.bindings.inputs)
            self.output_descriptors = tuple(SimpleNamespace(size=4, shape=b.shape) for b in self.bindings.outputs)

        def allocate_device_buffer(self, size):
            return AclDeviceBuffer(object(), size)

        def prepare_dataset_banks(self, *args, **kwargs):
            pass

        def zero_device_buffer(self, _):
            pass

    path = tmp_path / "model.om"
    path.write_bytes(b"fake OM")
    if kind == "silero":
        monkeypatch.setattr(silero_acl, "AclModel", Model)
        runner = silero_acl.SileroVadAclRunner(str(path), runtime_manager=provider)
        order = ["silero_vad"]
    elif kind == "fullsubnet":
        monkeypatch.setattr(fullsubnet_stateful_acl, "AclModel", Model)
        runner = fullsubnet_stateful_acl.StatefulAclFullSubNetRunner(str(path), str(path), runtime_manager=provider)
        order = ["fullsubnet_sb", "fullsubnet_fb"]
    else:
        from inference_manifest import ArtifactBindings, TensorBinding
        from inference_service.backends.ascend import model
        from voice_asr_service.transducer_ascend import ENCODER_FEATURES_SEMANTIC

        monkeypatch.setattr(model, "AclModel", Model)
        bindings = ArtifactBindings(
            inputs=(TensorBinding(semantic=ENCODER_FEATURES_SEMANTIC, index=0, dtype="float32", shape=(1, 80)),),
            outputs=(TensorBinding(semantic="output", index=0, dtype="float32", shape=(1, 80)),),
        )
        runner = ZipformerEncoderAclRunner(path, bindings, device_id=0, runtime_manager=provider)
        order = ["encoder"]
    with pytest.raises(BackendLifecycleError) as error:
        runner.close()
    assert error.value.close_pending and events == [order[0]]
    runner.close()
    runner.close()
    assert events == [order[0], *order, "lease"]


def test_speech_session_close_pending_survives_composite_and_handle_boundary():
    calls = []

    class Session(ModelSession):
        def __init__(self):
            super().__init__("speech-test", BackendCapabilities())

        def _load(self, context, rollback):
            pass

        def _execute(self, request, context):
            return {}

        def _close(self):
            calls.append("close")
            if len(calls) == 1:
                raise BackendLifecycleError("native cleanup pending", close_pending=True)

    context = RuntimeContext(SimpleNamespace(deployment=SimpleNamespace()))
    resources = SpeechDirectionSessionResources({"vad": (Session(), context)})
    owner = ModelRuntimeHandle(RuntimeAssembly(runtime_executor=SimpleNamespace(), owned_components=(resources,)))
    owner.load()
    with pytest.raises(ExecutionFailure) as error:
        owner.close()
    assert error.value.close_pending and not owner.closed
    owner.close()
    owner.close()
    assert owner.closed and calls == ["close", "close"]


@pytest.mark.parametrize("kind", ["silero", "fullsubnet", "zipformer"])
@pytest.mark.parametrize("external_rollback", [False, True])
def test_failed_runner_constructor_transfers_owner_and_preserves_load_error(
    tmp_path, monkeypatch, kind, external_rollback
):
    from inference_service.backends.ascend import model
    from inference_service.unified_runtime import LoadRollback
    from voice_asr_service.speech_direction import silero_acl
    from voice_asr_service.speech_direction.enhancement import fullsubnet_stateful_acl
    from voice_asr_service.transducer_ascend import ZipformerEncoderAclRunner

    events, blocked = [], [True]
    original = ValueError("descriptor failed")

    class Model:
        def __init__(self, *_):
            pass

        def load_descriptor(self):
            raise original

        def close(self):
            events.append("model")
            if blocked[0]:
                raise RuntimeError("release failed")

    lease = SimpleNamespace(close=lambda: events.append("lease"))
    provider = SimpleNamespace(acquire=lambda _: lease)
    path = tmp_path / "broken.om"
    path.write_bytes(b"test")
    rollback = LoadRollback() if external_rollback else None
    monkeypatch.setattr(model, "AclModel", Model)
    monkeypatch.setattr(silero_acl, "AclModel", Model)
    monkeypatch.setattr(fullsubnet_stateful_acl, "AclModel", Model)
    with pytest.raises(ValueError) as failure:
        if kind == "zipformer":
            ZipformerEncoderAclRunner(path, None, device_id=0, runtime_manager=provider, cleanup=rollback)
        elif kind == "silero":
            silero_acl.SileroVadAclRunner(str(path), runtime_manager=provider, cleanup=rollback)
        else:
            fullsubnet_stateful_acl.StatefulAclFullSubNetRunner(
                str(path), str(path), runtime_manager=provider, cleanup=rollback
            )
    assert failure.value is original and failure.value.close_pending
    owner = failure.value.cleanup_owner
    if external_rollback:
        assert owner is rollback and rollback.rollback()
    assert events == ["model"]
    blocked[0] = False
    owner.close()
    owner.close()
    assert events == ["model", "model", "lease"]


def test_asr_constructor_callsite_keeps_failed_encoder_until_cleanup(tmp_path, monkeypatch):
    from inference_service.backends.ascend import acl_runtime, model
    from voice_asr_service.asr_inference_module import ASRInferenceModule

    events, blocked = [], [True]
    original = ValueError("encoder descriptor failed")

    class Model:
        def __init__(self, *_):
            pass

        def load_descriptor(self):
            raise original

        def close(self):
            events.append("model")
            if blocked[0]:
                raise RuntimeError("release pending")

    provider = SimpleNamespace(
        acquire=lambda _: SimpleNamespace(close=lambda: events.append("lease")),
        close=lambda: events.append("provider"),
    )
    monkeypatch.setattr(acl_runtime, "AclRuntimeManager", lambda: provider)
    monkeypatch.setattr(model, "AclModel", Model)
    path = tmp_path / "encoder.om"
    path.write_bytes(b"test")
    module = ASRInferenceModule()
    with pytest.raises(ValueError) as failure:
        module._initialize_ascend_transducer(
            SimpleNamespace(deployment=SimpleNamespace(bindings={"encoder": None})),
            {"encoder": str(path)},
            SimpleNamespace(profile=SimpleNamespace(device_id=0)),
            0.0,
        )
    assert failure.value is original and failure.value.cleanup_owner is module
    assert events == ["model"]
    blocked[0] = False
    module.cleanup()
    module.cleanup()
    assert events == ["model", "model", "lease", "provider"]


def test_asr_initialize_preserves_onnx_import_failure_and_cleanup_owner(tmp_path, monkeypatch):
    import builtins
    import importlib.util

    from inference_service.backends.ascend import acl_runtime
    from voice_asr_service import asr_inference_module, transducer_ascend

    events, blocked = [], [True]
    provider = SimpleNamespace(close=lambda: events.append("provider"))
    monkeypatch.setattr(acl_runtime, "AclRuntimeManager", lambda: provider)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

    class Encoder:
        def __init__(self, *args, cleanup, **kwargs):
            cleanup.defer(self.close)

        def close(self):
            events.append("encoder")
            if blocked[0]:
                raise RuntimeError("encoder release pending")

    monkeypatch.setattr(transducer_ascend, "ZipformerEncoderAclRunner", Encoder)
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    validated = SimpleNamespace(
        identity=SimpleNamespace(interface="tensor_model", model_type="sherpa_onnx", operation="recognize"),
        bundle_root=tmp_path,
        resolved_artifacts={role: tmp_path / role for role in ("encoder", "decoder", "joiner", "tokens")},
        deployment=SimpleNamespace(
            device="cpu",
            bindings={"encoder": None},
            metadata={"decode_chunk_len": 32},
            role_runtime_profiles={"encoder": SimpleNamespace(backend="ascend", profile=SimpleNamespace(device_id=0))},
        ),
    )
    monkeypatch.setattr(asr_inference_module, "load_inference_manifest", lambda *_: validated)
    module = asr_inference_module.ASRInferenceModule()
    with pytest.raises(RuntimeError, match="onnxruntime unavailable") as caught:
        module.initialize(str(tmp_path), "ascend")
    assert caught.value.cleanup_owner is module and caught.value.close_pending
    assert isinstance(caught.value.__cause__, ImportError)
    assert events == ["encoder"]
    blocked[0] = False
    caught.value.cleanup_owner.close()
    assert events == ["encoder", "encoder", "provider"]


@pytest.mark.parametrize("failing", ["silero", "fullsubnet"])
def test_compatible_speech_chain_retries_pending_backend_without_reclosing_peer(tmp_path, failing):
    import numpy as np

    from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful import StatefulFullSubNetEnhancer
    from voice_asr_service.speech_direction.pipeline import DoaState, VadState
    from voice_asr_service.speech_direction.pipeline_streaming import (
        StreamingPipelineParams,
        StreamingSpeechDirectionPipeline,
    )
    from voice_asr_service.speech_direction.runtime import SpeechDirectionRuntime
    from voice_asr_service.speech_direction.speech_gate import SileroVadEngine

    events = []

    def backend(name):
        def close():
            events.append(name)
            if name == failing and events.count(name) == 1:
                raise BackendLifecycleError("ACL release pending", close_pending=True)

        return SimpleNamespace(backend="ascend", reset=lambda: None, close=close)

    path = tmp_path / "silero.om"
    path.write_bytes(b"test")
    silero = SileroVadEngine(str(path), inference_runner=backend("silero"))
    fullnet = StatefulFullSubNetEnhancer(backend("fullsubnet"))
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        silero,
        SimpleNamespace(angles=np.array([90.0])),
        StreamingPipelineParams(),
        VadState(),
        DoaState(),
    )
    runtime = SpeechDirectionRuntime(
        SimpleNamespace(audio=SimpleNamespace(sample_rate=16000, channels=6), speech_direction_max_age_ms=1000),
        pipeline,
        enable_capture=False,
    )
    runtime.start()
    try:
        with pytest.raises(ExecutionFailure) as failure:
            runtime.stop()
        assert failure.value.close_pending and not runtime._model_runtime_handle.closed
        runtime.stop()
        runtime.stop()
        assert runtime._model_runtime_handle.closed
        assert events.count(failing) == 2
        assert events.count("silero" if failing == "fullsubnet" else "fullsubnet") == 1
    finally:
        runtime.stop()


def test_asr_handle_load_failure_preserves_owner_after_constructor_handoff(tmp_path, monkeypatch):
    from voice_asr_service import asr_inference_module, transducer_ascend

    blocked, calls = [True], []

    class Encoder:
        def __init__(self, *args, cleanup, **kwargs):
            cleanup.defer(self.close)

        def close(self):
            calls.append("encoder_close")
            if blocked[0]:
                raise BackendLifecycleError("encoder cleanup pending", close_pending=True)

    class Runtime:
        def __init__(self, encoder, *args, **kwargs):
            self.encoder = encoder

        def load(self):
            raise RuntimeError("runtime installation failed")

        def step(self, *args):
            return {}

        def open_stream(self, *_):
            return object()

        def reset_stream(self, *_):
            return None

        def close_stream(self, *_):
            return None

        def close(self):
            self.encoder.close()

    validated = SimpleNamespace(
        identity=SimpleNamespace(interface="tensor_model", model_type="sherpa_onnx", operation="recognize"),
        bundle_root=tmp_path,
        resolved_artifacts={role: tmp_path / role for role in ("encoder", "decoder", "joiner", "tokens")},
        deployment=SimpleNamespace(
            device="cpu",
            bindings={"encoder": None},
            metadata={"decode_chunk_len": 32},
            role_runtime_profiles={"encoder": SimpleNamespace(backend="ascend", profile=SimpleNamespace(device_id=0))},
        ),
    )
    monkeypatch.setattr(asr_inference_module, "load_inference_manifest", lambda *_: validated)
    monkeypatch.setattr(transducer_ascend, "ZipformerEncoderAclRunner", Encoder)
    monkeypatch.setattr(transducer_ascend, "AscendTransducerRuntime", Runtime)
    monkeypatch.setattr(
        "inference_service.backends.ascend.acl_runtime.AclRuntimeManager", lambda: SimpleNamespace(close=lambda: None)
    )
    module = asr_inference_module.ASRInferenceModule()
    with pytest.raises(RuntimeError, match="runtime installation failed") as caught:
        module.initialize(str(tmp_path), "ascend")
    assert caught.value.cleanup_owner is module and caught.value.close_pending
    assert module._pending_cleanup is None  # Ownership has already moved to Handle.
    attempts = len(calls)
    blocked[0] = False
    caught.value.cleanup_owner.close()
    assert len(calls) == attempts + 1 and module._runtime_handle is None


def test_manifest_vad_keeps_handle_and_provider_when_close_needs_retry():
    from voice_asr_service.vad_runtime import ManifestVadRuntime

    events, blocked = [], [True]

    def close_handle():
        events.append("handle")
        if blocked[0]:
            raise BackendLifecycleError("VAD lease pending", close_pending=True)

    runtime = ManifestVadRuntime("bundle", "onnx")
    runtime._handle = SimpleNamespace(close=close_handle)
    runtime._providers = SimpleNamespace(close=lambda: events.append("provider"))
    with pytest.raises(BackendLifecycleError):
        runtime.close()
    assert events == ["handle"]
    blocked[0] = False
    runtime.close()
    runtime.close()
    assert events == ["handle", "handle", "provider"]


def test_manifest_vad_load_failure_keeps_original_failure_and_cleanup_owner(monkeypatch):
    from inference_service.unified_runtime import RuntimeProviders
    from voice_asr_service import vad_runtime

    blocked = [True]

    class Session(ModelSession):
        def __init__(self):
            super().__init__("vad", BackendCapabilities(stateful=True, resettable=True))

        def _load(self, context, rollback):
            rollback.defer(self._close)
            raise ValueError("original VAD load failure")

        def _execute(self, request, context):
            return {}

        def _close(self):
            if blocked[0]:
                raise BackendLifecycleError("VAD resource still busy", close_pending=True)

    identity = SimpleNamespace(interface="tensor_model", model_type="silero_vad", operation="vad")
    validated = SimpleNamespace(
        top_level_identity=identity,
        deployment=SimpleNamespace(
            backend="onnx",
            execution_contract=None,
            audio_contract=SimpleNamespace(
                sample_rate_hz=16000,
                frame_size=512,
                chunk_size=576,
                channels=1,
                sample_dtype="float32",
                execution_mode="streaming",
            ),
        ),
        deployment_fingerprint="deployment",
        runtime_profile_fingerprint="profile",
        integrity_status="verified",
    )
    providers = RuntimeProviders(object(), object())
    session = Session()
    dependencies = SimpleNamespace(
        providers=providers,
        registry_set=SimpleNamespace(
            session_builder_registry=SimpleNamespace(create=lambda *_args, **_kwargs: session),
            backend_registry=object(),
        ),
    )
    monkeypatch.setattr(vad_runtime, "load_inference_manifest", lambda *_: validated)
    monkeypatch.setattr(vad_runtime, "build_runtime_dependencies", lambda *_: dependencies)
    runtime = vad_runtime.ManifestVadRuntime("bundle", "onnx")
    with pytest.raises(ExecutionFailure) as caught:
        runtime.initialize()
    failure = caught.value
    assert isinstance(failure.cause, ExecutionFailure)
    assert "original VAD load failure" in failure.message
    assert failure.code == failure.cause.code and failure.evidence == failure.cause.evidence
    assert failure.close_pending and failure.cleanup_owner is runtime
    assert failure.details["rollback_errors"]
    assert runtime._handle is not None and not providers.closed
    blocked[0] = False
    failure.cleanup_owner.close()
    assert runtime._handle is None and providers.closed


@pytest.mark.parametrize("failure_at", ["node", "provider", "both"])
def test_speech_direction_main_always_shuts_down_ros_after_cleanup_errors(monkeypatch, failure_at):
    from voice_asr_service.speech_direction import node

    events = []

    def destroy():
        events.append("node")
        if failure_at in {"node", "both"}:
            raise RuntimeError("node cleanup pending")

    def close():
        events.append("provider")
        if failure_at in {"provider", "both"}:
            raise RuntimeError("provider cleanup pending")

    monkeypatch.setattr(node.rclpy, "init", lambda **_: None)
    monkeypatch.setattr(node.rclpy, "spin", lambda _: None)
    monkeypatch.setattr(node.rclpy, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(node, "SpeechDirectionNode", lambda **_: SimpleNamespace(destroy_node=destroy))
    monkeypatch.setattr(
        node,
        "build_model_service_runtime_dependencies",
        lambda: SimpleNamespace(registry_set=object(), providers=SimpleNamespace(close=close)),
    )
    with pytest.raises(RuntimeError, match="cleanup pending"):
        node.main()
    assert events == ["node", "provider", "shutdown"]


@pytest.mark.parametrize("timer_index", [1, 2])
@pytest.mark.parametrize("cleanup_pending", [False, True])
def test_speech_direction_timer_failure_releases_started_runtime(monkeypatch, timer_index, cleanup_pending):
    from inference_service.unified_runtime import RuntimeProviders
    from voice_asr_service.speech_direction import model_downloader, node

    events, timers = [], []
    logger = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None)
    config = SimpleNamespace(
        mount_yaw_deg=0,
        audio_topic="/audio",
        vad=SimpleNamespace(model_path="vad", backend="onnx"),
        fullnet=SimpleNamespace(
            ckpt="fullnet",
            backend="torch",
            stateful_fb_om_path="fb",
            stateful_sb_om_path="sb",
            stateful_manifest_path="manifest",
        ),
    )
    providers = RuntimeProviders(SimpleNamespace(close=lambda: events.append("provider")), object())

    def stop():
        events.append("stop")
        if cleanup_pending and events.count("stop") == 1:
            raise BackendLifecycleError("worker still exiting", close_pending=True)

    def build(self):
        events.append("start")
        self._runtime = SimpleNamespace(stop=stop, close=stop)

    def timer(*_):
        timers.append(True)
        if len(timers) == timer_index:
            raise ValueError("timer creation failed")
        return object()

    monkeypatch.setattr(node.Node, "__init__", lambda *_: None)
    monkeypatch.setattr(node.Node, "create_publisher", lambda *_, **__: object())
    monkeypatch.setattr(node.Node, "declare_parameter", lambda *_, **__: None)
    monkeypatch.setattr(node.Node, "get_parameter", lambda *_, **__: SimpleNamespace(value=""))
    monkeypatch.setattr(node.Node, "create_timer", timer)
    monkeypatch.setattr(node.Node, "get_logger", lambda _: logger)
    monkeypatch.setattr(node.Node, "destroy_node", lambda _: events.append("node") or True)
    monkeypatch.setattr(node.SpeechDirectionNode, "_declare_parameters", lambda _: None)
    monkeypatch.setattr(node.SpeechDirectionNode, "_load_config_from_parameters", lambda _: config)
    monkeypatch.setattr(node.SpeechDirectionNode, "_apply_bundle_artifacts", lambda *_: None)
    monkeypatch.setattr(node.SpeechDirectionNode, "_build_and_start", build)
    monkeypatch.setattr(model_downloader, "require_configured_models", lambda *_, **__: None)
    monkeypatch.setattr(node, "require_runtime_dependencies", lambda registry, providers, **_: (registry, providers))
    monkeypatch.setattr(
        node,
        "build_model_service_runtime_dependencies",
        lambda: SimpleNamespace(registry_set=object(), providers=providers),
    )
    monkeypatch.setattr(node.rclpy, "init", lambda **_: None)
    monkeypatch.setattr(node.rclpy, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(node.rclpy, "spin", lambda _: pytest.fail("constructor failed before spin"))
    with pytest.raises(ValueError, match="timer creation failed"):
        node.main()
    assert events == ["start", "stop", "node"] + (["stop"] if cleanup_pending else []) + ["provider", "shutdown"]
    assert providers.closed


def test_speech_direction_dependency_creation_failure_shuts_down_ros(monkeypatch):
    from voice_asr_service.speech_direction import node

    events = []

    def build():
        raise ValueError("dependencies failed")

    monkeypatch.setattr(node, "build_model_service_runtime_dependencies", build)
    monkeypatch.setattr(node.rclpy, "init", lambda **_: None)
    monkeypatch.setattr(node.rclpy, "shutdown", lambda: events.append("shutdown"))
    with pytest.raises(ValueError, match="dependencies failed"):
        node.main()
    assert events == ["shutdown"]
