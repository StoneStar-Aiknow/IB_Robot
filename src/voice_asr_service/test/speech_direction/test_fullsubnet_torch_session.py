"""Tests for the manifest-backed FullSubNet Torch session."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _SRC.parent
for package_root in (
    _SRC,
    _WORKSPACE_SRC / "inference_manifest",
    _WORKSPACE_SRC / "inference_service",
):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import voice_asr_service.speech_direction.enhancement.fullsubnet_stateful_torch as torch_executor_module  # noqa: E402
from inference_manifest import (  # noqa: E402
    ArtifactBindings,
    Deployment,
    DeploymentArtifact,
    DeploymentTarget,
    ExecutionContract,
    RoleRuntimeProfile,
    StateLink,
    TorchRuntimeProfile,
    ValidatedDeployment,
)
from inference_service.backends.errors import BackendInferenceError, BackendLoadError  # noqa: E402
from inference_service.backends.types import RuntimeContext  # noqa: E402
from inference_service.unified_runtime import ExecutionContext, ModelRequest  # noqa: E402
from voice_asr_service.speech_direction.contract import speech_direction_bindings  # noqa: E402
from voice_asr_service.speech_direction.fullsubnet_torch_session import (  # noqa: E402
    FullSubNetTorchSession,
    build_fullsubnet_torch_session,
)

_STATE_SUFFIXES = ("_hidden_in", "_hidden_out", "_cell_in", "_cell_out", ".state_in", ".state_out")


class _FakeExecutor:
    backend = "stateful_torch_cpu"

    def __init__(self, checkpoint, state_contract, *, device, timing_enabled, cleanup=None):
        self.checkpoint = checkpoint
        self.state_contract = state_contract
        self.device = device
        self.timing_enabled = timing_enabled
        self.resets = 0
        self.closed = False
        if cleanup is not None:
            cleanup.defer(self.close)

    def run_fb(self, frame):
        return np.zeros((4, 2, 257), dtype=np.float32)

    def run_sb(self, frame):
        return np.zeros((1028, 2, 2), dtype=np.float32)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True

    @property
    def last_timing_ms(self):
        return {"fb_infer_ms": 1.0}


def _torch_bindings(role: str) -> ArtifactBindings:
    full = speech_direction_bindings(role)
    return ArtifactBindings(
        inputs=tuple(b for b in full.inputs if not b.semantic.endswith(_STATE_SUFFIXES)),
        outputs=tuple(b for b in full.outputs if not b.semantic.endswith(_STATE_SUFFIXES)),
    )


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "fullsubnet"
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "cum_fullsubnet_best_model_218epochs.tar").write_bytes(b"checkpoint")
    (root / "assets" / "cum_fullsubnet_best_model_218epochs.manifest.json").write_bytes(b"{}")
    return root


def _deployment(device: str = "cpu") -> Deployment:
    links = tuple(
        StateLink(
            role="__runtime__",
            state_name=state_name,
            owner="session",
            source=f"state.{state_name}_in",
            target=f"state.{state_name}_out",
            scope="runtime",
            state_bank=f"fullsubnet_{role}.bank",
        )
        for role in ("fb", "sb")
        for state_name in ("hidden", "cell")
    )
    return Deployment(
        execution_contract=ExecutionContract(
            state_scope="stream",
            execution_structure="direct",
            cancellation_granularity="checkpoint",
            stateful=True,
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
            state_links=links,
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
        artifacts={
            "fullsubnet_fb": DeploymentArtifact(
                path="assets/cum_fullsubnet_best_model_218epochs.tar", format="torch", sha256=None
            ),
            "fullsubnet_sb": DeploymentArtifact(
                path="assets/cum_fullsubnet_best_model_218epochs.manifest.json", format="json", sha256=None
            ),
        },
        execution=("fullsubnet_fb", "fullsubnet_sb"),
        bindings={
            "fullsubnet_fb": _torch_bindings("fullsubnet_fb"),
            "fullsubnet_sb": _torch_bindings("fullsubnet_sb"),
        },
    )


def _validated(root: Path, deployment: Deployment) -> ValidatedDeployment:
    return ValidatedDeployment(
        bundle_root=root,
        manifest_path=root / "inference_manifest.json",
        manifest=SimpleNamespace(),
        deployment_name="torch_cpu",
        deployment=deployment,
        top_level_identity=SimpleNamespace(interface="tensor_model", model_type="fullsubnet", operation="enhance"),
        role_identities={},
        role_runtime_profiles={},
        selected_deployment=deployment,
        semantic_contract=SimpleNamespace(),
        resolved_artifacts={},
        role_artifact_bindings={},
        declared_metadata={},
        integrity_status=None,
        deployment_fingerprint="fingerprint",
        runtime_profile_fingerprint="profile-fingerprint",
    )


def _context(root: Path, deployment: Deployment, runtime_options=None) -> RuntimeContext:
    return RuntimeContext(_validated(root, deployment), runtime_options or {}, role="fullsubnet_fb")


@pytest.fixture()
def fake_executor(monkeypatch):
    instances: list[_FakeExecutor] = []

    def factory(checkpoint, state_contract, *, device, timing_enabled, cleanup=None):
        executor = _FakeExecutor(
            checkpoint, state_contract, device=device, timing_enabled=timing_enabled, cleanup=cleanup
        )
        instances.append(executor)
        return executor

    monkeypatch.setattr(torch_executor_module, "StatefulTorchFullSubNetExecutor", factory)
    return instances


def test_builder_loads_and_executes_fb_sb_roles(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    assert isinstance(session, FullSubNetTorchSession)
    session.load(context)

    assert len(fake_executor) == 1
    assert fake_executor[0].device == "cpu"
    assert fake_executor[0].checkpoint.endswith("cum_fullsubnet_best_model_218epochs.tar")

    request = ModelRequest({}, {})
    execution = ExecutionContext("fullsubnet-test")
    fb = session.execute_role(
        "fullsubnet_fb",
        {"host.fullsubnet.fb_spectrum": np.zeros((4, 2, 257), dtype=np.float32)},
        request,
        execution,
    )
    assert fb["host.fullsubnet.fb_features"].shape == (4, 2, 257)
    sb = session.execute_role(
        "fullsubnet_sb",
        {"host.fullsubnet.sb_features": np.zeros((1028, 2, 32), dtype=np.float32)},
        request,
        execution,
    )
    assert sb["host.fullsubnet.sb_mask"].shape == (1028, 2, 2)

    session.reset()
    assert fake_executor[0].resets == 1
    session.close()
    assert fake_executor[0].closed is True


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_fullsubnet_role_entry_rechecks_control_with_unstarted_evidence(tmp_path, fake_executor, monkeypatch, stop):
    from datetime import datetime, timedelta, timezone

    from inference_service.unified_runtime import Deadline, ExecutionFailure, OutcomeState

    context = _context(_bundle(tmp_path), _deployment())
    session = build_fullsubnet_torch_session(context)
    session.load(context)
    now = [datetime(2026, 11, 3, tzinfo=timezone.utc)]
    execution = ExecutionContext("entry", Deadline.after(1, now=now[0], clock=lambda: now[0]))
    original_check = session._check_not_started_context
    calls = []

    def check(ctx, phase):
        original_check(ctx, phase)
        if stop == "cancel":
            ctx.token.cancel()
        else:
            now[0] += timedelta(seconds=2)

    original_run = fake_executor[0].run_fb
    monkeypatch.setattr(fake_executor[0], "run_fb", lambda frame: (calls.append("fb"), original_run(frame))[1])
    monkeypatch.setattr(session, "_check_not_started_context", check)
    inputs = {"host.fullsubnet.fb_spectrum": np.zeros((4, 2, 257), dtype=np.float32)}
    try:
        with pytest.raises(ExecutionFailure) as caught:
            session.execute_role("fullsubnet_fb", inputs, ModelRequest(), execution)
        assert caught.value.evidence.state is OutcomeState.NOT_STARTED
        assert caught.value.evidence.outcome_known and not caught.value.evidence.state_mutated and not calls
        monkeypatch.setattr(session, "_check_not_started_context", original_check)
        session.execute_role("fullsubnet_fb", inputs, ModelRequest(), ExecutionContext("valid"))
        assert calls == ["fb"]
    finally:
        session.close()


@pytest.mark.parametrize("sync_fails", [False, True])
@pytest.mark.parametrize("failure_kind", ["kernel", "shape"])
@pytest.mark.parametrize("role", ["fb", "sb"])
def test_registered_torch_session_fences_execution_reset_and_retryable_close(
    tmp_path, monkeypatch, sync_fails, failure_kind, role
):
    from contextlib import nullcontext

    import torch

    from inference_service.backends.types import BackendState

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.fb_model = SimpleNamespace(
                sequence_model=torch.nn.Identity(),
                fc_output_layer=torch.nn.Identity(),
                activate_function=torch.nn.Identity(),
            )
            self.sb_model = SimpleNamespace(sequence_model=torch.nn.Identity(), fc_output_layer=torch.nn.Identity())

    root = _bundle(tmp_path)
    (root / "assets/cum_fullsubnet_best_model_218epochs.manifest.json").write_text(
        '{"norm_type":"cumulative_laplace_norm"}'
    )
    monkeypatch.setattr(torch_executor_module, "_load_model_class", lambda: TinyModel)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {}})
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    session.load(context)
    native = session._executor
    native.device = torch.device("cuda:0")
    events = []
    blocked = [sync_fails]

    def sync(device):
        events.append("sync")
        if blocked[0]:
            raise RuntimeError("device incomplete")

    monkeypatch.setattr(
        torch_executor_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device=lambda _: nullcontext(),
                synchronize=sync,
                current_stream=lambda device: SimpleNamespace(synchronize=lambda: sync(device)),
            ),
            inference_mode=nullcontext,
            from_numpy=lambda _: SimpleNamespace(to=lambda _: None),
        ),
    )

    def fail(*args):
        if failure_kind == "shape":
            return torch.zeros(1), None, None
        raise RuntimeError("kernel failed after submission")

    setattr(native, f"_{role}_model", fail)
    from inference_service.unified_runtime import ExecutionFailure, LifecycleState, ModelRuntimeHandle

    semantic, shape = (
        ("host.fullsubnet.fb_spectrum", (4, 2, 257)) if role == "fb" else ("host.fullsubnet.sb_features", (1028, 2, 32))
    )
    role_request = ModelRequest({semantic: np.zeros(shape, np.float32)})
    stream_runtime = SimpleNamespace(
        open_stream=lambda _: object(),
        step=lambda _, request, ctx: session.execute_role(f"fullsubnet_{role}", request.inputs, request, ctx),
        reset_stream=lambda _, ctx: session.reset(ctx),
        close_stream=lambda *_: None,
        close=session.close,
    )
    owner = ModelRuntimeHandle(
        stream_runtime,
        streaming_runtime=stream_runtime,
        state_scope="stream",
        state_bank_mode="runtime_exclusive",
        stateful=True,
        resettable=True,
    )
    owner.load()
    stream = owner.open_stream(ExecutionContext("open"))
    with pytest.raises(ExecutionFailure) as failure:
        owner.step(stream, role_request, ExecutionContext("fb"))
    assert failure.value.evidence.outcome_known is (failure_kind == "shape" or not sync_fails)
    assert (failure.value.code == "state_outcome_unknown") is not failure.value.evidence.outcome_known
    assert ("kernel failed" if failure_kind == "kernel" else "输出必须") in str(failure.value)
    assert native._poisoned and events == ["sync"]
    if failure_kind == "shape":
        from inference_service.unified_runtime import OutcomeState

        assert failure.value.evidence.state is OutcomeState.COMPLETED
        assert failure.value.evidence.state_mutated
    assert owner.state is LifecycleState.RESET_REQUIRED
    with pytest.raises(ExecutionFailure, match="recovery_required"):
        owner.step(stream, role_request, ExecutionContext("blocked"))

    monkeypatch.setattr(native, "_reset_states", lambda: events.append("reset"))
    if sync_fails:
        with pytest.raises(RuntimeError, match="device incomplete"):
            session.reset()
        assert "reset" not in events
        with pytest.raises(ExecutionFailure) as pending:
            owner.close()
        assert pending.value.close_pending and session._executor is native
        assert getattr(native, f"_{role}_model") is fail
        blocked[0] = False
    else:
        owner.reset(ExecutionContext("recover"))
        assert events[-3:] == ["sync", "reset", "sync"] and not native._poisoned and owner.ready
    owner.close()
    owner.close()
    assert session.health().state is BackendState.CLOSED and session._executor is None
    assert native._fb_model is None and native._fb_hidden is None


def test_constructor_pins_cuda_device_for_model_state_reset_and_close(tmp_path, monkeypatch):
    import torch

    current = [1]
    placements, synchronized = [], []

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.fb_model = SimpleNamespace(
                sequence_model=torch.nn.Identity(),
                fc_output_layer=torch.nn.Identity(),
                activate_function=torch.nn.Identity(),
            )
            self.sb_model = SimpleNamespace(sequence_model=torch.nn.Identity(), fc_output_layer=torch.nn.Identity())

    root = _bundle(tmp_path)
    monkeypatch.setattr(torch_executor_module, "_load_model_class", lambda: TinyModel)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device.index))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: synchronized.append(device.index)),
    )
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, device: placements.append(device.index) or self)
    zeros = torch.zeros

    def allocate(*args, device, **kwargs):
        placements.append(device.index)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", allocate)
    native = torch_executor_module.StatefulTorchFullSubNetExecutor(
        str(root / "assets/cum_fullsubnet_best_model_218epochs.tar"),
        device="cuda",
    )
    current[0] = 0
    native.reset()
    native.close()
    assert placements and set(placements) == {1}
    assert synchronized and set(synchronized) == {1}
    assert native.device == torch.device("cuda:1") and native._cleanup_complete


def test_fullsubnet_close_terminates_after_bounded_drain_failures(monkeypatch):
    import threading

    from inference_service.backends.errors import BackendLifecycleError
    from inference_service.model_sessions.torch_completion import RELEASE_DRAIN_ATTEMPTS

    events = []

    def sticky_drain(_device):
        events.append("drain")
        raise RuntimeError("sticky device error")

    monkeypatch.setattr(
        torch_executor_module,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(synchronize=sticky_drain)),
    )
    native = torch_executor_module.StatefulTorchFullSubNetExecutor.__new__(
        torch_executor_module.StatefulTorchFullSubNetExecutor
    )
    native._lock, native._closed, native._poisoned = threading.Lock(), False, False
    native._cleanup_complete, native._release_drain_failures = False, 0
    native.device = SimpleNamespace(type="cuda")
    native._fb_model, native._sb_model = object(), object()
    for name in ("_fb_hidden", "_fb_cell", "_sb_hidden", "_sb_cell"):
        setattr(native, name, object())

    for _ in range(RELEASE_DRAIN_ATTEMPTS):
        with pytest.raises(BackendLifecycleError) as pending:
            native.close()
        assert pending.value.close_pending and not native._cleanup_complete
        assert native._fb_model is not None
    assert events == ["drain"] * RELEASE_DRAIN_ATTEMPTS

    # 第 3 次关闭放弃等待：不再 drain，直接释放引用并完成 close。
    native.close()
    assert native._fb_model is None and native._sb_model is None
    assert native._fb_hidden is None and native._sb_cell is None
    assert native._cleanup_complete
    assert events == ["drain"] * RELEASE_DRAIN_ATTEMPTS


def test_role_execution_rejects_invalid_shapes(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    session.load(context)

    with pytest.raises(BackendInferenceError) as invalid:
        session.execute_role(
            "fullsubnet_fb",
            {"host.fullsubnet.fb_spectrum": np.zeros((1, 2, 257), dtype=np.float32)},
            ModelRequest({}, {}),
            ExecutionContext("bad-shape"),
        )
    assert invalid.value.code == "role_input_shape_mismatch"

    with pytest.raises(BackendInferenceError) as missing:
        session.execute_role(
            "fullsubnet_fb",
            {"other": np.zeros((4, 2, 257), dtype=np.float32)},
            ModelRequest({}, {}),
            ExecutionContext("bad-semantic"),
        )
    assert missing.value.code in {"missing_semantic_input", "role_fullsubnet_fb_input_semantic_mismatch"}


def test_request_execution_requires_host_orchestration(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    session.load(context)

    with pytest.raises(BackendInferenceError) as exc:
        session.execute(ModelRequest({}, {}), ExecutionContext("request"))
    assert exc.value.code == "host_orchestration_required"


@pytest.mark.parametrize("failure_at", ["sb", "states"])
def test_partial_torch_constructor_retains_rollback_until_device_drain(tmp_path, monkeypatch, failure_at):
    import gc
    import weakref

    import torch

    from inference_service.backends.errors import BackendLifecycleError

    references, blocked = [], [True]

    class Model(torch.nn.Module):
        def __init__(self, **_):
            super().__init__()

    class Part(torch.nn.Module):
        def __init__(self, model):
            super().__init__()

        def to(self, device):
            references.append(weakref.ref(self))
            if failure_at == "sb" and len(references) == 2:
                raise RuntimeError("SB move failed")
            return self

    def fail_states(_):
        raise RuntimeError("state allocation failed")

    def drain(*_):
        if blocked[0]:
            raise RuntimeError("device still active")

    root = _bundle(tmp_path)
    (root / "assets/cum_fullsubnet_best_model_218epochs.manifest.json").write_text(
        '{"norm_type":"cumulative_laplace_norm"}'
    )
    monkeypatch.setattr(torch_executor_module, "_load_model_class", lambda: Model)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(torch_executor_module, "StatefulFullBandT2", Part)
    monkeypatch.setattr(torch_executor_module, "StatefulSubBandT2", Part)
    monkeypatch.setattr(torch_executor_module.StatefulTorchFullSubNetExecutor, "_reset_states", fail_states)
    # close 经 drain_torch_device_before_release 在 torch_completion 命名空间内
    # 解析 drain,monkeypatch 必须打在真实解析点才能模拟“设备完成未知”。
    from inference_service.model_sessions import torch_completion

    monkeypatch.setattr(torch_completion, "drain_torch_device", drain)
    session = FullSubNetTorchSession()
    with pytest.raises(BackendLoadError) as caught:
        session.load(_context(root, _deployment()))
    assert not caught.value.rollback_clean and caught.value.rollback_errors
    assert references[0]() is not None
    with pytest.raises(BackendLifecycleError) as pending:
        session.close()
    assert pending.value.close_pending
    blocked[0] = False
    session.close()
    gc.collect()
    assert all(ref() is None for ref in references)


def test_unknown_runtime_options_are_rejected(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment(), {"device": "cuda"})
    session = build_fullsubnet_torch_session(context)
    with pytest.raises(BackendLoadError) as exc:
        session.load(context)
    assert exc.value.code == "invalid_runtime_options"


def test_builder_rejects_non_torch_identity(tmp_path) -> None:
    root = _bundle(tmp_path)
    deployment = _deployment()
    validated = _validated(root, deployment)
    object.__setattr__(
        validated,
        "top_level_identity",
        SimpleNamespace(interface="tensor_model", model_type="silero_vad", operation="vad"),
    )
    context = RuntimeContext(validated, {}, role="fullsubnet_fb")
    with pytest.raises(BackendLoadError) as exc:
        build_fullsubnet_torch_session(context)
    assert exc.value.code == "invalid_deployment"


@pytest.mark.parametrize("poisoned", [False, True])
def test_fullsubnet_healthy_reset_does_not_drain_other_streams(monkeypatch, poisoned):
    import threading

    events = []
    device = SimpleNamespace(type="cuda")
    stream = SimpleNamespace(synchronize=lambda: events.append("initialization-stream"))
    monkeypatch.setattr(
        torch_executor_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device=lambda value: nullcontext(),
                current_stream=lambda value: stream,
                synchronize=lambda value: events.append("device"),
            )
        ),
    )
    native = torch_executor_module.StatefulTorchFullSubNetExecutor.__new__(
        torch_executor_module.StatefulTorchFullSubNetExecutor
    )
    native._lock, native._closed, native._poisoned, native.device = threading.Lock(), False, poisoned, device
    native._reset_states = lambda: events.append("initialize")
    native.reset()
    assert events == (["device"] if poisoned else []) + ["initialize", "initialization-stream"]
    assert not native._poisoned
