from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from inference_manifest import load_inference_manifest
from inference_service.backends import (
    BACKEND_REGISTRY,
    BackendLoadError,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


def _make_context(
    root: Path,
    *,
    policy_type: str = "act",
    device: str = "cpu",
    runtime_options: dict[str, object] | None = None,
) -> RuntimeContext:
    root.mkdir()
    bundle_paths = create_policy_bundle(root, policy_type=policy_type)
    manifest = make_manifest(root, bundle_paths, deployment_name=device)
    manifest["deployments"][device]["device"] = device
    write_manifest(root, manifest)
    return RuntimeContext(load_inference_manifest(root, device), runtime_options=runtime_options or {})


def _install_fake_lerobot(monkeypatch, torch_module, calls, *, attention: bool = True, reset: bool = True):
    lerobot_module = ModuleType("lerobot")
    configs_module = ModuleType("lerobot.configs")
    policies_config_module = ModuleType("lerobot.configs.policies")
    policies_module = ModuleType("lerobot.policies")
    factory_module = ModuleType("lerobot.policies.factory")

    class FakeConfig:
        def __init__(self, policy_type: str, device: str, vlm_model_name: str | None = None) -> None:
            self.type = policy_type
            self.device = device
            self.vlm_model_name = vlm_model_name

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            bundle = Path(path)
            raw = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
            calls["config_path"] = path
            calls["config_kwargs"] = kwargs
            calls["persisted_device"] = raw["device"]
            return cls(raw["type"], raw["device"], raw.get("vlm_model_name"))

    class FakeModel:
        supports_attention = attention

        def __init__(self) -> None:
            self.parameter = torch_module.tensor([1.0], dtype=torch_module.float32)

        def to(self, device):
            self.parameter = self.parameter.to(device)
            return self

        def half(self):
            self.parameter = self.parameter.half()
            return self

        def bfloat16(self):
            self.parameter = self.parameter.bfloat16()
            return self

        def float(self):
            self.parameter = self.parameter.float()
            return self

    class FakePolicy:
        supports_attention = attention

        def __init__(self, config) -> None:
            self.config = config
            self.model = FakeModel()
            self.to_devices = []
            self.eval_calls = 0
            self.reset_calls = 0
            self.last_batch = None
            self.last_noise = None

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["policy_path"] = path
            calls["policy_kwargs"] = kwargs
            calls["runtime_device"] = kwargs["config"].device
            policy = cls(kwargs["config"])
            calls["policy"] = policy
            return policy

        def to(self, device):
            self.to_devices.append(str(device))
            self.model.to(device)
            return self

        def eval(self):
            self.eval_calls += 1
            return self

        def parameters(self):
            yield self.model.parameter

        def predict_action_chunk(self, batch, noise=None):
            self.last_batch = batch
            self.last_noise = noise
            return torch_module.arange(6, dtype=torch_module.float32, device=self.model.parameter.device).reshape(
                1, 2, 3
            )

        def select_action(self, batch):
            self.last_batch = batch
            return torch_module.arange(3, dtype=torch_module.float32, device=self.model.parameter.device).reshape(1, 3)

    if reset:

        def reset_policy(self):
            self.reset_calls += 1

        FakePolicy.reset = reset_policy

    def get_policy_class(policy_type):
        calls["policy_type"] = policy_type
        return FakePolicy

    def get_policy_config_class(policy_type):
        calls["config_policy_type"] = policy_type
        calls["config_resolved_before_load"] = "config_path" not in calls
        return FakeConfig

    def make_pre_post_processors(*, policy_cfg, pretrained_path, **kwargs):
        calls["processor_config"] = policy_cfg
        calls["processor_path"] = pretrained_path
        calls["processor_kwargs"] = kwargs

        def preprocessor(batch):
            return batch

        def postprocessor(action):
            return action

        calls["preprocessor"] = preprocessor
        calls["postprocessor"] = postprocessor
        return preprocessor, postprocessor

    policies_config_module.PreTrainedConfig = FakeConfig
    factory_module.get_policy_class = get_policy_class
    factory_module.get_policy_config_class = get_policy_config_class
    factory_module.make_pre_post_processors = make_pre_post_processors
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", policies_config_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_module)


def test_torch_backend_loads_original_bundle_and_preserves_native_inference_contract(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from inference_service.backends.torch import create_backend

    context = _make_context(tmp_path / "bundle", policy_type="pi05")
    config_path = context.validated_manifest.bundle_root / "config.json"
    original_config = config_path.read_bytes()
    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)

    def fail_tempdir(*args, **kwargs):
        raise AssertionError(f"TorchBackend attempted temporary materialization: {args}, {kwargs}")

    monkeypatch.setattr(tempfile, "mkdtemp", fail_tempdir)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_tempdir)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", fail_tempdir)
    backend = create_backend(context)
    assert backend.capabilities.resettable is False
    backend.load(context)

    bundle_path = str(context.validated_manifest.bundle_root)
    assert calls["config_path"] == bundle_path
    assert calls["config_kwargs"] == {"local_files_only": True}
    assert calls["policy_path"] == bundle_path
    assert calls["processor_path"] == bundle_path
    assert calls["persisted_device"] == "cuda"
    assert calls["runtime_device"] == "cpu"
    assert calls["policy_type"] == "pi05"
    assert calls["config_policy_type"] == "pi05"
    assert calls["config_resolved_before_load"] is True
    assert calls["policy_kwargs"]["config"] is calls["processor_config"]
    assert calls["policy_kwargs"]["local_files_only"] is True
    assert config_path.read_bytes() == original_config
    assert calls["policy"].to_devices == ["cpu"]
    assert calls["policy"].eval_calls == 1
    assert backend.policy_config.type == "pi05"
    assert backend.preprocessor is calls["preprocessor"]
    assert backend.postprocessor is calls["postprocessor"]
    assert backend.health().state is BackendState.READY
    assert backend.capabilities.resettable is True
    assert backend.capabilities.stateful is True
    assert backend.capabilities.supports_attention is True
    assert backend.capabilities.supports_cancellation is False
    assert backend.capabilities.thread_safe is False
    assert backend.capabilities.max_in_flight_per_instance == 1
    assert backend.capabilities.supports_multiple_instances is True
    assert backend.capabilities.resource_domain == "torch:cpu"
    assert backend.capabilities.resource_domain_limit == 1

    noise = torch.ones((1, 2, 3), dtype=torch.float64)
    result = backend.infer(
        InferenceRequest(
            request_id="chunk",
            inputs={
                "observation.state": np.array([[1, 2, 3]], dtype=np.float32),
                "task": "canonical task",
                "_noise": noise,
            },
            prompt="request prompt",
        )
    )

    assert tuple(result.action.shape) == (2, 3)
    assert result.actual_chunk_size == 2
    assert result.backend_latency_ms >= 0
    assert result.metadata["request_id"] == "chunk"
    assert result.metadata["policy_type"] == "pi05"
    assert result.metadata["device"] == "cpu"
    assert result.metadata["action_method"] == "predict_action_chunk"
    assert result.metadata["deployment_fingerprint"] == context.deployment_fingerprint
    assert result.metadata["external_noise"] is True
    assert calls["policy"].last_batch["task"] == "canonical task"
    assert "_noise" not in calls["policy"].last_batch
    assert calls["policy"].last_noise.dtype == calls["policy"].model.parameter.dtype
    assert calls["policy"].last_noise.device == calls["policy"].model.parameter.device

    selected = backend.infer(
        InferenceRequest(
            request_id="single",
            inputs={"observation.state": torch.zeros((1, 3)), "task": "canonical task"},
            metadata={"action_method": "select_action"},
        )
    )
    assert tuple(selected.action.shape) == (3,)
    assert selected.actual_chunk_size == 1
    assert calls["policy"].last_batch["task"] == "canonical task"

    backend.reset()
    assert calls["policy"].reset_calls == 1
    backend.close()
    backend.close()
    assert backend.policy is None
    assert backend.policy_config is None
    assert backend.health().state is BackendState.CLOSED


def test_torch_backend_resolves_local_semantic_assets_without_rewriting_bundle(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from inference_service.backends.torch import create_backend

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    bundle_paths = create_policy_bundle(bundle, policy_type="smolvla", local_tokenizer=True)
    write_manifest(bundle, make_manifest(bundle, bundle_paths))
    context = RuntimeContext(load_inference_manifest(bundle, "cpu"))
    original_config = (bundle / "config.json").read_bytes()
    original_preprocessor = (bundle / "policy_preprocessor.json").read_bytes()
    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)

    backend = create_backend(context)
    backend.load(context)

    tokenizer_path = str((bundle / "tokenizer").resolve())
    assert calls["policy_kwargs"]["config"].vlm_model_name == tokenizer_path
    assert calls["processor_kwargs"]["preprocessor_overrides"] == {
        "device_processor": {"device": "cpu"},
        "tokenizer_processor": {"tokenizer_name": tokenizer_path},
    }
    assert calls["processor_kwargs"]["postprocessor_overrides"] == {
        "device_processor": {"device": "cpu"},
    }
    assert (bundle / "config.json").read_bytes() == original_config
    assert (bundle / "policy_preprocessor.json").read_bytes() == original_preprocessor
    backend.close()


@pytest.mark.parametrize(
    ("model_dtype", "expected_dtype"),
    [("native", "float32"), ("fp16", "float16"), ("bf16", "bfloat16"), ("fp32", "float32")],
)
def test_torch_backend_applies_validated_model_dtype_runtime_option(monkeypatch, tmp_path, model_dtype, expected_dtype):
    torch = pytest.importorskip("torch")
    from inference_service.backends.torch import create_backend

    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    context = _make_context(
        tmp_path / "bundle",
        runtime_options={"model_dtype": model_dtype},
    )
    backend = create_backend(context)
    backend.load(context)

    assert str(calls["policy"].model.parameter.dtype).removeprefix("torch.") == expected_dtype
    assert calls["policy"].eval_calls == 1
    result = backend.infer(
        InferenceRequest(
            request_id="dtype",
            inputs={"observation.state": torch.zeros((1, 3))},
        )
    )
    assert result.metadata["model_dtype"] == model_dtype
    backend.close()


@pytest.mark.parametrize(
    "runtime_options",
    [{"model_dtype": "float64"}, {"unknown": True}],
)
def test_torch_backend_rejects_invalid_runtime_options(monkeypatch, tmp_path, runtime_options):
    torch = pytest.importorskip("torch")
    from inference_service.backends.torch import create_backend

    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    context = _make_context(tmp_path / "bundle", runtime_options=runtime_options)
    backend = create_backend(context)

    with pytest.raises(BackendLoadError) as error:
        backend.load(context)

    assert error.value.code == "invalid_runtime_options"
    backend.close()


def test_static_registry_creates_lazy_torch_factory_without_loading_policy(tmp_path):
    context = _make_context(tmp_path / "bundle")

    backend = BACKEND_REGISTRY.create(context)

    assert backend.name == "torch"
    assert backend.health().state is BackendState.CREATED
    assert backend.policy is None
    backend.close()


def test_two_cpu_torch_pipelines_construct_and_serialize_shared_device_domain(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from inference_service.pipeline import create_inference_pipeline

    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    context = _make_context(tmp_path / "bundle")
    first = create_inference_pipeline("first", context.validated_manifest)
    second = create_inference_pipeline("second", context.validated_manifest)
    first.load()
    second.load()

    active = 0
    max_active = 0
    lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()

    def blocking_chunk(batch, noise=None):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        entered.set()
        try:
            release.wait(timeout=2)
            return torch.zeros((1, 2, 6), dtype=torch.float32)
        finally:
            with lock:
                active -= 1

    first._backend.policy.predict_action_chunk = blocking_chunk
    second._backend.policy.predict_action_chunk = blocking_chunk
    errors = []

    def infer(pipeline, request_id):
        try:
            pipeline.infer(
                InferenceRequest(
                    request_id=request_id,
                    inputs={"observation.state": torch.zeros((1, 3))},
                )
            )
        except Exception as exc:
            errors.append(exc)

    first_thread = threading.Thread(target=infer, args=(first, "first"))
    second_thread = threading.Thread(target=infer, args=(second, "second"))
    first_thread.start()
    assert entered.wait(timeout=2)
    second_thread.start()
    time.sleep(0.05)

    assert max_active == 1
    release.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert errors == []
    assert max_active == 1
    first.close()
    second.close()


def test_torch_backend_reports_non_resettable_policy_and_unobservable_attention(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from inference_service.backends.torch import create_backend

    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls, attention=False, reset=False)
    context = _make_context(tmp_path / "bundle")
    backend = create_backend(context)
    backend.load(context)

    assert backend.capabilities.resettable is False
    assert backend.capabilities.stateful is False
    assert backend.capabilities.supports_attention is False
    backend.close()


def test_torch_backend_detects_hookable_act_decoder_attention():
    from inference_service.backends.torch import TorchBackend

    policy = SimpleNamespace(
        model=SimpleNamespace(
            decoder=SimpleNamespace(layers=[SimpleNamespace(multihead_attn=object())]),
        )
    )

    assert TorchBackend._observes_attention(policy) is True


def test_torch_backend_does_not_import_accelerator_extension_for_cpu(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import inference_service.backends.torch as torch_backend_module

    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    imported = []
    original_import = torch_backend_module.importlib.import_module

    def guarded_import(name):
        imported.append(name)
        if name == "torch_npu":
            raise AssertionError("CPU backend imported torch_npu")
        return original_import(name)

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", guarded_import)
    context = _make_context(tmp_path / "bundle")
    backend = torch_backend_module.create_backend(context)
    backend.load(context)
    backend.close()

    assert "torch_npu" not in imported


def test_torch_backend_reports_missing_torch_dependency(monkeypatch, tmp_path):
    import inference_service.backends.torch as torch_backend_module

    def unavailable(name):
        if name == "torch":
            raise ImportError("not installed")
        return __import__(name)

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", unavailable)
    context = _make_context(tmp_path / "bundle")
    backend = torch_backend_module.create_backend(context)
    with pytest.raises(BackendLoadError, match="PyTorch.*unavailable") as error:
        backend.load(context)
    assert error.value.code == "missing_dependency"
    backend.close()


def test_torch_backend_reports_missing_lerobot_dependency(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import inference_service.backends.torch as torch_backend_module

    original_import = torch_backend_module.importlib.import_module

    def unavailable(name):
        if name == "torch":
            return torch
        if name.startswith("lerobot"):
            raise ImportError("not installed")
        return original_import(name)

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", unavailable)
    context = _make_context(tmp_path / "bundle")
    backend = torch_backend_module.create_backend(context)
    with pytest.raises(BackendLoadError, match="LeRobot.*unavailable") as error:
        backend.load(context)
    assert error.value.code == "missing_dependency"
    backend.close()


def test_torch_backend_rejects_unavailable_cuda_before_lerobot_import(monkeypatch, tmp_path):
    import inference_service.backends.torch as torch_backend_module

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
    )
    imported = []

    def fake_import(name):
        imported.append(name)
        if name == "torch":
            return fake_torch
        raise AssertionError(f"unexpected import after unavailable device: {name}")

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", fake_import)
    context = _make_context(tmp_path / "bundle", device="cuda")
    backend = torch_backend_module.create_backend(context)
    with pytest.raises(BackendLoadError, match="cuda.*not available") as error:
        backend.load(context)
    assert error.value.code == "device_unavailable"
    assert imported == ["torch"]
    backend.close()


def test_torch_backend_imports_torch_npu_only_when_npu_is_selected(monkeypatch, tmp_path):
    import inference_service.backends.torch as torch_backend_module

    fake_torch = SimpleNamespace(device=lambda name: name)
    imported = []

    def fake_import(name):
        imported.append(name)
        if name == "torch":
            return fake_torch
        if name == "torch_npu":
            raise ImportError("not installed")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", fake_import)
    context = _make_context(tmp_path / "bundle", device="npu")
    backend = torch_backend_module.create_backend(context)
    with pytest.raises(BackendLoadError, match="torch_npu.*unavailable") as error:
        backend.load(context)
    assert error.value.code == "missing_dependency"
    assert imported == ["torch", "torch_npu"]
    backend.close()


@pytest.mark.parametrize("device", ["cuda", "mps", "npu"])
def test_torch_backend_optional_accelerator_load(monkeypatch, tmp_path, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("Torch NPU is unavailable")

    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    from inference_service.backends.torch import create_backend

    context = _make_context(tmp_path / "bundle", device=device)
    backend = create_backend(context)
    backend.load(context)
    assert calls["policy"].to_devices == [device]
    assert backend.capabilities.resettable is True
    backend.close()
