import threading
from types import SimpleNamespace

from voice_tts_service.voice_tts_node import VoiceTTSNode


class _Logger:
    def info(self, _message):
        pass

    def error(self, _message):
        pass


class _Backend:
    instances = []

    def __init__(self, _bundle, _runtime_options):
        self.load_calls = 0
        self.close_calls = 0
        self.runtime_version = "fake-acl"
        self.instances.append(self)

    def load(self):
        self.load_calls += 1

    def close(self):
        self.close_calls += 1


def _node():
    return SimpleNamespace(
        _backend=None,
        _bundle=object(),
        _bundle_path="/bundle",
        _deployment="ascend_310p",
        _prompt_profile="default",
        _device_id=0,
        _core=SimpleNamespace(backend=None),
        _runtime_state="unloaded",
        _init_error="",
        get_logger=lambda: _Logger(),
    )


def test_model_load_is_idempotent_and_unload_allows_reload(monkeypatch):
    _Backend.instances = []
    monkeypatch.setattr("voice_tts_service.voice_tts_node.deployment_backend", lambda _bundle: "om")
    monkeypatch.setattr("voice_tts_service.voice_tts_node.AscendOmBackend", _Backend)
    node = _node()

    assert VoiceTTSNode._load_model_locked(node) is True
    first = node._backend
    assert first.load_calls == 1
    assert node._core.backend is first
    assert node._runtime_state == "ready"

    assert VoiceTTSNode._load_model_locked(node) is False
    assert len(_Backend.instances) == 1

    assert VoiceTTSNode._unload_model_locked(node) is True
    assert first.close_calls == 1
    assert node._backend is None
    assert node._core.backend is None
    assert node._runtime_state == "unloaded"

    assert VoiceTTSNode._load_model_locked(node) is True
    assert len(_Backend.instances) == 2
    assert node._backend is _Backend.instances[1]


def test_unload_is_idempotent():
    node = _node()

    assert VoiceTTSNode._unload_model_locked(node) is False
    assert node._runtime_state == "unloaded"


def test_unload_handler_leaves_failure_state_to_lifecycle_helper():
    node = _node()
    node._backend_lock = threading.Lock()

    def fail_unload(target):
        target._runtime_state = "failed"
        target._init_error = "lifecycle cleanup failed"
        raise RuntimeError("service-visible failure")

    node._unload_model_locked = lambda: fail_unload(node)
    response = SimpleNamespace(success=True, message="")

    VoiceTTSNode._on_unload(node, None, response)

    assert response.success is False
    assert response.message == "service-visible failure"
    assert node._runtime_state == "failed"
    assert node._init_error == "lifecycle cleanup failed"
