import threading
from types import SimpleNamespace

from voice_tts_service.voice_tts_node import VoiceTTSNode


class _Logger:
    def info(self, _message):
        pass


class _Session:
    instances = []

    def __init__(self):
        self.close_calls = 0
        self.infer = object()
        self.instances.append(self)

    def close(self):
        self.close_calls += 1


def _node():
    node = SimpleNamespace(
        _session=None,
        _core=SimpleNamespace(infer=None),
        _init_error="",
        _deployment="ascend_310p",
        get_logger=lambda: _Logger(),
    )
    node._new_session = _Session
    return node


def test_model_load_is_idempotent_and_unload_allows_new_session():
    _Session.instances = []
    node = _node()

    assert VoiceTTSNode._load_model_locked(node) is True
    first = node._session
    assert node._core.infer is first.infer

    assert VoiceTTSNode._load_model_locked(node) is False
    assert len(_Session.instances) == 1

    assert VoiceTTSNode._unload_model_locked(node) is True
    assert first.close_calls == 1
    assert node._session is None
    assert node._core.infer is None

    assert VoiceTTSNode._load_model_locked(node) is True
    assert len(_Session.instances) == 2
    assert node._session is _Session.instances[1]


def test_unload_is_idempotent():
    node = _node()

    assert VoiceTTSNode._unload_model_locked(node) is False
    assert node._session is None


def test_unload_handler_reports_session_close_failure():
    node = _node()
    node._session_lock = threading.RLock()

    def fail_unload():
        node._init_error = "lifecycle cleanup failed"
        raise RuntimeError("service-visible failure")

    node._unload_model_locked = fail_unload
    response = SimpleNamespace(success=True, message="")

    VoiceTTSNode._on_unload(node, None, response)

    assert response.success is False
    assert response.message == "service-visible failure"
    assert node._init_error == "lifecycle cleanup failed"
