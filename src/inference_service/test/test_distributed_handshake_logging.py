from types import SimpleNamespace

import inference_service.pipeline_policy_node as pipeline_policy_module
from ibrobot_msgs.msg import InferencePipelineStatus
from inference_service.pipeline_policy_node import PipelinePolicyNode


class _Logger:
    def __init__(self):
        self.warning_messages = []
        self.info_messages = []

    def warning(self, message, **_kwargs):
        self.warning_messages.append(message)

    def info(self, message, **_kwargs):
        self.info_messages.append(message)


class _Session:
    def __init__(self, *, ready=False, heartbeat_error=None):
        self.ready = ready
        self.heartbeat_error = heartbeat_error
        self.session = ("", 0)

    def observe_cloud(self, _status):
        return SimpleNamespace(error=None, invalidated_request_ids=())

    def expire_heartbeat(self):
        return SimpleNamespace(error=self.heartbeat_error, invalidated_request_ids=())


def _node(session, logger, *, started_at=0.0, received_at=None):
    return SimpleNamespace(
        _require_edge_session=lambda: session,
        _video_stream_manager=None,
        _distributed_started_monotonic=started_at,
        _last_cloud_status_received_monotonic=received_at,
        _config=SimpleNamespace(heartbeat_topic="/inference/policy/heartbeat"),
        get_logger=lambda: logger,
        _complete_invalidated=lambda *_args: None,
    )


def test_check_heartbeat_explains_missing_cloud_heartbeat(monkeypatch):
    logger = _Logger()
    session = _Session()
    node = _node(session, logger, started_at=10.0)
    monkeypatch.setattr(pipeline_policy_module.time, "monotonic", lambda: 16.0)

    PipelinePolicyNode._check_heartbeat(node)

    assert len(logger.warning_messages) == 1
    assert "has not been heard from" in logger.warning_messages[0]
    assert "pure_inference_node" in logger.warning_messages[0]
    assert "/inference/policy/heartbeat" in logger.warning_messages[0]
    # The edge half being up while the cloud half is silent must not read as a
    # silent downgrade to single-machine inference.
    assert "not a fallback to local inference" in logger.warning_messages[0]


def test_check_heartbeat_warning_covers_the_startup_window_only(monkeypatch):
    logger = _Logger()
    session = _Session(ready=False)
    # Same moment as the test above, well past the warning delay: the only
    # difference is that a cloud status has been received.
    node = _node(session, logger, started_at=10.0, received_at=12.0)
    monkeypatch.setattr(pipeline_policy_module.time, "monotonic", lambda: 16.0)

    PipelinePolicyNode._check_heartbeat(node)

    assert logger.warning_messages == []


def test_cloud_status_logging_distinguishes_not_ready_and_handshake(monkeypatch):
    logger = _Logger()
    session = _Session()
    node = _node(session, logger, received_at=None)
    status = SimpleNamespace(ready=False, runtime_state="loading")
    monkeypatch.setattr(pipeline_policy_module, "status_from_message", lambda _message: status)

    message = SimpleNamespace(role=InferencePipelineStatus.ROLE_CLOUD)
    PipelinePolicyNode._cloud_status_callback(node, message)

    assert len(logger.warning_messages) == 1
    assert "discovered but is not ready" in logger.warning_messages[0]
    assert "runtime_state='loading'" in logger.warning_messages[0]


def test_cloud_status_logging_reports_handshake_establishment(monkeypatch):
    logger = _Logger()
    session = _Session()
    node = _node(session, logger)
    status = SimpleNamespace(ready=True, runtime_state="ready", session_id="session-a", session_generation=3)

    def observe_cloud(_status):
        session.ready = True
        return SimpleNamespace(invalidated_request_ids=(), error=None)

    session.observe_cloud = observe_cloud
    session.session = ("session-a", 3)
    monkeypatch.setattr(pipeline_policy_module, "status_from_message", lambda _message: status)

    message = SimpleNamespace(role=InferencePipelineStatus.ROLE_CLOUD)
    PipelinePolicyNode._cloud_status_callback(node, message)

    assert logger.info_messages == ["Cloud handshake established: session_id=session-a, session_generation=3"]
