from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from inference_manifest import load_inference_manifest
from inference_service.distributed import (
    CloudSession,
    DistributedCloudService,
    DistributedProtocolError,
    DistributedRequest,
    DistributedResult,
    EdgeSession,
    Operation,
    PeerRole,
    PipelineStatus,
    StructuredError,
    build_pipeline_identity,
)
from inference_service.distributed.ros_protocol import (
    decode_failure_result,
    request_from_message,
    request_to_message,
    result_from_message,
    result_to_message,
    status_from_message,
    status_to_message,
)
from inference_service.pipeline import PipelineState
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


def _identity(root: Path, pipeline_id: str = "policy"):
    root.mkdir()
    paths = create_policy_bundle(root)
    config_path = root / "config.json"
    config = config_path.read_text(encoding="utf-8").replace('"device": "cuda"', '"chunk_size": 4,\n  "device": "cuda"')
    config_path.write_text(config, encoding="utf-8")
    write_manifest(root, make_manifest(root, paths))
    return build_pipeline_identity(pipeline_id, load_inference_manifest(root, "cpu"))


def _handshake(edge: EdgeSession, cloud: CloudSession) -> PipelineStatus:
    edge.start()
    edge_status = edge.local_status()
    assert cloud.observe_edge(edge_status, backend_ready=True) is None
    cloud_status = cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=True,
        cancellation_supported=True,
    )
    update = edge.observe_cloud(cloud_status, now=10.0)
    assert update.error is None
    assert edge.ready
    return cloud_status


def test_identity_summary_is_canonical_and_includes_chunk_contract(tmp_path):
    identity = _identity(tmp_path / "bundle")

    assert identity.pipeline_id == "policy"
    assert [feature.semantic for feature in identity.policy.inputs] == [
        "observation.images.top",
        "observation.state",
    ]
    assert identity.policy.action_dimension == 6
    assert identity.policy.nominal_chunk_size == 4


def test_matching_handshake_gates_requests_and_routes_result(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)

    with pytest.raises(DistributedProtocolError) as not_ready:
        edge.prepare_request(Operation.INFER, "request-1")
    assert not_ready.value.code == "not_ready"

    cloud_status = _handshake(edge, cloud)
    request = edge.prepare_request(
        Operation.INFER,
        "request-1",
        inputs={"observation.state": np.zeros((1, 6), dtype=np.float32)},
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    cloud.validate_request(request)
    result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=request.request_id,
        session_id=cloud_status.session_id,
        session_generation=cloud_status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        action=np.zeros((2, 6), dtype=np.float32),
        actual_chunk_size=2,
        backend_latency_ms=1.5,
        backend_ready=True,
        backend_state="ready",
    )

    update = edge.accept_result(result)

    assert update.error is None


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("pipeline_id", "pipeline_id_mismatch"),
        ("bundle_digest", "bundle_digest_mismatch"),
        ("deployment_name", "deployment_mismatch"),
        ("deployment_fingerprint", "deployment_fingerprint_mismatch"),
    ],
)
def test_handshake_rejects_identity_mismatches(tmp_path, field, code):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    edge.start()
    remote_identity = replace(identity, **{field: f"different-{field}"})
    status = PipelineStatus(
        role=PeerRole.CLOUD,
        identity=remote_identity,
        sequence=1,
        session_id="session",
        session_generation=1,
        ready=True,
        runtime_state="ready",
    )

    update = edge.observe_cloud(status)

    assert update.error is not None
    assert update.error.code == code
    assert edge.state is PipelineState.HANDSHAKING


def test_heartbeat_loss_invalidates_all_in_flight_requests(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity, heartbeat_timeout=1.0)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "first")
    edge.prepare_request(Operation.INFER, "second")

    update = edge.expire_heartbeat(now=11.1)

    assert update.error is not None
    assert update.error.code == "heartbeat_expired"
    assert update.invalidated_request_ids == ("first", "second")
    assert edge.state is PipelineState.DEGRADED


def test_heartbeat_recovery_requires_a_new_session(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity, heartbeat_timeout=1.0)
    cloud = CloudSession(identity)
    ready = _handshake(edge, cloud)

    edge.expire_heartbeat(now=11.1)
    stale = edge.observe_cloud(replace(ready, sequence=ready.sequence + 1), now=11.2)

    assert stale.error is not None
    assert stale.error.code == "stale_status"
    assert edge.state is PipelineState.DEGRADED

    assert cloud.observe_edge(edge.local_status(), backend_ready=True) is None
    recovered = cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=False,
        cancellation_supported=False,
    )
    edge.observe_cloud(recovered, now=11.3)

    assert edge.ready
    assert recovered.session_id != ready.session_id
    assert recovered.session_generation > ready.session_generation


def test_cloud_restart_creates_new_session_and_stale_response_is_discarded(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    first_cloud = CloudSession(identity)
    first_status = _handshake(edge, first_cloud)
    old_request = edge.prepare_request(Operation.INFER, "old")

    restarted_cloud = CloudSession(identity)
    assert restarted_cloud.observe_edge(edge.local_status(), backend_ready=True) is None
    restarted_status = restarted_cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=False,
        cancellation_supported=False,
    )
    update = edge.observe_cloud(restarted_status)

    assert update.invalidated_request_ids == ("old",)
    assert restarted_status.session_id != first_status.session_id

    edge.prepare_request(Operation.INFER, "current")
    stale = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=old_request.request_id,
        session_id=old_request.session_id,
        session_generation=old_request.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        action=np.zeros((1, 6), dtype=np.float32),
        actual_chunk_size=1,
        backend_ready=True,
        backend_state="ready",
    )
    stale_update = edge.accept_result(stale)

    assert stale_update.error is not None
    assert stale_update.error.code == "stale_response"


def test_remote_backend_failure_revokes_ready_session(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    ready = _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "request")
    failed = replace(
        ready,
        sequence=ready.sequence + 1,
        ready=False,
        runtime_state="failed",
        error=StructuredError(
            code="device_lost",
            message="cloud accelerator failed",
            stage="backend",
            recoverable=True,
        ),
    )

    update = edge.observe_cloud(failed)

    assert update.invalidated_request_ids == ("request",)
    assert update.error is not None
    assert update.error.code == "device_lost"
    assert edge.state is PipelineState.DEGRADED


def test_cloud_rejects_unknown_or_stale_session_requests(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    request = edge.prepare_request(Operation.RESET, "reset")
    stale = replace(request, session_generation=request.session_generation + 1)

    with pytest.raises(DistributedProtocolError) as error:
        cloud.validate_request(stale)

    assert error.value.code == "session_generation_mismatch"


def test_cloud_rejects_replayed_request_ids(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    request = edge.prepare_request(Operation.RESET, "reset")
    cloud.validate_request(request)

    with pytest.raises(DistributedProtocolError) as error:
        cloud.validate_request(request)

    assert error.value.code == "duplicate_request_id"


def test_cancel_request_requires_target_and_tracks_capability(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "inference")

    cancel = edge.prepare_request(Operation.CANCEL, "cancel", target_request_id="inference")

    assert edge.cancellation_supported is True
    assert cancel.target_request_id == "inference"


def test_expired_deadline_is_rejected_before_transmission(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    _handshake(edge, cloud)

    with pytest.raises(DistributedProtocolError) as error:
        edge.prepare_request(
            Operation.INFER,
            "expired",
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

    assert error.value.code == "deadline_exceeded"


def test_structured_cloud_error_can_be_matched_without_action_payload(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    request = edge.prepare_request(Operation.INFER, "failed")
    result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=request.request_id,
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=False,
        backend_ready=True,
        backend_state="ready",
        error=StructuredError(
            code="invalid_input",
            message="tensor decode failed",
            stage="decode",
        ),
    )

    update = edge.accept_result(result)

    assert update.error is None


def test_result_operation_must_match_pending_request(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    edge.prepare_request(Operation.INFER, "request")
    mismatched = DistributedResult(
        operation=Operation.RESET,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        backend_ready=True,
        backend_state="ready",
    )

    update = edge.accept_result(mismatched)

    assert update.error is not None
    assert update.error.code == "stale_response"


def test_unknown_operation_error_can_complete_any_pending_operation(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    edge.prepare_request(Operation.RESET, "request")
    failed = DistributedResult(
        operation=Operation.UNKNOWN,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=False,
        backend_ready=False,
        backend_state="decode_failed",
        error=StructuredError(
            code="decode_failed",
            message="invalid operation",
            stage="decode",
        ),
    )

    update = edge.accept_result(failed)

    assert update.error is not None
    assert update.error.code == "remote_backend_unavailable"


def test_dispatch_request_cannot_be_invalidated_between_registration_and_send(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    observed: list[str] = []

    def sender(request):
        observed.append(request.request_id)
        failed = replace(
            status,
            sequence=status.sequence + 1,
            ready=False,
            runtime_state="failed",
            error=StructuredError(
                code="device_lost",
                message="cloud accelerator failed",
                stage="backend",
                recoverable=True,
            ),
        )
        update = edge.observe_cloud(failed)
        assert update.invalidated_request_ids == ("request",)

    request = edge.dispatch_request(Operation.INFER, "request", sender)

    assert request.request_id == "request"
    assert observed == ["request"]
    assert edge.state is PipelineState.DEGRADED


def test_cloud_startup_failure_is_reported_in_status_and_results(tmp_path):
    identity = _identity(tmp_path / "bundle")
    startup_error = StructuredError(
        code="missing_dependency",
        message="tcim runtime is unavailable",
        stage="startup",
    )
    service = DistributedCloudService(identity, None, startup_error=startup_error)
    edge = EdgeSession(identity)
    edge.start()

    status = service.observe_edge(edge.local_status())

    assert status.ready is False
    assert status.error == startup_error

    request = DistributedRequest(
        operation=Operation.RESET,
        pipeline_id=identity.pipeline_id,
        request_id="request",
        session_id="unavailable",
        session_generation=1,
        deployment_fingerprint=identity.deployment_fingerprint,
    )
    result = service.handle(request)

    assert result.success is False
    assert result.pipeline_id == identity.pipeline_id
    assert result.error == startup_error


def test_unknown_pipeline_error_uses_cloud_pipeline_identity(tmp_path):
    identity = _identity(tmp_path / "bundle")

    class Runtime:
        capabilities = type("Capabilities", (), {"resettable": False, "supports_cancellation": False})()

        @staticmethod
        def health():
            state = type("State", (), {"value": "ready"})()
            return type("Health", (), {"ready": True, "state": state})()

        @staticmethod
        def close():
            return None

    service = DistributedCloudService(identity, Runtime())
    edge = EdgeSession(identity)
    edge.start()
    cloud_status = service.observe_edge(edge.local_status())
    edge.observe_cloud(cloud_status)
    request = edge.prepare_request(Operation.RESET, "request")

    result = service.handle(replace(request, pipeline_id="unknown"))

    assert result.pipeline_id == identity.pipeline_id
    assert result.error is not None
    assert result.error.code == "pipeline_not_found"
    assert edge.accept_result(result).error is None


def test_ros_protocol_round_trips_status_request_and_result(tmp_path):
    identity = _identity(tmp_path / "bundle")
    edge = EdgeSession(identity)
    cloud = CloudSession(identity)
    status = _handshake(edge, cloud)
    request = edge.prepare_request(
        Operation.INFER,
        "request",
        inputs={"observation.state": np.zeros((1, 6), dtype=np.float32)},
        deadline=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    result = DistributedResult(
        operation=Operation.INFER,
        pipeline_id=identity.pipeline_id,
        request_id=request.request_id,
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint=identity.deployment_fingerprint,
        success=True,
        action=np.zeros((2, 6), dtype=np.float32),
        actual_chunk_size=2,
        backend_latency_ms=1.5,
        backend_ready=True,
        backend_state="ready",
    )

    assert status_from_message(status_to_message(status)) == status
    decoded_request = request_from_message(request_to_message(request))
    assert decoded_request.operation == request.operation
    assert decoded_request.request_id == request.request_id
    assert np.array_equal(decoded_request.inputs["observation.state"], request.inputs["observation.state"])
    decoded_result = result_from_message(result_to_message(result))
    assert decoded_result.actual_chunk_size == result.actual_chunk_size
    assert np.array_equal(decoded_result.action, result.action)


def test_decode_failure_uses_cloud_identity_and_unknown_operation(tmp_path):
    identity = _identity(tmp_path / "bundle")
    message = request_to_message(
        DistributedRequest(
            operation=Operation.RESET,
            pipeline_id=identity.pipeline_id,
            request_id="request",
            session_id="session",
            session_generation=1,
            deployment_fingerprint=identity.deployment_fingerprint,
        )
    )
    message.operation = 255
    message.pipeline_id = "unknown"

    result = decode_failure_result(
        message,
        ValueError("invalid operation"),
        identity.deployment_fingerprint,
        identity.pipeline_id,
    )

    assert result.operation is Operation.UNKNOWN
    assert result.pipeline_id == identity.pipeline_id
    assert result.error is not None
    assert result.error.code == "decode_failed"
