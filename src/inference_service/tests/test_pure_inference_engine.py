from __future__ import annotations

import inspect
import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from std_srvs.srv import Trigger

import inference_service.pipeline_policy_node as policy_node_module
from inference_service.backends import (
    BackendCapabilities,
    BackendDescriptor,
    BackendRegistry,
    ConformanceEvidence,
    InferenceRequest,
    LifecycleBackend,
    PartialLoadRollback,
    ResourceDomainAdmissions,
    RuntimeContext,
)
from inference_service.backends._legacy import BackendResult
from inference_service.core.pure_inference_engine import PureInferenceEngine
from inference_service.distributed import DistributedProtocolError, Operation, StructuredError
from inference_service.pipeline_policy_node import (
    DeadlineExceededError,
    ObservationNotReadyError,
    PipelineBusyError,
    PipelinePolicyNode,
    RequestCanceledError,
)
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


class _FacadeBackend(LifecycleBackend):
    def __init__(self) -> None:
        super().__init__("torch", BackendCapabilities(), domains=ResourceDomainAdmissions())
        self.requests: list[InferenceRequest] = []
        self.close_calls = 0

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        self.context = context

    def _infer(self, request: InferenceRequest) -> BackendResult:
        self.requests.append(request)
        return BackendResult(
            action=np.full((2, 6), 3.0, dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=0.1,
        )

    def _close(self) -> None:
        self.close_calls += 1


def test_rad_to_lerobot_preserves_float32_without_joint_conversion():
    node = SimpleNamespace(_joint_rad_limits=[])

    converted = PipelinePolicyNode._rad_to_lerobot(node, np.array([1.0, 2.0], dtype=np.float64))

    assert converted.dtype == np.float32
    assert converted.flags.c_contiguous


def test_rad_to_lerobot_returns_float32_after_joint_conversion():
    node = SimpleNamespace(_joint_rad_limits=[(-1.0, 1.0, 200.0, -100.0)])

    converted = PipelinePolicyNode._rad_to_lerobot(node, np.array([0.0], dtype=np.float32))

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, np.array([0.0], dtype=np.float32))


def test_rad_to_lerobot_converts_history_along_joint_axis():
    node = SimpleNamespace(_joint_rad_limits=[(-1.0, 1.0, 200.0, -100.0)])

    converted = PipelinePolicyNode._rad_to_lerobot(
        node,
        np.array([[-1.0, 5.0], [1.0, 6.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(converted, np.array([[-100.0, 5.0], [100.0, 6.0]], dtype=np.float32))


def test_to_policy_inputs_converts_numpy_observations_to_contiguous_tensors():
    image = np.zeros((2, 3, 4), dtype=np.float32).transpose(1, 0, 2)

    converted = PipelinePolicyNode._to_policy_inputs(
        {"observation.images.top": image, "observation.state": np.arange(6, dtype=np.float32)}
    )

    assert isinstance(converted["observation.images.top"], torch.Tensor)
    assert converted["observation.images.top"].shape == (3, 2, 4)
    assert converted["observation.images.top"].is_contiguous()
    assert converted["observation.state"].dtype == torch.float32


def test_required_observation_rejects_missing_and_stale_hold_values():
    spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top/image_raw",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[],
    )

    value, missing = PipelinePolicyNode._sample_observation(state, 1_000_000_000)

    assert value is None
    assert missing["reason"] == "missing"

    image = np.ones((3, 4, 5), dtype=np.float32)
    state.history.extend([(1_000_000_000, 1_000_000_000, image), (1_600_000_000, 1_600_000_000, image * 2)])

    value, issue = PipelinePolicyNode._sample_observation(state, 1_500_000_000)
    np.testing.assert_array_equal(value, image)
    assert issue is None

    value, future = PipelinePolicyNode._sample_observation(state, 999_999_999)
    assert value is None
    assert future["reason"] == "newer_than_request"

    value, stale = PipelinePolicyNode._sample_observation(state, 1_500_000_001)
    assert value is None
    assert stale["reason"] == "stale"
    assert stale["tolerance_ms"] == 500.0


@pytest.mark.parametrize(
    ("policy", "asof_tol_ms", "sample_time_ns", "expected_constraint"),
    [
        ("asof", 100, 1_100_000_001, "asof"),
        ("drop", 0, 1_050_000_000, "drop"),
    ],
)
def test_required_observation_preserves_alignment_strategy(policy, asof_tol_ms, sample_time_ns, expected_constraint):
    value = object()
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy=policy,
        asof_tol_ms=asof_tol_ms,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(1_000_000_000, 1_000_000_000, value)],
    )

    sampled, issue = PipelinePolicyNode._sample_observation(state, sample_time_ns)

    assert sampled is None
    assert issue["reason"] == "stale"
    assert issue["constraint"] == expected_constraint


def test_operation_admission_fails_fast_when_pipeline_is_busy():
    lock = threading.Lock()
    lock.acquire()
    node = SimpleNamespace(
        _operation_lock=lock,
        _reset_pending=threading.Event(),
        _config=SimpleNamespace(pipeline_id="policy"),
    )

    with pytest.raises(PipelineBusyError, match="already processing"):
        PipelinePolicyNode._acquire_operation(node)

    assert lock.locked()


def test_operation_admission_rejects_requests_while_reset_is_pending():
    node = SimpleNamespace(
        _operation_lock=threading.Lock(),
        _reset_pending=threading.Event(),
        _config=SimpleNamespace(pipeline_id="policy"),
    )
    node._reset_pending.set()

    with pytest.raises(PipelineBusyError, match="waiting to reset"):
        PipelinePolicyNode._acquire_operation(node)

    assert not node._operation_lock.locked()


def test_hold_history_window_ignores_asof_tolerance():
    spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top/image_raw",
        ros_type="sensor_msgs/msg/Image",
        resample_policy="hold",
        asof_tol_ms=1500,
        max_age_ms=500,
    )
    node = SimpleNamespace(
        _obs_specs=[spec],
        _frequency=20.0,
        _n_obs_steps=1,
        _subs={},
        _state_specs=[],
        _topic_to_qos={spec.topic: {}},
        _subscription_key=lambda current_spec: current_spec.key,
        create_subscription=lambda *_args, **_kwargs: object(),
        _observation_callback=lambda *_args: None,
    )
    message_module = ModuleType("rosidl_runtime_py.utilities")
    message_module.get_message = lambda _ros_type: object

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, message_module.__name__, message_module)
        PipelinePolicyNode._setup_observation_subscriptions(node)

    assert node._subs[spec.key].history_window_ns == 550_000_000


def test_history_window_includes_model_observation_horizon():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        ros_type="sensor_msgs/msg/JointState",
        resample_policy="hold",
        asof_tol_ms=0,
        max_age_ms=500,
    )
    node = SimpleNamespace(
        _obs_specs=[spec],
        _frequency=20.0,
        _n_obs_steps=4,
        _subs={},
        _state_specs=[spec],
        _topic_to_qos={spec.topic: {}},
        _subscription_key=lambda current_spec: current_spec.key,
        create_subscription=lambda *_args, **_kwargs: object(),
        _observation_callback=lambda *_args: None,
    )
    message_module = ModuleType("rosidl_runtime_py.utilities")
    message_module.get_message = lambda _ros_type: object

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, message_module.__name__, message_module)
        PipelinePolicyNode._setup_observation_subscriptions(node)

    assert node._subs[spec.key].history_window_ns == 700_000_000


def test_sample_observations_fails_closed_when_required_input_is_missing():
    spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top/image_raw",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _state_specs=[],
        _frequency=20.0,
        _n_obs_steps=1,
        _observation_lock=threading.Lock(),
        _sample_observation=PipelinePolicyNode._sample_observation,
        _sample_observation_history=PipelinePolicyNode._sample_observation_history,
    )

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000)

    assert error.value.code == "observation_not_ready"
    assert error.value.recoverable is True
    assert error.value.details["observations"][0]["reason"] == "missing"


def _rtp_video_node(last_sent: dict[str, int], *, buffer_entry: tuple[int, int, object] | None):
    """Fake node whose local buffer and sender-side record can disagree.

    ``buffer_entry`` seeds the subscription buffer (what the device received)
    while ``last_sent`` feeds the manager query (what left on the wire), so
    tests can pin the freshness gate to the wire-side view.
    """
    spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top/image_raw",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[buffer_entry] if buffer_entry is not None else [],
    )
    return SimpleNamespace(
        _subs={spec.key: state},
        _state_specs=[],
        _frequency=20.0,
        _n_obs_steps=1,
        _observation_lock=threading.Lock(),
        _sample_observation=PipelinePolicyNode._sample_observation,
        _sample_observation_history=PipelinePolicyNode._sample_observation_history,
        _video_stream_manager=SimpleNamespace(latest_sent_capture_ns=lambda key: last_sent.get(key, 0)),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
    )


def test_rtp_video_gate_fails_closed_when_nothing_left_the_device():
    # The local buffer holds a frame that is perfectly fresh for the buffer
    # gate, but the sender never put anything on the wire: the compute side
    # has no frame at all, so the decision must fail closed.
    node = _rtp_video_node({}, buffer_entry=(1_000_000_000, 1_000_000_000, object()))

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000, rtp_video_keys={"observation.images.top"})

    issue = error.value.details["observations"][0]
    assert issue["key"] == "observation.images.top"
    assert issue["reason"] == "video_not_sent"


def test_rtp_video_gate_reports_stale_wire_record_despite_fresh_buffer():
    # Fresh buffer entry, but the last frame that actually left the device is
    # older than max_age (network outage / encode backlog): the gate must
    # read the wire-side age, not the buffer-side age.
    node = _rtp_video_node(
        {"observation.images.top": 100_000_000},
        buffer_entry=(1_000_000_000, 1_000_000_000, object()),
    )

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000, rtp_video_keys={"observation.images.top"})

    issue = error.value.details["observations"][0]
    assert issue["reason"] == "video_send_stale"
    assert issue["age_ms"] == pytest.approx(900.0)


def test_rtp_video_gate_passes_on_recent_wire_send_and_skips_local_value():
    # A recent wire send passes even though the local buffer is empty; the
    # returned observations never carry the local video value because the
    # policy consumes decoded frames on the compute side.
    node = _rtp_video_node({"observation.images.top": 900_000_000}, buffer_entry=None)

    observations = PipelinePolicyNode._sample_observations(
        node, 1_000_000_000, rtp_video_keys={"observation.images.top"}
    )

    assert observations == {}


def test_rtp_video_gate_without_manager_fails_closed():
    # A distributed video key with no manager attached (rollover raced the
    # decision) must fail closed rather than read as forever-fresh.
    node = _rtp_video_node({}, buffer_entry=(1_000_000_000, 1_000_000_000, object()))
    del node._video_stream_manager

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000, rtp_video_keys={"observation.images.top"})

    assert error.value.details["observations"][0]["reason"] == "video_not_sent"


def test_sample_observations_reports_history_decode_failure(monkeypatch):
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        ros_type="test/State",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(1_000_000_000, 1_000_000_000, np.array([1.0], dtype=np.float32))],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _state_specs=[spec],
        _frequency=20.0,
        _n_obs_steps=2,
        _observation_lock=threading.Lock(),
        _sample_observation=PipelinePolicyNode._sample_observation,
        _sample_observation_history=PipelinePolicyNode._sample_observation_history,
        _subscription_key=lambda current_spec: current_spec.key,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
    )
    monkeypatch.setattr(policy_node_module, "decode_value", lambda _ros_type, _value, _spec: None)

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000)

    assert error.value.details["observations"][0]["reason"] == "decode_failed"


def test_sample_observations_builds_aligned_history_with_startup_padding(monkeypatch):
    state_spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        ros_type="test/State",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    image_spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top/image_raw",
        ros_type="test/Image",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=state_spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[
            (900_000_000, 900_000_000, np.array([1.0], dtype=np.float32)),
            (950_000_000, 950_000_000, np.array([2.0], dtype=np.float32)),
            (1_000_000_000, 1_000_000_000, np.array([3.0], dtype=np.float32)),
        ],
    )
    image = SimpleNamespace(
        spec=image_spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[
            (900_000_000, 900_000_000, np.full((1, 1, 1), 10.0, dtype=np.float32)),
            (1_000_000_000, 1_000_000_000, np.full((1, 1, 1), 20.0, dtype=np.float32)),
        ],
    )
    node = SimpleNamespace(
        _subs={state_spec.key: state, image_spec.key: image},
        _state_specs=[state_spec],
        _frequency=20.0,
        _n_obs_steps=4,
        _observation_lock=threading.Lock(),
        _sample_observation=PipelinePolicyNode._sample_observation,
        _sample_observation_history=PipelinePolicyNode._sample_observation_history,
        _subscription_key=lambda spec: spec.key,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
    )
    monkeypatch.setattr(policy_node_module, "decode_value", lambda _ros_type, value, _spec: value)

    observations = PipelinePolicyNode._sample_observations(node, 1_000_000_000)

    np.testing.assert_array_equal(
        observations[state_spec.key],
        np.array([[[1.0], [1.0], [2.0], [3.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        observations[image_spec.key][0, :, 0, 0, 0],
        np.array([10.0, 10.0, 10.0, 20.0], dtype=np.float32),
    )


def test_history_sampling_checks_live_age_only_for_latest_observation():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=100_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[
            (850_000_000, 850_000_000, "old"),
            (1_000_000_000, 1_000_000_000, "latest"),
        ],
    )

    values, issue = PipelinePolicyNode._sample_observation_history(
        state,
        [850_000_000, 900_000_000, 950_000_000, 1_000_000_000],
        1_050_000_000,
    )

    assert issue is None
    assert values == ["old", "old", "old", "latest"]


def test_sample_observations_merges_historical_state_sources_on_joint_axis(monkeypatch):
    first_spec = SimpleNamespace(
        key="observation.state", topic="/arm", ros_type="test/State", resample_policy="hold", asof_tol_ms=0
    )
    second_spec = SimpleNamespace(
        key="observation.state", topic="/base", ros_type="test/State", resample_policy="hold", asof_tol_ms=0
    )

    def key_for(spec):
        return f"{spec.key}_{spec.topic.replace('/', '_')}"

    first = SimpleNamespace(
        spec=first_spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(1_000_000_000, 1_000_000_000, np.array([1.0, 2.0], dtype=np.float32))],
    )
    second = SimpleNamespace(
        spec=second_spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(1_000_000_000, 1_000_000_000, np.array([3.0], dtype=np.float32))],
    )
    node = SimpleNamespace(
        _subs={key_for(first_spec): first, key_for(second_spec): second},
        _state_specs=[first_spec, second_spec],
        _frequency=20.0,
        _n_obs_steps=2,
        _observation_lock=threading.Lock(),
        _sample_observation=PipelinePolicyNode._sample_observation,
        _sample_observation_history=PipelinePolicyNode._sample_observation_history,
        _subscription_key=key_for,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
    )
    monkeypatch.setattr(policy_node_module, "decode_value", lambda _ros_type, value, _spec: value)

    observations = PipelinePolicyNode._sample_observations(node, 1_000_000_000)

    np.testing.assert_array_equal(
        observations["observation.state"],
        np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("policy", "asof_tol_ms", "expected_constraint"),
    [("asof", 10, "asof"), ("drop", 0, "drop")],
)
def test_history_sampling_preserves_alignment_constraints(policy, asof_tol_ms, expected_constraint):
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy=policy,
        asof_tol_ms=asof_tol_ms,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(900_000_000, 900_000_000, "value"), (1_000_000_000, 1_000_000_000, "latest")],
    )

    values, issue = PipelinePolicyNode._sample_observation_history(
        state,
        [900_000_000, 950_000_000, 1_000_000_000],
        1_000_000_000,
    )

    assert values is None
    assert issue["constraint"] == expected_constraint


def test_history_sampling_rejects_stale_latest_observation():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=100_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(1_000_000_000, 1_000_000_000, "latest")],
    )

    values, issue = PipelinePolicyNode._sample_observation_history(
        state,
        [950_000_000, 1_000_000_000],
        1_100_000_001,
    )

    assert values is None
    assert issue["constraint"] == "max_age"


def test_clear_observation_buffers_accepts_older_episode_timestamps():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(2_000_000_000, 2_000_000_000, np.ones(6, dtype=np.float32))],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _observation_lock=threading.Lock(),
        _observation_epoch=0,
        _observation_reset_cutoff_ns=0,
    )

    PipelinePolicyNode._clear_observation_buffers(node)
    accepted = PipelinePolicyNode._store_observation(
        node,
        spec.key,
        1,
        1_000_000_000,
        np.zeros(6, dtype=np.float32),
    )

    assert accepted is True
    assert state.history[0][0] == 1_000_000_000
    np.testing.assert_array_equal(state.history[0][2], np.zeros(6, dtype=np.float32))


def test_observation_from_previous_epoch_is_discarded_after_reset():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _observation_lock=threading.Lock(),
        _observation_epoch=0,
        _observation_reset_cutoff_ns=0,
    )
    old_epoch = node._observation_epoch

    PipelinePolicyNode._clear_observation_buffers(node)
    accepted = PipelinePolicyNode._store_observation(
        node,
        spec.key,
        old_epoch,
        2_000_000_000,
        np.ones(6, dtype=np.float32),
    )

    assert accepted is False
    assert state.history == []


def test_observation_callback_rejects_header_timestamp_before_reset_cutoff():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        stamp_src="header",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _observation_lock=threading.Lock(),
        _observation_epoch=1,
        _observation_reset_cutoff_ns=2_000_000_000,
        _state_specs=[spec],
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=3_000_000_000)),
        _subscription_key=lambda current_spec: current_spec.key,
        _store_observation=lambda *args, **kwargs: PipelinePolicyNode._store_observation(node, *args, **kwargs),
    )
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))

    PipelinePolicyNode._observation_callback(node, message, spec)

    assert state.history == []


def test_observation_callback_accepts_new_episode_after_ros_time_rewind():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        stamp_src="header",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=550_000_000,
        history=[],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _observation_lock=threading.Lock(),
        _observation_epoch=1,
        _observation_reset_cutoff_ns=10_000_000_000,
        _state_specs=[spec],
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_100_000_000)),
        _subscription_key=lambda current_spec: current_spec.key,
        _store_observation=lambda *args, **kwargs: PipelinePolicyNode._store_observation(node, *args, **kwargs),
    )
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))

    PipelinePolicyNode._observation_callback(node, message, spec)

    assert node._observation_reset_cutoff_ns == 0
    assert state.history == [(1_000_000_000, 1_100_000_000, message)]


def test_observation_callback_started_before_reset_cannot_write_new_epoch():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        stamp_src="header",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=550_000_000,
        history=[],
    )
    callback_started = threading.Event()
    callback_release = threading.Event()

    def now():
        callback_started.set()
        assert callback_release.wait(timeout=2)
        return SimpleNamespace(nanoseconds=1_100_000_000)

    node = SimpleNamespace(
        _subs={spec.key: state},
        _observation_lock=threading.Lock(),
        _observation_epoch=0,
        _observation_reset_cutoff_ns=0,
        _state_specs=[spec],
        get_clock=lambda: SimpleNamespace(now=now),
        _subscription_key=lambda current_spec: current_spec.key,
    )
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))
    callback = threading.Thread(target=PipelinePolicyNode._observation_callback, args=(node, message, spec))
    callback.start()
    assert callback_started.wait(timeout=2)

    PipelinePolicyNode._clear_observation_buffers(node, reset_time_ns=2_000_000_000)
    callback_release.set()
    callback.join(timeout=2)

    assert not callback.is_alive()
    assert node._observation_epoch == 1
    assert node._observation_reset_cutoff_ns == 2_000_000_000
    assert state.history == []


def test_far_future_observation_is_rejected_without_pruning_valid_history():
    spec = SimpleNamespace(
        key="observation.state",
        topic="/joint_states",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    current = np.zeros(6, dtype=np.float32)
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[(1_000_000_000, 1_000_000_000, current)],
    )
    node = SimpleNamespace(
        _subs={spec.key: state},
        _observation_lock=threading.Lock(),
        _observation_epoch=0,
        _observation_reset_cutoff_ns=0,
    )

    accepted = PipelinePolicyNode._store_observation(
        node,
        spec.key,
        0,
        10_000_000_000,
        np.ones(6, dtype=np.float32),
        receive_time_ns=1_100_000_000,
    )

    assert accepted is False
    assert len(state.history) == 1
    assert state.history[0][0] == 1_000_000_000
    assert state.history[0][1] == 1_000_000_000
    assert state.history[0][2] is current


def test_action_commit_rejects_cancel_requested_goal_without_publishing():
    published = []
    goal_handle = SimpleNamespace(is_cancel_requested=True, succeed=lambda: None)
    node = SimpleNamespace(
        _config=SimpleNamespace(execution_mode="monolithic"),
        _goal_state_lock=threading.Lock(),
        _cancel_requested_goals=set(),
        _completed_goals=set(),
        _action_pub=SimpleNamespace(publish=published.append),
        _lerobot_to_rad=lambda action: action,
        _fail_distributed_after_late_cancel=lambda _request_id: None,
    )

    with pytest.raises(RequestCanceledError, match="canceled before action publication"):
        PipelinePolicyNode._commit_action(node, goal_handle, "request", np.zeros((2, 6), dtype=np.float32))

    assert published == []


def test_action_commit_rejects_expired_deadline_without_publishing():
    published = []
    goal_handle = SimpleNamespace(is_cancel_requested=False, succeed=lambda: None)
    node = SimpleNamespace(
        _config=SimpleNamespace(execution_mode="monolithic"),
        _goal_state_lock=threading.Lock(),
        _cancel_requested_goals=set(),
        _completed_goals=set(),
        _action_pub=SimpleNamespace(publish=published.append),
        _lerobot_to_rad=lambda action: action,
    )

    with pytest.raises(DeadlineExceededError, match="expired before action publication"):
        PipelinePolicyNode._commit_action(
            node,
            goal_handle,
            "request",
            np.zeros((2, 6), dtype=np.float32),
            datetime.now(timezone.utc) - timedelta(seconds=1),
        )

    assert published == []


def test_distributed_backend_unavailable_result_fails_pending_operation(monkeypatch):
    pending = policy_node_module._PendingOperation(threading.Event(), Operation.INFER)
    remote_error = StructuredError(
        code="remote_backend_unavailable",
        message="cloud backend left READY",
        stage="readiness",
    )
    result = SimpleNamespace(request_id="request", error=None)
    update = SimpleNamespace(error=remote_error, canceled_request_id="", invalidated_request_ids=())
    node = SimpleNamespace(
        _config=SimpleNamespace(pipeline_id="policy"),
        _pending_lock=threading.RLock(),
        _pending={"request": pending},
        _require_edge_session=lambda: SimpleNamespace(accept_result=lambda _result: update),
        _complete_invalidated=lambda *_args: None,
        get_logger=lambda: SimpleNamespace(warning=lambda *_args: None),
    )
    monkeypatch.setattr(policy_node_module, "result_from_message", lambda _message: result)

    PipelinePolicyNode._distributed_result_callback(
        node,
        SimpleNamespace(
            request_id="request",
            pipeline_id="policy",
            session_id="",
            session_generation=0,
            deployment_fingerprint="",
            operation=int(Operation.INFER),
        ),
    )

    assert pending.event.is_set()
    assert pending.result is None
    assert pending.error == remote_error


def test_round_trip_rechecks_deadline_after_result_event(monkeypatch):
    deadline = datetime.now(timezone.utc) - timedelta(seconds=1)

    class _Session:
        def dispatch_request(self, operation, request_id, sender, **_kwargs):
            request = SimpleNamespace(
                operation=operation,
                request_id=request_id,
                deadline=deadline,
                session_id="session",
                session_generation=1,
                deployment_fingerprint="deployment",
            )
            sender(request)
            return request

        def abandon_request(self, _request_id):
            return None

    session = _Session()
    node = SimpleNamespace(
        _pending_lock=threading.RLock(),
        _pending={},
        _request_pub=SimpleNamespace(),
        _require_edge_session=lambda: session,
    )

    def publish(_request):
        pending = node._pending["request"]
        pending.result = SimpleNamespace(success=True, error=None, backend_ready=True)
        pending.event.set()

    node._request_pub.publish = publish
    monkeypatch.setattr(policy_node_module, "request_to_message", lambda request: request)

    with pytest.raises(DistributedProtocolError) as error:
        PipelinePolicyNode._round_trip(node, Operation.INFER, "request", deadline=deadline)

    assert error.value.code == "deadline_exceeded"


def test_health_exception_is_not_masked_by_previous_error():
    published = []
    node = SimpleNamespace(
        _config=SimpleNamespace(execution_mode="monolithic", pipeline_id="policy"),
        _require_manager=lambda: SimpleNamespace(
            health=lambda _pipeline_id: (_ for _ in ()).throw(RuntimeError("health unavailable"))
        ),
        _manifest=SimpleNamespace(fingerprint="deployment"),
        _last_error="older inference error",
        _health_pub=SimpleNamespace(publish=published.append),
    )

    PipelinePolicyNode._publish_health(node)

    assert published[0].message == "health unavailable"


def test_distributed_health_augments_existing_keys_with_stream_summaries():
    published = []
    snapshot = SimpleNamespace(
        observation_key="observation.images.top",
        stream_id="top",
        mode="rtp",
        configured_encoder_backend="auto",
        selected_encoder_backend="software",
        configured_decoder_backend="software",
        selected_decoder_backend="not-local",
        endpoint=("127.0.0.1", 5004),
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        security="none/trusted-network-only",
        lifecycle_state="configured",
        ready=False,
    )
    session = SimpleNamespace(
        ready=True,
        state=SimpleNamespace(value="ready"),
        session=("session", 1),
    )
    node = SimpleNamespace(
        _config=SimpleNamespace(execution_mode="distributed", pipeline_id="policy", deployment="cpu"),
        _manifest=SimpleNamespace(
            fingerprint="deployment",
            manifest=SimpleNamespace(bundle=SimpleNamespace(name="bundle")),
            deployment=SimpleNamespace(backend="torch"),
        ),
        _remote_state="ready",
        _require_edge_session=lambda: session,
        _video_stream_manager=SimpleNamespace(diagnostic_snapshots=lambda: (snapshot,)),
        _inference_count=3,
        _last_inference_time=1.5,
        _last_error="",
        _health_pub=SimpleNamespace(publish=published.append),
        _format_video_stream_diagnostic=PipelinePolicyNode._format_video_stream_diagnostic,
    )

    PipelinePolicyNode._publish_health(node)

    values = {item.key: item.value for item in published[0].values}
    assert values["pipeline_id"] == "policy"
    assert values["backend"] == "torch"
    assert values["inference_count"] == "3"
    assert values["video_stream.count"] == "1"
    summary = json.loads(values["video_stream.observation.images.top"])
    assert summary["endpoint"] == "127.0.0.1:5004"
    assert summary["security"] == "none/trusted-network-only"
    assert summary["ready"] is False


def test_late_distributed_cancel_fails_session_before_rejecting_action():
    published = []
    failure = SimpleNamespace(invalidated_request_ids=(), error=None)
    session = SimpleNamespace(fail_calls=[])
    session.fail = lambda error: session.fail_calls.append(error) or failure
    goal_handle = SimpleNamespace(is_cancel_requested=True, succeed=lambda: None)
    node = SimpleNamespace(
        _config=SimpleNamespace(execution_mode="distributed"),
        _goal_state_lock=threading.Lock(),
        _cancel_requested_goals=set(),
        _completed_goals=set(),
        _action_pub=SimpleNamespace(publish=published.append),
        _lerobot_to_rad=lambda action: action,
        _require_edge_session=lambda: session,
        _complete_invalidated=lambda *_args: None,
        _fail_distributed_after_late_cancel=lambda request_id: PipelinePolicyNode._fail_distributed_after_late_cancel(
            node, request_id
        ),
    )

    with pytest.raises(RequestCanceledError):
        PipelinePolicyNode._commit_action(node, goal_handle, "request", np.zeros((2, 6), dtype=np.float32))

    assert published == []
    assert len(session.fail_calls) == 1
    assert session.fail_calls[0].code == "cancellation_after_remote_completion"


def test_reset_preflight_failure_preserves_observation_history():
    cleared = []
    node = SimpleNamespace(
        _reset_pending=threading.Event(),
        _operation_lock=threading.Lock(),
        _config=SimpleNamespace(execution_mode="distributed", pipeline_id="policy", request_timeout=0.1),
        _reset_distributed_pipeline=lambda _deadline: (RuntimeError("reset unsupported"), False),
        _clear_observation_buffers=lambda: cleared.append(True),
        _last_error="",
    )

    response = PipelinePolicyNode._reset_callback(node, Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert response.message == "reset unsupported"
    assert cleared == []


def _bundle(root: Path) -> Path:
    root.mkdir()
    paths = create_policy_bundle(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"chunk_size": 2, "max_action_dim": 8})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    write_manifest(root, make_manifest(root, paths))
    return root


def _registry(monkeypatch, created: list[_FacadeBackend]) -> BackendRegistry:
    module = ModuleType("tests.pure_facade_backend")

    def create_backend(_context: RuntimeContext) -> _FacadeBackend:
        backend = _FacadeBackend()
        created.append(backend)
        return backend

    module.create_backend = create_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return BackendRegistry(
        {
            "torch": BackendDescriptor(
                name="torch",
                factory="tests.pure_facade_backend:create_backend",
                supported_identities=frozenset({("policy", "act", "predict")}),
                conformance_evidence=frozenset({ConformanceEvidence("policy", "act")}),
                target_validator=lambda deployment: None,
            )
        }
    )


def test_pure_engine_uses_validated_manifest_registry_pipeline_and_clean_shutdown(monkeypatch, tmp_path):
    created: list[_FacadeBackend] = []
    engine = PureInferenceEngine(
        _bundle(tmp_path / "bundle"),
        "cpu",
        pipeline_id="smoke",
        runtime_options={"trace": True},
        registry=_registry(monkeypatch, created),
    )

    noise = np.ones((1, 2, 8), dtype=np.float32)
    result = engine(
        {"observation.state": np.zeros((1, 6), dtype=np.float32)},
        prompt="pick banana",
        control_inputs={"noise": noise},
        capture_raw_action=True,
    )

    assert result.shape == (2, 6)
    assert result.chunk_size == 2
    assert engine.chunk_size == 2
    assert result.policy_type == "act"
    assert result.backend_type == "torch"
    np.testing.assert_array_equal(result.raw_action, result.action)
    assert created[0].requests[0].prompt == "pick banana"
    np.testing.assert_array_equal(created[0].requests[0].inputs["noise"], noise)
    assert created[0].requests[0].metadata["pipeline_id"] == "smoke"
    assert created[0].context.runtime_options == {"trace": True}
    assert engine.capabilities == created[0].capabilities
    assert engine.policy_metadata.policy_type == "act"
    assert engine.nominal_chunk_size == 2
    assert engine.max_action_dimension == 8
    engine.close()
    engine.close()
    assert created[0].close_calls == 1


def test_pure_engine_contains_no_backend_identity_dispatch():
    source = inspect.getsource(PureInferenceEngine)

    for backend_name in ("ascend", "hisilicon", "rknn", "hmm"):
        assert backend_name not in source
