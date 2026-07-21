#!/usr/bin/env python3
"""Frame-gated rosbag policy evaluation utilities.

This module keeps Level 1 policy evaluation on the live ROS boundary:
recorded ROS messages are republished to contract observation topics, then the
selected unified pipeline ``DispatchInfer`` action server is called for the
same timestamp. The pure helpers are intentionally separate from the ROS
client so validation and metrics can be tested without a running graph.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dataset_tools.policy_eval_compare import (
    compare_prediction_documents,
    default_plot_dir,
    write_compare_plots,
)

NS_PER_SEC = 1_000_000_000
HISTORICAL_TIMESTAMP_POLICIES = {"contract", "header"}


class SpecView(Protocol):
    key: str
    topic: str
    ros_type: str
    is_action: bool
    resample_policy: str
    asof_tol_ms: int
    stamp_src: str


@dataclass(frozen=True)
class ContractContext:
    robot_config_path: str
    policy_path: str | None
    required_input_features: list[str]
    robot_config: dict[str, Any]
    contract: Any
    observations: list[SpecView]
    actions: list[SpecView]
    fingerprint: str


@dataclass(frozen=True)
class CalibrationStatus:
    status: str
    path: str = ""
    message: str = ""
    paths: tuple[str, ...] = ()


@dataclass
class StreamRecord:
    spec: SpecView
    timestamps_ns: list[int] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    decoded_values: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class StreamDiagnostic:
    key: str
    topic: str
    ready: bool
    source_timestamp_ns: int | None
    age_ns: int | None


@dataclass(frozen=True)
class ReplayFrame:
    frame_index: int
    sample_timestamp_ns: int
    observation_messages: dict[str, Any]
    diagnostics: list[StreamDiagnostic]
    label_action: list[float] | None = None


@dataclass(frozen=True)
class PredictionRecord:
    frame_index: int
    sample_timestamp_ns: int
    inference_id: str
    status: str
    success: bool
    message: str
    error_code: str = ""
    error_stage: str = ""
    error_recoverable: bool = False
    error_details_json: str = ""
    deployment_fingerprint: str = ""
    inference_latency_ms: float | None = None
    chunk_size: int = 0
    action: list[Any] | None = None
    label_action: list[float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def load_required_input_features(policy_path: str | Path | None) -> list[str]:
    if not policy_path:
        return []
    config_path = Path(policy_path).expanduser() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"policy config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_features = config.get("input_features", {})
    if not isinstance(input_features, dict):
        raise ValueError(f"policy config input_features must be an object: {config_path}")
    return [str(key) for key in input_features]


def filter_observations_by_input_features(
    observations: list[SpecView], required_input_features: list[str]
) -> list[SpecView]:
    if not required_input_features:
        return observations
    required = set(required_input_features)
    return [spec for spec in observations if spec.key in required]


def load_contract_context(robot_config_path: str | Path, policy_path: str | Path | None = None) -> ContractContext:
    from robot_config.contract_utils import contract_fingerprint, iter_specs
    from robot_config.loader import build_contract_from_robot_config_dict, load_robot_config_dict

    robot_config = load_robot_config_dict(robot_config_path)
    contract = build_contract_from_robot_config_dict(robot_config)
    specs = list(iter_specs(contract))
    required_input_features = load_required_input_features(policy_path)
    observations = filter_observations_by_input_features(
        [spec for spec in specs if not spec.is_action], required_input_features
    )
    actions = [spec for spec in specs if spec.is_action]
    return ContractContext(
        robot_config_path=str(Path(robot_config_path).expanduser()),
        policy_path=str(Path(policy_path).expanduser()) if policy_path else None,
        required_input_features=required_input_features,
        robot_config=robot_config,
        contract=contract,
        observations=observations,
        actions=actions,
        fingerprint=contract_fingerprint(contract),
    )


def required_topics(specs: list[SpecView]) -> set[str]:
    return {spec.topic for spec in specs}


def missing_topics(specs: list[SpecView], topic_types: dict[str, str]) -> list[str]:
    return sorted(required_topics(specs) - set(topic_types.keys()))


def validate_required_topics(specs: list[SpecView], topic_types: dict[str, str]) -> None:
    missing = missing_topics(specs, topic_types)
    if missing:
        raise ValueError("rosbag is missing required contract topics: " + ", ".join(missing))


def validate_timestamp_compatibility(timestamp_policy: str, observations: list[SpecView]) -> None:
    if timestamp_policy not in HISTORICAL_TIMESTAMP_POLICIES:
        raise ValueError(
            "unified pipeline replay supports timestamp_policy 'header' or 'contract'; "
            "bag/receive timestamps cannot be reconstructed after ROS message republication"
        )
    non_header = sorted({spec.topic for spec in observations if spec.stamp_src != "header"})
    if non_header:
        raise ValueError(
            "historical replay requires every selected observation to use contract stamp_src='header'; "
            f"receive-time streams cannot match historical DispatchInfer timestamps: {non_header}"
        )


def stream_key_counts(specs: list[SpecView]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.key] = counts.get(spec.key, 0) + 1
    return counts


def stream_key_for_spec(spec: SpecView, key_counts: dict[str, int]) -> str:
    if key_counts.get(spec.key, 0) > 1 and (spec.key == "observation.state" or spec.is_action):
        topic_suffix = spec.topic.replace("/", "_").lstrip("_")
        return f"{spec.key}_{topic_suffix}" if topic_suffix else spec.key
    return spec.key


def _stamp_from_header_ns(msg: Any) -> int | None:
    try:
        stamp = msg.header.stamp
        timestamp_ns = int(stamp.sec) * NS_PER_SEC + int(stamp.nanosec)
        return timestamp_ns or None
    except (AttributeError, TypeError, ValueError):
        return None


def select_message_timestamp_ns(msg: Any, bag_timestamp_ns: int, spec: SpecView, timestamp_policy: str) -> int:
    policy = timestamp_policy.lower()
    if policy in {"bag", "receive"}:
        return int(bag_timestamp_ns)
    if policy == "header":
        return int(_stamp_from_header_ns(msg) or bag_timestamp_ns)
    if policy == "contract":
        if spec.stamp_src == "header":
            return int(_stamp_from_header_ns(msg) or bag_timestamp_ns)
        return int(bag_timestamp_ns)
    raise ValueError(f"unsupported timestamp policy: {timestamp_policy}")


def make_eval_ticks(
    stream_timestamps: list[list[int]],
    rate_hz: float,
    *,
    frame_limit: int | None = None,
    frame_stride: int = 1,
) -> list[int]:
    valid = [np.asarray(ts, dtype=np.int64) for ts in stream_timestamps if ts]
    if not valid:
        raise ValueError("cannot select evaluation frames from empty observation streams")
    if rate_hz <= 0:
        raise ValueError(f"contract rate_hz must be positive, got {rate_hz}")
    stride = max(1, int(frame_stride))
    step_ns = int(NS_PER_SEC / float(rate_hz))
    # Start ticks at the *latest* first-message timestamp across streams so every
    # stream has at least one message at or before the first tick. Using the
    # earliest first-message timestamp (min) would place ticks before some
    # streams have any data; with header-stamp policies this happens whenever
    # sensors have different capture-to-publish latencies (e.g. USB cameras
    # whose header stamp predates the bag receive time by hundreds of ms while
    # joint states are near-instant), yielding empty samples for the early frames.
    start_ns = int(max(ts.min() for ts in valid))
    end_ns = int(max(ts.max() for ts in valid))
    ticks = list(range(start_ns, end_ns + 1, step_ns))
    selected = ticks[::stride]
    if frame_limit is not None and frame_limit >= 0:
        selected = selected[: int(frame_limit)]
    return selected


def selected_indices_for_ticks(
    *,
    policy: str,
    timestamps_ns: list[int],
    ticks_ns: list[int],
    step_ns: int,
    tol_ns: int,
) -> list[int | None]:
    if not timestamps_ns:
        return [None] * len(ticks_ns)
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    selected: list[int | None] = []
    last_idx = 0
    for tick in ticks_ns:
        while last_idx + 1 < len(ts) and ts[last_idx + 1] <= tick:
            last_idx += 1
        idx: int | None
        if policy == "drop":
            idx = last_idx if ts[last_idx] <= tick and ts[last_idx] > tick - step_ns else None
        elif policy == "asof":
            idx = last_idx if ts[last_idx] <= tick and tick - ts[last_idx] <= tol_ns else None
        else:
            idx = last_idx if ts[last_idx] <= tick else None
        selected.append(idx)
    return selected


def build_replay_frames(
    observation_streams: dict[str, StreamRecord],
    ticks_ns: list[int],
    rate_hz: float,
    *,
    action_streams: dict[str, StreamRecord] | None = None,
) -> list[ReplayFrame]:
    step_ns = int(NS_PER_SEC / float(rate_hz))
    obs_indices = {
        key: selected_indices_for_ticks(
            policy=record.spec.resample_policy,
            timestamps_ns=record.timestamps_ns,
            ticks_ns=ticks_ns,
            step_ns=step_ns,
            tol_ns=max(0, int(record.spec.asof_tol_ms)) * 1_000_000,
        )
        for key, record in observation_streams.items()
    }
    action_indices = {
        key: selected_indices_for_ticks(
            policy="hold",
            timestamps_ns=record.timestamps_ns,
            ticks_ns=ticks_ns,
            step_ns=step_ns,
            tol_ns=0,
        )
        for key, record in (action_streams or {}).items()
    }

    frames: list[ReplayFrame] = []
    for frame_index, tick in enumerate(ticks_ns):
        messages: dict[str, Any] = {}
        diagnostics: list[StreamDiagnostic] = []
        for key, record in observation_streams.items():
            idx = obs_indices[key][frame_index]
            source_ts = record.timestamps_ns[idx] if idx is not None else None
            if idx is not None:
                messages[record.spec.topic] = record.messages[idx]
            diagnostics.append(
                StreamDiagnostic(
                    key=key,
                    topic=record.spec.topic,
                    ready=idx is not None,
                    source_timestamp_ns=source_ts,
                    age_ns=(tick - source_ts) if source_ts is not None else None,
                )
            )

        label_parts: list[Any] = []
        for key, record in (action_streams or {}).items():
            idx = action_indices[key][frame_index]
            if idx is not None and record.decoded_values:
                label_parts.append(np.asarray(record.decoded_values[idx], dtype=np.float32).reshape(-1))
        label_action = np.concatenate(label_parts).astype(float).tolist() if label_parts else None
        frames.append(
            ReplayFrame(
                frame_index=frame_index,
                sample_timestamp_ns=int(tick),
                observation_messages=messages,
                diagnostics=diagnostics,
                label_action=label_action,
            )
        )
    return frames


def inspect_calibration(robot_config: dict[str, Any]) -> CalibrationStatus:
    try:
        from robot_config.utils import resolve_calibration_paths_from_config

        paths = resolve_calibration_paths_from_config(robot_config)
    except Exception as exc:
        return CalibrationStatus(status="unknown", message=f"failed to resolve calibration path: {exc}")
    if not paths:
        return CalibrationStatus(status="pass_through_risk", message="robot_config has no calibration files")

    resolved_paths = [str(Path(path).expanduser()) for path in paths]
    missing_paths = [path for path in resolved_paths if not Path(path).exists()]
    primary_path = resolved_paths[0] if resolved_paths else ""
    if not missing_paths:
        return CalibrationStatus(
            status="available",
            path=primary_path,
            message="calibration files exist",
            paths=tuple(resolved_paths),
        )
    return CalibrationStatus(
        status="missing",
        path=primary_path,
        message=f"missing calibration files: {','.join(missing_paths)}",
        paths=tuple(resolved_paths),
    )


def action_from_variants(action_chunk: Any) -> list[Any] | None:
    for variant in getattr(action_chunk, "variants", []):
        if variant.key != "action":
            continue
        if variant.type == "float_32_array":
            array_msg = variant.float_32_array
        elif variant.type == "float_64_array":
            array_msg = variant.float_64_array
        else:
            continue
        shape = [int(dim.size) for dim in array_msg.layout.dim]
        data = np.asarray(array_msg.data, dtype=np.float64)
        if shape:
            data = data.reshape(shape)
        return data.tolist()
    return None


def prediction_document(
    *,
    context: ContractContext,
    backend_name: str,
    timestamp_policy: str,
    frame_stride: int,
    policy_state_mode: str,
    calibration_status: CalibrationStatus,
    records: list[PredictionRecord],
    planned_frame_count: int | None = None,
    bag_digest: str = "",
    policy_bundle_digest: str = "",
) -> dict[str, Any]:
    successful_records = [
        record for record in records if record.success and _action_dimension(record.action) is not None
    ]
    deployment_fingerprints = {record.deployment_fingerprint for record in successful_records}
    deployment_identity_consistent = (
        bool(successful_records) and "" not in deployment_fingerprints and len(deployment_fingerprints) == 1
    )
    deployment_fingerprint = next(iter(deployment_fingerprints)) if deployment_identity_consistent else ""
    action_dimensions = {_action_dimension(record.action) for record in successful_records}
    action_dimensions.discard(None)
    action_dim = next(iter(action_dimensions)) if len(action_dimensions) == 1 else None
    action_shapes = {_action_shape(record.action) for record in successful_records}
    action_shapes.discard(None)
    canonical_shape = next(iter(action_shapes)) if len(action_shapes) == 1 else None
    action_contract_consistent = (
        bool(successful_records)
        and action_dim is not None
        and canonical_shape is not None
        and all(
            type(record.chunk_size) is int and record.chunk_size == canonical_shape[0] for record in successful_records
        )
    )
    return {
        "schema_version": 1,
        "metadata": {
            "robot_config_path": context.robot_config_path,
            "policy_path": context.policy_path,
            "required_input_features": context.required_input_features,
            "contract_name": getattr(context.contract, "name", ""),
            "contract_fingerprint": context.fingerprint,
            "bag_digest": bag_digest,
            "policy_bundle_digest": policy_bundle_digest,
            "deployment_fingerprint": deployment_fingerprint,
            "deployment_fingerprints": sorted(deployment_fingerprints - {""}),
            "deployment_identity_consistent": deployment_identity_consistent,
            "timestamp_policy": timestamp_policy,
            "selected_frame_count": len(records),
            "planned_frame_count": int(planned_frame_count if planned_frame_count is not None else len(records)),
            "successful_frame_count": len(successful_records),
            "complete": len(records) == (planned_frame_count if planned_frame_count is not None else len(records))
            and len(successful_records) == len(records)
            and deployment_identity_consistent
            and action_contract_consistent,
            "frame_stride": int(frame_stride),
            "backend": {"name": backend_name},
            "policy_state_mode": policy_state_mode,
            "replay_timestamp_mode": "live_rebased" if policy_state_mode == "per_frame_reset" else "historical",
            "replay_publisher_reliability": "reliable",
            "calibration": asdict(calibration_status),
            "action_dim": action_dim,
        },
        "frames": [asdict(record) for record in records],
    }


def write_prediction_file(path: str | Path, document: dict[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _finite_action(action: list[Any] | None) -> bool:
    return _action_dimension(action) is not None


def _action_dimension(action: list[Any] | None) -> int | None:
    shape = _action_shape(action)
    return shape[-1] if shape is not None else None


def _action_shape(action: list[Any] | None) -> tuple[int, int] | None:
    if action is None:
        return None
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        return None
    return int(array.shape[0]), int(array.shape[1])


def path_digest(path: str | Path | None) -> str:
    if not path:
        return ""
    root = Path(path).expanduser()
    if not root.exists():
        return ""
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def policy_bundle_digest(path: str | Path | None) -> str:
    if not path:
        return ""
    root = Path(path).expanduser()
    manifest_path = root / "inference_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("bundle", {}).get("digest", {}).get("value")
        if isinstance(value, str) and value:
            return value
    return path_digest(root)


def load_prediction_file(path: str | Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key}")
            value[key] = item
        return value

    def reject_non_finite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    return json.loads(
        Path(path).expanduser().read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_non_finite_constant,
    )


def load_action_mean_std(policy_path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not policy_path:
        return None
    policy_root = Path(policy_path).expanduser()
    config_path = policy_root / "policy_postprocessor.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for step in config.get("steps", []):
        if step.get("registry_name") != "unnormalizer_processor":
            continue
        step_config = step.get("config", {})
        if step_config.get("norm_map", {}).get("ACTION") != "MEAN_STD":
            return None
        state_file = step.get("state_file")
        if not isinstance(state_file, str):
            return None
        state_path = policy_root / state_file
        if not state_path.is_file():
            return None
        from safetensors import safe_open

        with safe_open(state_path, framework="np") as state:
            keys = set(state.keys())
            if not {"action.mean", "action.std"}.issubset(keys):
                return None
            return state.get_tensor("action.mean"), state.get_tensor("action.std")
    return None


def _decode_contract_value(ros_type: str, msg: Any, spec: SpecView) -> Any:
    from robot_config.contract_utils import decode_value

    return decode_value(ros_type, msg, spec)


def read_rosbag_streams(
    bag_dir: str | Path,
    context: ContractContext,
    *,
    timestamp_policy: str,
    include_actions: bool = False,
    storage_id: str = "mcap",
) -> tuple[dict[str, str], dict[str, StreamRecord], dict[str, StreamRecord]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError("rosbag replay requires ROS 2 Python packages") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(Path(bag_dir).expanduser()), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    try:
        validate_required_topics(context.observations, topic_types)
    except ValueError as exc:
        if context.policy_path is None:
            raise ValueError(
                f"{exc}. If the policy uses only a subset of contract observations, pass --policy-path so "
                "policy_eval can apply the same config.json input_features filtering as the unified pipeline node."
            ) from exc
        raise
    if include_actions:
        validate_required_topics(context.actions, topic_types)

    obs_by_topic: dict[str, list[SpecView]] = {}
    for spec in context.observations:
        obs_by_topic.setdefault(spec.topic, []).append(spec)
    action_by_topic: dict[str, list[SpecView]] = {}
    if include_actions:
        for spec in context.actions:
            action_by_topic.setdefault(spec.topic, []).append(spec)
    all_topics = set(obs_by_topic) | set(action_by_topic)

    obs_key_counts = stream_key_counts(context.observations)
    action_key_counts = stream_key_counts(context.actions)
    obs_streams = {
        stream_key_for_spec(spec, obs_key_counts): StreamRecord(spec=spec, timestamps_ns=[], messages=[])
        for spec in context.observations
    }
    action_streams = {
        stream_key_for_spec(spec, action_key_counts): StreamRecord(spec=spec, timestamps_ns=[], decoded_values=[])
        for spec in context.actions
    }

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        if topic not in all_topics:
            continue
        msg = deserialize_message(data, get_message(topic_types[topic]))
        for spec in obs_by_topic.get(topic, []):
            ts_ns = select_message_timestamp_ns(msg, int(bag_ns), spec, timestamp_policy)
            record = obs_streams[stream_key_for_spec(spec, obs_key_counts)]
            record.timestamps_ns.append(ts_ns)
            record.messages.append(msg)
        for spec in action_by_topic.get(topic, []):
            ts_ns = select_message_timestamp_ns(msg, int(bag_ns), spec, timestamp_policy)
            record = action_streams[stream_key_for_spec(spec, action_key_counts)]
            record.timestamps_ns.append(ts_ns)
            record.decoded_values.append(_decode_contract_value(spec.ros_type, msg, spec))
    return topic_types, obs_streams, action_streams


def _time_msg(timestamp_ns: int):
    from builtin_interfaces.msg import Time

    msg = Time()
    msg.sec = int(timestamp_ns // NS_PER_SEC)
    msg.nanosec = int(timestamp_ns % NS_PER_SEC)
    return msg


def rebase_message_headers(messages: dict[str, Any], timestamp_ns: int) -> dict[str, Any]:
    """Copy replay messages and assign one live timestamp after a per-frame reset."""
    rebased = {}
    for topic, msg in messages.items():
        replay_msg = copy.deepcopy(msg)
        try:
            replay_msg.header.stamp.sec = int(timestamp_ns // NS_PER_SEC)
            replay_msg.header.stamp.nanosec = int(timestamp_ns % NS_PER_SEC)
        except AttributeError as exc:
            raise ValueError(f"cannot rebase header timestamp for replay topic {topic}") from exc
        rebased[topic] = replay_msg
    return rebased


def is_observation_not_ready(error_code: str | None) -> bool:
    return error_code == "observation_not_ready"


def replay_publisher_qos(qos: dict[str, Any] | None) -> dict[str, Any]:
    replay_qos = dict(qos or {})
    replay_qos["reliability"] = "reliable"
    replay_qos["history"] = "keep_last"
    replay_qos["depth"] = 1
    return replay_qos


def _run_capture(args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rosidl_runtime_py.utilities import get_message
        from std_srvs.srv import Trigger

        from ibrobot_msgs.action import DispatchInfer
        from robot_config.contract_utils import qos_profile_from_dict
    except ImportError as exc:
        raise RuntimeError("policy evaluation capture requires ROS 2 Python packages") from exc

    context = load_contract_context(args.robot_config, args.policy_path)
    validate_timestamp_compatibility(args.timestamp_policy, context.observations)
    calibration = inspect_calibration(context.robot_config)
    _, obs_streams, action_streams = read_rosbag_streams(
        args.bag_dir,
        context,
        timestamp_policy=args.timestamp_policy,
        include_actions=args.compare_labels,
        storage_id=args.storage_id,
    )
    ticks = make_eval_ticks(
        [record.timestamps_ns for record in obs_streams.values()],
        getattr(context.contract, "rate_hz", 10.0),
        frame_limit=args.frame_limit,
        frame_stride=args.frame_stride,
    )
    frames = build_replay_frames(
        obs_streams,
        ticks,
        getattr(context.contract, "rate_hz", 10.0),
        action_streams=action_streams if args.compare_labels else None,
    )

    rclpy.init(args=None)
    node = rclpy.create_node("policy_eval_replay_client")
    publishers = {}
    for spec in context.observations:
        msg_cls = get_message(spec.ros_type)
        qos = qos_profile_from_dict(replay_publisher_qos(getattr(spec, "qos", None))) or 10
        publishers[spec.topic] = node.create_publisher(msg_cls, spec.topic, qos)
    action_client = ActionClient(node, DispatchInfer, args.action_name)
    if not action_client.wait_for_server(timeout_sec=args.server_timeout_sec):
        node.destroy_node()
        rclpy.shutdown()
        raise TimeoutError(f"DispatchInfer action server not available: {args.action_name}")
    reset_client = None
    if args.policy_state_mode == "per_frame_reset":
        reset_client = node.create_client(Trigger, args.reset_service)
        if not reset_client.wait_for_service(timeout_sec=args.server_timeout_sec):
            node.destroy_node()
            rclpy.shutdown()
            raise TimeoutError(f"policy reset service not available: {args.reset_service}")

    records: list[PredictionRecord] = []
    try:
        for frame in frames:
            replay_timestamp_ns = frame.sample_timestamp_ns
            observation_messages = frame.observation_messages
            if reset_client is not None:
                future = reset_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=args.request_timeout_sec)
                reset_result = future.result() if future.done() else None
                if reset_result is None or not reset_result.success:
                    records.append(
                        PredictionRecord(
                            frame_index=frame.frame_index,
                            sample_timestamp_ns=frame.sample_timestamp_ns,
                            inference_id="",
                            status="reset_failed",
                            success=False,
                            message="policy reset failed before frame",
                            label_action=frame.label_action,
                            diagnostics={"streams": [asdict(item) for item in frame.diagnostics]},
                        )
                    )
                    if args.failure_policy == "stop":
                        break
                    continue
                replay_timestamp_ns = node.get_clock().now().nanoseconds
                observation_messages = rebase_message_headers(frame.observation_messages, replay_timestamp_ns)

            readiness_deadline = time.monotonic() + args.observation_ready_timeout_sec
            while True:
                for topic, msg in observation_messages.items():
                    publishers[topic].publish(msg)
                end_time = time.monotonic() + args.settle_sec
                while time.monotonic() < end_time:
                    rclpy.spin_once(node, timeout_sec=min(0.02, max(0.0, end_time - time.monotonic())))

                inference_id = args.inference_id_prefix + str(uuid.uuid4())
                goal = DispatchInfer.Goal()
                goal.obs_timestamp = _time_msg(replay_timestamp_ns)
                goal.prompt = args.prompt
                goal.inference_id = inference_id
                send_future = action_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(node, send_future, timeout_sec=args.request_timeout_sec)
                goal_handle = send_future.result() if send_future.done() else None
                if goal_handle is None or not goal_handle.accepted:
                    record = PredictionRecord(
                        frame_index=frame.frame_index,
                        sample_timestamp_ns=frame.sample_timestamp_ns,
                        inference_id=inference_id,
                        status="goal_rejected_or_timeout",
                        success=False,
                        message="DispatchInfer goal was rejected or timed out before acceptance",
                        label_action=frame.label_action,
                        diagnostics={"streams": [asdict(item) for item in frame.diagnostics]},
                    )
                else:
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(node, result_future, timeout_sec=args.request_timeout_sec)
                    result_response = result_future.result() if result_future.done() else None
                    if result_response is None:
                        record = PredictionRecord(
                            frame_index=frame.frame_index,
                            sample_timestamp_ns=frame.sample_timestamp_ns,
                            inference_id=inference_id,
                            status="result_timeout",
                            success=False,
                            message="DispatchInfer result timed out",
                            label_action=frame.label_action,
                            diagnostics={"streams": [asdict(item) for item in frame.diagnostics]},
                        )
                    else:
                        result = result_response.result
                        action = action_from_variants(result.action_chunk)
                        finite_action = _finite_action(action)
                        success = bool(result.success) and finite_action
                        error = result.error
                        record = PredictionRecord(
                            frame_index=frame.frame_index,
                            sample_timestamp_ns=frame.sample_timestamp_ns,
                            inference_id=inference_id,
                            status="ok" if success else "non_finite_action" if result.success else "inference_failed",
                            success=success,
                            message=result.message,
                            error_code=str(error.code),
                            error_stage=str(error.stage),
                            error_recoverable=bool(error.recoverable),
                            error_details_json=str(error.details_json),
                            deployment_fingerprint=str(result.deployment_fingerprint),
                            inference_latency_ms=float(result.inference_latency_ms),
                            chunk_size=int(result.chunk_size),
                            action=action if finite_action else None,
                            label_action=frame.label_action,
                            diagnostics={"streams": [asdict(item) for item in frame.diagnostics]},
                        )
                if record.success or not is_observation_not_ready(record.error_code):
                    break
                if time.monotonic() >= readiness_deadline:
                    break
            records.append(record)
            if not record.success and args.failure_policy == "stop":
                break
            if (len(records) % 50) == 0 or len(records) == 1:
                total = len(frames)
                pct = 100.0 * len(records) / total if total else 0.0
                latency_str = f" latency={record.inference_latency_ms:.0f}ms" if record.inference_latency_ms else ""
                print(
                    f"  [{len(records)}/{total}] {pct:.0f}% "
                    f"frame={record.frame_index} status={record.status}{latency_str}",
                    flush=True,
                )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    document = prediction_document(
        context=context,
        backend_name=args.backend_name,
        timestamp_policy=args.timestamp_policy,
        frame_stride=args.frame_stride,
        policy_state_mode=args.policy_state_mode,
        calibration_status=calibration,
        records=records,
        planned_frame_count=len(frames),
        bag_digest=path_digest(args.bag_dir),
        policy_bundle_digest=policy_bundle_digest(args.policy_path),
    )
    write_prediction_file(args.out, document)
    print(f"Wrote {len(records)} prediction frames to {args.out}")


def _run_compare(args: argparse.Namespace) -> None:
    reference = load_prediction_file(args.reference)
    action_stats = load_action_mean_std(reference.get("metadata", {}).get("policy_path"))
    results = []
    plot_files: list[str] = []
    plot_dir = default_plot_dir(args.reference, args.out) if args.plot_dir == "" else Path(args.plot_dir).expanduser()
    for candidate_path in args.candidates:
        candidate = load_prediction_file(candidate_path)
        result = compare_prediction_documents(
            reference,
            candidate,
            join_key=args.join_key,
            allow_incompatible=args.allow_incompatible,
            action_mean=action_stats[0] if action_stats is not None else None,
            action_std=action_stats[1] if action_stats is not None else None,
        )
        result["candidate_path"] = str(candidate_path)
        if args.plots:
            result_plot_files = write_compare_plots(
                reference=reference,
                candidate=candidate,
                candidate_path=candidate_path,
                output_dir=plot_dir,
                join_key=args.join_key,
                action_step=args.plot_action_step,
            )
            result["plot_files"] = result_plot_files
            plot_files.extend(result_plot_files)
        results.append(result)
    output = {"schema_version": 1, "comparisons": results}
    if args.out:
        write_prediction_file(args.out, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if plot_files:
        print("Wrote comparison plots:")
        for path in plot_files:
            print(f"  {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frame-gated rosbag policy evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Replay rosbag observations and capture DispatchInfer output")
    capture.add_argument("--bag-dir", required=True, help="ROS bag episode directory")
    capture.add_argument("--robot-config", required=True, help="robot_config YAML path")
    capture.add_argument(
        "--policy-path",
        default="",
        help="Optional pretrained policy directory; when set, config.json input_features filter contract observations",
    )
    capture.add_argument("--out", required=True, help="Prediction JSON output path")
    capture.add_argument("--backend-name", default="policy", help="Backend label stored in prediction metadata")
    capture.add_argument("--pipeline-id", default="policy", help="Named inference pipeline used for default endpoints")
    capture.add_argument("--action-name", default="", help="DispatchInfer action name override")
    capture.add_argument("--reset-service", default="", help="Pipeline reset service override")
    capture.add_argument("--storage-id", default="mcap")
    capture.add_argument("--timestamp-policy", choices=["contract", "header", "bag", "receive"], default="header")
    capture.add_argument("--frame-limit", type=int, default=None)
    capture.add_argument("--frame-stride", type=int, default=1)
    capture.add_argument("--settle-sec", type=float, default=0.05)
    capture.add_argument("--observation-ready-timeout-sec", type=float, default=5.0)
    capture.add_argument("--server-timeout-sec", type=float, default=10.0)
    capture.add_argument("--request-timeout-sec", type=float, default=30.0)
    capture.add_argument("--failure-policy", choices=["stop", "continue"], default="stop")
    capture.add_argument("--policy-state-mode", choices=["continuous", "per_frame_reset"], default="continuous")
    capture.add_argument("--prompt", default="")
    capture.add_argument("--inference-id-prefix", default="policy_eval-")
    capture.add_argument("--compare-labels", action="store_true")
    capture.set_defaults(func=_run_capture)

    compare = subparsers.add_parser("compare", help="Compare prediction JSON files")
    compare.add_argument("--reference", required=True, help="Reference prediction JSON")
    compare.add_argument(
        "--candidate", dest="candidates", action="append", required=True, help="Candidate prediction JSON"
    )
    compare.add_argument("--out", default="", help="Optional comparison JSON output path")
    compare.add_argument("--join-key", choices=["frame_index", "sample_timestamp_ns"], default="frame_index")
    compare.add_argument("--allow-incompatible", action="store_true")
    compare.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG line charts after comparison; use --no-plots to disable",
    )
    compare.add_argument(
        "--plot-dir",
        default="",
        help="Plot output directory; defaults to <out>_plots or <reference>_plots",
    )
    compare.add_argument(
        "--plot-action-step",
        type=int,
        default=0,
        help="Action chunk step used for raw action value plots",
    )
    compare.set_defaults(func=_run_compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capture":
        if not args.action_name:
            args.action_name = f"/inference/{args.pipeline_id}/dispatch"
        if not args.reset_service:
            args.reset_service = f"/inference/{args.pipeline_id}/reset"
    args.func(args)


if __name__ == "__main__":
    main()
