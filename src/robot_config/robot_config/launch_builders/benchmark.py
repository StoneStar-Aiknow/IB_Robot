"""Benchmark launch builder for production evaluation.

The builder remains provider-agnostic: it never imports LIBERO, robosuite,
MuJoCo, ``benchmark_runtime``, or ``benchmark_libero`` Python modules.

For production evaluation it owns the launch-time run identity boundary. It
allocates one safe, non-overwriting run ID, writes an ignored materialized
robot YAML containing that ID, and passes the materialized path to every
runtime role. Native and canonical artifact owners therefore read the same
final identity from the same SSOT snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.benchmark_endpoints import (
    BenchmarkEndpointError,
    resolve_benchmark_cloud_deployment,
    resolve_benchmark_endpoints,
    resolve_benchmark_inference,
    resolve_benchmark_parallel,
    resolve_benchmark_run_timeout,
)
from robot_config.benchmark_observation_transport import materialize_benchmark_observation_transport
from robot_config.inference_config import (
    ControlModeInferenceConfig,
    InferenceConfigError,
    parse_inference_config,
    scheduler_enabled_from_raw_config,
)
from robot_config.logger_utils import get_colored_logger
from robot_config.observation_transport import (
    observation_transport_fingerprint,
    robot_config_declares_rtp_observation,
    validate_robot_config_observation_transports,
)
from robot_config.utils import parse_bool, prepare_lerobot_env

logger = get_colored_logger("robot_config.benchmark")


class BenchmarkLaunchError(RuntimeError):
    """Raised when the benchmark nodes cannot be built from SSOT."""


def _benchmark_process_exit_handler(role: str, coordinated: bool = False):
    """Shut down the benchmark evaluation graph when a required node exits."""

    def _on_exit(event, context):
        if context.is_shutdown:
            return None
        returncode = event.returncode
        if role.endswith(":evaluator") and returncode == 0:
            if not coordinated:
                reason = "benchmark evaluation completed"
                logger.info(reason)
                return EmitEvent(event=Shutdown(reason=reason))
            logger.info("benchmark lane evaluator completed; waiting for coordinator fan-in")
            return None
        if role == "coordinator" and returncode == 0:
            return EmitEvent(event=Shutdown(reason="benchmark evaluation completed"))
        if role.endswith(":environment") and returncode == 0:
            reason = f"benchmark environment exited before evaluator completion: {role}"
            logger.error(reason)
            return EmitEvent(event=Shutdown(reason=reason))
        reason = f"benchmark required process {role!r} exited with return code {returncode}"
        logger.error(reason)
        return EmitEvent(event=Shutdown(reason=reason))

    return _on_exit


def _with_benchmark_lifecycle(node: Node, role: str, coordinated: bool = False) -> list:
    """Return one required Benchmark node and its narrowly scoped exit handler."""
    return [
        node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=node,
                on_exit=_benchmark_process_exit_handler(role, coordinated),
            )
        ),
    ]


def _require_non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkLaunchError(f"{label} must be a non-empty string in the SSOT")
    return value


_SAFE_RUN_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _require_safe_run_component(value: Any, label: str) -> str:
    component = _require_non_empty(value, label)
    if component != component.strip():
        raise BenchmarkLaunchError(f"{label} must not contain surrounding whitespace")
    if component in {".", ".."} or not _SAFE_RUN_COMPONENT.fullmatch(component):
        raise BenchmarkLaunchError(
            f"{label} must be one safe path component containing only letters, digits, '.', '_' or '-'"
        )
    return component


def _workspace_root(config_path: str) -> Path:
    resolved = Path(config_path).expanduser().resolve()
    repository = next((parent for parent in resolved.parents if (parent / ".git").exists()), None)
    if repository is not None:
        return repository
    workspace_value = os.environ.get("WORKSPACE")
    if workspace_value:
        workspace = Path(workspace_value).expanduser()
        if workspace.is_absolute() and (workspace / ".git").exists():
            return workspace.resolve()
    return resolved.parent


def _resolve_output_root(evaluation: dict[str, Any], config_path: str) -> Path:
    output = evaluation.get("output", {})
    if not isinstance(output, dict):
        raise BenchmarkLaunchError("benchmark.evaluation.output must be a mapping when present")
    root_value = output.get("root", "outputs/benchmark")
    root_text = _require_non_empty(root_value, "benchmark.evaluation.output.root")
    output_root = Path(root_text).expanduser()
    if not output_root.is_absolute():
        output_root = _workspace_root(config_path) / output_root
    return output_root.resolve()


def _runtime_config_path(config_path: str, run_id: str) -> Path:
    return _workspace_root(config_path) / "tmp" / "benchmark_run_configs" / f"{run_id}.yaml"


def _lane_config_path(config_path: str, run_id: str, lane_id: str) -> Path:
    return Path(config_path).parent / run_id / f"{lane_id}.yaml"


def _make_lane_config(
    robot_config: dict[str, Any], lane_id: str, lane_path: Path, *, pin_loopback: bool = True
) -> dict[str, Any]:
    lane_config = copy.deepcopy(robot_config)
    lane_config["_config_path"] = str(lane_path)
    endpoints = resolve_benchmark_endpoints(robot_config, lane_id=lane_id)
    contract = lane_config.get("contract")
    if isinstance(contract, dict):
        for section in ("observations", "actions"):
            entries = contract.get(section, [])
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                endpoint = entry.get("topic")
                if isinstance(endpoint, str) and endpoint.startswith("/"):
                    entry["topic"] = endpoints.scope_endpoint(endpoint)
                publish = entry.get("publish")
                if isinstance(publish, dict) and isinstance(publish.get("topic"), str):
                    topic = publish["topic"]
                    if topic.startswith("/"):
                        publish["topic"] = endpoints.scope_endpoint(topic)
        rtp_streams = [
            entry["transport"]
            for entry in contract.get("observations", [])
            if isinstance(entry, dict) and (entry.get("transport") or {}).get("mode") == "rtp"
        ]
        if rtp_streams:
            ports = [stream["endpoint"]["port"] for stream in rtp_streams]
            # Reserve a disjoint port range per lane, including sparse overrides
            # and the adjacent RTCP port. Preserve each stream's configured offset.
            stride = max(ports) - min(ports) + 2
            stride += stride % 2
            offset = int(lane_id.removeprefix("lane_")) * stride
            for stream in rtp_streams:
                # A colocated shared cloud owner binds every lane's receiver
                # endpoint on this host, so the profile-declared host (which
                # only matters for the two-machine routes) is pinned to
                # loopback. An external cloud owner keeps the declared host.
                if pin_loopback:
                    stream["endpoint"]["host"] = "127.0.0.1"
                port = stream["endpoint"]["port"] + offset
                if port > 65535:
                    raise BenchmarkLaunchError(f"RTP port allocation for {lane_id} exceeds 65535")
                stream["endpoint"]["port"] = port
                stream_id = f"{lane_id}_{stream['stream_id']}"
                if len(stream_id) > 63:
                    digest = hashlib.sha256(stream_id.encode()).hexdigest()[:16]
                    stream_id = f"{stream_id[:46]}_{digest}"
                stream["stream_id"] = stream_id

    control_mode = str(lane_config.get("default_control_mode", "model_inference"))
    inference = parse_inference_config(lane_config, control_mode)
    pipelines = lane_config["control_modes"][control_mode]["inference"]["pipelines"]
    for pipeline_id, pipeline in inference.pipelines.items():
        pipelines[pipeline_id]["transport"] = {
            key: f"{value}_{lane_id}" if key in {"node_name", "cloud_node_name"} else endpoints.scope_endpoint(value)
            for key, value in asdict(pipeline.transport).items()
            if value is not None
        }
    selection = lane_config["benchmark"]["evaluation"].get("observation_transport")
    if selection is not None:
        pipeline_id, _ = _selected_benchmark_pipeline(lane_config)
        selection["effective_fingerprint"] = observation_transport_fingerprint(
            lane_config, selection["mode"], pipeline_id
        )
        materialize_benchmark_observation_transport(lane_config)
    errors = validate_robot_config_observation_transports(lane_config)
    if errors:
        raise BenchmarkLaunchError("invalid benchmark lane transport:\n- " + "\n- ".join(errors))
    return lane_config


def _write_materialized_config(robot_config: dict[str, Any], run_id: str, destination: Path) -> None:
    snapshot = copy.deepcopy(robot_config)
    snapshot.pop("_config_path", None)
    snapshot.pop("_benchmark_lane_configs", None)
    benchmark = snapshot.get("benchmark")
    if not isinstance(benchmark, dict):
        raise BenchmarkLaunchError("benchmark section must be a mapping")
    evaluation = benchmark.get("evaluation")
    if not isinstance(evaluation, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping")
    evaluation["run_id"] = run_id
    encoded = yaml.safe_dump({"robot": snapshot}, sort_keys=False, allow_unicode=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise
    except OSError as exc:
        raise BenchmarkLaunchError(f"cannot materialize benchmark run config at {destination}: {exc}") from exc


def _prepare_run_identity(
    robot_config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> str:
    """Allocate one run ID and materialize it into a launch-owned YAML.

    Automatic IDs use UTC ``timestamp_suite_run-name``. If the same-second
    base is already represented by either an output directory or a runtime
    config reservation, ``_01``, ``_02``, ... are tried in order. Explicit
    IDs fail closed on either condition and are never suffixed.
    """
    benchmark = robot_config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise BenchmarkLaunchError("benchmark section must be a mapping")
    evaluation = benchmark.get("evaluation")
    if not isinstance(evaluation, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping")
    config_path = _require_non_empty(robot_config.get("_config_path"), "robot._config_path")
    output_root = _resolve_output_root(evaluation, config_path)

    explicit = evaluation.get("run_id")
    if explicit is not None:
        run_id = _require_safe_run_component(explicit, "benchmark.evaluation.run_id")
        destination = _runtime_config_path(config_path, run_id)
        if (output_root / run_id).exists() or destination.exists():
            raise BenchmarkLaunchError(f"benchmark run_id already exists and will not be overwritten: {run_id}")
        try:
            _write_materialized_config(robot_config, run_id, destination)
        except FileExistsError as exc:
            raise BenchmarkLaunchError(
                f"benchmark run_id already exists and will not be overwritten: {run_id}"
            ) from exc
    else:
        suite = _require_safe_run_component(evaluation.get("suite"), "benchmark.evaluation.suite")
        output = evaluation.get("output", {})
        if not isinstance(output, dict):
            raise BenchmarkLaunchError("benchmark.evaluation.output must be a mapping when present")
        run_name = _require_safe_run_component(
            output.get("run_name", "benchmark_evaluation"),
            "benchmark.evaluation.output.run_name",
        )
        timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        base = f"{timestamp}_{suite}_{run_name}"
        suffix = 0
        while True:
            run_id = base if suffix == 0 else f"{base}_{suffix:02d}"
            destination = _runtime_config_path(config_path, run_id)
            if (output_root / run_id).exists() or destination.exists():
                suffix += 1
                continue
            try:
                _write_materialized_config(robot_config, run_id, destination)
            except FileExistsError:
                suffix += 1
                continue
            break

    evaluation["run_id"] = run_id
    robot_config["_config_path"] = str(destination)
    return run_id


def _resolve_benchmark_endpoints(robot_config: dict) -> tuple[str, str, str, str, str]:
    """Resolve the benchmark namespace and service endpoints from the SSOT.

    Plan and finalization services are derived from the benchmark namespace;
    there is intentionally no synthetic status topic without a real owner.
    """
    try:
        endpoints = resolve_benchmark_endpoints(robot_config)
    except BenchmarkEndpointError as exc:
        raise BenchmarkLaunchError(f"benchmark endpoint resolution failed: {exc}") from exc
    namespace = endpoints.namespace
    plan_service = f"{namespace}/get_plan"
    finalize_service = f"{namespace}/finalize"
    return (
        namespace,
        endpoints.reset_service,
        endpoints.step_service,
        plan_service,
        finalize_service,
    )


def _lane_id(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise BenchmarkLaunchError("lane index must be a non-negative int")
    return f"lane_{index:03d}"


def _resolve_lane_endpoints(robot_config: dict, lane_id: str) -> tuple[str, str, str, str, str]:
    """Resolve all benchmark service names for one isolated evaluation lane."""
    try:
        endpoints = resolve_benchmark_endpoints(robot_config, lane_id=lane_id)
    except BenchmarkEndpointError as exc:
        raise BenchmarkLaunchError(f"benchmark endpoint resolution failed: {exc}") from exc
    namespace = endpoints.namespace
    return (
        namespace,
        endpoints.reset_service,
        endpoints.step_service,
        f"{namespace}/get_plan",
        f"{namespace}/finalize",
    )


def _resolve_evaluator_endpoints(robot_config: dict, *, lane_id: str = "") -> dict[str, str]:
    """Resolve the evaluator's target endpoints from the SSOT.

    The evaluator orchestrates:
      ResetBenchmark -> PreparePolicyEpisode -> RunPolicy -> StepBenchmark

    PreparePolicyEpisode and RunPolicy are served by the **action_dispatcher**
    node, NOT the inference_service. The inference_service serves DispatchInfer
    and health; the action_dispatcher wraps them behind its own episode gate.

    Endpoint derivation:
      - action_dispatcher node name is "action_dispatcher" (set by the
        execution launch builder). The node creates ``~/prepare_episode``
        and ``~/run_policy`` when executor_type=benchmark and
        scheduler_mode=wait_for_feedback, which expand to:
          /action_dispatcher/prepare_episode
          /action_dispatcher/run_policy
      - inference_health_topic comes from the inference pipeline transport
        config (the only endpoint that IS from the inference_service).

    Returns a dict with the evaluator's endpoint parameter names and
    resolved values.
    """
    # action_dispatcher endpoints — derived from the node name "action_dispatcher"
    # (set in execution.py generate_action_dispatcher_node). The node creates
    # ~/prepare_episode and ~/run_policy in benchmark + wait_for_feedback mode.
    # Endpoint strings are absolute and therefore bypass the evaluator's own
    # benchmark namespace. The dispatcher launch namespace is lane-specific.
    namespace = resolve_benchmark_endpoints(robot_config, lane_id=lane_id).namespace if lane_id else ""
    action_dispatcher_name = f"{namespace}/action_dispatcher"
    prepare_policy_episode = f"{action_dispatcher_name}/prepare_episode"
    run_policy_action = f"{action_dispatcher_name}/run_policy"

    # inference_health_topic — the only endpoint from the inference_service.
    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    try:
        inference = parse_inference_config(robot_config, control_mode)
    except InferenceConfigError:
        inference_health = ""
    else:
        if inference.enabled and inference.pipelines:
            _, pipeline = _selected_benchmark_pipeline(robot_config, inference=inference)
            inference_health = resolve_benchmark_endpoints(robot_config, lane_id=lane_id or None).scope_endpoint(
                pipeline.transport.health_topic
            )
        else:
            inference_health = ""

    return {
        "prepare_policy_episode_service": prepare_policy_episode,
        "run_policy_action_server": run_policy_action,
        "inference_health_topic": inference_health,
    }


def _selected_benchmark_pipeline(robot_config: dict, *, inference: ControlModeInferenceConfig | None = None):
    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    if inference is None:
        inference = parse_inference_config(robot_config, control_mode)
    if not inference.enabled or not inference.pipelines:
        raise BenchmarkLaunchError("benchmark RTP requires an enabled inference pipeline")
    executor = robot_config.get("control_modes", {}).get(control_mode, {}).get("executor", {}) or {}
    selected = executor.get("inference_pipeline")
    if selected is None:
        if len(inference.pipelines) != 1:
            raise BenchmarkLaunchError("benchmark executor must select one inference pipeline")
        selected = next(iter(inference.pipelines))
    if not isinstance(selected, str) or not selected:
        raise BenchmarkLaunchError("benchmark executor.inference_pipeline must select one pipeline")
    try:
        return selected, inference.pipelines[selected]
    except KeyError as exc:
        raise BenchmarkLaunchError(f"benchmark executor selects unknown pipeline {selected!r}") from exc


def _build_environment_node(
    robot_config: dict,
    benchmark: dict,
    reset_service: str,
    step_service: str,
    plan_service: str,
    finalize_service: str,
    lane_id: str = "",
    namespace: str = "",
    lane_index: int = 0,
    lane_count: int = 1,
    run_id: str = "",
) -> Node:
    """Build the production benchmark environment node."""
    return Node(
        package="benchmark_runtime",
        executable="benchmark_environment_node",
        name="benchmark_environment",
        namespace=namespace,
        parameters=[
            {
                "robot_config_path": str(robot_config["_config_path"]),
                "benchmark_type": _require_non_empty(benchmark.get("type"), "benchmark.type"),
                "adapter_name": _require_non_empty(benchmark.get("adapter"), "benchmark.adapter"),
                "reset_service": reset_service,
                "step_service": step_service,
                "plan_service": plan_service,
                "finalize_service": finalize_service,
                "lane_id": lane_id,
                "lane_index": lane_index,
                "lane_count": lane_count,
                "run_id": run_id,
            }
        ],
        output="screen",
    )


def _build_evaluator_node(
    robot_config: dict,
    benchmark: dict,
    endpoint_params: dict[str, str] | None = None,
    lane_id: str = "",
    namespace: str = "",
) -> Node:
    """Build the production benchmark evaluator node."""
    evaluation_section = benchmark.get("evaluation", {})
    if not isinstance(evaluation_section, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping when present")
    evaluation_enabled = bool(evaluation_section.get("enabled", False))

    params: dict[str, Any] = {
        "robot_config_path": str(robot_config["_config_path"]),
        "evaluation_enabled": evaluation_enabled,
        "lane_id": lane_id,
    }
    # Endpoint and run identity parameters are supplied by the caller.
    if endpoint_params:
        params.update(endpoint_params)

    return Node(
        package="benchmark_runtime",
        executable="benchmark_evaluator_node",
        name="benchmark_evaluator",
        namespace=namespace,
        parameters=[params],
        output="screen",
    )


def generate_benchmark_nodes(robot_config: dict, *, use_sim: object = False) -> list:
    """Build production benchmark nodes from the robot configuration.

    The builder references benchmark packages by ROS package/executable name
    and keeps provider imports outside ``robot_config``. When evaluation is
    enabled it launches both the environment and evaluator; otherwise it
    launches only the environment.

    evaluation.inference settings affect only parallel-enabled shared hosts,
    including single-lane shared hosts; serial mode still validates them.
    """

    _require_non_empty(robot_config.get("_config_path"), "robot._config_path")

    benchmark = robot_config.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise BenchmarkLaunchError("benchmark section must be a mapping")
    _require_non_empty(benchmark.get("type"), "benchmark.type")
    _require_non_empty(benchmark.get("adapter"), "benchmark.adapter")
    evaluation_section = benchmark.get("evaluation", {})
    if not isinstance(evaluation_section, dict):
        raise BenchmarkLaunchError("benchmark.evaluation must be a mapping when present")
    evaluation_enabled = bool(evaluation_section.get("enabled", False))

    control_mode = str(robot_config.get("default_control_mode", "model_inference"))
    if evaluation_enabled and scheduler_enabled_from_raw_config(robot_config, control_mode):
        raise BenchmarkLaunchError("benchmark evaluation does not support inference.scheduler.enable=true")
    try:
        parallel_enabled, lane_count = resolve_benchmark_parallel(evaluation_section)
        inference_limits = resolve_benchmark_inference(evaluation_section)
        if parallel_enabled and lane_count > 1:
            run_timeout_sec = resolve_benchmark_run_timeout(evaluation_section)
    except BenchmarkEndpointError as exc:
        raise BenchmarkLaunchError(str(exc)) from exc
    if parallel_enabled:
        _, pipeline = _selected_benchmark_pipeline(robot_config)
        # Under the explicit per-camera transport model the RTP declaration
        # lives on the contract entries, not on a top-level selection mapping.
        rtp_parallel = robot_config_declares_rtp_observation(robot_config)
        distributed_parallel = pipeline.execution_mode == "distributed"
        if rtp_parallel and not distributed_parallel:
            raise BenchmarkLaunchError("parallel RTP benchmark requires distributed inference")
        external_cloud = resolve_benchmark_cloud_deployment(evaluation_section) == "external"
        if external_cloud and not distributed_parallel:
            raise BenchmarkLaunchError("parallel.cloud_deployment=external requires distributed inference")

    # This is the sole benchmark transport materialization boundary. The
    # generic launch path only resolves the runtime target; every benchmark
    # node below consumes the resulting contract transport values.
    try:
        materialize_benchmark_observation_transport(robot_config)
        transport_errors = validate_robot_config_observation_transports(robot_config)
    except (TypeError, ValueError) as exc:
        raise BenchmarkLaunchError(f"invalid benchmark observation transport: {exc}") from exc
    if transport_errors:
        raise BenchmarkLaunchError("invalid benchmark observation transport:\n- " + "\n- ".join(transport_errors))

    run_id = _prepare_run_identity(robot_config) if evaluation_enabled else ""

    if not evaluation_enabled:
        namespace, reset_service, step_service, plan_service, finalize_service = _resolve_benchmark_endpoints(
            robot_config
        )
        environment_node = _build_environment_node(
            robot_config,
            benchmark,
            reset_service=reset_service,
            step_service=step_service,
            plan_service=plan_service,
            finalize_service=finalize_service,
        )
        logger.info("benchmark evaluation disabled: launching environment node only")
        return [environment_node]

    nodes: list = []
    if not parallel_enabled:
        namespace, reset_service, step_service, plan_service, finalize_service = _resolve_benchmark_endpoints(
            robot_config
        )
        environment_node = _build_environment_node(
            robot_config,
            benchmark,
            reset_service,
            step_service,
            plan_service,
            finalize_service,
            run_id=run_id,
        )
        endpoint_params = {
            "benchmark_namespace": namespace,
            "benchmark_reset_service": reset_service,
            "benchmark_step_service": step_service,
            "benchmark_plan_service": plan_service,
            "benchmark_finalize_service": finalize_service,
            **_resolve_evaluator_endpoints(robot_config),
            "run_id": run_id,
        }
        evaluator_node = _build_evaluator_node(robot_config, benchmark, endpoint_params=endpoint_params)
        return [
            *_with_benchmark_lifecycle(environment_node, "benchmark:environment"),
            *_with_benchmark_lifecycle(evaluator_node, "benchmark:evaluator"),
        ]
    lane_configs: list[dict[str, Any]] = []
    for index in range(lane_count):
        lane_id = _lane_id(index)
        lane_path = _lane_config_path(robot_config["_config_path"], run_id, lane_id)
        lane_config = _make_lane_config(robot_config, lane_id, lane_path, pin_loopback=not external_cloud)
        _write_materialized_config(lane_config, run_id, lane_path)
        namespace, reset_service, step_service, plan_service, finalize_service = _resolve_lane_endpoints(
            robot_config, lane_id
        )
        evaluator_action_endpoints = _resolve_evaluator_endpoints(robot_config, lane_id=lane_id)
        environment_node = _build_environment_node(
            lane_config,
            benchmark,
            reset_service=reset_service,
            step_service=step_service,
            plan_service=plan_service,
            finalize_service=finalize_service,
            lane_id=lane_id,
            namespace=namespace,
            lane_index=index,
            lane_count=lane_count,
            run_id=run_id,
        )
        endpoint_params = {
            "benchmark_namespace": namespace,
            "benchmark_reset_service": reset_service,
            "benchmark_step_service": step_service,
            "benchmark_plan_service": plan_service,
            "benchmark_finalize_service": finalize_service,
            "prepare_policy_episode_service": evaluator_action_endpoints["prepare_policy_episode_service"],
            "run_policy_action_server": evaluator_action_endpoints["run_policy_action_server"],
            "inference_health_topic": evaluator_action_endpoints["inference_health_topic"],
            "run_id": run_id,
        }
        evaluator_node = _build_evaluator_node(
            lane_config,
            benchmark,
            endpoint_params=endpoint_params,
            lane_id=lane_id,
            namespace=namespace,
        )
        # Multi-lane successful exits wait for coordinator fan-in; one lane
        # still shuts down the graph when its evaluator completes.
        nodes.extend(_with_benchmark_lifecycle(environment_node, f"{lane_id}:environment", lane_count > 1))
        nodes.extend(_with_benchmark_lifecycle(evaluator_node, f"{lane_id}:evaluator", lane_count > 1))
        _, pipeline = _selected_benchmark_pipeline(lane_config)
        transport = pipeline.transport
        shared_config = {
            "pipeline_id": pipeline.pipeline_id,
            "model_path": str(pipeline.model_path),
            "deployment": pipeline.deployment,
            "request_timeout": pipeline.request_timeout,
            "runtime_options_json": json.dumps(dict(pipeline.runtime_options), sort_keys=True, separators=(",", ":")),
            "robot_config_path": str(lane_path),
            "request_topic": transport.request_topic or "",
            "result_topic": transport.result_topic or "",
            "heartbeat_topic": transport.heartbeat_topic or "",
            "video_descriptor_topic": transport.video_descriptor_topic or "",
            "video_status_topic": transport.video_status_topic or "",
        }
        if not distributed_parallel:
            shared_config.update(
                {
                    "node_name": transport.node_name,
                    "execution_mode": pipeline.execution_mode,
                    "default_task": pipeline.default_task,
                    "use_sim": parse_bool(use_sim, default=False),
                    "action_server": transport.action_server,
                    "reset_service": transport.reset_service,
                    "health_topic": transport.health_topic,
                    "action_topic": transport.action_topic,
                }
            )
        lane_configs.append(shared_config)
    # Distributed lanes (DDS or RTP delivery) share one cloud owner; only a
    # monolithic parallel profile falls back to the shared policy frontends.
    if external_cloud and distributed_parallel:
        export_path = _lane_config_path(robot_config["_config_path"], run_id, _lane_id(0)).parent / "cloud_configs.json"
        export_path.write_text(
            json.dumps({"cloud_configs": lane_configs, **inference_limits}, sort_keys=True), encoding="utf-8"
        )
        logger.info(
            "external shared cloud: start the owner on the cloud host with "
            f"'ros2 launch inference_service benchmark_shared_cloud.launch.py cloud_configs_file:={export_path}'"
        )
    else:
        shared_parameter = "cloud_configs_json" if distributed_parallel else "lane_configs_json"
        shared_executable = "benchmark_shared_cloud_node" if distributed_parallel else "benchmark_shared_policy_node"
        shared = Node(
            package="inference_service",
            executable=shared_executable,
            name="benchmark_shared_cloud" if distributed_parallel else "benchmark_shared_policy",
            env=prepare_lerobot_env(),
            parameters=[{shared_parameter: json.dumps(lane_configs), **inference_limits}],
            output="screen",
        )
        nodes.insert(0, shared)
        nodes[1:1] = _with_benchmark_lifecycle(shared, "benchmark:shared_policy", lane_count > 1)[1:]
    robot_config["_benchmark_lane_configs"] = [
        str(_lane_config_path(robot_config["_config_path"], run_id, _lane_id(i))) for i in range(lane_count)
    ]
    if lane_count > 1:
        nodes.append(
            Node(
                package="benchmark_runtime",
                executable="benchmark_coordinator_node",
                name="benchmark_coordinator",
                parameters=[
                    {
                        "robot_config_path": str(robot_config["_config_path"]),
                        "run_id": run_id,
                        "lane_count": lane_count,
                        "wait_timeout_sec": run_timeout_sec,
                    }
                ],
                output="screen",
            )
        )
        coordinator = nodes[-1]
        nodes.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=coordinator,
                    on_exit=_benchmark_process_exit_handler("coordinator"),
                )
            )
        )
    logger.info(f"benchmark evaluation enabled: launching {lane_count} lane(s)")
    return nodes
