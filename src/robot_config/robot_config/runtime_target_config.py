"""SSOT expansion of automatically managed runtime scheduling participants."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from inference_manifest.runtime_identity import (
    is_canonical_device,
    physical_device,
    runtime_composition_fingerprint,
    validate_runtime_device_visibility,
)
from inference_manifest.runtime_target import (
    DEFAULT_ADMISSION_SAFETY_MARGIN_MS,
    DEFAULT_RUNTIME_ENDPOINT,
    RUNTIME_LIMIT_DEFAULTS,
    RuntimeExecutionTarget,
    identifier,
    validate_profile_observation_inputs,
    validate_runtime_endpoint,
    validate_runtime_group_budget,
)
from inference_manifest.runtime_target import validate_runtime_limits as _validate_runtime_limits

from .runtime_scheduling import resource_id_for_device, scheduling_available

_LIMIT_FIELDS = set(RUNTIME_LIMIT_DEFAULTS)

_MEMBER_FIELDS = {
    "hardware_resource_id",
    "default_priority",
    "request_timeout_ms",
    *_LIMIT_FIELDS,
    "profile_path",
    "profile_observation_inputs",
    "runtime_version",
}
_RESOURCE_FIELDS = {
    "physical_device",
    "mode",
    "verified_concurrency",
    "hardware_profile_fingerprint",
    "managed_coverage",
}


def runtime_scheduler_block(robot_config):
    """Read the single robot-level policy without changing control modes."""
    from .inference_config import _runtime_scheduler_block

    block = robot_config.get("runtime_scheduler", {})
    if not scheduling_available(robot_config):
        return {}
    _runtime_scheduler_block(robot_config)
    validate_runtime_fields(block, active=scheduling_available(robot_config))
    return block


def _fields(value, allowed, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path} has unsupported fields: {sorted(unknown)}")


def validate_runtime_fields(block, *, active=True):
    for name in ("managed_perception",):
        members = block.get(name, {})
        if not isinstance(members, Mapping):
            raise ValueError(f"runtime_scheduler.{name} must be a mapping")
        for member, value in members.items():
            _fields(value, _MEMBER_FIELDS, f"runtime_scheduler.{name}.{member}")
            validate_profile_observation_inputs(value.get("profile_observation_inputs", []))
    resources = block.get("resources", {})
    if not isinstance(resources, Mapping):
        raise ValueError("runtime_scheduler.resources must be a mapping")
    for resource, value in resources.items():
        _fields(value, _RESOURCE_FIELDS, f"runtime_scheduler.resources.{resource}")
    if not active:
        return
    from .inference_config import _require_pos_seconds

    _require_pos_seconds(
        block.get("startup_readiness_timeout", 60.0), "robot.runtime_scheduler.startup_readiness_timeout"
    )
    if block.get("priority_zero_policy", "fifo") not in {"fifo", "edf"}:
        raise ValueError("runtime_scheduler.priority_zero_policy must be fifo or edf")
    endpoint = block.get("runtime_endpoint", DEFAULT_RUNTIME_ENDPOINT)
    validate_runtime_endpoint(endpoint)
    validate_runtime_endpoint(endpoint + "/ready")
    prediction = block.get("priority_zero_deadline_admission", {})
    _fields(prediction, {"enable", "safety_margin_ms"}, "runtime_scheduler.priority_zero_deadline_admission")
    if type(prediction.get("enable", False)) is not bool:
        raise ValueError("runtime_scheduler.priority_zero_deadline_admission.enable must be boolean")
    if prediction.get("enable") and block.get("priority_zero_policy", "fifo") != "fifo":
        raise ValueError("runtime_scheduler.priority_zero_deadline_admission: runtime profile admission requires FIFO")
    margin = prediction.get("safety_margin_ms", DEFAULT_ADMISSION_SAFETY_MARGIN_MS)
    if type(margin) is not int or margin < 0:
        raise ValueError(
            "invalid deadline safety margin: runtime_scheduler.priority_zero_deadline_admission.safety_margin_ms"
        )
    for name in ("managed_perception",):
        for member, value in block.get(name, {}).items():
            validate_runtime_limits(
                {key: value[key] for key in _LIMIT_FIELDS if key in value}, f"runtime_scheduler.{name}.{member}"
            )
            timeout = value.get("request_timeout_ms", 5000)
            if type(timeout) is not int or not 1 <= timeout <= 2**31 - 1:
                raise ValueError(f"runtime_scheduler.{name}.{member}.request_timeout_ms must be a positive integer")
            priority = value.get("default_priority", 0)
            if type(priority) is not int or priority < 0:
                raise ValueError(f"runtime_scheduler.{name}.{member}.default_priority must be a non-negative integer")
    _normalize_runtime_resources(block)


def _normalize_runtime_resources(block):
    resources = {}
    devices = {}
    prediction = block.get("priority_zero_deadline_admission", {})
    for resource_id, value in block.get("resources", {}).items():
        path = f"runtime_scheduler.resources.{resource_id}"
        try:
            identifier(resource_id, "hardware resource ID")
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        device = value.get("physical_device")
        if not is_canonical_device(device):
            raise ValueError(f"{path}.physical_device must identify a canonical local device")
        if device in devices:
            raise ValueError(f"physical device cannot have multiple resource aliases: {devices[device]}, {resource_id}")
        devices[device] = resource_id
        mode = value.get("mode", "serial")
        if mode not in {"serial", "ascend_concurrent"}:
            raise ValueError(f"{path}: unsupported resource mode {mode!r}")
        verified = value.get("verified_concurrency", False)
        if type(verified) is not bool:
            raise ValueError(f"{path}.verified_concurrency must be a boolean")
        if mode == "ascend_concurrent" and (not device.startswith("local/npu:") or not verified):
            raise ValueError(f"{path}: concurrent resource requires an explicitly verified Ascend device")
        for name in ("hardware_profile_fingerprint",):
            if not isinstance(value.get(name, ""), str):
                raise ValueError(f"{path}.{name} must be a string")
            if prediction.get("enable", False) and not value.get(name, "").strip():
                raise ValueError(f"{path}.{name} must be nonempty for runtime prediction")
        if prediction.get("enable", False) and mode != "serial":
            raise ValueError(f"{path}: runtime prediction requires serial resources")
        coverage = value.get("managed_coverage", False)
        if type(coverage) is not bool:
            raise ValueError(f"{path}.managed_coverage must be a boolean")
        if prediction.get("enable", False) and not coverage:
            raise ValueError(f"{path}: runtime prediction requires explicit managed_coverage=true")
        resources[resource_id] = {
            "physical_device": device,
            "mode": mode,
            "verified_concurrency": verified,
            "hardware_profile_fingerprint": value.get("hardware_profile_fingerprint", ""),
            "predictive_admission": prediction.get("enable", False),
            "admission_safety_margin_ms": prediction.get("safety_margin_ms", DEFAULT_ADMISSION_SAFETY_MARGIN_MS),
            "managed_coverage": coverage,
        }
    return resources


def build_runtime_resources(robot_config, runtime_targets=None):
    """Combine explicit resource rules with devices discovered from managed hosts."""
    block = runtime_scheduler_block(robot_config)
    if not scheduling_available(robot_config):
        return {}
    declarations = dict(block.get("resources", {}))
    for target in runtime_targets or ():
        declarations.setdefault(target["hardware_resource_id"], {"physical_device": target["physical_device"]})
    resources = _normalize_runtime_resources({**block, "resources": declarations})
    if runtime_targets is not None:
        # Only the Global composition can attest device-wide coverage. Host
        # descriptors deliberately contain no claim about other participants.
        unmanaged_participants = tuple(_unmanaged_runtime_participants(robot_config))
        for resource_id, resource in resources.items():
            uncovered = [
                name
                for name, device in unmanaged_participants
                if device is None or device == resource["physical_device"]
            ]
            resource["managed_coverage"] = not uncovered and declarations[resource_id].get("managed_coverage", True)
            if uncovered and resource["predictive_admission"]:
                raise ValueError(f"prediction blocked by unmanaged device participants: {uncovered}")
    return resources


def validate_runtime_limits(value, path):
    """Validate bounded host retention identically for policy and perception."""
    _fields(value, _LIMIT_FIELDS, path)
    try:
        _validate_runtime_limits(value)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_runtime_target_directory(
    robot_config,
    control_mode=None,
    *,
    inference=None,
    instance_ids=None,
    perception_only=False,
    policy_only=False,
    include_policy=True,
):
    block = runtime_scheduler_block(robot_config)
    if not scheduling_available(robot_config):
        return ()
    if perception_only and policy_only:
        raise ValueError("perception_only and policy_only are mutually exclusive")
    selected = {} if policy_only else block.get("managed_perception", {})
    resources = build_runtime_resources(robot_config)
    mode = control_mode or robot_config.get("default_control_mode", "model_inference")
    predictive = bool(block.get("priority_zero_deadline_admission", {}).get("enable", False))
    from .inference_config import parse_inference_config
    from .perception_runtime_config import parse_perception_runtime_config, select_perception_services

    if inference is None and not perception_only and include_policy:
        inference = (
            parse_inference_config(robot_config, mode) if mode in robot_config.get("control_modes", {}) else None
        )
    enabled_services = () if policy_only else parse_perception_runtime_config(robot_config).enabled_services
    available = {service.instance_id for service in enabled_services}
    if missing := set(selected) - available:
        raise ValueError(
            f"runtime_scheduler.managed_perception references unavailable services {sorted(missing)}; "
            f"available: {sorted(available)}"
        )
    services = {
        service.instance_id: service
        for service in select_perception_services(enabled_services, instance_ids=instance_ids)
    }
    selected = {name: selected.get(name, {}) for name in services}
    targets = []
    generation = _digest(block)

    def target(kind, instance_id, validated, options, *, required, adapter, runtime_options, stage=None):
        from inference_manifest import ManifestError, validate_deployment_artifact_integrity

        from .inference_config import InferenceConfigError, _resolve_profile_path

        path = f"runtime_scheduler target {kind}/{instance_id}"
        # Policy parsing already validated its artifacts; all stages share that
        # immutable validated manifest within this launch composition.
        if kind != "policy":
            try:
                validate_deployment_artifact_integrity(validated)
            except ManifestError as exc:
                raise InferenceConfigError(f"{path}: {exc}") from exc
        options = dict(options)
        device = physical_device(
            validated, runtime_options, torch_device_index=runtime_options.get("torch_device_index")
        )
        resource_id = resource_id_for_device(resources, device, options.get("hardware_resource_id"))
        if resource_id not in resources:
            resources.update(
                _normalize_runtime_resources({**block, "resources": {resource_id: {"physical_device": device}}})
            )
        options["hardware_resource_id"] = resource_id
        resource = resources[resource_id]
        runtime_version = options.get("runtime_version", "")
        if not isinstance(runtime_version, str):
            raise ValueError(f"{path}.runtime_version must be a string")
        validate_runtime_device_visibility(device, f"{kind}/{instance_id}")
        if resource.get("physical_device") != device:
            raise ValueError(f"{path}: configured physical device disagrees with manifest {device}")
        mode = resource.get("mode", "serial")
        if mode not in {"serial", "ascend_concurrent"}:
            raise ValueError(f"{path}: unsupported resource mode {mode!r} in resources.{resource_id}")
        if mode == "ascend_concurrent" and (
            validated.deployment.backend != "ascend" or resource.get("verified_concurrency") is not True
        ):
            raise ValueError(f"{path}: concurrent resource requires explicitly verified Ascend deployment")
        if mode == "ascend_concurrent" and options.get("default_priority", 0) > 7:
            raise ValueError(f"{path}.default_priority must fit the Ascend range 0..7")
        contract = getattr(validated.deployment, "execution_contract", None)
        if (
            mode == "ascend_concurrent"
            and contract is not None
            and (bool(getattr(contract, "stateful", False)) or bool(getattr(contract, "state_links", ())))
        ):
            raise ValueError(
                f"{path}: ascend_concurrent does not support stateful execution contracts or state_links; "
                "use a serial resource for dataset/state-bank models"
            )
        if stage and mode != "ascend_concurrent":
            raise ValueError(f"{path}: independent mixed policy requires verified Ascend concurrency")
        if predictive and (mode != "serial" or not options.get("profile_path")):
            raise ValueError(
                f"{path}: runtime prediction requires serial resources and runtime_execution profiles for every target"
            )
        if predictive and not runtime_version.strip():
            raise ValueError(f"{path}.runtime_version must be nonempty for runtime prediction")
        if "profile_path" in options:
            profile_path = _resolve_profile_path(options["profile_path"], path)
            if profile_path is not None:
                options["profile_path"] = str(profile_path)
        model = validated.manifest.model
        suffix = f"/{stage}" if stage else ""
        result = {
            "target_id": f"{kind}/{instance_id}{suffix}",
            "endpoint": f"/inference/runtime/hosts/{kind}/{instance_id}{suffix}",
            "hardware_resource_id": resource_id,
            "physical_device": device,
            "deployment_fingerprint": validated.fingerprint,
            "composition_fingerprint": runtime_composition_fingerprint(adapter, runtime_options, stage=stage),
            "lifecycle_group": f"{kind}/{instance_id}",
            "configuration_generation": generation,
            "interface": model.interface,
            "model_type": model.model_type,
            "operation": model.operation or "policy",
            "required": required,
            "template_only": adapter == "model_runtime",
            "execution_kind": "stage" if stage else "executor",
            "resource_mode": mode,
            "hardware_profile_fingerprint": resource.get("hardware_profile_fingerprint", ""),
            "runtime_version": runtime_version,
            "predictive_admission": predictive,
            "admission_safety_margin_ms": block.get("priority_zero_deadline_admission", {}).get(
                "safety_margin_ms", DEFAULT_ADMISSION_SAFETY_MARGIN_MS
            ),
            **dict(options),
        }
        targets.append(RuntimeExecutionTarget(**result).to_dict())

    for instance_id, options in selected.items():
        service = services[instance_id]
        target(
            "perception",
            instance_id,
            service.validated_manifest,
            options,
            required=service.required,
            adapter=service.adapter_class,
            runtime_options=dict(service.runtime_options),
        )
    if not perception_only and include_policy and inference and inference.enabled:
        for pipeline in inference.pipelines.values():
            if pipeline.execution_mode == "distributed":
                # The deployment belongs to CloudBackendRuntime; the local
                # edge only transports observations and validates actions.
                continue
            if not pipeline.runtime_policy_json:
                continue
            if pipeline.hardware_resource_id not in resources:
                resources.update(
                    _normalize_runtime_resources(
                        {
                            **block,
                            "resources": {
                                pipeline.hardware_resource_id: {
                                    "physical_device": physical_device(
                                        pipeline.validated_manifest,
                                        pipeline.runtime_options,
                                        torch_device_index=pipeline.runtime_options.get("torch_device_index"),
                                    )
                                }
                            },
                        }
                    )
                )
            independent = pipeline.scheduling and pipeline.scheduling.stage_policy == "independent"
            stages = ("visual", "action") if independent else (None,)
            for stage in stages:
                priority = (
                    pipeline.scheduling.frame_base_priority if stage == "visual" else inference.inference_priority
                )
                options = {
                    **dict(pipeline.runtime_limits),
                    "hardware_resource_id": pipeline.hardware_resource_id,
                    "default_priority": priority,
                    "request_timeout_ms": math.ceil(pipeline.request_timeout * 1000),
                    "runtime_version": pipeline.runtime_version,
                }
                if pipeline.runtime_profile_path:
                    options["profile_path"] = str(pipeline.runtime_profile_path)
                if pipeline.profile_observation_inputs:
                    options["profile_observation_inputs"] = list(pipeline.profile_observation_inputs)
                target(
                    "policy",
                    pipeline.pipeline_id,
                    pipeline.validated_manifest,
                    options,
                    required=pipeline.required,
                    adapter="policy_executor_v1",
                    runtime_options=dict(pipeline.runtime_options),
                    stage=stage,
                )
    if not (perception_only or policy_only):
        build_runtime_resources(robot_config, targets)
    # Validate aliases before any processes or models are started.
    mapping = {}
    for item in targets:
        previous = mapping.setdefault(item["physical_device"], item["hardware_resource_id"])
        if previous != item["hardware_resource_id"]:
            raise ValueError("physical device cannot have multiple resource aliases")
    if (
        not perception_only
        and include_policy
        and inference is not None
        and inference.enabled
        and (targets or resources)
    ):
        from .inference_config import _parse_scheduler, _validate_global_endpoint_conflicts

        scheduler = inference.scheduler or _parse_scheduler(block)
        _validate_global_endpoint_conflicts(inference.pipelines, scheduler.global_endpoints)
    validate_runtime_endpoints(robot_config, targets, services.values(), inference)
    groups = {}
    for item in targets:
        groups.setdefault(item["lifecycle_group"], []).append(item)
    for group, members in groups.items():
        try:
            validate_runtime_group_budget(members)
        except ValueError as exc:
            raise ValueError(f"runtime_scheduler lifecycle group {group}: {exc}") from exc
    return tuple(targets)


def validate_runtime_endpoints(robot_config, targets, services=(), inference=None):
    """Reserve control and business names together before ROS entities exist."""
    from .inference_config import _active_interface_fields

    scheduler = inference.scheduler if inference is not None else None
    owners = {}

    def reserve(endpoint, owner):
        if not endpoint:
            return
        if endpoint in owners:
            raise ValueError(f"runtime endpoint conflict: {endpoint!r} used by {owners[endpoint]} and {owner}")
        owners[endpoint] = owner

    runtime_endpoint = robot_config.get("runtime_scheduler", {}).get("runtime_endpoint", DEFAULT_RUNTIME_ENDPOINT)
    reserve(runtime_endpoint, "runtime Global")
    reserve(runtime_endpoint + "/ready", "runtime Global readiness")
    if scheduler is not None:
        for name in ("readiness", "open_session", "dispatch", "close_session"):
            reserve(getattr(scheduler.global_endpoints, name), f"Global {name}")
    for target in targets:
        reserve(target["endpoint"], target["target_id"])
    for service in services:
        reserve(service.endpoint, f"service {service.instance_id}")
    if robot_config.get("voice_tts", {}).get("enabled", False):
        reserve(robot_config["voice_tts"].get("service_name", "/voice_tts/synthesize"), "TTS service")
    if inference is not None and inference.enabled:
        for pipeline in inference.pipelines.values():
            for field in _active_interface_fields(scheduled=bool(pipeline.runtime_policy_json)):
                reserve(getattr(pipeline.transport, field), f"policy {pipeline.pipeline_id}.{field}")


def _unmanaged_runtime_participants(robot_config):
    """Fail closed for model consumers whose migration is outside PR-16.

    Later business PRs replace these conservative unknown-device entries with
    their own configured deployment identities. They cannot certify coverage
    merely by sharing a physical device with a compiled target.
    """
    for name in ("voice_asr", "voice_tts", "speech_direction", "semantic_mapping"):
        block = robot_config.get(name, {})
        if (
            isinstance(block, Mapping)
            and block.get("enabled", False)
            and (name != "semantic_mapping" or not block.get("query_only", False))
        ):
            yield name, None
    for name in robot_config.get("_launch_native_model_hosts", {}):
        yield f"local/{name}", None
