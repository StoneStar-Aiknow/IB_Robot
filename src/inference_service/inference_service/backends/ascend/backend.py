"""Manifest-driven Ascend ACL backend for ACT and PI0.5 OM deployments."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding
from inference_manifest.json_utils import load_json_strict
from inference_service.backends.ascend.acl_runtime import ACL_RUNTIME_MANAGER, AclRuntimeLease, AclRuntimeManager
from inference_service.backends.ascend.graspgen_runtime import (
    DDPMSchedulerNumpy,
    GraspGenRuntimeConfig,
    build_pointnet_geometry,
    matrix_to_rt,
    rt_to_matrix,
)
from inference_service.backends.ascend.graspgen_runtime import (
    group_features as graspgen_group_features,
)
from inference_service.backends.ascend.model import AclDeviceBuffer, AclModel, numpy_dtype
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendResult,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import BoundInputs, ExecutionFrame, ExecutionPlan
from inference_service.pi05_schedule import PI05DenoisingSchedule, load_pi05_schedule

_ALLOWED_RUNTIME_OPTIONS = frozenset({"device_id", "acl_config_path", "random_seed", "curvature_log_path"})
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_TIME_SEMANTICS = frozenset({"time", "timestep", "action.time", "_time"})
_GRASPGEN_EXECUTION = (
    "generator_sa1",
    "generator_sa2",
    "generator_encoder_head",
    "discriminator_sa1",
    "discriminator_sa2",
    "discriminator_encoder_head",
    "denoiser",
    "discriminator_head",
)
_GRASP_NOISE_SEMANTICS = frozenset({"noise", "diffusion.noise", "_noise"})
_GRASP_TIME_SEMANTICS = frozenset({"time", "timestep", "diffusion.time", "_time"})
_GRASP_SAMPLE_SEMANTICS = frozenset({"sample", "diffusion.sample", "_sample"})
_GRASPGEN_GENERATOR_EMBEDDING_SEMANTICS = (
    "internal.graspgen.generator_embedding",
    "generator_embedding",
    "object_embedding",
)
_GRASPGEN_DISCRIMINATOR_EMBEDDING_SEMANTICS = (
    "internal.graspgen.discriminator_embedding",
    "discriminator_embedding",
    "object_embedding",
)


class AscendBackend(LifecycleBackend):
    """Own ACL runtime state and execute manifest-declared OM roles."""

    def __init__(
        self,
        device_id: int,
        *,
        runtime_manager: AclRuntimeManager = ACL_RUNTIME_MANAGER,
        diagnostic_capture=None,
        diagnostic_schedule: PI05DenoisingSchedule | None = None,
        diagnostic_schedule_source: str | None = None,
    ) -> None:
        if diagnostic_capture is not None and not callable(diagnostic_capture):
            raise TypeError("diagnostic_capture must be callable or None")
        super().__init__(
            "ascend",
            BackendCapabilities(
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                resource_domain=f"ascend:{device_id}",
                max_in_flight_per_resource_domain=1,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._device_id = device_id
        self._runtime_manager = runtime_manager
        self._lease: AclRuntimeLease | None = None
        self._models: dict[str, AclModel] = {}
        self._shared_buffers: list[AclDeviceBuffer] = []
        self._context: RuntimeContext | None = None
        self._options: dict[str, object] = {}
        self._policy_config: dict[str, object] = {}
        self._denoising_schedule: PI05DenoisingSchedule | None = None
        self._denoising_schedule_source: str | None = None
        self._denoising_schedule_override = False
        self._curvature_log_path: Path | None = None
        self._random: np.random.Generator | None = None
        self._diagnostic_capture = diagnostic_capture
        self._diagnostic_schedule = diagnostic_schedule
        self._diagnostic_schedule_source = diagnostic_schedule_source

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        deployment = context.deployment
        if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
            raise BackendLoadError("AscendBackend requires a compiled ascend deployment", code="invalid_deployment")
        if context.policy.policy_type not in {"act", "pi05", "graspgen"}:
            raise BackendLoadError(
                f"AscendBackend does not support policy family {context.policy.policy_type!r}",
                code="unsupported_policy_backend_pair",
            )
        options = self._validate_runtime_options(context.runtime_options)
        if int(options["device_id"]) != self._device_id:
            raise BackendLoadError(
                "Ascend backend device_id changed after construction", code="deployment_context_mismatch"
            )
        if any(deployment.artifacts[role].format != "om" for role in deployment.execution):
            raise BackendLoadError("Ascend execution artifacts must use format 'om'", code="invalid_artifact_format")
        if not (deployment.target.runtime.startswith("acl") or deployment.target.runtime.startswith("ascend")):
            raise BackendLoadError(
                f"Ascend target.runtime {deployment.target.runtime!r} is not in the ACL runtime family",
                code="incompatible_backend_target",
            )

        if context.policy.policy_type == "act":
            expected_execution: tuple[str, ...] = ("policy",)
        elif context.policy.policy_type == "pi05":
            expected_execution = ("vlm", "action_expert")
        else:
            expected_execution = _GRASPGEN_EXECUTION
        execution_match = deployment.execution == expected_execution
        if not execution_match:
            raise BackendLoadError(
                f"Ascend {context.policy.policy_type} requires execution {list(expected_execution)}, got "
                f"{list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        if any(link.producer_binding == "input" for link in deployment.device_links):
            raise BackendLoadError(
                "Ascend does not support input-sourced device links",
                code="unsupported_device_link_source",
            )

        if context.policy.policy_type != "pi05" and options["curvature_log_path"] is not None:
            raise BackendLoadError(
                "Ascend PI0.5 schedule diagnostics are only valid for a PI0.5 deployment",
                code="invalid_runtime_options",
            )

        denoising_schedule = None
        denoising_schedule_source = None
        denoising_schedule_override = False
        curvature_log_path = None
        if context.policy.policy_type == "pi05":
            denoising_schedule, denoising_schedule_source, denoising_schedule_override = self._load_denoising_schedule(
                context,
                diagnostic_schedule=self._diagnostic_schedule,
                diagnostic_schedule_source=self._diagnostic_schedule_source,
                require_velocity=options["curvature_log_path"] is not None,
            )
            if options["curvature_log_path"] is not None:
                curvature_log_path = Path(str(options["curvature_log_path"])).expanduser().resolve()
                try:
                    curvature_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with curvature_log_path.open("a", encoding="utf-8"):
                        pass
                except OSError as exc:
                    raise BackendLoadError(
                        f"Unable to open PI0.5 curvature log {curvature_log_path}: {exc}",
                        code="invalid_runtime_options",
                    ) from exc

        lease = self._runtime_manager.acquire(self._device_id, options["acl_config_path"])
        rollback.defer(lease.close)
        models: dict[str, AclModel] = {}
        shared_buffers: list[AclDeviceBuffer] = []

        def cleanup_loaded_resources() -> None:
            errors: list[Exception] = []
            for model in reversed(tuple(models.values())):
                try:
                    model.close()
                except Exception as exc:
                    errors.append(exc)
            for shared in reversed(shared_buffers):
                try:
                    lease.acl.rt.free(shared.pointer)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))

        rollback.defer(cleanup_loaded_resources)
        for role in deployment.execution:
            model = AclModel(lease, role, self._require_artifact(context, role), deployment.bindings[role])
            model.load_descriptor()
            models[role] = model

        input_overrides: dict[str, dict[int, AclDeviceBuffer]] = {role: {} for role in deployment.execution}
        output_overrides: dict[str, dict[int, AclDeviceBuffer]] = {role: {} for role in deployment.execution}
        shared_outputs: dict[tuple[str, int], AclDeviceBuffer] = {}
        for link in deployment.device_links:
            producer_binding = self._binding_for_semantic(
                deployment.bindings[link.producer].outputs, link.semantic, "producer output"
            )
            consumer_binding = self._binding_for_semantic(
                deployment.bindings[link.consumer].inputs, link.semantic, "consumer input"
            )
            if producer_binding.index is None or consumer_binding.index is None:
                raise BackendLoadError(
                    f"Ascend device link {link.semantic!r} requires explicit runtime indices",
                    code="invalid_device_link",
                )
            producer_desc = models[link.producer].output_descriptors[int(producer_binding.index)]
            consumer_desc = models[link.consumer].input_descriptors[int(consumer_binding.index)]
            if producer_desc.size != consumer_desc.size:
                raise BackendLoadError(
                    f"Ascend device link {link.semantic!r} size differs between producer "
                    f"({producer_desc.size}) and consumer ({consumer_desc.size})",
                    code="device_link_size_mismatch",
                )
            producer_key = (link.producer, int(producer_binding.index))
            shared = shared_outputs.get(producer_key)
            if shared is None:
                pointer, ret = lease.acl.rt.malloc(producer_desc.size, 0)
                if ret != 0:
                    raise RuntimeError(f"acl.rt.malloc(device link {link.semantic}) failed with ACL error code {ret}")
                shared = AclDeviceBuffer(pointer=pointer, size=producer_desc.size)
                shared_outputs[producer_key] = shared
                shared_buffers.append(shared)
            output_overrides[link.producer][int(producer_binding.index)] = shared
            input_overrides[link.consumer][int(consumer_binding.index)] = shared
        for role in deployment.execution:
            models[role].prepare_datasets(
                input_overrides=input_overrides[role],
                output_overrides=output_overrides[role],
            )

        policy_config = self._load_policy_config(context.validated_manifest.bundle_root / "config.json")
        if context.policy.policy_type == "pi05":
            self._validate_pi05_config(policy_config)
        elif context.policy.policy_type == "graspgen":
            self._validate_graspgen_config(policy_config)
        self._lease = lease
        self._models = models
        self._shared_buffers = shared_buffers
        self._context = context
        self._options = options
        self._policy_config = policy_config
        self._denoising_schedule = denoising_schedule
        self._denoising_schedule_source = denoising_schedule_source
        self._denoising_schedule_override = denoising_schedule_override
        self._curvature_log_path = curvature_log_path
        self._random = np.random.default_rng(options["random_seed"])

    def _infer(self, request: InferenceRequest) -> BackendResult:
        context = self._context
        if context is None:
            raise BackendInferenceError("AscendBackend is not fully loaded", code="runtime_not_loaded")
        plan, role_inputs = self._request_execution(request)
        started = time.perf_counter()
        policy_type = context.policy.policy_type
        graspgen_timing_ms = None
        graspgen_denoiser_steps = None
        if policy_type == "act":
            outputs = self._infer_act(plan, role_inputs)
        elif policy_type == "pi05":
            outputs = self._infer_pi05(plan, role_inputs)
        else:
            outputs, graspgen_timing_ms, graspgen_denoiser_steps = self._infer_graspgen(plan, role_inputs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if policy_type == "graspgen":
            primary = self._raw_grasp_outputs(outputs, plan)
            return BackendResult(
                action=outputs,
                actual_chunk_size=int(primary["grasp.poses"].shape[0]),
                backend_latency_ms=latency_ms,
                metadata={
                    "request_id": request.request_id,
                    "device_id": self._device_id,
                    "target_soc": context.target.soc if context.target is not None else None,
                    "deployment_name": context.deployment_name,
                    "deployment_fingerprint": context.deployment_fingerprint,
                    "graspgen_timing_ms": graspgen_timing_ms,
                    "graspgen_denoiser_steps": graspgen_denoiser_steps,
                },
            )
        action = self._raw_action(outputs, plan)
        metadata = {
            "request_id": request.request_id,
            "device_id": self._device_id,
            "target_soc": context.target.soc if context.target is not None else None,
            "deployment_name": context.deployment_name,
            "deployment_fingerprint": context.deployment_fingerprint,
        }
        if self._denoising_schedule is not None:
            metadata["denoising_schedule"] = {
                "name": self._denoising_schedule.name,
                "step_count": self._denoising_schedule.step_count,
                "source": self._denoising_schedule_source,
            }
            if self._denoising_schedule_override:
                metadata["denoising_schedule"]["override"] = True
        return BackendResult(
            action=outputs,
            actual_chunk_size=self._chunk_size(action),
            backend_latency_ms=latency_ms,
            metadata=metadata,
        )

    def _close(self) -> None:
        lease = self._lease
        models = self._models
        shared_buffers = self._shared_buffers
        self._lease = None
        self._models = {}
        self._shared_buffers = []
        self._context = None
        self._options = {}
        self._policy_config = {}
        self._denoising_schedule = None
        self._denoising_schedule_source = None
        self._denoising_schedule_override = False
        self._curvature_log_path = None
        self._random = None
        errors: list[Exception] = []
        for model in reversed(tuple(models.values())):
            try:
                model.close()
            except Exception as exc:
                errors.append(exc)
        if lease is not None:
            for shared in reversed(shared_buffers):
                try:
                    lease.acl.rt.free(shared.pointer)
                except Exception as exc:
                    errors.append(exc)
        if lease is not None:
            try:
                lease.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _infer_act(self, plan: ExecutionPlan, role_inputs: Mapping[str, BoundInputs]) -> dict[int, np.ndarray]:
        if plan.role_names != ("policy",):
            raise BackendInferenceError("Ascend ACT requires one policy role", code="invalid_request")
        return self._models["policy"].execute(role_inputs["policy"])

    def _infer_pi05(
        self, plan: ExecutionPlan, role_inputs: Mapping[str, BoundInputs]
    ) -> dict[str, dict[int, np.ndarray]]:
        if plan.role_names != ("vlm", "action_expert"):
            raise BackendInferenceError("Ascend PI0.5 requires vlm and action_expert roles", code="invalid_request")
        frame = ExecutionFrame(plan)
        try:
            frame.begin_role("vlm")
            vlm_read_indices = self._host_output_indices(plan, "vlm")
            if self._diagnostic_capture is not None:
                vlm_read_indices.update(
                    int(binding.index) for binding in plan.role("vlm").bindings.outputs if binding.index is not None
                )
                self._capture_bound_inputs("vlm", role_inputs["vlm"])
            vlm_outputs = self._models["vlm"].execute(role_inputs["vlm"], read_outputs=vlm_read_indices)
            semantic_vlm_outputs = self._semantic_outputs(plan, "vlm", vlm_outputs)
            self._capture_vlm_outputs(semantic_vlm_outputs)
            frame.finish_role("vlm", semantic_vlm_outputs)

            host_inputs = frame.begin_role("action_expert")
            action_bindings = plan.role("action_expert").bindings
            noise_binding = self._single_semantic_binding(action_bindings.inputs, _NOISE_SEMANTICS, "noise")
            time_binding = self._single_semantic_binding(action_bindings.inputs, _TIME_SEMANTICS, "time")
            action_binding = self._binding_for_semantic(action_bindings.outputs, "action", "action output")
            initial_inputs = self._bound_by_index(role_inputs["action_expert"])
            for binding in action_bindings.inputs:
                if binding.semantic in host_inputs and binding.index is not None:
                    initial_inputs[int(binding.index)] = host_inputs[binding.semantic]

            noise = initial_inputs.get(int(noise_binding.index))
            if noise is None:
                noise = self._sample_noise(noise_binding)
            self._capture("ae_in_noise", noise)
            self._capture("ae_in_past_kv", semantic_vlm_outputs.get("internal.past_kv"))
            self._capture("ae_in_prefix_pad_masks", semantic_vlm_outputs.get("internal.prefix_pad_masks"))
            outputs: dict[int, np.ndarray] = {}
            if self._denoising_schedule is None:
                steps = int(self._policy_config["num_inference_steps"])
                for step in range(steps):
                    inputs = dict(initial_inputs)
                    inputs[int(noise_binding.index)] = np.ascontiguousarray(
                        noise, dtype=numpy_dtype(noise_binding.dtype)
                    )
                    inputs[int(time_binding.index)] = np.full(
                        self._static_shape(time_binding),
                        1.0 - step / steps,
                        dtype=numpy_dtype(time_binding.dtype),
                    )
                    self._capture(f"ae_in_time_step{step:02d}", inputs[int(time_binding.index)])
                    outputs = self._models["action_expert"].execute(
                        inputs,
                        read_outputs={int(action_binding.index)},
                    )
                    noise = outputs[int(action_binding.index)]
                    self._capture(f"x_t_step{step:02d}", noise)
            else:
                timesteps = self._denoising_schedule.timesteps
                self._capture("timesteps", np.asarray(timesteps, dtype=np.float32))
                x_t = np.asarray(noise, dtype=np.float32)
                velocity_trace: list[np.ndarray] | None = [] if self._curvature_log_path is not None else None
                for step, (timestep, next_timestep) in enumerate(pairwise(timesteps)):
                    dt = next_timestep - timestep
                    inputs = dict(initial_inputs)
                    inputs[int(noise_binding.index)] = np.ascontiguousarray(x_t, dtype=numpy_dtype(noise_binding.dtype))
                    inputs[int(time_binding.index)] = np.full(
                        self._static_shape(time_binding),
                        timestep,
                        dtype=numpy_dtype(time_binding.dtype),
                    )
                    self._capture(f"ae_in_time_step{step:02d}", inputs[int(time_binding.index)])
                    self._capture(f"dt_step{step:02d}", np.asarray(dt, dtype=np.float32))
                    outputs = self._models["action_expert"].execute(
                        inputs,
                        read_outputs={int(action_binding.index)},
                    )
                    velocity = outputs[int(action_binding.index)]
                    if velocity_trace is not None:
                        velocity_trace.append(np.asarray(velocity).copy())
                    self._capture(f"velocity_step{step:02d}", velocity)
                    x_t = x_t + np.float32(dt) * np.asarray(velocity, dtype=np.float32)
                    self._capture(f"x_t_step{step:02d}", x_t)
                if velocity_trace is not None:
                    self._write_curvature_log(velocity_trace)
                outputs[int(action_binding.index)] = np.ascontiguousarray(x_t, dtype=numpy_dtype(action_binding.dtype))
            frame.finish_role("action_expert")
            return {"action_expert": outputs}
        finally:
            frame.close()

    def _infer_graspgen(
        self, plan: ExecutionPlan, role_inputs: Mapping[str, BoundInputs]
    ) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, float], int]:
        if plan.role_names != _GRASPGEN_EXECUTION:
            raise BackendInferenceError(
                f"Ascend GraspGen requires execution {list(_GRASPGEN_EXECUTION)}, got {list(plan.role_names)}",
                code="invalid_request",
            )
        total_started = time.perf_counter()
        timing_ms: dict[str, float] = {}
        config = self._require_graspgen_runtime_config()
        stage_started = time.perf_counter()
        points_payload = self._graspgen_input_points(role_inputs)
        scaled_points, object_center = self._prepare_graspgen_points(points_payload, config)
        timing_ms["input_prepare"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        geometry = build_pointnet_geometry(
            scaled_points,
            npoints=config.npoints,
            radii=config.radii,
            nsamples=config.nsamples,
        )
        timing_ms["pointnet_geometry"] = (time.perf_counter() - stage_started) * 1000.0
        generator_device_linked = self._has_graspgen_device_link(
            plan,
            producer="generator_encoder_head",
            consumer="denoiser",
            semantics=_GRASPGEN_GENERATOR_EMBEDDING_SEMANTICS,
        )
        discriminator_device_linked = self._has_graspgen_device_link(
            plan,
            producer="discriminator_encoder_head",
            consumer="discriminator_head",
            semantics=_GRASPGEN_DISCRIMINATOR_EMBEDDING_SEMANTICS,
        )
        generator_embedding = self._run_graspgen_encoder(
            "generator", geometry, timing_ms, read_embedding=not generator_device_linked
        )
        discriminator_embedding = self._run_graspgen_encoder(
            "discriminator", geometry, timing_ms, read_embedding=not discriminator_device_linked
        )
        sample = self._run_graspgen_denoiser(plan, generator_embedding, config, timing_ms)
        stage_started = time.perf_counter()
        poses = rt_to_matrix(sample, config.kappa)
        discriminator_input = matrix_to_rt(poses, config.kappa)
        timing_ms["pose_conversion"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        head_inputs = {
            self._graspgen_input_index("discriminator_head", "sample_rt"): discriminator_input,
        }
        if discriminator_embedding is not None:
            head_inputs[
                self._graspgen_input_index("discriminator_head", _GRASPGEN_DISCRIMINATOR_EMBEDDING_SEMANTICS)
            ] = discriminator_embedding
        pose_index = self._graspgen_output_index("discriminator_head", "grasp.poses")
        confidence_index = self._graspgen_output_index("discriminator_head", "grasp.confidence")
        head_outputs = self._models["discriminator_head"].execute(
            head_inputs,
            read_outputs={confidence_index},
        )
        timing_ms["discriminator_head"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        confidence = np.asarray(head_outputs[confidence_index], dtype=np.float32).reshape(-1)
        poses[:, :3, 3] += object_center
        head_result: dict[int, np.ndarray] = {
            pose_index: np.asarray(poses, dtype=np.float32),
            confidence_index: confidence,
        }
        timing_ms["output_finalize"] = (time.perf_counter() - stage_started) * 1000.0
        timing_ms["total"] = (time.perf_counter() - total_started) * 1000.0
        return {"discriminator_head": head_result}, timing_ms, config.diffusion_steps

    def _run_graspgen_encoder(
        self,
        prefix: str,
        geometry,
        timing_ms: dict[str, float],
        *,
        read_embedding: bool,
    ) -> np.ndarray | None:
        sa1_name = f"{prefix}_sa1"
        sa2_name = f"{prefix}_sa2"
        head_name = f"{prefix}_encoder_head"
        stage_started = time.perf_counter()
        sa1_outputs = self._models[sa1_name].execute(
            {self._graspgen_input_index(sa1_name, "observation.object_points"): geometry.stage1_input}
        )
        timing_ms[sa1_name] = (time.perf_counter() - stage_started) * 1000.0
        host_started = time.perf_counter()
        sa1_features = sa1_outputs[self._graspgen_output_index(sa1_name, "features")]
        stage2_input = np.concatenate(
            [geometry.stage2_relative_xyz, graspgen_group_features(sa1_features, geometry.stage2_indices)],
            axis=1,
        )
        timing_ms[f"{prefix}_encoder_host"] = (time.perf_counter() - host_started) * 1000.0
        stage_started = time.perf_counter()
        sa2_outputs = self._models[sa2_name].execute(
            {self._graspgen_input_index(sa2_name, "stage2_features"): stage2_input}
        )
        timing_ms[sa2_name] = (time.perf_counter() - stage_started) * 1000.0
        host_started = time.perf_counter()
        sa2_features = sa2_outputs[self._graspgen_output_index(sa2_name, "features")]
        global_xyz = geometry.stage2_xyz.T[None, :, None, :]
        global_features = sa2_features[:, :, None, :]
        head_input = np.ascontiguousarray(np.concatenate([global_xyz, global_features], axis=1), dtype=np.float32)
        timing_ms[f"{prefix}_encoder_host"] += (time.perf_counter() - host_started) * 1000.0
        stage_started = time.perf_counter()
        embedding_semantics = (
            _GRASPGEN_GENERATOR_EMBEDDING_SEMANTICS
            if prefix == "generator"
            else _GRASPGEN_DISCRIMINATOR_EMBEDDING_SEMANTICS
        )
        embedding_index = self._graspgen_output_index(head_name, embedding_semantics)
        head_outputs = self._models[head_name].execute(
            {self._graspgen_input_index(head_name, "global_features"): head_input},
            read_outputs={embedding_index} if read_embedding else set(),
        )
        timing_ms[head_name] = (time.perf_counter() - stage_started) * 1000.0
        return head_outputs[embedding_index] if read_embedding else None

    def _run_graspgen_denoiser(
        self,
        plan: ExecutionPlan,
        generator_embedding: np.ndarray | None,
        config: GraspGenRuntimeConfig,
        timing_ms: dict[str, float],
    ) -> np.ndarray:
        setup_started = time.perf_counter()
        denoiser_name = "denoiser"
        bindings = plan.role(denoiser_name).bindings
        sample_binding = self._single_semantic_binding(bindings.inputs, _GRASP_SAMPLE_SEMANTICS, "sample")
        self._single_semantic_binding(bindings.inputs, _GRASP_TIME_SEMANTICS, "time")
        noise_shape = self._static_shape(sample_binding)
        if self._random is None:
            raise BackendInferenceError("Ascend random generator is unavailable", code="runtime_not_loaded")
        sample = np.ascontiguousarray(
            self._random.standard_normal(noise_shape).astype(numpy_dtype(sample_binding.dtype))
        )
        position_scheduler = DDPMSchedulerNumpy(config.diffusion_steps, "scaled_linear")
        rotation_scheduler = DDPMSchedulerNumpy(config.diffusion_steps, "squaredcos_cap_v2")
        object_embedding_index = self._graspgen_input_index(denoiser_name, _GRASPGEN_GENERATOR_EMBEDDING_SEMANTICS)
        sample_index = self._graspgen_input_index(denoiser_name, "diffusion.sample")
        time_index = self._graspgen_input_index(denoiser_name, "diffusion.time")
        predicted_noise_index = self._graspgen_output_index(denoiser_name, "predicted_noise")
        batch_size = noise_shape[0]
        variance_shape = (batch_size, 3)
        timing_ms["denoiser_setup"] = (time.perf_counter() - setup_started) * 1000.0
        loop_started = time.perf_counter()
        execute_ms = 0.0
        for timestep in position_scheduler.timesteps:
            inputs = {
                sample_index: np.ascontiguousarray(sample, dtype=np.float32),
                time_index: np.asarray([timestep], dtype=np.float32),
            }
            if generator_embedding is not None:
                inputs[object_embedding_index] = generator_embedding
            execute_started = time.perf_counter()
            outputs = self._models[denoiser_name].execute(inputs)
            execute_ms += (time.perf_counter() - execute_started) * 1000.0
            predicted_noise = outputs[predicted_noise_index]
            position_noise = self._sample_diffusion_noise(variance_shape, timestep)
            rotation_noise = self._sample_diffusion_noise(variance_shape, timestep)
            position, _ = position_scheduler.step(predicted_noise[:, :3], int(timestep), sample[:, :3], position_noise)
            rotation, _ = rotation_scheduler.step(predicted_noise[:, 3:], int(timestep), sample[:, 3:], rotation_noise)
            sample = np.ascontiguousarray(np.concatenate([position, rotation], axis=-1), dtype=np.float32)
        loop_ms = (time.perf_counter() - loop_started) * 1000.0
        timing_ms["denoiser_execute"] = execute_ms
        timing_ms["denoiser_host"] = max(0.0, loop_ms - execute_ms)
        return sample

    def _sample_diffusion_noise(self, shape: tuple[int, ...], timestep: np.ndarray) -> np.ndarray | None:
        if int(timestep) <= 0:
            return None
        if self._random is None:
            raise BackendInferenceError("Ascend random generator is unavailable", code="runtime_not_loaded")
        return np.ascontiguousarray(self._random.standard_normal(shape).astype(np.float32))

    @staticmethod
    def _has_graspgen_device_link(
        plan: ExecutionPlan,
        *,
        producer: str,
        consumer: str,
        semantics: tuple[str, ...],
    ) -> bool:
        return any(
            link.producer == producer and link.consumer == consumer and link.semantic in semantics
            for link in plan.device_links
        )

    def _graspgen_input_index(self, role: str, semantic: str | tuple[str, ...]) -> int:
        bindings = self._context.deployment.bindings[role].inputs if self._context else ()
        semantics = (semantic,) if isinstance(semantic, str) else semantic
        match = [binding for binding in bindings if binding.semantic in semantics]
        if len(match) != 1 or match[0].index is None:
            raise BackendInferenceError(
                f"Ascend GraspGen role {role!r} input semantics {semantics!r} are not uniquely bound",
                code="invalid_graspgen_binding",
            )
        return int(match[0].index)

    def _graspgen_output_index(self, role: str, semantic: str | tuple[str, ...]) -> int:
        bindings = self._context.deployment.bindings[role].outputs if self._context else ()
        semantics = (semantic,) if isinstance(semantic, str) else semantic
        match = [binding for binding in bindings if binding.semantic in semantics]
        if len(match) != 1 or match[0].index is None:
            raise BackendInferenceError(
                f"Ascend GraspGen role {role!r} output semantics {semantics!r} are not uniquely bound",
                code="invalid_graspgen_binding",
            )
        return int(match[0].index)

    @staticmethod
    def _graspgen_input_points(role_inputs: Mapping[str, BoundInputs]) -> np.ndarray:
        first_role = _GRASPGEN_EXECUTION[0]
        bound = role_inputs.get(first_role)
        if bound is None or not bound.tensors:
            raise BackendInferenceError(
                "Ascend GraspGen requires object points on the first execution role",
                code="invalid_request",
            )
        tensor = bound.tensors[0]
        value = np.asarray(tensor.value)
        if value.ndim != 2 or value.shape[-1] != 3:
            raise BackendInferenceError(
                f"Ascend GraspGen observation.object_points must have shape [N,3], got {value.shape}",
                code="invalid_request",
            )
        return value.astype(np.float32, copy=False)

    @staticmethod
    def _prepare_graspgen_points(points: np.ndarray, config: GraspGenRuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
        clean = np.asarray(points, dtype=np.float32)
        clean = clean[np.isfinite(clean).all(axis=1)]
        if len(clean) == 0:
            raise BackendInferenceError(
                "Ascend GraspGen object point cloud contains no finite points",
                code="invalid_request",
            )
        if len(clean) > config.point_count:
            rng = np.random.default_rng(0)
            indices = np.sort(rng.choice(len(clean), config.point_count, replace=False))
            clean = clean[indices]
        center = clean.mean(axis=0, dtype=np.float32)
        scaled = np.ascontiguousarray((clean - center) * np.float32(config.kappa), dtype=np.float32)
        return scaled, center

    @staticmethod
    def _raw_grasp_outputs(
        outputs: Mapping[str, Mapping[int, np.ndarray]], plan: ExecutionPlan
    ) -> dict[str, np.ndarray]:
        role_outputs = outputs.get("discriminator_head")
        if role_outputs is None:
            raise BackendInferenceError(
                "Ascend GraspGen runtime did not return discriminator_head outputs",
                code="missing_grasp_output",
            )
        head_role = plan.role("discriminator_head")
        pose_binding = next((b for b in head_role.bindings.outputs if b.semantic == "grasp.poses"), None)
        confidence_binding = next((b for b in head_role.bindings.outputs if b.semantic == "grasp.confidence"), None)
        if pose_binding is None or pose_binding.index is None:
            raise BackendInferenceError(
                "Ascend GraspGen deployment is missing grasp.poses output binding",
                code="missing_grasp_output",
            )
        if confidence_binding is None or confidence_binding.index is None:
            raise BackendInferenceError(
                "Ascend GraspGen deployment is missing grasp.confidence output binding",
                code="missing_grasp_output",
            )
        return {
            "grasp.poses": np.asarray(role_outputs[int(pose_binding.index)]),
            "grasp.confidence": np.asarray(role_outputs[int(confidence_binding.index)]),
        }

    def _require_graspgen_runtime_config(self) -> GraspGenRuntimeConfig:
        if self._context is None:
            raise BackendInferenceError("AscendBackend is not fully loaded", code="runtime_not_loaded")
        if not isinstance(self._policy_config, Mapping):
            raise BackendInferenceError(
                "Ascend GraspGen requires a mapping policy config", code="invalid_policy_config"
            )
        return GraspGenRuntimeConfig.from_manifest(self._policy_config)

    @staticmethod
    def _validate_graspgen_config(config: Mapping[str, object]) -> None:
        for key in ("kappa", "diffusion_steps", "grasp_batch_size", "point_count"):
            value = config.get(key)
            if value is None:
                continue
            if isinstance(value, float | int):
                continue
            raise BackendLoadError(
                f"Ascend GraspGen requires numeric {key!r} in LeRobot config, got {type(value).__name__}",
                code="invalid_policy_config",
            )

    def _sample_noise(self, binding: TensorBinding) -> np.ndarray:
        if self._random is None:
            raise BackendInferenceError("Ascend random generator is unavailable", code="runtime_not_loaded")
        return np.ascontiguousarray(
            self._random.standard_normal(self._static_shape(binding)).astype(numpy_dtype(binding.dtype))
        )

    def _capture_bound_inputs(self, role: str, inputs: BoundInputs) -> None:
        image_index = 0
        for tensor in sorted(inputs.tensors, key=lambda item: int(item.index or 0)):
            semantic = tensor.semantic
            if role == "vlm" and semantic.startswith("observation.images."):
                name = f"vlm_in_image_{image_index}"
                image_index += 1
            elif role == "vlm" and semantic == "observation.language.tokens":
                name = "vlm_in_lang_tokens"
            elif role == "vlm" and semantic == "observation.language.attention_mask":
                name = "vlm_in_lang_masks"
            elif role == "vlm" and semantic == "prefix_att_2d_masks_4d":
                name = "vlm_in_prefix_mask_4d"
            else:
                name = f"{role}_in_{semantic}"
            self._capture(name, tensor.value)

    def _capture_vlm_outputs(self, outputs: Mapping[str, np.ndarray]) -> None:
        self._capture("past_kv_tensor", outputs.get("internal.past_kv"))
        self._capture("prefix_pad_masks", outputs.get("internal.prefix_pad_masks"))

    def _capture(self, name: str, value: object | None) -> None:
        if self._diagnostic_capture is not None and value is not None:
            self._diagnostic_capture(name, value)

    @staticmethod
    def _curvature_scores(velocities: list[np.ndarray], eps: float = 1e-6) -> list[float]:
        if not velocities:
            return []
        if len(velocities) == 1:
            return [0.0]
        scores = []
        for current, following in pairwise(velocities):
            current_flat = current.reshape(current.shape[0], -1).astype(np.float32, copy=False)
            following_flat = following.reshape(following.shape[0], -1).astype(np.float32, copy=False)
            difference = np.linalg.norm(following_flat - current_flat, axis=1)
            magnitude = np.linalg.norm(current_flat, axis=1) + eps
            scores.append(float(np.mean(difference / magnitude)))
        scores.append(scores[-1])
        return scores

    def _write_curvature_log(self, velocities: list[np.ndarray]) -> None:
        if self._curvature_log_path is None or self._denoising_schedule is None:
            return
        record = {
            "schedule": self._denoising_schedule.to_dict(),
            "curvature_scores": self._curvature_scores(velocities),
        }
        try:
            with self._curvature_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as exc:
            raise BackendInferenceError(
                f"Unable to write PI0.5 curvature log {self._curvature_log_path}: {exc}",
                code="curvature_log_failed",
            ) from exc

    @staticmethod
    def _request_execution(request: InferenceRequest) -> tuple[ExecutionPlan, Mapping[str, BoundInputs]]:
        plan = request.inputs.get("execution_plan")
        role_inputs = request.inputs.get("role_inputs")
        if not isinstance(plan, ExecutionPlan):
            raise BackendInferenceError("AscendBackend request is missing execution_plan", code="invalid_request")
        if not isinstance(role_inputs, Mapping):
            raise BackendInferenceError("AscendBackend request is missing role_inputs", code="invalid_request")
        for role in plan.role_names:
            if not isinstance(role_inputs.get(role), BoundInputs):
                raise BackendInferenceError(
                    f"AscendBackend role {role!r} inputs are not bound tensors",
                    code="invalid_request",
                )
        return plan, role_inputs

    @staticmethod
    def _host_output_indices(plan: ExecutionPlan, role: str) -> set[int]:
        semantics = {link.semantic for link in plan.host_links if link.producer == role}
        return {
            int(binding.index)
            for binding in plan.role(role).bindings.outputs
            if binding.semantic in semantics and binding.index is not None
        }

    @staticmethod
    def _semantic_outputs(plan: ExecutionPlan, role: str, outputs: Mapping[int, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            binding.semantic: outputs[int(binding.index)]
            for binding in plan.role(role).bindings.outputs
            if binding.index is not None and int(binding.index) in outputs
        }

    @staticmethod
    def _raw_action(outputs: object, plan: ExecutionPlan) -> np.ndarray:
        action_role = next(
            role for role in plan.roles if any(binding.semantic == "action" for binding in role.bindings.outputs)
        )
        binding = next(binding for binding in action_role.bindings.outputs if binding.semantic == "action")
        role_outputs = (
            outputs[action_role.name] if isinstance(outputs, Mapping) and action_role.name in outputs else outputs
        )
        if not isinstance(role_outputs, Mapping) or binding.index not in role_outputs:
            raise BackendInferenceError(
                "Ascend runtime did not return the bound action output", code="missing_action_output"
            )
        action = np.asarray(role_outputs[binding.index])
        if action.ndim == 1 and all(dimension > 0 for dimension in binding.shape):
            expected_size = int(np.prod(binding.shape, dtype=np.int64))
            if action.size == expected_size:
                action = action.reshape(binding.shape)
        return action

    @staticmethod
    def _chunk_size(action: np.ndarray) -> int:
        if action.ndim < 2 or action.shape[-2] < 1:
            raise BackendInferenceError(
                f"Ascend action output has invalid shape {action.shape}",
                code="invalid_action_shape",
            )
        return int(action.shape[-2])

    @staticmethod
    def _bound_by_index(inputs: BoundInputs) -> dict[int, np.ndarray]:
        return {int(tensor.index): tensor.value for tensor in inputs.tensors if tensor.index is not None}

    @staticmethod
    def _single_semantic_binding(
        bindings: tuple[TensorBinding, ...], semantics: frozenset[str], description: str
    ) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic in semantics]
        if len(matches) != 1 or matches[0].index is None:
            raise BackendLoadError(
                f"Ascend PI0.5 requires exactly one indexed {description} binding",
                code="invalid_pi05_bindings",
            )
        return matches[0]

    @staticmethod
    def _binding_for_semantic(bindings: tuple[TensorBinding, ...], semantic: str, description: str) -> TensorBinding:
        matches = [binding for binding in bindings if binding.semantic == semantic]
        if len(matches) != 1:
            raise BackendLoadError(
                f"Ascend deployment requires exactly one {description} binding for {semantic!r}",
                code="invalid_bindings",
            )
        return matches[0]

    @staticmethod
    def _static_shape(binding: TensorBinding) -> tuple[int, ...]:
        if any(dimension < 1 for dimension in binding.shape):
            raise BackendInferenceError(
                f"Ascend runtime-generated input {binding.semantic!r} requires a static shape, got {binding.shape}",
                code="dynamic_runtime_input",
            )
        return binding.shape

    @staticmethod
    def _require_artifact(context: RuntimeContext, role: str) -> Path:
        try:
            path = context.resolved_artifacts[role]
        except KeyError as exc:
            raise BackendLoadError(
                f"Ascend deployment is missing artifact role {role!r}", code="missing_artifact_role"
            ) from exc
        if not path.is_file():
            raise BackendLoadError(f"Ascend artifact {role!r} is not a regular file: {path}", code="invalid_artifact")
        return path

    @staticmethod
    def _load_denoising_schedule(
        context: RuntimeContext,
        *,
        diagnostic_schedule: PI05DenoisingSchedule | None = None,
        diagnostic_schedule_source: str | None = None,
        require_velocity: bool = False,
    ) -> tuple[PI05DenoisingSchedule | None, str | None, bool]:
        deployment = context.deployment
        assert isinstance(deployment, CompiledDeployment)
        action_outputs = [
            binding for binding in deployment.bindings["action_expert"].outputs if binding.semantic == "action"
        ]
        if len(action_outputs) != 1:
            raise BackendLoadError(
                "Ascend PI0.5 requires exactly one action output binding",
                code="invalid_pi05_bindings",
            )
        runtime_name = action_outputs[0].runtime_name or ""
        output_name = next(
            (part for part in reversed(runtime_name.split(":")) if part in {"action", "velocity", "v_t"}),
            runtime_name,
        )
        velocity_mode = output_name in {"velocity", "v_t"}
        if (diagnostic_schedule is not None or require_velocity) and not velocity_mode:
            raise BackendLoadError(
                "Ascend PI0.5 schedule diagnostics require a velocity/v_t Action Expert runtime output",
                code="invalid_runtime_options",
            )
        if diagnostic_schedule is not None:
            return diagnostic_schedule, diagnostic_schedule_source or "diagnostic", True

        artifact = deployment.artifacts.get("denoising_schedule")
        if artifact is None:
            if velocity_mode:
                raise BackendLoadError(
                    "Ascend PI0.5 velocity output requires a denoising_schedule artifact",
                    code="missing_denoising_schedule",
                )
            return None, None, False
        if artifact.format != "json":
            raise BackendLoadError(
                "Ascend PI0.5 denoising_schedule artifact must use format 'json'",
                code="invalid_denoising_schedule",
            )
        if not velocity_mode:
            raise BackendLoadError(
                "Ascend PI0.5 denoising_schedule requires a velocity/v_t Action Expert runtime output",
                code="invalid_denoising_schedule",
            )
        try:
            schedule = load_pi05_schedule(context.resolved_artifacts["denoising_schedule"])
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to load PI0.5 denoising schedule {artifact.path}: {exc}",
                code="invalid_denoising_schedule",
            ) from exc
        return schedule, artifact.path, False

    @staticmethod
    def _load_policy_config(path: Path) -> dict[str, object]:
        try:
            value = load_json_strict(path)
        except Exception as exc:
            raise BackendLoadError(
                f"Unable to read LeRobot config {path}: {exc}", code="invalid_policy_config"
            ) from exc
        if not isinstance(value, dict):
            raise BackendLoadError(f"LeRobot config must be an object: {path}", code="invalid_policy_config")
        return value

    @staticmethod
    def _validate_pi05_config(config: Mapping[str, object]) -> None:
        for key in ("chunk_size", "max_action_dim", "num_inference_steps"):
            value = config.get(key)
            if type(value) is not int or value < 1:
                raise BackendLoadError(
                    f"Ascend PI0.5 requires positive integer {key!r} in LeRobot config",
                    code="invalid_policy_config",
                )

    @staticmethod
    def _validate_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
        unknown = sorted(set(options) - _ALLOWED_RUNTIME_OPTIONS)
        if unknown:
            raise BackendLoadError(f"unknown Ascend runtime options: {unknown}", code="invalid_runtime_options")
        device_id = options.get("device_id", 0)
        if type(device_id) is not int or device_id < 0:
            raise BackendLoadError("Ascend device_id must be a non-negative integer", code="invalid_runtime_options")
        acl_config_path = options.get("acl_config_path")
        if acl_config_path is not None and (type(acl_config_path) is not str or not acl_config_path.strip()):
            raise BackendLoadError("Ascend acl_config_path must be a non-empty string", code="invalid_runtime_options")
        random_seed = options.get("random_seed")
        if random_seed is not None and type(random_seed) is not int:
            raise BackendLoadError("Ascend random_seed must be an integer or null", code="invalid_runtime_options")
        diagnostic_paths = {}
        for name in ("curvature_log_path",):
            value = options.get(name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise BackendLoadError(
                    f"Ascend {name} must be a non-empty string or null",
                    code="invalid_runtime_options",
                )
            diagnostic_paths[name] = value
        return {
            "device_id": device_id,
            "acl_config_path": acl_config_path,
            "random_seed": random_seed,
            **diagnostic_paths,
        }


def create_backend(context: RuntimeContext) -> AscendBackend:
    """Lazy registry factory for Ascend ACL execution."""

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment) or deployment.backend != "ascend":
        raise BackendLoadError("AscendBackend requires a compiled ascend deployment", code="invalid_deployment")
    options = AscendBackend._validate_runtime_options(context.runtime_options)
    return AscendBackend(int(options["device_id"]))
