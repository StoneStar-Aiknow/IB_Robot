"""Host-orchestrated GraspGen execution over the eight compiled Ascend OM roles."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding
from inference_service.backends.ascend.model import numpy_dtype
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.types import RuntimeContext
from inference_service.model_sessions.ascend import AscendOmModelSession
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest
from model_utils.graspgen_contract import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_DIFFUSION_SAMPLE,
    GRASPGEN_DIFFUSION_TIME,
    GRASPGEN_EXECUTION,
    GRASPGEN_GLOBAL_FEATURES,
    GRASPGEN_GRASP_RT,
    GRASPGEN_GRASP_SCORES,
    GRASPGEN_GROUPED_POINTS,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
    GRASPGEN_PREDICTED_NOISE,
    GRASPGEN_STAGE2_FEATURES,
    GRASPGEN_STAGE_FEATURES,
)

from .graspgen_adapter import GraspGenConfig
from .graspgen_geometry import (
    DDPMSchedulerNumpy,
    PointNetGeometry,
    build_pointnet_geometry,
    group_features,
    matrix_to_rt,
    rt_to_matrix,
)


class GraspGenAscendSession(AscendOmModelSession):
    """Drive the GraspGen roles one at a time, computing what the compiled graph cannot.

    GraspGen is not one graph: PointNet++ sampling and grouping, the DDPM denoising loop
    and the SO(3) pose conversion all happen between OM invocations. The generic session
    walks ``execution`` straight through, so this subclass overrides only ``_execute`` and
    reuses the inherited loading, device-link wiring and per-role binding logic unchanged.

    Every intermediate travels under a ``host.graspgen.*`` key in a local dict, so the only
    tensors crossing the ``ModelSession`` boundary are the two the manifest declares.
    """

    execution_contract = "request-iterative"
    orchestration_visibility = "session"

    allowed_runtime_options = AscendOmModelSession.allowed_runtime_options | {"random_seed"}

    def __init__(self, device_id: int = 0, *, config: GraspGenConfig, **kwargs) -> None:
        super().__init__(device_id, **kwargs)
        self._config = config
        self._random: np.random.Generator | None = None

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        deployment = context.deployment
        if isinstance(deployment, CompiledDeployment) and deployment.execution != GRASPGEN_EXECUTION:
            raise BackendLoadError(
                f"GraspGen requires execution {list(GRASPGEN_EXECUTION)}, got {list(deployment.execution)}",
                code="invalid_execution_plan",
            )
        seed = context.runtime_options.get("random_seed")
        if seed is not None and type(seed) is not int:
            raise BackendLoadError("GraspGen random_seed must be an integer or null", code="invalid_runtime_options")
        super()._load(context, rollback)
        self._random = np.random.default_rng(seed)

    def _close(self) -> None:
        self._random = None
        super()._close()

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        config = self._config
        points = np.asarray(request.inputs[GRASPGEN_POINT_CLOUD_SEMANTIC], dtype=np.float32)
        geometry = build_pointnet_geometry(points, npoints=config.npoints, radii=config.radii, nsamples=config.nsamples)

        # ``values`` accumulates every tensor the roles exchange. _run_role writes each
        # role's outputs back into it, so a device-linked embedding that stays on the card
        # is simply absent - and _role_inputs skips linked slots anyway.
        values: dict[str, object] = {}
        self._run_encoder("generator", geometry, values)
        self._run_encoder("discriminator", geometry, values)
        sample = self._run_denoiser(values)

        poses = rt_to_matrix(sample, config.kappa)
        self._bind(values, "discriminator_head", GRASPGEN_GRASP_RT, matrix_to_rt(poses, config.kappa))
        head_outputs = self._run_role(GRASPGEN_EXECUTION.index("discriminator_head"), "discriminator_head", values)
        try:
            scores = np.asarray(head_outputs[GRASPGEN_GRASP_SCORES], dtype=np.float32)
        except KeyError as exc:
            raise BackendInferenceError(
                "GraspGen discriminator head did not return grasp scores", code="missing_runtime_output"
            ) from exc
        return {
            GRASPGEN_POSE_SEMANTIC: np.ascontiguousarray(poses, dtype=np.float32),
            GRASPGEN_CONFIDENCE_SEMANTIC: np.ascontiguousarray(scores.reshape(-1), dtype=np.float32),
        }

    def _run_encoder(self, prefix: str, geometry: PointNetGeometry, values: dict[str, object]) -> None:
        """Run one PointNet++ encoder chain, grouping between its stages on the host."""
        sa1, sa2, head = f"{prefix}_sa1", f"{prefix}_sa2", f"{prefix}_encoder_head"

        self._bind(values, sa1, GRASPGEN_GROUPED_POINTS, geometry.stage1_input)
        sa1_features = self._staged_features(sa1, values)
        stage2_input = np.concatenate(
            [geometry.stage2_relative_xyz, group_features(sa1_features, geometry.stage2_indices)], axis=1
        )

        self._bind(values, sa2, GRASPGEN_STAGE2_FEATURES, stage2_input)
        sa2_features = self._staged_features(sa2, values)
        head_input = np.concatenate([geometry.stage2_xyz.T[None, :, None, :], sa2_features[:, :, None, :]], axis=1)

        self._bind(values, head, GRASPGEN_GLOBAL_FEATURES, head_input)
        self._run_role(GRASPGEN_EXECUTION.index(head), head, values)

    def _staged_features(self, role: str, values: dict[str, object]) -> np.ndarray:
        outputs = self._run_role(GRASPGEN_EXECUTION.index(role), role, values)
        try:
            return np.asarray(outputs[GRASPGEN_STAGE_FEATURES], dtype=np.float32)
        except KeyError as exc:
            raise BackendInferenceError(
                f"GraspGen role {role!r} did not return set-abstraction features", code="missing_runtime_output"
            ) from exc

    def _run_denoiser(self, values: dict[str, object]) -> np.ndarray:
        config = self._config
        role_index = GRASPGEN_EXECUTION.index("denoiser")
        sample_binding = self._binding("denoiser", "inputs", GRASPGEN_DIFFUSION_SAMPLE)
        time_binding = self._binding("denoiser", "inputs", GRASPGEN_DIFFUSION_TIME)
        noise_shape = self._static_shape(sample_binding)
        # The scheduler state stays float32 whatever the compiled slots declare; only the
        # tensors handed to ACL are narrowed, so a float16 deployment does not degrade the
        # denoising arithmetic itself.
        sample = np.ascontiguousarray(self._standard_normal(noise_shape), dtype=np.float32)
        position_scheduler = DDPMSchedulerNumpy(config.diffusion_steps, "scaled_linear")
        rotation_scheduler = DDPMSchedulerNumpy(config.diffusion_steps, "squaredcos_cap_v2")
        variance_shape = (noise_shape[0], 3)
        for timestep in position_scheduler.timesteps:
            self._bind(values, "denoiser", GRASPGEN_DIFFUSION_SAMPLE, sample)
            self._bind(values, "denoiser", GRASPGEN_DIFFUSION_TIME, np.full(self._static_shape(time_binding), timestep))
            outputs = self._run_role(role_index, "denoiser", values)
            try:
                predicted_noise = np.asarray(outputs[GRASPGEN_PREDICTED_NOISE], dtype=np.float32)
            except KeyError as exc:
                raise BackendInferenceError(
                    "GraspGen denoiser did not return predicted noise", code="missing_runtime_output"
                ) from exc
            # Position and rotation are denoised under different beta schedules, and each
            # draws its own variance noise - matching the upstream two-scheduler runtime.
            position, _ = position_scheduler.step(
                predicted_noise[:, :3], int(timestep), sample[:, :3], self._variance_noise(variance_shape, timestep)
            )
            rotation, _ = rotation_scheduler.step(
                predicted_noise[:, 3:], int(timestep), sample[:, 3:], self._variance_noise(variance_shape, timestep)
            )
            sample = np.ascontiguousarray(np.concatenate([position, rotation], axis=-1), dtype=np.float32)
        return sample

    def _variance_noise(self, shape: tuple[int, ...], timestep: object) -> np.ndarray | None:
        if int(timestep) <= 0:
            return None
        return np.ascontiguousarray(self._standard_normal(shape), dtype=np.float32)

    def _standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._random is None:
            raise BackendInferenceError("GraspGen random generator is unavailable", code="runtime_not_loaded")
        return self._random.standard_normal(shape)

    def _bind(self, values: dict[str, object], role: str, semantic: str, value: np.ndarray) -> None:
        """Publish a host tensor under its semantic, narrowed to the slot's declared dtype.

        ACL copies raw bytes and checks only the byte count, so a float32 host array fed to
        a float16 input fails as an opaque size mismatch. The manifest binding, not a
        float32 assumption, owns the ABI.
        """
        dtype = numpy_dtype(self._binding(role, "inputs", semantic).dtype)
        values[semantic] = np.ascontiguousarray(value, dtype=dtype)

    def _binding(self, role: str, direction: str, semantic: str) -> TensorBinding:
        bindings = getattr(self._loaded_deployment().bindings[role], direction)
        matches = [binding for binding in bindings if binding.semantic == semantic]
        if len(matches) != 1 or matches[0].index is None:
            raise BackendInferenceError(
                f"GraspGen role {role!r} does not uniquely bind {direction[:-1]} {semantic!r}",
                code="invalid_graspgen_binding",
            )
        return matches[0]

    @staticmethod
    def _static_shape(binding: TensorBinding) -> tuple[int, ...]:
        if any(dimension < 1 for dimension in binding.shape):
            raise BackendInferenceError(
                f"GraspGen host-generated input {binding.semantic!r} needs a static shape, got {list(binding.shape)}",
                code="dynamic_runtime_input",
            )
        return tuple(binding.shape)
