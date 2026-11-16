"""GraspGen planning wrapper orchestrating the ROS-free planning modules."""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from inference_service.runtime_composition import (
    require_runtime_dependencies,
)
from inference_service.unified_runtime import RegistrySet, RuntimeProviders

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "manipulation_service GraspGen dependencies are missing. Ensure IB-Robot "
        "default setup completed with `./scripts/setup.sh` (GraspGen is installed "
        "by default on Ubuntu; skipped on openEuler Embedded)."
    ) from exc

from .candidate_width import estimate_candidate_target_widths
from .grasp_types import GraspCandidate, GraspDiagnostic
from .graspgen_backend import (
    DEFAULT_MODEL_DIR,
    LOCAL_BACKEND_REQUIRES_CUDA,
    VALID_INFERENCE_BACKENDS,
    LocalPipelineBackend,
    cuda_status,
    load_grasp_metadata,
)
from .point_cloud_completion import (
    complete_object_cloud_from_mask_depth,
    complete_object_cloud_prismatic_extrude,
    complete_scene_cloud_table_holes,
)
from .point_clouds import (
    align_mask_to_depth,
    depth_and_segmentation_to_point_clouds,
    downsample_points,
    point_cloud_outlier_removal,
    populate_depth_mask_diagnostics,
)
from .table_plane import (
    DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
    VALID_TABLETOP_FILTER_MODES,
    TablePlane,
    TablePlaneFit,
    TabletopFilterParams,
    fit_table_plane_ransac,
    run_tabletop_filter_stage,
    start_execution_table_fit,
    tabletop_clearance_sweep,
)

logger = logging.getLogger(__name__)


class GraspGenWrapper:
    def __init__(
        self,
        gripper_config: str = "graspgen_robotiq_2f_140.yml",
        model_dir: str | None = None,
        device: str = "cuda",
        collision_gripper: str | None = None,
        inference_backend: str = "local_cuda",
        local_manifest_path: str = "",
        local_deployment_name: str = "",
        ascend_local_manifest_path: str = "",
        ascend_local_deployment_name: str = "ascend_310p",
        ascend_local_device_id: int = 0,
        ascend_local_random_seed: int | None = None,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ):
        backend = inference_backend.strip().lower()
        if backend not in VALID_INFERENCE_BACKENDS:
            raise ValueError(
                f"invalid inference_backend {inference_backend!r}; expected one of {VALID_INFERENCE_BACKENDS}"
            )
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self.inference_backend = backend

        base = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        gripper_cfg_path = self._resolve_config(gripper_config, base)

        if not gripper_cfg_path.exists():
            raise FileNotFoundError(
                f"Gripper config not found: {gripper_cfg_path}\n"
                f"Download model weights with:\n"
                f'  HF_ENDPOINT=https://hf-mirror.com python3 -c "'
                f"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('adithyamurali/GraspGenModels', "
                f"'checkpoints/{gripper_config}', "
                f"local_dir='{base}')\""
            )

        logger.info("Loading GraspGen config from %s", gripper_cfg_path)
        self._cfg = None
        self._gripper_name, self._inference_point_count = load_grasp_metadata(gripper_cfg_path)

        if backend == "local_cuda":
            if device != "cuda":
                raise RuntimeError(f"{LOCAL_BACKEND_REQUIRES_CUDA} Requested device={device!r}.")
            if not torch.cuda.is_available():
                raise RuntimeError(f"{LOCAL_BACKEND_REQUIRES_CUDA} CUDA status: {cuda_status()}.")
            self.device = "cuda"
            self._sampler = None
            self._ascend_local_client = LocalPipelineBackend(
                local_manifest_path,
                deployment_name=local_deployment_name or "torch_cuda",
                registry_set=registry_set,
                providers=providers,
            )
            logger.info("GraspGen Torch pipeline ready (gripper=%s)", self._gripper_name)
        elif backend == "ascend_local":
            self.device = "ascend"
            self._sampler = None
            self._ascend_local_client = LocalPipelineBackend(
                ascend_local_manifest_path,
                deployment_name=local_deployment_name or ascend_local_deployment_name,
                device_id=ascend_local_device_id,
                random_seed=ascend_local_random_seed,
                registry_set=registry_set,
                providers=providers,
            )
            logger.info(
                "Ascend local GraspGen backend ready (manifest=%s, deployment=%s, gripper=%s)",
                ascend_local_manifest_path,
                ascend_local_deployment_name,
                self._gripper_name,
            )

        collision_name = collision_gripper or self._gripper_name
        self._collision_gripper_name = collision_name
        self._gripper_info = None
        self._collision_mesh = None
        self._collision_vertices = None
        if backend == "local_cuda":
            self._ensure_collision_geometry()

    def _ensure_collision_geometry(self) -> None:
        if self._collision_vertices is not None:
            return
        try:
            from grasp_gen.robot import get_gripper_info
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GraspGen source package is required when collision filtering, source-gripper "
                "tabletop sweep, or full debug rendering is enabled"
            ) from exc
        self._gripper_info = get_gripper_info(self._collision_gripper_name)
        self._collision_mesh = self._gripper_info.collision_mesh
        self._collision_vertices = np.asarray(self._collision_mesh.vertices)

    @staticmethod
    def _resolve_config(filename: str, model_dir: Path) -> Path:
        p = Path(filename)
        if p.is_absolute():
            return p

        bundle_root = model_dir.resolve()
        adapter_path = bundle_root / "assets" / "adapter.json"
        if adapter_path.is_file():
            try:
                adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unable to read GraspGen bundle adapter: {adapter_path}") from exc
            if not isinstance(adapter, dict):
                raise ValueError(f"GraspGen bundle adapter must contain a JSON object: {adapter_path}")
            configured_path = adapter.get("gripper_config")
            if isinstance(configured_path, str) and configured_path.strip():
                relative_path = Path(configured_path)
                if relative_path.is_absolute():
                    raise ValueError(
                        f"GraspGen bundle adapter gripper_config must be bundle-relative: {configured_path!r}"
                    )
                resolved_path = (bundle_root / relative_path).resolve()
                try:
                    resolved_path.relative_to(bundle_root)
                except ValueError as exc:
                    raise ValueError(
                        f"GraspGen bundle adapter gripper_config escapes the bundle: {configured_path!r}"
                    ) from exc
                if not resolved_path.is_file():
                    raise FileNotFoundError(f"GraspGen bundle adapter gripper_config does not exist: {resolved_path}")
                return resolved_path

        candidates = [
            model_dir / "checkpoints" / filename,
            model_dir / filename,
        ]
        env_model_dir = os.environ.get("GRASPGEN_MODEL_DIR")
        if env_model_dir:
            candidates.append(Path(env_model_dir) / "checkpoints" / filename)
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def plan_grasps(
        self,
        depth_image: np.ndarray,
        segmentation_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
        depth_scale: float = 1000.0,
        grasp_threshold: float = 0.5,
        num_grasps: int = 2000,
        topk_num_grasps: int = 300,
        min_grasps: int = 80,
        max_tries: int = 4,
        enable_collision_filter: bool = True,
        collision_threshold: float = 0.005,
        max_scene_points: int = 8192,
        enable_tabletop_filter: bool = True,
        enable_source_gripper_tabletop_sweep: bool = DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
        require_tabletop_filter: bool = True,
        tabletop_clearance: float = 0.003,
        tabletop_pregrasp_distance: float = 0.08,
        tabletop_pregrasp_steps: int = 5,
        tabletop_ransac_threshold: float = 0.006,
        tabletop_min_inlier_ratio: float = 0.15,
        tabletop_filter_mode: str = "strict",
        adaptive_tabletop_low_profile_height: float = 0.035,
        adaptive_tabletop_clearance_min: float = 0.001,
        adaptive_tabletop_clearance_max: float = 0.002,
        adaptive_tabletop_clearance_height_ratio: float = 0.12,
        adaptive_tabletop_pregrasp_min: float = 0.02,
        adaptive_tabletop_pregrasp_height_ratio: float = 2.0,
        adaptive_tabletop_hard_floor: float = -0.002,
        adaptive_tabletop_auto_tune: bool = True,
        adaptive_tabletop_retry_clearances: tuple[float, ...] | list[float] = (0.002, 0.001),
        adaptive_tabletop_retry_pregrasp_height_ratio: float = 1.5,
        adaptive_tabletop_retry_hard_floor: float = -0.003,
        target_width_axis_local: tuple[float, float, float] | list[float] = (1.0, 0.0, 0.0),
        target_width_percentile_low: float = 5.0,
        target_width_percentile_high: float = 95.0,
        target_width_min_m: float = 0.005,
        target_width_max_m: float = 0.14,
        enable_object_cloud_completion: bool = False,
        object_cloud_completion_mode: str = "none",
        object_cloud_completion_max_points: int = 5000,
        object_cloud_completion_kernel_size: int = 5,
        object_cloud_completion_min_neighbors: int = 6,
        enable_object_cloud_prismatic_extrude: bool = False,
        object_cloud_prismatic_extrude_max_points: int = 8000,
        object_cloud_prismatic_extrude_layers: int = 8,
        enable_scene_cloud_table_holes: bool = False,
        scene_cloud_table_holes_max_points: int = 8000,
    ) -> tuple[list[GraspCandidate], GraspDiagnostic]:
        plan_started = time.perf_counter()

        diag = GraspDiagnostic()
        inputs = self._validate_plan_inputs(
            diag,
            depth_image=depth_image,
            segmentation_mask=segmentation_mask,
            depth_scale=depth_scale,
            object_cloud_completion_mode=object_cloud_completion_mode,
            enable_object_cloud_completion=enable_object_cloud_completion,
            tabletop_filter_mode=tabletop_filter_mode,
        )
        if inputs is None:
            return [], diag
        depth_m, mask_for_pc, completion_mode, completion_enabled, mode = inputs

        clouds = self._build_point_clouds(
            diag,
            depth_m=depth_m,
            mask_for_pc=mask_for_pc,
            camera_intrinsics=camera_intrinsics,
            completion_enabled=completion_enabled,
            completion_mode=completion_mode,
            object_cloud_completion_max_points=object_cloud_completion_max_points,
            object_cloud_completion_kernel_size=object_cloud_completion_kernel_size,
            object_cloud_completion_min_neighbors=object_cloud_completion_min_neighbors,
            enable_object_cloud_prismatic_extrude=enable_object_cloud_prismatic_extrude,
            object_cloud_prismatic_extrude_max_points=object_cloud_prismatic_extrude_max_points,
            object_cloud_prismatic_extrude_layers=object_cloud_prismatic_extrude_layers,
            enable_scene_cloud_table_holes=enable_scene_cloud_table_holes,
            scene_cloud_table_holes_max_points=scene_cloud_table_holes_max_points,
            tabletop_ransac_threshold=tabletop_ransac_threshold,
            tabletop_min_inlier_ratio=tabletop_min_inlier_ratio,
        )
        if clouds is None:
            return [], diag
        scene_pc, object_pc_shell, shared_table_fit = clouds

        start_execution_table_fit(diag)
        diag.point_cloud_preprocess_ms = (time.perf_counter() - plan_started) * 1000.0

        prepared = self._filter_outliers_and_downsample(diag, object_pc_shell)
        if prepared is None:
            return [], diag
        object_pc_clean, object_pc_inference = prepared

        stage_started = time.perf_counter()
        grasps, confidences = self._run_batched_inference(
            object_pc_inference,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            min_grasps=min_grasps,
            max_tries=max_tries,
        )
        diag.model_inference_ms = (time.perf_counter() - stage_started) * 1000.0
        postprocess_started = time.perf_counter()

        filtered = self._collision_filter_stage(
            diag,
            grasps=grasps,
            confidences=confidences,
            scene_pc=scene_pc,
            object_pc_clean=object_pc_clean,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            enable_collision_filter=enable_collision_filter,
            collision_threshold=collision_threshold,
            max_scene_points=max_scene_points,
        )
        if filtered is None:
            return [], diag
        grasps_final, confidences_final = filtered

        if enable_tabletop_filter:
            params = TabletopFilterParams(
                mode=mode,
                enable_source_gripper_tabletop_sweep=enable_source_gripper_tabletop_sweep,
                require_tabletop_filter=require_tabletop_filter,
                tabletop_min_inlier_ratio=tabletop_min_inlier_ratio,
                tabletop_clearance=tabletop_clearance,
                tabletop_pregrasp_distance=tabletop_pregrasp_distance,
                tabletop_pregrasp_steps=tabletop_pregrasp_steps,
                adaptive_tabletop_low_profile_height=adaptive_tabletop_low_profile_height,
                adaptive_tabletop_clearance_min=adaptive_tabletop_clearance_min,
                adaptive_tabletop_clearance_max=adaptive_tabletop_clearance_max,
                adaptive_tabletop_clearance_height_ratio=adaptive_tabletop_clearance_height_ratio,
                adaptive_tabletop_pregrasp_min=adaptive_tabletop_pregrasp_min,
                adaptive_tabletop_pregrasp_height_ratio=adaptive_tabletop_pregrasp_height_ratio,
                adaptive_tabletop_hard_floor=adaptive_tabletop_hard_floor,
                adaptive_tabletop_auto_tune=adaptive_tabletop_auto_tune,
                adaptive_tabletop_retry_clearances=tuple(adaptive_tabletop_retry_clearances),
                adaptive_tabletop_retry_pregrasp_height_ratio=adaptive_tabletop_retry_pregrasp_height_ratio,
                adaptive_tabletop_retry_hard_floor=adaptive_tabletop_retry_hard_floor,
            )
            grasps_final, confidences_final = run_tabletop_filter_stage(
                grasps_final=grasps_final,
                confidences_final=confidences_final,
                scene_pc=scene_pc,
                shared_table_fit=shared_table_fit,
                object_pc_clean=object_pc_clean,
                diag=diag,
                params=params,
                sweep_fn=self._filter_tabletop_clearance,
            )
            if grasps_final is None:
                return [], diag

        return self._assemble_candidates(
            diag,
            grasps_final=grasps_final,
            confidences_final=confidences_final,
            object_pc_clean=object_pc_clean,
            postprocess_started=postprocess_started,
            plan_started=plan_started,
            target_width_axis_local=target_width_axis_local,
            target_width_percentile_low=target_width_percentile_low,
            target_width_percentile_high=target_width_percentile_high,
            target_width_min_m=target_width_min_m,
            target_width_max_m=target_width_max_m,
        )

    def _validate_plan_inputs(
        self,
        diag: GraspDiagnostic,
        *,
        depth_image: np.ndarray,
        segmentation_mask: np.ndarray,
        depth_scale: float,
        object_cloud_completion_mode: str,
        enable_object_cloud_completion: bool,
        tabletop_filter_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, str, bool, str] | None:
        """Validate inputs and populate the depth/mask diagnostics (D2 stage 1).

        Returns None exactly where plan_grasps returned ``[], diag`` early
        (the caller then returns ``[], diag``).
        """
        completion_mode = object_cloud_completion_mode.strip().lower()
        if completion_mode not in {"none", "mask_depth_inpaint"}:
            raise ValueError(
                f"invalid object_cloud_completion_mode {object_cloud_completion_mode!r}; "
                "expected one of ['none', 'mask_depth_inpaint']"
            )
        completion_enabled = bool(enable_object_cloud_completion and completion_mode == "mask_depth_inpaint")
        diag.object_cloud_completion_enabled = completion_enabled
        diag.object_cloud_completion_mode = completion_mode if completion_enabled else "none"
        mode = tabletop_filter_mode.strip().lower()
        if mode not in VALID_TABLETOP_FILTER_MODES:
            raise ValueError(
                f"invalid tabletop_filter_mode {tabletop_filter_mode!r}; "
                f"expected one of {sorted(VALID_TABLETOP_FILTER_MODES)}"
            )
        diag.tabletop_filter_mode = mode

        depth_m = depth_image.astype(np.float32) / np.float32(depth_scale)

        mask_for_pc = align_mask_to_depth(segmentation_mask, depth_m, diag)
        populate_depth_mask_diagnostics(mask_for_pc, depth_m, diag)

        if diag.mask_pixel_count == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = "segmentation mask is empty after alignment to depth image"
            logger.warning(diag.failure_reason)
            return None
        if diag.valid_depth_pixel_count == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = "depth image contains no finite positive depth pixels"
            logger.warning(diag.failure_reason)
            return None
        if diag.valid_depth_in_mask_count == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                "segmentation mask has no valid depth pixels — check RGB/depth alignment, "
                "depth holes on the target, or depth encoding/scale"
            )
            logger.warning(diag.failure_reason)
            return None
        return depth_m, mask_for_pc, completion_mode, completion_enabled, mode

    def _build_point_clouds(
        self,
        diag: GraspDiagnostic,
        *,
        depth_m: np.ndarray,
        mask_for_pc: np.ndarray,
        camera_intrinsics: np.ndarray,
        completion_enabled: bool,
        completion_mode: str,
        object_cloud_completion_max_points: int,
        object_cloud_completion_kernel_size: int,
        object_cloud_completion_min_neighbors: int,
        enable_object_cloud_prismatic_extrude: bool,
        object_cloud_prismatic_extrude_max_points: int,
        object_cloud_prismatic_extrude_layers: int,
        enable_scene_cloud_table_holes: bool,
        scene_cloud_table_holes_max_points: int,
        tabletop_ransac_threshold: float,
        tabletop_min_inlier_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, TablePlaneFit] | None:
        """Convert to point clouds, fit the shared table, complete the clouds (D2 stages 2-4).

        Returns None where plan_grasps returned ``[], diag`` early. The
        background execution-table fit is started by the caller right after
        this stage, as in the original flow.
        """
        fx = float(camera_intrinsics[0, 0])
        fy = float(camera_intrinsics[1, 1])
        cx = float(camera_intrinsics[0, 2])
        cy = float(camera_intrinsics[1, 2])

        scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
            depth_image=depth_m,
            segmentation_mask=mask_for_pc,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            target_object_id=1,
            remove_object_from_scene=True,
        )

        diag.object_point_count = len(object_pc)
        diag.object_point_count_raw = len(object_pc)
        diag.scene_point_count = len(scene_pc) if scene_pc is not None else 0
        diag.scene_point_count_raw = diag.scene_point_count
        diag.object_pc_raw = object_pc.astype(np.float32, copy=True)
        logger.info("Point clouds: scene=%d, object=%d", diag.scene_point_count, diag.object_point_count)

        if len(object_pc) == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                "object point cloud is empty after conversion despite non-empty mask/depth diagnostics — "
                "check mask labels and RGB-depth registration"
            )
            logger.warning(diag.failure_reason)
            return None

        shared_table_fit = TablePlaneFit(None)
        if scene_pc is not None and len(scene_pc) > 0:
            shared_min_inlier_ratio = min(float(tabletop_min_inlier_ratio), 0.10)
            table_fit_started = time.perf_counter()
            shared_table_fit = fit_table_plane_ransac(
                scene_pc,
                positive_reference=object_pc.mean(axis=0),
                distance_threshold=float(tabletop_ransac_threshold),
                min_inlier_ratio=shared_min_inlier_ratio,
            )
            diag.tabletop_ransac_duration_ms = (time.perf_counter() - table_fit_started) * 1000.0
            diag.tabletop_ransac_iterations = shared_table_fit.iterations_evaluated
        shared_table = shared_table_fit.plane

        if completion_enabled:
            completed_points = complete_object_cloud_from_mask_depth(
                depth_m,
                mask_for_pc,
                fx,
                fy,
                cx,
                cy,
                max_added_points=int(object_cloud_completion_max_points),
                kernel_size=int(object_cloud_completion_kernel_size),
                min_neighbors=int(object_cloud_completion_min_neighbors),
            )
            if len(completed_points):
                object_pc = np.vstack([object_pc, completed_points.astype(object_pc.dtype, copy=False)])
            diag.object_completion_added_count = int(len(completed_points))
            diag.object_point_count_completed = int(len(object_pc))
            diag.object_point_count = int(len(object_pc))
            logger.info(
                "Object PC completion: mode=%s raw=%d depth_added=%d completed=%d",
                completion_mode,
                diag.object_point_count_raw,
                len(completed_points),
                diag.object_point_count_completed,
            )
        else:
            diag.object_point_count_completed = int(len(object_pc))

        diag.object_prismatic_extrude_enabled = bool(enable_object_cloud_prismatic_extrude)
        object_pc_shell = object_pc
        if enable_object_cloud_prismatic_extrude and scene_pc is not None and len(scene_pc) > 0:
            prismatic_points = complete_object_cloud_prismatic_extrude(
                object_pc,
                scene_pc,
                table_plane=shared_table,
                max_added_points=int(object_cloud_prismatic_extrude_max_points),
                num_layers=int(object_cloud_prismatic_extrude_layers),
            )
            if len(prismatic_points):
                object_pc = np.vstack([object_pc, prismatic_points.astype(object_pc.dtype, copy=False)])
            diag.object_prismatic_extrude_added_count = int(len(prismatic_points))
            diag.object_point_count_completed = int(len(object_pc))
            object_pc_shell = object_pc
            logger.info(
                "Object PC prismatic extrude: added=%d completed=%d",
                len(prismatic_points),
                diag.object_point_count_completed,
            )
        diag.object_pc_after_completion = object_pc_shell.astype(np.float32, copy=True)

        # Optionally generate a local dense table patch around the target footprint.
        diag.scene_cloud_table_holes_enabled = bool(enable_scene_cloud_table_holes)
        if scene_pc is not None and len(scene_pc) > 0:
            if enable_scene_cloud_table_holes:
                scene_hole_pts = complete_scene_cloud_table_holes(
                    scene_pc,
                    footprint_points=object_pc,
                    table_plane=shared_table,
                    max_added_points=int(scene_cloud_table_holes_max_points),
                )
                if len(scene_hole_pts):
                    scene_pc = np.vstack([scene_pc, scene_hole_pts.astype(scene_pc.dtype, copy=False)])
                    logger.info("Scene cloud table patch: added=%d total=%d", len(scene_hole_pts), len(scene_pc))
                diag.scene_table_hole_added_count = int(len(scene_hole_pts))
            diag.scene_point_count = int(len(scene_pc))
            diag.scene_pc_after_completion = scene_pc.astype(np.float32, copy=True)

        return scene_pc, object_pc_shell, shared_table_fit

    def _filter_outliers_and_downsample(
        self,
        diag: GraspDiagnostic,
        object_pc_shell: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Outlier removal and inference downsampling (D2 stage 5).

        Returns None where plan_grasps returned ``[], diag`` early.
        """
        # GraspGen gets a hollow surface shell: visible object points, optional
        # mask-depth surface inpaint, and silhouette side walls to the table. We
        # never add interior volume samples.
        stage_started = time.perf_counter()
        outlier_input = downsample_points(object_pc_shell, self._inference_point_count)
        if len(outlier_input) < len(object_pc_shell):
            logger.info(
                "Object PC downsampled before outlier removal: %d -> %d points",
                len(object_pc_shell),
                len(outlier_input),
            )
        object_pc_tensor = torch.from_numpy(outlier_input).float()
        pc_filtered, _ = point_cloud_outlier_removal(object_pc_tensor)
        object_pc_clean = pc_filtered.numpy()
        diag.object_point_count = len(object_pc_clean)
        logger.info("Object PC for GraspGen inference (hollow shell): %d points", len(object_pc_clean))

        if len(object_pc_clean) < 20:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                f"object point cloud too sparse after outlier removal ({len(object_pc_clean)} points < 20)"
            )
            logger.warning(diag.failure_reason)
            return None

        object_pc_inference = downsample_points(object_pc_clean, self._inference_point_count)
        diag.object_point_count_graspgen_input = len(object_pc_inference)
        diag.object_pc_inference_input = object_pc_inference.astype(np.float32, copy=True)
        if len(object_pc_inference) < len(object_pc_clean):
            logger.info(
                "Object PC downsampled for GraspGen inference: %d -> %d points",
                len(object_pc_clean),
                len(object_pc_inference),
            )

        diag.outlier_removal_ms = (time.perf_counter() - stage_started) * 1000.0
        return object_pc_clean, object_pc_inference

    def _collision_filter_stage(
        self,
        diag: GraspDiagnostic,
        *,
        grasps,
        confidences,
        scene_pc: np.ndarray | None,
        object_pc_clean: np.ndarray,
        grasp_threshold: float,
        num_grasps: int,
        enable_collision_filter: bool,
        collision_threshold: float,
        max_scene_points: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Decode the inference output and run the collision filter (D2 stage 7).

        Returns None where plan_grasps returned ``[], diag`` early (zero-grasp
        and all-collision guards).
        """
        diag.raw_grasp_count = len(grasps)

        if len(grasps) == 0:
            diag.failure_stage = "graspgen_inference"
            diag.failure_reason = (
                f"GraspGen produced zero grasps (threshold={grasp_threshold}, "
                f"num_grasps={num_grasps}, object_points={len(object_pc_clean)}) — "
                f"object geometry may be unsuitable for this gripper or threshold too high"
            )
            logger.warning(diag.failure_reason)
            return None

        grasps_np = grasps.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        grasps_np[:, 3, 3] = 1.0

        if enable_collision_filter and scene_pc is not None and len(scene_pc) > 0:
            self._ensure_collision_geometry()
            from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

            pc_mean = scene_pc.mean(axis=0)
            scene_centered = scene_pc - pc_mean
            grasps_centered = grasps_np.copy()
            grasps_centered[:, :3, 3] -= pc_mean

            if len(scene_centered) > max_scene_points:
                idx = np.random.choice(len(scene_centered), max_scene_points, replace=False)
                scene_for_collision = scene_centered[idx]
            else:
                scene_for_collision = scene_centered

            collision_free_mask = filter_colliding_grasps(
                scene_pc=scene_for_collision,
                grasp_poses=grasps_centered,
                gripper_collision_mesh=self._collision_mesh,
                collision_threshold=collision_threshold,
            )

            diag.collision_filter_before = len(grasps_np)
            diag.collision_filter_after = int(collision_free_mask.sum())

            grasps_final = grasps_centered[collision_free_mask].copy()
            grasps_final[:, :3, 3] += pc_mean
            confidences_final = confidences_np[collision_free_mask]
            logger.info(
                "Collision filter: %d/%d grasps remaining",
                diag.collision_filter_after,
                diag.collision_filter_before,
            )
            if diag.collision_filter_after == 0:
                diag.failure_stage = "collision_filter"
                diag.failure_reason = (
                    f"all {diag.collision_filter_before} grasps filtered by collision — "
                    f"object is likely too close to scene obstacles for this gripper"
                )
                logger.warning(diag.failure_reason)
                return None
            return grasps_final, confidences_final

        diag.collision_filter_before = len(grasps_np)
        diag.collision_filter_after = len(grasps_np)
        return grasps_np, confidences_np

    def _assemble_candidates(
        self,
        diag: GraspDiagnostic,
        *,
        grasps_final: np.ndarray,
        confidences_final: np.ndarray,
        object_pc_clean: np.ndarray,
        postprocess_started: float,
        plan_started: float,
        target_width_axis_local,
        target_width_percentile_low: float,
        target_width_percentile_high: float,
        target_width_min_m: float,
        target_width_max_m: float,
    ) -> tuple[list[GraspCandidate], GraspDiagnostic]:
        """Estimate widths, assemble and sort the candidates (D2 stage 9)."""
        width_axis_local = np.asarray(target_width_axis_local, dtype=np.float64)
        width_estimates = estimate_candidate_target_widths(
            object_pc_clean,
            grasps_final,
            width_axis_local,
            percentile_low=target_width_percentile_low,
            percentile_high=target_width_percentile_high,
            min_width_m=target_width_min_m,
            max_width_m=target_width_max_m,
        )
        results = []
        for i, width_estimate in enumerate(width_estimates):
            target_width_m, target_width_quality, width_axis_camera, min_offset, max_offset = width_estimate
            results.append(
                GraspCandidate(
                    pose_4x4=grasps_final[i].astype(np.float32),
                    confidence=float(confidences_final[i]),
                    target_width_m=target_width_m,
                    target_width_quality=target_width_quality,
                    width_axis_camera=width_axis_camera,
                    target_width_min_offset_m=min_offset,
                    target_width_max_offset_m=max_offset,
                )
            )

        results.sort(key=lambda g: g.confidence, reverse=True)
        diag.grasp_postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        diag.plan_total_ms = (time.perf_counter() - plan_started) * 1000.0
        logger.info(
            "GraspGen plan stages_ms: point_cloud=%.1f outlier=%.1f model=%.1f postprocess=%.1f total=%.1f",
            diag.point_cloud_preprocess_ms,
            diag.outlier_removal_ms,
            diag.model_inference_ms,
            diag.grasp_postprocess_ms,
            diag.plan_total_ms,
        )
        return results, diag

    def _run_batched_inference(
        self,
        object_pc: np.ndarray,
        *,
        grasp_threshold: float,
        num_grasps: int,
        topk_num_grasps: int,
        min_grasps: int = 80,
        max_tries: int = 4,
    ):
        poses, confidence, _ = self._ascend_local_client.sample(
            object_pc,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            min_grasps=min_grasps,
            max_tries=max_tries,
        )
        return torch.from_numpy(poses), torch.from_numpy(confidence)

    @property
    def gripper_name(self) -> str:
        return self._gripper_name

    def warmup(self) -> bool:
        """Run one deterministic local inference to initialize lazy runtime state."""
        rng = np.random.default_rng(0)
        point_count = max(20, int(self._inference_point_count))
        object_pc = rng.uniform(
            low=(-0.03, -0.02, 0.35),
            high=(0.03, 0.02, 0.40),
            size=(point_count, 3),
        ).astype(np.float32)
        self._ascend_local_client.warmup(object_pc)
        return True

    @property
    def collision_gripper_name(self) -> str:
        return self._collision_gripper_name

    def _filter_tabletop_clearance(
        self,
        grasps: np.ndarray,
        confidences: np.ndarray,
        table: TablePlane,
        clearance: float,
        pregrasp_distance: float,
        pregrasp_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_collision_geometry()
        return tabletop_clearance_sweep(
            grasps,
            confidences,
            table=table,
            clearance=clearance,
            pregrasp_distance=pregrasp_distance,
            pregrasp_steps=pregrasp_steps,
            collision_vertices=self._collision_vertices,
        )
