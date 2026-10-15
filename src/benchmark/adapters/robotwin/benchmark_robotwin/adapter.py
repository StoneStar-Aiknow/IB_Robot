"""RoboTwin benchmark adapter — drives ``libs/robotwin`` Base_Task via the
generic ``BenchmarkAdapter`` protocol.

Key API translation (RoboTwin is NOT gym-shaped):

- ``reset()``  → ``env.setup_demo(seed=...)`` + ``env.get_obs()``
- ``step()``   → ``env.take_action(action, action_type)`` + ``env.get_obs()``
- ``render()`` → ``None`` (headless)
- ``close()``  → ``env.close_env()``
- success      → ``env.eval_success``
- terminated    → ``env.eval_success or env.take_action_cnt >= env.step_lim``

The task environment instance is created once per task and reused across
episodes — mirroring the upstream RoboTwin eval loop, which calls
``setup_demo(now_ep_num=...)`` repeatedly on one instance instead of paying
the SAPIEN engine + asset load per episode.

Heavy Sapien/RoboTwin imports are deferred to the first ``reset()``; the
plugin descriptor, discovery, and ``configure()`` never touch them.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_robotwin.action_codec import validate_robotwin_action_dict
from benchmark_robotwin.evaluation import RobotwinPlanRequest, resolve_plan_request
from benchmark_robotwin.observation_codec import convert_observation
from benchmark_robotwin.plan_provider import RobotwinPlanProvider
from benchmark_robotwin.plan_resolver import RobotwinPlanResolver
from benchmark_robotwin.version_probe import default_robotwin_root, probe_robotwin_provider
from benchmark_runtime.adapter import BenchmarkAdapter
from benchmark_runtime.io_descriptor import BenchmarkIODescriptor, FeatureDescriptor
from benchmark_runtime.models import (
    BenchmarkCapabilities,
    BenchmarkEnvironmentConfig,
    BenchmarkTask,
    NativeArtifact,
    ResetRequest,
    ResetResult,
    StepResult,
)
from benchmark_runtime.plan import BenchmarkPlan


class RobotwinAdapterError(RuntimeError):
    """Raised for any adapter-level contract violation."""


_FROZEN_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
_FROZEN_IMAGE_SIZE = (480, 640)
_FROZEN_CONTROL_MODE = "qpos"
_DEFAULT_ARM_DIM = 6
_DEFAULT_ACTION_TYPE = "qpos"
_DEFAULT_EMBODIMENT = "aloha-agilex"
_DEFAULT_TASK_CONFIG = "demo_clean"
_DEFAULT_ENV_CFG_TYPE = "aloha_agilex"
_SEED_EPISODE_OFFSET = 100000


def _build_io_descriptor(arm_dim: int) -> BenchmarkIODescriptor:
    """Build the policy I/O boundary for one embodiment.

    The state/action width is derived from ``arm_dim`` so the descriptor,
    the action codec, and the observation codec always agree on the same
    ``(arm_dim + 1) * 2`` dual-arm layout.
    """

    state_dim = (arm_dim + 1) * 2
    return BenchmarkIODescriptor(
        observations=(
            FeatureDescriptor("observation.images.cam_high", "image", "uint8", (480, 640, 3), "HWC"),
            FeatureDescriptor("observation.images.cam_left_wrist", "image", "uint8", (480, 640, 3), "HWC"),
            FeatureDescriptor("observation.images.cam_right_wrist", "image", "uint8", (480, 640, 3), "HWC"),
            FeatureDescriptor("observation.state", "state", "float32", (state_dim,), "C"),
        ),
        actions=(FeatureDescriptor("action", "action", "float32", (state_dim,), "C"),),
    )


class RobotwinAdapter(BenchmarkAdapter):
    """Phase 1 single-task RoboTwin adapter."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config: BenchmarkEnvironmentConfig | None = None
        self._plan: BenchmarkPlan | None = None
        self._plan_request: RobotwinPlanRequest | None = None
        self._task_config_args: dict[str, Any] = {}
        self._task_name: str = ""
        self._action_type: str = _DEFAULT_ACTION_TYPE
        self._arm_dim: int = _DEFAULT_ARM_DIM
        self._io_descriptor: BenchmarkIODescriptor = _build_io_descriptor(_DEFAULT_ARM_DIM)
        self._env: Any = None
        self._env_generation: int = 0
        self._task_environment_close_count: int = 0
        self._close_count: int = 0
        self._closed: bool = False
        self._last_head_frame: Any = None
        self._last_raw_observation: Any = None
        self._capabilities = BenchmarkCapabilities(
            supports_render=False,
            supports_init_state=True,
            supports_native_artifact=True,
            supports_reward=True,
            supports_success=True,
        )

    @property
    def capabilities(self) -> BenchmarkCapabilities:
        return self._capabilities

    def get_io_descriptor(self) -> BenchmarkIODescriptor:
        """Declare the canonical RobotWin policy I/O boundary."""
        return self._io_descriptor

    @property
    def active_task_identity(self) -> tuple[str, int] | None:
        with self._lock:
            if self._env is None or not self._task_name:
                return None
            return (self._plan_request.suite if self._plan_request else "robotwin", 0)

    @property
    def close_count(self) -> int:
        with self._lock:
            return self._close_count

    def configure(self, config: BenchmarkEnvironmentConfig) -> None:
        with self._lock:
            if self._closed:
                raise RobotwinAdapterError("adapter is closed")
            self._config = config

            options = config.options

            camera_names = options.get("camera_names", list(_FROZEN_CAMERA_NAMES))
            if tuple(camera_names) != _FROZEN_CAMERA_NAMES:
                raise RobotwinAdapterError(f"camera_names must be {_FROZEN_CAMERA_NAMES}, got {camera_names}")
            image_size = options.get("image_size", list(_FROZEN_IMAGE_SIZE))
            if tuple(image_size) != _FROZEN_IMAGE_SIZE:
                raise RobotwinAdapterError(f"image_size must be {_FROZEN_IMAGE_SIZE}, got {image_size}")
            control_mode = options.get("control_mode", _FROZEN_CONTROL_MODE)
            if control_mode != _FROZEN_CONTROL_MODE:
                raise RobotwinAdapterError(f"control_mode must be '{_FROZEN_CONTROL_MODE}', got '{control_mode}'")

            # Lane capability is negotiated from the SSOT like LIBERO: the
            # environment node validates the active parallel lane count
            # against this bound before materializing lanes.
            max_lane_count = options.get("max_lane_count", 1)
            if not isinstance(max_lane_count, int) or isinstance(max_lane_count, bool) or max_lane_count < 1:
                raise RobotwinAdapterError(f"benchmark.max_lane_count must be a positive int, got {max_lane_count!r}")
            self._capabilities = replace(self._capabilities, max_lane_count=max_lane_count)

            self._task_name = str(options.get("task_name", "")).strip()
            if not self._task_name:
                raise RobotwinAdapterError("benchmark.task_name is required")

            task_config_name = str(options.get("task_config", _DEFAULT_TASK_CONFIG)).strip()
            self._action_type = str(options.get("action_type", _DEFAULT_ACTION_TYPE)).strip()
            embodiment = str(options.get("embodiment", _DEFAULT_EMBODIMENT)).strip()
            self._arm_dim = int(options.get("arm_dim", _DEFAULT_ARM_DIM))
            env_cfg_type = str(options.get("env_cfg_type", _DEFAULT_ENV_CFG_TYPE)).strip()

            probe_robotwin_provider()
            self._robotwin_root = default_robotwin_root()

            if str(self._robotwin_root) not in sys.path:
                sys.path.insert(0, str(self._robotwin_root))

            self._task_config_args = self._load_task_config(task_config_name, embodiment)
            embodiment_arm_dim = int(self._task_config_args.pop("_embodiment_arm_dim"))
            if self._arm_dim != embodiment_arm_dim:
                raise RobotwinAdapterError(
                    f"benchmark.arm_dim={self._arm_dim} does not match embodiment {embodiment!r} "
                    f"joint count {embodiment_arm_dim}"
                )
            self._io_descriptor = _build_io_descriptor(self._arm_dim)

            plan_request = resolve_plan_request(options)
            if plan_request is None:
                plan_request = RobotwinPlanRequest(
                    suite="robotwin",
                    task_name=self._task_name,
                    task_config=task_config_name,
                    embodiment=embodiment,
                    env_cfg_type=env_cfg_type,
                    task_order_index=0,
                    tasks=(0,),
                    episodes_per_task=1,
                    max_steps=1000,
                    seed=10000,
                    save_sim_states=False,
                    video_enabled=False,
                    write_native=False,
                    write_canonical=False,
                    startup_timeout_sec=120.0,
                    max_duration_sec=300.0,
                )
            self._plan_request = plan_request
            self._plan = RobotwinPlanResolver(RobotwinPlanProvider()).resolve(plan_request)

    def _load_task_config(self, task_config_name: str, embodiment: str) -> dict[str, Any]:
        import yaml

        config_path = self._robotwin_root / "env_cfg" / "task_config" / f"{task_config_name}.yml"
        if not config_path.is_file():
            raise RobotwinAdapterError(f"task_config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise RobotwinAdapterError(f"task_config must be a Mapping, got {type(data).__name__}")

        # The robot YAML is the SSOT: its embodiment selection always wins
        # over whatever the upstream task_config yml happens to pin.
        data["embodiment"] = [embodiment]
        data.setdefault("render_freq", 0)
        data.setdefault("save_data", False)

        embodiment_index_path = self._robotwin_root / "env_cfg" / "task_config" / "_embodiment_config.yml"
        if embodiment_index_path.is_file():
            with embodiment_index_path.open("r", encoding="utf-8") as f:
                _embodiment_types = yaml.safe_load(f) or {}

            def _resolve_embodiment_file(emb_name: str) -> Path:
                entry = _embodiment_types.get(emb_name)
                if not isinstance(entry, dict) or "file_path" not in entry:
                    raise RobotwinAdapterError(f"embodiment '{emb_name}' missing in {embodiment_index_path}")
                return self._resolve_embodiment_path(entry["file_path"])

            def _load_embodiment_cfg(emb_name: str) -> dict:
                config = _resolve_embodiment_file(emb_name) / "config.yml"
                if not config.is_file():
                    raise RobotwinAdapterError(f"embodiment config.yml not found: {config}")
                with config.open(encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

            embodiment_list = data["embodiment"]
            if len(embodiment_list) == 1:
                data["left_robot_file"] = str(_resolve_embodiment_file(embodiment_list[0]))
                data["right_robot_file"] = data["left_robot_file"]
                data["dual_arm_embodied"] = True
                embodiment_cfg = _load_embodiment_cfg(embodiment_list[0])
                data["left_embodiment_config"] = embodiment_cfg
                data["right_embodiment_config"] = embodiment_cfg
            elif len(embodiment_list) == 3:
                data["left_robot_file"] = str(_resolve_embodiment_file(embodiment_list[0]))
                data["right_robot_file"] = str(_resolve_embodiment_file(embodiment_list[1]))
                data["embodiment_dis"] = embodiment_list[2]
                data["dual_arm_embodied"] = False
                data["left_embodiment_config"] = _load_embodiment_cfg(embodiment_list[0])
                data["right_embodiment_config"] = _load_embodiment_cfg(embodiment_list[1])
            else:
                raise RobotwinAdapterError(
                    f"embodiment list must have 1 or 3 entries, got {len(embodiment_list)}: {embodiment_list}"
                )

            arm_joints = data["left_embodiment_config"].get("arm_joints_name")
            if not isinstance(arm_joints, list) or not arm_joints or not isinstance(arm_joints[0], list):
                raise RobotwinAdapterError(f"embodiment {embodiment!r} config has no usable arm_joints_name")
            data["_embodiment_arm_dim"] = len(arm_joints[0])
        else:
            raise RobotwinAdapterError(f"embodiment index not found: {embodiment_index_path}")

        data.setdefault("eval_mode", True)
        data.setdefault(
            "data_type",
            {
                "rgb": True,
                "third_view": False,
                "depth": False,
                "pointcloud": False,
                "endpose": True,
                "qpos": True,
                "mesh_segmentation": False,
                "actor_segmentation": False,
                "observer": False,
            },
        )
        data.setdefault(
            "domain_randomization",
            {
                "random_background": False,
                "cluttered_table": False,
                "clean_background_rate": 1,
                "random_head_camera_dis": 0,
                "random_table_height": 0,
                "random_light": False,
                "crazy_random_light_rate": 0,
            },
        )
        data.setdefault(
            "camera",
            {
                "head_camera_type": "D435",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": True,
            },
        )
        return data

    def _resolve_embodiment_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self._robotwin_root / path
        return path.resolve()

    def get_plan(self) -> BenchmarkPlan:
        with self._lock:
            if self._plan is None:
                raise NotImplementedError("RobotwinAdapter does not have a resolved plan")
            return self._plan

    def list_tasks(self) -> list[BenchmarkTask]:
        with self._lock:
            if self._plan is None:
                return []
            suite = self._plan.suite
            return [
                BenchmarkTask(
                    suite=suite,
                    task_id=task.task_id,
                    name=task.name,
                    prompt=task.prompt,
                )
                for task in self._plan.tasks
            ]

    def reset(self, request: ResetRequest) -> ResetResult:
        with self._lock:
            self._require_usable()
            assert self._plan_request is not None

            if request.task_id != 0:
                raise RobotwinAdapterError(f"Phase 1 supports only task_id 0; got {request.task_id}")

            actual_seed = request.seed + _SEED_EPISODE_OFFSET * (1 + request.init_state_id)

            # One SAPIEN instance per task (upstream eval-loop semantics):
            # only the first reset creates it; later episodes reuse it.
            reused_environment = self._env is not None
            if not reused_environment:
                self._create_task_environment()

            try:
                self._env.setup_demo(
                    now_ep_num=request.init_state_id,
                    seed=actual_seed,
                    is_test=True,
                    **self._task_config_args,
                )
            except Exception as exc:
                self._clear_active_environment(close=True)
                raise RobotwinAdapterError(f"setup_demo failed: {exc}") from exc

            try:
                raw_obs = self._env.get_obs()
                observations = convert_observation(raw_obs, arm_dim=self._arm_dim)
            except Exception as exc:
                self._clear_active_environment(close=True)
                raise RobotwinAdapterError(f"post-reset obs conversion failed: {exc}") from exc

            self._last_raw_observation = raw_obs
            self._last_head_frame = observations.get("observation.images.cam_high")
            task_prompt = self._plan.tasks[0].prompt if self._plan else self._task_name
            step_lim = getattr(self._env, "step_lim", None) or 1000

            metadata = {
                "suite": self._plan_request.suite,
                "task_id": 0,
                "task_name": self._task_name,
                "seed": request.seed,
                "actual_seed": actual_seed,
                "episode_index": request.init_state_id,
                "requested_init_state_id": request.init_state_id,
                "actual_init_state_id": request.init_state_id,
                "init_state_count": max(1, self._plan_request.episodes_per_task),
                "init_state_wrapped": False,
                "task_environment_generation": self._env_generation,
                "task_environment_reused": reused_environment,
                "task_environment_close_count": self._task_environment_close_count,
                "step_limit": step_lim,
                "action_type": self._action_type,
                "arm_dim": self._arm_dim,
                "embodiment": self._plan_request.embodiment,
                "task_config": self._plan_request.task_config,
            }

            return ResetResult(
                observations=observations,
                task_prompt=task_prompt,
                metadata=metadata,
            )

    def step(self, action: dict[str, np.ndarray]) -> StepResult:
        with self._lock:
            self._require_usable()
            if self._env is None:
                raise RobotwinAdapterError("step called before reset")

            canonical = validate_robotwin_action_dict(action, arm_dim=self._arm_dim)

            try:
                self._env.take_action(canonical, action_type=self._action_type)
            except Exception as exc:
                raise RobotwinAdapterError(f"take_action failed: {exc}") from exc

            try:
                raw_obs = self._env.get_obs()
                observations = convert_observation(raw_obs, arm_dim=self._arm_dim)
            except Exception as exc:
                raise RobotwinAdapterError(f"post-step obs conversion failed: {exc}") from exc

            self._last_raw_observation = raw_obs
            self._last_head_frame = observations.get("observation.images.cam_high")
            eval_success = bool(getattr(self._env, "eval_success", False))
            take_action_cnt = int(getattr(self._env, "take_action_cnt", 0))
            step_lim = int(getattr(self._env, "step_lim", 1000))
            terminated = eval_success or take_action_cnt >= step_lim

            return StepResult(
                observations=observations,
                terminated=terminated,
                truncated=False,
                reward=float(eval_success),
                is_success=eval_success,
                standard_metrics={
                    "reward": float(eval_success),
                    "take_action_cnt": take_action_cnt,
                    "step_limit": step_lim,
                },
                native_metrics={
                    "robotwin_eval_success": eval_success,
                    "robotwin_take_action_cnt": take_action_cnt,
                },
                info={
                    "action_type": self._action_type,
                    "arm_dim": self._arm_dim,
                },
            )

    def render(self) -> np.ndarray | None:
        return self._last_head_frame if self._last_head_frame is not None else None

    def collect_native_artifacts(self) -> list[NativeArtifact]:
        return []

    def on_task_finalized(self, suite: str, task_id: int) -> None:
        with self._lock:
            del suite, task_id
            self._clear_active_environment(close=True)

    def get_native_reset_payload(self) -> dict[str, Any] | None:
        with self._lock:
            if self._env is None:
                return None
            return {
                "observations": self._last_raw_observation,
                "native_env": self._env,
                "sim_state": None,
            }

    def get_native_step_payload(self) -> dict[str, Any] | None:
        return self.get_native_reset_payload()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clear_active_environment(close=True)
            self._close_count += 1

    def _require_usable(self) -> None:
        if self._closed:
            raise RobotwinAdapterError("adapter is closed")
        if self._config is None:
            raise RobotwinAdapterError("adapter not configured")
        if self._plan is None:
            raise RobotwinAdapterError("adapter has no resolved plan")

    def _create_task_environment(self) -> None:
        try:
            module = importlib.import_module(f"envs.{self._task_name}")
        except ImportError as exc:
            raise RobotwinAdapterError(f"failed to import envs.{self._task_name}: {exc}") from exc

        try:
            env_class = getattr(module, self._task_name)
        except AttributeError as exc:
            raise RobotwinAdapterError(f"envs.{self._task_name} has no class {self._task_name}: {exc}") from exc

        try:
            self._env = env_class()
        except Exception as exc:
            raise RobotwinAdapterError(f"failed to instantiate {self._task_name}(): {exc}") from exc

        self._env_generation += 1

    def _clear_active_environment(self, *, close: bool) -> None:
        env = self._env
        self._env = None
        self._last_raw_observation = None
        self._last_head_frame = None
        if close and env is not None:
            with contextlib.suppress(Exception):
                env.close_env()
            viewer = getattr(env, "viewer", None)
            if viewer is not None:
                with contextlib.suppress(Exception):
                    viewer.close()
            self._task_environment_close_count += 1
