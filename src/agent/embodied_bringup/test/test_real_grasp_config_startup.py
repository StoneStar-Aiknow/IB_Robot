"""Real-config startup regression for every grasp-enabled robot YAML (issue #177).

Loads each in-tree robot YAML that declares a ``grasp_execution`` section
through the production loader once per navigation stage, prepares the runtime
binding exactly as the production launch does, derives the
``pick_executor_node`` parameters through ``generate_embodied_nodes``, and
constructs :class:`PickExecutorNode` for every pair whose effective
``grasp_execution.enabled`` is true. The hard-coded mandatory pairs fail the
test when any of them is missing or cannot start; the inheriting configs'
grasp-disabled default stages must generate no pick executor; the providerless
handeye configs must resolve their IK worker launch from the configuration
alone.

Scope note: only configs declaring ``grasp_execution`` are loaded. Loader-level
validity of unrelated configs is robot_config's own test surface; at the base
commit a known-invalid non-grasp config (``unitree_g1``) would otherwise fail
this suite for reasons outside the grasp pipeline.

Model resources (D5 step 3): perception services validate against
metadata-only fixture bundles under an isolated ``WORKSPACE`` (the configs'
relative ``models/...`` paths resolve there), and the node's strict GraspGen
identity loader validates against a synthetic strict-valid bundle whose
declared files exist and whose selected deployments (``ascend_310p`` and
``torch_cuda``) are declared. No validator is replaced and no configuration
field is removed; the only config mutations are the supported launch
preparations — the ``with_embodied:=true`` override applied by the production
``embodied_config_overrides_hook`` and the simulated runtime target written
back before the live-binding snapshot — and neither rewrites grasp fields.

The launch-parameter decoding helpers are copied from
``test_embodied_launch_builder.py`` (the launch ``Node`` stores parameter
dicts with launch-substitution values): this module owns its copies so the
real-config suite stays self-contained.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy
from launch_ros.actions import Node
from rclpy.parameter import Parameter

from embodied_bringup.launch_builders.embodied import generate_embodied_nodes
from inference_manifest import BundleFile, canonical_bundle_digest
from manipulation_execution.pick.node import PickExecutorNode
from robot_config.interface_binding import bind_robot_interfaces
from robot_config.launch_builders.embodied import embodied_config_overrides_hook
from robot_config.loader import _load_robot_section, load_robot_config_dict
from robot_config.runtime_target import RuntimeTarget, resolve_runtime_target
from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import render_robot_description, resolve_profile_argument
from robot_runtime.peripherals import merge_peripherals
from robot_runtime.profile import load_profile

ROBOTS_DIR = Path(__file__).resolve().parents[3] / "robot_config" / "config" / "robots"

#: Hard-coded mandatory coverage, independent of the YAML filter (D5 step 1).
MANDATORY_PAIRS = {
    ("lekiwi_handeye_realsense_grasp", ""),
    ("lekiwi_handeye_realsense_grasp_pc", ""),
    ("lekiwi_nav_grasp", "grasp"),
    ("lekiwi_nav_grasp", "hybrid"),
    ("lekiwi_nav_grasp_sound_real", "grasp"),
    ("lekiwi_hybrid_switch_test", "grasp"),
}
#: Their default (hybrid) stage disables grasp: no pick executor may appear.
GRASP_DISABLED_DEFAULTS = ("lekiwi_nav_grasp_sound_real", "lekiwi_hybrid_switch_test")

#: Metadata fixtures for every enabled perception service the grasp configs
#: reference (bundle dir name under models/ -> semantic identity family).
PERCEPTION_FIXTURE_BUNDLES = (
    "yolox_x_640",
    "pear_parameter_network",
    "grounding_dino_swint_seq8_1280x720",
    "sam2.1_hiera_tiny_prompt_ascend",
    "sam2.1_hiera_tiny",
    "ram_plus_swin_large_14m",
    "siglip2_so400m_patch14_384",
    "grounded_sam2_swint_ogc",
)


# ---------------------------------------------------------------------------
# Model resource fixtures (D5 step 3)
# ---------------------------------------------------------------------------


def _uuid(index: int) -> str:
    return f"123e4567-e89b-42d3-a456-4266141740{index:02d}"


def _deployments(index: int) -> dict:
    def deployment(offset: int, runtime_profile: dict) -> dict:
        return {
            "uuid": _uuid(index * 10 + offset),
            "revision": 1,
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": runtime_profile,
        }

    return {
        "ascend_310p": deployment(
            1,
            {"backend": "ascend", "target": {"runtime": "acl", "soc": "Ascend310P1"}, "profile": {"device_id": 0}},
        ),
        "torch_cuda": deployment(
            2, {"backend": "torch", "target": {"runtime": "torch"}, "profile": {"device": "cuda"}}
        ),
        "torch_cpu": deployment(3, {"backend": "torch", "target": {"runtime": "torch"}, "profile": {"device": "cpu"}}),
    }


def _write_tensor_model_bundle(
    root: Path, name: str, index: int, *, model_type: str, operation: str, semantic_identity: dict
) -> None:
    """Write a schema-v3 metadata bundle whose declared files also exist (strict-loadable)."""
    bundle = root / name
    (bundle / "assets").mkdir(parents=True)
    (bundle / "assets" / "identity.txt").write_text(name, encoding="utf-8")
    entry = BundleFile(path="assets/identity.txt")
    manifest = {
        "schema_version": 3,
        "bundle": {
            "uuid": _uuid(index),
            "revision": 1,
            "name": f"test-{name}",
            "files": [entry.model_dump(mode="json")],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(_uuid(index), 1, f"test-{name}", [entry]),
            },
        },
        "model": {
            "interface": "tensor_model",
            "model_type": model_type,
            "operation": operation,
            "inputs": [{"semantic": "features", "dtype": "float32", "shape": [1]}],
            "outputs": [{"semantic": "scores", "dtype": "float32", "shape": [1]}],
            "semantic_identity": semantic_identity,
        },
        "deployments": _deployments(index),
    }
    (bundle / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_graspgen_bundle(root: Path) -> None:
    """Strict-valid synthetic GraspGen bundle: declared files exist, both
    selected deployments (ascend_310p, torch_cuda) are declared, identities
    are valid. Strict loading instantiates no backend."""
    _write_tensor_model_bundle(
        root,
        "graspgen",
        9,
        model_type="graspgen",
        operation="generate_grasps",
        semantic_identity={
            "logical_model_revision": "graspgen-test@v1",
            "output_semantics": "grasp-pose-matrix-confidence-sorted-v1",
            "preprocessing_contract": "object-points-outlier-filtered-centered-kappa-scaled-v1",
        },
    )


def _fixture_model_workspace(tmp_path: Path, monkeypatch) -> None:
    """Redirect relative model bundle paths to metadata fixtures.

    The robot YAMLs' ``models/...`` paths resolve against ``WORKSPACE``; the
    real tree is a gitignored download. The fixtures carry the same identity
    and semantic contract the manifests declare, without the weights.
    """
    models = tmp_path / "fixture-models"
    for index, name in enumerate(PERCEPTION_FIXTURE_BUNDLES, start=1):
        embedding = (
            {
                "embedding": {
                    "embedding_space_id": "siglip2-test-space:v1",
                    "dimension": 4,
                    "normalization": "l2",
                    "image_preprocessing": "siglip2-image-v1",
                    "text_preprocessing": "siglip2-text-v1",
                }
            }
            if name.startswith("siglip2")
            else {}
        )
        _write_tensor_model_bundle(
            models,
            name,
            index,
            model_type=name.split("_")[0],
            operation="encode" if name.startswith("siglip2") else "detect",
            semantic_identity={
                "logical_model_revision": f"{name}@v1",
                "preprocessing_contract": f"{name}-pre-v1",
                "output_semantics": f"{name}-output-v1",
                **embedding,
            },
        )
    _write_graspgen_bundle(models)
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    # A symlinked models/ keeps absolute $(env WORKSPACE)-independent paths
    # inside the fixture tree too.
    (tmp_path / "models").symlink_to(models, target_is_directory=True)


def _isolate_home(tmp_path: Path, monkeypatch) -> None:
    """Isolated HOME plus the approved front-camera artifact the hybrid stage needs.

    An existing malformed camera-calibration artifact under the real HOME
    breaks loading (loader camera-calibration path); the approved artifact
    satisfies the front camera's mounting-transform requirement.
    """
    artifact = tmp_path / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    import yaml

    artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "approved",
                "device": {"name": "front_camera", "serial": "043322073551"},
                "transform": {
                    "parent_frame": "base_link",
                    "child_frame": "camera_front_optical_frame",
                    "translation": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))


# ---------------------------------------------------------------------------
# Launch parameter decoding (copied from test_embodied_launch_builder.py)
# ---------------------------------------------------------------------------


def _normalize_launch_param_mapping(raw_params):
    normalized = {}
    for raw_key, raw_value in raw_params.items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            value = "".join(getattr(item, "text", str(item)) for item in raw_value)
        else:
            value = raw_value
        normalized[key] = value
    return normalized


def _decode_launch_json_string(raw_value: str):
    normalized = raw_value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return json.loads(normalized)


def _decode_launch_string(raw_value: str):
    normalized = raw_value.strip()
    if normalized.endswith("\n..."):
        normalized = normalized[:-4].rstrip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return normalized


# ---------------------------------------------------------------------------
# Production preparation (D5 step 2)
# ---------------------------------------------------------------------------


def _lekiwi_runtime_description(config: dict) -> dict:
    """Build the public description exactly as the LeKiwi runtime launch does.

    Mirrors ``lekiwi_robot/launch/runtime.launch.py`` (profile load, deployment
    peripheral merge including the loader-recorded disabled entries, the
    effective FAST-LIO override, one rendered URDF feeding the public model)
    and ``robot_config/launch_builders/runtime.py``'s fragment construction.
    """
    provider = config["runtime"]["provider"]
    profile_path = resolve_profile_argument(provider, config["runtime"]["profile"])
    profile = load_profile(profile_path)
    runtime = config.get("runtime") or {}
    application = list(config.get("peripherals") or []) + list(runtime.get("_disabled_peripherals") or [])
    fast_lio = (config.get("navigation") or {}).get("fast_lio") or {}
    fragment: dict = {}
    if application:
        fragment["peripherals"] = application
    if fast_lio.get("enabled"):
        fragment["fast_lio"] = fast_lio
    merged_peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    fast_lio_config = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
    robot_description = render_robot_description(profile, Path(profile_path), simulated=True)
    return build_description(
        {**profile, "fast_lio": fast_lio_config},
        peripherals=merged_peripherals,
        use_sim=True,
        robot_description=robot_description,
        sim_platform="sdk",
    )


def _prepare_for_embodied(config: dict) -> tuple[dict, str]:
    """Deferred load -> with_embodied override + simulation target -> runtime binding.

    The supported launch preparation this suite may apply is exactly the
    production one: the ``with_embodied`` CLI override and the simulated
    runtime target written back before the live-binding snapshot
    (launch_registry/preparation.py step 13a) — nothing that rewrites grasp
    fields. Returns the prepared config and its active control mode.
    """
    # The supported launch override is the only config change this suite makes.
    embodied_config_overrides_hook(config, {"with_embodied": "true", "with_perception": ""})
    if (config.get("runtime") or {}).get("provider"):
        runtime_target = resolve_runtime_target(config, "", True)
        config.setdefault("runtime", {})["target"] = runtime_target.value
        assert runtime_target is RuntimeTarget.SIMULATION, runtime_target
        description = _lekiwi_runtime_description(config)
        assert description["execution"] == "simulated", description["execution"]
        config = bind_robot_interfaces(config, description)
        assert config["runtime"]["target"] == "simulation", config["runtime"]["target"]
    active_control_mode = str(config.get("default_control_mode", "moveit_planning"))
    return config, active_control_mode


def _pick_executor_parameters(config: dict, active_control_mode: str) -> tuple[dict, str]:
    nodes = generate_embodied_nodes(
        config,
        active_control_mode,
        motion_authorized=False,
        include_visual_games=False,
        include_perception=True,
        use_sim=True,
    )
    pick = next(
        (node for node in nodes if node.__dict__.get("_Node__node_name") == "pick_executor_node"),
        None,
    )
    assert pick is not None, "generate_embodied_nodes produced no pick_executor_node"
    assert isinstance(pick, Node)
    parameters = _normalize_launch_param_mapping(pick._Node__parameters[0])
    backend = _decode_launch_string(str(parameters["kinematics_backend"]))
    assert backend == ("runtime" if (config.get("runtime") or {}).get("provider") else "legacy_moveit")
    return parameters, backend


def _parameter_overrides(parameters: dict) -> list[Parameter]:
    """Clean launch-normalized parameter values back to plain Python values.

    ``launch_ros`` normalizes dict parameter strings through ``yaml.dump```,
    which appends the YAML document-end marker to plain scalars and quotes
    flow collections, so every string parameter needs the marker and the
    quoting stripped before ``Parameter`` construction. The ``*_json``
    parameters stay strings — the node parses them itself.
    """
    overrides = []
    for name, value in parameters.items():
        if isinstance(value, str):
            value = _decode_launch_string(value)
        overrides.append(Parameter(name, value=value))
    return overrides


def _construct_pick_executor(parameters: dict) -> PickExecutorNode:
    return PickExecutorNode(parameter_overrides=_parameter_overrides(parameters))


# ---------------------------------------------------------------------------
# Enumeration (D5 step 1)
# ---------------------------------------------------------------------------


def _stage_names(path: Path) -> list[str]:
    """Stage names from the inheritance-merged unresolved section (raw reader)."""
    _, merged = _load_robot_section(path)
    stages = merged.get("nav_stages") or {}
    return [""] if not stages else [str(name) for name in stages]


def _declares_grasp_execution(merged: dict) -> bool:
    """Whether any stage or the unresolved top level declares grasp_execution.

    A YAML may enable grasp only through a ``nav_stages.<stage>`` override
    (the mapping/navigation stages of a grasp config commonly disable it
    there); those configs must still be enumerated, so the selector checks
    the top level AND every stage override of the inheritance-merged
    unresolved section.
    """
    if isinstance(merged.get("grasp_execution"), dict):
        return True
    stages = merged.get("nav_stages") or {}
    return any(isinstance(stage, dict) and isinstance(stage.get("grasp_execution"), dict) for stage in stages.values())


def _grasp_enabled_pairs() -> list[tuple[str, str, dict]]:
    """(config, stage, loaded config) for every pair with grasp_execution enabled."""
    pairs: list[tuple[str, str, dict]] = []
    for path in sorted(ROBOTS_DIR.glob("*.yaml")):
        _, merged = _load_robot_section(path)
        if not _declares_grasp_execution(merged):
            continue
        for stage in _stage_names(path):
            config = load_robot_config_dict(
                path,
                nav_stage=stage,
                defer_interface_binding=True,
                require_local_model_artifacts=False,
            )
            if (config.get("grasp_execution") or {}).get("enabled"):
                pairs.append((path.stem, stage, config))
    return pairs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_only_grasp_declaration_is_enumerated():
    """Regression (D5 step 1): a stage-only grasp declaration still selects the YAML.

    A config may carry ``grasp_execution`` exclusively inside a
    ``nav_stages.<stage>`` override (with the unresolved top level not
    declaring it at all); the enumeration predicate must select such a YAML
    or that stage's pick executor would never be constructed.
    """
    assert _declares_grasp_execution({"nav_stages": {"grasp": {"grasp_execution": {"enabled": True}}}})
    assert _declares_grasp_execution({"grasp_execution": {"enabled": True}})
    # No grasp_execution anywhere: not selected, whatever else the stages declare.
    assert not _declares_grasp_execution({"nav_stages": {"grasp": {"fast_lio": {"enabled": True}}}})
    assert not _declares_grasp_execution({})


def test_every_grasp_enabled_real_config_starts_the_pick_executor(tmp_path, monkeypatch):
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), (
        "root conftest domain isolation is not active; node tests would share a ROS domain"
    )
    _isolate_home(tmp_path, monkeypatch)
    _fixture_model_workspace(tmp_path, monkeypatch)

    pairs = _grasp_enabled_pairs()
    selected = {(name, stage) for name, stage, _ in pairs}
    assert selected, "the real-config filter selected no grasp-enabled pair"
    missing = MANDATORY_PAIRS - selected
    assert not missing, f"mandatory grasp pairs missing from the real-config filter: {sorted(missing)}"

    failures: list[str] = []
    rclpy.init()
    try:
        for name, stage, config in pairs:
            node = None
            backend = "?"
            try:
                prepared, active_control_mode = _prepare_for_embodied(config)
                parameters, backend = _pick_executor_parameters(prepared, active_control_mode)
                node = _construct_pick_executor(parameters)
                assert node._grasp_geometry.__name__ == "so101_suite.gripper_geometry", (name, stage)
                assert node._wrist_guard.__name__ == "so101_suite.wrist_guard", (name, stage)
                assert node._mesh_directory is not None and node._mesh_directory.is_dir(), (name, stage)
            except Exception as exc:  # noqa: BLE001 - report every pair, then fail once
                failures.append(f"{name}/{stage or 'default'} (backend={backend}): {exc!r}")
            finally:
                if node is not None:
                    node.destroy_node()
    finally:
        rclpy.shutdown()
    assert not failures, "real grasp configs failed to start pick_executor_node:\n" + "\n".join(failures)


def test_grasp_disabled_default_stages_start_no_pick_executor(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    _fixture_model_workspace(tmp_path, monkeypatch)

    for name in GRASP_DISABLED_DEFAULTS:
        config = load_robot_config_dict(
            ROBOTS_DIR / f"{name}.yaml",
            defer_interface_binding=True,
            require_local_model_artifacts=False,
        )
        assert (config.get("grasp_execution") or {}).get("enabled") is False, name
        prepared, active_control_mode = _prepare_for_embodied(config)
        nodes = generate_embodied_nodes(
            prepared,
            active_control_mode,
            motion_authorized=False,
            include_visual_games=False,
            include_perception=True,
            use_sim=True,
        )
        names = [node.__dict__.get("_Node__node_name") for node in nodes]
        assert "pick_executor_node" not in names, f"{name} default stage started a pick executor"


def test_handeye_grasp_launch_auto_starts_parallel_ik_workers(tmp_path, monkeypatch):
    """Re-based from the synthetic-dict version: the real handeye configs drive the check."""
    import importlib.util

    _isolate_home(tmp_path, monkeypatch)
    _fixture_model_workspace(tmp_path, monkeypatch)

    launch_path = Path(__file__).parents[1] / "launch" / "embodied_pipeline.launch.py"
    spec = importlib.util.spec_from_file_location("embodied_pipeline_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("lekiwi_handeye_realsense_grasp", "lekiwi_handeye_realsense_grasp_pc"):
        config = load_robot_config_dict(
            ROBOTS_DIR / f"{name}.yaml",
            defer_interface_binding=True,
            require_local_model_artifacts=False,
        )
        workers_launch = config["grasp_execution"]["ik"]["workers_launch"]
        assert "$(find so101_motion)/launch/ik_workers.launch.py" in workers_launch, name

        action = module._parallel_ik_worker_action(config, "false")

        assert action is not None, name
        assert action.__class__.__name__ == "IncludeLaunchDescription", name
        arguments = dict(action._IncludeLaunchDescription__launch_arguments)
        assert arguments["worker_count"] == "4", name
        assert arguments["namespace_prefix"] == "ik_worker", name
        assert arguments["joint_state_topic"] == "/arm_joint_state_broadcaster/joint_states", name
        # The launch file the YAML names exists in the installed share; nothing
        # synthetic was injected by the test.
        from robot_config.utils import resolve_ros_path

        resolved = Path(resolve_ros_path(workers_launch))
        assert resolved.is_file() and "so101_motion" in resolved.parts, (name, resolved)

    # Runtime-backed configs take their workers from the profile, not a launch include.
    nav_config = load_robot_config_dict(
        ROBOTS_DIR / "lekiwi_nav_grasp.yaml",
        nav_stage="grasp",
        defer_interface_binding=True,
        require_local_model_artifacts=False,
    )
    assert module._parallel_ik_worker_action(nav_config, "false") is None
