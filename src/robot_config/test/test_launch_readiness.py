"""Unit tests for launch readiness helpers."""

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from launch import LaunchContext
from launch.actions import RegisterEventHandler
from launch_ros.actions import Node

from inference_manifest import BundleFile, canonical_bundle_digest
from robot_config.inference_config import InferenceConfigError
from robot_config.launch_builders import tracing as tracing_builder
from robot_config.launch_builders.control import (
    generate_controller_spawners,
    generate_ros2_control_nodes,
)
from robot_config.launch_builders.execution import (
    _attention_viz_request,
    generate_action_dispatcher_node,
    generate_execution_nodes,
    generate_inference_node,
)
from robot_config.launch_builders.navigation import generate_navigation_nodes
from robot_config.launch_builders.perception import generate_camera_nodes
from robot_config.launch_builders.sim_backend import get_sim_backend
from robot_config.launch_builders.teleop import generate_teleop_nodes
from robot_config.loader import load_robot_config_dict
from robot_config.wait_for_controllers import missing_inactive_controllers

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "robot.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("robot_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
robot_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(robot_launch)

_BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
_DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"


def _block_voice_asr_service_import(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("voice_asr_service"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def _text(substitutions):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in substitutions)


def _node_parameters(node):
    def decode_parameter(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode_parameter(tuple(item)) for item in value]
        text = _text(value)
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text.strip()

    raw = node._Node__parameters[0]
    parsed = {}
    for key, value in raw.items():
        name = _text(key)
        parsed[name] = decode_parameter(value)
    return parsed


def _node_remappings(node):
    remappings = []
    for src, dst in node._Node__remappings:
        remappings.append((_text(src), _text(dst)))
    return remappings


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_inference_bundle(root, deployments=None):
    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "type": "act",
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})
    (root / "model.safetensors").write_bytes(b"test-weights")
    paths = ("config.json", "model.safetensors", "policy_postprocessor.json", "policy_preprocessor.json")
    entries = [BundleFile(path=path) for path in paths]
    deployment_values = deployments or {"cpu": {"backend": "torch", "device": "cpu"}}
    deployment_values = {
        name: {"uuid": _DEPLOYMENT_UUID, "revision": 1, **value} for name, value in deployment_values.items()
    }
    _write_json(
        root / "inference_manifest.json",
        {
            "schema_version": 2,
            "bundle": {
                "uuid": _BUNDLE_UUID,
                "revision": 1,
                "name": root.name,
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(_BUNDLE_UUID, 1, root.name, entries),
                },
            },
            "deployments": deployment_values,
        },
    )
    return root


def _inference_robot_config(config_path, pipelines, *, selection=None):
    executor = {"type": "topic", "mode": "model_inference"}
    if selection is not None:
        executor["inference_pipeline"] = selection
    return {
        "_config_path": str(config_path),
        "name": "test_robot",
        "joints": {"all": ["1", "2", "3", "4", "5", "6"]},
        "control_modes": {
            "model_inference": {
                "inference": {"enabled": True, "pipelines": pipelines},
                "executor": executor,
            }
        },
    }


def test_missing_inactive_controllers_returns_only_non_active():
    controllers = [
        SimpleNamespace(name="joint_state_broadcaster", state="active"),
        SimpleNamespace(name="arm_position_controller", state="inactive"),
        SimpleNamespace(name="gripper_position_controller", state="active"),
    ]

    pending = missing_inactive_controllers(
        controllers,
        ["joint_state_broadcaster", "arm_position_controller", "missing_controller"],
    )

    assert pending == ["arm_position_controller", "missing_controller"]


def test_generate_controller_spawners_groups_activation():
    spawners = generate_controller_spawners(
        ["joint_state_broadcaster", "arm_position_controller"],
        use_sim=True,
    )

    assert len(spawners) == 1
    assert isinstance(spawners[0], Node)
    cmd_text = [item[0].text for item in spawners[0].cmd if item and hasattr(item[0], "text")]
    assert "--controller-manager" in cmd_text
    assert "controller_manager" in cmd_text
    assert "--activate-as-group" in cmd_text


def _relay_targets(nodes):
    """Collect (source_topic, target_topic) pairs from robot_config topic_relay nodes."""
    pairs = []
    for node in nodes:
        if getattr(node, "_Node__package", None) != "robot_config":
            continue
        args = getattr(node, "_Node__arguments", None) or []
        if len(args) >= 2:
            pairs.append((_text(args[0]), _text(args[1])))
    return pairs


def test_realsense_camera_topics_are_remapped_to_robot_config_contract_names():
    nodes = generate_camera_nodes(
        {
            "peripherals": [
                {
                    "type": "camera",
                    "name": "wrist",
                    "driver": "realsense",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "align_depth": True,
                    "enable_pointcloud": True,
                }
            ]
        },
        use_sim=False,
    )

    relay_pairs = _relay_targets(nodes)
    assert (
        "/camera/wrist_camera/color/image_raw",
        "/camera/wrist/image_raw",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/color/camera_info",
        "/camera/wrist/camera_info",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/aligned_depth_to_color/image_raw",
        "/camera/wrist/depth/image_rect_raw",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/aligned_depth_to_color/image_raw",
        "/camera/wrist/aligned_depth_to_color/image_raw",
    ) in relay_pairs
    assert (
        "/camera/wrist_camera/aligned_depth_to_color/camera_info",
        "/camera/wrist/aligned_depth_to_color/camera_info",
    ) in relay_pairs


def test_start_actions_handler_snapshots_action_list():
    original_actions = ["first"]
    handler = robot_launch._start_actions_on_success(
        original_actions,
        success_message="ok",
        failure_reason="failed",
    )
    original_actions.append("second")

    returned_actions = handler(SimpleNamespace(returncode=0), None)

    assert returned_actions == ["first"]


def test_robot_launch_loads_without_voice_asr_service_when_asr_disabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    launch_spec = importlib.util.spec_from_file_location("robot_launch_without_voice_asr", _LAUNCH_PATH)
    assert launch_spec is not None
    assert launch_spec.loader is not None
    launch_module = importlib.util.module_from_spec(launch_spec)

    launch_spec.loader.exec_module(launch_module)

    assert launch_module.generate_launch_description() is not None


def test_voice_asr_builder_skips_missing_package_when_disabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

    nodes = generate_voice_asr_nodes({"voice_asr": {"enabled": False}})

    assert nodes == []


def test_voice_asr_builder_reports_missing_package_when_enabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

    with pytest.raises(
        ModuleNotFoundError,
        match="voice_asr.enabled=true requires the voice_asr_service package to be installed",
    ):
        generate_voice_asr_nodes({"voice_asr": {"enabled": True}})


def test_controller_startup_timeout_comes_from_yaml_mapping():
    robot_config = {
        "controller_startup_timeout": {
            "sim": 42.5,
            "hardware": 7.5,
        }
    }

    assert robot_launch._resolve_controller_startup_timeout(robot_config, use_sim=True) == 42.5
    assert robot_launch._resolve_controller_startup_timeout(robot_config, use_sim=False) == 7.5


def test_gazebo_start_backend_uses_readiness_probe_instead_of_timer():
    adapter = get_sim_backend("gazebo")
    actions, create_node = adapter.start_backend({"name": "test_robot", "gazebo_world_name": "demo"})

    assert isinstance(create_node, Node)
    assert any(isinstance(action, Node) for action in actions)
    assert any(isinstance(action, RegisterEventHandler) for action in actions)


def test_shared_loader_preserves_source_path_metadata():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_single_arm.yaml"
    robot_config = load_robot_config_dict(config_path)

    assert robot_config["name"] == "so101_single_arm"
    assert robot_config["_config_path"] == str(config_path.resolve())


def test_launch_loader_uses_shared_dict_loader():
    robot_config = robot_launch.load_robot_config("so101_single_arm")

    assert robot_config["name"] == "so101_single_arm"
    assert robot_config["_config_path"].endswith("config/robots/so101_single_arm.yaml")


def test_default_trace_session_auto_suffixes_on_collision(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tracing_builder, "_trace_session_exists", lambda name: name == tracing_builder.DEFAULT_TRACE_SESSION_NAME
    )
    monkeypatch.setattr(
        tracing_builder,
        "datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _fmt: "20260428_180000")),
    )

    session_name, trace_dir = tracing_builder._resolve_trace_session(
        tracing_builder.DEFAULT_TRACE_SESSION_NAME,
        tmp_path,
    )

    assert session_name == "ib_robot_trace_20260428_180000"
    assert trace_dir == tmp_path / session_name


def test_custom_trace_session_collision_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing_builder, "_trace_session_exists", lambda _name: True)

    try:
        tracing_builder._resolve_trace_session("custom_trace", tmp_path)
    except RuntimeError as exc:
        assert "custom_trace" in str(exc)
    else:
        raise AssertionError("Expected custom tracing session collision to raise RuntimeError")


def test_generate_navigation_nodes_for_lekiwi_mode():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": True,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        }
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_lekiwi_sim_uses_sim_controller_override():
    config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "lekiwi.yaml"
    robot_config = load_robot_config_dict(config_path)

    _nodes, controller_names, _deferred_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=True,
        auto_start_controllers="true",
    )

    assert controller_names == ["joint_state_broadcaster", "base_controller"]


def test_generate_navigation_nodes_honors_force_enable_override():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": False,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        },
        force_enable=True,
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_launch_setup_enables_navigation_when_requested():
    context = LaunchContext()
    context.launch_configurations["robot_config"] = "lekiwi"
    context.launch_configurations["use_sim"] = "false"
    context.launch_configurations["auto_start_controllers"] = "false"
    context.launch_configurations["control_mode"] = "teleop"
    context.launch_configurations["with_navigation"] = "true"
    context.launch_configurations["navigation_mode"] = "full"

    actions = robot_launch.launch_setup(context)
    nav_nodes = [
        action for action in actions if isinstance(action, Node) and action.node_package == "robot_localization"
    ]

    assert len(nav_nodes) == 1


def test_launch_setup_uses_mock_sim_backend_without_controllers(tmp_path):
    src_config_path = Path(__file__).resolve().parents[1] / "config" / "robots" / "so101_single_arm.yaml"
    robot_config = load_robot_config_dict(src_config_path)
    robot_config.pop("_config_path", None)

    config_path = tmp_path / "so101_mock.yaml"
    config_path.write_text(yaml.safe_dump({"robot": robot_config}, sort_keys=False), encoding="utf-8")

    context = LaunchContext()
    context.launch_configurations["robot_config"] = "so101_single_arm"
    context.launch_configurations["config_path"] = str(config_path)
    context.launch_configurations["use_sim"] = "true"
    context.launch_configurations["sim_platform"] = "mock"
    context.launch_configurations["auto_start_controllers"] = "true"
    context.launch_configurations["control_mode"] = "model_inference"
    context.launch_configurations["with_navigation"] = "false"

    actions = robot_launch.launch_setup(context)
    node_packages = [action.node_package for action in actions if isinstance(action, Node)]

    assert node_packages.count("hardware_mock") == 1
    assert "controller_manager" not in node_packages
    assert "ros_gz_sim" not in node_packages

    timed_nodes = [
        action
        for action in actions
        if isinstance(action, Node) and action.node_package in {"inference_service", "action_dispatch"}
    ]
    assert len(timed_nodes) == 2
    assert all(_node_parameters(node)["use_sim_time"] is False for node in timed_nodes)

    inference_nodes = [node for node in timed_nodes if node.node_package == "inference_service"]
    assert _node_parameters(inference_nodes[0])["use_sim"] is True


def test_launch_loader_preserves_config_path_for_runtime_consumers():
    robot_config = robot_launch.load_robot_config("lekiwi")

    assert robot_config["name"] == "lekiwi"
    assert robot_config["_config_path"].endswith("config/robots/lekiwi.yaml")


def test_inference_execution_mode_cli_override_targets_one_pipeline():
    robot_config = {
        "control_modes": {
            "model_inference": {
                "inference": {
                    "pipelines": {
                        "primary": {"execution_mode": "monolithic"},
                        "backup": {"execution_mode": "monolithic"},
                    }
                }
            }
        }
    }
    context = LaunchContext()
    context.launch_configurations["inference_pipeline"] = "backup"
    context.launch_configurations["inference_execution_mode"] = "distributed"

    robot_launch._apply_inference_cli_overrides(context, robot_config, "model_inference")

    pipelines = robot_config["control_modes"]["model_inference"]["inference"]["pipelines"]
    assert pipelines["primary"]["execution_mode"] == "monolithic"
    assert pipelines["backup"]["execution_mode"] == "distributed"


def test_inference_execution_mode_cli_override_requires_pipeline():
    context = LaunchContext()
    context.launch_configurations["inference_execution_mode"] = "distributed"

    with pytest.raises(ValueError, match="requires inference_pipeline"):
        robot_launch._apply_inference_cli_overrides(context, {}, "model_inference")


def test_inference_execution_mode_cli_override_rejects_unknown_pipeline():
    robot_config = {
        "control_modes": {"model_inference": {"inference": {"pipelines": {"policy": {"execution_mode": "monolithic"}}}}}
    }
    context = LaunchContext()
    context.launch_configurations["inference_pipeline"] = "missing"
    context.launch_configurations["inference_execution_mode"] = "distributed"

    with pytest.raises(ValueError, match="unknown pipeline"):
        robot_launch._apply_inference_cli_overrides(context, robot_config, "model_inference")


def test_generate_one_pipeline_node_uses_only_unified_parameters(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {
            "policy": {
                "model_path": str(bundle),
                "deployment": "cpu",
                "execution_mode": "monolithic",
                "request_timeout": 2.5,
                "default_task": "pick banana",
                "runtime_options": {"perf_enabled": True, "perf_log_every": 3},
                "transport": {
                    "action_server": "/custom/dispatch",
                    "reset_service": "/custom/reset",
                    "health_topic": "/custom/health",
                    "action_topic": "/custom/actions",
                },
            }
        },
    )

    nodes = generate_inference_node(robot_config, "model_inference")

    assert len(nodes) == 1
    assert nodes[0].node_executable == "pipeline_policy_node"
    params = _node_parameters(nodes[0])
    assert params["pipeline_id"] == "policy"
    assert params["model_path"] == str(bundle.resolve())
    assert params["deployment"] == "cpu"
    assert params["request_timeout"] == 2.5
    assert params["default_task"] == "pick banana"
    assert json.loads(params["runtime_options_json"]) == {"perf_enabled": True, "perf_log_every": 3}
    assert params["action_server"] == "/custom/dispatch"
    assert params["reset_service"] == "/custom/reset"
    assert params["health_topic"] == "/custom/health"
    assert params["action_topic"] == "/custom/actions"
    for legacy_field in ("checkpoint", "model", "models", "device", "policy_path"):
        assert legacy_field not in params


def test_generate_two_pipeline_nodes_and_route_dispatcher_to_selected_pipeline(tmp_path):
    first = _create_inference_bundle(tmp_path / "first")
    second = _create_inference_bundle(tmp_path / "second")
    pipelines = {
        "primary": {"model_path": str(first), "deployment": "cpu", "execution_mode": "monolithic"},
        "backup": {
            "model_path": str(second),
            "deployment": "cpu",
            "execution_mode": "monolithic",
            "transport": {
                "action_server": "/backup/dispatch",
                "reset_service": "/backup/reset",
            },
        },
    }
    robot_config = _inference_robot_config(tmp_path / "robot.yaml", pipelines, selection="backup")

    nodes = generate_execution_nodes(robot_config, "model_inference")

    inference_nodes = [node for node in nodes if node.node_package == "inference_service"]
    assert [node.node_executable for node in inference_nodes] == ["pipeline_policy_node", "pipeline_policy_node"]
    dispatcher = next(node for node in nodes if node.node_package == "action_dispatch")
    params = _node_parameters(dispatcher)
    assert params["inference_action_server"] == "/backup/dispatch"
    assert params["inference_reset_service"] == "/backup/reset"
    assert params["inference_timeout_sec"] == 5.0


def test_multiple_pipelines_require_explicit_executor_selection(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    pipelines = {
        "first": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
        "second": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
    }
    robot_config = _inference_robot_config(tmp_path / "robot.yaml", pipelines)

    with pytest.raises(InferenceConfigError, match="executor.inference_pipeline"):
        generate_action_dispatcher_node(robot_config, "model_inference")


def test_launch_generation_rejects_invalid_deployment(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {"policy": {"model_path": str(bundle), "deployment": "missing", "execution_mode": "monolithic"}},
    )

    with pytest.raises(InferenceConfigError, match="Deployment 'missing'"):
        generate_inference_node(robot_config, "model_inference")


def test_launch_generation_emits_distributed_edge_with_transport_parameters(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {"edge": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "distributed"}},
    )

    nodes = generate_inference_node(robot_config, "model_inference")

    assert len(nodes) == 1
    assert nodes[0].node_executable == "pipeline_policy_node"
    params = _node_parameters(nodes[0])
    assert params["execution_mode"] == "distributed"
    assert params["request_topic"] == "/inference/edge/request"
    assert params["result_topic"] == "/inference/edge/result"
    assert params["heartbeat_topic"] == "/inference/edge/heartbeat"
    assert params["video_descriptor_topic"] == "/inference/edge/video/descriptors"
    assert params["video_status_topic"] == "/inference/edge/video/status"


def test_launch_generation_supports_mixed_execution_modes(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {
            "local": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
            "edge": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "distributed"},
        },
        selection="local",
    )

    nodes = generate_execution_nodes(robot_config, "model_inference")

    inference_nodes = [node for node in nodes if node.node_package == "inference_service"]
    assert len(inference_nodes) == 2
    modes = [_node_parameters(node)["execution_mode"] for node in inference_nodes]
    assert modes == ["monolithic", "distributed"]


def test_launch_generation_rejects_pipeline_endpoint_conflicts(tmp_path):
    bundle = _create_inference_bundle(tmp_path / "bundle")
    robot_config = _inference_robot_config(
        tmp_path / "robot.yaml",
        {
            "first": {"model_path": str(bundle), "deployment": "cpu", "execution_mode": "monolithic"},
            "second": {
                "model_path": str(bundle),
                "deployment": "cpu",
                "execution_mode": "monolithic",
                "transport": {"action_server": "/inference/first/dispatch"},
            },
        },
        selection="first",
    )

    with pytest.raises(InferenceConfigError, match="endpoint conflict"):
        generate_inference_node(robot_config, "model_inference")


def test_attention_viz_request_uses_robot_config_only():
    enabled, mode, _viz_config = _attention_viz_request({"attention_viz": {"enabled": False, "mode": "file"}})

    assert enabled is False
    assert mode == "file"


def test_generate_teleop_nodes_injects_target_joint_names_into_device_config():
    nodes = generate_teleop_nodes(
        {
            "joints": {
                "arm": ["1", "2", "3", "4", "5"],
                "gripper": ["6"],
            },
            "teleoperation": {
                "enabled": True,
                "active_device": "left_leader",
                "devices": [
                    {
                        "name": "left_leader",
                        "type": "leader_arm",
                        "port": "/dev/ttyACM0",
                        "target": {
                            "arm_joint_names": ["joint1_left", "joint2_left"],
                            "gripper_joint_names": ["joint6_left"],
                        },
                    }
                ],
            },
        }
    )

    params = _node_parameters(nodes[0])
    device_config = json.loads(params["device_config"].strip("'"))

    assert device_config["arm_joint_names"] == ["joint1_left", "joint2_left"]
    assert device_config["gripper_joint_names"] == ["joint6_left"]


def test_generate_joy_teleop_nodes_for_mobile_base():
    nodes = generate_teleop_nodes(
        {
            "teleoperation": {
                "enabled": True,
                "active_device": "lekiwi_gamepad",
                "devices": [
                    {
                        "name": "lekiwi_gamepad",
                        "type": "joy_teleop",
                        "config_path": "$(find robot_config)/config/lekiwi/lekiwi_teleop.yaml",
                        "input_device": "/dev/input/js0",
                    }
                ],
            }
        }
    )

    assert len(nodes) == 2
    assert all(isinstance(node, Node) for node in nodes)
