"""The X2 grasp binds the OmniHand runtime; each host deployment serves the grasp models its accelerator runs.

Pure configuration checks: no ROS graph, no vendor SDK, no GPU. The shared
aimdk_x2_grasp.yaml names no models; aimdk_x2_grasp_cuda.yaml and
aimdk_x2_grasp_310p.yaml each add theirs. The perception service declarations
are parsed with the same parser the launch uses; loading a host deployment
resolves its bundles and skips when they are not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from robot_config.interface_binding import InterfaceBindingError, bind_robot_interfaces, required_interface_ids
from robot_config.loader import load_robot_config_dict, validate_robot_config_dict
from robot_runtime.capabilities import all_capabilities

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG = ROBOTS / "aimdk_x2_grasp.yaml"
CUDA_BUNDLES = ("grounded_sam2_swint_ogc", "graspgenx_omnihand_left_x2")
ASCEND_BUNDLES = ("grounding_dino_swint_seq8_1280x720", "sam2.1_hiera_tiny_prompt_ascend", "graspgenx_omnihand_left_x2")
#: Each host deployment: the bundles it serves, and the deployments they run on that host's accelerator.
HOSTS = {
    "aimdk_x2_grasp_cuda": (CUDA_BUNDLES, {"torch_cuda"}),
    "aimdk_x2_grasp_310p": (ASCEND_BUNDLES, {"ascend_310p", "torch_npu"}),
}
# src/robot_config/test/ -> src/
PROFILE = (
    Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
)
WORKSPACE = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def merged():
    # The grasp backends' runtime topics are live logical bindings, resolved at launch.
    return load_robot_config_dict(str(CONFIG), defer_interface_binding=True)


def _host(name):
    # Loading a host deployment resolves its perception bundles: skip, rather than error, without them.
    bundles, _ = HOSTS[name]
    missing = [
        bundle for bundle in bundles if not (WORKSPACE / "models" / bundle / "inference_manifest.json").is_file()
    ]
    if missing:
        pytest.skip(f"model bundles not installed under models/: {missing}")
    return load_robot_config_dict(str(ROBOTS / f"{name}.yaml"), defer_interface_binding=True)


@pytest.fixture(params=sorted(HOSTS), scope="module")
def host(request):
    return request.param, _host(request.param)


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_configuration_is_valid(merged):
    validate_robot_config_dict(merged, deferred_interfaces=True)


def test_the_grasp_topics_bind_against_the_omnihand_runtime(merged, profile):
    from robot_runtime.interface_description import build_description, description_digest
    from robot_runtime.profile import load_profile

    assert "joint.arm_stream" in required_interface_ids(merged)
    with pytest.raises(InterfaceBindingError, match="description_required"):
        load_robot_config_dict(str(CONFIG))
    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    bound = bind_robot_interfaces(merged, description)
    assert bound["runtime"]["interface_description"]["interfaces"]["joint.arm_stream"]["endpoint"]
    del description["interfaces"]["joint.hand_stream"]
    description.pop("digest")
    description["digest"] = description_digest(description)
    with pytest.raises(InterfaceBindingError, match="unknown_interface.*joint.hand_stream"):
        bind_robot_interfaces(merged, description)


def test_it_selects_the_omnihand_runtime_profile(merged, profile):
    assert merged["runtime"]["provider"] == "aimdk_robot"
    assert merged["runtime"]["profile"] == "x2_ultra_omnihand"
    assert merged["type"] == profile["runtime"]["type"] == "x2_ultra_omnihand"


def test_agent_deployment_keeps_the_grasp_executor_and_robot_identity():
    base = _host("aimdk_x2_grasp_310p")
    config = load_robot_config_dict(str(ROBOTS / "aimdk_x2_grasp_310p_agent.yaml"), defer_interface_binding=True)
    validate_robot_config_dict(config, deferred_interfaces=True)
    assert config["name"] == base["name"]
    assert config["runtime"] == base["runtime"]
    assert config["grasp_execution"] == base["grasp_execution"]
    assert config["perception_services"] == base["perception_services"]
    assert config["embodied"]["skill_catalog_profile"] == base["embodied"]["skill_catalog_profile"]
    assert config["embodied"]["entry_mode"] == "agent"
    agent = config["embodied"]["agent"]
    assert agent["enabled"] and agent["incubation"] and agent["execution_enabled"]
    assert agent["test_allowlist"] == ["pick_object"]
    assert agent["planner"]["mode"] == "vlm" and agent["planner"]["api_protocol"] == "chat_completions"
    assert agent["voice_bridge"]["enabled"] is False
    for key in ("ledger_path", "conversation_path", "deployment_lock_path"):
        assert Path(agent[key]).is_absolute() and not agent[key].startswith("/tmp/")


def test_arm_limits_match_the_mirror_until_the_configurations_merge(merged):
    # Two copies of the motion-control limits exist until the grasp and mirror
    # configurations become one; a change to either must reach the other.
    # The mirror's limits sit in its own file, so it is read without binding the
    # runtime interface description its full load requires.
    mirror = yaml.safe_load((ROBOTS / "aimdk_x2_magicmirror.yaml").read_text(encoding="utf-8"))["robot"]
    assert merged["arm_limits"] == mirror["motion_imitation_stream"]["mock_playback"]["arm_limits"]


def test_required_capabilities_are_all_provided_by_the_omnihand_profile(merged, profile):
    required = set(merged["capabilities"]["requires"])
    assert required <= all_capabilities()
    missing = sorted(required - set(profile["capabilities"]))
    assert missing == [], f"deployment requires capabilities the OmniHand runtime does not declare: {missing}"
    assert {"hand.multi_joint", "hand.gesture"} <= required
    assert "gripper.1d" not in required


def test_the_shared_grasp_names_no_models_and_runs_the_pipeline_beside_them_with_its_head_on_the_orin(merged):
    grasp = merged["grasp_execution"]
    options = grasp["backend_options"]
    assert grasp["enabled"] and grasp["backend"] == "aimdk_grasp.remote_head:x2_remote_grasp_backends"
    # No accelerator is chosen here; the aimdk_grasp backends refuse it on its own.
    assert not (merged.get("perception_services") or {}).get("services")
    assert "model_deployment" not in grasp and "segment_service" not in options
    assert grasp["model_bundle_path"] == "$(env WORKSPACE)/models/graspgenx_omnihand_left_x2"
    # The split is the X2's own (aimdk_grasp.split): nothing of it in the shared schema.
    assert "planning_host" not in grasp
    assert options["head"]["cpu_isolation"] == {"node": "4-7", "others": "1-3"}
    assert options["watch"] == "stop" and options["head"]["status_timeout_s"] == 1.0
    assert (options["watch_frame_timeout_s"], options["watch_unknown_timeout_s"]) == (1.0, 4.0)
    assert grasp["pipeline"]["release_after_success"] is True


def test_each_host_deployment_serves_its_models_on_its_own_accelerator(host, monkeypatch):
    from robot_config.perception_runtime_config import parse_perception_runtime_config

    name, config = host
    validate_robot_config_dict(config, deferred_interfaces=True)
    _, deployments = HOSTS[name]
    services = config["perception_services"]["services"]
    assert {service["deployment"] for service in services} == deployments
    monkeypatch.setenv("WORKSPACE", os.fspath(WORKSPACE))
    runtime = parse_perception_runtime_config(config)
    assert {service.instance_id for service in runtime.enabled_services} == {service["id"] for service in services}
    # Everything but the models is the shared deployment's.
    shared = load_robot_config_dict(str(CONFIG), defer_interface_binding=True)
    assert config["name"] == shared["name"] and config["embodied"] == shared["embodied"]
    for key in ("backend", "command_interfaces", "model_bundle_path", "pipeline"):
        assert config["grasp_execution"][key] == shared["grasp_execution"][key]


def test_the_executor_identity_names_the_deployment_the_grasp_service_runs(host):
    name, config = host
    grasp = config["grasp_execution"]
    services = {service["endpoint"]: service for service in config["perception_services"]["services"]}
    options = grasp["backend_options"]
    generation = services[options["generate_grasps_service"]]
    assert grasp["model_bundle_path"] == "$(env WORKSPACE)/" + generation["bundle_path"]
    assert grasp["model_deployment"] == generation["deployment"]
    assert options["detect_service"] in services
    detector = services[options["detect_service"]]
    if options["segment_service"]:
        # A detector without masks (GroundingDINO) needs the segmentation service it is paired with.
        assert services[options["segment_service"]]["service_type"] == "ibrobot_msgs/srv/SegmentDetections"
        assert detector["adapter_class"].endswith(":GroundingDINORawDetectPlugin")
    else:
        assert detector["adapter_class"].endswith(":GroundingDetectPlugin")  # grounded SAM2 returns masks


def test_the_cuda_host_runs_grounded_sam2_and_the_310p_host_the_ascend_pair_and_graspgenx_on_the_npu():
    cuda, ascend = _host("aimdk_x2_grasp_cuda"), _host("aimdk_x2_grasp_310p")
    assert [service["bundle_path"] for service in cuda["perception_services"]["services"]] == [
        "models/grounded_sam2_swint_ogc",
        "models/graspgenx_omnihand_left_x2",
    ]
    assert cuda["grasp_execution"]["model_deployment"] == "torch_cuda"
    assert [(s["bundle_path"], s["deployment"]) for s in ascend["perception_services"]["services"]] == [
        ("models/grounding_dino_swint_seq8_1280x720", "ascend_310p"),
        ("models/sam2.1_hiera_tiny_prompt_ascend", "ascend_310p"),
        ("models/graspgenx_omnihand_left_x2", "torch_npu"),
    ]
    assert ascend["grasp_execution"]["model_deployment"] == "torch_npu"
    assert ascend["grasp_execution"]["backend_options"]["segment_service"] == "/perception/grasp/segment_detections"


def test_pick_object_switches_the_runtime_to_the_policy_stream(merged):
    assert merged["skill_required_control_mode"] == "moveit_planning"
    assert merged["control_modes"]["moveit_planning"]["runtime_mode"] == "policy_stream"
    assert merged["embodied"]["enabled"] and merged["embodied"]["skill_catalog_profile"] == "aimdk_x2_grasp"
