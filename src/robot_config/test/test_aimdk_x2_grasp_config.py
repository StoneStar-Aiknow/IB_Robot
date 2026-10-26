"""The X2 grasp deployment binds the OmniHand runtime and serves the grasp models.

Pure configuration checks: no ROS graph, no vendor SDK, no GPU. The host
deployments load through the canonical loader, which runs the complete schema;
bundle validation needs the model bundles and skips when they are not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config, load_robot_config_dict, validate_config
from robot_runtime.capabilities import all_capabilities

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG = ROBOTS / "aimdk_x2_grasp.yaml"
# src/robot_config/test/ -> src/
PROFILE = (
    Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
)
WORKSPACE = Path(__file__).resolve().parents[3]
# Loading a deployment resolves its perception bundles: without them every test
# here would error in its fixture instead of skipping. The 310P deployment adds
# the separate detection and segmentation bundles it serves.
pytestmark = pytest.mark.model_bundle(
    "grounded_sam2_swint_ogc",
    "graspgenx_omnihand_left_x2",
    "grounding_dino_swint_seq8_1280x720",
    "sam2.1_hiera_tiny_prompt_ascend",
)


@pytest.fixture(scope="module")
def merged():
    return load_robot_config_dict(str(CONFIG))


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_configuration_is_valid():
    assert validate_config(load_robot_config(str(CONFIG))) == []


def test_it_selects_the_omnihand_runtime_profile(merged, profile):
    assert merged["runtime"]["provider"] == "aimdk_robot"
    assert merged["runtime"]["profile"] == "x2_ultra_omnihand"
    assert merged["type"] == profile["runtime"]["type"] == "x2_ultra_omnihand"


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


#: What each host deployment serves, and the deployment its executor names.
DEPLOYMENTS = {
    "aimdk_x2_grasp_cuda": {
        "services": {"grasp_grounding", "grasp_generation"},
        "model_deployment": "torch_cuda",
    },
    "aimdk_x2_grasp_310p": {
        "services": {"grasp_grounding", "grasp_segmentation", "grasp_generation"},
        "model_deployment": "torch_npu",
    },
}


@pytest.mark.parametrize("deployment", tuple(DEPLOYMENTS))
def test_the_host_deployments_load_and_validate_whole(monkeypatch, deployment):
    """Each host deployment loads through the canonical loader, whole, as on the robot.

    The whole file must load, not just the perception block:
    load_robot_config_dict runs the complete schema, so a grasp_execution key
    the schema does not know fails here, not when the host launches the
    runtime and the model services. The executor's backend options wait for
    the backend that consumes them; until then these files declare only what
    the schema and the launch can stand behind.
    """
    from robot_config.perception_runtime_config import parse_perception_runtime_config

    monkeypatch.setenv("WORKSPACE", os.fspath(WORKSPACE))
    config = load_robot_config_dict(str(ROBOTS / f"{deployment}.yaml"))
    runtime = parse_perception_runtime_config(config)
    expected = DEPLOYMENTS[deployment]
    assert {service.instance_id for service in runtime.enabled_services} == expected["services"]
    for service in runtime.enabled_services:
        assert service.bundle_path.is_dir(), f"{service.instance_id}: {service.bundle_path}"
    assert config["grasp_execution"]["model_deployment"] == expected["model_deployment"]


def test_the_shared_configuration_serves_no_model_itself(merged):
    """Launching the shared file alone must not look like a host that serves the models."""
    assert not ((merged.get("perception_services") or {}).get("services") or [])
