"""The X2 magic-mirror deployment: the mirror and the visual interaction in one launch.

Closes the gap this package's own visual-interaction test pointed at: no
test_aimdk_x2_magicmirror_config.py existed. The combined entry adds a
visual_interaction section and an utterance topic for the voice beside the
inherited mirror stack, and switches off the two driving modes the skills
base leaves on — this file pins all three, plus the topic wiring that makes
the two features one deployment rather than two that happen to start together.

Full canonical loading cannot run here: the entry declares Ascend perception
bundles under $WORKSPACE/models, deployment material a checkout does not
carry, and validate_robot_config_dict resolves them strictly. The section is
loaded through the inheritance merge only (the seam test_aero_hand_config.py
uses), which is the layer the new wiring lives at; bundle resolution stays
covered by test_perception_runtime_config.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from launch_ros.actions import Node

from robot_config.loader import _load_robot_section

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG = ROBOTS / "aimdk_x2_magicmirror.yaml"
# src/robot_config/test/ -> src/
PROFILE = (
    Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
)


@pytest.fixture(scope="module")
def raw():
    """The file as written, before inheritance is resolved."""
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["robot"]


@pytest.fixture(scope="module")
def config():
    # base_config chains resolved, nothing else run — see the module docstring.
    return _load_robot_section(CONFIG)[1]


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def bound(config, profile):
    """The config with the description interface_binding writes at launch.

    Built by the same function the runtime builds its description with, from
    the real profile: the mirror builder reads ``joint.arm_stream`` (joint
    order, modes) out of it for the mock reset pose, so a hand-written minimal
    description would not exercise the shipped wiring.
    """
    import copy

    from robot_runtime.interface_description import build_description

    bound = copy.deepcopy(config)
    bound["runtime"]["interface_description"] = build_description(profile)
    return bound


def test_it_inherits_the_skills_deployment(raw):
    assert raw["base_config"] == "aimdk_x2_skills.yaml"
    assert raw["name"] == "aimdk_x2_magicmirror"


class TestTheCombinedEntry:
    def test_the_mirror_s_own_mode_stays_its_sole_producer(self, config):
        """The observation gate the visual interaction builder runs must not be
        read as "no motion anywhere": the mirror keeps its mode, declared the
        way the mirror builder's sole-producer check requires."""
        mode = config["control_modes"]["model_inference"]
        assert mode["runtime_mode"] == "policy_stream"
        assert mode["inference"]["enabled"] is False
        assert mode["executor"]["enabled"] is False
        assert config["skill_required_control_mode"] == "model_inference"

    @pytest.mark.parametrize("producer", ["inference", "executor"])
    def test_no_control_mode_launches_a_motion_producer(self, config, producer):
        """A missing enabled key counts as enabled, as execution.py reads it.

        teleop and base_navigation are inherited through the skills config and
        both drive the robot; the mirror demo uses neither, so each is switched
        off here — and the visual interaction's observation gate would refuse
        the config otherwise.
        """
        for name, mode in config["control_modes"].items():
            section = mode.get(producer)
            if section is None:
                continue
            assert section.get("enabled", True) is False, f"{name}.{producer}"

    def test_the_visual_interaction_builder_accepts_this_config(self, config, profile):
        """The gate that would refuse a motion producer must pass on the real file."""
        from visual_interaction.launch_builders.visual_interaction import generate_visual_interaction_nodes

        assert len(generate_visual_interaction_nodes(bound(config, profile))) == 1

    def test_the_mirror_builder_still_builds_its_whole_stack(self, config, profile):
        """The combined entry is an extension of the mirror entry, not a
        replacement: session, its release handler, the operator and the voice
        all keep launching."""
        from manipulation_execution.launch_builders.mirror_session import generate_mirror_session_nodes

        actions = generate_mirror_session_nodes(bound(config, profile))

        executables = [action._Node__node_executable for action in actions if isinstance(action, Node)]
        assert executables == ["mirror_session_node", "mirror_operator_node", "mirror_voice_node"]
        # Private accessors: the public properties only resolve after launch
        # executes the action (same pattern as the builder tests).
        assert len(actions) == 4  # plus the process-exit handler running the release node


class TestTheWiring:
    """The topics that make this one deployment: the interaction's rounds,
    gate and replies all land on endpoints the mirror side already owns."""

    def test_the_actor_state_comes_from_the_imitation_stream(self, config):
        """The same PEAR pipeline the arms follow decides who the rounds are
        about; spelling a second topic here is how the two drift apart."""
        assert (
            config["visual_interaction"]["actor_state_topic"] == config["motion_imitation_stream"]["actor_state_topic"]
        )

    def test_the_gate_reads_the_session_state_topic(self, config):
        from embodied_common.mirror_session_contract import STATE_TOPIC

        assert config["visual_interaction"]["mirror_state_topic"] == STATE_TOPIC

    def test_the_voice_consumes_the_interaction_s_topic(self, config):
        from embodied_common.visual_interaction_contract import UTTERANCE_TOPIC

        assert config["mirror_session"]["voice"]["utterance_topic"] == UTTERANCE_TOPIC

    def test_it_reads_the_same_half_resolution_stream(self, config):
        """The relay topic the imitation stream reads, not a second one; the
        stream itself was deliberately left untouched."""
        assert (
            config["visual_interaction"]["frame_source"]["topic"]
            == config["motion_imitation_stream"]["frame_source"]["topic"]
        )

    def test_the_frame_source_qos_is_stated(self, config):
        """best_effort over volatile, the camera's own pairing for a live feed."""
        qos = config["visual_interaction"]["frame_source"]["qos"]

        assert qos["reliability"] == "best_effort"
        assert qos["durability"] == "volatile"

    def test_the_operator_reads_the_rear_head_camera(self, config):
        """Covering the front centre camera trips its auto gain and brightens
        the frame again, so the gestures read the rear head camera instead."""
        operator = config["mirror_session"]["operator"]

        assert operator["camera_topic"] == "/aima/hal/sensor/rgb_head_rear/rgb_image/compressed"


class TestTheSection:
    def test_the_vl_is_the_scripted_one_here(self, config):
        """The board overlay swaps in internvl; this file must stay launchable
        on a workstation."""
        assert config["visual_interaction"]["vl"]["implementation"] == "fake"

    def test_the_categories_are_the_whole_whitelist(self, config):
        from embodied_common.visual_interaction_contract import CATEGORIES

        assert tuple(config["visual_interaction"]["categories"]) == CATEGORIES

    def test_the_session_section_is_carried_over(self, config):
        """The values the standalone visual interaction entry ships with; the
        combined entry must not quietly loosen one."""
        section = config["visual_interaction"]

        assert section["session"]["confirmation_count"] >= 2
        assert section["session"]["max_session_sec"] > section["session"]["absent_reset_sec"]
        assert section["session"]["max_evidence_age_sec"] > (
            section["session"]["confirmation_count"] * section["round_period_sec"]
        )
