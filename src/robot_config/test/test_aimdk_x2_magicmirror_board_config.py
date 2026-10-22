"""The X2 magic-mirror board overlay: the combined entry with the real VL.

The overlay is the only place a machine-specific literal is allowed to live
(the demo root the vendor ships on the board). Everything else must arrive
through inheritance untouched — especially the wiring the combined entry
exists for. Loaded through the inheritance merge for the same reason as
test_aimdk_x2_magicmirror_config.py: the base entry's Ascend bundles are
deployment material a checkout does not carry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import _load_robot_section

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG = ROBOTS / "aimdk_x2_magicmirror_board.yaml"
# src/robot_config/test/ -> src/
PROFILE = (
    Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
)


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["robot"]


@pytest.fixture(scope="module")
def config():
    return _load_robot_section(CONFIG)[1]


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_it_overlays_the_workstation_entry(raw):
    assert raw["base_config"] == "aimdk_x2_magicmirror.yaml"
    assert raw["name"] == "aimdk_x2_magicmirror_board"


class TestTheOverlay:
    def test_only_the_vl_is_swapped(self, config):
        """The one intended difference; an absolute root, as the builder checks."""
        assert config["visual_interaction"]["vl"] == {
            "implementation": "internvl",
            "demo_root": "/root/vlx_internvl_310p",
        }

    def test_everything_else_is_inherited(self, config):
        """The wiring the combined entry exists for survives the overlay."""
        frame_topic = config["motion_imitation_stream"]["frame_source"]["topic"]
        assert config["visual_interaction"]["actor_state_topic"] == "/motion_imitation_stream/actor_state"
        assert config["visual_interaction"]["mirror_state_topic"] == "/mirror_session/state"
        assert config["mirror_session"]["voice"]["utterance_topic"] == "/visual_interaction/utterance"
        assert config["visual_interaction"]["frame_source"]["topic"] == frame_topic
        for name in ("teleop", "base_navigation"):
            assert config["control_modes"][name]["executor"]["enabled"] is False, name

    def test_the_builder_accepts_the_board_config(self, config, profile):
        """The internvl path passes the same gate the fake one does — the demo
        root is absolute and the wiring keys are inherited."""
        import copy

        from visual_interaction.launch_builders.visual_interaction import generate_visual_interaction_nodes

        bound = copy.deepcopy(config)
        bound["runtime"]["interface_description"] = {
            "robot": {
                "id": "x2",
                "type": profile["runtime"]["type"],
                "runtime_name": profile["runtime"]["name"],
                "runtime_version": profile["runtime"]["version"],
            },
            "interfaces": {},
        }

        assert len(generate_visual_interaction_nodes(bound)) == 1
