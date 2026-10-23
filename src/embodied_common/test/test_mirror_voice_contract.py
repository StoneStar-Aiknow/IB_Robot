"""The mirror voice contract: the section its builder and its node both read.

Covers the keys this feature added to a colleague's module — the utterance
topic and the ASCII-letter rule shared by the launch-time phrase check and the
run-time utterance filter — plus the module's own rule that it stay
dependency-free, which the launch builder's import budget leans on.
"""

import subprocess
import sys

import pytest

from embodied_common import mirror_voice_contract as contract


def test_the_module_stays_dependency_free():
    """Importing it must not pull in ROS, a launch builder or a sibling package.

    The mirror session's launch builder imports it while building parameters,
    and the voice node imports it at startup; anything it drags in lands in
    both import budgets.
    """
    script = (
        "import sys; import embodied_common.mirror_voice_contract; "
        "leaked = sorted("
        "  name for name in sys.modules"
        "  if name.split('.')[0] in {'rclpy', 'launch', 'robot_config', 'ibrobot_msgs', 'manipulation_execution'}"
        ");"
        "assert not leaked, leaked"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


class TestUtteranceTopic:
    def test_the_default_is_today_s_behaviour(self):
        """A section without the key speaks transition phrases only."""
        assert contract.parse_voice_section({"enabled": True}).utterance_topic == ""

    def test_an_absolute_name_is_kept(self):
        section = contract.parse_voice_section({"enabled": True, "utterance_topic": "/visual_interaction/utterance"})

        assert section.utterance_topic == "/visual_interaction/utterance"

    @pytest.mark.parametrize("value", ["visual_interaction/utterance", "utterance", 7, ["/x"]])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(ValueError, match=r"utterance_topic: must be empty or an absolute ROS name"):
            contract.parse_voice_section({"utterance_topic": value})

    def test_it_survives_the_json_round_trip(self):
        """The node reads the section as the one JSON parameter the builder sends."""
        section = contract.parse_voice_section({"utterance_topic": "/visual_interaction/utterance"})

        assert contract.from_json(contract.to_json(section)) == section

    def test_the_round_trip_of_the_default_stays_empty(self):
        section = contract.parse_voice_section({"enabled": True})

        assert contract.from_json(contract.to_json(section)) == section


class TestAsciiLetters:
    """One rule in two places: phrases are checked at launch, utterances at run time."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("你举起的玩偶真有趣。", False),
            ("启动中", False),
            ("3 号展台的帽子", False),
            ("Hello", True),
            ("这个 toy 很有趣", True),
            ("", False),
        ],
    )
    def test_it_matches_exactly_the_letters(self, text, expected):
        assert contract.has_ascii_letters(text) is expected

    def test_it_agrees_with_the_phrase_check(self):
        """A phrase ZipVoice cannot speak is refused at launch by the same rule
        the node applies to a run-time utterance; the two must not drift."""
        with pytest.raises(ValueError, match="ASCII letters"):
            contract.parse_voice_section({"phrases": {"ready": ["OK"]}})

        assert contract.has_ascii_letters("OK")
