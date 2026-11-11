"""The voice_mirror_all entry turns internvl_voice on and changes nothing else.

The feature's only edit to an existing config is one marked block in the
entry. The loader overwrites a repeated key silently, so a block that also
wrote ``mirror_session:`` or ``visual_interaction:`` would replace the
inherited section without a sound. The check here is that the entry with the
block cut out merges to exactly the entry with it, minus ``internvl_voice``,
and that the block itself is read by the contract without a single warning —
a warning means some item fell back to its default. Comparing against the
file's own text, not a commit, keeps the check true when other features'
blocks are merged in around ours.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from embodied_common import internvl_voice_contract as contract
from robot_config.loader import _load_robot_section

PACKAGE = Path(__file__).resolve().parents[1]
ROBOTS = PACKAGE / "config" / "robots"
ENTRY = ROBOTS / "aimdk_x2_voice_mirror_all.yaml"
GRASP = ROBOTS / "aimdk_x2_voice_mirror_grasp.yaml"
BLOCK_START = "  # feat/x2-internvl-voice\n"
BLOCK_END = "  # /feat/x2-internvl-voice\n"


@pytest.fixture(scope="module")
def text():
    return ENTRY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def config():
    """The entry merged through its base_config chain, as the launch loader merges it.

    The merge only: the full loader also requires the Ascend bundles under
    $WORKSPACE/models, deployment material a checkout does not carry (same
    reason as test_aimdk_x2_magicmirror_board_config.py).
    """
    return _load_robot_section(ENTRY)[1]


def _without_block(text: str) -> str:
    """The entry with our marked block, and the blank line that sets it off, taken out."""
    start = text.index(BLOCK_START)
    end = text.index(BLOCK_END, start) + len(BLOCK_END)
    if text[:start].endswith("\n\n"):
        start -= 1
    return text[:start] + text[end:]


def _merged(text: str, tmp_path: Path) -> dict:
    """``text`` as the entry, merged with the real base_config chain next to it."""
    robots = tmp_path / "robots"
    shutil.copytree(ROBOTS, robots)
    (robots / ENTRY.name).write_text(text, encoding="utf-8")
    return _load_robot_section(robots / ENTRY.name)[1]


class TestTheBlock:
    def test_it_is_there_exactly_once_between_its_markers(self, text):
        assert text.count(BLOCK_START) == 1
        assert text.count(BLOCK_END) == 1
        assert text.index(BLOCK_START) < text.index(BLOCK_END)
        assert text.count("\n  internvl_voice:") == 1

    def test_it_adds_only_its_own_key(self, text):
        start = text.index(BLOCK_START)
        block = yaml.safe_load(text[start : text.index(BLOCK_END, start)])

        assert list(block) == ["internvl_voice"]
        assert yaml.safe_load(text)["robot"]["internvl_voice"] == block["internvl_voice"]

    def test_it_normalizes_without_a_warning(self, config):
        settings, warnings = contract.normalize(config["internvl_voice"])

        assert warnings == []
        assert settings is not None
        assert settings.look is not None

    def test_it_is_written_at_the_contract_s_defaults(self, config):
        """The defaults stand in for a bad item, so they must be what is deployed.

        Except asr: feat/x2-retarget-tune deploys "session", the mode that stopped the
        STANDBY/MIRRORING stalls on the robot. And zll (11-10) turned the key actions on,
        with the v2 table that has their lines.
        """
        settings = contract.normalize(config["internvl_voice"])[0]
        defaults = contract.Settings()
        look = replace(defaults.look, key_actions=replace(defaults.look.key_actions, enabled=True))

        assert settings == replace(
            defaults,
            asr="session",
            clips_table="$(find robot_config)/config/mirror_voice/x2_clips_v2.yaml",
            clip_dir="assets/audio/x2_mirror_clips_v2",
            look=look,
        )

    def test_its_table_is_the_installed_one(self, config):
        """``$(find robot_config)`` against the source tree, laid out as the share dir is."""
        shipped = str(PACKAGE / "config" / "mirror_voice" / "x2_clips_v2.yaml")
        default = str(PACKAGE / "config" / "mirror_voice" / "x2_clips_v1.yaml")

        bundle = contract.build_bundle(config["internvl_voice"], "/root/tree", default, lambda package: str(PACKAGE))

        assert bundle.warnings == ()
        assert bundle.settings.clips_table == shipped
        assert len(bundle.table.clips) == 64
        assert bundle.clip_dirs == (
            "/root/tree/assets/audio/x2_mirror_clips_v2",
            "/root/tree/assets/audio/x2_mirror_clips_v1",
        )


class TestNothingElseChanges:
    def test_the_entry_without_the_block_merges_to_the_same_config(self, text, config, tmp_path):
        without = _merged(_without_block(text), tmp_path)
        current = dict(config)
        block = current.pop("internvl_voice")

        assert block
        assert "internvl_voice" not in without
        for section in ("mirror_session", "visual_interaction", "voice_asr", "cbf_safety"):
            assert current[section] == without[section], section
        assert current == without

    def test_nothing_of_ours_is_outside_the_block(self, text):
        """Only our block is checked: other features merged into this entry may change the rest."""
        outside = _without_block(text)

        assert "feat/x2-internvl-voice" not in outside
        assert "x2_clips_" not in outside
        assert not any(line.lstrip().startswith("internvl_voice:") for line in outside.splitlines())


class TestGraspProfile:
    def test_commissioned_grasp_entry_enables_mirror_voice(self):
        """The commissioned entry retains the mirror feedback enabled on the robot."""
        config = _load_robot_section(GRASP)[1]

        assert config["internvl_voice"]["enabled"] is True
        settings, warnings = contract.normalize(config["internvl_voice"])
        assert settings is not None and warnings == []
        assert settings.clips_table == "$(find robot_config)/config/mirror_voice/x2_clips_v2.yaml"
        assert config["mirror_session"]["voice"]["enabled"] is True
