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
import subprocess
from pathlib import Path

import pytest
import yaml

from embodied_common import internvl_voice_contract as contract
from robot_config.loader import _load_robot_section

BASELINE = "676fa608"
PACKAGE = Path(__file__).resolve().parents[1]
ROBOTS = PACKAGE / "config" / "robots"
ENTRY = ROBOTS / "aimdk_x2_voice_mirror_all.yaml"
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
        """The defaults stand in for a bad item, so they must be what is deployed."""
        assert contract.normalize(config["internvl_voice"])[0] == contract.Settings()

    def test_its_table_is_the_installed_one(self, config):
        """``$(find robot_config)`` against the source tree, laid out as the share dir is."""
        default = str(PACKAGE / "config" / "mirror_voice" / "x2_clips_v1.yaml")

        bundle = contract.build_bundle(config["internvl_voice"], "/root/tree", default, lambda package: str(PACKAGE))

        assert bundle.warnings == ()
        assert bundle.settings.clips_table == default
        assert len(bundle.table.clips) == 61
        assert bundle.clip_dirs == ("/root/tree/assets/audio/x2_mirror_clips_v1",)


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

    def test_the_entry_without_the_block_is_the_baseline_file(self, text):
        """Optional: holds only until another feature edits this entry; skips without the commit."""
        repo = subprocess.run(
            ["git", "-C", str(PACKAGE), "rev-parse", "--show-toplevel"], capture_output=True, check=False, text=True
        )
        if repo.returncode != 0:
            pytest.skip("not a git checkout")
        path = ENTRY.relative_to(Path(repo.stdout.strip())).as_posix()
        baseline = subprocess.run(
            ["git", "-C", str(PACKAGE), "show", f"{BASELINE}:{path}"], capture_output=True, check=False, text=True
        )
        if baseline.returncode != 0:
            pytest.skip(f"{BASELINE} is not reachable")

        assert _without_block(text) == baseline.stdout
