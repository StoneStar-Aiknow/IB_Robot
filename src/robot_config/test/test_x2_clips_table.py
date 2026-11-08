"""The X2 mirror clip table: 61 prerecorded lines, text only, installed with robot_config.

The WAVs are recordings of these exact texts, so the table is generated from
the release's clips.tsv and never edited by hand. What the mirror relies on is
checked here: the contract reads it without a single warning, every event and
the standby words have a line, every pool the look maps to exists, and nothing
the mirror says by itself can be heard as a command — a look line or a standby
word containing a hotword or a voice-triggerable skill alias would start that
skill whenever the ASR is not muted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from embodied_common import internvl_voice_contract as contract
from robot_config.loader import _load_robot_section

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
TABLE = CONFIG_DIR / "mirror_voice" / "x2_clips_v1.yaml"
HOTWORDS = CONFIG_DIR / "hotwords" / "x2_commands.txt"
ENTRY = CONFIG_DIR / "robots" / "aimdk_x2_voice_mirror_all.yaml"
# src/robot_config/test/ -> src/
SKILLS = Path(__file__).resolve().parents[2] / "agent" / "skill_catalog" / "config" / "skills"

# Every pool the look maps to, selects from or falls back on (plan §4.2–4.4).
LOOK_POOLS = (
    "hand.heart",
    "hand.thumb",
    "hand.yeah",
    "hand.any",
    "big.wave",
    "big.open",
    "big.any",
    "pose.any",
    "dance.any",
    "other.any",
    "seq.repeat",
    "seq.still",
    "seq.switch",
    "seq.invite",
)

# The only lines allowed to contain a command phrase, and the phrase each may
# contain: both answer a wave and name it (wave_hand aliases 招手 / 打招呼).
# zll approved these lines and their audio. They are safe only because the ASR
# is deaf while a clip plays — internvl_voice.asr ``mute`` or ``off``; with
# ``keep`` they would start wave_hand, which the test below refuses.
ALLOWED_TRIGGER_LINES = {
    "big_wave_02": {"招手"},
    "big_wave_03": {"打招呼"},
}
ASR_MODES_DEAF_DURING_PLAYBACK = ("mute", "off")


@pytest.fixture(scope="module")
def parsed():
    return contract.parse_clip_table(yaml.safe_load(TABLE.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def table(parsed):
    return parsed[0]


def _triggers() -> dict[str, str]:
    """Phrase -> where it comes from: hotwords, allowlisted skill aliases, ASR corrections."""
    agent = ROBOT["embodied"]["agent"]
    triggers = {}
    for line in HOTWORDS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            triggers[line.split()[0]] = "hotword"
    for skill in agent["test_allowlist"]:
        manifest = yaml.safe_load((SKILLS / skill / "manifest.yaml").read_text(encoding="utf-8"))
        triggers[skill] = f"skill {skill}"
        for alias in manifest["description"].get("aliases_zh", []):
            triggers.setdefault(alias, f"skill {skill}")
    # A heard mishearing is rewritten to its command before the planner sees it.
    for canonical, variants in agent["voice_bridge"]["text_corrections"].items():
        for variant in variants:
            triggers.setdefault(variant, f"correction to {canonical}")
    return triggers


ROBOT = _load_robot_section(ENTRY)[1]
TRIGGERS = _triggers()


def _spoken_unprompted():
    """Ids of the lines the mirror says by itself: look lines and standby words."""
    table, _ = contract.parse_clip_table(yaml.safe_load(TABLE.read_text(encoding="utf-8")))
    return [clip_id for clip_id in table.clips if not table.pool_of(clip_id).startswith("evt.")]


def test_it_parses_without_a_warning(parsed):
    assert parsed[1] == []


def test_it_is_version_1_of_61_lines(table):
    assert (table.version, table.wav_suffix, len(table.clips)) == (1, "_v1.wav", 61)


@pytest.mark.parametrize("pool", contract.REQUIRED_POOLS)
def test_every_event_and_the_standby_words_have_a_line(table, pool):
    assert table.pools().get(pool)


@pytest.mark.parametrize("pool", LOOK_POOLS)
def test_every_pool_the_look_uses_exists(table, pool):
    assert table.pools().get(pool)


def test_there_are_no_pools_beyond_those(table):
    assert set(table.pools()) == {*contract.REQUIRED_POOLS, *LOOK_POOLS}


def test_the_triggers_were_found():
    """An empty trigger list would pass the check below vacuously."""
    sources = set(TRIGGERS.values())
    assert "hotword" in sources
    assert "skill wave_hand" in sources
    assert "停止照镜子" in TRIGGERS


@pytest.mark.parametrize("clip_id", _spoken_unprompted())
def test_no_unprompted_line_contains_a_command(table, clip_id):
    text = table.clips[clip_id]
    hits = {phrase: source for phrase, source in TRIGGERS.items() if phrase in text}
    allowed = ALLOWED_TRIGGER_LINES.get(clip_id, set())

    unexpected = {phrase: source for phrase, source in hits.items() if phrase not in allowed}
    assert unexpected == {}, f"{clip_id} {text!r} contains {unexpected}"


@pytest.mark.parametrize("clip_id", sorted(ALLOWED_TRIGGER_LINES))
def test_each_allowed_line_still_needs_its_allowance(table, clip_id):
    """A stale entry would quietly allow a command phrase into a re-recorded line."""
    text = table.clips[clip_id]

    assert all(phrase in text and phrase in TRIGGERS for phrase in ALLOWED_TRIGGER_LINES[clip_id])


def test_the_allowed_lines_are_spoken_with_the_asr_deaf():
    """The allowance holds only while the deployed entry keeps the ASR off during playback."""
    settings, _ = contract.normalize(ROBOT["internvl_voice"])

    assert settings.asr in ASR_MODES_DEAF_DURING_PLAYBACK
