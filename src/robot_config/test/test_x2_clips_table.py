"""The X2 mirror clip tables: v1 (61 lines) and v2 (v1 plus 3 for open arms), text only.

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
# version -> (wav suffix, lines); v2 is the one the entry ships.
TABLES = {1: ("_v1.wav", 61), 2: ("_v2.wav", 64)}
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
ASR_MODES_DEAF_DURING_PLAYBACK = ("mute", "off", "session")


def _path(version: int) -> Path:
    return CONFIG_DIR / "mirror_voice" / f"x2_clips_v{version}.yaml"


@pytest.fixture(scope="module", params=sorted(TABLES), ids=lambda version: f"v{version}")
def parsed(request):
    return contract.parse_clip_table(yaml.safe_load(_path(request.param).read_text(encoding="utf-8")))


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


def _read(version: int):
    return contract.parse_clip_table(yaml.safe_load(_path(version).read_text(encoding="utf-8")))[0]


# v2 holds every v1 line with the same text (checked below), so checking v2's lines checks both.
SHIPPED = _read(max(TABLES))


def _spoken_unprompted():
    """Ids of the lines the mirror says by itself: look lines and standby words."""
    return [clip_id for clip_id in SHIPPED.clips if not SHIPPED.pool_of(clip_id).startswith("evt.")]


def test_it_parses_without_a_warning(parsed):
    assert parsed[1] == []


def test_its_version_suffix_and_size(table):
    assert (table.wav_suffix, len(table.clips)) == TABLES[table.version]


def test_v2_is_v1_plus_the_open_arms_lines():
    """The 61 v1 lines are the same recordings under v2's names, so their text must not move."""
    v1, v2 = _read(1), _read(2)

    assert {k: v for k, v in v2.clips.items() if k in v1.clips} == dict(v1.clips)
    assert sorted(set(v2.clips) - set(v1.clips)) == ["big_open_02", "big_open_03", "big_open_04"]


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
def test_no_unprompted_line_contains_a_command(clip_id):
    text = SHIPPED.clips[clip_id]
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
