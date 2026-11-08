"""The ``internvl_voice`` section of the X2 mirror: prerecorded clips and the InternVL look.

When the section is present and its ``enabled`` is not ``false``, the mirror
speaks prerecorded clips only — transition lines, standby words and a line
matched to what InternVL sees — and never synthesises. Left out, or with
``enabled: false``, everything is the 676fa608 baseline.

Two launch builders (mirror_session and visual_interaction, both in
robot_config) call ``build_bundle`` and hand their node one JSON parameter:
``playback_json`` for mirror_voice, ``look_json`` for visual_interaction. The
nodes read it back with ``parse_playback_json`` / ``parse_look_json`` and
depend only on the dict those return, not on ``Settings``.

Unlike the strict voice section (mirror_voice_contract), a wrong value here
never fails: each bad item warns and falls back to its own default, so a typo
cannot silence the mirror or bring back synthesis. Every warning starts with
the path of the bad value, e.g. ``internvl_voice.look.nice``. Only a broken
``look`` turns the look off (``Settings.look`` is ``None``); playback goes on.

Clip ids are ``<group>_<sub>_<NN>``; the pool of an id is ``<group>.<sub>``
(``evt_ready_01`` -> ``evt.ready``) and its WAV is ``<id><wav_suffix>``.
``evt.*`` pools are the four transition events, ``standby.any`` the standby
words, and every other pool is a look pool.

Keep it dependency-free (stdlib and PyYAML): both builders import it while
building parameters and both nodes import it at startup.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

import yaml

SECTION_PATH = "internvl_voice"
TABLE_PATH = "clips_table"

# The spoken events, in the order a round goes through them, and their pools.
EVENTS = ("starting", "ready", "mirroring", "goodbye")
EVENT_POOLS: Mapping[str, str] = MappingProxyType({event: f"evt.{event}" for event in EVENTS})
STANDBY_POOL = "standby.any"
# Pools a usable table must have at least one clip in.
REQUIRED_POOLS = (*EVENT_POOLS.values(), STANDBY_POOL)

ASR_MODES = ("mute", "off", "keep")

DEFAULT_CLIPS_TABLE = "$(find robot_config)/config/mirror_voice/x2_clips_v1.yaml"
DEFAULT_CLIP_DIR = "assets/audio/x2_mirror_clips_v1"
DEFAULT_DEMO_ROOT = "/root/vlx_internvl_310p"
DEFAULT_TABLE_VERSION = 1
DEFAULT_WAV_SUFFIX = "_v1.wav"

CLIP_ID_PATTERN = re.compile(r"^[a-z]+_[a-z]+_[0-9]{2}$")
# A clip is a recording of its text; an ASCII letter or digit has no fixed reading.
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
_FIND = re.compile(r"\$\(find\s+([A-Za-z0-9_]+)\)")

# No delay or interval in a 60 s session is longer than this.
_MAX_SEC = 60.0

Interval = tuple[float, float]


class _Bad(ValueError):
    """One item is wrong; the message says why, the caller adds the path."""


@dataclass(frozen=True)
class StandbySettings:
    first_after_sec: Interval = (1.0, 3.0)
    gap_sec: Interval = (2.0, 5.0)


@dataclass(frozen=True)
class LookSettings:
    # Always true here: a look that is off is ``Settings.look is None``.
    enabled: bool = True
    demo_root: str = DEFAULT_DEMO_ROOT
    # 0: start the next look as soon as the last one ends.
    round_period_sec: float = 0.0
    # 0: look all through MIRRORING; > 0: only this long before a line is due.
    window_sec: float = 0.0
    quick_gap_sec: float = 2.0
    nice: int = 10
    crop_actor: bool = True
    refine_hand: bool = True
    verify: bool = True
    first_after_sec: Interval = (2.0, 4.0)
    gap_sec: Interval = (3.0, 6.0)
    max_per_round: int = 5
    fresh_sec: float = 5.0
    pool_cooldown_sec: float = 15.0
    fallback_after_sec: float = 2.5
    latest_start_sec: float = 2.0
    dump_dir: str = ""


@dataclass(frozen=True)
class Settings:
    # Always true here: a section that is off normalizes to ``None``.
    enabled: bool = True
    clips_table: str = DEFAULT_CLIPS_TABLE
    clip_dir: str = DEFAULT_CLIP_DIR
    asr: str = "mute"
    asr_resume_delay_sec: float = 0.8
    asr_control_topic: str = "/voice_control"
    standby: StandbySettings = field(default_factory=StandbySettings)
    # None when the look is off or its section is broken.
    look: LookSettings | None = field(default_factory=LookSettings)


# ---------------------------------------------------------------------------
# Item checks: each returns the normalized value or raises _Bad.


def _seconds(lo: float = 0.0, hi: float = _MAX_SEC, *, positive: bool = False) -> Callable[[Any], float]:
    def check(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _Bad(f"must be a number, not {value!r}")
        value = float(value)
        if value != value or value < lo or value > hi or (positive and value <= 0):
            low = f"> {lo:g}" if positive else f">= {lo:g}"
            raise _Bad(f"{value:g} is out of range ({low} and <= {hi:g})")
        return value

    return check


def _integer(lo: int, hi: int) -> Callable[[Any], int]:
    def check(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise _Bad(f"must be an integer from {lo} to {hi}, not {value!r}")
        return value

    return check


def _interval(value: Any) -> Interval:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise _Bad(f"must be [low, high] in seconds, not {value!r}")
    lo, hi = (_seconds()(item) for item in value)
    if lo > hi:
        raise _Bad(f"low {lo:g} is above high {hi:g}")
    return (lo, hi)


def _flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise _Bad(f"must be true or false, not {value!r}")
    return value


def _text(*, allow_empty: bool = False) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if not isinstance(value, str):
            raise _Bad(f"must be a string, not {value!r}")
        value = value.strip()
        if not value and not allow_empty:
            raise _Bad("must not be empty")
        return value

    return check


def _choice(options: tuple[str, ...]) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if value not in options:
            raise _Bad(f"must be one of {list(options)}, not {value!r}")
        return value

    return check


def _topic(value: Any) -> str:
    value = _text()(value)
    if not value.startswith("/"):
        raise _Bad(f"must be an absolute ROS name, not {value!r}")
    return value


_TOP_CHECKS: Mapping[str, Callable[[Any], Any]] = MappingProxyType(
    {
        "clips_table": _text(),
        "clip_dir": _text(),
        "asr": _choice(ASR_MODES),
        "asr_resume_delay_sec": _seconds(0.0, 10.0),
        "asr_control_topic": _topic,
    }
)
_STANDBY_CHECKS: Mapping[str, Callable[[Any], Any]] = MappingProxyType(
    {"first_after_sec": _interval, "gap_sec": _interval}
)
_LOOK_CHECKS: Mapping[str, Callable[[Any], Any]] = MappingProxyType(
    {
        "demo_root": _text(),
        "round_period_sec": _seconds(),
        "window_sec": _seconds(),
        "quick_gap_sec": _seconds(),
        "nice": _integer(0, 19),
        "crop_actor": _flag,
        "refine_hand": _flag,
        "verify": _flag,
        "first_after_sec": _interval,
        "gap_sec": _interval,
        "max_per_round": _integer(1, 20),
        "fresh_sec": _seconds(positive=True),
        "pool_cooldown_sec": _seconds(),
        "fallback_after_sec": _seconds(),
        "latest_start_sec": _seconds(positive=True),
        "dump_dir": _text(allow_empty=True),
    }
)


def _items(
    data: Mapping[str, Any],
    checks: Mapping[str, Callable[[Any], Any]],
    defaults: Any,
    path: str,
    warnings: list[str],
    *,
    known: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Every key of ``checks`` from ``data``, a bad or missing one at its default."""
    unknown = sorted(str(key) for key in data if key not in checks and key not in known)
    if unknown:
        warnings.append(f"{path}: unknown keys {unknown} ignored; known: {[*known, *checks]}")
    values = {}
    for key, check in checks.items():
        default = getattr(defaults, key)
        if key not in data:
            values[key] = default
            continue
        try:
            values[key] = check(data[key])
        except _Bad as exc:
            warnings.append(f"{path}.{key}: {exc}; using the default {_shown(default)}")
            values[key] = default
    return values


def _shown(value: Any) -> str:
    return json.dumps(list(value) if isinstance(value, tuple) else value, ensure_ascii=False)


def _switch(data: Mapping[str, Any], path: str, warnings: list[str]) -> bool:
    """``enabled``: only an explicit ``false`` turns the section off."""
    if "enabled" not in data:
        warnings.append(f"{path}.enabled: missing; read as true")
        return True
    enabled = data["enabled"]
    if enabled is False:
        return False
    if enabled is not True:
        warnings.append(f"{path}.enabled: {enabled!r} is not true or false; read as true")
    return True


# ---------------------------------------------------------------------------
# The section.


def normalize(raw: Any) -> tuple[Settings | None, list[str]]:
    """The section over the defaults, plus a warning per bad item.

    ``None`` (section left out) or ``enabled: false`` is the baseline and
    returns ``None``. Anything else is on: a bad item warns and keeps its
    default, a broken or disabled ``look`` leaves ``Settings.look`` ``None``.
    """
    warnings: list[str] = []
    if raw is None:
        return None, warnings
    if raw is False:
        warnings.append(f"{SECTION_PATH}: must be a mapping; false read as off")
        return None, warnings
    if not isinstance(raw, Mapping):
        warnings.append(f"{SECTION_PATH}: must be a mapping, not {raw!r}; every item at its default")
        return Settings(), warnings
    if not _switch(raw, SECTION_PATH, warnings):
        return None, warnings

    top = _items(raw, _TOP_CHECKS, Settings, SECTION_PATH, warnings, known=("enabled", "standby", "look"))
    return Settings(
        enabled=True,
        standby=_standby(raw.get("standby"), warnings),
        look=_look(raw["look"], warnings) if "look" in raw else LookSettings(),
        **top,
    ), warnings


def _standby(value: Any, warnings: list[str]) -> StandbySettings:
    path = f"{SECTION_PATH}.standby"
    if value is None:
        return StandbySettings()
    if not isinstance(value, Mapping):
        warnings.append(f"{path}: must be a mapping, not {value!r}; using the defaults")
        return StandbySettings()
    return StandbySettings(**_items(value, _STANDBY_CHECKS, StandbySettings, path, warnings))


def _look(value: Any, warnings: list[str]) -> LookSettings | None:
    path = f"{SECTION_PATH}.look"
    if not isinstance(value, Mapping):
        warnings.append(f"{path}: must be a mapping, not {value!r}; the look is off")
        return None
    if not _switch(value, path, warnings):
        return None
    return LookSettings(enabled=True, **_items(value, _LOOK_CHECKS, LookSettings, path, warnings, known=("enabled",)))


def resolve(path: str, workspace: str | None) -> str:
    """``path`` as is when absolute, else under ``workspace`` (``$WORKSPACE``)."""
    path = os.path.expanduser(path)
    if os.path.isabs(path) or not workspace:
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(workspace, path))


# ---------------------------------------------------------------------------
# The clip table.


def pool_of(clip_id: str) -> str:
    """``evt_ready_01`` -> ``evt.ready``."""
    if not CLIP_ID_PATTERN.match(clip_id):
        raise ValueError(f"{clip_id!r} is not a clip id <group>_<sub>_<NN>")
    group, sub, _ = clip_id.split("_")
    return f"{group}.{sub}"


def is_look_pool(pool: str) -> bool:
    """Every pool but the transition events and the standby words."""
    return not pool.startswith("evt.") and pool != STANDBY_POOL and not pool.startswith("standby.")


@dataclass(frozen=True)
class ClipTable:
    version: int = DEFAULT_TABLE_VERSION
    wav_suffix: str = DEFAULT_WAV_SUFFIX
    # Clip id -> text, in table order; read-only.
    clips: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def pool_of(self, clip_id: str) -> str:
        return pool_of(clip_id)

    def pools(self) -> dict[str, tuple[str, ...]]:
        """Pool -> its clip ids, both in table order."""
        pools: dict[str, list[str]] = {}
        for clip_id in self.clips:
            pools.setdefault(pool_of(clip_id), []).append(clip_id)
        return {pool: tuple(ids) for pool, ids in pools.items()}

    def wav_name(self, clip_id: str) -> str:
        return f"{clip_id}{self.wav_suffix}"


def _wav_suffix(value: Any) -> str:
    value = _text()(value)
    if "/" in value or not value.endswith(".wav"):
        raise _Bad(f"must be a file name ending in .wav, not {value!r}")
    return value


def parse_clip_table(mapping: Any) -> tuple[ClipTable, list[str]]:
    """The table over the defaults; a bad entry is dropped with a warning.

    An entry needs an id ``<group>_<sub>_<NN>`` and a non-empty text with no
    ASCII letter or digit, unique in the table. A required pool left empty
    (any event, ``standby.any``) warns too; ``missing_pools`` lists them.
    """
    warnings: list[str] = []
    if not isinstance(mapping, Mapping):
        warnings.append(f"{TABLE_PATH}: must be a mapping, not {type(mapping).__name__}; the table is empty")
        return ClipTable(), warnings
    checks = {"version": _integer(1, 999), "wav_suffix": _wav_suffix}
    head = _items(mapping, checks, ClipTable, TABLE_PATH, warnings, known=("clips",))

    clips: dict[str, str] = {}
    raw_clips = mapping.get("clips")
    if not isinstance(raw_clips, Mapping):
        warnings.append(f"{TABLE_PATH}.clips: must be a mapping from clip id to text; the table is empty")
        raw_clips = {}
    seen: dict[str, str] = {}
    for clip_id, text in raw_clips.items():
        where = f"{TABLE_PATH}.clips.{clip_id}"
        if not isinstance(clip_id, str) or not CLIP_ID_PATTERN.match(clip_id):
            warnings.append(f"{where}: the id is not <group>_<sub>_<NN> (lowercase, two digits); dropped")
            continue
        if not isinstance(text, str) or not text.strip():
            warnings.append(f"{where}: the text must be a non-empty string; dropped")
            continue
        text = text.strip()
        if _ASCII_ALNUM.search(text):
            warnings.append(f"{where}: {text!r} has an ASCII letter or digit; dropped")
            continue
        if text in seen:
            warnings.append(f"{where}: {text!r} is already {seen[text]}; dropped")
            continue
        seen[text] = clip_id
        clips[clip_id] = text

    table = ClipTable(clips=MappingProxyType(clips), **head)
    for pool in missing_pools(table):
        warnings.append(f"{TABLE_PATH}: pool {pool} has no clip")
    return table, warnings


def missing_pools(table: ClipTable) -> list[str]:
    """The required pools (events, ``standby.any``) the table has no clip in."""
    pools = table.pools()
    return [pool for pool in REQUIRED_POOLS if pool not in pools]


def load_clip_table(path: str) -> tuple[ClipTable | None, list[str]]:
    """The table in a YAML file; ``None`` when it cannot be read or lacks a required pool."""
    try:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        return None, [f"{TABLE_PATH}: cannot read {path} ({exc})"]
    table, warnings = parse_clip_table(data)
    if missing_pools(table):
        warnings.append(f"{TABLE_PATH}: {path} lacks {missing_pools(table)}; not used")
        return None, warnings
    return table, warnings


# ---------------------------------------------------------------------------
# What the builders compute.


@dataclass(frozen=True)
class Bundle:
    # None when the section is off (baseline); then table is None too.
    settings: Settings | None
    table: ClipTable | None
    # Where to look for the WAVs, in order: the configured dir, then the default.
    clip_dirs: tuple[str, ...]
    warnings: tuple[str, ...]


def expand_find(text: str, find: Callable[[str], str]) -> str:
    """Every ``$(find pkg)`` replaced by ``find(pkg)``, the package's share dir."""
    return _FIND.sub(lambda match: str(find(match.group(1))), text)


def build_bundle(
    raw: Any,
    workspace: str | None,
    default_table: str,
    find: Callable[[str], str],
) -> Bundle:
    """The section, its clip table and its paths, ready for the two JSON parameters.

    ``default_table`` is the path of the table installed with robot_config; it
    stands in when the configured one cannot be read or is unusable.
    ``clip_dir`` and ``look.demo_root`` come back resolved against
    ``workspace``, ``clips_table`` as the path actually read.
    """
    settings, warnings = normalize(raw)
    if settings is None:
        return Bundle(None, None, (), tuple(warnings))

    try:
        configured = resolve(expand_find(settings.clips_table, find), workspace)
    except Exception as exc:  # noqa: BLE001 - any find() failure falls back to the default table
        warnings.append(f"{SECTION_PATH}.clips_table: cannot expand {settings.clips_table!r} ({exc})")
        configured = ""
    table, table_warnings = load_clip_table(configured) if configured else (None, [])
    warnings.extend(table_warnings)
    used = configured
    if table is None:
        used = os.path.normpath(default_table)
        if used != configured:
            warnings.append(f"{SECTION_PATH}.clips_table: using the installed table {used}")
            table, table_warnings = load_clip_table(used)
            warnings.extend(table_warnings)
        if table is None:
            warnings.append(f"{SECTION_PATH}.clips_table: no usable table; nothing to play")
            table = ClipTable()

    if not workspace:
        warnings.append(f"{SECTION_PATH}: WORKSPACE is not set; relative paths stay relative")
    clip_dirs = tuple(dict.fromkeys(resolve(path, workspace) for path in (settings.clip_dir, DEFAULT_CLIP_DIR)))
    look = settings.look
    if look is not None:
        look = replace(look, demo_root=resolve(look.demo_root, workspace))
    settings = replace(settings, clips_table=used, clip_dir=clip_dirs[0], look=look)
    return Bundle(settings, table, clip_dirs, tuple(warnings))


# ---------------------------------------------------------------------------
# mirror_voice's parameter ``clips_json``.


def playback_json(settings: Settings, table: ClipTable, clip_dirs: tuple[str, ...] | list[str]) -> str:
    """What mirror_voice needs to speak the clips, as one string parameter."""
    pools = table.pools()
    data = {
        "version": table.version,
        "clip_dirs": list(clip_dirs),
        "wav_suffix": table.wav_suffix,
        "clips": dict(table.clips),
        "events": {event: list(pools.get(pool, ())) for event, pool in EVENT_POOLS.items()},
        "standby": {
            "ids": list(pools.get(STANDBY_POOL, ())),
            "first_after_sec": list(settings.standby.first_after_sec),
            "gap_sec": list(settings.standby.gap_sec),
        },
        "asr": {
            "mode": settings.asr,
            "resume_delay_sec": settings.asr_resume_delay_sec,
            "topic": settings.asr_control_topic,
        },
    }
    return json.dumps(data, ensure_ascii=False)


def _load_json(text: Any, path: str, warnings: list[str]) -> Mapping[str, Any]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        warnings.append(f"{path}: not JSON ({exc}); every field at its default")
        return {}
    if not isinstance(data, Mapping):
        warnings.append(f"{path}: must be a JSON object; every field at its default")
        return {}
    return data


def _clip_dirs(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise _Bad(f"must be a non-empty list of directories, not {value!r}")
    return [_text()(item) for item in value]


def _id_list(value: Any, clips: Mapping[str, str], path: str, warnings: list[str]) -> list[str]:
    if not isinstance(value, list):
        warnings.append(f"{path}: must be a list of clip ids; none")
        return []
    ids = []
    for clip_id in value:
        if clip_id in clips:
            ids.append(clip_id)
        else:
            warnings.append(f"{path}: {clip_id!r} is not a clip; dropped")
    return ids


def parse_playback_json(text: str) -> tuple[dict[str, Any], list[str]]:
    """``playback_json`` read back; a bad field warns and takes its default.

    Intervals come back as ``(lo, hi)`` floats; ids in ``events`` and
    ``standby.ids`` are only ones present in ``clips``.
    """
    path = "clips_json"
    warnings: list[str] = []
    data = _load_json(text, path, warnings)
    defaults = Settings()

    head = _items(
        {key: data[key] for key in ("version", "clip_dirs", "wav_suffix") if key in data},
        {"version": _integer(1, 999), "clip_dirs": _clip_dirs, "wav_suffix": _wav_suffix},
        _PlaybackDefaults,
        path,
        warnings,
    )
    head["clip_dirs"] = list(head["clip_dirs"])
    unknown = sorted(str(key) for key in data if key not in _PLAYBACK_KEYS)
    if unknown:
        warnings.append(f"{path}: unknown keys {unknown} ignored")

    clips: dict[str, str] = {}
    raw_clips = data.get("clips", {})
    if not isinstance(raw_clips, Mapping):
        warnings.append(f"{path}.clips: must be an object from clip id to text; none")
        raw_clips = {}
    for clip_id, clip_text in raw_clips.items():
        if CLIP_ID_PATTERN.match(str(clip_id)) and isinstance(clip_text, str) and clip_text.strip():
            clips[clip_id] = clip_text.strip()
        else:
            warnings.append(f"{path}.clips.{clip_id}: not a clip id with a text; dropped")

    raw_events = data.get("events", {})
    if not isinstance(raw_events, Mapping):
        warnings.append(f"{path}.events: must be an object; none")
        raw_events = {}
    events = {event: _id_list(raw_events.get(event, []), clips, f"{path}.events.{event}", warnings) for event in EVENTS}

    raw_standby = data.get("standby", {})
    if not isinstance(raw_standby, Mapping):
        warnings.append(f"{path}.standby: must be an object; the defaults")
        raw_standby = {}
    standby = {"ids": _id_list(raw_standby.get("ids", []), clips, f"{path}.standby.ids", warnings)}
    standby.update(
        _items(
            {key: raw_standby[key] for key in _STANDBY_CHECKS if key in raw_standby},
            _STANDBY_CHECKS,
            defaults.standby,
            f"{path}.standby",
            warnings,
        )
    )

    raw_asr = data.get("asr", {})
    if not isinstance(raw_asr, Mapping):
        warnings.append(f"{path}.asr: must be an object; the defaults")
        raw_asr = {}
    asr = _items(
        raw_asr,
        {
            "mode": _TOP_CHECKS["asr"],
            "resume_delay_sec": _TOP_CHECKS["asr_resume_delay_sec"],
            "topic": _TOP_CHECKS["asr_control_topic"],
        },
        _AsrDefaults,
        f"{path}.asr",
        warnings,
    )
    return {**head, "clips": clips, "events": events, "standby": standby, "asr": asr}, warnings


_PLAYBACK_KEYS = ("version", "clip_dirs", "wav_suffix", "clips", "events", "standby", "asr")


class _PlaybackDefaults:
    version = DEFAULT_TABLE_VERSION
    clip_dirs = (DEFAULT_CLIP_DIR,)
    wav_suffix = DEFAULT_WAV_SUFFIX


class _AsrDefaults:
    mode = Settings.asr
    resume_delay_sec = Settings.asr_resume_delay_sec
    topic = Settings.asr_control_topic


# ---------------------------------------------------------------------------
# visual_interaction's parameter ``look_json``.

LOOK_KEYS = tuple(item.name for item in fields(LookSettings))


def look_json(settings: Settings, table: ClipTable) -> str:
    """What the look needs, as one string parameter: every ``look`` key plus the look pools."""
    if settings.look is None:
        raise ValueError(f"{SECTION_PATH}.look is off; there is no look_json")
    data: dict[str, Any] = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in ((key, getattr(settings.look, key)) for key in LOOK_KEYS)
    }
    data["pools"] = {
        pool: [[clip_id, table.clips[clip_id]] for clip_id in ids]
        for pool, ids in table.pools().items()
        if is_look_pool(pool)
    }
    return json.dumps(data, ensure_ascii=False)


def parse_look_json(text: str) -> tuple[dict[str, Any], list[str]]:
    """``look_json`` read back; a bad field warns and takes its default.

    Intervals come back as ``(lo, hi)`` floats and ``pools`` as
    ``{pool: ((id, text), ...)}`` with look pools only.
    """
    path = "look_json"
    warnings: list[str] = []
    data = _load_json(text, path, warnings)
    values = _items(data, _LOOK_CHECKS, LookSettings, path, warnings, known=("enabled", "pools"))
    if data.get("enabled", True) is not True:
        warnings.append(f"{path}.enabled: {data.get('enabled')!r} read as true")

    pools: dict[str, tuple[tuple[str, str], ...]] = {}
    raw_pools = data.get("pools", {})
    if not isinstance(raw_pools, Mapping):
        warnings.append(f"{path}.pools: must be an object; none")
        raw_pools = {}
    for pool, entries in raw_pools.items():
        where = f"{path}.pools.{pool}"
        if not isinstance(pool, str) or not is_look_pool(pool) or not isinstance(entries, list):
            warnings.append(f"{where}: not a look pool with a list of clips; dropped")
            continue
        kept = []
        for entry in entries:
            ok = isinstance(entry, list | tuple) and len(entry) == 2 and all(isinstance(item, str) for item in entry)
            if not ok or not CLIP_ID_PATTERN.match(entry[0]) or pool_of(entry[0]) != pool or not entry[1].strip():
                warnings.append(f"{where}: {entry!r} is not [id, text] of this pool; dropped")
                continue
            kept.append((entry[0], entry[1].strip()))
        if kept:
            pools[pool] = tuple(kept)
    return {"enabled": True, **values, "pools": pools}, warnings
