"""Recorded replies of a deployment: the clip manifest shared by the launch, the executor and the voice bridge.

A manifest (``commissioning/audio/summit_clips.yaml``) lists, per code, a logical clip name, a
playback mode and the lines recorded for it::

    codes:
      Q17A: {clip: price_first, mode: sequence, consumer: skill:answer_price,
             lines: [{file: Q17A_01.wav, text: ..., sha256: ...}, ...]}

``random`` plays one file of the code; ``sequence`` plays all of them in order, installed as
one joined ``<code>_seq.wav``. Two views are derived here, so the files are named once:

- ``clip_files``: clip name -> files, for ``runtime_play_audio`` skills;
- ``spoken_clips``: spoken text -> files, for the voice bridge, which plays the recording of
  any text it would otherwise synthesize when that text is one of the recorded lines.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MODES = ("random", "sequence")


def spoken_key(text: str) -> str:
    """The text as heard: letters and digits only, case-folded, so punctuation never decides a match."""
    return "".join(char.casefold() for char in unicodedata.normalize("NFKC", str(text)) if char.isalnum())


def _entries(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ValueError("clip manifest must be a schema_version 1 mapping")
    codes = manifest.get("codes")
    if not isinstance(codes, Mapping) or not codes:
        raise ValueError("clip manifest must list codes")
    seen: set[str] = set()
    for code, entry in codes.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"clip manifest code {code!r} must be a mapping")
        clip = entry.get("clip")
        if not isinstance(clip, str) or not clip.strip() or clip in seen:
            raise ValueError(f"clip manifest code {code!r} needs a unique clip name")
        seen.add(clip)
        if entry.get("mode") not in MODES:
            raise ValueError(f"clip manifest code {code!r}: mode must be one of {MODES}")
        lines = entry.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"clip manifest code {code!r} needs lines")
        for line in lines:
            if (
                not isinstance(line, Mapping)
                or not str(line.get("file", "")).strip()
                or not spoken_key(line.get("text", ""))
            ):
                raise ValueError(f"clip manifest code {code!r}: every line needs a file and a text")
    return codes


def _files(code: str, entry: Mapping[str, Any], clip_dir: Path) -> list[str]:
    if entry["mode"] == "sequence":
        return [str(clip_dir / f"{code}_seq.wav")]
    return [str(clip_dir / str(line["file"])) for line in entry["lines"]]


def clip_files(manifest: Mapping[str, Any], clip_dir: str | Path) -> dict[str, list[str]]:
    """Clip name -> the files one playback chooses from."""
    clip_dir = Path(clip_dir)
    return {str(entry["clip"]): _files(code, entry, clip_dir) for code, entry in _entries(manifest).items()}


def spoken_clips(manifest: Mapping[str, Any], clip_dir: str | Path) -> dict[str, list[str]]:
    """Spoken text -> the files that say it (any line of a random code; all lines of a sequence)."""
    clip_dir = Path(clip_dir)
    spoken: dict[str, list[str]] = {}
    for code, entry in _entries(manifest).items():
        files = _files(code, entry, clip_dir)
        if entry["mode"] == "sequence":
            texts = ["".join(str(line["text"]) for line in entry["lines"])]
        else:
            texts = [str(line["text"]) for line in entry["lines"]]
        for text in texts:
            key = spoken_key(text)
            if key in spoken and spoken[key] != files:
                raise ValueError(f"clip manifest: the text {text!r} is recorded under two codes")
            spoken[key] = files
    return spoken


def skill_clips(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Skill name -> its clip name, for the codes a runtime_play_audio skill plays."""
    skills = {}
    for entry in _entries(manifest).values():
        consumer = str(entry.get("consumer", ""))
        if consumer.startswith("skill:"):
            skills[consumer.removeprefix("skill:")] = str(entry["clip"])
    return skills


def load_manifest(path: str | Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _entries(data)
    return data


def choose_clip_file(candidates: Any, *, last: str = "", exists=None, rng=None) -> str:
    """One file of ``candidates`` (a path or a list), never ``last`` when there is another choice.

    Files that ``exists`` rejects are skipped, so one missing recording does not silence the
    clip; empty when none is left.
    """
    import random

    paths = [candidates] if isinstance(candidates, str) else list(candidates or ())
    paths = [str(path).strip() for path in paths if str(path).strip()]
    if exists is not None:
        paths = [path for path in paths if exists(path)]
    if not paths:
        return ""
    fresh = [path for path in paths if path != last] or paths
    return (rng or random).choice(fresh)
