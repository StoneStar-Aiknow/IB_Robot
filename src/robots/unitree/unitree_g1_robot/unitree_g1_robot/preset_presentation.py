"""Project this profile's vendor action numbers into catalog display metadata.

The runtime profile (``profiles/<name>.yaml``) is the only source of the
vendor firmware action numbers: ``vendor.arm_actions.<skill>.action_id`` is
what the runtime sends to the firmware and what admission checks. The catalog
manifests (``src/agent/skill_catalog/config/skills/<skill>/manifest.yaml``)
are the only source the Gateway compiles and the CLI/Web display.

This module is the single bridge between the two, and it is deliberately
*display only*:

* it writes one optional entry into the manifest's existing
  ``capability.tags`` list, ``unitree_action_id_<N>``;
* ``release_arm`` additionally carries ``control_ownership`` because it
  returns control to the operator instead of running a gesture;
* it never writes an execution field, a binding, a recovery policy or an
  admission rule, and nothing in the tag is an execution entry point.

The package test ``test_catalog_action_id_presentation.py`` runs the same
projection with ``write=False``, so a profile id change (for example
``arms_horizontal: 15 -> 16``) fails until the manifests are regenerated.

Usage::

    python3 -m unitree_g1_robot.preset_presentation --check
    python3 -m unitree_g1_robot.preset_presentation --write
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ACTION_ID_TAG_PREFIX = "unitree_action_id_"
CONTROL_OWNERSHIP_TAG = "control_ownership"
PRESET_TAG = "preset"
# Control-ownership operations hand the arm back to the operator instead of
# performing a gesture. Listing one here only selects a display badge.
CONTROL_OWNERSHIP_ACTIONS = frozenset({"release_arm"})

_ACTION_ID_TAG = re.compile(rf"^{ACTION_ID_TAG_PREFIX}(?P<action_id>[0-9]{{1,6}})$")
_TAGS_KEY = re.compile(r"^(?P<indent>[ ]*)tags:[ ]*(?P<inline>.*)$")
_CHILD_KEY = re.compile(r"^(?P<indent>[ ]*)[a-z_]+:")
_SEQUENCE_ITEM = re.compile(r"^(?P<indent>[ ]*)-[ ]+(?P<value>.*)$")
_MAX_ACTION_ID = 999_999


@dataclass(frozen=True)
class PresentationUpdate:
    """One manifest whose ``capability.tags`` list differs from the profile."""

    skill: str
    path: Path
    action_id: int
    current_tags: tuple[str, ...]
    expected_tags: tuple[str, ...]
    text: str
    updated_text: str

    @property
    def changed(self) -> bool:
        return self.current_tags != self.expected_tags


def action_id_tag(action_id: int) -> str:
    """Return the display tag that carries one vendor action number."""
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise ValueError("action_id must be an integer")
    if not 0 <= action_id <= _MAX_ACTION_ID:
        raise ValueError(f"action_id {action_id} is outside 0..{_MAX_ACTION_ID}")
    return f"{ACTION_ID_TAG_PREFIX}{action_id}"


def parse_action_id_tag(tag: str) -> int | None:
    """Return the number carried by a vendor action tag, or ``None``."""
    match = _ACTION_ID_TAG.match(tag)
    return int(match.group("action_id")) if match else None


def vendor_action_ids(profile: Mapping[str, Any]) -> dict[str, int]:
    """Read ``vendor.arm_actions`` and return skill name -> firmware action id."""
    vendor = profile.get("vendor")
    if not isinstance(vendor, Mapping):
        raise ValueError("profile has no vendor block")
    actions = vendor.get("arm_actions")
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("profile has no vendor.arm_actions mapping")
    result: dict[str, int] = {}
    for name, entry in actions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("vendor.arm_actions key must be a skill name")
        if not isinstance(entry, Mapping) or not isinstance(entry.get("action_id"), int):
            raise ValueError(f"vendor.arm_actions.{name}.action_id must be an integer")
        result[name] = entry["action_id"]
    return result


def expected_presentation_tags(skill: str, action_id: int) -> tuple[str, ...]:
    """Return the presentation tags this projection owns for one skill."""
    tags = [PRESET_TAG]
    if skill in CONTROL_OWNERSHIP_ACTIONS:
        tags.append(CONTROL_OWNERSHIP_TAG)
    tags.append(action_id_tag(action_id))
    return tuple(tags)


def _managed(tag: str) -> bool:
    return tag == CONTROL_OWNERSHIP_TAG or parse_action_id_tag(tag) is not None


def _rewrite_tags(text: str, skill: str, expected: Sequence[str]) -> str:
    """Replace only the ``capability.tags`` list, keeping every other byte."""
    lines = text.splitlines(keepends=True)
    capability_index = next((index for index, line in enumerate(lines) if line.rstrip("\r\n") == "capability:"), None)
    if capability_index is None:
        raise ValueError(f"{skill}: manifest has no top-level capability block")
    tags_index = None
    child_indent = None
    for index in range(capability_index + 1, len(lines)):
        raw = lines[index].rstrip("\r\n")
        if raw and not raw.startswith((" ", "#")):
            break
        if not raw.strip():
            continue
        match = _TAGS_KEY.match(raw)
        if child_indent is None:
            child_indent = match.group("indent") if match is not None else _CHILD_KEY.match(raw).group("indent")
        if match is not None and match.group("indent") == child_indent:
            tags_index = index
            break
    if tags_index is None:
        raise ValueError(f"{skill}: manifest capability has no tags list to carry display metadata")
    key = _TAGS_KEY.match(lines[tags_index].rstrip("\r\n"))
    if key is None:  # pragma: no cover - guarded by the loop above
        raise ValueError(f"{skill}: manifest capability has no tags list to carry display metadata")
    inline = key.group("inline").strip()
    newline = "\r\n" if lines[tags_index].endswith("\r\n") else "\n"
    if inline:
        if not inline.startswith("["):
            raise ValueError(f"{skill}: unsupported tags value {inline!r}")
        replacement = [
            f"{key.group('indent')}tags: {yaml.safe_dump(list(expected), default_flow_style=True).strip()}{newline}"
        ]
        head = lines[:tags_index]
        end = tags_index + 1
    else:
        item_indent = None
        end = tags_index + 1
        while end < len(lines):
            raw = lines[end].rstrip("\r\n")
            item = _SEQUENCE_ITEM.match(raw)
            if item is None:
                break
            if item_indent is None:
                item_indent = item.group("indent")
            end += 1
        if item_indent is None:
            item_indent = f"{key.group('indent')}  "
        replacement = [f"{item_indent}- {value}{newline}" for value in expected]
        head = lines[: tags_index + 1]
    return "".join([*head, *replacement, *lines[end:]])


def plan_updates(profile_path: Path, skills_root: Path) -> list[PresentationUpdate]:
    """Return one update per profile action, in profile order."""
    profile = yaml.safe_load(Path(profile_path).read_text(encoding="utf-8"))
    if not isinstance(profile, Mapping):
        raise ValueError(f"{profile_path}: profile is not a mapping")
    updates: list[PresentationUpdate] = []
    for skill, action_id in vendor_action_ids(profile).items():
        path = Path(skills_root) / skill / "manifest.yaml"
        if not path.is_file():
            raise ValueError(f"{skill}: catalog manifest {path} is missing")
        text = path.read_text(encoding="utf-8")
        manifest = yaml.safe_load(text)
        if not isinstance(manifest, Mapping) or manifest.get("name") != skill:
            raise ValueError(f"{skill}: catalog manifest name does not match the profile key")
        capability = manifest.get("capability")
        if not isinstance(capability, Mapping):
            raise ValueError(f"{skill}: catalog manifest has no capability block")
        tags = capability.get("tags")
        if not isinstance(tags, list) or PRESET_TAG not in tags:
            raise ValueError(f"{skill}: catalog manifest must already declare the {PRESET_TAG!r} tag")
        current = tuple(str(tag) for tag in tags)
        expected = expected_presentation_tags(skill, action_id)
        # Keep any classification this projection does not own, in its order.
        preserved = tuple(tag for tag in current if tag != PRESET_TAG and not _managed(tag))
        expected = (*expected, *preserved)
        updates.append(
            PresentationUpdate(
                skill=skill,
                path=path,
                action_id=action_id,
                current_tags=current,
                expected_tags=expected,
                text=text,
                updated_text=_rewrite_tags(text, skill, expected),
            )
        )
    return updates


def apply_updates(updates: Sequence[PresentationUpdate], *, write: bool) -> list[PresentationUpdate]:
    """Write the changed manifests, or return them untouched for a check."""
    changed = [update for update in updates if update.changed]
    if write:
        for update in changed:
            update.path.write_text(update.updated_text, encoding="utf-8")
    return changed


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "agent" / "skill_catalog" / "config" / "profiles").is_dir():
            return candidate
    raise ValueError(f"cannot find the IB_Robot repository root from {start}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, help="runtime profile that owns vendor.arm_actions")
    parser.add_argument("--skills-root", type=Path, help="catalog config/skills directory")
    parser.add_argument("--write", action="store_true", help="write the manifests instead of checking them")
    parser.add_argument("--check", action="store_true", help="report drift and exit 1 (the default)")
    args = parser.parse_args(argv)
    root = _repository_root(Path(__file__).resolve())
    profile = args.profile or root / "src/robots/unitree/unitree_g1_robot/profiles/g1_29dof_arm_presets.yaml"
    skills_root = args.skills_root or root / "src/agent/skill_catalog/config/skills"
    changed = apply_updates(plan_updates(profile, skills_root), write=args.write)
    for update in changed:
        print(f"{update.skill}: {list(update.current_tags)} -> {list(update.expected_tags)}")
    if args.write:
        print(f"updated {len(changed)} manifest(s)")
        return 0
    if changed:
        print("catalog presentation tags are stale; rerun with --write")
        return 1
    print("catalog presentation tags are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
