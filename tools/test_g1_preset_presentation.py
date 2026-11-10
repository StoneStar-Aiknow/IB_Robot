"""The catalog's display tags must stay bound to this profile's action numbers.

``vendor.arm_actions`` is the only source of the vendor firmware action ids.
The catalog manifests are the only source the CLI and the Web display. This
suite keeps the two equal and proves the projection stays presentation-only:
a profile id edit that is not followed by a manifest regeneration fails here,
and no skill outside this profile may display a vendor action number.
"""

import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_preset_presentation import (  # noqa: E402
    ACTION_ID_TAG_PREFIX,
    CONTROL_OWNERSHIP_ACTIONS,
    CONTROL_OWNERSHIP_TAG,
    PRESET_TAG,
    apply_updates,
    parse_action_id_tag,
    plan_updates,
    vendor_action_ids,
)

REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE = REPOSITORY / "src/robots/unitree/unitree_g1_robot/profiles/g1_29dof_arm_presets.yaml"
CATALOG_ROOT = REPOSITORY / "src" / "agent" / "skill_catalog" / "config"
SKILLS_ROOT = CATALOG_ROOT / "skills"
CATALOG_PROFILE = CATALOG_ROOT / "profiles" / "unitree_g1_arm_presets.yaml"
ACTION_TAG = re.compile(r"^[a-z][a-z0-9]*_action_id_[0-9]{1,6}$")


def _profile() -> dict:
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_shipped_catalog_tags_reproduce_from_the_profile():
    updates = plan_updates(PROFILE, SKILLS_ROOT)

    assert [update.skill for update in updates] == list(vendor_action_ids(_profile()))
    assert len(updates) == 23
    for update in updates:
        assert not update.changed, f"{update.skill}: rerun --write to bind the profile id"
        assert update.expected_tags == update.current_tags
        assert update.action_id == vendor_action_ids(_profile())[update.skill]


def test_a_profile_id_change_makes_the_catalog_tags_stale(tmp_path):
    """A renumbered firmware action must fail until the manifest is regenerated."""
    skills = tmp_path / "skills"
    shutil.copytree(SKILLS_ROOT, skills)
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    original = profile["vendor"]["arm_actions"]["arms_horizontal"]["action_id"]
    profile["vendor"]["arm_actions"]["arms_horizontal"]["action_id"] = original + 1
    profile["vendor"]["preset_commissioning"]["allowed_arm_actions"]["arms_horizontal"]["action_id"] = original + 1
    drifted = tmp_path / "profile.yaml"
    drifted.write_text(yaml.safe_dump(profile, allow_unicode=True), encoding="utf-8")

    updates = {update.skill: update for update in plan_updates(drifted, skills)}

    arms = updates["arms_horizontal"]
    assert arms.changed
    assert arms.current_tags == (PRESET_TAG, f"{ACTION_ID_TAG_PREFIX}{original}")
    assert arms.expected_tags == (PRESET_TAG, f"{ACTION_ID_TAG_PREFIX}{original + 1}")
    assert [name for name, update in updates.items() if update.changed] == ["arms_horizontal"]


def test_regenerating_a_stale_catalog_keeps_every_other_manifest_byte(tmp_path):
    skills = tmp_path / "skills"
    shutil.copytree(SKILLS_ROOT, skills)
    before = {path: path.read_text(encoding="utf-8") for path in skills.rglob("*.yaml")}
    for update in plan_updates(PROFILE, skills):
        manifest = yaml.safe_load(update.text)
        manifest["capability"]["tags"] = [PRESET_TAG]
        update.path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    stale = {update.skill: update for update in plan_updates(PROFILE, skills)}
    assert all(update.changed for update in stale.values())

    apply_updates(list(stale.values()), write=True)

    assert not [update for update in plan_updates(PROFILE, skills) if update.changed]
    for path, original in before.items():
        if path.parent.name in stale:
            continue
        assert path.read_text(encoding="utf-8") == original
    for name, update in stale.items():
        regenerated = yaml.safe_load((skills / name / "manifest.yaml").read_text(encoding="utf-8"))
        assert tuple(regenerated["capability"]["tags"]) == update.expected_tags
        assert regenerated["capability"]["summary"] == yaml.safe_load(update.text)["capability"]["summary"]


def test_action_tags_are_unique_and_fit_the_presentation_grammar():
    updates = plan_updates(PROFILE, SKILLS_ROOT)

    ids = [update.action_id for update in updates]
    assert len(ids) == len(set(ids))
    for update in updates:
        tag = f"{ACTION_ID_TAG_PREFIX}{update.action_id}"
        assert parse_action_id_tag(tag) == update.action_id
        assert re.fullmatch(r"[a-z][a-z0-9_]{0,31}", tag)
        assert tag in update.current_tags


def test_catalog_profile_enables_exactly_the_profiled_actions():
    catalog_profile = yaml.safe_load(CATALOG_PROFILE.read_text(encoding="utf-8"))
    enabled = [entry["name"] for entry in catalog_profile["enabled_skills"]]

    assert enabled == list(vendor_action_ids(_profile()))


def test_only_release_arm_carries_the_control_ownership_badge():
    updates = plan_updates(PROFILE, SKILLS_ROOT)
    with_ownership = {update.skill for update in updates if CONTROL_OWNERSHIP_TAG in update.current_tags}

    assert with_ownership == set(CONTROL_OWNERSHIP_ACTIONS) == {"release_arm"}
    for update in updates:
        if update.skill in CONTROL_OWNERSHIP_ACTIONS:
            assert CONTROL_OWNERSHIP_TAG in update.expected_tags
        else:
            assert CONTROL_OWNERSHIP_TAG not in update.expected_tags


def test_the_control_ownership_badge_grants_nothing():
    """The badge is display data; admission and recovery policy stay untouched."""
    manifest = yaml.safe_load((SKILLS_ROOT / "release_arm" / "manifest.yaml").read_text(encoding="utf-8"))
    capability = manifest["capability"]

    assert manifest["semantic_level"] == "atomic_operator"
    assert capability["recovery_policy"] == "never_retry"
    assert capability["required_capabilities"] == ["motion.named"]
    assert set(capability).isdisjoint({"vendor", "action_id", "control_ownership", "auto_recover"})
    assert CONTROL_OWNERSHIP_TAG in capability["tags"]
    # No other tag grants an entry point: the only execution mapping is the motion binding.
    implementation = yaml.safe_load(
        (SKILLS_ROOT / "release_arm" / "implementations" / "g1_preset_v1.yaml").read_text(encoding="utf-8")
    )
    assert implementation["binding"] == {"motion": "release_arm"}
    assert set(implementation).isdisjoint({"action_id", "vendor_action_id", "control_ownership"})


def test_no_skill_outside_this_profile_displays_a_vendor_action_id():
    profiled = set(vendor_action_ids(_profile()))
    offenders = []
    for manifest_path in sorted(SKILLS_ROOT.glob("*/manifest.yaml")):
        name = manifest_path.parent.name
        tags = (yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}).get("capability", {}).get("tags") or []
        action_tags = [tag for tag in tags if ACTION_TAG.match(str(tag))]
        if name in profiled:
            assert len(action_tags) == 1, f"{name}: expected exactly one vendor action id tag"
        else:
            offenders.extend((name, tag) for tag in action_tags)
        if name not in CONTROL_OWNERSHIP_ACTIONS:
            assert CONTROL_OWNERSHIP_TAG not in tags, f"{name}: control ownership is release_arm only"
    assert offenders == []


def test_release_arm_is_never_inserted_into_another_catalog_entry():
    """Listing release_arm must not make it a recovery or finishing step."""
    references = []
    for path in sorted(SKILLS_ROOT.rglob("*.yaml")):
        if path.relative_to(SKILLS_ROOT).parts[0] == "release_arm":
            continue
        payload = path.read_text(encoding="utf-8")
        payload = re.sub(r"^#.*$", "", payload, flags=re.MULTILINE)
        if re.search(r"\brelease_arm\b", payload):
            references.append(str(path.relative_to(CATALOG_ROOT)))
    assert references == []


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("unitree_action_id_99", 99),
        ("unitree_action_id_0", 0),
        ("unitree_action_id_15", 15),
        ("unitree_action_id_", None),
        ("unitree_action_id_abc", None),
        ("unitree_action_id_1234567", None),
        ("preset", None),
        ("control_ownership", None),
    ],
)
def test_action_id_tag_parsing_is_exact(tag, expected):
    assert parse_action_id_tag(tag) == expected
