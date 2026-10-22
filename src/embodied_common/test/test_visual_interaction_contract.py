"""The visual interaction contract is the one place its shared names live."""

import subprocess
import sys

import pytest

from embodied_common import visual_interaction_contract as contract


def test_contract_module_stays_dependency_free():
    """Importing it must not pull in ROS, a launch builder or a sibling registry.

    The launch builder, the session node and a future speech coordinator all
    import this module; anything it drags in lands in each of their import
    budgets.
    """
    script = (
        "import sys; import embodied_common.visual_interaction_contract; "
        "leaked = sorted("
        "  name for name in sys.modules"
        "  if name.split('.')[0] in {'rclpy', 'launch', 'robot_config', 'ibrobot_msgs'}"
        ");"
        "assert not leaked, leaked"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_speakable_kinds_never_include_a_silent_kind():
    assert contract.KIND_NONE not in contract.SPEAKABLE_KINDS
    assert contract.KIND_UNCERTAIN not in contract.SPEAKABLE_KINDS
    assert set(contract.SPEAKABLE_KINDS) < set(contract.KINDS)


def test_none_and_uncertain_are_distinct_kinds():
    """A quiet run must be tellable from a blind one, so they never collapse."""
    assert contract.KIND_NONE != contract.KIND_UNCERTAIN
    assert contract.SUPPRESS_NO_NEW_TOPIC != contract.SUPPRESS_UNCERTAIN


def test_no_action_kind_until_a_gesture_path_is_accepted():
    """Gestures need a keypoint model and a measured latency first.

    Reporting one has to be impossible, not merely discouraged, or the feature
    can claim a capability nothing has verified.
    """
    assert "action" not in contract.KINDS


def test_categories_are_a_closed_whitelist_without_a_fallback():
    """A generic member would hand the sensitivity judgement back to the model."""
    assert contract.CATEGORIES == ("toy", "sign", "hat", "badge")
    for fallback in ("other", "unknown", "display_object", "object", "misc", ""):
        assert fallback not in contract.CATEGORIES


def test_pendant_stays_out_of_the_first_version():
    """Small-object visibility and religious-symbol refusal are accepted apart."""
    for excluded in ("pendant", "necklace", "medal", "medical", "book", "blade"):
        assert excluded not in contract.CATEGORIES


def test_unknown_is_a_region_but_never_a_category():
    assert "unknown" in contract.REGIONS
    assert "unknown" not in contract.CATEGORIES


def test_every_coverage_state_blocks_speaking_again():
    """Coverage outranks event numbering: no state may leave a topic re-sayable.

    Putting an object down and raising it again mints a new episode_seq, and
    the first version cannot tell a second object from the same one returning.
    """
    assert set(contract.BLOCKING_COVERAGE_STATES) == set(contract.COVERAGE_STATES)


def test_suppress_reasons_separate_working_silence_from_failure():
    assert contract.SUPPRESS_NO_NEW_TOPIC in contract.SUPPRESS_REASONS
    assert contract.SUPPRESS_MODEL_NOT_READY in contract.SUPPRESS_REASONS


def test_mirror_inactive_is_a_parked_feature_not_a_failure():
    """In the magicmirror entry the rounds suspend while the mirror is not
    MIRRORING: a bystander must be able to tell that from a broken model or a
    decided silence, which is the difference the status topic exists for."""
    assert contract.SUPPRESS_MIRROR_INACTIVE in contract.SUPPRESS_REASONS
    assert contract.SUPPRESS_MIRROR_INACTIVE != contract.SUPPRESS_MODEL_NOT_READY
    assert contract.SUPPRESS_MIRROR_INACTIVE != contract.SUPPRESS_NO_NEW_TOPIC
    assert contract.SUPPRESS_MIRROR_INACTIVE != contract.SUPPRESS_MUTED


def test_value_sets_have_no_duplicates_and_no_blanks():
    for name in ("KINDS", "SPEAKABLE_KINDS", "CATEGORIES", "REGIONS", "COVERAGE_STATES", "SUPPRESS_REASONS"):
        values = getattr(contract, name)
        assert isinstance(values, tuple), name
        assert len(set(values)) == len(values), name
        assert all(isinstance(value, str) and value for value in values), name


def test_topics_are_absolute_and_distinct():
    assert contract.UTTERANCE_TOPIC.startswith("/")
    assert contract.STATUS_TOPIC.startswith("/")
    assert contract.UTTERANCE_TOPIC != contract.STATUS_TOPIC
    assert contract.NODE_NAME in contract.UTTERANCE_TOPIC


def test_confirmation_needs_at_least_two_windows():
    """One window is a single frame set, which is what the second window checks."""
    assert contract.DEFAULT_CONFIRMATION_COUNT >= 2


def test_timing_defaults_are_positive_and_ordered():
    assert contract.DEFAULT_MAX_EVIDENCE_AGE_SEC > 0
    assert contract.DEFAULT_MIN_GAP_SEC > 0
    assert contract.DEFAULT_ABSENT_RESET_SEC > 0
    # A session must outlast the absence timer, or it would expire first and the
    # missed-hand-over guard could never fire.
    assert contract.DEFAULT_MAX_SESSION_SEC > contract.DEFAULT_ABSENT_RESET_SEC
    assert 0 < contract.DEFAULT_MIN_REPLY_CHARS < contract.DEFAULT_MAX_REPLY_CHARS


def test_no_actor_matches_the_actor_selector_sentinel():
    """The value is repeated to keep this module dependency-free; it must agree.

    Skipped rather than failed when manipulation_execution is not importable:
    embodied_common must not gain a dependency on it just to check a constant.
    """
    actor_selector = pytest.importorskip("manipulation_execution.actor_selector")

    assert contract.NO_ACTOR == actor_selector.NO_ACTOR


def test_the_message_carries_the_text_itself():
    """A visual game looks its announcement up by handler; this one does not.

    The reply is generated per round, so a consumer must read ``text`` rather
    than re-deriving words from the enum fields.
    """
    message_module = pytest.importorskip("ibrobot_msgs.msg")
    utterance = message_module.VisualInteractionUtterance()

    fields = utterance.get_fields_and_field_types()
    assert fields["text"] == "string"
    assert fields["suppress_reason"] == "string"
    assert fields["session_id"] == "string"
    assert fields["actor_epoch"] == "uint32"
    assert fields["kind"] == "string"
    # Default-constructed, it says nothing and claims no reason: a publisher has
    # to fill both in, so an unset message can never read as a candidate.
    assert utterance.text == ""
    assert utterance.suppress_reason == ""
