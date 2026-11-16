from types import SimpleNamespace

import pytest

from embodied_common import held_custody_contract as custody

ARMS = ("left", "right")
PICK, PLACE = custody.OWNER_PICK, custody.OWNER_PLACE


def _c(state, owner="", reason="", goal_active=False):
    return SimpleNamespace(state=state, owner=owner, reason_code=reason, goal_active=goal_active)


EMPTY = _c(custody.EMPTY)
HELD_BY_PICK = _c(custody.HELD, PICK)


def test_every_state_has_owner_and_reason_rules_and_unknown_strings_read_as_unknown():
    assert set(custody.STATE_OWNERS) == set(custody.STATE_REASONS) == custody.STATES
    for value in (None, "", "empty", "picked", "uncertain", 3):
        assert custody.normalize_state(value) == custody.UNKNOWN
    for state in custody.STATES:
        assert custody.normalize_state(state) == state


def test_only_empty_admits_new_tasks():
    assert not custody.blocks_new_tasks(custody.EMPTY)
    for state in (custody.HELD, custody.RECOVERING, custody.UNKNOWN, "picked", None):
        assert custody.blocks_new_tasks(state)


def test_a_dead_owners_last_message_never_admits_motion():
    assert custody.fresh_custody(EMPTY, 0.5) is EMPTY
    assert custody.fresh_custody(EMPTY, custody.CUSTODY_STALE_S) is EMPTY
    for age in (custody.CUSTODY_STALE_S + 0.1, float("inf"), float("nan"), -1.0):
        assert custody.fresh_custody(EMPTY, age) is None
    # The latched sample a restarted Gateway receives first may be a dead owner's.
    assert custody.fresh_custody(EMPTY, 0.0, first_sample=True) is None
    assert custody.blocked_arms({"left": custody.fresh_custody(EMPTY, 9.0)}, ["left"]) == ("left",)
    assert custody.CUSTODY_REPUBLISH_S < custody.CUSTODY_STALE_S


@pytest.mark.parametrize(
    ("state", "owner", "reason"),
    [
        (custody.EMPTY, "", ""),
        (custody.EMPTY, "", custody.DROPPED),
        (custody.HELD, PICK, ""),
        (custody.HELD, PLACE, ""),
        (custody.HELD, PICK, custody.NO_PUT_BACK),
        (custody.RECOVERING, "", ""),
        (custody.UNKNOWN, "", custody.STARTUP_UNVERIFIED),
        (custody.UNKNOWN, PLACE, custody.PLACE_UNCERTAIN),
        (custody.UNKNOWN, PICK, custody.OPERATOR_TAKEOVER),
    ],
)
def test_publishable_combinations(state, owner, reason):
    custody.validate_custody(state, owner, reason)


@pytest.mark.parametrize(
    ("state", "owner", "reason"),
    [
        (custody.EMPTY, PICK, ""),
        (custody.EMPTY, "", custody.STARTUP_UNVERIFIED),
        (custody.HELD, "", ""),
        (custody.HELD, PICK, custody.GRIP_UNCERTAIN),
        (custody.HELD, PLACE, custody.NO_PUT_BACK),
        (custody.RECOVERING, PICK, ""),
        (custody.UNKNOWN, "", ""),
        (custody.UNKNOWN, "", custody.DROPPED),
        ("picked", PICK, ""),
    ],
)
def test_combinations_the_owner_must_never_publish(state, owner, reason):
    with pytest.raises(ValueError):
        custody.validate_custody(state, owner, reason)


def test_a_held_arm_admits_carrying_and_placing_it_from_any_root_task():
    custodies = {"left": HELD_BY_PICK, "right": EMPTY}
    assert custody.blocked_arms(custodies, [], moves_body=True, all_arms=ARMS) == ()
    assert custody.blocked_arms(custodies, ["left"], placing=True) == ()
    # Nothing else may use the held arm: a second pick, a gesture.
    assert custody.blocked_arms(custodies, ["left"]) == ("left",)
    assert custody.blocked_arms(custodies, ["right"]) == ()
    assert custody.free_arms(custodies, ARMS) == ("right",)
    assert custody.free_arms({"left": HELD_BY_PICK}, ARMS) == ()
    # A recovery admits nothing, not even a place.
    assert custody.blocked_arms({"left": _c(custody.RECOVERING)}, ["left"], placing=True) == ("left",)


def test_an_arm_a_place_holds_admits_nothing_until_it_is_given_back():
    custodies = {"left": _c(custody.HELD, PLACE), "right": EMPTY}
    # A crashed place leaves nobody judging drops: no walk, no second place.
    assert custody.blocked_arms(custodies, [], moves_body=True, all_arms=ARMS) == ("left",)
    assert custody.blocked_arms(custodies, ["left"], placing=True) == ("left",)
    assert custody.blocked_arms(custodies, ["right"]) == ()


def test_an_arm_with_a_running_goal_admits_nothing():
    # A pick before its lift still reads EMPTY, but its goal is running.
    picking = _c(custody.EMPTY, goal_active=True)
    assert custody.blocked_arms({"left": picking}, ["left"]) == ("left",)
    assert custody.blocked_arms({"left": picking, "right": EMPTY}, [], moves_body=True, all_arms=ARMS) == ("left",)
    assert custody.free_arms({"left": picking, "right": EMPTY}, ARMS) == ("right",)


def test_a_restarted_gateway_waits_until_no_arm_runs_anything_it_lost_the_lease_for():
    assert custody.settled_for_admission({"left": EMPTY, "right": HELD_BY_PICK}, ARMS)
    for busy in (_c(custody.EMPTY, goal_active=True), _c(custody.RECOVERING), _c(custody.HELD, PLACE), None):
        assert not custody.settled_for_admission({"left": EMPTY, "right": busy}, ARMS)
    assert not custody.settled_for_admission({"left": EMPTY}, ARMS)
    with pytest.raises(ValueError):
        custody.settled_for_admission({"left": EMPTY}, [])


def test_one_arms_problem_stops_that_arm_only():
    custodies = {"left": _c(custody.UNKNOWN, reason=custody.GRIP_UNCERTAIN), "right": EMPTY}
    assert custody.blocked_arms(custodies, ["right"]) == ()
    assert custody.blocked_arms(custodies, ["left"], placing=True) == ("left",)
    assert custody.blocked_arms(custodies, [], moves_body=True, all_arms=ARMS) == ("left",)
    assert custody.cancels_on_unknown("left", ["left"])
    assert custody.cancels_on_unknown("left", ["right"], moves_body=True)
    assert not custody.cancels_on_unknown("left", ["right"])
    assert not custody.cancels_on_unknown("left", ["left"], operator_command=True)


def test_a_body_motion_checks_every_configured_arm_even_before_their_custody_arrives():
    assert custody.blocked_arms({}, [], moves_body=True, all_arms=ARMS) == ARMS
    assert custody.blocked_arms({"left": EMPTY}, [], moves_body=True, all_arms=ARMS) == ("right",)
    with pytest.raises(ValueError):
        custody.blocked_arms({"left": EMPTY}, [], moves_body=True)


def test_only_the_picks_custody_judges_drops():
    assert custody.judges_drops(custody.HELD, PICK)
    assert not custody.judges_drops(custody.HELD, PLACE)
    # The put-back and the pick's own release open the hand on purpose.
    assert not custody.judges_drops(custody.RECOVERING, "")


def test_only_the_pick_hands_the_object_over_and_only_its_place_renews():
    assert custody.accepts_claim(custody.HELD, PICK, 30.0)
    assert custody.accepts_claim(custody.HELD, PICK, custody.CLAIM_LEASE_MAX_S)
    for lease in (0.0, -1.0, custody.CLAIM_LEASE_MAX_S + 1, float("inf"), float("nan")):
        assert not custody.accepts_claim(custody.HELD, PICK, lease)
        assert not custody.accepts_renewal(custody.HELD, PLACE, token_matches=True, lease_s=lease)
    assert not custody.accepts_claim(custody.HELD, PLACE, 30.0)
    assert not custody.accepts_claim(custody.RECOVERING, "", 30.0)
    # A long place renews with its own token; nobody else can.
    assert custody.accepts_renewal(custody.HELD, PLACE, token_matches=True, lease_s=30.0)
    assert not custody.accepts_renewal(custody.HELD, PLACE, token_matches=False, lease_s=30.0)
    assert not custody.accepts_renewal(custody.HELD, PICK, token_matches=True, lease_s=30.0)


def test_a_held_object_is_put_back_only_when_the_gateway_idles():
    assert not custody.needs_put_back(custody.HELD, PICK, "", idle_s=custody.IDLE_PUT_BACK_S - 0.1)
    assert custody.needs_put_back(custody.HELD, PICK, "", idle_s=custody.IDLE_PUT_BACK_S)
    assert custody.needs_put_back(custody.HELD, PICK, "", idle_s=5.0, limit_s=5.0)
    assert not custody.needs_put_back(custody.HELD, PICK, custody.NO_PUT_BACK, idle_s=1e6)
    assert not custody.needs_put_back(custody.HELD, PLACE, "", idle_s=1e6)
    for state in (custody.EMPTY, custody.RECOVERING, custody.UNKNOWN):
        assert not custody.needs_put_back(state, PICK, "", idle_s=1e6)


def test_a_refused_put_back_keeps_holding_and_lets_the_operator_take_over():
    assert custody.accepts_put_back(custody.HELD, PICK)
    assert not custody.accepts_put_back(custody.HELD, PLACE)
    assert custody.state_for_put_back(contact_known=True, body_moved=False) == (custody.RECOVERING, "", "")
    for contact_known, body_moved in ((True, True), (False, False)):
        result = custody.state_for_put_back(contact_known=contact_known, body_moved=body_moved)
        assert result == (custody.HELD, PICK, custody.NO_PUT_BACK)
        custody.validate_custody(*result)
        assert custody.blocked_arms({"left": _c(*result)}, [], moves_body=True, all_arms=("left",)) == ()
    # Otherwise the arm would hold forever if no place ever comes.
    for command in custody.OPERATOR_COMMANDS:
        assert custody.admits_operator_command(custody.HELD, command, owner=PICK, reason_code=custody.NO_PUT_BACK)
        assert not custody.admits_operator_command(custody.HELD, command, owner=PICK)
    takeover = custody.state_for_operator_takeover()
    custody.validate_custody(*takeover)
    assert takeover[0] == custody.UNKNOWN


@pytest.mark.parametrize("reason", sorted(custody.REVERIFIABLE_REASONS))
def test_an_unknown_custody_that_checks_out_goes_back_to_held_without_an_operator(reason):
    ok = dict(record_has_object=True, grip_fresh_held=True, arm_at_record_pose=True, no_motion_since=True)
    result = custody.state_after_reverify(reason, **ok)
    assert result == (custody.HELD, PICK, "")
    custody.validate_custody(*result)
    for name in ok:
        assert custody.state_after_reverify(reason, **{**ok, name: False}) is None


def test_a_reverified_custody_keeps_its_refused_put_back():
    ok = dict(record_has_object=True, grip_fresh_held=True, arm_at_record_pose=True, no_motion_since=True)
    result = custody.state_after_reverify(custody.GRIP_UNCERTAIN, **ok, no_put_back=True)
    assert result == (custody.HELD, PICK, custody.NO_PUT_BACK)
    assert not custody.needs_put_back(result[0], result[1], result[2], idle_s=1e6)


def test_a_faulty_record_never_reverifies():
    ok = dict(record_has_object=True, grip_fresh_held=True, arm_at_record_pose=True, no_motion_since=True)
    assert custody.state_after_reverify(custody.CUSTODY_FAULT, **ok) is None


def test_a_restart_is_empty_only_on_fresh_empty_hand_feedback():
    assert custody.state_at_startup(empty_evidence=True) == (custody.EMPTY, "", "")
    state, owner, reason = custody.state_at_startup(empty_evidence=False)
    custody.validate_custody(state, owner, reason)
    assert (state, reason) == (custody.UNKNOWN, custody.STARTUP_UNVERIFIED)
    assert reason in custody.REVERIFIABLE_REASONS


def test_the_operators_reset_is_checked_when_it_arrives_and_only_in_unknown():
    ok = dict(operator_confirmed=True, empty_evidence=True, settled_home=True)
    assert custody.state_after_reset(custody.UNKNOWN, **ok) == (custody.EMPTY, "", "")
    for state in (custody.HELD, custody.RECOVERING, custody.EMPTY):
        assert custody.state_after_reset(state, **ok) is None
    for name in ok:
        assert custody.state_after_reset(custody.UNKNOWN, **{**ok, name: False}) is None


def test_a_confirmed_drop_is_empty_and_returns_home_once_without_an_operator():
    state, owner, reason = custody.state_after_drop(empty_evidence=True)
    custody.validate_custody(state, owner, reason)
    assert (state, reason) == (custody.EMPTY, custody.DROPPED)
    assert custody.returns_home_after_drop(state, reason)
    assert not custody.admits_operator_command(state, custody.RETURN_HOME)
    assert not custody.returns_home_after_drop(custody.EMPTY, "")
    state, owner, reason = custody.state_after_drop(empty_evidence=False)
    custody.validate_custody(state, owner, reason)
    assert (state, reason) == (custody.UNKNOWN, custody.GRIP_UNCERTAIN)


def test_operator_commands_pass_in_unknown_and_never_elsewhere():
    for command in custody.OPERATOR_COMMANDS:
        assert custody.admits_operator_command(custody.UNKNOWN, command)
        assert custody.admits_operator_command("picked", command)  # unreadable state reads as UNKNOWN
        for state in (custody.EMPTY, custody.RECOVERING):
            assert not custody.admits_operator_command(state, command)
        assert not custody.admits_operator_command(custody.HELD, command, owner=PLACE)
    assert not custody.admits_operator_command(custody.UNKNOWN, "CLOSE_HAND")


@pytest.mark.parametrize(
    ("reported", "reason", "empty_evidence", "held_evidence", "expected"),
    [
        (custody.EMPTY, "", True, False, (custody.EMPTY, "", "")),
        (custody.EMPTY, "", False, False, (custody.UNKNOWN, "", custody.PLACE_UNCERTAIN)),
        (custody.EMPTY, custody.DROPPED, True, False, (custody.EMPTY, "", custody.DROPPED)),
        (custody.EMPTY, custody.DROPPED, False, False, (custody.UNKNOWN, "", custody.GRIP_UNCERTAIN)),
        (custody.HELD, "", False, True, (custody.HELD, PICK, "")),
        (custody.HELD, "", False, False, (custody.UNKNOWN, "", custody.PLACE_UNCERTAIN)),
        (custody.UNKNOWN, "", True, True, (custody.UNKNOWN, "", custody.PLACE_UNCERTAIN)),
    ],
)
def test_every_return_leads_to_a_publishable_state(reported, reason, empty_evidence, held_evidence, expected):
    result = custody.state_after_return(reported, reason, empty_evidence=empty_evidence, held_evidence=held_evidence)
    assert result == expected
    custody.validate_custody(*result)


@pytest.mark.parametrize(
    ("reported", "reason"),
    [(custody.RECOVERING, ""), ("released", ""), (custody.HELD, custody.DROPPED), (custody.EMPTY, "LOST")],
)
def test_a_place_cannot_report_anything_else(reported, reason):
    with pytest.raises(ValueError):
        custody.state_after_return(reported, reason, empty_evidence=True, held_evidence=True)


def test_refusal_codes_never_double_as_custody_reasons():
    reasons = set().union(*custody.STATE_REASONS.values())
    assert not custody.CUSTODY_ERRORS & reasons
    assert {PLACE} == custody.CLAIMANTS


def test_short_feedback_dropouts_do_not_stop_the_arm():
    assert 0 < custody.GRIP_FEEDBACK_GRACE_S < custody.CUSTODY_STALE_S < custody.IDLE_PUT_BACK_S


def test_epochs_differ_per_process_start():
    first, second = custody.new_custody_epoch(), custody.new_custody_epoch()
    assert first and second and first != second


def test_wire_message_carries_every_contract_field():
    from ibrobot_msgs.action import PickObject, PlaceObject
    from ibrobot_msgs.msg import HeldObject
    from ibrobot_msgs.srv import AuthorizeHeldReset, ClaimHeldObject, ReturnHeldObject

    held = HeldObject()
    for name in ("custody_epoch", "state", "owner", "reason_code", "arm_side", "goal_active", "contact_positions"):
        assert hasattr(held, name), name
    assert not hasattr(held, "grip") and not hasattr(held, "root_task_id")
    held.state = custody.EMPTY
    custody.validate_custody(held.state, held.owner, held.reason_code)
    assert custody.blocked_arms({"left": held}, ["left"]) == ()  # a message works as a custody
    for request in (
        ClaimHeldObject.Request(),
        ReturnHeldObject.Request(),
        AuthorizeHeldReset.Request(),
        PickObject.Goal(),
        PlaceObject.Goal(),
    ):
        for name in ("custody_id", "custody_epoch"):
            assert hasattr(request, name), (type(request).__name__, name)
        if not isinstance(request, PlaceObject.Goal):
            assert hasattr(request, "request_id"), type(request).__name__
    # A place names the custody and the arm it places, so two held objects never mix up.
    assert hasattr(PlaceObject.Goal(), "arm_side")
    assert hasattr(ReturnHeldObject.Request(), "state") and not hasattr(ReturnHeldObject.Request(), "outcome")
    assert not hasattr(AuthorizeHeldReset.Response(), "remaining_sec")
    for response in (ClaimHeldObject.Response(), ReturnHeldObject.Response(), AuthorizeHeldReset.Response()):
        assert hasattr(response, "error_code"), type(response).__name__
    assert not hasattr(ClaimHeldObject.Response(), "reason_code")
    assert not hasattr(ReturnHeldObject.Response(), "reason_code")
