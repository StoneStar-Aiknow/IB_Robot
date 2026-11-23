"""Held-object custody vocabulary shared by its owner and the Capability Gateway.

The executor that owns custody publishes ``ibrobot_msgs/HeldObject``; the
Gateway admits motion from that topic only. Both take every state, owner,
reason, command and refusal code from this module, so neither keeps its own
copy of the other's strings. Refusals go in a response's ``error_code``;
``reason_code`` always means the custody's own reason.

Custody is kept per arm (``HeldObject.arm_side``, the arm's configured side
name, also on single-arm robots): a problem with one arm stops that arm only,
and the other keeps working. The Gateway still runs one root task at a time
(its single execution lease): "keeps working" means tasks on the other arm are
admitted, not that both arms move at once.

Four states, and who acts on each:

* ``EMPTY``: nothing is held; new tasks are admitted. Entered only on fresh
  empty-hand feedback, never on a timeout or a lost response, and after a
  restart only that way too, whatever the record says (``state_at_startup``).
  ``DROPPED`` marks a drop the feedback confirmed: the Gateway opens the hand
  and returns home once, before it admits another root task. The owner clears
  ``DROPPED`` when the arm's next pick starts.
* ``HELD``: an object is held by its ``pick``, or by the ``place`` that claimed
  it. Holding is a safe state, so nothing hurries it. While the pick holds it,
  whatever root task asks, the arm admits only body motions (carrying it) and
  a place of this very object (``blocked_arms``; the place names the custody);
  a pick never picks with it (``free_arms``). While a place holds it, the arm
  admits nothing, not even a walk, until the place gives it back: a crashed
  place leaves nobody judging drops until its claim expires. A place renews
  its claim with its token (``accepts_renewal``). A place that cannot place it
  gives it back to the pick. Only after the Gateway has been idle (no root
  task running at all) for ``IDLE_PUT_BACK_S`` with the pick still holding it
  does the Gateway put it back (``needs_put_back``): release at the verified
  source contact, then return home. When the contact is not verified or the
  body (base, torso) moved since the grasp, the same arm joints would release
  it somewhere else, so the owner refuses: the object stays held with
  ``NO_PUT_BACK`` and the operator is alerted (``state_for_put_back``); the
  operator may then take over with ``OPERATOR_COMMANDS``. The owner's record
  keeps the body pose at the grasp and the ``NO_PUT_BACK`` latch, so both
  survive a restart and a re-verification. While the Gateway is lost the
  object stays held. Only the pick's custody judges drops (``judges_drops``).
* ``RECOVERING``: the owner opens the hand on purpose: the put-back, or the
  pick's own release after success (``release_after_success``); a pick that
  does not release keeps holding the object (HELD). It runs under the Gateway's lease
  and a total deadline, and can be stopped. Success (the arm back home, fresh
  empty-hand feedback) ends in ``EMPTY``; anything else ends in ``UNKNOWN``.
  There is no second automatic put-back.
* ``UNKNOWN``: anything that cannot be told for sure about that arm. The
  Gateway cancels the motions still running on it (``cancels_on_unknown``; a
  motion of the body uses every arm) and admits nothing on it but the
  operator's ``OPERATOR_COMMANDS`` for that arm (open the hand, return home),
  each as often as needed and under the Gateway's single lease, so the other
  arm stands still meanwhile. A return home plans with the hand's held object:
  its geometry when known, otherwise a bound for the largest object the hand
  may hold; the other arm's later tasks treat the stopped arm and that object
  the same way. There are two ways out:

  - back to ``HELD`` on its own, without an operator, only when the record
    describes the object, fresh feedback shows it still gripped, the arm is
    measured at the recorded pose and nothing moved the arm since the custody
    became UNKNOWN (``state_after_reverify``);
  - to ``EMPTY`` when the operator confirms it (``AuthorizeHeldReset``) at a
    moment when fresh feedback shows an empty hand and the arm has settled at
    the robot's configured home (never a home read from the custody record,
    which may be the very thing that failed); otherwise the confirmation is
    refused (``state_after_reset``).

  While the hand's feedback is lost the state stays ``UNKNOWN`` until it is
  repaired: an empty hand cannot be proven (on a robot that detects its hands
  only at boot, that means a reboot).

Only an operator identity may send ``OPERATOR_COMMANDS``; an Agent's caller
scope never qualifies, whatever its name.

The owner persists its record before every publish, republishes every
``CUSTODY_REPUBLISH_S``, always publishes a non-empty ``custody_id`` (a fresh
one for a custody it cannot match, e.g. after a restart) and sets
``goal_active`` while a goal of its own (a pick, a put-back) runs on the arm.
The Gateway measures a custody's age from when it received it (never from the
stamp: robot and host clocks differ) and ignores the first, latched sample
after it subscribes; an older or missing custody reads as UNKNOWN
(``fresh_custody``), so a dead owner's last message never admits motion. A
Gateway that starts admits nothing until every configured arm reports a
settled custody (``settled_for_admission``): a running goal it no longer holds
a lease for must end first.

For the Gateway (a later change): the put-back, the operator commands and the
return home after a drop are motions the Gateway itself starts, so they need
their own execution-lease owner kinds and a dispatch binding of their own
(``PickObject`` ``MODE_RECOVER`` carries one like any delegated goal).

No ROS or package dependencies.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping

SCHEMA_VERSION = 1

# ---- states ----
EMPTY = "EMPTY"
HELD = "HELD"
RECOVERING = "RECOVERING"
UNKNOWN = "UNKNOWN"
STATES = frozenset({EMPTY, HELD, RECOVERING, UNKNOWN})

# ---- owners: who holds a HELD object; UNKNOWN keeps the last one, if known ----
OWNER_PICK = "pick"
OWNER_PLACE = "place"
OWNERS = frozenset({OWNER_PICK, OWNER_PLACE})
STATE_OWNERS = {
    EMPTY: frozenset({""}),
    HELD: OWNERS,
    RECOVERING: frozenset({""}),
    UNKNOWN: frozenset({"", *OWNERS}),
}

# ---- reasons ----
#: EMPTY: the object dropped and fresh feedback confirms the hand empty.
DROPPED = "DROPPED"
#: HELD: the automatic put-back was refused (no verified contact, or the body
#: moved since the grasp); it stays held, no further automatic put-back, and
#: the operator is alerted and may take over.
NO_PUT_BACK = "NO_PUT_BACK"
#: UNKNOWN: a restart without fresh empty-hand feedback.
STARTUP_UNVERIFIED = "STARTUP_UNVERIFIED"
#: UNKNOWN: the hand's feedback cannot tell whether it holds the object (stale
#: or lost for longer than GRIP_FEEDBACK_GRACE_S, or a drop it cannot confirm).
GRIP_UNCERTAIN = "GRIP_UNCERTAIN"
#: UNKNOWN: the place cannot tell what happened (it said so, its claim expired,
#: a claim or return answer was lost, or its HELD report did not check out).
PLACE_UNCERTAIN = "PLACE_UNCERTAIN"
#: UNKNOWN: a pick stopped mid-motion with an uncertain hand or arm.
PICK_INTERRUPTED = "PICK_INTERRUPTED"
#: UNKNOWN: the put-back or the pick's own release failed or was stopped.
RECOVERY_FAILED = "RECOVERY_FAILED"
#: UNKNOWN: the operator took over a HELD (NO_PUT_BACK) arm with a command.
OPERATOR_TAKEOVER = "OPERATOR_TAKEOVER"
#: UNKNOWN: the custody state machine itself failed; its record is not trusted.
CUSTODY_FAULT = "CUSTODY_FAULT"
STATE_REASONS = {
    EMPTY: frozenset({"", DROPPED}),
    HELD: frozenset({"", NO_PUT_BACK}),
    RECOVERING: frozenset({""}),
    UNKNOWN: frozenset(
        {
            STARTUP_UNVERIFIED,
            GRIP_UNCERTAIN,
            PLACE_UNCERTAIN,
            PICK_INTERRUPTED,
            RECOVERY_FAILED,
            OPERATOR_TAKEOVER,
            CUSTODY_FAULT,
        }
    ),
}
#: UNKNOWN reasons a re-verification may end (not a fault of the record itself).
REVERIFIABLE_REASONS = STATE_REASONS[UNKNOWN] - {CUSTODY_FAULT}

#: The put-back runs PickObject MODE_RECOVER: release at the verified source
#: contact, then return home. It is the only automatic recovery.
PUT_BACK_SKILL = "pick_object"
#: How long the Gateway must be idle, with the pick still holding the object,
#: before it puts it back. Configurable.
IDLE_PUT_BACK_S = 60.0
#: How long hand feedback may be stale or unreadable while HELD before the
#: custody becomes UNKNOWN (GRIP_UNCERTAIN): short dropouts do not stop the arm.
GRIP_FEEDBACK_GRACE_S = 1.0
#: The owner republishes its custody at least this often.
CUSTODY_REPUBLISH_S = 1.0
#: The Gateway reads a custody received longer ago than this as UNKNOWN.
CUSTODY_STALE_S = 3.0
#: The longest lease a place may claim or renew (ClaimHeldObject.lease_s).
CLAIM_LEASE_MAX_S = 120.0

# ---- operator commands, each naming one arm ----
#: Open the hand where it is (a plain command: the operator is ready for it).
OPEN_HAND = "OPEN_HAND"
#: Return the arm home on a path checked against the current scene.
RETURN_HOME = "RETURN_HOME"
OPERATOR_COMMANDS = frozenset({OPEN_HAND, RETURN_HOME})

#: Only a place claims; the put-back runs PickObject MODE_RECOVER instead.
CLAIMANTS = frozenset({OWNER_PLACE})
#: What a place may report when it gives the custody back (ReturnHeldObject):
#: EMPTY (with DROPPED if it fell), HELD (with the arm back at the handover pose
#: it claimed it at; the owner checks) or UNKNOWN.
RETURN_STATES = frozenset({EMPTY, HELD, UNKNOWN})

# ---- why a custody service refuses (the response's error_code) ----
#: The request was decided on something gone: another custody, owner epoch or
#: claim token.
STALE_REQUEST = "STALE_REQUEST"
#: The current state does not allow it (e.g. a reset while the hand is not
#: shown empty at home).
WRONG_STATE = "WRONG_STATE"
#: The claim names another target than the one held.
TARGET_MISMATCH = "TARGET_MISMATCH"
#: Malformed: unknown claimant or state, lease out of range, a reset without
#: the operator's explicit confirmation and note.
INVALID_REQUEST = "INVALID_REQUEST"
#: The same request_id came back with a different payload.
REQUEST_CONFLICT = "REQUEST_CONFLICT"
CUSTODY_ERRORS = frozenset({STALE_REQUEST, WRONG_STATE, TARGET_MISMATCH, INVALID_REQUEST, REQUEST_CONFLICT})


def new_custody_epoch() -> str:
    """A fresh epoch for one custody owner process; never reused after a restart."""
    return uuid.uuid4().hex


def normalize_state(value: object) -> str:
    """A tolerant consumer's view: anything not a known state is UNKNOWN."""
    return value if isinstance(value, str) and value in STATES else UNKNOWN


def fresh_custody(custody: object, age_s: float, *, first_sample: bool = False, limit_s: float = CUSTODY_STALE_S):
    """The Gateway's custody of one arm, or None (read as UNKNOWN).

    ``age_s`` counts from when the Gateway received it; the first, latched
    sample after subscribing is ignored until the owner publishes again.
    """
    fresh = not first_sample and math.isfinite(age_s) and 0 <= age_s <= limit_s
    return custody if fresh else None


def _view(custody: object) -> tuple[str, str, str, bool]:
    if custody is None:
        return UNKNOWN, "", "", False
    return (
        normalize_state(getattr(custody, "state", None)),
        getattr(custody, "owner", ""),
        getattr(custody, "reason_code", ""),
        getattr(custody, "goal_active", False) is True,
    )


def blocks_new_tasks(state: object) -> bool:
    """True in every state but EMPTY, for one arm's custody."""
    return normalize_state(state) != EMPTY


def blocked_arms(
    custodies: Mapping[str, object],
    task_arms: Iterable[str],
    *,
    placing: bool = False,
    moves_body: bool = False,
    all_arms: Iterable[str] = (),
) -> tuple:
    """The arms whose custody keeps a task out; empty means it may run.

    ``custodies`` maps each arm side to its custody (anything with ``state``,
    ``owner`` and ``goal_active``, e.g. a HeldObject message, already through
    ``fresh_custody``); an arm without one reads as UNKNOWN. ``task_arms`` are
    the arms the task moves; ``placing`` says it places the object those arms
    hold (the place names the custody). A task that moves the body uses every
    arm of the robot, so ``all_arms`` (from the robot's configuration, not from
    the custodies seen so far) is required then. An arm with a running goal
    admits nothing; a HELD arm the pick holds admits a body motion and a place;
    a HELD arm a place holds, RECOVERING and UNKNOWN admit nothing.
    """
    task_arms = set(task_arms)
    used = set(task_arms)
    if moves_body:
        all_arms = set(all_arms)
        if not all_arms:
            raise ValueError("a body motion needs the robot's configured arms")
        used |= all_arms

    def blocks(arm):
        state, owner, _, goal_active = _view(custodies.get(arm))
        if goal_active:
            return True
        if state == HELD:
            return owner != OWNER_PICK or (arm in task_arms and not placing)
        return blocks_new_tasks(state)

    return tuple(sorted(arm for arm in used if blocks(arm)))


def free_arms(custodies: Mapping[str, object], candidates: Iterable[str]) -> tuple:
    """The candidate arms a pick may choose from (an empty, idle hand each)."""
    free = []
    for arm in candidates:
        state, _, _, goal_active = _view(custodies.get(arm))
        if state == EMPTY and not goal_active:
            free.append(arm)
    return tuple(free)


def settled_for_admission(custodies: Mapping[str, object], all_arms: Iterable[str]) -> bool:
    """Whether a Gateway that just started may admit anything: every configured
    arm reports a fresh custody with no goal running, no place holding it and
    no recovery under way (it holds no lease for any of them)."""
    all_arms = tuple(all_arms)
    if not all_arms:
        raise ValueError("admission needs the robot's configured arms")
    for arm in all_arms:
        if arm not in custodies or custodies[arm] is None:
            return False
        state, owner, _, goal_active = _view(custodies[arm])
        if goal_active or state == RECOVERING or (state == HELD and owner == OWNER_PLACE):
            return False
    return True


def cancels_on_unknown(
    unknown_arm: str, motion_arms: Iterable[str], *, moves_body: bool = False, operator_command: bool = False
) -> bool:
    """Whether a running motion is cancelled when ``unknown_arm`` becomes UNKNOWN."""
    return not operator_command and (moves_body or unknown_arm in set(motion_arms))


def judges_drops(state: object, owner: object) -> bool:
    """Whether the owner's own feedback may declare a drop: only while the pick
    holds it (a place, the put-back and the pick's own release open the hand on
    purpose; a place reports drops itself)."""
    return normalize_state(state) == HELD and owner == OWNER_PICK


def _lease_ok(lease_s: float) -> bool:
    return math.isfinite(lease_s) and 0 < lease_s <= CLAIM_LEASE_MAX_S


def accepts_claim(state: object, owner: object, lease_s: float) -> bool:
    """Whether the owner hands the custody to a place (epoch, custody_id and
    target matching): only from the pick, for a lease in (0, CLAIM_LEASE_MAX_S]."""
    return normalize_state(state) == HELD and owner == OWNER_PICK and _lease_ok(lease_s)


def accepts_renewal(state: object, owner: object, *, token_matches: bool, lease_s: float) -> bool:
    """Whether the owner renews a place's claim: the place holds it and presents
    its own token, for a lease in (0, CLAIM_LEASE_MAX_S]."""
    held_by_place = normalize_state(state) == HELD and owner == OWNER_PLACE
    return held_by_place and token_matches and _lease_ok(lease_s)


def needs_put_back(
    state: object, owner: object, reason_code: object, idle_s: float, limit_s: float = IDLE_PUT_BACK_S
) -> bool:
    """Whether the Gateway puts the object back now: the pick still holds it, no
    put-back was refused, and the Gateway has run no root task for ``idle_s``."""
    held_by_pick = normalize_state(state) == HELD and owner == OWNER_PICK
    return held_by_pick and reason_code != NO_PUT_BACK and idle_s >= limit_s


def accepts_put_back(state: object, owner: object) -> bool:
    """Whether the owner takes a put-back request (custody_id and epoch matching):
    only while the pick holds it. It then moves to ``state_for_put_back``."""
    return normalize_state(state) == HELD and owner == OWNER_PICK


def state_for_put_back(*, contact_known: bool, body_moved: bool) -> tuple[str, str, str]:
    """(state, owner, reason_code) once the owner took a put-back request:
    RECOVERING, or still HELD with NO_PUT_BACK when the same arm joints would
    release it somewhere else. ``body_moved`` compares the body pose now with
    the one the record kept at the grasp."""
    if contact_known and not body_moved:
        return RECOVERING, "", ""
    return HELD, OWNER_PICK, NO_PUT_BACK


def state_at_startup(*, empty_evidence: bool) -> tuple[str, str, str]:
    """(state, owner, reason_code) after an owner restart, whatever the record
    says; a record describing the object may then re-verify it."""
    return (EMPTY, "", "") if empty_evidence else (UNKNOWN, "", STARTUP_UNVERIFIED)


def state_after_reverify(
    reason_code: str,
    *,
    record_has_object: bool,
    grip_fresh_held: bool,
    arm_at_record_pose: bool,
    no_motion_since: bool,
    no_put_back: bool = False,
) -> tuple[str, str, str] | None:
    """(HELD, pick, reason) when an UNKNOWN custody checks out as still held; else None.

    Every condition must hold: a trusted record describing the object, fresh
    feedback showing it gripped, the arm measured at the recorded pose, and no
    motion of that arm (operator commands included) since it became UNKNOWN.
    After an owner restart (STARTUP_UNVERIFIED) the owner cannot see the time
    it was down, so ``no_motion_since`` must come from a source that survived
    the restart (e.g. the runtime's command log), else it is False. A record
    with the ``NO_PUT_BACK`` latch (``no_put_back``) keeps it.
    """
    checks = (record_has_object, grip_fresh_held, arm_at_record_pose, no_motion_since)
    if reason_code in REVERIFIABLE_REASONS and all(checks):
        return HELD, OWNER_PICK, NO_PUT_BACK if no_put_back else ""
    return None


def state_after_reset(
    state: object, *, operator_confirmed: bool, empty_evidence: bool, settled_home: bool
) -> tuple | None:
    """(EMPTY, "", "") when the operator's confirmation arrives for an UNKNOWN
    custody while fresh feedback shows an empty hand settled at the configured
    home; else None (refused with WRONG_STATE, nothing is remembered)."""
    ok = normalize_state(state) == UNKNOWN and operator_confirmed and empty_evidence and settled_home
    return (EMPTY, "", "") if ok else None


def returns_home_after_drop(state: object, reason_code: object) -> bool:
    """Whether the Gateway opens the hand and returns home (once per custody_id)."""
    return normalize_state(state) == EMPTY and reason_code == DROPPED


def admits_operator_command(state: object, command: object, *, owner: object = "", reason_code: object = "") -> bool:
    """An operator command for one arm passes while that arm's custody is
    UNKNOWN, or HELD by the pick with NO_PUT_BACK (the owner then moves it to
    ``state_for_operator_takeover`` before the command runs)."""
    if command not in OPERATOR_COMMANDS:
        return False
    state = normalize_state(state)
    return state == UNKNOWN or (state == HELD and owner == OWNER_PICK and reason_code == NO_PUT_BACK)


def state_for_operator_takeover() -> tuple[str, str, str]:
    """(state, owner, reason_code) when the operator takes over a HELD arm."""
    return UNKNOWN, OWNER_PICK, OPERATOR_TAKEOVER


def state_after_drop(*, empty_evidence: bool) -> tuple[str, str, str]:
    """(state, owner, reason_code) once a held object is judged dropped."""
    return (EMPTY, "", DROPPED) if empty_evidence else (UNKNOWN, "", GRIP_UNCERTAIN)


def state_after_return(
    state: str, reason_code: str = "", *, empty_evidence: bool, held_evidence: bool
) -> tuple[str, str, str]:
    """(state, owner, reason_code) after a place gives the custody back.

    The owner checks the place's report: EMPTY needs fresh empty-hand feedback
    (``empty_evidence``); HELD needs fresh grip feedback and the arm measured at
    the handover pose (``held_evidence``), and goes back to the pick.
    """
    if state not in RETURN_STATES or reason_code not in {"", DROPPED} or (reason_code and state != EMPTY):
        raise ValueError(f"a place cannot return {state!r} with {reason_code!r}")
    if state == HELD and held_evidence:
        return HELD, OWNER_PICK, ""
    if state == EMPTY and reason_code == DROPPED:
        return state_after_drop(empty_evidence=empty_evidence)
    if state == EMPTY and empty_evidence:
        return EMPTY, "", ""
    return UNKNOWN, "", PLACE_UNCERTAIN


def validate_custody(state: str, owner: str, reason_code: str = "") -> None:
    """Raise ValueError for a combination the custody owner must never publish."""
    if state not in STATES:
        raise ValueError(f"unknown custody state {state!r}")
    if owner not in STATE_OWNERS[state]:
        raise ValueError(f"owner {owner!r} is not allowed in {state}")
    if reason_code not in STATE_REASONS[state]:
        raise ValueError(f"reason {reason_code!r} is not allowed in {state}")
    if reason_code == NO_PUT_BACK and owner != OWNER_PICK:
        raise ValueError("only the pick's custody is put back")
