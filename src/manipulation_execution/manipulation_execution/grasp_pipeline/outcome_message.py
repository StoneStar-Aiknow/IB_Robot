"""The ``PickObject`` result message: what an Agent needs to decide its next step.

``error_code`` stays the machine-readable code; the message says, in order:
the outcome, where the arm and the object are now, every retry the pipeline
made and why, what the caller should do next, and the last failure's detail.
An Agent passes the message on to the user and keeps it in its conversation,
so it is plain sentences without internal names.

    PICK FAILED [NO_EXECUTABLE_CANDIDATE] No safe grasp could be planned for the target.
    State: the arm did not move; the object was not touched.
    Retries (2): 1. reobserve after OBSERVATION_UNAVAILABLE: robot sensor data dropped out; ...
    Next: ask the user to move the object closer to the robot or turn it, then try again.
    Detail: RuntimeError: No candidate produced a complete three-segment plan
"""

from __future__ import annotations

from .pipeline import PickOutcome

#: What each code means, and what the caller should do next.
FAILURES: dict[str, tuple[str, str]] = {
    "INVALID_TARGET": (
        "The target name is empty or not usable.",
        "give a short object name, such as 'an apple'.",
    ),
    "TARGET_NOT_VISIBLE": (
        "The target was not seen by the head camera.",
        "check the object name, or ask the user to put the object in front of the robot, then try again.",
    ),
    "NO_EXECUTABLE_CANDIDATE": (
        "No safe grasp could be planned for the target.",
        "ask the user to move the object closer to the robot or turn it, then try again.",
    ),
    "REOBSERVE_REQUIRED": (
        "The scene kept changing while the grasp was planned.",
        "keep the object and the robot still, then try again.",
    ),
    "OBSERVATION_UNAVAILABLE": (
        "The robot's camera or joint data was not available.",
        "wait a few seconds and try again; if it repeats, the operator should check the robot's network and load.",
    ),
    "TARGET_MOVED": (
        "The object moved while the arm approached it; the arm went back.",
        "keep the object still, then try again.",
    ),
    "GRASP_VERIFICATION_FAILED": (
        "The hand did not keep hold of the object.",
        "look at the table again before retrying: the object may have moved or fallen.",
    ),
    "SKILL_TIMEOUT": (
        "The pick ran out of time.",
        "try again; if it repeats, the operator should check the robot's load.",
    ),
    "TASK_TIMEOUT": (
        "The task budget ran out before the pick started.",
        "try again with a new command.",
    ),
    "CANCELED": ("The pick was cancelled.", "nothing to do unless the user asks again."),
    "RPC_FAILED": (
        "A grasp service did not answer.",
        "try again later; if it repeats, the operator should check the grasp services.",
    ),
    "RPC_TIMEOUT": (
        "A grasp service did not answer in time.",
        "try again later; if it repeats, the operator should check the grasp services.",
    ),
    "NOT_READY": ("The grasp services are still starting.", "wait about a minute and try again."),
    "SERVICE_UNAVAILABLE": (
        "A grasp service is not running.",
        "the operator should check the grasp services.",
    ),
    "MOTION_FAILED": (
        "The arm did not follow its motion and was stopped.",
        "do not retry; the operator must check the robot first.",
    ),
    "WATCH_FAILED": (
        "The robot could no longer see whether the target moved and stopped the arm.",
        "do not retry; the operator must check the robot first.",
    ),
    "EFFECTOR_FAILED": (
        "The hand did not finish its action and the arm was stopped.",
        "do not retry; the operator must check the robot first.",
    ),
    "INTERNAL_ERROR": (
        "The grasp software failed unexpectedly.",
        "do not retry; report it to the operator.",
    ),
}
_UNKNOWN = ("The pick failed.", "report it to the operator before retrying.")

#: Why the pipeline retried, per retry kind and code.
RETRY_REASONS: dict[str, str] = {
    "OBSERVATION_UNAVAILABLE": "robot sensor data dropped out",
    "NO_EXECUTABLE_CANDIDATE": "no safe grasp could be planned from that view",
    "REOBSERVE_REQUIRED": "the scene or the robot changed while the grasp was planned",
    "TARGET_MOVED": "the object moved during the approach and the arm went back",
    "GRASP_VERIFICATION_FAILED": "the hand did not keep hold of the object",
}
_RETRY_KINDS = {
    "reobserve": "looked again",
    "replan": "planned again",
    "regrasp": "grasped again",
}
#: Detail beyond this many characters is cut; the full text is in the pipeline report.
DETAIL_MAX = 300


def outcome_message(outcome: PickOutcome) -> str:
    lines = []
    if outcome.success:
        lines.append(f"PICK SUCCEEDED. {_success_summary(outcome)}")
    else:
        what, _ = FAILURES.get(outcome.error_code, _UNKNOWN)
        lines.append(f"PICK FAILED [{outcome.error_code or 'UNKNOWN'}] {what}")
    lines.append(f"State: {_state(outcome)}")
    if outcome.retries:
        lines.append(
            f"Retries ({len(outcome.retries)}): " + "; ".join(_retry(i, r) for i, r in enumerate(outcome.retries, 1))
        )
    if not outcome.success:
        lines.append(f"Next: {FAILURES.get(outcome.error_code, _UNKNOWN)[1]}")
        if outcome.message:
            lines.append(f"Detail: {_cut(outcome.message)}")
    return "\n".join(lines)


def _success_summary(outcome: PickOutcome) -> str:
    if outcome.released_after_success:
        return "The object was grasped, lifted and put back where it was."
    return "The object was grasped and lifted."


def _executed(outcome: PickOutcome) -> bool:
    return bool(outcome.report.get("executions"))


def _state(outcome: PickOutcome) -> str:
    if not outcome.at_rest:
        held = " and may still hold the object" if outcome.held else ""
        return f"the arm was stopped where it was{held}; it needs the operator before the next command."
    if outcome.success:
        if outcome.released_after_success:
            return "the arm is back at home with the hand open; the object is on the table."
        return "the hand holds the object."
    if not _executed(outcome):
        return "the arm did not move; the object was not touched."
    last = (outcome.report["executions"][-1] or {}).get("phases") or []
    lost = any(phase.get("state") == "grip_lost" for phase in last)
    if outcome.error_code == "GRASP_VERIFICATION_FAILED":
        if lost:
            return "the object slipped out after the lift and may have moved or fallen; the arm is back at home, hand open."
        return "the hand closed on nothing (the object may have been pushed); the arm is back at home, hand open."
    if outcome.error_code == "TARGET_MOVED":
        return "the arm went back to home; the hand is open."
    return "the arm is back at home with the hand open; an earlier attempt touched the object (see retries)."


def _retry(index: int, retry: dict) -> str:
    code = retry.get("code", "")
    kind = _RETRY_KINDS.get(retry.get("retry", ""), "tried again")
    reason = RETRY_REASONS.get(code) or _cut(retry.get("message", ""), 120) or code
    return f"{index}. {kind} after {code}: {reason}"


def _cut(text: str, limit: int = DETAIL_MAX) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
