"""Only an adjacent, same-target place step asks pick to retain custody."""

# HeldObject states in which a pick executor still owns a carried object.
HELD_CUSTODY_STATES = frozenset({"picked", "claimed", "recovering", "uncertain"})


def motion_stop_is_unknown(result):
    """Controller cleanup uncertainty is separate from object custody uncertainty."""
    if getattr(result, "error_code", "") in {
        "CANCEL_CLEANUP_TIMEOUT",
        "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
        "SKILL_CANCEL_TIMEOUT",
        "POSE_STOP_UNKNOWN",
    }:
        return True
    detail = getattr(result, "execution_detail", None)
    return getattr(detail, "schema_version", 0) in {1, 2} and detail.recovery_state == "UNKNOWN"


def retains_for_place(steps, index):
    if not 0 <= index < len(steps) - 1:
        return False
    pick, place = steps[index : index + 2]
    return bool(
        pick.skill_name == "pick_object"
        and place.skill_name == "place_in_container"
        and pick.target_name.strip()
        and pick.target_name.strip().casefold() == place.target_name.strip().casefold()
        and place.container_name.strip()
    )
