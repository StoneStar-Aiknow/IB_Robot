"""Launch wiring for the X2 mirror photo node (``manipulation_execution.mirror_photo_node``).

``mirror_session`` appends the photo node from here, after the voice node,
when ``mirror_session.photo.enabled`` is true (feat/x2-photo-capture). Photos
are an extra and the mirror must start without them: a wrong ``photo``
section, no camera, or any other error while working out the node only logs a
WARNING and leaves the node out; it never fails the launch. Without a
``photo`` block, or with ``enabled: false``, the launch is exactly what it was
before the feature.

The section is checked by ``mirror_photo.config``, which the node reads again
from ``photo_json``. The node's inputs are the session's: the camera (the
compressed stream PEAR reads, the half-resolution one on the X2 entries) and
the PEAR frames; the hand frames only when the retarget player follows the
hands, otherwise the gesture trigger has nothing to read and stays quiet. The
parameter names are listed here rather than imported, as the operator's and
the voice's are; ``test_mirror_photo_launch`` holds them against the node's
declaration. Every import is function-local or stdlib: the module stays
within the registry import budget.
"""

from __future__ import annotations

from collections.abc import Mapping

PHOTO_OFF = "no photo node, the mirror starts without photos"


def photo_parameters(robot_config: Mapping, config: Mapping, session: Mapping, warn) -> dict | None:
    """The photo node's parameters, or None if it is not to start; ``warn`` says why when it is wrong.

    ``config`` is ``mirror_session``; ``session`` the session node's parameters
    (its camera, PEAR topic and status period).
    """
    from manipulation_execution.launch_builders.motion_imitation_stream import mock_hand_frame_topic
    from manipulation_execution.mirror_photo.config import SECTION_PATH, parse_photo_section, to_json

    section = config.get("photo")
    if section is None:
        return None
    try:
        photo = parse_photo_section(section)
    except ValueError as exc:
        warn(f"{exc}; {PHOTO_OFF}")
        return None
    if not photo.enabled:
        return None
    camera_topic = session.get("camera_topic", "")
    if not camera_topic:
        warn(f"{SECTION_PATH}.enabled is true but motion_imitation_stream names no compressed camera; {PHOTO_OFF}")
        return None
    pear_frame_topic = session.get("pear_frame_topic", "")
    if not pear_frame_topic:
        warn(f"{SECTION_PATH}: no PEAR frame topic, so no held-pose trigger and no person box")
    return {
        "photo_json": to_json(photo),
        "camera_topic": camera_topic,
        "pear_frame_topic": pear_frame_topic,
        "hand_frame_topic": mock_hand_frame_topic(robot_config),
        "status_period_sec": float(session["status_period_sec"]),
    }


__all__ = ["PHOTO_OFF", "photo_parameters"]
