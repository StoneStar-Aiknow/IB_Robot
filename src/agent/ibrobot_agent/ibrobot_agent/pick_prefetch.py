"""Start a pick's perception as soon as the Agent's plan names its target.

The pick executor's PrefetchGrasp service (manipulation_execution) captures and
perceives without moving anything, so it needs no admission: the Agent asks for
it once the plan is validated, before the Gateway admits and dispatches it. The
Gateway asks again when the pick starts; a request for the same target joins
the running prefetch and binds its task. Whatever is prefetched is used only by
a pick goal for the same target with the robot where it was at the capture.
"""

from __future__ import annotations

import logging

PICK_SKILL = "pick_object"
PLACE_SKILL = "place_in_container"

_LOGGER = logging.getLogger(__name__)


def first_pick(steps) -> tuple[str, str] | None:
    """(target, container) when the workflow starts with a pick, else None.

    Only a first step: a pick after other motions starts from another posture,
    so a capture made now would be refused anyway. The container is the one of
    a place of the same target right after it (skill_library.carry_handover
    retains_for_place), else ''.
    """
    if not steps or steps[0].skill_name != PICK_SKILL:
        return None
    target = steps[0].target_name.strip()
    if not target:
        return None
    container = ""
    if len(steps) > 1:
        place = steps[1]
        if (
            place.skill_name == PLACE_SKILL
            and place.target_name.strip().casefold() == target.casefold()
            and place.container_name.strip()
        ):
            container = place.container_name.strip()
    return target, container


class RosPickPrefetch:
    """``AgentService(workflow_planned=...)``: calls the pick executor's PrefetchGrasp, never waiting."""

    def __init__(self, node, service_name: str):
        from ibrobot_msgs.srv import PrefetchGrasp

        self._request_class = PrefetchGrasp.Request
        self._client = node.create_client(PrefetchGrasp, service_name)
        self._service_name = service_name

    def __call__(self, steps) -> None:
        request = first_pick(steps)
        if request is None:
            return
        if not self._client.service_is_ready():
            _LOGGER.info("pick prefetch skipped: %s is not available", self._service_name)
            return
        target, container = request
        # No dispatch binding yet: the Gateway's request at the pick's start binds the task.
        self._client.call_async(self._request_class(target_query=target, container_query=container))
        _LOGGER.info("pick prefetch requested at plan time: target=%r container=%r", target, container)
