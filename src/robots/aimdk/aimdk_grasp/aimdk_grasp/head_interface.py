"""The interfaces of the X2 grasp head: names both hosts agree on.

The head (``aimdk_grasp.grasp_head``) runs beside the head camera and the CUDA
GPU (the Orin): capture and registration, the collision scene, cuRobo and the
target watch. The grasp pipeline (``aimdk_grasp.remote_head``) runs beside the
runtime and the models (the 310P or 203) and calls it under
``backend_options.head.namespace`` (``aimdk_grasp.split``), with the
``aimdk_grasp_msgs`` types.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NAMESPACE = "/x2_grasp_head"
#: How often the head publishes the armed watch's status (and at once when the target moved).
STATUS_PERIOD_S = 0.1
#: Statuses queued per drain; the pipeline host drains at every poll of its executor.
STATUS_DEPTH = 10


def session_id(client: str, number: int) -> str:
    """A watch session's name: the pipeline process that armed it and its arm count there (1, 2, ...).

    The count orders one client's sessions, so the head can refuse a command
    that arrives after a newer session of the same client (a timed-out ARM, a
    late DISARM); a restarted pipeline process is a new client.
    """
    return f"{client}:{int(number)}"


def parse_session(text: str) -> tuple[str, int]:
    """``(client, number)`` of a session_id; ValueError when it is not one."""
    client, _, number = str(text).rpartition(":")
    if not client or not number.isdigit() or int(number) < 1:
        raise ValueError(f"not a watch session id: {text!r}")
    return client, int(number)


#: The head is still warming up (cuRobo compiles its kernels): no capture yet.
NOT_READY = "NOT_READY"
#: Nothing supports the target in the scene: usually not the target at all.
NO_SUPPORT = "NO_SUPPORT"
#: The capture, scene or plan named is not (or no longer) kept by the head.
UNKNOWN = "UNKNOWN_REQUEST"


@dataclass(frozen=True)
class HeadEndpoints:
    capture: str
    scene: str
    plan: str
    refine: str
    watch: str
    status: str
    plan_place: str


def head_endpoints(namespace: str = DEFAULT_NAMESPACE) -> HeadEndpoints:
    root = "/" + namespace.strip("/")
    return HeadEndpoints(
        capture=f"{root}/capture",
        scene=f"{root}/complete_scene",
        plan=f"{root}/plan",
        refine=f"{root}/refine",
        watch=f"{root}/watch",
        status=f"{root}/watch_status",
        plan_place=f"{root}/plan_place",
    )
