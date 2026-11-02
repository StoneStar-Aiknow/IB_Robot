"""The X2 grasp on two hosts: a temporary deployment of the X2, nothing other robots use.

The robot-side host (the 310P or 203) runs the runtime, the Gateway, the grasp
models and manipulation_execution's grasp pipeline with
``aimdk_grasp.remote_head:x2_remote_grasp_backends``. The Orin runs the grasp
head (``aimdk_grasp.grasp_head``): the head camera's capture and registration,
the collision scene, cuRobo and the target watch, beside the camera's raw depth
and the CUDA GPU. ``launch/x2_grasp_head.launch.py`` starts the head with the
same robot configuration and binds it to the runtime the other host runs.

Everything about the split is configured in ``grasp_execution.backend_options.head``
and read here; the robot-agnostic pipeline and launch builders know nothing of
it. Ending the split means selecting ``aimdk_grasp.backend:x2_grasp_backends``
and dropping the ``head`` block in the X2 configuration.

``backend_options.head``:

- ``namespace``: where the head serves (default ``/x2_grasp_head``).
- ``status_timeout_s``: how long the pipeline host goes without a watch status
  of its session before the watch counts as unavailable, on its own clock.
- ``cpu_isolation``: ``{node, others}``, Linux CPU lists ("4-7", "0,2-3") for
  the head node and for every other process its launch starts; they must not
  overlap.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .head_interface import DEFAULT_NAMESPACE, STATUS_PERIOD_S

REMOTE_BACKEND = "aimdk_grasp.remote_head:x2_remote_grasp_backends"
HEAD_EXECUTABLE = "x2_grasp_head_node"
RUNTIME_PROVIDER = "aimdk_robot"
#: Robot types the grasp head plans for: the left arm with the left OmniHand.
ROBOT_TYPES = frozenset({"x2_ultra_omnihand"})
#: Ten status periods. On the robot network the runtime's views reached the
#: Orin at 50 Hz, ~5 ms old (2026-10-22); the head publishes every STATUS_PERIOD_S.
STATUS_TIMEOUT_S = 1.0
#: Longer than this without a status and the arm would approach blind for too long to call it watched.
MAX_STATUS_TIMEOUT_S = 10.0

_CPU_RANGE = re.compile(r"(\d+)(?:-(\d+))?")


def parse_cpu_list(text: str) -> frozenset[int]:
    """The CPUs a Linux CPU list ("4-7", "0,2-3") names; raises ValueError when malformed or empty."""
    cpus: set[int] = set()
    for part in str(text).split(","):
        match = _CPU_RANGE.fullmatch(part.strip())
        if match is None:
            raise ValueError(f"malformed CPU list {text!r}")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise ValueError(f"malformed CPU list {text!r}")
        cpus.update(range(first, last + 1))
    return frozenset(cpus)


@dataclass(frozen=True)
class CpuIsolation:
    node: frozenset[int]
    others: frozenset[int]


@dataclass(frozen=True)
class HeadOptions:
    namespace: str = DEFAULT_NAMESPACE
    status_timeout_s: float = STATUS_TIMEOUT_S
    cpu_isolation: CpuIsolation | None = None


def head_options(options: Mapping[str, Any]) -> HeadOptions:
    """``backend_options.head`` checked; raises ValueError naming what is wrong."""
    given = options.get("head") or {}
    if not isinstance(given, Mapping):
        raise ValueError("backend_options.head must be a mapping")
    unknown = sorted(set(given) - {"namespace", "status_timeout_s", "cpu_isolation"})
    if unknown:
        raise ValueError(f"backend_options.head has unknown keys {unknown}")
    namespace = given.get("namespace", DEFAULT_NAMESPACE)
    if not isinstance(namespace, str) or not namespace.strip("/"):
        raise ValueError("backend_options.head.namespace must be a non-empty string")
    timeout = given.get("status_timeout_s", STATUS_TIMEOUT_S)
    if not (
        not isinstance(timeout, bool)
        and isinstance(timeout, int | float)
        and math.isfinite(timeout)
        and 2 * STATUS_PERIOD_S <= timeout <= MAX_STATUS_TIMEOUT_S
    ):
        raise ValueError(
            f"backend_options.head.status_timeout_s must be a finite number of seconds in "
            f"[{2 * STATUS_PERIOD_S:g}, {MAX_STATUS_TIMEOUT_S:g}]"
        )
    isolation = given.get("cpu_isolation")
    cpus = None
    if isolation is not None:
        if not isinstance(isolation, Mapping) or set(isolation) != {"node", "others"}:
            raise ValueError("backend_options.head.cpu_isolation must name exactly node and others")
        try:
            node, others = parse_cpu_list(isolation["node"]), parse_cpu_list(isolation["others"])
        except ValueError as exc:
            raise ValueError(f"backend_options.head.cpu_isolation: {exc}") from exc
        if node & others:
            raise ValueError("backend_options.head.cpu_isolation.node and others must not share CPUs")
        cpus = CpuIsolation(node, others)
    return HeadOptions(namespace, float(timeout), cpus)


def split_config(robot_config: Mapping[str, Any]) -> HeadOptions:
    """The head's options of an X2 configuration that splits its grasp; RuntimeError for any other.

    The head plans the left arm of an OmniHand X2 for the pipeline that
    ``REMOTE_BACKEND`` runs beside an aimdk_robot runtime: a configuration of
    another robot, another runtime or another backend has no grasp head.
    """
    source = robot_config.get("_config_path", "the robot configuration")
    grasp = robot_config.get("grasp_execution") or {}
    if not grasp.get("enabled") or grasp.get("backend") != REMOTE_BACKEND:
        raise RuntimeError(
            f"{source}: the X2 grasp head serves only grasp_execution.backend {REMOTE_BACKEND!r} "
            f"(enabled); this configuration selects {grasp.get('backend')!r}"
        )
    provider = (robot_config.get("runtime") or {}).get("provider")
    if provider != RUNTIME_PROVIDER:
        raise RuntimeError(f"{source}: the X2 grasp head binds an {RUNTIME_PROVIDER} runtime, not {provider!r}")
    if robot_config.get("type") not in ROBOT_TYPES:
        raise RuntimeError(
            f"{source}: the X2 grasp head plans for {sorted(ROBOT_TYPES)}, not robot type {robot_config.get('type')!r}"
        )
    try:
        return head_options(grasp.get("backend_options") or {})
    except ValueError as exc:
        raise RuntimeError(f"{source}: {exc}") from exc


def isolate_launch_cpus(head: HeadOptions) -> None:
    """Keep the calling launch, and so every process it starts, off the head node's cores.

    On the X2's Orin cuRobo shares the CPUs with the vendor stack; confined with
    the rest of its launch elsewhere, a plan-only request took 5.0 s instead of
    7.7-10 s (2026-10-22), as the X2 pipeline did with taskset.
    """
    isolation = head.cpu_isolation
    if isolation is None:
        return
    missing = sorted((isolation.others | isolation.node) - set(range(os.cpu_count() or 0)))
    if missing:
        raise RuntimeError(f"backend_options.head.cpu_isolation names CPUs this host does not have: {missing}")
    # sched_setaffinity covers one thread; the process forks its children from any of them.
    for task in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(task), isolation.others)


def head_node(robot_config: Mapping[str, Any], head: HeadOptions):
    """The grasp head node, bound to the robot's live interface description and pinned to its cores."""
    from launch_ros.actions import Node

    runtime = robot_config.get("runtime") or {}
    description = runtime.get("interface_description", {}) if runtime.get("provider") else {}
    isolation = head.cpu_isolation
    return Node(
        package="aimdk_grasp",
        executable=HEAD_EXECUTABLE,
        name=HEAD_EXECUTABLE,
        output="screen",
        parameters=[
            {
                "grasp_execution_json": json.dumps(robot_config.get("grasp_execution") or {}),
                "interface_description_json": json.dumps(description),
            }
        ],
        prefix=None if isolation is None else "taskset -c " + ",".join(map(str, sorted(isolation.node))),
    )
