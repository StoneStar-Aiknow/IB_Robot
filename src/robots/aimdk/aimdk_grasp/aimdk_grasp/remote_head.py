"""The X2 grasp pipeline beside the runtime, with its head on another host.

``grasp_execution.backend: "aimdk_grasp.remote_head:x2_remote_grasp_backends"``
selects it (``aimdk_grasp.split``). The pipeline, the perception models, the arm
and the hand run here (the 310P or 203, beside the runtime); capture and
registration, the collision scene, cuRobo and the target watch run in the grasp
head (``aimdk_grasp.grasp_head``, the Orin, beside the head camera and the CUDA
GPU). Only compressed captures, target masks, grasp candidates, plans and the
watch's status cross between them; the head keeps each capture for the
requests that follow it. Nothing here needs CUDA.

The watch is judged here, not taken on the head's word: a status counts only
for the session this host armed, in sequence, while it keeps arriving on this
host's clock and its frames keep advancing (RemoteWatch).
"""

from __future__ import annotations

import contextlib
import itertools
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import (
    Flag,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    Observation,
    Proposals,
)
from manipulation_execution.grasp_pipeline.perception import PerceptionServices
from manipulation_execution.grasp_pipeline.streaming import received_at_or_stale

from . import head_wire as wire
from .backend import WARMUP_BUDGET_S, arm_and_hand, watch_timeouts, x2_robot
from .execution_watch import WATCH_FAILED, WatchState, WatchStatus, WatchTimeouts, watch_failure
from .head_interface import NO_SUPPORT, STATUS_DEPTH, head_endpoints, session_id
from .proposer import X2Proposer
from .scene_estimation import NoSupportError
from .split import STATUS_TIMEOUT_S, head_options
from .target_watch import CLOSING_PHASES, WATCHED_PHASES

RPC_FAILED = "RPC_FAILED"
#: The longest a watch command may take; the watch only observes, so a slow one is skipped.
WATCH_CALL_S = 1.0
#: Close cannot start until the head confirms its dispatch gate is closed.
WATCH_CLOSE_BARRIER_S = 0.1
# The head joins its watcher for up to 2 s on DISARM; waiting less loses the report.
WATCH_DISARM_S = 3.0
_HEAD_PLAN_ID = "x2_grasp_head_plan_id"


@dataclass(frozen=True)
class RemoteCapture:
    """``Observation.robot_state`` of a capture the grasp head keeps."""

    capture_id: str
    report: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteScene:
    """``Proposals.scene``: the scene the grasp head completed for a capture."""

    capture_id: str
    report: Mapping = field(default_factory=dict)


class X2GraspHeadClient:
    """Calls into the grasp head; the node must be spun by another thread (the pipeline's executor)."""

    def __init__(self, node, namespace: str, *, callback_group=None):
        from aimdk_grasp_msgs.action import PlanGraspCandidates
        from aimdk_grasp_msgs.msg import WatchStatus as WatchStatusMsg
        from aimdk_grasp_msgs.srv import CaptureGraspObservation, CompleteGraspScene, RefineGraspPlan, WatchGraspTarget
        from rclpy.action import ActionClient
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        from robot_runtime.on_demand import OnDemandSubscriptions

        names = head_endpoints(namespace)
        self.types = (CaptureGraspObservation, CompleteGraspScene, RefineGraspPlan, WatchGraspTarget)
        self.capture_client = node.create_client(CaptureGraspObservation, names.capture, callback_group=callback_group)
        self.scene_client = node.create_client(CompleteGraspScene, names.scene, callback_group=callback_group)
        self.refine_client = node.create_client(RefineGraspPlan, names.refine, callback_group=callback_group)
        self.watch_client = node.create_client(WatchGraspTarget, names.watch, callback_group=callback_group)
        self.plan_client = ActionClient(node, PlanGraspCandidates, names.plan, callback_group=callback_group)
        self.plan_type = PlanGraspCandidates
        self._received: list = []
        self._statuses = OnDemandSubscriptions(node.context, f"{node.get_name()}_head_watch")
        self._statuses.subscribe(
            WatchStatusMsg,
            names.status,
            QoSProfile(
                depth=STATUS_DEPTH, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
            ),
            STATUS_DEPTH,
            lambda message, received: self._received.append((message, received_at_or_stale(received))),
        )

    def statuses(self) -> list:
        """``(WatchStatus, received)`` pairs taken since the last call; ``received`` on this host's monotonic clock.

        A status the middleware did not time is dated ``-inf``, so the watch
        reads it as not heard from: an undated status cannot show the head is
        still reporting.
        """
        self._statuses.drain()
        taken, self._received = self._received, []
        return taken

    @staticmethod
    def wait(future, name: str, deadline: float, cancelled: Flag, on_cancel=None):
        while not future.done():
            if cancelled():
                future.cancel()
                if on_cancel is not None:
                    on_cancel()
                raise GraspFailure("CANCELED", f"cancelled waiting for {name}")
            if time.monotonic() >= deadline:
                future.cancel()
                if on_cancel is not None:
                    on_cancel()
                raise GraspFailure(RPC_FAILED, f"{name} timed out")
            time.sleep(0.002)
        return future.result()

    def call(self, client, request, deadline: float, cancelled: Flag = lambda: False):
        name = client.srv_name
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=max(0.0, min(2.0, deadline - time.monotonic()))
        ):
            raise GraspFailure(RPC_FAILED, f"{name} is not available")
        response = self.wait(client.call_async(request), name, deadline, cancelled)
        if response is None:
            raise GraspFailure(RPC_FAILED, f"{name} returned nothing")
        return response

    def complete_scene(
        self, observation: Observation, mask: np.ndarray, timeout_s: float, *, cancelled: Flag
    ) -> RemoteScene:
        """``X2Proposer``'s remote_scene: NoSupportError when nothing supports the target."""
        capture = _remote_capture(observation)
        request = self.types[1].Request(
            capture_id=capture.capture_id, target_mask=wire.encode_mask(mask), timeout_s=max(0.0, timeout_s)
        )
        response = self.call(
            self.scene_client,
            request,
            time.monotonic() + max(0.0, timeout_s) + 1.0,
            cancelled,
        )
        if not response.success:
            if response.error_code == NO_SUPPORT:
                raise NoSupportError(response.message)
            raise GraspFailure(response.error_code or RPC_FAILED, response.message)
        return RemoteScene(capture.capture_id, wire.from_json(response.report_json))

    def close(self) -> None:
        self._statuses.close()


def _remote_capture(observation: Observation) -> RemoteCapture:
    capture = observation.robot_state
    if not isinstance(capture, RemoteCapture):
        raise GraspFailure("INTERNAL_ERROR", "the X2 grasp head plans only from its own captures")
    return capture


class RemoteHeadCamera:
    """``ObservationSource``: one capture by the grasp head, decoded here."""

    def __init__(self, head: X2GraspHeadClient):
        self.head = head

    def capture(self, *, deadline: float, cancelled: Flag) -> Observation:
        request = self.head.types[0].Request(timeout_s=max(0.0, deadline - time.monotonic()))
        asked = time.monotonic()
        response = self.head.call(self.head.capture_client, request, deadline + 1.0, cancelled)
        answered = time.monotonic()
        if not response.success:
            raise GraspFailure(response.error_code or RPC_FAILED, response.message)
        report = wire.from_json(response.report_json)
        # The head's own breakdown, plus what the call around it cost here: the
        # rest of rpc_s past the head's capture_s is encoding, transfer and the reply.
        report.update(rpc_s=answered - asked, reply_bytes=len(response.rgb.data) + len(response.depth.data))
        try:
            observation = Observation(
                rgb=wire.decode_rgb(response.rgb),
                depth_m=wire.decode_depth(response.depth),
                intrinsics=wire.matrix_from_list(response.intrinsics, (3, 3)),
                frame_id=response.rgb.header.frame_id,
                stamp_ns=int(response.stamp_ns),
                base_frame=response.base_frame,
                base_from_camera=wire.matrix_from_list(response.base_from_camera, (4, 4)),
                robot_state=RemoteCapture(response.capture_id, report),
                report=report,
            )
        except ValueError as exc:
            raise GraspFailure("OBSERVATION_UNAVAILABLE", f"the grasp head's capture does not decode: {exc}") from exc
        report["decode_s"] = time.monotonic() - answered
        return observation


class RemotePlanner:
    """``GraspPlanner``: the grasp head plans against the scene it completed for the capture."""

    def __init__(self, head: X2GraspHeadClient):
        self.head = head

    @staticmethod
    def _with_plan_id(plan: GraspPlan, plan_id: str) -> GraspPlan:
        return GraspPlan(
            plan.steps,
            plan.candidate_index,
            plan.release,
            {**dict(plan.report), _HEAD_PLAN_ID: plan_id},
        )

    @staticmethod
    def _plan_id(plan: GraspPlan) -> str:
        plan_id = plan.report.get(_HEAD_PLAN_ID)
        if not isinstance(plan_id, str) or not plan_id:
            raise GraspFailure("INTERNAL_ERROR", "refine needs a plan the grasp head made")
        return plan_id

    def plan(self, observation: Observation, proposals: Proposals, *, deadline: float, cancelled: Flag) -> GraspPlan:
        capture = _remote_capture(observation)
        goal = self.head.plan_type.Goal(
            capture_id=capture.capture_id,
            grasps=wire.poses_to_list(proposals.grasps),
            confidence=np.asarray(proposals.confidence, dtype=np.float64).tolist(),
            timeout_s=max(0.0, deadline - time.monotonic()),
        )
        client = self.head.plan_client
        if not client.server_is_ready() and not client.wait_for_server(
            timeout_sec=max(0.0, min(2.0, deadline - time.monotonic()))
        ):
            raise GraspFailure(RPC_FAILED, "the grasp head's planner is not available")
        handle = self.head.wait(client.send_goal_async(goal), "plan goal", deadline + 1.0, cancelled)
        if handle is None or not handle.accepted:
            raise GraspFailure(RPC_FAILED, "the grasp head refused the plan goal")
        outcome = self.head.wait(
            handle.get_result_async(), "plan", deadline + 1.0, cancelled, on_cancel=handle.cancel_goal_async
        )
        result = outcome.result
        if not result.success:
            raise GraspFailure(result.error_code or RPC_FAILED, result.message, retryable=bool(result.retryable))
        return self._with_plan_id(wire.plan_from_msg(result.plan), result.plan_id)

    def refine(self, plan: GraspPlan, completed: int, *, deadline: float, cancelled: Flag) -> GraspPlan:
        plan_id = self._plan_id(plan)
        request = self.head.types[2].Request(
            plan_id=plan_id, completed=int(completed), timeout_s=max(0.0, deadline - time.monotonic())
        )
        response = self.head.call(self.head.refine_client, request, deadline + 1.0, cancelled)
        if not response.success:
            raise GraspFailure(response.error_code or RPC_FAILED, response.message)
        refined = wire.plan_from_msg(response.plan)
        # The steps already run stay the caller's objects (GraspPlanner.refine).
        refined = self._with_plan_id(
            GraspPlan(
                plan.steps[: completed + 1] + refined.steps[completed + 1 :],
                refined.candidate_index,
                refined.release,
                refined.report,
            ),
            plan_id,
        )
        return refined


class RemoteWatch:
    """``TargetWatch``: the grasp head compares images; this host decides from its statuses.

    Every command names its session (``head_interface.session_id``): a new one
    per arm, numbered in arming order, so the head refuses a late command of an
    earlier goal. A status counts only when it names the armed session and
    comes after the last one taken. Every status taken is read, not just the
    newest: a move or a failure the head reports is kept for the session even
    if a later status says the watched steps ended.

    The watch has failed when the head reports it unavailable or undecided for
    too long, or, judged on this host's clock while the watched steps run,
    when no status came for ``status_timeout_s`` or the head's fresh frames
    stopped advancing for longer than its frame timeout plus that (a head that
    lives but whose camera or watcher stalled, or never heard the step start).
    Head and host clocks are never compared. When the watched steps end (the
    first step after them starts) this host judges its own timing one last
    time; after that nothing new is judged, but what the head already found and
    reports later (its last frame's votes) still counts.
    """

    def __init__(
        self,
        head: X2GraspHeadClient,
        *,
        mode: str = "stop",
        timeouts: WatchTimeouts = WatchTimeouts(),
        status_timeout_s: float = STATUS_TIMEOUT_S,
        clock=time.monotonic,
    ):
        if mode not in ("stop", "log"):
            raise ValueError("a remote watch is stop or log; off runs none")
        self.head, self.mode, self.timeouts, self.status_timeout_s, self.clock = (
            head,
            mode,
            timeouts,
            status_timeout_s,
            clock,
        )
        # This process's sessions: a restarted pipeline is a new client to the head.
        self._client, self._arms = uuid.uuid4().hex, itertools.count(1)
        self._session, self._error = "", None
        self._reset()

    def _reset(self) -> None:
        self._sequence, self._status, self._received_at = 0, None, None
        self._frames, self._progress_at, self._armed_at = 0, None, None
        self._watching_since, self._ended = None, False
        self._closing_confirmed = False
        self._fault, self._moved, self._statuses = None, None, 0

    def _command(self, command: int, session: str, timeout_s: float = WATCH_CALL_S, **fields_) -> Any:
        request = self.head.types[3].Request(command=command, session_id=session, **fields_)
        return self.head.call(self.head.watch_client, request, time.monotonic() + timeout_s)

    def arm(self, observation: Observation, proposals: Proposals, plan: GraspPlan) -> None:
        """Arm a new session and wait for its first status; in ``stop`` raise WATCH_FAILED when none comes."""
        self._session, self._error = "", None
        self._reset()
        session = session_id(self._client, next(self._arms))
        try:
            capture = _remote_capture(observation)
            response = self._command(self.head.types[3].Request.ARM, session, capture_id=capture.capture_id)
            if not response.success:
                raise GraspFailure(WATCH_FAILED, f"the grasp head's watch did not start: {response.message}")
            self._session, self._armed_at = session, self.clock()
            self._await_status()
        except GraspFailure as failure:
            self._error, self._session = f"not started: {failure}", ""
            self._fault = f"watch unavailable: {self._error}"
            # The head may have armed after all (a reply that came too late); a
            # watcher left running would later halt another grasp. If the ARM has
            # not reached the head yet, this DISARM makes the head refuse it later.
            with contextlib.suppress(GraspFailure):
                self._command(self.head.types[3].Request.DISARM, session, timeout_s=WATCH_DISARM_S)
            if self.mode == "stop":
                raise GraspFailure(WATCH_FAILED, f"target watch {self._error}") from failure

    def _await_status(self) -> None:
        """The ARM reply alone does not show the watch runs: wait for the session's first status."""
        while True:
            self._take()
            if self._status is not None:
                return
            waited = self.clock() - self._armed_at
            if waited > self.status_timeout_s:
                raise GraspFailure(WATCH_FAILED, f"no watch status from the grasp head {waited:.1f} s after arming")
            time.sleep(0.01)

    def _take(self) -> None:
        """Read every status of the session that arrived, in order; keep what any of them found."""
        for message, received in self.head.statuses():
            if not self._session or message.session_id != self._session or message.sequence <= self._sequence:
                continue
            self._sequence, self._status, self._received_at = message.sequence, message, received
            self._statuses += 1
            if message.fresh_frames > self._frames:
                self._frames, self._progress_at = message.fresh_frames, received
            state = WatchState(message.state)
            if state is WatchState.MOVED and self._moved is None:
                self._moved = message.reason or "target moved"
            self._found(
                watch_failure(
                    WatchStatus(state, message.phase, unknown_s=message.unknown_s, reason=message.reason),
                    self.timeouts,
                )
            )

    def _found(self, fault: str | None) -> None:
        if fault is not None and self._fault is None:
            self._fault = fault

    def phase(self, name: str, event: str) -> None:
        if not self._session:
            return
        if event == "start":
            if name in WATCHED_PHASES:
                if self._watching_since is None:
                    self._watching_since = self.clock()
            elif self._watching_since is not None and not self._ended:
                # The last judgement of this host's own timing, up to the end of the watched steps.
                self._take()
                self._found(self._timing())
                self._ended = True
        if event == "start" and name in CLOSING_PHASES and not self._closing_confirmed:
            try:
                response = self._command(
                    self.head.types[3].Request.PHASE,
                    self._session,
                    timeout_s=WATCH_CLOSE_BARRIER_S,
                    phase_name=name,
                    phase_event=event,
                )
                if not response.success:
                    raise GraspFailure(RPC_FAILED, response.message or "the phase command was refused")
            except GraspFailure as failure:
                fault = f"watch unavailable: the grasp head did not close its dispatch gate: {failure}"
                self._found(fault)
            self._closing_confirmed = True
            return
        # Other phases are advisory and do not delay the arm. A phase that does
        # not arrive shows as frames that stop advancing (failed).
        client = self.head.watch_client
        if client.service_is_ready():
            client.call_async(
                self.head.types[3].Request(
                    command=self.head.types[3].Request.PHASE,
                    session_id=self._session,
                    phase_name=name,
                    phase_event=event,
                )
            )

    def moved(self) -> str | None:
        if not self._session:
            return None
        self._take()
        return self._moved if self.mode == "stop" else None

    def failed(self) -> str | None:
        """In ``stop``, why the watch no longer vouches for the target; ``log`` only records it."""
        if not self._session:
            return None
        self._take()
        if not self._ended:
            self._found(self._timing())
        return self._fault if self.mode == "stop" else None

    def _timing(self) -> str | None:
        """What this host's clock says while the watched steps run: statuses and fresh frames still coming."""
        now, status = self.clock(), self._status
        heard = self._received_at if status is not None else self._armed_at
        if now - heard > self.status_timeout_s:
            return f"watch unavailable: no watch status from the grasp head for {now - heard:.1f} s"
        if status is None or self._watching_since is None:
            return None
        stalled = now - max(self._watching_since, self._progress_at or self._watching_since)
        if stalled > self.timeouts.frame_s + self.status_timeout_s:
            state = WatchState(status.state).name.lower()
            return (
                f"watch unavailable: the grasp head took no fresh frame for {stalled:.1f} s "
                f"(its watch: {state}{': ' + status.reason if status.reason else ''})"
            )
        return None

    def disarm(self) -> Mapping:
        session, self._session = self._session, ""
        summary = dict(mode=self.mode, fault=self._fault, statuses=self._statuses, moved=self._moved)
        if self._status is not None:
            summary["last_state"] = WatchState(self._status.state).name.lower()
        if not session:
            return dict(summary, error=self._error)
        try:
            response = self._command(self.head.types[3].Request.DISARM, session, timeout_s=WATCH_DISARM_S)
        except GraspFailure as failure:
            return dict(error=f"disarm: {failure}", host=summary)
        if not response.success:
            return dict(error=f"disarm: {response.message}", host=summary)
        return dict(wire.from_json(response.report_json), host=summary)


def x2_remote_grasp_backends(
    *, node, grasp_execution: Mapping[str, Any], description: Mapping[str, Any]
) -> GraspBackends:
    from rclpy.callback_groups import ReentrantCallbackGroup

    robot = x2_robot(grasp_execution)
    options = robot.options
    split = head_options(options)
    group = ReentrantCallbackGroup()
    closers: list = []
    head = X2GraspHeadClient(node, split.namespace, callback_group=group)
    closers.append(head.close)
    try:
        services = PerceptionServices(
            node,
            detect=options["detect_service"],
            segment=options["segment_service"] or None,
            generate_grasps=options["generate_grasps_service"],
            timeout_s=float(options["service_timeout_s"]),
            callback_group=group,
        )
        camera = RemoteHeadCamera(head)
        proposer = X2Proposer(services, None, settings=robot.perception, remote_scene=head.complete_scene)
        executor, effector = arm_and_hand(node, robot, description, closers)
    except BaseException:
        for close in reversed(closers):
            close()
        raise
    watch = (
        RemoteWatch(
            head, mode=options["watch"], timeouts=watch_timeouts(options), status_timeout_s=split.status_timeout_s
        )
        if options["watch"] != "off"
        else None
    )

    def prepare() -> None:
        """Wait for the head, then warm the models on one of its captures."""
        deadline = time.monotonic() + WARMUP_BUDGET_S
        try:
            observation = camera.capture(deadline=deadline, cancelled=lambda: False)
            proposer.propose(observation, options["warmup_prompt"], deadline=deadline, cancelled=lambda: False)
        except GraspFailure as failure:
            raise RuntimeError(f"warmup on {options['warmup_prompt']!r}: {failure}") from failure

    def close() -> None:
        for closer in reversed(closers):
            closer()

    return GraspBackends(camera, proposer, RemotePlanner(head), executor, effector, watch, close=close, prepare=prepare)
