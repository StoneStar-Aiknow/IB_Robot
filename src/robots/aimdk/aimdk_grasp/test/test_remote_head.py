"""The X2 grasp pipeline's remote head adapters against a fake grasp head, over real topics."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("zstandard")
pytest.importorskip("aimdk_grasp_msgs")

from aimdk_grasp import head_wire as wire  # noqa: E402
from aimdk_grasp.execution_watch import WatchState, WatchTimeouts  # noqa: E402
from aimdk_grasp.head_interface import NO_SUPPORT, NOT_READY, head_endpoints  # noqa: E402
from aimdk_grasp.remote_head import (  # noqa: E402
    RemoteCapture,
    RemoteHeadCamera,
    RemotePlanner,
    RemoteWatch,
    X2GraspHeadClient,
)
from aimdk_grasp.scene_estimation import NoSupportError  # noqa: E402
from aimdk_grasp_msgs.action import PlanGraspCandidates  # noqa: E402
from aimdk_grasp_msgs.msg import WatchStatus  # noqa: E402
from aimdk_grasp_msgs.srv import (  # noqa: E402
    CaptureGraspObservation,
    CompleteGraspScene,
    RefineGraspPlan,
    WatchGraspTarget,
)
from rclpy.action import ActionServer, CancelResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402

from manipulation_execution.grasp_pipeline.contracts import (  # noqa: E402
    ArmSegment,
    Effect,
    EffectorAction,
    GraspFailure,
    GraspPlan,
    Move,
    Proposals,
    SegmentKind,
)

NAMESPACE = "/test_grasp_head"
JOINTS = ("a", "b")


def plan(shift: float = 0.0) -> GraspPlan:
    approach = ArmSegment(
        "approach", SegmentKind.TRAJECTORY, JOINTS, np.array([[0.0, 0.0], [0.1 + shift, 0.2]]), np.array([0.0, 1.0])
    )
    lift = ArmSegment("lift", SegmentKind.TARGET, JOINTS, np.array([[0.1 + shift, 0.2], [0.1 + shift, 0.3]]))
    return GraspPlan(
        (Effect(EffectorAction.PRESHAPE), Move(approach, refine_after=True), Effect(EffectorAction.CLOSE), Move(lift)),
        candidate_index=2,
    )


class FakeHead(Node):
    def __init__(self):
        super().__init__("fake_grasp_head")
        names, group = head_endpoints(NAMESPACE), ReentrantCallbackGroup()
        self.ready, self.plan_seconds, self.requests, self.arm_seconds = True, 0.0, [], 0.0
        self.arm_refusal, self.sessions = "", []
        self.phase_refusal = ""
        self.cancelled = threading.Event()
        self.scene_entered = threading.Event()
        self.scene_release = None
        self.phase_entered = threading.Event()
        self.phase_release = None
        # What the head's watch says, published every 50 ms for the armed session while ``pumping``.
        self.session, self.sequence, self.pumping = "", 0, True
        self.state, self.frames, self.advancing, self.unknown_s, self.reason = WatchState.WAITING, 0, True, 0.0, ""
        self.create_service(CaptureGraspObservation, names.capture, self._capture, callback_group=group)
        self.create_service(CompleteGraspScene, names.scene, self._scene, callback_group=group)
        self.create_service(RefineGraspPlan, names.refine, self._refine, callback_group=group)
        self.create_service(WatchGraspTarget, names.watch, self._watch, callback_group=group)
        ActionServer(
            self,
            PlanGraspCandidates,
            names.plan,
            execute_callback=self._plan,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.statuses = self.create_publisher(
            WatchStatus,
            names.status,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        self.create_timer(0.05, self._pump, callback_group=group)

    def _pump(self):
        if not (self.session and self.pumping):
            return
        if self.advancing and self.state in (WatchState.STABLE, WatchState.UNKNOWN):
            self.frames += 1
        self.publish(self.session)

    def publish(self, session, sequence=None, **fields):
        self.sequence += 1
        status = dict(state=int(self.state), fresh_frames=self.frames, unknown_s=self.unknown_s, reason=self.reason)
        status.update(fields)
        self.statuses.publish(
            WatchStatus(session_id=session, sequence=self.sequence if sequence is None else sequence, **status)
        )

    def _capture(self, request, response):
        self.requests.append(("capture", request.timeout_s))
        if not self.ready:
            response.error_code, response.message = NOT_READY, "warming up"
            return response
        rgb = np.zeros((4, 6, 3), np.uint8)
        rgb[..., 0] = 200
        response.rgb, response.depth = wire.encode_rgb(rgb), wire.encode_depth(np.full((4, 6), 0.75, np.float32))
        response.rgb.header.frame_id = "optical"
        response.success, response.capture_id, response.stamp_ns = True, "c1", 123
        response.intrinsics = np.eye(3).ravel().tolist()
        response.base_frame, response.base_from_camera = "torso_link", np.eye(4).ravel().tolist()
        response.report_json = '{"wait_s": 0.01}'
        return response

    def _scene(self, request, response):
        self.requests.append(("scene", request.capture_id, wire.decode_mask(request.target_mask).sum()))
        if self.scene_release is not None:
            self.scene_entered.set()
            self.scene_release.wait(5.0)
        if request.capture_id == "unsupported":
            response.error_code, response.message = NO_SUPPORT, "nothing under it"
        else:
            response.success, response.report_json = True, '{"scene_s": 0.08}'
        return response

    def _plan(self, goal_handle):
        request = goal_handle.request
        self.requests.append(("plan", request.capture_id, len(request.grasps) // 16, list(request.confidence)))
        ended = time.monotonic() + self.plan_seconds
        while time.monotonic() < ended:
            if goal_handle.is_cancel_requested:
                self.cancelled.set()
                goal_handle.canceled()
                return PlanGraspCandidates.Result(success=False, error_code="CANCELED")
            time.sleep(0.01)
        goal_handle.succeed()
        if request.capture_id == "unplannable":
            return PlanGraspCandidates.Result(success=False, error_code="NO_EXECUTABLE_CANDIDATE", message="none")
        if request.capture_id == "flicker":
            return PlanGraspCandidates.Result(
                success=False, error_code="REOBSERVE_REQUIRED", message="the target flickered", retryable=True
            )
        return PlanGraspCandidates.Result(success=True, plan_id="p1", plan=wire.plan_to_msg(plan()))

    def _refine(self, request, response):
        self.requests.append(("refine", request.plan_id, request.completed))
        response.success, response.plan = True, wire.plan_to_msg(plan(shift=0.05))
        return response

    def _watch(self, request, response):
        self.requests.append(("watch", request.command, request.capture_id, request.phase_name, request.phase_event))
        self.sessions.append(request.session_id)
        response.success = True
        if request.command == WatchGraspTarget.Request.ARM:
            time.sleep(self.arm_seconds)
            if self.arm_refusal:
                response.success, response.message = False, self.arm_refusal
                return response
            self.session, self.sequence, self.state = request.session_id, 0, WatchState.WAITING
        elif request.command == WatchGraspTarget.Request.PHASE and request.phase_event == "start":
            if request.phase_name == "close" and self.phase_release is not None:
                self.phase_entered.set()
                self.phase_release.wait(5.0)
            if request.phase_name == "close" and self.phase_refusal:
                response.success, response.message = False, self.phase_refusal
                return response
            if request.phase_name in ("approach", "grasp"):
                self.state = WatchState.STABLE if self.state is WatchState.WAITING else self.state
            elif self.state is not WatchState.WAITING:  # the watched steps are over (the close started)
                self.state = WatchState.ENDED
        elif request.command == WatchGraspTarget.Request.DISARM:
            self.session = ""
            response.report_json = '{"frames": 12}'
        return response


@pytest.fixture
def ros():
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    head = FakeHead()
    pipeline = Node("remote_head_test")
    client = X2GraspHeadClient(pipeline, NAMESPACE, callback_group=ReentrantCallbackGroup())
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(head)
    executor.add_node(pipeline)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield head, client
    client.close()
    executor.shutdown()
    thread.join(timeout=5.0)
    head.destroy_node()
    pipeline.destroy_node()
    if owned and rclpy.ok():
        rclpy.shutdown()


def capture(client):
    return RemoteHeadCamera(client).capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)


def test_a_capture_decodes_into_an_observation_the_head_keeps(ros):
    head, client = ros
    observation = capture(client)
    assert observation.robot_state == RemoteCapture("c1", {"wait_s": 0.01})
    assert (observation.frame_id, observation.stamp_ns, observation.base_frame) == ("optical", 123, "torso_link")
    assert observation.rgb.shape == (4, 6, 3) and observation.rgb[..., 0].mean() > 190
    np.testing.assert_allclose(observation.depth_m, 0.75)
    head.ready = False
    with pytest.raises(GraspFailure) as raised:
        capture(client)
    assert raised.value.code == NOT_READY


def test_a_scene_without_support_raises_what_the_proposer_falls_back_on(ros):
    head, client = ros
    observation = capture(client)
    mask = np.zeros((4, 6), bool)
    mask[1:3, 2:4] = True
    scene = client.complete_scene(observation, mask, 5.0, cancelled=lambda: False)
    assert (scene.capture_id, scene.report) == ("c1", {"scene_s": 0.08})
    assert head.requests[-1] == ("scene", "c1", 4)
    unsupported = observation.__class__(**{**vars(observation), "robot_state": RemoteCapture("unsupported")})
    with pytest.raises(NoSupportError, match="nothing under it"):
        client.complete_scene(unsupported, mask, 5.0, cancelled=lambda: False)


def test_cancelling_a_scene_call_stops_waiting_for_an_unresponsive_head(ros):
    head, client = ros
    observation = capture(client)
    mask = np.ones((4, 6), bool)
    head.scene_release = threading.Event()
    cancelled = threading.Event()

    def cancel_after_request_arrives():
        assert head.scene_entered.wait(2.0)
        cancelled.set()

    canceller = threading.Thread(target=cancel_after_request_arrives)
    canceller.start()
    began = time.monotonic()
    try:
        with pytest.raises(GraspFailure) as raised:
            client.complete_scene(observation, mask, 120.0, cancelled=cancelled.is_set)
    finally:
        head.scene_release.set()
        canceller.join(2.0)

    assert raised.value.code == "CANCELED" and not canceller.is_alive()
    assert time.monotonic() - began < 1.0


def test_a_plan_and_its_refinement_cross_and_keep_the_steps_already_run(ros):
    head, client = ros
    observation = capture(client)
    grasps = np.tile(np.eye(4), (3, 1, 1))
    proposals = Proposals(np.ones((4, 6), bool), np.zeros((5, 3)), grasps, np.array([0.9, 0.8, 0.7]))
    planner = RemotePlanner(client)
    planned = planner.plan(observation, proposals, deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert head.requests[-1] == ("plan", "c1", 3, [0.9, 0.8, 0.7])
    assert planned.candidate_index == 2 and [step.name for step in planned.steps] == [
        "preshape",
        "approach",
        "close",
        "lift",
    ]
    assert planned.report["x2_grasp_head_plan_id"] == "p1"
    copied = GraspPlan(planned.steps, planned.candidate_index, planned.release, dict(planned.report))
    refined = planner.refine(copied, 1, deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert head.requests[-1] == ("refine", "p1", 1)
    assert refined.steps[:2] == planned.steps[:2]  # the same step objects
    assert refined.steps[3].segment.end[0] == pytest.approx(0.15)
    planner.refine(refined, 2, deadline=time.monotonic() + 5.0, cancelled=lambda: False)  # still the head's plan p1
    assert head.requests[-1] == ("refine", "p1", 2)
    with pytest.raises(GraspFailure, match="plan the grasp head made") as raised:
        planner.refine(plan(), 1, deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert raised.value.code == "INTERNAL_ERROR"
    unplannable = observation.__class__(**{**vars(observation), "robot_state": RemoteCapture("unplannable")})
    with pytest.raises(GraspFailure) as raised:
        planner.plan(unplannable, proposals, deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert raised.value.code == "NO_EXECUTABLE_CANDIDATE" and not raised.value.retryable
    flicker = observation.__class__(**{**vars(observation), "robot_state": RemoteCapture("flicker")})
    with pytest.raises(GraspFailure) as raised:  # the pipeline observes again, as on a single host
        planner.plan(flicker, proposals, deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert raised.value.code == "REOBSERVE_REQUIRED" and raised.value.retryable


def test_cancelling_a_plan_cancels_it_on_the_head(ros):
    head, client = ros
    observation = capture(client)
    head.plan_seconds = 5.0
    proposals = Proposals(np.ones((4, 6), bool), np.zeros((5, 3)), np.tile(np.eye(4), (1, 1, 1)), np.array([0.9]))
    asked = time.monotonic() + 0.3
    with pytest.raises(GraspFailure) as raised:
        RemotePlanner(client).plan(
            observation, proposals, deadline=time.monotonic() + 10.0, cancelled=lambda: time.monotonic() > asked
        )
    assert raised.value.code == "CANCELED" and head.cancelled.wait(3.0)


QUICK = WatchTimeouts(frame_s=0.3, unknown_s=0.5)


def remote_watch(client, mode="stop"):
    return RemoteWatch(client, mode=mode, timeouts=QUICK, status_timeout_s=0.3)


def wait(predicate, timeout_s=3.0):
    ended = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < ended:
        time.sleep(0.01)
    return predicate()


def watched(head, client, mode="stop"):
    """A watch armed on the head's capture and into its approach."""
    watch = remote_watch(client, mode)
    watch.arm(capture(client), None, None)
    watch.phase("preshape", "start")
    watch.phase("approach", "start")
    assert wait(lambda: head.state is WatchState.STABLE)
    return watch


def reads(watch, state):
    """Whether the watch has taken a status in ``state`` (statuses are taken only when asked)."""
    watch._take()
    return watch._status is not None and watch._status.state == state


def commands(head):
    return [entry[1:] for entry in head.requests if entry[0] == "watch"]


def test_a_watch_is_armed_on_a_session_of_its_own_and_reports_a_move_of_that_session_only(ros):
    head, client = ros
    watch = remote_watch(client)
    assert watch.moved() is None and watch.failed() is None
    watch.arm(capture(client), None, None)
    session = head.session
    assert session.endswith(":1") and watch.failed() is None  # arm returned on the session's first status
    watch.phase("approach", "start")
    time.sleep(0.4)  # statuses keep coming and frames advance: still watching
    assert watch.failed() is None and watch.moved() is None
    head.publish("an earlier session", state=int(WatchState.MOVED), reason="old news")
    head.publish(session, sequence=1, state=int(WatchState.MOVED), reason="a replay")
    time.sleep(0.3)
    assert watch.moved() is None and watch.failed() is None
    head.state, head.reason = WatchState.MOVED, "target moved (votes [0.4, 0.9])"
    assert wait(lambda: watch.moved() is not None)
    assert watch.moved() == "target moved (votes [0.4, 0.9])" and watch.failed() is None
    report = watch.disarm()
    assert report["frames"] == 12 and report["host"]["moved"] == watch._moved and report["host"]["fault"] is None
    assert watch.moved() is None and watch.failed() is None  # disarmed
    assert [(command, capture_id) for command, capture_id, *_ in commands(head)] == [
        (WatchGraspTarget.Request.ARM, "c1"),
        (WatchGraspTarget.Request.PHASE, ""),
        (WatchGraspTarget.Request.DISARM, ""),
    ]
    # Every command names the session; the next arm is a newer one of the same client.
    assert head.sessions == [session] * 3
    watch.arm(capture(client), None, None)
    client_id, number = head.session.rsplit(":", 1)
    assert client_id == session.rsplit(":", 1)[0] and number == "2"
    watch.disarm()


def test_a_watch_the_head_does_not_start_refuses_the_execution(ros):
    head, client = ros
    head.arm_refusal = "not started: TypeError: the X2 watch compares X2 captures"
    with pytest.raises(GraspFailure) as raised:
        remote_watch(client).arm(capture(client), None, None)
    assert raised.value.code == "WATCH_FAILED" and "did not start: not started: TypeError" in str(raised.value)
    assert [command for command, *_ in commands(head)] == [
        WatchGraspTarget.Request.ARM,
        WatchGraspTarget.Request.DISARM,
    ]


def test_an_arm_the_head_acknowledges_without_ever_sending_a_status_is_no_watch(ros):
    head, client = ros
    head.pumping = False
    began = time.monotonic()
    with pytest.raises(GraspFailure, match="no watch status from the grasp head") as raised:
        remote_watch(client).arm(capture(client), None, None)
    assert raised.value.code == "WATCH_FAILED" and time.monotonic() - began < 2.0
    assert commands(head)[-1][0] == WatchGraspTarget.Request.DISARM


def test_an_arm_that_answers_too_late_is_disarmed_so_no_watcher_is_left_running(ros):
    head, client = ros
    head.arm_seconds = 1.5  # past the client's wait: the head armed, the client never heard
    with pytest.raises(GraspFailure, match="timed out"):
        remote_watch(client).arm(capture(client), None, None)
    assert [command for command, *_ in commands(head)] == [
        WatchGraspTarget.Request.ARM,
        WatchGraspTarget.Request.DISARM,
    ]
    watch = remote_watch(client, mode="log")  # log goes on without the watch, and says so
    watch.arm(capture(client), None, None)
    watch.phase("approach", "start")
    assert watch.failed() is None and watch.disarm()["error"].startswith("not started")


def test_statuses_that_stop_mid_approach_fail_the_execution(ros):
    head, client = ros
    watch = watched(head, client)
    assert watch.failed() is None
    head.pumping = False  # the head exited or the network went down
    assert wait(lambda: watch.failed() is not None)
    assert watch.failed().startswith("watch unavailable: no watch status from the grasp head for")
    assert watch.moved() is None
    assert watch.disarm()["host"]["fault"] == watch._fault and "no watch status" in watch._fault


def test_a_status_the_middleware_did_not_time_is_not_a_status_heard_from(ros):
    """CycloneDDS records no receive time; an undated status cannot show the head still reports.

    The status may have waited in the reader queue for seconds, so dating it at
    the drain would hide a head that stopped reporting behind its own backlog.
    """
    head, client = ros
    watch = watched(head, client)
    assert watch.failed() is None
    head.pumping = False  # only the undated status is left to vouch for the head
    _reader, on_sample, _depth = watch.head._statuses._entries[0]
    on_sample(
        WatchStatus(session_id=head.session, sequence=10**6, state=int(WatchState.STABLE), fresh_frames=10**6),
        None,
    )
    assert watch.failed().startswith("watch unavailable: no watch status from the grasp head for")


def test_a_head_that_keeps_reporting_but_takes_no_new_frame_fails_the_execution(ros):
    head, client = ros
    watch = watched(head, client)
    head.advancing = False  # alive, but its camera or watcher stalled behind a "stable" it no longer refreshes
    assert wait(lambda: watch.failed() is not None)
    assert watch.failed().startswith("watch unavailable: the grasp head took no fresh frame for")
    assert "(its watch: stable)" in watch.failed()


def test_a_head_that_never_heard_the_approach_start_fails_the_execution(ros):
    head, client = ros
    watch = remote_watch(client)
    watch.arm(capture(client), None, None)
    head.pumping, head.state = True, WatchState.WAITING
    watch._watching_since = time.monotonic()  # this host is in the approach; the PHASE never arrived
    assert wait(lambda: watch.failed() is not None)
    assert "took no fresh frame" in watch.failed() and "(its watch: waiting)" in watch.failed()


@pytest.mark.parametrize(
    ("state", "unknown_s", "expected"),
    [
        (WatchState.UNAVAILABLE, 0.0, "watch unavailable: no fresh image for 1.2 s"),
        (WatchState.UNKNOWN, 0.6, "target not observable for 0.6 s"),
    ],
)
def test_what_the_head_reports_as_unavailable_or_long_undecided_fails_the_execution(ros, state, unknown_s, expected):
    head, client = ros
    watch = watched(head, client)
    head.state, head.unknown_s, head.reason = WatchState.UNKNOWN, 0.2, "target 10% visible"
    assert wait(lambda: reads(watch, WatchState.UNKNOWN))
    assert watch.failed() is None  # briefly undecided: go on
    head.state, head.unknown_s, head.reason = state, unknown_s, "no fresh image for 1.2 s"
    assert wait(lambda: watch.failed() is not None)
    assert watch.failed().startswith(expected)


def test_nothing_is_judged_before_the_approach_or_once_the_close_started(ros):
    head, client = ros
    watch = remote_watch(client)
    watch.arm(capture(client), None, None)
    watch.phase("preshape", "start")
    time.sleep(0.5)  # WAITING with no frames is what the head says before the approach
    assert watch.failed() is None
    watch.phase("approach", "start")
    watch.phase("approach", "end")
    watch.phase("close", "start")  # the hand is meant to touch the target now
    head.pumping = False
    time.sleep(0.5)
    assert watch.failed() is None and watch.moved() is None
    watch.disarm()


def test_close_waits_until_the_head_confirms_its_dispatch_gate_is_closed(ros):
    head, client = ros
    watch = watched(head, client)
    head.phase_release = threading.Event()
    returned = threading.Event()
    errors = []

    def close():
        try:
            watch.phase("close", "start")
        except Exception as exc:  # noqa: BLE001 - surfaced in the test thread
            errors.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=close)
    thread.start()
    assert head.phase_entered.wait(1.0)
    assert not returned.wait(0.01)  # the hand's close caller is still behind the phase barrier
    head.phase_release.set()
    thread.join(1.0)
    assert returned.is_set() and not thread.is_alive() and not errors


@pytest.mark.parametrize(("mode", "stops"), [("stop", True), ("log", False)])
def test_a_refused_close_barrier_is_reported_through_failed_without_phase_raising(ros, mode, stops):
    head, client = ros
    watch = watched(head, client, mode=mode)
    head.phase_refusal = "watch command rejected"

    watch.phase("close", "start")

    fault = "watch unavailable: the grasp head did not close its dispatch gate"
    assert (watch.failed() is not None) is stops
    assert watch._fault.startswith(fault) and "watch command rejected" in watch._fault
    assert watch.disarm()["host"]["fault"].startswith(fault)


def test_in_log_mode_failures_and_moves_are_recorded_but_never_stop_the_execution(ros):
    head, client = ros
    watch = watched(head, client, mode="log")
    head.state, head.reason = WatchState.MOVED, "target moved"
    assert wait(lambda: reads(watch, WatchState.MOVED))
    assert watch.moved() is None
    head.pumping = False
    time.sleep(0.5)
    assert watch.failed() is None
    host = watch.disarm()["host"]
    assert host["moved"] == "target moved" and host["fault"].startswith("watch unavailable: no watch status")
    with pytest.raises(ValueError, match="off runs none"):
        RemoteWatch(client, mode="off")


def queued(head, state):
    """Wait until the head has published a status in ``state`` (it then sits unread in the watch's queue)."""
    published = []
    publish = head.statuses.publish

    def spy(message):
        published.append(message.state)
        publish(message)

    head.statuses.publish = spy
    assert wait(lambda: int(state) in published)


def test_a_move_reported_as_the_grasp_ends_still_counts_after_the_next_step_starts(ros):
    """grasp -> settle_grasp -> close: a move of the last frames must stop the close."""
    head, client = ros
    watch = watched(head, client)
    watch.phase("grasp", "start")
    head.state, head.reason = WatchState.MOVED, "target moved (votes [0.4, 0.9])"
    queued(head, WatchState.MOVED)
    watch.phase("grasp", "end")
    watch.phase("settle_grasp", "start")  # the head then reports ENDED, after the MOVED still queued
    assert wait(lambda: head.state is WatchState.ENDED)
    time.sleep(0.15)  # an ENDED status follows the MOVED in the same drain
    assert watch.moved() == "target moved (votes [0.4, 0.9])"
    watch.phase("close", "start")
    assert watch.moved() == "target moved (votes [0.4, 0.9])" and watch.failed() is None


def test_a_move_or_failure_the_head_reports_only_after_the_step_ended_still_counts(ros):
    head, client = ros
    watch = watched(head, client)
    watch.phase("settle_grasp", "start")
    assert wait(lambda: head.state is WatchState.ENDED)
    assert watch.moved() is None and watch.failed() is None
    head.pumping = False
    head.publish(head.session, state=int(WatchState.MOVED), reason="the last frame's votes")
    assert wait(lambda: watch.moved() == "the last frame's votes")
    watch.disarm()
    head.pumping = True
    other = watched(head, client)
    other.phase("settle_grasp", "start")
    head.pumping = False
    # The head's watch ended undecided for too long: its last word is that failure.
    head.publish(head.session, state=int(WatchState.UNKNOWN), unknown_s=0.9, reason="hand covers the target")
    assert wait(lambda: other.failed() is not None)
    assert other.failed().startswith("target not observable for 0.9 s") and other.moved() is None


def test_a_status_gap_during_the_watched_steps_is_judged_when_they_end_even_if_nobody_asked(ros):
    head, client = ros
    watch = watched(head, client)
    head.pumping = False
    time.sleep(0.5)  # past the 0.3 s status timeout while the arm moved (between polls)
    watch.phase("settle_grasp", "start")
    assert watch.failed().startswith("watch unavailable: no watch status from the grasp head for")
