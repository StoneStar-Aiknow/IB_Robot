"""The X2 grasp head's requests, over fake head parts: what is kept, refused and reported."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("zstandard")
pytest.importorskip("aimdk_grasp_msgs")

from aimdk_grasp import grasp_head  # noqa: E402
from aimdk_grasp import head_wire as wire  # noqa: E402
from aimdk_grasp.backend import HeadParts  # noqa: E402
from aimdk_grasp.execution_watch import WatchState, WatchStatus  # noqa: E402
from aimdk_grasp.head_interface import NO_SUPPORT, NOT_READY, UNKNOWN  # noqa: E402
from aimdk_grasp.proposer import PerceptionSettings  # noqa: E402
from aimdk_grasp.scene_estimation import NoSupportError  # noqa: E402
from aimdk_grasp_msgs.action import PlanGraspCandidates  # noqa: E402
from aimdk_grasp_msgs.srv import (  # noqa: E402
    CaptureGraspObservation,
    CompleteGraspScene,
    RefineGraspPlan,
    WatchGraspTarget,
)

from manipulation_execution.pick.pipeline.contracts import (  # noqa: E402
    ArmSegment,
    Effect,
    EffectorAction,
    GraspFailure,
    GraspPlan,
    Move,
    Observation,
    SegmentKind,
)

H, W = 4, 6


def a_plan() -> GraspPlan:
    line = ArmSegment("approach", SegmentKind.TARGET, ("a",), np.array([[0.0], [0.1]]))
    return GraspPlan((Move(line), Effect(EffectorAction.CLOSE)), candidate_index=0)


class Camera:
    def __init__(self):
        self.calls, self.stamp, self.fail = [], 100, None

    def capture(self, *, deadline, cancelled):
        if self.fail is not None:
            raise self.fail
        self.stamp += 1
        self.calls.append({"stamp_ns": self.stamp, "wait_s": 0.01})
        self.calls.append({"stamp_ns": self.stamp + 1000, "wait_s": 9.0})  # a watch frame meanwhile
        frame = SimpleNamespace(
            depth=np.full((H, W), 0.8, np.float32), intrinsics="k", depth_scale=1.0, frame_id="optical"
        )
        registered = SimpleNamespace(
            frame=frame,
            transform=SimpleNamespace(stamp_ns=self.stamp),
            received_at=1.0,
            image_feedback="image feedback",
        )
        state = SimpleNamespace(live=SimpleNamespace(observation=registered, feedback="current feedback"))
        return Observation(
            np.zeros((H, W, 3), np.uint8),
            frame.depth,
            np.eye(3),
            "optical",
            self.stamp,
            "torso_link",
            np.eye(4),
            state,
            report={"stamp_ns": self.stamp, "wait_s": 0.01},
        )


class Imu:
    def __init__(self):
        self.events = []

    def pause_after(self, stamp):
        self.events.append(("pause_after", stamp))

    def pause(self):
        self.events.append(("pause",))

    def resume(self):
        self.events.append(("resume",))


class Planner:
    def __init__(self):
        self.asked, self.fail = [], None

    def plan(self, observation, proposals, *, deadline, cancelled):
        self.asked.append((observation, proposals))
        if self.fail is not None:
            raise self.fail
        return a_plan()

    def refine(self, plan, completed, *, deadline, cancelled):
        self.asked.append(("refine", plan, completed))
        return plan


class Watch:
    def __init__(self):
        self.armed, self.refuse, self.started, self.disarmed = [], None, True, 0
        self.arm_seconds = 0.0
        self.state = WatchStatus(WatchState.WAITING)

    def arm(self, observation, proposals, plan):
        time.sleep(self.arm_seconds)
        self.armed.append((observation, proposals.target_mask.sum()))
        if self.refuse is not None:
            raise self.refuse

    def phase(self, name, event):
        self.armed.append((name, event))

    def status(self):
        return self.state

    def disarm(self):
        self.disarmed += 1
        return {"frames": 3}


@pytest.fixture
def head(monkeypatch):
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    parts = HeadParts(Camera(), "spheres", SimpleNamespace(prepare=lambda *a: None, preload=lambda *a: None), Planner())
    parts.imu, parts.watch = Imu(), Watch()
    robot = SimpleNamespace(
        options={"head": {"namespace": "/test_head"}, "watch": "stop"}, perception=PerceptionSettings(), hand="hand"
    )
    monkeypatch.setattr(grasp_head, "x2_robot", lambda config: robot)
    monkeypatch.setattr(grasp_head, "head_parts", lambda node, robot, description, closers, on_moved: parts)
    monkeypatch.setattr(grasp_head, "check_initial_hand", lambda hand, admitted: None)
    monkeypatch.setattr(grasp_head, "measured_spheres", lambda model, feedback: "measured spheres")
    monkeypatch.setattr(grasp_head, "prepare_scene", lambda depth, *a, **k: ("prepared", k["robot_spheres_torso"]))
    unsupported = threading.Event()

    def complete(prepared, mask, *, lazy_validation):
        if unsupported.is_set():
            raise NoSupportError("nothing under it")
        return ("scene", None if mask is None else int(mask.sum()))

    monkeypatch.setattr(grasp_head, "complete_scene", complete)
    monkeypatch.setattr(grasp_head, "mask_depth_to_points", lambda mask, *a, **k: np.zeros((int(mask.sum()), 3)))
    node = grasp_head.X2GraspHeadNode()
    published = []
    monkeypatch.setattr(node._statuses, "publish", published.append)
    yield SimpleNamespace(node=node, parts=parts, unsupported=unsupported, published=published)
    node.close()
    node.destroy_node()
    if owned and rclpy.ok():
        rclpy.shutdown()


class HeldElsewhere:
    """Hold ``lock`` on another thread (a planner), as long as the ``with`` block runs."""

    def __init__(self, lock):
        self.lock, self.taken, self.done = lock, threading.Event(), threading.Event()

    def _hold(self):
        with self.lock:
            self.taken.set()
            self.done.wait(10.0)

    def __enter__(self):
        self.thread = threading.Thread(target=self._hold, daemon=True)
        self.thread.start()
        assert self.taken.wait(5.0)
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join(5.0)


def call(handler, kind, **fields):
    return handler(kind.Request(**fields), kind.Response())


def mask(rows=slice(1, 3)):
    value = np.zeros((H, W), bool)
    value[rows, 2:4] = True
    return wire.encode_mask(value)


def ready(node):
    assert node._ready.wait(5.0)


def test_captures_wait_for_the_warmup_and_start_their_scene(head):
    node, parts = head.node, head.parts
    ready(node)
    response = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0)
    assert response.success and response.capture_id and response.rgb.header.frame_id == "optical"
    np.testing.assert_allclose(wire.decode_depth(response.depth), 0.8)
    assert parts.imu.events[-1] == ("pause_after", response.stamp_ns)
    assert node._capture(response.capture_id).prepared.result(5.0) == ("prepared", "measured spheres")
    node._ready.clear()
    refused = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0)
    assert (refused.success, refused.error_code) == (False, NOT_READY)


def test_a_scene_completes_with_its_mask_or_says_why_not(head):
    node = head.node
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id
    unknown = call(node._on_scene, CompleteGraspScene, capture_id="nope", target_mask=mask(), timeout_s=1.0)
    assert unknown.error_code == UNKNOWN
    wrong = wire.encode_mask(np.ones((H + 1, W), bool))
    assert call(
        node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=wrong, timeout_s=1.0
    ).error_code == ("INTERNAL_ERROR")
    head.unsupported.set()
    response = call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    assert (response.success, response.error_code) == (False, NO_SUPPORT)
    head.unsupported.clear()
    response = call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    assert response.success and "scene_s" in wire.from_json(response.report_json)
    assert node._capture(capture_id).estimate == ("scene", 4)


def test_the_scene_stages_wait_while_the_gpu_is_held(head):
    """A planner capturing CUDA graphs holds the head's GPU; scene work on the GPU then must not run."""
    node, gpu = head.node, head.parts.gpu
    ready(node)
    with gpu:  # this thread stands in for the planner
        response = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0)
        prepared = node._capture(response.capture_id).prepared
        assert response.success and not prepared.done()
        scene = threading.Thread(
            target=call,
            args=(node._on_scene, CompleteGraspScene),
            kwargs=dict(capture_id=response.capture_id, target_mask=mask(), timeout_s=5.0),
            daemon=True,
        )
        scene.start()
        scene.join(0.3)
        assert scene.is_alive() and not prepared.done()
    scene.join(5.0)
    assert not scene.is_alive() and node._capture(response.capture_id).estimate == ("scene", 4)


def test_a_scene_behind_a_long_plan_gives_up_at_its_deadline(head):
    node, gpu = head.node, head.parts.gpu
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id
    node._capture(capture_id).prepared.result(5.0)
    with HeldElsewhere(gpu):  # a plan on another thread
        began = time.monotonic()
        response = call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=0.3)
    assert (response.success, response.error_code) == (False, "SKILL_TIMEOUT") and time.monotonic() - began < 1.0


def test_a_capture_reports_its_own_timings_and_an_error_is_answered_not_raised(head):
    node, parts = head.node, head.parts
    ready(node)
    response = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0)
    assert wire.from_json(response.report_json) == {"stamp_ns": response.stamp_ns, "wait_s": 0.01}
    parts.camera.fail = RuntimeError("Warp CUDA error 900")
    capture = node._answering(node._on_capture)
    response = capture(CaptureGraspObservation.Request(timeout_s=1.0), CaptureGraspObservation.Response())
    assert (response.success, response.error_code) == (False, "INTERNAL_ERROR") and "Warp" in response.message


def test_a_scene_that_cannot_be_built_has_no_plan_as_on_a_single_host(head, monkeypatch):
    node = head.node
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id

    def thin(prepared, mask, *, lazy_validation):
        raise ValueError("Target mask has insufficient valid depth after self-filtering")

    monkeypatch.setattr(grasp_head, "complete_scene", thin)
    response = call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    assert (response.success, response.error_code) == (False, "NO_EXECUTABLE_CANDIDATE")


def _plan(node, capture_id, grasps=2):
    goal = PlanGraspCandidates.Goal(
        capture_id=capture_id,
        grasps=wire.poses_to_list(np.tile(np.eye(4), (grasps, 1, 1))),
        confidence=[0.5] * grasps,
        timeout_s=5.0,
    )
    ended = []
    handle = SimpleNamespace(
        request=goal, is_cancel_requested=False, succeed=lambda: ended.append("succeeded"), canceled=lambda: None
    )
    return node._on_plan(handle), ended


def test_a_plan_uses_the_completed_scene_and_wakes_the_imu_only_with_a_plan(head):
    node, parts = head.node, head.parts
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id
    result, _ = _plan(node, capture_id)
    assert (result.success, result.error_code) == (False, UNKNOWN)  # no scene yet
    call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    result, ended = _plan(node, capture_id, grasps=3)
    observation, proposals = parts.planner.asked[-1]
    assert result.success and ended == ["succeeded"] and parts.imu.events[-1] == ("resume",)
    assert proposals.scene == ("scene", 4) and proposals.grasps.shape == (3, 4, 4) and len(proposals.target_points) == 4
    assert wire.plan_from_msg(result.plan).candidate_index == 0
    refined = call(node._on_refine, RefineGraspPlan, plan_id=result.plan_id, completed=0, timeout_s=1.0)
    assert refined.success and parts.planner.asked[-1][0] == "refine"
    parts.planner.fail = GraspFailure("NO_EXECUTABLE_CANDIDATE", "none")
    result, _ = _plan(node, capture_id)
    assert (result.success, result.error_code, result.retryable) == (False, "NO_EXECUTABLE_CANDIDATE", False)
    parts.planner.fail = GraspFailure("REOBSERVE_REQUIRED", "the target flickered", retryable=True)
    result, _ = _plan(node, capture_id)
    assert (result.error_code, result.retryable) == ("REOBSERVE_REQUIRED", True)
    parts.planner.fail = KeyError("left_elbow_joint")
    result, _ = _plan(node, capture_id)
    assert (result.success, result.error_code) == (False, "INTERNAL_ERROR")
    assert parts.imu.events[-1] == ("pause",)


ARM, PHASE, DISARM = (WatchGraspTarget.Request.ARM, WatchGraspTarget.Request.PHASE, WatchGraspTarget.Request.DISARM)


def watch(node, command, session, **fields):
    return call(node._on_watch, WatchGraspTarget, command=command, session_id=session, **fields)


def scene_ready(node):
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id
    call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    return capture_id


def test_the_watch_is_armed_on_a_completed_scene_and_publishes_its_state_for_that_session_only(head):
    node, parts = head.node, head.parts
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id
    assert not watch(node, ARM, "p:1", capture_id=capture_id).success  # no scene yet
    call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    assert not watch(node, ARM, "", capture_id=capture_id).success
    head.published.clear()
    node._publish_status()
    assert head.published == []  # nothing is armed
    assert watch(node, ARM, "p:2", capture_id=capture_id).success
    assert [(status.session_id, status.sequence, status.state) for status in head.published] == [("p:2", 1, 0)]
    parts.watch.state = WatchStatus(WatchState.STABLE, "approach", 4, 1234, 0.0, "")
    assert watch(node, PHASE, "p:2", phase_name="approach", phase_event="start").success
    assert parts.watch.armed[-2][1] == 4 and parts.watch.armed[-1] == ("approach", "start")
    parts.watch.state = WatchStatus(WatchState.MOVED, "grasp", 5, 1300, 0.0, "target moved (votes [0.4, 0.9])")
    node._publish_status()  # the timer, and the watch at once when the target moved
    last = head.published[-1]
    assert (last.session_id, last.sequence, last.state, last.phase) == ("p:2", 3, int(WatchState.MOVED), "grasp")
    assert (last.fresh_frames, last.last_image_stamp_ns, last.reason) == (5, 1300, "target moved (votes [0.4, 0.9])")
    disarmed = watch(node, DISARM, "p:2")
    assert disarmed.success and wire.from_json(disarmed.report_json) == {"frames": 3}
    count = len(head.published)
    node._publish_status()
    assert len(head.published) == count  # disarmed: the session is over


def test_a_late_command_of_an_earlier_session_never_touches_the_armed_one(head):
    node, parts = head.node, head.parts
    capture_id = scene_ready(node)
    assert watch(node, ARM, "p:7", capture_id=capture_id).success
    # The previous goal's DISARM and PHASE, delivered after this goal armed.
    late = watch(node, DISARM, "p:6")
    assert not late.success and "'p:6' is not the armed one (p:7)" in late.message
    assert not watch(node, PHASE, "p:6", phase_name="close", phase_event="start").success
    assert parts.watch.disarmed == 0 and ("close", "start") not in parts.watch.armed
    assert node._session == "p:7"
    # Its ARM, delivered late (the client gave up waiting): older than the armed session.
    stale = watch(node, ARM, "p:6", capture_id=capture_id)
    assert not stale.success and "not newer than p:7" in stale.message and node._session == "p:7"
    assert not watch(node, ARM, "p:7", capture_id=capture_id).success  # nor a repeat of the armed one
    assert not watch(node, DISARM, "q:7").success  # another pipeline process's session
    assert not watch(node, DISARM, "p-7").success  # not a session id
    assert parts.watch.disarmed == 0
    assert watch(node, DISARM, "p:7").success and parts.watch.disarmed == 1


def test_an_arm_whose_disarm_arrived_first_is_refused(head):
    """The client gave up on an ARM and disarmed it; the ARM then arrives after its own DISARM."""
    node, parts = head.node, head.parts
    capture_id = scene_ready(node)
    assert not watch(node, DISARM, "p:3").success  # nothing armed yet; the head remembers the session
    late = watch(node, ARM, "p:3", capture_id=capture_id)
    assert not late.success and node._session == "" and len(parts.watch.armed) == 0
    assert watch(node, ARM, "p:4", capture_id=capture_id).success  # the client's next session arms


def test_a_disarm_that_overtakes_its_slow_arm_still_ends_that_session(head):
    """Commands are applied one at a time, in the order the head serves them."""
    node, parts = head.node, head.parts
    capture_id = scene_ready(node)
    parts.watch.arm_seconds = 0.3
    arming = threading.Thread(target=watch, args=(node, ARM, "p:1"), kwargs=dict(capture_id=capture_id))
    arming.start()
    time.sleep(0.1)  # the ARM is inside watch.arm
    disarmed = watch(node, DISARM, "p:1")
    arming.join(5.0)
    assert disarmed.success and parts.watch.disarmed == 1 and node._session == ""


def test_a_watch_that_does_not_start_is_refused_and_publishes_nothing(head):
    node, parts = head.node, head.parts
    ready(node)
    capture_id = call(node._on_capture, CaptureGraspObservation, timeout_s=1.0).capture_id
    call(node._on_scene, CompleteGraspScene, capture_id=capture_id, target_mask=mask(), timeout_s=1.0)
    parts.watch.refuse = GraspFailure("WATCH_FAILED", "target watch not started: TypeError: no X2 capture")
    response = watch(node, ARM, "p:1", capture_id=capture_id)
    assert not response.success and "not started" in response.message
    head.published.clear()
    node._publish_status()
    assert head.published == []
