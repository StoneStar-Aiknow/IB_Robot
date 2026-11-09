"""The X2 target watch adapter and the chest IMU it takes body rotation from."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.execution_watch import (
    ChestImu,
    WatchState,
    WatchStatus,
    WatchTimeouts,
    X2TargetWatch,
    moved_reason,
    watch_failure,
)
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.joints import BODY_JOINTS
from aimdk_grasp.observation import AdmittedState, LiveState, RegisteredObservation
from aimdk_grasp.target_watch import WatchConfig
from aimdk_grasp.types import RGBDFrame
from test_target_watch import K, far_spheres, scene, watcher_inputs

from manipulation_execution.pick.pipeline.contracts import GraspFailure, Observation, Proposals

STILL, MOVED = scene(), scene(target_at=(30, 55))


class LevelImu:
    """A chest IMU that always reads level and counts the pauses it is asked for."""

    def __init__(self):
        self.orientation = SimpleNamespace(at=lambda stamp_ns: np.array([0.0, 0.0, 0.0, 1.0]))
        self.pauses = 0

    def pause(self):
        self.pauses += 1


@pytest.fixture(autouse=True)
def no_watcher_left_running(monkeypatch):
    """Disarm every watch a test armed: a watcher left polling at period 0 starves the next tests."""
    armed_watches = []
    arm = X2TargetWatch.arm

    def tracked(self, *args, **kwargs):
        armed_watches.append(self)
        return arm(self, *args, **kwargs)

    monkeypatch.setattr(X2TargetWatch, "arm", tracked)
    yield
    for watch in armed_watches:
        if watch._watcher is not None:
            watch._watcher.stop()


@pytest.fixture(autouse=True)
def one_moving_sphere(monkeypatch):
    monkeypatch.setattr("aimdk_grasp.target_watch.moving_sphere_indices", lambda: np.array([0]))


def planned(image=STILL):
    rgb, depth, mask = image
    feedback = Feedback(1e-8, dict.fromkeys(BODY_JOINTS, 0.0), np.zeros(10))
    registered = RegisteredObservation(
        RGBDFrame(rgb, depth, K, "optical"), ImageTransform(10, "optical", "torso_link", np.eye(4)), 1e-8, feedback
    )
    admitted = AdmittedState(LiveState(registered, feedback), 1e-8, time.monotonic())
    k = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1.0]])
    observation = Observation(rgb, depth, k, "optical", 10, "torso_link", np.eye(4), robot_state=admitted)
    return observation, Proposals(mask, np.zeros((4, 3)), np.eye(4)[None], [0.9])


def frames(*images):
    ok = SimpleNamespace(healthy=True)
    # Each frame arrives as it is taken: the watch takes older ones as stale.
    _, capture, _ = watcher_inputs([(rgb, depth, ok) for rgb, depth, _ in images], ["approach"], time.monotonic)
    return capture


def wait(predicate, timeout_s=5.0):
    ended = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < ended:
        time.sleep(0.01)
    return predicate()


def armed(capture, **options):
    watch = X2TargetWatch(capture, far_spheres, period_s=0.0, **options)
    watch.arm(*planned(), plan=None)
    return watch


def test_in_stop_mode_a_moved_target_halts_the_approach_once_the_votes_decide():
    watch = armed(frames(STILL, MOVED), imu=LevelImu())
    assert watch.moved() is None
    watch.phase("approach", "start")
    assert wait(lambda: watch.moved() is not None)
    assert watch.moved().startswith("target moved (votes [")
    report = watch.disarm()
    assert report["moved"] and report["mode"] == "stop" and report["halted"] == watch.moved()
    assert report["imu"] and "records" not in report and report["votes"]["moved"] >= 3


def test_arming_again_stops_the_watcher_armed_before():
    """An old watcher would wake on the new phases and compare against an old image and mask."""
    watch = armed(frames(STILL, STILL), imu=LevelImu())
    first = watch._watcher
    watch.arm(*planned(), plan=None)
    assert watch._watcher is not first and not first._thread.is_alive()
    watch.disarm()


def test_in_log_mode_the_votes_are_recorded_and_nothing_halts():
    watch = armed(frames(STILL, MOVED), mode="log")
    watch.phase("approach", "start")
    assert wait(lambda: watch._watcher.watch.moved)
    assert watch.moved() is None
    report = watch.disarm()
    assert report["moved"] and report["halted"] is None and not report["imu"]


def test_without_the_imu_the_background_measures_the_camera_shift():
    watch = armed(frames(STILL, MOVED))
    watch.phase("approach", "start")
    assert wait(lambda: watch.moved() is not None)
    records = watch._watcher.records
    report = watch.disarm()
    assert not report["imu"] and report["votes"]["unknown"] == 0
    assert all("background_shift_px" in record for record in records)


def test_a_frame_whose_camera_shift_nothing_measures_votes_unknown_when_stopping():
    flat = [(np.full_like(rgb, 128), depth, mask) for rgb, depth, mask in (STILL, MOVED)]
    for (rgb, _, mask), (still_rgb, _, _) in zip(flat, (STILL, MOVED), strict=True):
        rgb[mask] = still_rgb[mask]  # only the target is textured
    watch = X2TargetWatch(frames(*flat), far_spheres, period_s=0.0)
    observation, proposals = planned(flat[0])
    watch.arm(observation, proposals, None)
    watch.phase("approach", "start")
    assert wait(lambda: len(watch._watcher.records) >= 6)
    assert watch.moved() is None
    assert watch.disarm()["votes"]["unknown"] >= 6


def test_frames_are_taken_from_the_approach_until_the_close():
    captures = []
    capture = frames(STILL)

    def counted():
        captures.append(1)
        return capture()

    watch = armed(counted, imu=LevelImu())
    watch.phase("preshape", "start")
    time.sleep(0.1)
    assert not captures  # nothing before the first arm segment
    watch.phase("approach", "start")
    assert wait(lambda: len(captures) >= 3)
    watch.phase("close", "start")
    assert wait(lambda: not watch._watcher._thread.is_alive())
    taken = len(captures)
    time.sleep(0.1)
    assert len(captures) == taken and watch.disarm()["votes"]["still"] == taken


def unwatchable():
    """A planned observation the X2 watch cannot compare (not an X2 capture)."""
    observation, proposals = planned()
    observation = Observation(
        observation.rgb, observation.depth_m, observation.intrinsics, "optical", 10, "torso_link", np.eye(4)
    )
    return observation, proposals


def test_in_stop_mode_a_watch_that_cannot_start_refuses_the_execution():
    watch = X2TargetWatch(frames(STILL), far_spheres)
    with pytest.raises(GraspFailure) as raised:
        watch.arm(*unwatchable(), None)
    assert raised.value.code == "WATCH_FAILED" and "not started: TypeError" in str(raised.value)
    assert not watch.started and watch.status().state is WatchState.UNAVAILABLE
    report = watch.disarm()
    assert report["error"] == "not started: TypeError: the X2 watch compares X2 captures"
    assert report["state"] == "unavailable" and report["fault"].startswith("watch unavailable: not started")
    with pytest.raises(ValueError, match="watch mode"):
        X2TargetWatch(frames(STILL), far_spheres, mode="off")


def test_in_log_mode_a_watch_that_cannot_start_is_recorded_and_the_execution_goes_on():
    watch = X2TargetWatch(frames(STILL), far_spheres, mode="log")
    watch.arm(*unwatchable(), None)
    watch.phase("approach", "start")
    assert watch.moved() is None and watch.failed() is None
    assert watch.disarm()["fault"].startswith("watch unavailable: not started: TypeError")


QUICK = WatchTimeouts(frame_s=0.3, unknown_s=0.5, image_age_s=0.5)


def test_a_still_target_reads_stable_once_the_approach_starts_and_ended_from_the_close_on():
    watch = armed(frames(STILL), imu=LevelImu(), timeouts=QUICK)
    time.sleep(0.4)  # longer than the frame timeout, but nothing is watched yet
    assert watch.status().state is WatchState.WAITING and watch.failed() is None
    watch.phase("approach", "start")
    assert wait(lambda: watch.status().state is WatchState.STABLE)
    status = watch.status()
    assert status.phase == "approach" and status.fresh_frames >= 1 and status.last_image_stamp_ns > 10
    assert watch.failed() is None
    watch.phase("close", "start")
    time.sleep(0.4)  # no frames any more: the watched steps are over, not the watch broken
    assert watch.status().state is WatchState.ENDED and watch.failed() is None
    report = watch.disarm()
    assert report["state"] == "ended" and report["fault"] is None


def test_a_moved_target_reads_moved_and_is_not_a_failure():
    watch = armed(frames(STILL, MOVED), imu=LevelImu(), timeouts=QUICK)
    watch.phase("approach", "start")
    assert wait(lambda: watch.status().state is WatchState.MOVED)
    assert watch.status().reason == watch.moved() and watch.failed() is None


def stalled(after, *images):
    """Frames from ``images`` until ``after`` are taken; then always the last one again (a camera that froze)."""
    capture = frames(*images)
    taken, last = [0], []

    def frozen():
        if taken[0] < after:
            taken[0] += 1
            last[:] = [capture()]
        return last[0]

    return frozen


def test_a_camera_that_stops_updating_makes_the_watch_unavailable_mid_approach():
    watch = armed(stalled(2, STILL), imu=LevelImu(), timeouts=QUICK)
    watch.phase("approach", "start")
    assert wait(lambda: watch.status().state is WatchState.STABLE)
    assert wait(lambda: watch.failed() is not None, 3.0)
    reason = watch.failed()
    assert reason.startswith("watch unavailable: no fresh image for") and "(not newer)" in reason
    assert watch.status().fresh_frames == 2 and watch.moved() is None
    assert watch.disarm()["fault"] == watch._fault


def test_captures_that_keep_failing_make_the_watch_unavailable_not_dead_on_the_first():
    capture, calls = frames(STILL), [0]

    def flaky():
        calls[0] += 1
        if calls[0] in (2, 3) or calls[0] > 6:
            raise RuntimeError("No recent RGB-D set has image-time feedback and TF")
        return capture()

    watch = armed(flaky, imu=LevelImu(), timeouts=QUICK)
    watch.phase("approach", "start")
    assert wait(lambda: watch.status().fresh_frames >= 4)  # two failures did not end the watch
    assert wait(lambda: watch.failed() is not None, 3.0)
    assert "capture failed: RuntimeError: No recent RGB-D set" in watch.failed()


def test_an_old_image_is_not_a_fresh_frame():
    capture = frames(STILL)

    def old():
        state = capture()
        state.observation.received_at -= 2.0  # taken now, but two seconds after it arrived
        return state

    watch = armed(old, imu=LevelImu(), timeouts=QUICK)
    watch.phase("approach", "start")
    assert wait(lambda: watch.failed() is not None, 3.0)
    assert "image 2.0" in watch.failed() and watch.status().fresh_frames == 0


def test_a_watch_whose_comparison_breaks_is_unavailable_at_once():
    watch = armed(frames(STILL), imu=LevelImu(), timeouts=WatchTimeouts(frame_s=30.0))
    watch._watcher.watch.update = lambda *a, **k: 1 / 0
    watch.phase("approach", "start")
    assert wait(lambda: watch.failed() is not None)
    assert watch.failed() == "watch unavailable: watch stopped: ZeroDivisionError: division by zero"


def flat_frames():
    flat = [(np.full_like(rgb, 128), depth, mask) for rgb, depth, mask in (STILL, MOVED)]
    for (rgb, _, mask), (still_rgb, _, _) in zip(flat, (STILL, MOVED), strict=True):
        rgb[mask] = still_rgb[mask]  # only the target is textured: no camera shift is measured
    return flat


def test_frames_that_never_decide_are_tolerated_for_a_while_then_fail_the_execution():
    flat = flat_frames()
    watch = X2TargetWatch(frames(*flat), far_spheres, period_s=0.0, timeouts=QUICK)
    observation, proposals = planned(flat[0])
    watch.arm(observation, proposals, None)
    watch.phase("approach", "start")
    assert wait(lambda: watch.status().state is WatchState.UNKNOWN)
    assert watch.failed() is None  # briefly undecided: go on
    assert wait(lambda: watch.failed() is not None, 3.0)
    assert watch.failed().startswith("target not observable for") and "no camera shift" in watch.failed()
    assert watch.status().state is WatchState.UNKNOWN and watch.moved() is None


def test_in_log_mode_failures_are_recorded_but_never_stop_the_execution():
    watch = armed(stalled(1, STILL), mode="log", timeouts=QUICK)
    watch.phase("approach", "start")
    assert wait(lambda: watch.status().state is WatchState.UNAVAILABLE, 3.0)
    assert watch.failed() is None and watch.moved() is None
    assert watch.disarm()["fault"].startswith("watch unavailable: no fresh image")


def test_votes_that_decide_moved_just_after_the_grasp_ended_are_still_reported():
    """The last frame of the grasp segment is compared after the next step began."""
    late, release = threading.Event(), threading.Event()
    capture = frames(STILL, MOVED, MOVED, MOVED)
    taken = [0]

    def slow_last():
        taken[0] += 1
        if taken[0] == 4:
            late.set()
            release.wait(5.0)  # compared only once the next step has started
        return capture()

    watch = armed(slow_last, imu=LevelImu(), timeouts=QUICK)
    watch.phase("grasp", "start")
    assert late.wait(5.0) and watch.moved() is None
    watch.phase("settle_grasp", "start")
    release.set()
    assert wait(lambda: watch.moved() is not None)
    assert watch.status().state is WatchState.MOVED and watch.failed() is None
    assert watch.disarm()["state"] == "moved"


def test_a_move_blocked_after_close_does_not_cross_the_watch_status_interface():
    """Raw votes can finish after close, but the accepted verdict is the interface boundary."""
    entered, release = threading.Event(), threading.Event()
    capture = frames(MOVED)

    def slow_capture():
        entered.set()
        assert release.wait(5.0)
        return capture()

    watch = armed(
        slow_capture,
        imu=LevelImu(),
        timeouts=QUICK,
        config=WatchConfig(moved_frames=1, window_frames=1),
    )
    watch.phase("approach", "start")
    assert entered.wait(5.0)
    watch.phase("close", "start")
    release.set()
    assert wait(lambda: not watch._watcher.running)

    assert watch._watcher.watch.moved and watch._watcher.notified is None
    assert watch.moved() is None and watch.status().state is WatchState.ENDED
    assert watch.disarm()["state"] == "ended"


def undecided(timeouts=QUICK):
    flat = flat_frames()
    watch = X2TargetWatch(frames(*flat), far_spheres, period_s=0.0, timeouts=timeouts)
    observation, proposals = planned(flat[0])
    watch.arm(observation, proposals, None)
    watch.phase("grasp", "start")
    return watch


def test_a_failure_reached_before_the_watched_steps_end_is_still_reported_after_them():
    watch = undecided()
    time.sleep(0.7)  # undecided past the 0.5 s limit, and nobody asked meanwhile
    watch.phase("settle_grasp", "start")
    assert watch.failed().startswith("target not observable for")
    time.sleep(0.2)
    status = watch.status()
    assert status.state is WatchState.UNKNOWN and watch.failed().startswith("target not observable for")


def test_undecided_frames_that_had_not_reached_the_limit_when_the_watch_ended_fail_nothing():
    watch = undecided(WatchTimeouts(frame_s=0.3, unknown_s=2.0))
    assert wait(lambda: watch.status().state is WatchState.UNKNOWN)
    watch.phase("settle_grasp", "start")
    time.sleep(2.2)  # past the limit, but after the watched steps
    assert watch.status().state is WatchState.ENDED and watch.failed() is None


def test_which_states_fail_a_watch():
    timeouts = WatchTimeouts(unknown_s=2.0)
    assert watch_failure(WatchStatus(WatchState.UNAVAILABLE, reason="x"), timeouts) == "watch unavailable: x"
    assert watch_failure(WatchStatus(WatchState.UNKNOWN, unknown_s=1.9), timeouts) is None
    assert watch_failure(WatchStatus(WatchState.UNKNOWN, unknown_s=2.1, reason="hidden"), timeouts).endswith("hidden")
    for state in (WatchState.WAITING, WatchState.STABLE, WatchState.MOVED, WatchState.ENDED):
        assert watch_failure(WatchStatus(state, unknown_s=99.0), timeouts) is None
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, 61.0, "1"):
        with pytest.raises(ValueError, match="frame_s must be a finite number"):
            WatchTimeouts(frame_s=bad)
    assert WatchTimeouts(unknown_s=60).unknown_s == 60


def test_the_imu_is_paused_once_the_execution_it_watched_ends():
    imu = LevelImu()
    watch = armed(frames(STILL), imu=imu)
    watch.phase("approach", "start")
    assert imu.pauses == 0
    watch.disarm()
    assert imu.pauses == 1
    unstarted = X2TargetWatch(frames(STILL), far_spheres, imu=imu)
    unstarted.disarm()
    assert imu.pauses == 2


def test_the_moved_reason_names_its_votes():
    record = dict(depth_changed_fraction=0.41234, rgb_changed_fraction=0.9)
    assert moved_reason(record) == "target moved (votes [0.412, 0.9])"


def test_the_chest_imu_pauses_after_an_image_or_while_idle_and_catches_up_on_resume():
    rclpy = pytest.importorskip("rclpy")
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Imu

    from robot_config.contract_utils import qos_profile_from_dict
    from robot_runtime.interface_description import build_description
    from robot_runtime.profile import load_profile

    profile = os.path.join(os.path.dirname(__file__), "..", "..", "aimdk_robot", "profiles", "x2_ultra_omnihand.yaml")
    description = build_description(load_profile(profile), use_sim=True)
    interface = description["interfaces"]["imu.chest"]
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    node = rclpy.create_node("chest_imu_test")
    qos = qos_profile_from_dict(interface["qos"])
    qos.depth = 500
    publisher = node.create_publisher(Imu, interface["endpoint"], qos)
    imu = ChestImu(node, description)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    t0 = 7 * 10**9

    def publish(first, last):
        for index in range(first, last):
            message = Imu()
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(t0 + index * 2_000_000, 10**9)
            message.orientation.w = 1.0
            publisher.publish(message)

    def known(index):
        return imu.orientation.at(t0 + index * 2_000_000 - 1_000_000) is not None

    try:
        assert wait(lambda: publisher.get_subscription_count() > 0)
        publish(0, 10)
        assert wait(lambda: known(9))
        imu.pause_after(t0 + 10 * 2_000_000)  # keeps 20 ms past the image, sample 20
        publish(10, 40)
        assert wait(lambda: known(20))
        time.sleep(0.2)
        assert not known(22)  # paused: later samples wait in the queue
        imu.resume()
        assert wait(lambda: known(39))
        imu.pause()  # idle: nothing is taken until the next image or plan
        assert imu.paused
        publish(40, 60)
        time.sleep(0.2)
        assert not known(41)
        imu.resume()
        assert not imu.paused and wait(lambda: known(59))
    finally:
        imu.close()
        executor.shutdown()
        spinner.join(timeout=5.0)
        node.destroy_node()
        if owned and rclpy.ok():
            rclpy.shutdown()
