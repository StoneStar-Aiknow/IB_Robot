"""The X2 target watch adapter and the chest IMU it takes body rotation from."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.execution_watch import ChestImu, X2TargetWatch, moved_reason
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.joints import BODY_JOINTS
from aimdk_grasp.observation import AdmittedState, LiveState, RegisteredObservation
from aimdk_grasp.types import RGBDFrame
from test_target_watch import K, far_spheres, scene, watcher_inputs

from manipulation_execution.grasp_pipeline.contracts import Observation, Proposals

STILL, MOVED = scene(), scene(target_at=(30, 55))


class LevelImu:
    """A chest IMU that always reads level and counts the pauses it is asked for."""

    def __init__(self):
        self.orientation = SimpleNamespace(at=lambda stamp_ns: np.array([0.0, 0.0, 0.0, 1.0]))
        self.pauses = 0

    def pause(self):
        self.pauses += 1


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
    _, capture, _ = watcher_inputs([(rgb, depth, ok) for rgb, depth, _ in images], ["approach"])
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


def test_a_watch_that_cannot_start_leaves_the_execution_alone():
    watch = X2TargetWatch(frames(STILL), far_spheres)
    observation, proposals = planned()
    observation = Observation(
        observation.rgb, observation.depth_m, observation.intrinsics, "optical", 10, "torso_link", np.eye(4)
    )
    watch.arm(observation, proposals, None)
    watch.phase("approach", "start")
    assert watch.moved() is None
    assert watch.disarm() == dict(mode="stop", error="not started: TypeError: the X2 watch compares X2 captures")
    with pytest.raises(ValueError, match="watch mode"):
        X2TargetWatch(frames(STILL), far_spheres, mode="off")


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
