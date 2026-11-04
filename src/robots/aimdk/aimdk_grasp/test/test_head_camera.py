"""The head camera observation source, against a fake head over the X2 runtime's described topics."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("warp")

from aimdk_grasp.feedback import FeedbackHistory  # noqa: E402
from aimdk_grasp.hand import HandMapping  # noqa: E402
from aimdk_grasp.head_camera import BASE_FRAME, HeadCamera, register_set  # noqa: E402
from aimdk_grasp.joints import BODY_JOINTS, HEAD, LEFT_HAND  # noqa: E402
from aimdk_grasp.observation import AdmittedState, RegisteredObservation  # noqa: E402
from aimdk_grasp.robot_model import planning_model  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, JointState  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from test_observation import OPEN  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: E402

from ibrobot_msgs.msg import RuntimeStatus  # noqa: E402
from manipulation_execution.grasp_pipeline.contracts import GraspFailure  # noqa: E402
from robot_config.contract_utils import qos_profile_from_dict  # noqa: E402
from robot_runtime.interface_description import build_description  # noqa: E402
from robot_runtime.profile import load_profile  # noqa: E402

PROFILE = Path(__file__).resolve().parents[2] / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
W, H, FOCAL = 16, 12, 12.0
COLOR, DEPTH = "head_color_optical", "head_depth_optical"


def description():
    return build_description(load_profile(os.fspath(PROFILE)), use_sim=True)


def encoded(stamp: int) -> float:
    """A value that names its image: TF x and head yaw both carry it."""
    return (stamp % 10**9) * 1e-9


def info(frame: str, header) -> CameraInfo:
    k = [FOCAL, 0.0, (W - 1) / 2, 0.0, FOCAL, (H - 1) / 2, 0.0, 0.0, 1.0]
    return CameraInfo(header=header, width=W, height=H, distortion_model="plumb_bob", d=[0.0] * 5, k=k)


class FakeHead(Node):
    """The runtime's camera, joint views and status, plus the platform TF, all at the image stamps."""

    def __init__(self, interfaces, open_hand):
        super().__init__("fake_head")
        self.open_hand, self.tf_on = open_hand, True
        # Joint views stop while False (a feedback gap); the runtime's lifecycle as published.
        self.views_on, self.lifecycle = True, "ACTIVE"

        def publisher(kind, name):
            interface = interfaces[name]
            return self.create_publisher(kind, interface["endpoint"], qos_profile_from_dict(interface.get("qos")) or 10)

        self.images = [
            publisher(kind, f"camera.rgbd_head_front.{name}")
            for kind, name in (
                (Image, "color"),
                (Image, "depth"),
                (CameraInfo, "color_info"),
                (CameraInfo, "depth_info"),
            )
        ]
        self.views = {
            name: (publisher(JointState, name), interfaces[name]["joint_names"])
            for name in ("joint.state", "joint.head_state", "joint.hand_state")
        }
        self.status = publisher(RuntimeStatus, "runtime.status")
        self.tf = TransformBroadcaster(self)
        depth_in_color = TransformStamped()
        depth_in_color.header.frame_id, depth_in_color.child_frame_id = COLOR, DEPTH
        depth_in_color.transform.rotation.w = 1.0
        StaticTransformBroadcaster(self).sendTransform(depth_in_color)
        self.create_timer(1 / 30, self._tick)

    def _tick(self):
        now = self.get_clock().now()
        stamp = now.nanoseconds
        self.status.publish(RuntimeStatus(lifecycle=self.lifecycle, stamp=now.to_msg()))
        for pub, names in self.views.values() if self.views_on else ():
            values = {joint: 0.0 for joint in BODY_JOINTS}
            values[HEAD[0]] = encoded(stamp)
            values.update(zip(LEFT_HAND, self.open_hand, strict=True))
            message = JointState(name=list(names), position=[values.get(joint, 0.0) for joint in names])
            message.header.stamp = now.to_msg()
            pub.publish(message)
        if self.tf_on:
            camera = TransformStamped()
            camera.header.stamp, camera.header.frame_id, camera.child_frame_id = now.to_msg(), BASE_FRAME, COLOR
            camera.transform.translation.x, camera.transform.translation.z = encoded(stamp), 0.4
            camera.transform.rotation.w = 1.0
            self.tf.sendTransform(camera)
        color, depth = (
            Image(encoding="rgb8", width=W, height=H, step=3 * W, data=bytes(range(256)) * 2 + bytes(64)),
            Image(encoding="16uc1", width=W, height=H, step=2 * W, data=np.full((H, W), 1250, "<u2").tobytes()),
        )
        headers = [Header(frame_id=frame, stamp=now.to_msg()) for frame in (COLOR, DEPTH)]
        color.header, depth.header = headers
        for pub, message in zip(
            self.images, (color, depth, info(COLOR, headers[0]), info(DEPTH, headers[1])), strict=True
        ):
            pub.publish(message)


#: The head's GPU lock (head_parts shares one with the planner).
GPU = threading.RLock()


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


@pytest.fixture
def head():
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    open_hand = HandMapping(planning_model(), {"open": OPEN}).preset("open")
    fake = FakeHead(description()["interfaces"], open_hand)
    consumer = rclpy.create_node("head_camera_test")
    camera = HeadCamera(consumer, description(), FeedbackHistory(), device="cpu", gpu_lock=GPU)
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (fake, consumer):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield fake, camera
    finally:
        camera.close()
        executor.shutdown()
        thread.join(timeout=5.0)  # before its nodes' timers are destroyed under it
        for node in (consumer, fake):
            node.destroy_node()
        if owned and rclpy.ok():
            rclpy.shutdown()


def capture(camera, timeout_s=10.0, cancelled=lambda: False):
    """Retry until the graph is up; the source itself waits only CAPTURE_WAIT_S."""
    ended = time.monotonic() + timeout_s
    while True:
        try:
            return camera.capture(deadline=time.monotonic() + 1.0, cancelled=cancelled)
        except GraspFailure:
            if time.monotonic() > ended:
                raise
            time.sleep(0.1)


def test_a_capture_is_registered_with_feedback_and_tf_from_its_own_image_time(head):
    _, camera = head
    observation = capture(camera)
    assert (observation.frame_id, observation.base_frame) == (COLOR, BASE_FRAME)
    np.testing.assert_allclose(observation.depth_m, 1.25)
    assert observation.rgb.shape == (H, W, 3) and observation.intrinsics[0, 0] == FOCAL
    admitted = observation.robot_state
    assert isinstance(admitted, AdmittedState)
    registered = admitted.live.observation
    assert isinstance(registered, RegisteredObservation) and registered.calibration_id
    assert admitted.live.feedback.received_at >= registered.image_feedback.received_at
    assert 0 <= admitted.admitted_at - registered.received_at <= 0.5
    named = encoded(observation.stamp_ns)
    assert observation.base_from_camera[0, 3] == pytest.approx(named)
    assert registered.image_feedback.positions[HEAD[0]] == pytest.approx(named)
    assert registered.received_at == pytest.approx(observation.stamp_ns * 1e-9)
    call = camera.calls[-1]
    assert call["total_s"] >= call["registration_s"] and call["image_age_s"] >= 0.0


def test_a_capture_after_a_quiet_spell_takes_a_set_from_now_not_from_before(head):
    _, camera = head
    first = capture(camera)
    time.sleep(2.0)  # nothing is taken meanwhile; the camera keeps publishing
    second = capture(camera)
    age_s = (camera._node.get_clock().now().nanoseconds - second.stamp_ns) * 1e-9
    assert second.stamp_ns - first.stamp_ns >= 1.5e9 and age_s < 0.5


def test_the_current_feedback_is_taken_after_a_slow_registration(head, monkeypatch):
    from aimdk_grasp import head_camera

    _, camera = head
    capture(camera)
    register = head_camera.register_set

    def slow(*args, **kwargs):
        time.sleep(0.3)  # the views keep arriving meanwhile
        return register(*args, **kwargs)

    monkeypatch.setattr(head_camera, "register_set", slow)
    observation = camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    current = observation.robot_state.live.feedback.received_at
    # Fresh by the 0.1 s gate: read after the registration, not before it.
    assert current - observation.stamp_ns * 1e-9 >= 0.25


def test_a_capture_registers_only_once_the_gpu_is_free(head):
    """While the planner holds the GPU (cuRobo capturing CUDA graphs), registering would fail."""
    _, camera = head
    capture(camera)
    taken = []
    with GPU:  # this thread stands in for the planner
        worker = threading.Thread(
            target=lambda: taken.append(camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)),
            daemon=True,
        )
        worker.start()
        worker.join(0.3)
        assert worker.is_alive() and not taken
    worker.join(5.0)
    assert taken and camera.calls[-1]["gpu_wait_s"] >= 0.25


def test_a_capture_behind_a_long_plan_gives_up_at_its_deadline_or_on_cancel(head):
    _, camera = head
    capture(camera)
    with HeldElsewhere(GPU):  # a plan that outlasts the capture
        began = time.monotonic()
        with pytest.raises(GraspFailure) as raised:
            camera.capture(deadline=time.monotonic() + 0.3, cancelled=lambda: False)
        assert raised.value.code == "OBSERVATION_UNAVAILABLE" and "GPU was busy" in str(raised.value)
        assert time.monotonic() - began < 1.0
        asked = time.monotonic() + 0.2
        with pytest.raises(GraspFailure) as raised:
            camera.capture(deadline=time.monotonic() + 30.0, cancelled=lambda: time.monotonic() > asked)
        assert raised.value.code == "CANCELED"


def test_without_tf_a_capture_reports_the_missing_transform(head):
    fake, camera = head
    capture(camera)
    fake.tf_on = False
    time.sleep(0.4)  # the newest sets outrun the last head TF
    started = time.monotonic()
    with pytest.raises(GraspFailure, match="image-time feedback and TF") as raised:
        camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert raised.value.code == "OBSERVATION_UNAVAILABLE" and time.monotonic() - started < 1.0
    assert raised.value.retryable  # nothing has moved: the pipeline observes again


def after(seconds, action):
    timer = threading.Timer(seconds, action)
    timer.start()
    return timer


def test_a_feedback_gap_shorter_than_the_wait_is_waited_out(head):
    """Live 2026-11-03: feedback reaching the head stopped for ~400 ms; 0.2 s of waiting failed the pick."""
    fake, camera = head
    capture(camera)
    fake.views_on = False
    time.sleep(0.1)  # the newest images now have no feedback near them
    resumed = after(0.3, lambda: setattr(fake, "views_on", True))
    try:
        observation = camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    finally:
        resumed.cancel()
    assert observation.report["wait_s"] >= 0.2 and observation.report["rounds"] > 1


def test_feedback_that_turns_unhealthy_briefly_is_waited_out_then_refused(head):
    """An unhealthy runtime verdict or unsynchronized views waited CAPTURE_WAIT_S like a missing sample."""
    from aimdk_grasp.head_camera import CAPTURE_WAIT_S

    fake, camera = head
    capture(camera)
    fake.lifecycle = "FAULT"
    time.sleep(0.1)
    recovered = after(0.3, lambda: setattr(fake, "lifecycle", "ACTIVE"))
    try:
        observation = camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    finally:
        recovered.cancel()
    assert any("Healthy image-time feedback" in reason for reason in observation.report["invalid"])
    fake.lifecycle = "FAULT"
    time.sleep(0.1)
    started = time.monotonic()
    with pytest.raises(GraspFailure, match="Healthy image-time feedback") as raised:
        camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert raised.value.code == "OBSERVATION_UNAVAILABLE" and raised.value.retryable
    assert CAPTURE_WAIT_S <= time.monotonic() - started < CAPTURE_WAIT_S + 0.5


def test_a_stale_capture_is_taken_again_once_then_refused(head):
    _, camera = head
    capture(camera)
    calls = len(camera.calls)
    camera.clock = lambda: time.time() + 1.0  # every capture looks a second old
    with pytest.raises(GraspFailure, match="stale") as raised:
        camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert raised.value.code == "OBSERVATION_UNAVAILABLE" and len(camera.calls) == calls + 2
    assert camera.calls[-1]["feedback_wait_s"] >= 0.09  # it waited for fresh feedback before recapturing


def test_a_feedback_sample_just_late_waits_for_the_next_one_instead_of_a_new_image(head):
    """Recapturing costs a whole image and registration; the next feedback sample comes within a view period."""
    _, camera = head
    capture(camera)
    calls = len(camera.calls)
    caught_up = time.monotonic() + 0.06  # the feedback stream lags, then catches up
    camera.clock = lambda: time.time() + (0.2 if time.monotonic() < caught_up else 0.0)
    observation = camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: False)
    assert len(camera.calls) == calls + 1 and observation.report["recaptured"] is False
    assert observation.report["feedback_wait_s"] >= 0.03
    report = observation.report  # the capture's own entry, with the whole capture's time
    assert report["stamp_ns"] == observation.stamp_ns and report["capture_s"] >= report["total_s"]


def test_a_cancel_stops_the_wait(head):
    fake, camera = head
    fake.tf_on = False
    with pytest.raises(GraspFailure) as raised:
        camera.capture(deadline=time.monotonic() + 5.0, cancelled=lambda: True)
    assert raised.value.code == "CANCELED"


def test_the_images_are_received_in_a_participant_of_their_own(head):
    """Best-effort readers that share a participant with the images are handed every fragment of every frame."""
    fake, camera = head
    capture(camera)
    color = description()["interfaces"]["camera.rgbd_head_front.color"]["endpoint"]

    def participant(endpoints, node_name):
        prefixes = {bytes(info.endpoint_gid[:12]) for info in endpoints if info.node_name == node_name}
        assert len(prefixes) == 1, f"{node_name} endpoints: {endpoints}"
        return prefixes.pop()

    ended = time.monotonic() + 5.0
    while True:
        try:
            images = participant(fake.get_subscriptions_info_by_topic(color), "head_camera_test_camera")
            consumer = participant(fake.get_publishers_info_by_topic("/rosout"), "head_camera_test")
            break
        except AssertionError:
            if time.monotonic() > ended:
                raise
            time.sleep(0.1)
    assert images != consumer


def test_a_description_without_the_head_camera_is_refused():
    broken = description()
    del broken["interfaces"]["camera.rgbd_head_front.depth_info"]
    with pytest.raises(ValueError, match="camera.rgbd_head_front.depth_info"):
        HeadCamera(object(), broken, FeedbackHistory())


def test_a_set_whose_parts_are_apart_in_time_is_not_registered():
    def header(frame, stamp_ns):
        message = Header(frame_id=frame)
        message.stamp.sec, message.stamp.nanosec = divmod(stamp_ns, 10**9)
        return message

    stamp = 5 * 10**9
    color = Image(header=header(COLOR, stamp), encoding="rgb8", width=W, height=H, step=3 * W, data=bytes(3 * W * H))
    depth = Image(
        header=header(DEPTH, stamp + 60_000_000), encoding="16uc1", width=W, height=H, step=2 * W, data=bytes(2 * W * H)
    )
    with pytest.raises(ValueError, match="synchronization tolerance"):
        register_set(color, depth, info(COLOR, color.header), info(DEPTH, depth.header), np.eye(4), device="cpu")
