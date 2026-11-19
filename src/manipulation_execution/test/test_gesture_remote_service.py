"""A real loopback ROS service, synthetic pixels, no model or robot publishers."""

import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from embodied_common.hand_gesture_contract import RequestMeta, ResultMeta
from ibrobot_msgs.srv import ClassifyHandGesture
from manipulation_execution.hand_perception.remote_transport import ModelServiceTransport

CLIENT = "00000000-0000-4000-8000-000000000001"
SESSION = "00000000-0000-4000-8000-000000000002"
SERVER = "00000000-0000-4000-8000-000000000003"
ENGINE = "a" * 64


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def service():
    import os

    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    rclpy.init()
    server, client = Node("gesture_test_server"), Node("gesture_test_client")
    seen, release, entered = [], threading.Event(), threading.Event()
    finished = threading.Event()
    block = [False]

    def handle(request, reply):
        meta = RequestMeta.from_json(request.metadata_json)
        seen.append((meta, request.image))
        if meta.op == "infer" and block[0]:
            entered.set()
            assert release.wait(3.0), "test must release its service callback"
        reply.result_json = ResultMeta(
            meta,
            SERVER,
            ENGINE,
            status="retired" if meta.op == "retire" else "ok",
            gesture="holy" if meta.op == "infer" else "none",
            score=0.9 if meta.op == "infer" else 0.0,
            idle=True,
            retired_through=meta.retire_through,
        ).to_json()
        reply.model.ready = True
        reply.model.deployment_fingerprint = "b" * 64
        reply.model.backend = "mock"
        reply.inference_time_ms, reply.success = 1.0, True
        finished.set()
        return reply

    server.create_service(ClassifyHandGesture, "/test_hand_gesture/classify", handle)
    transport = ModelServiceTransport(client, "/test_hand_gesture/classify")
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(server)
    executor.add_node(client)
    stopping, spin_errors = threading.Event(), []

    def spin():
        try:
            while not stopping.is_set():
                executor.spin_once(timeout_sec=0.02)
        except Exception as exc:
            spin_errors.append(exc)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    assert wait_for(transport.ready)
    try:
        yield transport, seen, block, entered, release, finished
    finally:
        release.set()
        stopping.set()
        executor.wake()
        thread.join(3.0)
        executor.shutdown(timeout_sec=3.0)
        server.destroy_node()
        client.destroy_node()
        rclpy.shutdown()
        assert not thread.is_alive()
        assert not spin_errors


def test_native_typed_request_keeps_pixels_stamp_and_metadata_atomic(service):
    transport, seen, *_ = service
    stamp = time.time_ns()
    meta = RequestMeta(
        CLIENT,
        time.monotonic_ns(),
        1,
        2,
        op="infer",
        session_token=SESSION,
        actor_id=3,
        stamp_ns=stamp,
        pear_stamp_ns=stamp - 10_000_000,
        crop="both",
        roi=(-10, 20, 90, 120),
        source_size=(640, 480),
        budget_ms=400,
    )
    pixels = np.full((320, 320, 3), 17, np.uint8)
    future = transport.begin(meta, pixels, "real_camera_frame")
    assert wait_for(future.done)
    reply = future.result()
    echoed, image = seen[-1]
    assert echoed == meta and bytes(image.data) == pixels.tobytes()
    assert image.header.stamp.sec * 1_000_000_000 + image.header.stamp.nanosec == stamp
    assert image.header.frame_id == "real_camera_frame" and image.step == 960
    assert reply.success and ResultMeta.from_json(reply.result_json).request == meta
    transport.reap(future)


def test_health_probe_has_no_image_or_classification_work(service):
    transport, seen, *_ = service
    meta = RequestMeta(CLIENT, time.monotonic_ns(), 1, 1)
    future = transport.begin(meta, None, "")
    assert wait_for(future.done)
    assert seen[-1][0].op == "probe" and len(seen[-1][1].data) == 0
    assert ResultMeta.from_json(future.result().result_json).gesture == "none"
    transport.reap(future)


def test_cancelled_image_request_does_not_claim_server_work_stopped(service):
    transport, _, block, entered, release, finished = service
    block[0] = True
    stamp = time.time_ns()
    meta = RequestMeta(
        CLIENT,
        time.monotonic_ns(),
        1,
        2,
        op="infer",
        session_token=SESSION,
        actor_id=3,
        stamp_ns=stamp,
        pear_stamp_ns=stamp,
        crop="both",
        roi=(0, 0, 320, 320),
        source_size=(640, 480),
        budget_ms=400,
    )
    future = transport.begin(meta, np.zeros((320, 320, 3), np.uint8), "camera")
    assert entered.wait(2.0) and not future.done()
    future.cancel()
    assert future.cancelled()
    assert not finished.wait(0.05)  # Cancellation ended only the local future, not the service callback.
    release.set()
    assert finished.wait(2.0)
    probe = transport.begin(RequestMeta(CLIENT, meta.producer_start_mono_ns, 1, 3), None, "")
    assert wait_for(probe.done)  # A later serialized service call also completed before destroying executor handles.
    transport.reap(probe)
    transport.reap(future)
