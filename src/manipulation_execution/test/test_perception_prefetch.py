"""Perception started before the goal, and the container handed from the pick to its place."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from test_grasp_pipeline import Proposer, Source, StreamingPlanner, observation, run, streaming
from test_place_pipeline import fixture

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, Observation
from manipulation_execution.grasp_pipeline.pipeline import GraspPipeline, PickMode, PipelineConfig
from manipulation_execution.grasp_pipeline.place_pipeline import PlacePipeline
from manipulation_execution.grasp_pipeline.prefetch import (
    PerceptionPrefetch,
    container_change,
    container_from_msg,
    container_snapshot,
    container_to_msg,
    decode_mask,
    encode_mask,
)

ON = PipelineConfig(prefetch_perception=True)


class Posture:
    """Measured joints: ``values`` in turn, the last one repeated."""

    def __init__(self, *values):
        self.values, self.calls = [np.asarray(v, dtype=float) for v in values] or [np.zeros(3)], 0

    def __call__(self):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def prefetching(posture=None, **options):
    backends, executor, effector, planner, watch = streaming()
    backends = type(backends)(**{**backends.__dict__, "posture": posture or Posture(), **options})
    return GraspPipeline(backends, ON, clock=lambda: 0.0), backends, planner


def wait_finished(pipeline):
    assert pipeline.prefetch._job.finished.wait(5.0)


def goal(pipeline, query="red bottle"):
    return pipeline.run(query, mode=PickMode.EXECUTE, deadline=100.0, cancelled=lambda: False)


def test_a_prefetched_capture_is_the_goals_observation_when_nothing_moved():
    pipeline, backends, _ = prefetching()
    assert pipeline.prefetch.start("red bottle") == ""
    outcome = goal(pipeline)
    assert outcome.success and outcome.held is True
    assert backends.observation.captures == 1  # the prefetch's, none by the goal
    assert outcome.report["perception_prefetch"]["used"] is True
    assert outcome.completed_phases[:3] == ["observe", "perceive", "plan"]


@pytest.mark.parametrize(
    "posture,reason",
    [
        (Posture(np.zeros(3), np.zeros(3), np.full(3, 0.1)), "the robot moved after the capture"),
        (Posture(np.zeros(3), np.full(3, 0.1)), "RuntimeError: the robot moved during the capture"),
    ],
)
def test_a_robot_that_moved_around_the_capture_is_observed_again(posture, reason):
    pipeline, backends, _ = prefetching(posture)
    pipeline.prefetch.start("red bottle")
    wait_finished(pipeline)
    outcome = goal(pipeline)
    assert outcome.success
    assert backends.observation.captures == 2
    assert outcome.report["perception_prefetch"]["used"] is False
    assert outcome.report["perception_prefetch"]["reason"] == reason


def test_another_target_is_stopped_and_the_goal_observes_its_own():
    pipeline, backends, _ = prefetching()
    pipeline.prefetch.start("a lemon")
    wait_finished(pipeline)
    outcome = goal(pipeline)
    assert outcome.success and backends.observation.captures == 2
    assert outcome.report["perception_prefetch"] == {"used": False, "reason": "prefetched 'a lemon'"} | {
        k: v for k, v in outcome.report["perception_prefetch"].items() if k.endswith("_s")
    }
    assert pipeline.prefetch._job.cancelled.is_set()


def test_a_retry_observes_again():
    pipeline, backends, planner = prefetching()
    planner.failures = [GraspFailure("NO_EXECUTABLE_CANDIDATE", "nothing", retryable=True)]
    pipeline.prefetch.start("red bottle")
    outcome = goal(pipeline)
    assert outcome.success and outcome.attempts == 2
    assert backends.observation.captures == 2  # prefetch for attempt 1, a new capture for attempt 2


def test_a_failed_prefetch_leaves_the_goal_as_it_was():
    pipeline, backends, _ = prefetching()
    backends.observation.failures = [GraspFailure("OBSERVATION_UNAVAILABLE", "no frame")]
    pipeline.prefetch.start("red bottle")
    outcome = goal(pipeline)
    assert outcome.success and backends.observation.captures == 2
    assert "OBSERVATION_UNAVAILABLE" in outcome.report["perception_prefetch"]["reason"]


def test_without_posture_or_config_nothing_is_prefetched():
    backends, *_ = streaming()
    assert GraspPipeline(backends, ON).prefetch is None  # no posture: a capture could not be checked
    with_posture = type(backends)(**{**backends.__dict__, "posture": Posture()})
    assert GraspPipeline(with_posture, PipelineConfig()).prefetch is None
    outcome = run(backends, config=ON)
    assert outcome.success and "perception_prefetch" not in outcome.report


def test_an_old_capture_is_not_used():
    now = [0.0]
    prefetch = PerceptionPrefetch(
        capture=Source().capture, propose=Proposer().propose, posture=Posture(), max_age_s=6.0, clock=lambda: now[0]
    )
    prefetch.start("red bottle")
    assert prefetch._job.perceived.wait(5.0)
    now[0] = 7.0
    taken, report = prefetch.take("red bottle", deadline=100.0, cancelled=lambda: False)
    assert taken is None and report["reason"] == "older than 6 s"


def test_a_prefetch_serves_one_goal():
    prefetch = PerceptionPrefetch(capture=Source().capture, propose=Proposer().propose, posture=Posture())
    prefetch.start("red bottle")
    assert prefetch.take("red bottle", deadline=float("inf"), cancelled=lambda: False)[0] is not None
    taken, report = prefetch.take("red bottle", deadline=float("inf"), cancelled=lambda: False)
    assert taken is None and report["reason"] == "already used by a goal"


def test_the_goal_waits_for_a_running_prefetch_of_its_target():
    gate = threading.Event()
    source = Source()

    def slow_capture(**kwargs):
        assert gate.wait(5.0)
        return source.capture(**kwargs)

    prefetch = PerceptionPrefetch(capture=slow_capture, propose=Proposer().propose, posture=Posture())
    prefetch.start("red bottle")
    threading.Timer(0.05, gate.set).start()
    taken, report = prefetch.take("red bottle", deadline=float("inf"), cancelled=lambda: False)
    assert taken is not None and report["used"] is True and source.captures == 1


def test_the_gateways_request_joins_the_agents_prefetch_and_binds_its_task():
    source = Source()
    prefetch = PerceptionPrefetch(
        capture=source.capture,
        propose=Proposer().propose,
        posture=Posture(),
        detect_container=lambda *a, **k: mask(),
    )
    assert prefetch.start("An apple", "a black bowl") == ""  # the Agent: no task yet
    first = prefetch._job
    assert prefetch.start("an apple", "a black bowl", root_task_id="task-1") == ""  # the Gateway
    assert prefetch._job is first and not first.cancelled.is_set() and first.root_task_id == "task-1"
    assert first.finished.wait(5.0) and source.captures == 1
    assert "joined_after_s" in first.report
    assert prefetch.container("task-1")[0] is not None
    taken, report = prefetch.take("an apple", deadline=float("inf"), cancelled=lambda: False)
    assert taken is not None and "joined_after_s" in report


@pytest.mark.parametrize(
    "second",
    [
        dict(query="an orange", container_query="a black bowl", root_task_id="task-1"),  # another target
        dict(query="an apple", container_query="", root_task_id="task-1"),  # no place after it
        dict(query="an apple", container_query="a black bowl", root_task_id="task-2"),  # another task
    ],
)
def test_a_different_request_restarts_the_prefetch(second):
    source = Source()
    prefetch = PerceptionPrefetch(capture=source.capture, propose=Proposer().propose, posture=Posture())
    prefetch.start("an apple", "a black bowl", root_task_id="task-1" if second["root_task_id"] == "task-2" else "")
    first = prefetch._job
    prefetch.start(**second)
    assert prefetch._job is not first and first.cancelled.is_set()


def test_a_used_failed_or_old_prefetch_is_not_joined():
    now = [0.0]
    source = Source()
    prefetch = PerceptionPrefetch(
        capture=source.capture, propose=Proposer().propose, posture=Posture(), clock=lambda: now[0]
    )
    prefetch.start("an apple")
    used = prefetch._job
    assert prefetch.take("an apple", deadline=float("inf"), cancelled=lambda: False)[0] is not None
    prefetch.start("an apple")
    assert prefetch._job is not used  # already taken by a goal
    failed = prefetch._job
    assert failed.perceived.wait(5.0)
    failed.error = "OBSERVATION_UNAVAILABLE: no frame"
    prefetch.start("an apple")
    assert prefetch._job is not failed
    old = prefetch._job
    now[0] = prefetch.max_age_s
    prefetch.start("an apple")
    assert prefetch._job is not old


def mask():
    m = np.zeros((4, 4), bool)
    m[1:3, 1:3] = True
    return m


def test_the_container_is_detected_on_the_prefetched_capture_for_its_own_task():
    detected = []

    def detect(observation, query, *, deadline, cancelled):
        detected.append(query)
        return mask()

    prefetch = PerceptionPrefetch(
        capture=Source().capture, propose=Proposer().propose, posture=Posture(), detect_container=detect
    )
    prefetch.start("an apple", "a black bowl", root_task_id="task-1")
    assert prefetch._job.finished.wait(5.0)
    container, _ = prefetch.container("task-1")
    assert detected == ["a black bowl"] and container.query == "a black bowl" and container.mask.sum() == 4
    assert prefetch.container("task-2")[0] is None
    prefetch.start("an apple", root_task_id="task-3")  # a pick that is not followed by a place
    assert prefetch._job.finished.wait(5.0) and detected == ["a black bowl"]
    assert prefetch.container("task-3")[0] is None


def frame(stamp, *, rgb=0, depth=1.0, shift=0.0):
    pose = np.eye(4)
    pose[0, 3] = shift
    return Observation(
        rgb=np.full((4, 4, 3), rgb, np.uint8),
        depth_m=np.full((4, 4), depth, np.float32),
        intrinsics=np.eye(3),
        frame_id="camera",
        stamp_ns=stamp,
        base_frame="base",
        base_from_camera=pose,
    )


@pytest.mark.parametrize(
    "later,query,reason",
    [
        (frame(2), "a black bowl", ""),
        (frame(2), "the red box", "another container was prefetched"),
        (frame(1), "a black bowl", "the place capture is not newer than the prefetched one"),
        (frame(2, shift=0.01), "a black bowl", "the camera moved"),
        (frame(2, rgb=200), "a black bowl", "the container region changed"),
        (frame(2, depth=1.5), "a black bowl", "the container region changed"),
    ],
)
def test_a_prefetched_container_stands_only_for_an_unchanged_view(later, query, reason):
    prefetched = container_snapshot(frame(1), mask(), "a black bowl")
    assert container_change(prefetched, later, query)[0] == reason


def test_the_place_reuses_an_unchanged_container_and_detects_a_changed_one():
    held, backends = fixture()
    proposed, reused = [], []
    container = backends.proposer.propose()
    backends = type(backends)(
        **{
            **backends.__dict__,
            "proposer": SimpleNamespace(
                propose=lambda *a, **k: proposed.append(1) or container,
                from_mask=lambda observation, m, *a, **k: reused.append(m.sum()) or container,
            ),
        }
    )
    backends.observation.captures = 1  # the place's capture is newer than the pick's (stamp 1)
    still = container_snapshot(observation(1), np.ones((4, 4), bool), "the blue bowl on the left")
    outcome = PlacePipeline(backends).run(
        held, "the blue bowl on the left", deadline=float("inf"), cancelled=lambda: False, prefetched=still
    )
    assert outcome.success and reused == [16] and proposed == []
    assert outcome.report["container_prefetch"]["used"] is True

    held, backends2 = fixture()
    backends2 = type(backends2)(**{**backends2.__dict__, "proposer": backends.proposer})
    backends2.observation.captures = 1
    other = container_snapshot(observation(1), np.ones((4, 4), bool), "a black bowl")
    outcome = PlacePipeline(backends2).run(
        held, "the blue bowl on the left", deadline=float("inf"), cancelled=lambda: False, prefetched=other
    )
    assert outcome.success and proposed == [1]
    assert outcome.report["container_prefetch"] == {
        "used": False,
        "reason": "another container was prefetched",
        "age_s": outcome.report["container_prefetch"]["age_s"],
    }


def test_a_container_survives_the_claim_message():
    from sensor_msgs.msg import Image

    from ibrobot_msgs.msg import PrefetchedContainer

    prefetched = container_snapshot(frame(1, rgb=7, depth=0.8), mask(), "a black bowl")
    message = container_to_msg(prefetched, PrefetchedContainer, encode_mask=encode_mask, image_class=Image)
    back = container_from_msg(message, decode_mask=decode_mask)
    assert back.query == "a black bowl" and back.roi == prefetched.roi and back.stamp_ns == 1
    assert np.array_equal(back.mask, prefetched.mask) and np.array_equal(back.rgb, prefetched.rgb)
    assert np.allclose(back.depth, prefetched.depth) and np.allclose(back.base_from_camera, np.eye(4))
    assert container_from_msg(PrefetchedContainer(), decode_mask=decode_mask) is None
    assert container_to_msg(None, PrefetchedContainer, encode_mask=encode_mask, image_class=Image).query == ""


def test_the_claim_message_carries_pixels_as_byte_arrays(monkeypatch):
    """rclpy checks every byte of bytes assigned to a uint8[] field (70-80 ms per ROI); array('B') skips it."""
    from array import array

    import sensor_msgs.msg

    class Plain:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    monkeypatch.setattr(sensor_msgs.msg, "CompressedImage", Plain)
    prefetched = container_snapshot(frame(1, rgb=7, depth=0.8), mask(), "a black bowl")

    class Message:
        pass

    message = container_to_msg(prefetched, Message, encode_mask=encode_mask, image_class=Plain)
    assert all(isinstance(part.data, array) for part in (message.mask, message.rgb, message.depth))


def test_custody_hands_the_container_to_the_claim():
    from manipulation_execution.held_custody import HeldCustody, container_msg

    held, _ = fixture()
    custody = HeldCustody()
    prefetched = container_snapshot(frame(1), mask(), "a black bowl")
    custody.accept(held, recover=lambda: None, release=lambda: None, holding=lambda: "", container=prefetched)
    assert custody.container is prefetched
    assert container_msg(custody.container).query == "a black bowl"
    assert container_msg(None).query == ""


def test_prefetch_settings_are_validated():
    with pytest.raises(ValueError, match="prefetch_perception"):
        PipelineConfig(prefetch_perception=1)
    with pytest.raises(ValueError, match="prefetch_max_age_s"):
        PipelineConfig(prefetch_max_age_s=0)
    assert StreamingPlanner  # the planner the pipeline tests share
