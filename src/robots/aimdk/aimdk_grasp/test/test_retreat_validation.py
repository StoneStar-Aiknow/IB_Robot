"""Read-only X2 retreat validation on fresh fake RGB-D and real seven-joint paths."""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp import place_request, proposer, retreat_validation, scene_estimation
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.revalidation import RevalidationConfig

from manipulation_execution.grasp_pipeline.contracts import ArmSegment, Move, SegmentKind


@pytest.fixture
def rig(monkeypatch):
    clock = [100.0]
    hand = np.zeros(10)
    positions = dict.fromkeys((*LEFT_ARM, *RIGHT_ARM, *HEAD), 0.0)
    checked, captures, scenes, attachments = [], [], [], []
    status = {"obstacle": False, "age_during_check": 0.0, "drift_after_check": False}
    original = Feedback(clock[0], positions, hand.copy())

    def live(fresh=False):
        fb = replace(original, received_at=clock[0]) if fresh else original
        if fresh and status["drift_after_check"] and checked:
            fb = replace(fb, positions={**fb.positions, LEFT_ARM[0]: 0.1})
        return SimpleNamespace(
            feedback=fb, observation=SimpleNamespace(received_at=clock[0] if fresh else 100.0, image_feedback=fb)
        )

    def capture(**kwargs):
        captures.append("new RGB-D")
        return SimpleNamespace(robot_state=SimpleNamespace(live=live()), stamp_ns=123)

    def valid(path):
        checked.append(path.copy())
        clock[0] += status["age_during_check"]
        answer = np.ones(len(path), dtype=bool)
        if status["obstacle"]:
            answer[len(path) // 2] = False
        return answer

    @contextmanager
    def session(feedback, scene, *, attached):
        attachments.append(attached)
        yield SimpleNamespace(valid_positions=valid)

    def complete(prepared, mask, *, lazy_validation):
        scenes.append((mask, lazy_validation))
        return "new world"

    monkeypatch.setattr(proposer, "prepare_capture_scene", lambda *a: "new cloud")
    monkeypatch.setattr(scene_estimation, "complete_scene", complete)
    monkeypatch.setattr(place_request, "_hand_cover", lambda *a: np.array([[0, 0, 0, 0.02]]))
    monkeypatch.setattr(retreat_validation, "require_executable", lambda *a: None)
    head = SimpleNamespace(
        camera=SimpleNamespace(capture=capture, capture_state=lambda **kw: live(True), clock=lambda: clock[0]),
        gpu=threading.RLock(),
        spheres=object(),
        service=SimpleNamespace(session=session),
    )
    robot = SimpleNamespace(
        side="left",
        hand=SimpleNamespace(preset=lambda name: hand),
        planning=SimpleNamespace(revalidation=RevalidationConfig()),
        perception=SimpleNamespace(scene=object()),
        limits={},
        gravity=object(),
    )
    segment = ArmSegment(
        "retreat", SegmentKind.TRAJECTORY, LEFT_ARM, np.array([np.zeros(7), np.ones(7) * 0.1]), np.array([0.0, 1.0])
    )
    return SimpleNamespace(
        head=head,
        robot=robot,
        steps=(Move(segment),),
        hand=hand,
        checked=checked,
        captures=captures,
        scenes=scenes,
        status=status,
        attachments=attachments,
    )


def validate(rig):
    return retreat_validation.validate_retreat(
        rig.head, rig.robot, rig.steps, rig.hand, deadline=time.monotonic() + 10, cancelled=lambda: False
    )


def test_fresh_complete_world_and_every_intermediate_sample_are_checked_without_motion(rig):
    report = validate(rig)
    assert rig.captures == ["new RGB-D"]
    assert rig.scenes == [(None, False)]  # No stale target mask / lazy fine scene.
    assert len(rig.checked) == 1 and len(rig.checked[0]) >= 11
    assert report["checked_samples"] == len(rig.checked[0])
    np.testing.assert_allclose(rig.attachments[0], [[0, 0, 0, 0.02]])


@pytest.mark.parametrize("problem", ["obstacle", "scene_age", "robot_moved", "hand", "start", "effect"])
def test_retreat_validation_refuses_unsafe_or_expired_evidence(rig, problem):
    if problem == "obstacle":
        rig.status["obstacle"] = True
    elif problem == "scene_age":
        rig.status["age_during_check"] = 0.6
    elif problem == "robot_moved":
        rig.status["drift_after_check"] = True
    elif problem == "hand":
        rig.hand[4] = 1.0
    elif problem == "start":
        segment = rig.steps[0].segment
        rig.steps = (
            Move(ArmSegment(segment.name, segment.kind, segment.joint_names, segment.positions + 0.1, segment.times)),
        )
    else:
        from manipulation_execution.grasp_pipeline.contracts import Effect, EffectorAction

        rig.steps = (Effect(EffectorAction.RELEASE),)
    with pytest.raises((ValueError, RuntimeError)):
        validate(rig)


@pytest.fixture
def carrying(rig, monkeypatch):
    from manipulation_execution.grasp_pipeline.contracts import Effect, EffectorAction

    rig.steps = (*rig.steps, Effect(EffectorAction.RELEASE))
    rig.held = SimpleNamespace(arm_side="left", joint_names=LEFT_ARM, joint_positions=np.zeros(7))
    rig.contact = np.ones(7) * 0.1
    rig.support_checks = []
    monkeypatch.setattr(
        retreat_validation, "_carried_scene", lambda prepared, *a: (prepared, np.array([[0, 0, 0, 0.05]]))
    )
    monkeypatch.setattr(place_request, "opening_cover", lambda *a: np.array([[0, 0, 0, 0.08]]))
    monkeypatch.setattr(retreat_validation, "_require_support", lambda *a: rig.support_checks.append(True))
    return rig


def validate_carrying(rig):
    return retreat_validation.validate_retreat(
        rig.head,
        rig.robot,
        rig.steps,
        rig.hand,
        held=rig.held,
        support=object(),
        contact=rig.contact,
        deadline=time.monotonic() + 10,
        cancelled=lambda: False,
    )


def test_carrying_validation_checks_object_attachment_support_and_opening_without_motion(carrying):
    report = validate_carrying(carrying)
    assert carrying.support_checks == [True]
    assert carrying.captures == ["new RGB-D"]
    assert len(carrying.checked) == 2
    np.testing.assert_allclose(carrying.attachments[0], [[0, 0, 0, 0.05]])
    np.testing.assert_allclose(carrying.attachments[1], [[0, 0, 0, 0.08]])
    assert report["arm_side"] == "left"


@pytest.mark.parametrize("problem", ["wrong_contact", "wrong_side", "moved_arm", "support", "opening"])
def test_carrying_validation_refuses_invalid_source_release(carrying, monkeypatch, problem):
    if problem == "wrong_contact":
        carrying.contact += 0.1
    elif problem == "wrong_side":
        carrying.held.arm_side = "right"
    elif problem == "moved_arm":
        carrying.held.joint_positions += 0.1
    elif problem == "support":

        def reject(*args):
            raise ValueError("support moved")

        monkeypatch.setattr(retreat_validation, "_require_support", reject)
    else:

        @contextmanager
        def session(*a, attached):
            yield SimpleNamespace(valid_positions=lambda path: np.full(len(path), attached[0, 3] < 0.08))

        carrying.head.service.session = session
    with pytest.raises(ValueError):
        validate_carrying(carrying)


@pytest.mark.parametrize("problem", ["none", "height", "normal", "footprint"])
def test_release_requires_original_support_under_original_object(problem):
    from aimdk_grasp.scene_estimation import SupportPlane

    points = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]])
    reference = SupportPlane(np.zeros(3), np.array([0, 0, 1.0]), points)
    current = reference
    if problem == "none":
        reference = None
    elif problem == "height":
        current = replace(current, point=np.array([0, 0, 0.1]))
    elif problem == "normal":
        current = replace(current, normal=np.array([0, 1.0, 0]))
    else:
        current = replace(current, inliers=points + [3, 0, 0])
    model = SimpleNamespace(
        spheres=lambda *a: None, last_link_frames=np.eye(4)[None, None, :3], link_indices={"R_palm": 0}
    )
    held = SimpleNamespace(points_in_palm=np.array([[0, 0, 0.05], [0.01, 0, 0.06]]).ravel())
    with pytest.raises(ValueError):
        retreat_validation._require_support(SimpleNamespace(support=current), reference, model, None, held, np.zeros(7))


def test_carried_scene_filters_only_measured_hand_object_envelope_in_camera_frame(monkeypatch):
    @dataclass
    class Prepared:
        points: np.ndarray
        keep: np.ndarray
        planning_all: np.ndarray
        spheres: np.ndarray

    prepared = Prepared(np.array([[-1.0, 0, 0], [0.0, 0, 0]]), np.ones(2, bool), np.ones(2, bool), np.empty((0, 4)))
    observation = SimpleNamespace(base_from_camera=np.eye(4))
    observation.base_from_camera[0, 3] = 1.0
    model = SimpleNamespace(last_link_frames=np.eye(4)[None, None, :3], link_indices={"R_palm": 0})
    cover = np.array([[0, 0, 0, 0.05]])
    monkeypatch.setattr(place_request, "placement_cover", lambda *a: cover)
    filtered, attached = retreat_validation._carried_scene(
        prepared,
        observation,
        model,
        None,
        object(),
        SimpleNamespace(validation_self_filter_margin_m=0.01, self_filter_margin_m=0.03),
    )
    assert filtered.keep.tolist() == [False, True] and filtered.planning_all.tolist() == [False, True]
    np.testing.assert_array_equal(attached, cover)
    np.testing.assert_array_equal(filtered.spheres, cover)
