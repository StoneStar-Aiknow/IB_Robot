"""The X2 proposer: detection to candidates and scene, with the support fallback, against fake models."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.frames import ImageTransform
from aimdk_grasp.joints import BODY_JOINTS
from aimdk_grasp.observation import AdmittedState, LiveState, RegisteredObservation
from aimdk_grasp.proposer import NO_EXECUTABLE_CANDIDATE, PerceptionSettings, X2Proposer, detection_prompt
from aimdk_grasp.scene_estimation import SceneEstimate
from aimdk_grasp.types import CameraIntrinsics, RGBDFrame

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, Observation
from manipulation_execution.grasp_pipeline.perception import Detection

SIZE, STAMP, FRAME = 50, 123_000_000, "optical"
ON_TABLE = np.s_[20:28, 20:28]  # 0.1 m above the table
BELOW = np.s_[5:13, 35:43]  # through a hole in the table, 0.5 m below it


@pytest.fixture(autouse=True)
def unbounded_left_arm_reach(monkeypatch):
    """The synthetic camera is far from the real X2 shoulder; the reach crop keeps everything."""
    from aimdk_grasp import scene_estimation

    monkeypatch.setattr(scene_estimation, "left_arm_reach", lambda: (np.zeros(3), np.inf))


def box(region):
    rows, cols = region
    return np.array([cols.start, rows.start, cols.stop, rows.stop], dtype=np.float32)


def observation():
    depth = np.ones((SIZE, SIZE), dtype=np.float32)
    depth[ON_TABLE] = 0.9
    depth[BELOW] = 1.5
    matrix = np.diag([1.0, -1.0, -1.0, 1.0])
    matrix[:3, 3] = [1.0, 2.0, 2.0]
    frame = RGBDFrame(np.zeros((SIZE, SIZE, 3), np.uint8), depth, CameraIntrinsics(50, 50, 25, 25), FRAME)
    feedback = Feedback(STAMP * 1e-9, dict.fromkeys(BODY_JOINTS, 0.0), np.zeros(10))
    registered = RegisteredObservation(
        frame, ImageTransform(STAMP, FRAME, "torso_link", matrix), STAMP * 1e-9, feedback, "calibration"
    )
    k = np.array([[50.0, 0, 25], [0, 50.0, 25], [0, 0, 1]])
    return Observation(
        frame.rgb,
        depth,
        k,
        FRAME,
        STAMP,
        "torso_link",
        matrix,
        robot_state=AdmittedState(LiveState(registered, feedback), STAMP * 1e-9, 0.0),
    )


class FakeModels:
    """Stands in for PerceptionServices: fixed detections, box masks, three grasps per cloud."""

    def __init__(self, detections):
        self.detections, self.segmented, self.clouds, self.calls = detections, [], [], []

    def detect(self, rgb, frame_id, stamp_ns, prompt, *, box_threshold, text_threshold, cancelled):
        assert (frame_id, stamp_ns, prompt) == (FRAME, STAMP, "red cup .")
        self.calls.append(dict(service="detect"))
        return list(self.detections)

    def masks(self, rgb, frame_id, stamp_ns, detections, *, cancelled):
        self.calls.append(dict(service="segment"))
        self.segmented.extend(detections)
        masks = []
        for detection in detections:
            x0, y0, x1, y1 = (int(value) for value in detection.box_xyxy)
            mask = np.zeros((SIZE, SIZE), bool)
            mask[y0:y1, x0:x1] = True
            masks.append(mask)
        return masks

    def generate_grasps(self, points, frame_id, stamp_ns, *, max_grasps, min_confidence, cancelled):
        assert (max_grasps, min_confidence) == (400, -1.0)
        self.calls.append(dict(service="grasps"))
        self.clouds.append(points)
        poses = np.stack([np.eye(4)] * 3)
        poses[:, :3, 3] = points.mean(axis=0)
        return poses, np.array([0.9, 0.8, 0.7])


class NoSpheres:
    def spheres(self, feedback, positions):
        return np.empty((len(positions), 0, 4)), np.ones(len(positions), bool)


def propose(models, *, cancelled=lambda: False, preload=None, overlap=True):
    proposer = X2Proposer(models, NoSpheres(), settings=PerceptionSettings(scene_overlap=overlap), preload=preload)
    return proposer.propose(observation(), "  red   cup ", deadline=time.monotonic() + 30.0, cancelled=cancelled)


@pytest.mark.parametrize("overlap", [True, False])
def test_the_target_comes_with_its_candidates_and_the_scene_the_planner_preloads(overlap):
    models = FakeModels([Detection("cup", 0.8, box(ON_TABLE))])
    loaded = []
    proposals = propose(models, preload=lambda start, scene: loaded.append((start, scene)), overlap=overlap)
    assert proposals.target_mask[ON_TABLE].all() and proposals.target_mask.sum() == 64
    assert len(proposals.target_points) == 64 and np.allclose(proposals.target_points[:, 2], 0.9)
    assert proposals.grasps.shape == (3, 4, 4) and list(proposals.confidence) == [0.9, 0.8, 0.7]
    assert isinstance(proposals.scene, SceneEstimate) and [scene for _, scene in loaded] == [proposals.scene]
    assert loaded[0][0].received_at == pytest.approx(STAMP * 1e-9)  # the admitted current feedback
    assert proposals.scene.stamp_ns == STAMP and proposals.scene.support.target_heights_m["median"] == pytest.approx(
        0.1
    )
    report = proposals.report
    assert report["prompt"] == "red cup ." and report["candidate_count"] == 3
    assert {"detect", "segment", "clean", "cloud", "grasp", "scene", "scene_load"} <= set(
        report["perception_timings_s"]
    )
    assert [call["service"] for call in report["model_rpc"]] == ["detect", "segment", "grasps"]


def test_a_detection_with_no_support_beneath_gives_way_to_the_next_one():
    below = Detection("cup", 0.9, box(BELOW))
    duplicate = Detection("cup", 0.85, box(BELOW) + 1)  # the same object again
    on_table = Detection("cup", 0.6, box(ON_TABLE))
    models = FakeModels([below, duplicate, on_table])
    proposals = propose(models)
    assert models.segmented == [below, on_table]
    assert proposals.report["detection"]["score"] == pytest.approx(0.6)
    rejected = proposals.report["perception_timings_s"]["rejected_detections"]
    assert [row["score"] for row in rejected] == [pytest.approx(0.9)]


def test_only_unsupported_detections_end_the_pick():
    with pytest.raises(GraspFailure) as raised:
        propose(FakeModels([Detection("cup", 0.9, box(BELOW))]))
    assert raised.value.code == NO_EXECUTABLE_CANDIDATE and "NoSupportError" in str(raised.value)


def test_an_unseen_target_is_reported_as_not_visible_not_as_a_planning_failure():
    """The caller must tell "not seen" (look elsewhere, ask) from "seen but not graspable"."""
    with pytest.raises(GraspFailure, match="No target detected for 'red cup .'") as raised:
        propose(FakeModels([]))
    assert raised.value.code == "TARGET_NOT_VISIBLE" and not raised.value.retryable


def test_too_little_depth_ends_the_pick():
    with pytest.raises(GraspFailure, match="insufficient valid depth") as raised:
        propose(FakeModels([Detection("cup", 0.9, np.array([0, 0, 3, 3], np.float32))]))
    assert raised.value.code == NO_EXECUTABLE_CANDIDATE and not raised.value.retryable


def test_a_failed_preload_is_reported_and_left_to_the_planner():
    def fail(start, scene):
        raise RuntimeError("cuda busy")

    proposals = propose(FakeModels([Detection("cup", 0.8, box(ON_TABLE))]), preload=fail)
    assert proposals.report["scene_preload_error"] == "RuntimeError: cuda busy"


def test_a_cancel_stops_between_stages():
    models = FakeModels([Detection("cup", 0.8, box(ON_TABLE))])
    with pytest.raises(GraspFailure) as raised:
        propose(models, cancelled=lambda: bool(models.calls))
    assert raised.value.code == "CANCELED" and not models.clouds


def test_the_prompt_is_the_phrase_grounding_dino_expects():
    assert detection_prompt(" a  bottle ") == "a bottle ."
    assert detection_prompt("bottle .") == "bottle ."


@pytest.mark.parametrize(
    ("query", "prompt"),
    [(" 苹果。 ", "an apple ."), ("红色苹果", "a red apple ."), ("瓶子", "a bottle ."), ("杯子", "a cup .")],
)
def test_chinese_target_nouns_become_english_detector_prompts(query, prompt):
    assert detection_prompt(query) == prompt


@pytest.mark.parametrize("query", ["", " 。 ", "左边的苹果", "苹果和杯子", "抓取苹果"])
def test_ambiguous_or_untranslated_targets_fail_before_any_perception_call(query):
    models = FakeModels([])
    proposer = X2Proposer(models, NoSpheres())
    with pytest.raises(GraspFailure) as raised:
        proposer.propose(observation(), query, deadline=time.monotonic() + 30.0, cancelled=lambda: False)
    assert raised.value.code == "INVALID_TARGET"
    assert not raised.value.retryable and not models.calls


@pytest.mark.parametrize("overlap", [True, False])
def test_a_scene_finished_by_the_grasp_head_keeps_the_support_fallback(overlap):
    from aimdk_grasp.scene_estimation import NoSupportError

    below, on_table = Detection("cup", 0.9, box(BELOW)), Detection("cup", 0.6, box(ON_TABLE))
    models = FakeModels([below, on_table])
    asked = []

    def remote_scene(seen, mask, timeout_s, *, cancelled):
        asked.append((seen, mask.copy(), timeout_s, cancelled))
        if mask[BELOW].all():
            raise NoSupportError("nothing under it")
        return SimpleNamespace(capture_id="c1", report={"scene_s": 0.08})

    plain = replace(observation(), robot_state="the planning host's capture")  # no X2 state needed here
    proposer = X2Proposer(models, None, settings=PerceptionSettings(scene_overlap=overlap), remote_scene=remote_scene)
    proposals = proposer.propose(plain, "red cup", deadline=time.monotonic() + 30.0, cancelled=lambda: False)
    assert [seen for seen, _, _, _ in asked] == [plain, plain]
    assert asked[1][1][ON_TABLE].all() and asked[1][1].sum() == 64 and 0 < asked[1][2] <= 30.0
    assert all(not cancelled() for *_, cancelled in asked)
    assert proposals.scene.capture_id == "c1" and proposals.report["scene_host"] == {"scene_s": 0.08}
    assert len(proposals.target_points) == 64 and proposals.grasps.shape == (3, 4, 4)
    with pytest.raises(ValueError, match="sphere model"):
        X2Proposer(models, None)


def test_a_cancel_releases_the_overlapped_remote_scene_worker():
    models = FakeModels([Detection("cup", 0.8, box(ON_TABLE))])
    cancel_at = time.monotonic() + 0.1
    exited = []

    def cancelled():
        return time.monotonic() >= cancel_at

    def remote_scene(seen, mask, timeout_s, *, cancelled):
        while not cancelled():
            time.sleep(0.002)
        exited.append(True)
        raise GraspFailure("CANCELED", "goal cancelled")

    proposer = X2Proposer(models, None, remote_scene=remote_scene)
    began = time.monotonic()
    with pytest.raises(GraspFailure) as raised:
        proposer.propose(observation(), "red cup", deadline=began + 30.0, cancelled=cancelled)

    assert raised.value.code == "CANCELED" and exited
    assert time.monotonic() - began < 1.0


# -- the upright image the models see ---------------------------------------------------------------------------------

#: torso_link <- rgbd_head_front measured on the X2 (2026-10-26): image right is the robot's left, image down is up.
X2_HEAD = np.array([[0.01, 0.64, 0.76], [1.0, -0.01, -0.01], [0.0, 0.76, -0.64]])


def camera(roll_deg, *, down_deg=40.0):
    """A camera looking forward (+x) and down_deg below the horizon, rolled about its optical axis."""
    from scipy.spatial.transform import Rotation

    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # columns: right, down, forward
    tilt = Rotation.from_euler("y", down_deg, degrees=True).as_matrix()
    roll = Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = tilt @ level @ roll
    return pose


def test_the_x2_head_camera_is_turned_half_way_and_a_camera_looking_straight_down_not_at_all():
    from aimdk_grasp.proposer import upright_turns

    pose = np.eye(4)
    pose[:3, :3] = X2_HEAD
    assert upright_turns(pose) == 2
    assert upright_turns(observation().base_from_camera) == 0  # the fixture camera looks straight down
    assert upright_turns(camera(180.0, down_deg=80.0)) == 0  # within ~15 degrees of vertical: no up to find


@pytest.mark.parametrize("roll_deg", [0, 30, 90, 180, 270, -100])
def test_whatever_the_roll_the_turned_image_shows_up_at_the_top(roll_deg):
    from aimdk_grasp.proposer import ImageTurn, upright_turns

    pose = camera(roll_deg)
    k = np.array([[300.0, 0, 320], [0, 300.0, 120], [0, 0, 1]])
    height, width = 240, 640
    rows, cols = np.mgrid[0:height, 0:width]
    rays = np.linalg.inv(k) @ np.stack([cols.ravel(), rows.ravel(), np.ones(rows.size)])
    heights = (pose[:3, :3] @ rays)[2].reshape(height, width)  # how far up each pixel looks, in the base
    turned = ImageTurn(upright_turns(pose), height, width).image(heights)
    third = turned.shape[0] // 3
    assert turned[:third].mean() > turned[-third:].mean() + 0.2


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_box_found_in_the_turned_image_is_the_same_pixels_in_the_camera_image(turns):
    from aimdk_grasp.proposer import ImageTurn

    height, width = 30, 50
    marked = np.zeros((height, width), bool)
    marked[4:11, 7:20] = True
    turn = ImageTurn(turns, height, width)
    rows, cols = np.nonzero(turn.image(marked))
    found = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], np.float32)
    assert np.array_equal(turn.box_back(found), box(np.s_[4:11, 7:20]))
    assert np.array_equal(turn.box_turned(box(np.s_[4:11, 7:20])), found)
    assert np.array_equal(turn.mask_back(turn.image(marked)), marked)


class MarkerModels(FakeModels):
    """A detector that boxes the bright pixels of the image it is given, with or without their mask."""

    def __init__(self, *, with_mask):
        super().__init__([])
        self.with_mask, self.seen = with_mask, []

    def detect(self, rgb, frame_id, stamp_ns, prompt, *, box_threshold, text_threshold, cancelled):
        self.calls.append(dict(service="detect"))
        self.seen.append(rgb.copy())
        bright = rgb[..., 0] > 0
        rows, cols = np.nonzero(bright)
        found = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], np.float32)
        return [Detection("cup", 0.8, found, bright if self.with_mask else None)]

    def masks(self, rgb, frame_id, stamp_ns, detections, *, cancelled):
        self.seen.append(rgb.copy())
        return super().masks(rgb, frame_id, stamp_ns, detections, cancelled=cancelled)


@pytest.mark.parametrize("with_mask", [True, False])
def test_an_upside_down_camera_is_detected_upright_and_its_target_found_where_it_is(with_mask):
    upside_down = observation()
    rgb = np.zeros((SIZE, SIZE, 3), np.uint8)
    rgb[ON_TABLE] = 255
    rgb[20, 20] = (255, 0, 0)  # the target's top-left corner, as the camera sees it
    pose = np.array(upside_down.base_from_camera)
    pose[:3, :3] = X2_HEAD
    upside_down = replace(upside_down, rgb=rgb, base_from_camera=pose, robot_state="the head's capture")
    models = MarkerModels(with_mask=with_mask)
    proposer = X2Proposer(
        models,
        None,
        remote_scene=lambda seen, mask, timeout_s, *, cancelled: SimpleNamespace(capture_id="c1", report={}),
    )
    proposals = proposer.propose(upside_down, "red cup", deadline=time.monotonic() + 30.0, cancelled=lambda: False)
    assert all(np.array_equal(seen, np.rot90(rgb, 2)) for seen in models.seen)  # the models saw it upright
    assert len(models.seen) == (1 if with_mask else 2)
    assert proposals.target_mask[ON_TABLE].all() and proposals.target_mask.sum() == 64
    assert proposals.report["detection"]["box_xyxy"] == [20.0, 20.0, 28.0, 28.0]
    assert proposals.report["detection_turns"] == 2 and len(proposals.target_points) == 64
