"""Both physical finger models participate in sweeps and dual-arm trajectory checks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.hand_restore import HandSweepCheck, RestoreError
from aimdk_grasp.hand_restore_geometry import DualHandCollision, sphere_box_clearance
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.robot_model import config_directory, planning_model
from test_hand import PRESETS


@pytest.mark.parametrize("margin", [0.0, 0.02, 0.08])
def test_sphere_broad_phase_matches_exact_acceptance_in_mixed_batches(margin):
    check = object.__new__(HandSweepCheck)
    check.model = SimpleNamespace(links=["R_palm", "torso"], link_slices={"R_palm": slice(0, 3), "torso": slice(3, 8)})
    check.ignored = set()
    rng = np.random.default_rng(71)
    for offset in (0.02, 0.15, 2.0):
        spheres = rng.normal(0, 0.02, (17, 8, 4))
        spheres[..., 3] = 0.025
        spheres[:, 3:, 0] += offset
        spheres[0, 3:, 0] += 10.0  # Mix a trivially separated frame with closer ones.
        a, b = spheres[:, :3], spheres[:, 3:]
        exact = float(np.min(np.linalg.norm(a[:, :, None, :3] - b[:, None, :, :3], axis=-1) - 0.05))
        if exact < margin - 1e-6:
            with pytest.raises(RestoreError, match="clearance"):
                check.check_spheres(spheres, body_margin=margin)
        else:
            lower = check.check_spheres(spheres, body_margin=margin)
            assert margin - 1e-6 <= lower <= exact + 1e-10


@pytest.fixture(scope="module")
def geometry():
    checks = {
        side: HandSweepCheck(HandMapping(planning_model(config_directory(side)), PRESETS, side=side))
        for side in ("left", "right")
    }
    return DualHandCollision(checks)


def feedback(geometry):
    body = dict(zip(LEFT_ARM + RIGHT_ARM + HEAD, [0.4, 0, 0, -1.2, 0, 0, 0] * 2 + [0, 0], strict=True))
    return Feedback(
        1.0,
        body,
        geometry.checks["left"].mapping.preset("open"),
        right_hand=geometry.checks["right"].mapping.preset("open"),
    )


@pytest.mark.parametrize("finger_radius", [0.025, 0.1])
def test_both_real_models_check_a_short_arm_path_with_preserved_fingers(geometry, finger_radius):
    observed = feedback(geometry)
    q = observed.vector(LEFT_ARM + RIGHT_ARM)
    end = q.copy()
    end[[0, 7]] -= 0.15
    if finger_radius == 0.1:
        # The larger preservation envelope cannot be certified at this wrist/thumb gap.
        with pytest.raises(RestoreError, match="insufficient hand clearance"):
            geometry.check_arm_path(observed, LEFT_ARM + RIGHT_ARM, [q, end], finger_radius=finger_radius)
    else:
        geometry.check_arm_path(observed, LEFT_ARM + RIGHT_ARM, [q, end], finger_radius=finger_radius)


def test_rigid_pair_reuse_agrees_with_checking_all_real_fk_frames(geometry, monkeypatch):
    original = HandSweepCheck.check_spheres
    compared = []

    def compare(check, spheres, **kwargs):
        outcomes = []
        for options in (kwargs, {**kwargs, "pair_constant": None}):
            try:
                value = original(check, spheres, **options)
                outcomes.append((None, value))
            except RestoreError as exc:
                outcomes.append((exc, None))
        assert (outcomes[0][0] is None) == (outcomes[1][0] is None)
        compared.append(True)
        if outcomes[0][0] is not None:
            raise outcomes[0][0]
        return outcomes[0][1]

    monkeypatch.setattr(HandSweepCheck, "check_spheres", compare)
    observed = feedback(geometry)
    q = observed.vector(LEFT_ARM + RIGHT_ARM)
    end = q.copy()
    end[[0, 7]] -= 0.15
    end[[6, 13]] += 0.05  # A varying wrist must not be classified as a shared rigid ancestor.
    geometry.check_arm_path(observed, LEFT_ARM + RIGHT_ARM, [q, end], finger_radius=0.025)
    assert len(compared) >= 2


def test_right_sweep_uses_right_native_mapping_and_both_models(geometry):
    observed = feedback(geometry)
    right = observed.right_hand.copy()
    right[4] += 0.1
    assert geometry.sweep("right").check(observed, [right, observed.right_hand]) > 0
    invalid = dict(observed.positions, right_wrist_roll_joint=3.0)
    with pytest.raises(RestoreError, match="limits"):
        geometry.sweep("right").check(replace(observed, positions=invalid), [right])


def test_inter_hand_collision_is_checked_even_if_each_individual_model_passes(geometry, monkeypatch):
    # One sphere per side isolates the gap that a single-hand model cannot see.
    sphere = np.array([[[0.3, 0, 0.2, 0.02]]])
    checks = {
        side: SimpleNamespace(
            place=lambda *_a, **_k: sphere,
            check_spheres=lambda *_a, **_k: 1.0,
            model=SimpleNamespace(links=["R_palm"], sphere_links=["R_palm"]),
        )
        for side in ("left", "right")
    }
    observed = feedback(geometry)
    monkeypatch.setattr(geometry, "checks", checks)
    monkeypatch.setattr(geometry, "indices", {"left": [0], "right": [0]})
    monkeypatch.setattr(geometry, "leg_hulls", {})
    with pytest.raises(RestoreError, match="left/right hands"):
        geometry.check(observed, {side: observed.hand(side) for side in checks})


def test_path_checks_the_middle_not_just_the_endpoints(geometry, monkeypatch):
    observed = feedback(geometry)
    q = observed.vector(LEFT_ARM + RIGHT_ARM)
    end = q.copy()
    end[0] -= 0.15
    seen = []

    def collision(_feedback, _hands, *, left_arm, right_arm, arm_delta, sample_offset=0, **_):
        seen.extend(left_arm[:, 0])
        assert np.max(arm_delta) > 0
        if np.any((left_arm[:, 0] > 0.30) & (left_arm[:, 0] < 0.35)):
            raise RestoreError("HAND_SWEEP_COLLISION", "intermediate hand collision")

    monkeypatch.setattr(geometry, "check", collision)
    with pytest.raises(RestoreError, match="intermediate"):
        geometry.check_arm_path(observed, LEFT_ARM + RIGHT_ARM, [q, end])
    assert any(0.30 < value < 0.35 for value in seen)


def test_path_cancellation_stops_geometry_work_before_dispatch(geometry):
    observed = feedback(geometry)

    def stop():
        raise RestoreError("POSE_CANCELLED", "cancelled during collision checking")

    with pytest.raises(RestoreError, match="cancelled"):
        geometry.check_arm_path(observed, LEFT_ARM + RIGHT_ARM, [observed.vector(LEFT_ARM + RIGHT_ARM)], guard=stop)


def test_lower_body_boxes_are_solid_and_rotation_is_respected():
    identity = [0, 0, 0, 1, 0, 0, 0]
    spheres = np.array([[0, 0, 0, 0.1], [1.2, 0, 0, 0.1]])
    np.testing.assert_allclose(sphere_box_clearance(spheres, [2, 1, 1], identity), [-0.6, 0.1])
    rotated = [0, 0, 0, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    assert sphere_box_clearance(np.array([[0, 0.9, 0, 0.1]]), [2, 1, 1], rotated)[0] < 0


def test_common_shoulder_motion_cancels_but_wrist_motion_does_not(geometry):
    bounds = geometry.bounds["left"]
    observed = feedback(geometry)
    base = bounds.angles(observed, np.zeros(14))
    shoulder = np.zeros(14)
    shoulder[0] = 0.1
    wrist = np.zeros(14)
    wrist[5] = 0.1
    pair = ("left_wrist_yaw_link", "R_thumb_abad")
    original = bounds.pair(*pair, base)
    assert original > 0
    assert bounds.pair(*pair, bounds.angles(observed, shoulder)) == pytest.approx(original)
    assert bounds.pair(*pair, bounds.angles(observed, wrist)) > original


@pytest.mark.parametrize("side", ["left", "right"])
def test_finger_drift_expands_collision_bounds_without_expanding_head(geometry, monkeypatch, side):
    import aimdk_grasp.hand_restore_geometry as module

    bounds = geometry.bounds[side]
    observed = feedback(geometry)
    expanded = bounds.angles(observed, np.zeros(14))
    monkeypatch.setattr(module, "PATH_SHAPE_HOLD_RAD", 0.005)
    previous = bounds.angles(observed, np.zeros(14))
    assert np.all(expanded >= previous)
    assert np.any(expanded > previous)
    for index, source in enumerate(bounds.model.fk_source):
        if bounds.model.fk_inputs[source] in HEAD:
            assert expanded[index] == previous[index]
