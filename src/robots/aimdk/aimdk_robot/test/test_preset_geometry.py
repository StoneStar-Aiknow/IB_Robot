"""Behavior tests for a numerical guard, using clearly synthetic geometry."""

import copy
import json

import numpy as np
import pytest
from aimdk_robot.preset_geometry import NamedHoldGeometryGuard, canonical_digest, load_guard


def _record():
    models = []
    for side, source, sign in (("left", 0, 1), ("right", 7, -1)):
        bounds = np.zeros((2, 34))
        bounds[:, source] = 2.0
        models.append(
            {
                "side": side,
                "joints": [
                    {
                        "name": f"{side}_synthetic",
                        "parent": 0,
                        "origin": np.eye(4).tolist(),
                        "axis": [0, 0, 1],
                        "source": source,
                        "coefficients": [0, 1],
                        "limits": [-4, 4],
                    }
                ],
                "spheres": [[sign, 0, 0, 0.05], [sign, -0.5, 0, 0.05]],
                "sphere_frames": [1, 1],
                "head_check_indices": [0, 1],
                "hand_indices": [0],
                "self_pairs": [[0, 1]],
                "motion_bounds_m_per_rad": bounds.tolist(),
            }
        )
    value = {
        "schema_version": 1,
        "name": "overhead_heart",
        "validated": True,
        "required_head_clearance_m": 0.20,
        "native_limits_rad": [[-4, 4]] * 34,
        "native_speed_radps": [10] * 34,
        "native_uncertainty_rad": [0.001] * 34,
        "head_envelope_spheres": [[4, 0, 0, 0.1]],
        "head_envelope_covers_allowed_motion": True,
        "complete_articulated_hands": True,
        "continuous_motion_bounds_verified": True,
        "hand_clearance_m": 0.01,
        "models": models,
        "sources": ["synthetic unit-test geometry; not a robot certificate"],
    }
    value["sha256"] = canonical_digest(value)
    return value


def _guard(**changes):
    value = _record()
    value.update(changes)
    value["sha256"] = canonical_digest(value)
    return NamedHoldGeometryGuard(value)


def _validate(guard, positions=None, times=None):
    return guard.validate_path(
        name="overhead_heart",
        positions=[[0.0] * 34] if positions is None else positions,
        times=[0.0] if times is None else times,
    )


def test_synthetic_clear_pose_passes_without_claiming_a_robot_certificate():
    ok, reason = _validate(_guard())
    assert ok, reason


@pytest.mark.parametrize(
    "field",
    [
        "validated",
        "head_envelope_covers_allowed_motion",
        "complete_articulated_hands",
        "continuous_motion_bounds_verified",
    ],
)
def test_missing_certification_refuses_execution(field):
    ok, _ = _validate(_guard(**{field: False}))
    assert not ok


def test_missing_file_and_invalid_digest_refuse_execution(tmp_path):
    assert not _validate(load_guard(tmp_path / "absent.json"))[0]
    value = _record()
    value["native_speed_radps"][0] = 9
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value))
    assert not _validate(load_guard(path))[0]


def test_head_collision_is_refused_even_with_safe_native_angles():
    guard = _guard(head_envelope_spheres=[[1, 0, 0, 0.1]])
    assert not _validate(guard)[0]


def test_clear_endpoints_do_not_hide_an_unsafe_middle_of_a_path():
    guard = _guard(head_envelope_spheres=[[1, 0, 0, 0.1]])
    start, finish = np.zeros(34), np.zeros(34)
    start[0], finish[0] = -1.0, 1.0
    assert _validate(guard, [start], [0])[0]
    assert _validate(guard, [finish], [0])[0]
    assert not _validate(guard, [start, finish], [0, 1])[0]


def test_native_limit_speed_and_shape_failures_are_refused():
    guard = _guard()
    invalid = np.zeros(34)
    invalid[0] = 4.1
    assert not _validate(guard, [invalid], [0])[0]
    invalid[0] = 1.0
    assert not _validate(guard, [np.zeros(34), invalid], [0, 0.01])[0]
    assert not _validate(guard, [[0] * 33], [0])[0]
    assert not _validate(guard, [np.zeros(34), np.zeros(34)], [0, 0])[0]


def test_nonrigid_compiled_transform_is_refused():
    value = copy.deepcopy(_record())
    value["models"][0]["joints"][0]["origin"][0][0] = 2.0
    value["sha256"] = canonical_digest(value)
    assert not _validate(NamedHoldGeometryGuard(value))[0]


def test_easing_peak_speed_is_checked_instead_of_average_speed_only():
    guard = _guard(native_speed_radps=[0.6] * 34)
    start, finish = np.zeros(34), np.zeros(34)
    finish[0] = 0.5
    assert not _validate(guard, [start, finish], [0, 1])[0]


def test_nonfinite_clearance_cannot_bypass_head_checks():
    record = _record()
    record["required_head_clearance_m"] = float("nan")
    assert not _validate(NamedHoldGeometryGuard(record))[0]
