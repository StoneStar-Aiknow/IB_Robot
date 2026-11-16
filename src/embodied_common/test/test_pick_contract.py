import math

import pytest

from embodied_common.pick_contract import (
    FAILURE_CLASSES,
    GRASP_SPEC_INVALID,
    GRASP_SPEC_UNSUPPORTED,
    GraspCapability,
    GraspSpec,
    check_grasp_spec,
)


def test_no_spec_constrains_nothing():
    assert GraspSpec().requested() == ()
    assert GraspSpec.from_value(None) == GraspSpec()
    assert GraspCapability().unsupported(GraspSpec()) == ()


def test_full_spec_lists_what_it_constrains():
    spec = GraspSpec(1, "handle", ["rim"], "pinch", ["thumb", "index"], 2.5)
    assert spec.requested() == ("part", "avoid_parts", "grasp_type", "fingers", "target_force_n")
    assert spec.avoid_parts == ("rim",) and spec.fingers == ("thumb", "index")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 0, "part": "handle"},
        {"schema_version": 1, "part": " handle"},
        {"schema_version": 1, "part": "x" * 65},
        {"schema_version": 1, "avoid_parts": "rim"},
        {"schema_version": 1, "avoid_parts": ["rim", "rim"]},
        {"schema_version": 1, "fingers": [""]},
        {"schema_version": 1, "fingers": [f"f{i}" for i in range(17)]},
        {"schema_version": 1, "target_force_n": -1.0},
        {"schema_version": 1, "target_force_n": math.nan},
        {"schema_version": 1, "target_force_n": True},
        {"schema_version": 1, "part": "rim", "avoid_parts": ["rim"]},
    ],
)
def test_invalid_specs_are_refused(kwargs):
    with pytest.raises(ValueError):
        GraspSpec(**kwargs)


def test_a_gripper_that_pinches_and_grasps_refuses_the_rest_instead_of_ignoring_it():
    gripper = GraspCapability(grasp_types=("pinch", "power"))
    assert gripper.unsupported(GraspSpec(1, grasp_type="pinch")) == ()
    assert gripper.unsupported(GraspSpec(1, grasp_type="tripod")) == ("grasp_type",)
    spec = GraspSpec(1, "handle", ["rim"], "power", ["thumb"], 3.0)
    assert gripper.unsupported(spec) == ("part", "avoid_parts", "fingers", "target_force_n")


def test_a_hand_honours_its_declared_fingers_only():
    hand = GraspCapability(("pinch",), ("thumb", "index", "middle"), part_targeting=True, force_control=True)
    assert hand.unsupported(GraspSpec(1, "handle", (), "pinch", ("thumb", "index"), 1.0)) == ()
    assert hand.unsupported(GraspSpec(1, fingers=("thumb", "little"))) == ("fingers",)


def test_failure_classes_cover_cancel_and_timeout():
    assert {"CANCELED", "TIMEOUT", "UNSUPPORTED"} <= FAILURE_CLASSES


def test_generated_message_roundtrip():
    from ibrobot_msgs.action import PickObject
    from ibrobot_msgs.msg import GraspSpec as WireSpec

    spec = GraspSpec(1, "handle", ("rim", "spout"), "power", ("thumb",), 4.0)
    wire = WireSpec()
    spec.fill(wire)
    assert GraspSpec.from_value(wire) == spec
    assert GraspSpec.from_value(WireSpec()) == GraspSpec()

    goal, result = PickObject.Goal(), PickObject.Result()
    assert goal.MODE_RECOVER == 3 and hasattr(goal, "grasp_spec")
    assert not hasattr(goal, "retain_for_place")  # without release the pick keeps holding
    for name in ("custody_id", "failure_class", "recoverable", "measured_phase", "execution_history"):
        assert hasattr(result, name), name


def test_an_executor_tells_a_newer_spec_from_a_malformed_one():
    hand = GraspCapability(grasp_types=("pinch",), part_targeting=True)
    assert check_grasp_spec(None, hand) == (GraspSpec(), "")
    spec, code = check_grasp_spec({**_mapping(), "grasp_type": "pinch", "part": "handle"}, hand)
    assert code == "" and spec.part == "handle"
    # A caller on a newer version is refused as unsupported, never as malformed.
    assert check_grasp_spec({**_mapping(), "schema_version": 2}, hand)[1] == GRASP_SPEC_UNSUPPORTED
    assert check_grasp_spec({**_mapping(), "grasp_type": "power"}, hand)[1] == GRASP_SPEC_UNSUPPORTED
    assert check_grasp_spec({**_mapping(), "target_force_n": -1.0}, hand)[1] == GRASP_SPEC_INVALID
    assert check_grasp_spec({"schema_version": 1}, hand)[1] == GRASP_SPEC_INVALID


def _mapping():
    return {
        "schema_version": 1,
        "part": "",
        "avoid_parts": [],
        "grasp_type": "",
        "fingers": [],
        "target_force_n": 0.0,
    }
