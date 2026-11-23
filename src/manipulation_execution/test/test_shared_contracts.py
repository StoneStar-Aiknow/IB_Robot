"""Contract types shared by pick and place: what a backend may and may not express."""

import numpy as np
import pytest

from manipulation_execution.shared.contracts import (
    ArmSegment,
    ContainerEstimate,
    Effect,
    EffectorAction,
    GraspFailure,
    GraspPlan,
    GraspVerdict,
    HeldObject,
    Move,
    PlacePlan,
    SegmentKind,
    fill_custody_msg,
    fill_held_msg,
    held_from_msg,
)


def _move(name="transfer"):
    return Move(ArmSegment(name, SegmentKind.TARGET, ("j",), np.array([[0.0], [1.0]])))


def test_effects_default_to_the_effectors_choice_and_compare_as_before():
    assert Effect(EffectorAction.CLOSE) == Effect(EffectorAction.CLOSE, fingers=(), target_force_n=0.0)
    assert Effect(EffectorAction.PRESHAPE, grasp_type="pinch").name == "preshape"
    assert Effect(EffectorAction.CLOSE, fingers=["thumb", "index"], target_force_n=2.0).fingers == ("thumb", "index")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": EffectorAction.CLOSE, "grasp_type": "pinch"},
        {"action": EffectorAction.RELEASE, "fingers": ("thumb",)},
        {"action": EffectorAction.PRESHAPE, "target_force_n": 1.0},
        {"action": EffectorAction.CLOSE, "target_force_n": -1.0},
        {"action": EffectorAction.CLOSE, "target_force_n": float("nan")},
    ],
)
def test_effect_parameters_belong_to_their_action(kwargs):
    with pytest.raises(ValueError):
        Effect(**kwargs)


def test_failures_carry_a_known_class_and_default_their_reason():
    failure = GraspFailure("PLANNING_FAILED", "no candidate", failure_class="PLANNING", recoverable=True)
    assert failure.reason_code == "PLANNING_FAILED" and failure.recoverable
    assert GraspFailure("X", "m").failure_class == ""
    with pytest.raises(ValueError):
        GraspFailure("X", "m", failure_class="OOPS")
    with pytest.raises(ValueError):
        GraspVerdict(True, failure_class="GRIP")
    assert GraspVerdict(False, failure_class="VERIFICATION").failure_class == "VERIFICATION"


def test_grasp_plan_keeps_its_carry_alternative():
    steps = (_move("approach"), Effect(EffectorAction.CLOSE))
    plan = GraspPlan(steps, 0, carry_steps=[_move("carry")])
    assert plan.carry_steps == (plan.carry_steps[0],) and plan.carry_steps[0].name == "carry"
    assert GraspPlan(steps, 0).carry_steps is None
    for carry in ((), (Effect(EffectorAction.RELEASE),)):
        with pytest.raises(ValueError):
            GraspPlan(steps, 0, carry_steps=carry)


def _held(**overrides):
    values = dict(
        custody_id="c1",
        target_query="cup",
        arm_side="left",
        joint_names=("a", "b"),
        joint_positions=(0.0, 0.1),
        home_positions=(0.0, 0.0),
        radius_m=0.05,
        points_in_palm=(0.0, 0.0, 0.1),
        effector_state_json="{}",
    )
    values.update(overrides)
    return HeldObject(**values)


def test_held_object_validation():
    assert _held(contact_positions=(0.2, 0.3)).contact_positions == (0.2, 0.3)
    for overrides in (
        {"contact_positions": (0.2,)},
        {"contact_positions": (0.2, float("nan"))},
        {"points_in_palm": (0.0, 0.1)},
        {"radius_m": 0.0},
        {"joint_names": ("a", "a")},
        {"return_positions": (0.0, 0.0, 1.0, 1.0), "return_times": (0.0, 0.0)},
    ):
        with pytest.raises(ValueError):
            _held(**overrides)
    _held(return_positions=(0.0, 0.0, 1.0, 1.0), return_times=(0.0, 0.5))


def test_held_object_round_trips_through_the_wire_message():
    from dataclasses import fields

    from embodied_common import held_custody_contract as custody
    from ibrobot_msgs.msg import HeldObject as Message

    held = _held(
        lift_m=0.12,
        observation_id="obs-7",
        lifted_at_ns=1_700_000_000_123_456_789,
        frame_id="left_palm",
        return_positions=(0.0, 0.0, 1.0, 1.0),
        return_times=(0.0, 0.5),
        contact_positions=(0.2, 0.3),
    )
    message = Message()
    fill_held_msg(
        message,
        held,
        custody_epoch="e1",
        state=custody.HELD,
        owner=custody.OWNER_PICK,
    )
    assert message.header.frame_id == "left_palm"
    assert (message.custody_epoch, message.state) == ("e1", custody.HELD)
    back = held_from_msg(message)
    # Every field survives, including the two that change shape on the wire.
    for item in fields(HeldObject):
        assert getattr(back, item.name) == getattr(held, item.name), item.name


def test_a_held_message_refuses_an_unpublishable_custody():
    from embodied_common import held_custody_contract as custody
    from ibrobot_msgs.msg import HeldObject as Message

    message = Message()
    with pytest.raises(ValueError):
        fill_held_msg(message, _held(), custody_epoch="e1", state=custody.HELD, owner="")
    with pytest.raises(ValueError):
        fill_held_msg(message, _held(), custody_epoch="", state=custody.HELD, owner=custody.OWNER_PICK)
    with pytest.raises(ValueError):
        fill_held_msg(
            message, _held(), custody_epoch="e1", state=custody.HELD, owner=custody.OWNER_PICK, reason_code="X"
        )
    assert message.custody_id == ""  # nothing was set
    with pytest.raises(ValueError):
        held_from_msg(Message())  # an EMPTY custody carries no object


def test_a_custody_without_an_object_is_published_through_the_same_checks():
    from embodied_common import held_custody_contract as custody
    from ibrobot_msgs.msg import HeldObject as Message

    message = Message()
    fill_custody_msg(
        message,
        custody_id="c1",
        arm_side="left",
        custody_epoch="e1",
        state=custody.UNKNOWN,
        reason_code=custody.STARTUP_UNVERIFIED,
    )
    assert (message.custody_id, message.arm_side, message.state) == ("c1", "left", custody.UNKNOWN)
    assert message.goal_active is False
    assert message.reason_code == custody.STARTUP_UNVERIFIED and message.custody_epoch == "e1"
    for kwargs in (
        {"state": custody.UNKNOWN},  # no reason
        {"state": custody.HELD, "owner": custody.OWNER_PICK},  # HELD must describe its object
        {"state": custody.EMPTY, "custody_epoch": ""},
        {"state": custody.EMPTY, "arm_side": ""},
        {"state": custody.EMPTY, "custody_id": ""},
    ):
        values = {"custody_id": "c2", "arm_side": "left", "custody_epoch": "e1", **kwargs}
        with pytest.raises(ValueError):
            fill_custody_msg(Message(), **values)


def test_a_place_releases_once_and_never_closes():
    container = ContainerEstimate(np.eye(3), np.array([0.0, 0.0, 1.0]), 0.1)
    release = (Effect(EffectorAction.RELEASE),)
    PlacePlan((_move(),), release, (), container)
    with pytest.raises(ValueError):
        PlacePlan((_move(),), release + release, (), container)
    with pytest.raises(ValueError):
        PlacePlan((_move(), Effect(EffectorAction.RELEASE)), release, (), container)
    with pytest.raises(ValueError):
        PlacePlan((_move(),), (Effect(EffectorAction.CLOSE), *release), (), container)
