"""Unit tests for the shared safe-stop core.

Pure-Python. Covers: plan derivation (zeros/hold + joint mapping), joint-state
validation (names/length/finite/reorder), and the command construction rules
— zeros zeroed; hold prefers last action then unique-joint-mapped observation;
a hold channel with neither raises fatal (no fabricated value). Mirrors the
so101_single_arm contract shape: arm action.0..4 (hold) + gripper action.5 (hold).
"""

from __future__ import annotations

import pytest

from action_dispatch.safe_stop import (
    SafeStopError,
    SafeStopPlan,
    build_safe_stop_plan,
    construct_safety_command,
    validate_joint_state,
)


class _Spec:
    """Minimal SpecView-like stub matching the safe_stop reader."""

    def __init__(self, names, behavior, topic="/c", ros_type="Float64MultiArray", clamp=None):
        self.names = names
        self.safety_behavior = behavior
        self.topic = topic
        self.ros_type = ros_type
        self.clamp = clamp


JOINT_ORDER = ["1", "2", "3", "4", "5", "6"]


def _arm_specs():
    return [
        _Spec(["action.0", "action.1", "action.2", "action.3", "action.4"], "hold", "/arm"),
        _Spec(["action.5"], "hold", "/gripper"),
    ]


# ---------------------------------------------------------------------------
# Plan derivation.
# ---------------------------------------------------------------------------


def test_build_plan_maps_hold_channels_to_joint_indices():
    plan = build_safe_stop_plan(action_specs=_arm_specs(), joint_order=JOINT_ORDER)
    assert isinstance(plan, SafeStopPlan)
    assert plan.total_positions == 6
    arm, gripper = plan.channels
    assert arm.safety_behavior == "hold"
    assert arm.channel_names == ["action.0", "action.1", "action.2", "action.3", "action.4"]
    assert arm.joint_indices == [0, 1, 2, 3, 4]  # 1:1 positional map
    assert gripper.joint_indices == [5]


def test_build_plan_maps_named_hold_actions_by_joint_identity() -> None:
    plan = build_safe_stop_plan(
        action_specs=[_Spec(["joint3", "joint1"], "hold", "/arm")],
        joint_order=["joint1", "joint2", "joint3"],
    )

    assert plan.channels[0].joint_indices == [2, 0]


def test_build_plan_zeros_channel_has_no_joint_map():
    plan = build_safe_stop_plan(
        action_specs=[_Spec(["action.0", "action.1"], "zeros", "/base")], joint_order=JOINT_ORDER
    )
    ch = plan.channels[0]
    assert ch.safety_behavior == "zeros"
    assert ch.joint_indices == [None, None]


def test_build_plan_rejects_unknown_safety_behavior():
    with pytest.raises(SafeStopError, match="unknown safety_behavior"):
        build_safe_stop_plan(action_specs=[_Spec(["action.0"], "coast")], joint_order=JOINT_ORDER)


# ---------------------------------------------------------------------------
# Joint-state validation.
# ---------------------------------------------------------------------------


def test_validate_joint_state_accepts_complete_finite_sample():
    snap = validate_joint_state(
        joint_names=["6", "1", "2", "3", "4", "5"],  # out of order
        positions=[0.6, 0.1, 0.2, 0.3, 0.4, 0.5],
        expected_joint_order=JOINT_ORDER,
    )
    assert snap.valid
    # reordered into expected order
    assert snap.positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_validate_joint_state_rejects_wrong_name_set():
    snap = validate_joint_state(
        joint_names=["1", "2", "3"], positions=[0.1, 0.2, 0.3], expected_joint_order=JOINT_ORDER
    )
    assert not snap.valid


def test_validate_joint_state_rejects_non_finite():
    snap = validate_joint_state(
        joint_names=JOINT_ORDER, positions=[0.1, 0.2, float("nan"), 0.4, 0.5, 0.6], expected_joint_order=JOINT_ORDER
    )
    assert not snap.valid


def test_validate_joint_state_rejects_length_mismatch():
    snap = validate_joint_state(
        joint_names=JOINT_ORDER, positions=[0.1, 0.2, 0.3, 0.4, 0.5], expected_joint_order=JOINT_ORDER
    )
    assert not snap.valid


# ---------------------------------------------------------------------------
# Command construction.
# ---------------------------------------------------------------------------


def _valid_snapshot(positions):
    return validate_joint_state(joint_names=JOINT_ORDER, positions=positions, expected_joint_order=JOINT_ORDER)


def test_zeros_channel_zeroed():
    plan = build_safe_stop_plan(
        action_specs=[_Spec(["action.0", "action.1"], "zeros", "/base")], joint_order=JOINT_ORDER
    )
    cmds = construct_safety_command(plan=plan, last_action=[0.9, 0.9], joint_snapshot=_valid_snapshot([0.1] * 6))
    assert cmds == [[0.0, 0.0]]


def test_hold_channel_prefers_last_action_over_observation():
    plan = build_safe_stop_plan(action_specs=_arm_specs(), joint_order=JOINT_ORDER)
    cmds = construct_safety_command(
        plan=plan,
        last_action=[0.10, 0.11, 0.12, 0.13, 0.14, 0.99],  # arm + gripper last target
        joint_snapshot=_valid_snapshot([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert cmds == [[0.10, 0.11, 0.12, 0.13, 0.14], [0.99]]


def test_hold_channel_falls_back_to_joint_observation_without_last_action():
    plan = build_safe_stop_plan(action_specs=_arm_specs(), joint_order=JOINT_ORDER)
    cmds = construct_safety_command(
        plan=plan, last_action=None, joint_snapshot=_valid_snapshot([0.5, 0.6, 0.7, 0.8, 0.9, 0.15])
    )
    assert cmds == [[0.5, 0.6, 0.7, 0.8, 0.9], [0.15]]


def test_hold_channel_fabricates_nothing_when_neither_available():
    plan = build_safe_stop_plan(action_specs=_arm_specs(), joint_order=JOINT_ORDER)
    # no last action, invalid snapshot -> fatal (no fabricated value)
    from action_dispatch.safe_stop import JointSnapshot

    with pytest.raises(SafeStopError, match="refusing to fabricate"):
        construct_safety_command(plan=plan, last_action=None, joint_snapshot=JointSnapshot(valid=False))


def test_hold_channel_partial_last_action_filled_by_observation():
    plan = build_safe_stop_plan(action_specs=_arm_specs(), joint_order=JOINT_ORDER)
    # last action too short (missing gripper) -> arm uses last, gripper uses obs
    cmds = construct_safety_command(
        plan=plan,
        last_action=[0.10, 0.11, 0.12, 0.13, 0.14],  # 5 positions, gripper missing
        joint_snapshot=_valid_snapshot([0.0, 0.0, 0.0, 0.0, 0.0, 0.42]),
    )
    assert cmds[0] == [0.10, 0.11, 0.12, 0.13, 0.14]
    assert cmds[1] == [0.42]


def test_clamp_applied_to_hold_channel():
    plan = build_safe_stop_plan(
        action_specs=[_Spec(["action.0", "action.1"], "hold", "/arm", clamp=(0.0, 1.0))], joint_order=JOINT_ORDER
    )
    cmds = construct_safety_command(plan=plan, last_action=[1.5, -0.5], joint_snapshot=_valid_snapshot([0.0] * 6))
    assert cmds == [[1.0, 0.0]]  # clamped into [0.0, 1.0]


def test_command_length_matches_positions():
    plan = build_safe_stop_plan(action_specs=_arm_specs(), joint_order=JOINT_ORDER)
    cmds = construct_safety_command(
        plan=plan, last_action=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], joint_snapshot=_valid_snapshot([0.0] * 6)
    )
    assert sum(len(c) for c in cmds) == plan.total_positions == 6


# ---------------------------------------------------------------------------
# Clamping a zeros channel does not change the zeros.
# ---------------------------------------------------------------------------


def test_zeros_channel_clamp_irrelevant():
    plan = build_safe_stop_plan(
        action_specs=[_Spec(["action.0"], "zeros", "/base", clamp=(-1.0, 1.0))], joint_order=JOINT_ORDER
    )
    cmds = construct_safety_command(plan=plan, last_action=[9.0], joint_snapshot=_valid_snapshot([0.0] * 6))
    assert cmds == [[0.0]]
