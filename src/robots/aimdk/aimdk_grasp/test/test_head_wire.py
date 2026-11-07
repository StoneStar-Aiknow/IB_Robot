"""The X2 grasp pipeline's data survives the trip to the grasp head and back."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("aimdk_grasp_msgs")
pytest.importorskip("cv2")
pytest.importorskip("zstandard")

from aimdk_grasp import head_wire as wire  # noqa: E402

from manipulation_execution.grasp_pipeline.contracts import (  # noqa: E402
    ArmSegment,
    ContainerEstimate,
    Effect,
    EffectorAction,
    GraspPlan,
    Hold,
    Move,
    PlacePlan,
    SegmentKind,
)

JOINTS = ("a", "b", "c")


def _plan(release):
    approach = ArmSegment(
        "approach", SegmentKind.TRAJECTORY, JOINTS, np.linspace([0, 0, 0], [0.1, -0.2, 0.3], 5), np.linspace(0, 1, 5)
    )
    lift = ArmSegment("lift", SegmentKind.TARGET, JOINTS, np.array([[0.1, -0.2, 0.3], [0.1, -0.1, 0.3]]))
    return GraspPlan(
        (
            Effect(EffectorAction.PRESHAPE),
            Move(approach, refine_after=True),
            Effect(EffectorAction.CLOSE),
            Hold(0.5, name="settle"),
            Move(lift),
        ),
        candidate_index=7,
        release=release,
        report={"wall_s": 1.25, "candidates": [3, 7]},
    )


@pytest.mark.parametrize("release", [None, (Effect(EffectorAction.RELEASE),)])
def test_a_plan_comes_back_as_it_was_sent(release):
    plan = _plan(release)
    back = wire.plan_from_msg(wire.plan_to_msg(plan))
    assert back.candidate_index == 7 and back.report == plan.report
    assert (back.release is None) == (release is None)
    for sent, received in zip(plan.steps + (plan.release or ()), back.steps + (back.release or ()), strict=True):
        assert type(received) is type(sent) and received.name == sent.name
        if isinstance(sent, Move):
            assert received.refine_after == sent.refine_after and received.segment.kind is sent.segment.kind
            np.testing.assert_array_equal(received.segment.positions, sent.segment.positions)
            if sent.segment.times is None:
                assert received.segment.times is None
            else:
                np.testing.assert_array_equal(received.segment.times, sent.segment.times)


def test_a_step_whose_positions_do_not_fill_rows_is_refused():
    message = wire.step_to_msg(_plan(None).steps[1])
    message.positions = message.positions[:-1]
    with pytest.raises(wire.WireError, match="whole rows"):
        wire.step_from_msg(message)


def test_carry_and_place_paths_survive_the_generated_wire_types():
    from aimdk_grasp_msgs.action import PlanPlace

    source = _plan(None)
    carry = replace(source, carry_steps=source.steps)
    received = wire.plan_from_msg(wire.plan_to_msg(carry))
    assert len(received.carry_steps) == len(carry.carry_steps)
    np.testing.assert_array_equal(received.carry_steps[1].segment.positions, carry.carry_steps[1].segment.positions)

    container = ContainerEstimate([[0, 0, 0.1], [0.5, 0, 0.1], [0, 0.5, 0.1]], [0, 0, 1], 0.1)
    plan = PlacePlan(
        (source.steps[1],), (Effect(EffectorAction.RELEASE),), (Move(source.steps[1].segment.reversed()),), container
    )
    result = wire.place_to_result(plan, PlanPlace.Result())
    restored = wire.place_from_result(result)
    assert result.success and restored.release[0].action is EffectorAction.RELEASE
    np.testing.assert_array_equal(restored.transfer[0].segment.positions, plan.transfer[0].segment.positions)
    np.testing.assert_array_equal(restored.retreat[0].segment.positions, plan.retreat[0].segment.positions)
    np.testing.assert_array_equal(restored.container.opening, plan.container.opening)


def test_depth_comes_back_to_the_millimetre_with_its_holes():
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.2, 3.0, (48, 64)).astype(np.float32)
    depth[0, :5] = [0.0, np.nan, np.inf, -1.0, 70.0]  # nothing measured, or out of range
    back = wire.decode_depth(wire.encode_depth(depth))
    assert back.dtype == np.float32 and back.shape == depth.shape
    np.testing.assert_array_equal(back[0, :5], 0.0)
    assert np.max(np.abs(back[1:] - depth[1:])) <= 0.0005 + 1e-6


def test_a_mask_and_a_colour_image_come_back():
    rng = np.random.default_rng(2)
    mask = rng.random((48, 64)) > 0.7
    np.testing.assert_array_equal(wire.decode_mask(wire.encode_mask(mask)), mask)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 2] = 200, 40  # red stays red: the channels are not swapped
    back = wire.decode_rgb(wire.encode_rgb(rgb))
    assert back.shape == rgb.shape and np.abs(back.astype(int) - rgb).mean() < 3


def test_data_in_another_format_is_refused():
    depth = wire.encode_depth(np.ones((4, 4), dtype=np.float32))
    depth.format = "16UC1; zstd mm; 5x4"
    with pytest.raises(wire.WireError, match="size"):
        wire.decode_depth(depth)
    depth.format = "16UC1; png"
    with pytest.raises(wire.WireError, match="format"):
        wire.decode_depth(depth)
    with pytest.raises(wire.WireError, match="format"):
        wire.decode_rgb(wire.encode_mask(np.ones((4, 4), dtype=bool)))


def test_poses_and_matrices_keep_their_shape():
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [1, 2, 3]
    np.testing.assert_array_equal(wire.poses_from_list(wire.poses_to_list(poses)), poses)
    np.testing.assert_array_equal(wire.matrix_from_list(wire.matrix_to_list(np.eye(3), (3, 3)), (3, 3)), np.eye(3))
    with pytest.raises(wire.WireError):
        wire.matrix_to_list(np.full((4, 4), np.nan), (4, 4))
