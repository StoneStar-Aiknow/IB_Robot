"""Episode context: allowlisted inputs, the request shape, and nothing carried across decisions."""

from __future__ import annotations

import json

import pytest
from agent_eval.episode_context import (
    BodyFacts,
    EpisodeContext,
    EpisodeInfo,
    EpisodeObservation,
    RecipeFacts,
    approach_name,
)

PRESETS = {
    "horizontal": (0.0, 0.0, 0.7071068, 0.7071068),
    "top_down": (-0.5, 0.5, 0.5, 0.5),
    "top_down_rotated": (0.0, 0.7071068, 0.0, 0.7071068),
    "tilted_45": (-0.2705981, 0.2705981, 0.6532815, 0.6532815),
}
BODY = BodyFacts(sides=("left", "right"), approach_presets=PRESETS, approach_axis="x")
RECIPE = RecipeFacts(guide="Pick the object up. Say DONE when lifted.", fingers_on_table_z=0.795)


def observation(left_z=0.92, left_opening=1.0, tick=5, **extra):
    data = {
        "instruction": "Pick up the shell.",
        "tick": tick,
        "step_lim": 200,
        "arms": {
            "left": {
                "tcp_xyz": [-0.3, -0.22, left_z],
                "tcp_quat": PRESETS["top_down"],
                "gripper_opening": left_opening,
            },
            "right": {"tcp_xyz": [0.3, -0.22, 0.92], "tcp_quat": PRESETS["horizontal"], "wrist_joint_rad": 0.5},
        },
    }
    data.update(extra)
    return EpisodeObservation.from_mapping(data)


def answer(text):
    return {"state": "ANSWERED", "event_type": "answered", "user_message": text, "events": []}


def test_evaluator_and_answer_fields_never_reach_the_request_text():
    leaky = {
        "instruction": "Pick up the shell.",
        "tick": 3,
        "step_lim": 200,
        "native_success": "LEAK_NATIVE_SUCCESS",
        "reward": "LEAK_REWARD",
        "native_score": "LEAK_SCORE",
        "object_pose": {"x": "LEAK_OBJECT_X"},
        "info": {"success": "LEAK_INFO"},
        "arms": {
            "left": {"tcp_xyz": [0.1, 0.2, 0.9], "object_pose": "LEAK_ARM_OBJECT", "reward": "LEAK_ARM_REWARD"},
            "right": {"tcp_xyz": [0.3, 0.2, 0.9]},
        },
    }
    obs = EpisodeObservation.from_mapping(leaky)
    ctx = EpisodeContext(RECIPE, BODY)
    ctx.reset(EpisodeInfo("ep1"))
    text = ctx.build_request_text(obs)
    assert "LEAK" not in text
    assert "LEAK" not in json.dumps(ctx.state_view(obs))
    assert set(vars(obs)) == {"instruction", "tick", "step_lim", "arms"}


def test_the_request_is_a_function_of_the_current_observation_only():
    ctx = EpisodeContext(RECIPE, BODY)
    ctx.reset(EpisodeInfo("ep1", 1, tag="abcd1234"))
    first = ctx.build_request_text(observation())
    ctx.build_request_text(observation(left_z=0.80, left_opening=0.3))  # a decision in between
    assert ctx.build_request_text(observation()) == first
    assert first.startswith("EPISODE abcd1234\nTASK: Pick up the shell.\nROBOT STATE:\n")
    assert "left gripper: grasp point x=-0.300 y=-0.220 z=0.920 (fingers_above_table=0.125 m)" in first
    assert "pose=top_down" in first and "wrist joint=+29 deg" in first
    assert "TIME: 5 of 200 control ticks used" in first
    assert first.endswith(f"RULES: {RECIPE.guide}")
    ctx.reset(EpisodeInfo("ep2"))
    assert not ctx.build_request_text(observation()).startswith("EPISODE")


def test_done_is_an_end_request_only_when_answered():
    ctx = EpisodeContext(RECIPE, BODY)
    assert ctx.is_done_request(answer("done."))
    assert not ctx.is_done_request(answer("not finished"))
    assert not ctx.is_done_request({"state": "SUCCEEDED", "user_message": "DONE"})


def test_the_shipped_recipe_request_fits_the_agent_request_contract():
    from agent_eval.task_spec import load_recipe

    from ibrobot_agent.contracts import MAX_REQUEST_TEXT_LENGTH

    ctx = EpisodeContext(load_recipe("general_pickup_x5").facts(), BODY)
    ctx.reset(EpisodeInfo("ep1", tag="abcd1234"))
    long_instruction = observation(instruction="Pick up the black-handled whisk and lift it off the table.")
    assert len(ctx.build_request_text(long_instruction)) < MAX_REQUEST_TEXT_LENGTH


def test_gripper_pose_is_named_from_the_body_presets():
    assert approach_name((0.0, 0.0, 0.7072, 0.707), PRESETS, "x") == "horizontal"
    assert approach_name((0.5, -0.5, -0.5, -0.5), PRESETS, "x") == "top_down"  # -q is the same orientation
    assert approach_name((-0.1228, 0.1228, 0.6964, 0.6964), PRESETS, "x") == "custom (points +0.00,+0.94,-0.34)"
    assert approach_name(None, PRESETS, "x") == "?"
    # top_down twisted 180 deg about the approach axis grasps the same way
    assert approach_name((-0.7071, 0.0, 0.7071, 0.0), PRESETS, "x") == "top_down_rotated"
    # with a z approach axis and no presets the pointing direction is the tool z axis
    assert approach_name((0.0, 0.0, 0.0, 1.0), {}, "z") == "custom (points +0.00,+0.00,+1.00)"


def test_single_arm_body_without_presets_is_supported():
    ctx = EpisodeContext(RECIPE, BodyFacts(sides=("main",)))
    ctx.reset(EpisodeInfo("ep1"))
    obs = EpisodeObservation.from_mapping({"instruction": "Pick.", "arms": {"main": {"tcp_xyz": [0, 0, 1]}}})
    text = ctx.build_request_text(obs)
    assert "main gripper: grasp point x=0.000 y=0.000 z=1.000" in text
    assert "TIME: None of None control ticks used" in text


def test_invalid_recipes_and_bodies_are_refused():
    with pytest.raises(ValueError):
        RecipeFacts(guide=" ", fingers_on_table_z=0)
    with pytest.raises(ValueError):
        RecipeFacts(guide="say done", fingers_on_table_z=0, done_word="done")
    with pytest.raises(ValueError):
        BodyFacts(sides=())
