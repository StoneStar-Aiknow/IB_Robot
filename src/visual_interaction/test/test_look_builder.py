"""The builder's internvl_voice switch: the look node, or exactly the 676fa608 output.

Without ``internvl_voice`` (or with it, or its look, off) every generated
action must be what 676fa608 generated; that is checked against the 676fa608
builder itself, read out of git. With the look on and this section off, one
look node: InternVL, the section's topics, a nice prefix.
"""

from __future__ import annotations

import copy
import json
import subprocess
import types
from pathlib import Path

import pytest
from test_launch_builder import config, node_parameters

from visual_interaction.launch_builders import visual_interaction as builder
from visual_interaction.launch_builders.visual_interaction import generate_visual_interaction_nodes

BASELINE = "676fa608"
_BUILDER_PATH = "src/visual_interaction/visual_interaction/launch_builders/visual_interaction.py"
_REPO = Path(__file__).resolve().parents[3]
_HALF = "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"
_TOPIC_SOURCE = {
    "topic": _HALF,
    "qos": {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5},
}


@pytest.fixture(scope="module")
def baseline():
    """The 676fa608 builder module, or a skip where git or the commit is not there."""
    try:
        source = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{BASELINE}:{_BUILDER_PATH}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"the {BASELINE} builder is not readable from git: {error}")
    module = types.ModuleType("visual_interaction_builder_676fa608")
    exec(compile(source, f"{BASELINE}:{_BUILDER_PATH}", "exec"), module.__dict__)
    return module


def _substitutions(values) -> list:
    if isinstance(values, str):
        return [("str", values)]
    return [(type(item).__name__, item.describe()) for item in values or ()]


def describe(node) -> dict:
    """Everything a launch Node carries that this builder sets."""
    return {
        "type": type(node).__name__,
        "package": node._Node__package,
        "executable": node._Node__node_executable,
        "name": node._Node__node_name,
        "parameters": node_parameters(node),
        "arguments": [_substitutions(item) for item in (node._Node__arguments or ())],
        "prefix": _substitutions(node.process_description.prefix),
        "cmd": [_substitutions(item) for item in node.process_description._Executable__cmd],
        "output": _substitutions(node._ExecuteLocal__output),
        "sigterm_timeout": _substitutions(node._ExecuteLocal__sigterm_timeout),
    }


def look_section(**look) -> dict:
    return {"enabled": True, "look": {"enabled": True, "demo_root": "/root/vlx_internvl_310p", **look}}


def look_config(section_overrides=None, **look) -> dict:
    overrides = {
        "enabled": False,
        "frame_source": _TOPIC_SOURCE,
        "actor_state_topic": "/motion_imitation_stream/actor_state",
    }
    overrides["mirror_state_topic"] = "/mirror_session/state"
    overrides.update(section_overrides or {})
    robot = config(**overrides)
    robot["internvl_voice"] = look_section(**look)
    return robot


class Logged:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def info(self, text: str) -> None:
        self.infos.append(text)


@pytest.fixture
def logged(monkeypatch):
    """The builder's log, and robot_config's share dir pointed at its source (no install here)."""
    log = Logged()
    monkeypatch.setattr(builder, "_logger", lambda: log)
    monkeypatch.setattr(builder, "_find_share", lambda package: str(_REPO / "src" / package))
    monkeypatch.setenv("WORKSPACE", str(_REPO))
    return log


# ---------------------------------------------------------------------------
# The baseline is untouched.

_BASELINE_CASES = {
    "fake": {},
    "internvl": {"vl": {"implementation": "internvl", "demo_root": "/root/vlx_internvl_310p", "max_new_tokens": 16}},
    "chatter": {"mode": "chatter", "voice_status_topic": "/mirror_voice/status", "chatter": {"gap_min_sec": 4.0}},
    "topic": {"frame_source": _TOPIC_SOURCE, "mirror_state_topic": "/mirror_session/state"},
    "disabled": {"enabled": False},
}


@pytest.mark.parametrize("case", sorted(_BASELINE_CASES))
@pytest.mark.parametrize(
    "voice",
    [
        None,
        {"enabled": False, "look": {"enabled": True}},
        {"enabled": True, "look": {"enabled": False}},
        "absent",
    ],
    ids=["no-block", "block-off", "look-off", "no-section"],
)
def test_without_the_look_the_actions_are_the_baseline_s(baseline, logged, case, voice):
    robot = config(**_BASELINE_CASES[case])
    if voice == "absent":
        del robot["visual_interaction"]
    elif voice is not None:
        robot["internvl_voice"] = voice
    expected = baseline.generate_visual_interaction_nodes(copy.deepcopy(robot))
    actual = generate_visual_interaction_nodes(copy.deepcopy(robot))
    assert [describe(node) for node in actual] == [describe(node) for node in expected]
    assert logged.warnings == []


def test_the_baseline_under_rt_infer_is_the_baseline_s(baseline, logged, monkeypatch):
    robot = config(vl={"implementation": "internvl", "demo_root": "/root/vlx_internvl_310p"})
    robot["rt_infer"] = {"enable": True, "bundle_version": "2026.11.05"}
    # One temp file per call otherwise, which is all that would differ.
    monkeypatch.setattr(
        "inference_service.launch_builders.perception_models.write_rt_infer_snapshot",
        lambda _config: Path("/tmp/rt_infer_snapshot.json"),
    )
    try:
        expected = baseline.generate_visual_interaction_nodes(copy.deepcopy(robot))
    except Exception as error:  # noqa: BLE001 - the rt_infer snapshot needs a writable runtime dir
        pytest.skip(f"rt_infer wrapping is not available here: {error}")
    actual = generate_visual_interaction_nodes(copy.deepcopy(robot))
    assert [describe(node) for node in actual] == [describe(node) for node in expected]


def test_absent_and_disabled_blocks_generate_the_same(logged):
    robot = config()
    off = copy.deepcopy(robot)
    off["internvl_voice"] = {"enabled": False}
    assert [describe(node) for node in generate_visual_interaction_nodes(robot)] == [
        describe(node) for node in generate_visual_interaction_nodes(off)
    ]


# ---------------------------------------------------------------------------
# The look node.


def test_look_on_and_the_section_off_launch_the_look_node(logged):
    (node,) = generate_visual_interaction_nodes(look_config())
    shown = describe(node)
    assert (shown["package"], shown["executable"]) == ("visual_interaction", "visual_interaction_node")
    assert shown["prefix"] == [("TextSubstitution", "'nice -n 10'")]
    parameters = shown["parameters"]
    assert parameters["vl_implementation"] == "internvl"
    assert parameters["vl_demo_root"] == "/root/vlx_internvl_310p"
    assert parameters["frames_per_window"] == 1
    assert parameters["camera_topic"] == _HALF
    assert json.loads(parameters["camera_topic_qos_json"])["reliability"] == "best_effort"
    assert parameters["actor_state_topic"] == "/motion_imitation_stream/actor_state"
    assert parameters["mirror_state_topic"] == "/mirror_session/state"
    assert parameters["voice_status_topic"] == "/mirror_voice/status"
    assert "mode" not in parameters
    look = json.loads(parameters["look_json"])
    assert look["round_period_sec"] == 0.0 and look["window_sec"] == 0.0
    assert {"hand.heart", "seq.invite", "other.any"} <= set(look["pools"])
    assert not any(pool.startswith(("evt.", "standby.")) for pool in look["pools"])


def test_the_look_is_internvl_even_when_the_section_says_fake(logged):
    (node,) = generate_visual_interaction_nodes(look_config({"vl": {"implementation": "fake"}}))
    assert node_parameters(node)["vl_implementation"] == "internvl"


def test_the_look_settings_reach_the_node(logged):
    (node,) = generate_visual_interaction_nodes(
        look_config(nice=5, window_sec=4.0, round_period_sec=2.5, verify=False, demo_root="/opt/vl")
    )
    shown = describe(node)
    assert shown["prefix"] == [("TextSubstitution", "'nice -n 5'")]
    assert shown["parameters"]["vl_demo_root"] == "/opt/vl"
    look = json.loads(shown["parameters"]["look_json"])
    assert (look["window_sec"], look["round_period_sec"], look["verify"]) == (4.0, 2.5, False)


def test_the_section_s_voice_status_topic_wins(logged):
    (node,) = generate_visual_interaction_nodes(look_config({"voice_status_topic": "/other_voice/status"}))
    assert node_parameters(node)["voice_status_topic"] == "/other_voice/status"


def test_both_on_keep_only_the_section_s_node_and_warn(baseline, logged):
    robot = config()
    robot["internvl_voice"] = look_section()
    expected = baseline.generate_visual_interaction_nodes(copy.deepcopy(robot))
    actual = generate_visual_interaction_nodes(robot)
    assert [describe(node) for node in actual] == [describe(node) for node in expected]
    assert [warning for warning in logged.warnings if "only the visual_interaction node" in warning]


def test_the_look_without_a_section_to_take_the_camera_from_is_no_node(logged):
    robot = look_config()
    del robot["visual_interaction"]
    assert generate_visual_interaction_nodes(robot) == []
    assert any("no visual_interaction section" in warning for warning in logged.warnings)


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"frame_source": {"topic": _HALF}}, "frame_source"),
        ({"actor_state_topic": "relative"}, "actor_state_topic"),
        ({"mirror_state_stale_sec": "soon"}, "mirror_state_stale_sec"),
        ({"buffer_sec": [6]}, "float"),
    ],
)
def test_a_broken_section_costs_the_look_only_and_warns(logged, overrides, why):
    assert generate_visual_interaction_nodes(look_config(overrides)) == []
    assert [warning for warning in logged.warnings if "no look" in warning and why in warning]


def test_a_motion_producer_beside_the_look_costs_the_look_only(logged):
    robot = look_config()
    robot["control_modes"] = {"teleop": {"inference": {"enabled": True}}}
    assert generate_visual_interaction_nodes(robot) == []
    assert [warning for warning in logged.warnings if "no look" in warning and "observation-only" in warning]


def test_the_shipped_voice_mirror_all_config_launches_the_look(logged):
    """The real merge chain: visual_interaction off, internvl_voice.look on."""
    try:
        from robot_config.loader import _load_robot_section
    except ImportError as error:
        pytest.skip(f"robot_config's loader needs the workspace packages: {error}")
    path = _REPO / "src" / "robot_config" / "config" / "robots" / "aimdk_x2_voice_mirror_all.yaml"
    _, robot = _load_robot_section(path)
    robot.setdefault("runtime", {})["interface_description"] = config()["runtime"]["interface_description"]
    (node,) = generate_visual_interaction_nodes(robot)
    parameters = node_parameters(node)
    assert parameters["vl_implementation"] == "internvl"
    assert parameters["camera_topic"] == _HALF
    assert describe(node)["prefix"] == [("TextSubstitution", "'nice -n 10'")]
