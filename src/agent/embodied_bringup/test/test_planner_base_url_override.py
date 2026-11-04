"""The planner endpoint follows the launch argument, not a profile edit.

The compute pack's planner host is whatever machine serves the model on the
network the pack is plugged into, so it changes on every site move while the
profile itself should not. These cases pin the two properties the launch
override relies on: an empty argument leaves the profile exactly as written,
and a deployment without an agent planner is never given one.
"""

from embodied_bringup.launch_builders.embodied import apply_planner_base_url

PROFILE_URL = "http://10.0.0.1:18094/v1"
LAUNCH_URL = "http://10.42.0.100:18094/v1"


def _profile(base_url: str = PROFILE_URL) -> dict:
    return {
        "embodied": {
            "agent": {
                "planner": {
                    "mode": "vlm",
                    "provider": "openai_compatible",
                    "base_url": base_url,
                    "model": "qwen3.8-27b",
                }
            }
        }
    }


def test_launch_argument_replaces_only_the_endpoint():
    robot_config = _profile()
    apply_planner_base_url(robot_config, LAUNCH_URL)
    planner = robot_config["embodied"]["agent"]["planner"]
    assert planner["base_url"] == LAUNCH_URL
    assert planner["mode"] == "vlm"
    assert planner["provider"] == "openai_compatible"
    assert planner["model"] == "qwen3.8-27b"


def test_empty_argument_keeps_the_profile_endpoint():
    for value in ("", "   ", None):
        robot_config = _profile()
        apply_planner_base_url(robot_config, value)
        assert robot_config["embodied"]["agent"]["planner"]["base_url"] == PROFILE_URL


def test_argument_is_trimmed():
    robot_config = _profile()
    apply_planner_base_url(robot_config, f"  {LAUNCH_URL}  ")
    assert robot_config["embodied"]["agent"]["planner"]["base_url"] == LAUNCH_URL


def test_deployment_without_an_agent_planner_is_never_given_one():
    without_planner = {"embodied": {"agent": {"test_allowlist": ["wave_hand"]}}}
    apply_planner_base_url(without_planner, LAUNCH_URL)
    assert "planner" not in without_planner["embodied"]["agent"]

    for robot_config in ({}, {"embodied": {}}, {"embodied": {"agent": None}}, {"embodied": None}):
        apply_planner_base_url(robot_config, LAUNCH_URL)
        embodied = robot_config.get("embodied")
        if isinstance(embodied, dict):
            assert "planner" not in embodied
