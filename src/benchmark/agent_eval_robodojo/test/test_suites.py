"""The shipped RoboDojo suites: found by name through the plugin package, the pickup planner view."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from agent_eval.suite import load_suite

SUITES = Path(__file__).resolve().parents[1] / "config" / "suites"


def test_shipped_e2_suite_is_found_by_name_and_parses():
    suite = load_suite("robodojo_general_pickup_e2")  # share/agent_eval_robodojo/config/suites
    assert "agent_eval_robodojo" in suite.path and suite.path.endswith("robodojo_general_pickup_e2.yaml")
    assert suite.platform == "robodojo"
    names = [layout["name"] for layout in suite.layouts]
    assert names == ["shovel_9001", "conch_9002", "whisk_9012"] * 2
    assert sum(layout["episodes"] for layout in suite.layouts) == 30
    # The recipe is written for the six pickup skills: no hold skill in the planner's view.
    assert suite.robot_config_overrides["embodied"]["execution"]["simulation_hold"] is None
    assert suite.robot_config_overrides["embodied"]["skill_catalog_profile"] == "robodojo_x5_pickup"


def test_overrides_derive_a_run_local_robot_config(tmp_path, monkeypatch):
    from agent_eval.suite import derived_robot_config, with_overrides

    monkeypatch.setenv("PLANNER_URL", "https://llm.example/v1")
    suite = with_overrides(
        load_suite(str(SUITES / "robodojo_general_pickup_e2.yaml")),
        ["embodied.agent.planner.base_url=${PLANNER_URL}", "embodied.agent.planner.api_key_env=MY_KEY"],
    )
    derived = derived_robot_config(suite, tmp_path)
    with open(derived.robot_config, encoding="utf-8") as fh:
        robot = yaml.safe_load(fh)["robot"]
    planner = robot["embodied"]["agent"]["planner"]
    assert planner["base_url"] == "https://llm.example/v1" and planner["api_key_env"] == "MY_KEY"
    assert planner["catalog_view"] == "full" and planner["model"]  # untouched keys stay
    assert "simulation_hold" not in robot["embodied"]["execution"]
    assert "hold_simulation" not in robot["embodied"]["agent"]["test_allowlist"]
    with pytest.raises(ValueError):
        with_overrides(suite, ["no-equals-sign"])


def test_the_smoke_suite_runs_one_episode_of_the_e2_planner_view():
    smoke = load_suite("robodojo_general_pickup_smoke")
    e2 = load_suite("robodojo_general_pickup_e2")
    assert [(layout["name"], layout["episodes"]) for layout in smoke.layouts] == [("shovel_9001", 1)]
    assert smoke.robot_config_overrides == e2.robot_config_overrides
