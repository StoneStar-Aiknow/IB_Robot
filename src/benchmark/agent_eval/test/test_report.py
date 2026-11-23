"""Run report: deterministic hashes, honest nulls, canonical counters, Wilson numbers."""

from __future__ import annotations

import json

import pytest
from agent_eval.report import (
    NO_USAGE_REASON,
    CanonicalRun,
    aggregate,
    build_report,
    canonical_record,
    episode_entry,
    planner_identity,
    prompt_hash,
    wilson_interval,
    write_report,
)
from agent_eval.task_spec import load_recipe, load_task_spec
from agent_eval_testkit import CONFIG, fixture_body

from benchmark_runtime.canonical_report import reduce_episode_records


def summarize(episodes, planned=None):
    """The canonical summary of report entries (one task), as run_suite reduces them."""
    records = [
        canonical_record(ep, run_id="r", suite="s", task_id=0, task_name="l", episode_index=i)
        for i, ep in enumerate(episodes)
    ]
    return reduce_episode_records(records, run_id="r", suite="s", planned_episodes=planned or len(episodes))


def record(success, seed=0, detail=None, finish="agent_done"):
    native = (
        None if success is None else {"native_success": success, "native_score": 1.0 if success else 0.0, "seed": seed}
    )
    return {
        "session_id": f"s{seed:04d}",
        "generation": seed,
        "finish_reason": finish,
        "decisions": [
            {"terminal": {"state": "SUCCEEDED", "detail": detail}, "events": []},
            {"terminal": {"state": "ANSWERED"}, "events": []},
        ],
        "native_result": native,
        "episode_end": {"sim_time_s": 3.2},
    }


def test_wilson_interval_for_the_old_e2_result():
    low, high = wilson_interval(25, 30)
    assert 0.66 < low < 0.67 and 0.92 < high < 0.94
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)[0] == 0.0


def test_tokens_are_null_with_a_reason_unless_events_carry_usage():
    entry = episode_entry(record(True))
    assert entry["tokens"] is None and entry["tokens_reason"] == NO_USAGE_REASON
    assert entry["cost"] is None and entry["cost_reason"]
    entry = episode_entry(record(True, detail={"usage": {"prompt_tokens": 100, "completion_tokens": 7}}))
    assert entry["tokens"] == {"prompt_tokens": 100, "completion_tokens": 7} and entry["tokens_reason"] is None


def test_agent_done_is_not_success_and_missing_verdicts_are_not_counted():
    episodes = [episode_entry(record(False, 1)), episode_entry(record(None, 2), seed=2), episode_entry(record(True, 3))]
    assert episodes[0]["finish_reason"] == "agent_done" and episodes[0]["native_success"] is False
    assert episodes[1]["failure_reason"] == "no native result received"
    assert episodes[1]["infrastructure_error"].startswith("no native result received")
    agg = aggregate(episodes, summarize(episodes))
    counters = agg["counters"]
    assert (counters["started_episodes"], counters["completed_native_episodes"]) == (3, 2)
    assert counters["successful_native_episodes"] == 1 and counters["infrastructure_error_episodes"] == 1
    assert agg["run_status"] == "partial" and agg["complete"] is False
    assert "coarse alarm" in agg["wilson_95"]["note"]
    assert episodes[2]["skill_outcomes"] == {"ANSWERED": 1, "SUCCEEDED": 1}


def test_planner_identity_never_contains_keys():
    robot = {
        "embodied": {
            "agent": {
                "planner": {
                    "provider": "p",
                    "model": "m",
                    "base_url": "https://u:pw@llm.example:8443/v1",
                    "api_key": "sk-XX",
                    "api_key_env": "K",
                }
            }
        }
    }
    identity = planner_identity(robot)
    assert identity == {"mode": None, "provider": "p", "model": "m", "base_url_host": "llm.example"}


def test_report_hashes_are_deterministic(tmp_path):
    spec = load_task_spec(str(CONFIG / "tasks" / "general_pickup.yaml"))
    recipe = load_recipe(str(CONFIG / "recipes" / "general_pickup_x5.yaml"))
    kwargs = dict(
        run_id="r1",
        repo_dir=tmp_path,  # not a git work tree
        body=fixture_body(),
        robot={},
        task_spec=spec,
        recipe=recipe,
        layouts=[],
        episodes=[episode_entry(record(True))],
    )
    kwargs["summary"] = summarize(kwargs["episodes"])
    a, b = build_report(**kwargs), build_report(**kwargs)
    assert a == b
    assert a["recipe_task_hash"] == prompt_hash(recipe.guide, spec.source)
    assert all(len(v) == 64 for k, v in a["config_hashes"].items() if k not in ("suite", "robot_config_sources"))
    assert a["git"] == {"commit": None, "dirty": None}
    report_path = write_report(a, tmp_path / "out")
    assert json.loads(report_path.read_text())["run_id"] == "r1"


@pytest.mark.parametrize("k, n", [(30, 30), (1, 1)])
def test_wilson_interval_stays_in_bounds(k, n):
    low, high = wilson_interval(k, n)
    assert 0.0 <= low <= high <= 1.0


def test_platform_seed_wins_over_the_client_pass_seed():
    entry = episode_entry({"native_result": {"seed": 3, "native_success": True}}, layout="a", seed=9002)
    assert entry["seed"] == 3 and entry["client_seed"] == 9002


def test_interrupted_episodes_are_neither_success_nor_failure_and_mark_the_run_incomplete():
    judged_ok = episode_entry(record(True, 0))
    judged_fail = episode_entry(record(False, 1))
    interrupted = episode_entry(
        {
            "native_result": {"native_success": None, "failure_reason": "interrupted", "interrupted": True},
            "episode_end": {"interrupted": True, "reason": "platform connection lost (eof)"},
            "decisions": [],
        },
        seed=2,
    )
    missing = episode_entry({"decisions": []}, seed=3)
    episodes = [judged_ok, judged_fail, interrupted, missing]
    assert interrupted["native_success"] is None and interrupted["interrupted"] is True
    assert interrupted["infrastructure_error"] == "interrupted: platform connection lost (eof)"
    agg = aggregate(episodes, summarize(episodes))
    counters = agg["counters"]
    assert (counters["started_episodes"], counters["completed_native_episodes"]) == (4, 2)
    assert (counters["successful_native_episodes"], counters["native_failed_episodes"]) == (1, 1)
    assert counters["infrastructure_error_episodes"] == 2
    assert (agg["interrupted"], agg["unjudged"]) == (1, 2)
    # both rates side by side, no headline; the interrupted episode is in neither numerator nor judged
    assert agg["rates"] == {"native_model_success_rate": 0.5, "planned_success_rate": 0.25}
    assert "headline" not in json.dumps(agg) and "success_rate_all_attempts" not in agg
    assert agg["run_status"] == "partial" and agg["complete"] is False


def test_a_canonical_record_of_an_aborted_episode_takes_the_client_seed(tmp_path):
    aborted = episode_entry(
        {
            "session_id": "s0004",
            "generation": 4,
            "native_result": {"native_success": None, "failure_reason": "interrupted", "interrupted": True},
            "episode_end": {"interrupted": True, "reason": "session aborted by the caller"},
            "finish_reason": "stale_observation",
            "decisions": [],
        },
        layout="shovel_9001",
        seed=9001,
    )
    aborted["pass"] = "p01"
    record = canonical_record(aborted, run_id="r", suite="s", task_id=0, task_name="shovel_9001", episode_index=0)
    assert record.native_result_available is False and record.is_success is None and record.seed == 9001
    assert record.infrastructure_error == "interrupted: session aborted by the caller"
    assert record.metadata["seed_source"] == "client_pass" and record.metadata["layout"] == "shovel_9001"


def test_the_canonical_run_files(tmp_path):
    run = CanonicalRun(
        tmp_path / "canonical", run_id="r", suite="s", planned={0: {"task_name": "a", "planned_episodes": 3}}
    )
    for seed, success in ((0, True), (1, False)):
        run.append({**episode_entry(record(success, seed), layout="a"), "pass": "p01"}, 0)
    summary = run.finalize(stopped=False)
    assert summary.run_status == "partial"  # 2 of 3 planned
    manifest = json.loads((tmp_path / "canonical" / "run_manifest.json").read_text())
    assert manifest["status"] == "partial" and manifest["planned_episodes"] == 3
    assert json.loads((tmp_path / "canonical" / "summary.json").read_text())["counters"]["planned_episodes"] == 3
    lines = (tmp_path / "canonical" / "episodes.jsonl").read_text().splitlines()
    assert [(json.loads(line)["seed"], json.loads(line)["is_success"]) for line in lines] == [(0, True), (1, False)]
