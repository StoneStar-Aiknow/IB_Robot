"""TaskSpec / Recipe schemas: shipped files load; answers and unknown fields are refused."""

from __future__ import annotations

import copy

import pytest
import yaml
from agent_eval.task_spec import SpecError, load_recipe, load_task_spec, parse_recipe, parse_task_spec
from agent_eval_testkit import CONFIG


def shipped_task() -> dict:
    return yaml.safe_load((CONFIG / "tasks" / "general_pickup.yaml").read_text(encoding="utf-8"))


def shipped_recipe() -> dict:
    return yaml.safe_load((CONFIG / "recipes" / "general_pickup_x5.yaml").read_text(encoding="utf-8"))


def test_shipped_task_spec_loads():
    spec = load_task_spec(str(CONFIG / "tasks" / "general_pickup.yaml"))
    assert (spec.task, spec.platform_task, spec.recipe) == ("general_pickup", "general_pickup", "general_pickup_x5")
    assert (spec.max_decisions, spec.decision_timeout_s) == (40, 300.0)


@pytest.mark.parametrize(
    "path, value",
    [
        (("object_pose",), {"x": 0.1, "y": 0.0, "z": 0.8}),
        (("target",), [0.1, 0.0]),
        (("success_threshold",), 0.1),
        (("observation", "object_position"), [0.1, 0.0]),
        (("decision_budget", "goal_height"), 0.1),
        (("layout",), "e2_layout_1"),
        (("answer",), "under the cup"),
    ],
)
def test_answer_like_fields_are_refused(path, value):
    data = shipped_task()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(SpecError, match="answer-like field"):
        parse_task_spec(data)


def test_unknown_fields_and_ungrantable_observations_are_refused():
    data = shipped_task()
    data["notes"] = "free text"
    with pytest.raises(SpecError, match="unknown keys"):
        parse_task_spec(data)
    data = shipped_task()
    data["observation"]["fields"].append("native_info")
    with pytest.raises(SpecError, match="not grantable"):
        parse_task_spec(data)


def test_recipe_schema_is_strict():
    assert load_recipe(str(CONFIG / "recipes" / "general_pickup_x5.yaml")).name == "general_pickup_x5"
    for mutate, match in (
        (lambda d: d["thresholds"].pop("fingers_on_table_z"), "missing"),
        (lambda d: d["thresholds"].update(object_z=0.8), "unknown keys"),
        (lambda d: d.update(done_word="FINISHED"), "done_word"),
        (lambda d: d.update(max_idle_replies=0), "positive"),
    ):
        data = copy.deepcopy(shipped_recipe())
        mutate(data)
        with pytest.raises(SpecError, match=match):
            parse_recipe(data)


def test_config_hashes_are_canonical():
    a = parse_task_spec(shipped_task())
    reordered = dict(reversed(list(shipped_task().items())))
    assert parse_task_spec(reordered).sha256 == a.sha256
    changed = shipped_task()
    changed["decision_budget"]["max_decisions"] = 41
    assert parse_task_spec(changed).sha256 != a.sha256
