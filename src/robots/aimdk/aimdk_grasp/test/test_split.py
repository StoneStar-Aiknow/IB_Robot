"""The X2 grasp on two hosts: its head's options, its launch entry and who it refuses."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aimdk_grasp import split
from aimdk_grasp.split import REMOTE_BACKEND, CpuIsolation, HeadOptions, head_options, parse_cpu_list, split_config

LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "x2_grasp_head.launch.py"
HEAD = {"namespace": "/x2_grasp_head", "status_timeout_s": 1.0, "cpu_isolation": {"node": "4-7", "others": "1-3"}}


def x2(**changes):
    config = {
        "name": "aimdk_x2_grasp",
        "type": "x2_ultra_omnihand",
        "runtime": {"provider": "aimdk_robot", "profile": "x2_ultra_omnihand"},
        "grasp_execution": {"enabled": True, "backend": REMOTE_BACKEND, "backend_options": {"head": dict(HEAD)}},
    }
    config.update(changes)
    return config


def test_cpu_lists_read_as_linux_writes_them():
    assert parse_cpu_list("4-7") == {4, 5, 6, 7}
    assert parse_cpu_list("0, 2-3") == {0, 2, 3}
    for malformed in ("", "4-", "7-4", "a", "1,,2"):
        with pytest.raises(ValueError, match="malformed CPU list"):
            parse_cpu_list(malformed)


def test_the_head_options_default_to_the_x2_head_and_refuse_what_is_wrong():
    assert head_options({}) == HeadOptions()
    assert head_options({"head": HEAD}) == HeadOptions(
        "/x2_grasp_head", 1.0, CpuIsolation(frozenset({4, 5, 6, 7}), frozenset({1, 2, 3}))
    )

    def refused(**changes):
        with pytest.raises(ValueError) as raised:
            head_options({"head": {**HEAD, **changes}})
        return str(raised.value)

    assert "must not share CPUs" in refused(cpu_isolation={"node": "3-7", "others": "0-3"})
    assert "exactly node and others" in refused(cpu_isolation={"node": "4-7"})
    assert "malformed CPU list" in refused(cpu_isolation={"node": "4-", "others": "0-3"})
    for bad in (0.1, float("nan"), float("inf"), True, 11.0, "1"):
        assert "status_timeout_s must be a finite number of seconds in [0.2, 10]" in refused(status_timeout_s=bad)
    assert "namespace must be a non-empty string" in refused(namespace="/")
    assert "unknown keys ['package']" in refused(package="aimdk_grasp")


def test_only_an_x2_configuration_that_splits_its_grasp_has_a_head():
    assert split_config(x2()).cpu_isolation.node == {4, 5, 6, 7}
    single = x2(grasp_execution={"enabled": True, "backend": "aimdk_grasp.backend:x2_grasp_backends"})
    with pytest.raises(RuntimeError, match="serves only grasp_execution.backend"):
        split_config(single)
    with pytest.raises(RuntimeError, match="serves only"):
        split_config(x2(grasp_execution={**x2()["grasp_execution"], "enabled": False}))
    with pytest.raises(RuntimeError, match="binds an aimdk_robot runtime, not 'so101_runtime'"):
        split_config(x2(runtime={"provider": "so101_runtime"}))
    with pytest.raises(RuntimeError, match="not robot type 'x2_ultra'"):
        split_config(x2(type="x2_ultra"))  # no OmniHand: nothing for the head to plan
    with pytest.raises(RuntimeError, match="cpu_isolation"):
        split_config(x2(grasp_execution={**x2()["grasp_execution"], "backend_options": {"head": {"cpu_isolation": 1}}}))


def test_the_launch_keeps_its_own_threads_off_the_head_nodes_cores(monkeypatch):
    pinned = []
    monkeypatch.setattr(split.os, "listdir", lambda path: ["11", "12"] if path == "/proc/self/task" else [])
    monkeypatch.setattr(split.os, "sched_setaffinity", lambda task, cpus: pinned.append((task, set(cpus))))
    monkeypatch.setattr(split.os, "cpu_count", lambda: 8)
    head = split_config(x2())
    split.isolate_launch_cpus(head)
    assert pinned == [(11, {1, 2, 3}), (12, {1, 2, 3})]
    pinned.clear()
    split.isolate_launch_cpus(HeadOptions())
    assert not pinned
    monkeypatch.setattr(split.os, "cpu_count", lambda: 6)
    with pytest.raises(RuntimeError, match=r"does not have: \[6, 7\]"):
        split.isolate_launch_cpus(head)
    assert not pinned


def _prefix(node) -> str:
    return "".join(getattr(part, "text", "") for part in node.process_description.prefix)


def test_the_head_node_runs_on_its_own_cores_with_the_robots_grasp_configuration():
    pytest.importorskip("launch_ros")
    config = x2(runtime={"provider": "aimdk_robot", "interface_description": {"interfaces": {"x": {}}}})
    node = split.head_node(config, split_config(config))
    assert (vars(node).get("_Node__package"), vars(node).get("_Node__node_executable")) == (
        "aimdk_grasp",
        "x2_grasp_head_node",
    )
    assert _prefix(node) == "taskset -c 4,5,6,7"
    params = {
        "".join(getattr(part, "text", str(part)) for part in key): value
        for key, value in node._Node__parameters[0].items()
    }

    def decoded(name):
        # launch_ros keeps a parameter as substitutions; a JSON string comes back YAML-quoted.
        value = params[name]
        text = "".join(getattr(part, "text", str(part)) for part in value) if isinstance(value, tuple) else value
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]
        return json.loads(text)

    assert decoded("grasp_execution_json") == config["grasp_execution"]
    assert decoded("interface_description_json") == {"interfaces": {"x": {}}}
    assert "taskset" not in _prefix(split.head_node(config, HeadOptions()))


def _launch_module():
    spec = importlib.util.spec_from_file_location("x2_grasp_head_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_head_binds_to_the_runtime_the_robot_side_host_runs_and_refuses_any_other_robot(monkeypatch, tmp_path):
    pytest.importorskip("launch")
    module = _launch_module()
    path = tmp_path / "robot.yaml"
    path.write_text(json.dumps({"robot": x2()}), encoding="utf-8")
    loads = []

    def load(config_path, **options):
        loads.append(options)
        return json.loads(Path(config_path).read_text())["robot"]

    monkeypatch.setattr(module, "load_robot_config_dict", load)
    monkeypatch.setattr(module, "check_launch_dds", lambda _context: None)
    isolated, bound = [], []
    monkeypatch.setattr(module, "isolate_launch_cpus", lambda head: isolated.append(head))

    def bind(config, construct_consumers, **options):
        bound.append(options)
        effective = dict(config, runtime=dict(config["runtime"], interface_description={"interfaces": {}}))
        return construct_consumers(effective)

    monkeypatch.setattr(module, "generate_bound_runtime_actions", bind)
    context = SimpleNamespace(launch_configurations={"config_path": str(path), "control_mode": "moveit_planning"})
    (node,) = module.launch_setup(context)
    assert loads[0] == {"defer_interface_binding": True, "require_local_model_artifacts": False}
    assert vars(node).get("_Node__node_executable") == "x2_grasp_head_node"
    assert bound[0]["start_provider"] is False and isolated[0].cpu_isolation.node == {4, 5, 6, 7}
    # The head waits for the robot side however long it takes, unless told otherwise.
    assert bound[0]["readiness_timeout_s"] == float("inf")
    module.launch_setup(
        SimpleNamespace(launch_configurations={**context.launch_configurations, "runtime_timeout": "30"})
    )
    assert bound[1]["readiness_timeout_s"] == 30.0
    other = x2(name="so101", type="so101_follower", runtime={"provider": "so101_runtime"})
    path.write_text(json.dumps({"robot": other}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binds an aimdk_robot runtime"):
        module.launch_setup(context)
    assert len(bound) == 2  # refused before binding anything
