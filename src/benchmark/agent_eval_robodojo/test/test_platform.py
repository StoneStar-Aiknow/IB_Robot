"""The RoboDojo agent_eval plugin builds its commands from options/env, never from code paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_eval.platform import load_platform


def suite(robot_config="robodojo_x5"):
    return SimpleNamespace(robot_config=robot_config, task="general_pickup", name="s")


def test_plugin_is_registered_and_needs_a_robodojo_root(monkeypatch):
    monkeypatch.delenv("ROBODOJO_ROOT", raising=False)
    plugin = load_platform("robodojo")
    with pytest.raises(RuntimeError, match="ROBODOJO_ROOT"):
        plugin.stack_command(suite())


def test_client_command_runs_one_layout_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_ROOT", str(tmp_path))
    plugin = load_platform("robodojo", {"gpu": 1})
    layout = {"name": "conch_9002", "platform_options": {"port": 35000}}
    spec = plugin.client_command(suite(), layout=layout, seed=9002, episodes=5, run_id="r1", pass_id="p01")
    assert spec.argv[0] == "bash" and spec.argv[1].endswith("robodojo_client/run_eval_client.sh")
    assert spec.env["SEED"] == "9002" and spec.env["EVAL_NUM"] == "5" and spec.env["TASK"] == "general_pickup"
    assert spec.env["PORT"] == "35000" and spec.env["GPU"] == "1"
    # Only what the client launcher reads (its allowlist); no evaluation variables ride along.
    assert not [name for name in spec.env if name.startswith("AGENT_EVAL")]
    assert spec.env["ROBODOJO_ROOT"] == str(tmp_path)


def test_stack_uses_the_derived_config_file_when_there_is_one(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_ROOT", str(tmp_path))
    derived = tmp_path / "robot_config.robodojo_x5.yaml"
    derived.write_text("robot: {}\n")
    plugin = load_platform("robodojo")
    assert f"config_path:={derived}" in plugin.stack_command(suite(str(derived))).argv
    assert "robot_config:=robodojo_x5" in plugin.stack_command(suite()).argv


def test_each_pass_gets_its_own_robodojo_run_and_video_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_ROOT", str(tmp_path))
    plugin = load_platform("robodojo")
    layout = {"name": "conch_9002"}
    ids, dirs = set(), set()
    for pass_id in ("p02", "p05"):  # the same layout and seed run twice in one suite
        spec = plugin.client_command(suite(), layout=layout, seed=9002, episodes=5, run_id="r1", pass_id=pass_id)
        ids.add(spec.env["ROBODOJO_RUN_ID"])
        video = plugin.video_dir(suite(), run_id="r1", layout=layout, seed=9002, pass_id=pass_id)
        assert video.endswith(spec.env["ROBODOJO_RUN_ID"])
        dirs.add(video)
    assert len(ids) == 2 and len(dirs) == 2
    identity = plugin.platform_identity(suite())
    assert identity["task"] == "general_pickup" and "task_config_sha256" in identity


def test_every_variable_the_plugin_sets_passes_the_client_launchers_allowlist(monkeypatch, tmp_path):
    from pathlib import Path

    from ament_index_python.packages import get_package_share_directory

    monkeypatch.setenv("ROBODOJO_ROOT", str(tmp_path))
    monkeypatch.setenv("CONDA_SH", str(tmp_path / "conda.sh"))
    plugin = load_platform("robodojo")
    spec = plugin.client_command(suite(), layout={"name": "l"}, seed=1, episodes=1, run_id="r", pass_id="p01")
    script = Path(get_package_share_directory("robodojo_robot")) / "robodojo_client" / "run_eval_client.sh"
    text = script.read_text(encoding="utf-8")
    start = text.index("keep=(") + len("keep=(")
    allowed = set(text[start : text.index(")", start)].split())
    assert set(spec.env) <= allowed, sorted(set(spec.env) - allowed)
