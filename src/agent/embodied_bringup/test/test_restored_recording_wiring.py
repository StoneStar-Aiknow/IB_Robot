"""Regression coverage for restored manifest-to-runtime audio wiring."""

from pathlib import Path

import pytest
import yaml

from embodied_bringup.launch_builders.embodied import runtime_play_audio_clips, runtime_spoken_clips


def test_manifest_resolves_lists_and_recorded_feedback():
    root = Path(__file__).resolve().parents[4]
    config = {
        "enabled": True,
        "clip_manifest": str(root / "commissioning/audio/summit_clips.yaml"),
        "clip_dir": str(root / "commissioning/audio/summit"),
    }
    clips = runtime_play_audio_clips(config)
    assert len(clips["prepare_done"]) == 3
    assert all(Path(path).is_file() for path in clips["prepare_done"])
    spoken = runtime_spoken_clips({"embodied": {"runtime_play_audio": config}})
    assert len(spoken["准备好啦"]) == 3
    assert len(spoken["好啦动作做完啦"]) == 3
    manifest = yaml.safe_load(Path(config["clip_manifest"]).read_text())
    for code in ("P01", "P02", "P03", "P04"):
        assert manifest["codes"][code]["consumer"] == "speech"


def test_gift_wake_reminder_uses_recorded_variants_without_immediate_repeats():
    from embodied_common.recorded_clips import spoken_key
    from ibrobot_agent.voice_bridge import choose_recorded_line
    from robot_config.loader import _load_robot_section

    root = Path(__file__).resolve().parents[4]
    _, robot = _load_robot_section(root / "src/robot_config/config/robots/aimdk_x2_summit.yaml")
    text = robot["embodied"]["agent"]["planner"]["answer_rules"]["lines"]["already_given"]
    clips = runtime_spoken_clips(robot)
    candidates = clips[spoken_key(text)]
    assert len(set(candidates)) == 3
    assert {Path(path).name for path in candidates} == {f"Q02_wake_{index:02}.wav" for index in (1, 2, 3)}
    last, previous = {}, ""
    for _ in range(12):
        selected = choose_recorded_line(text, clips, last, exists=lambda _: True)
        assert selected in candidates and selected != previous
        previous = selected


@pytest.mark.parametrize(("message", "code"), [("completed", "F18"), ("ended", "F19")])
def test_gift_terminal_messages_use_recorded_variants_without_immediate_repeats(message, code):
    from embodied_common.recorded_clips import spoken_key
    from ibrobot_agent.voice_bridge import choose_recorded_line
    from robot_config.loader import _load_robot_section

    root = Path(__file__).resolve().parents[4]
    _, robot = _load_robot_section(root / "src/robot_config/config/robots/aimdk_x2_summit.yaml")
    text = robot["embodied"]["agent"]["voice_bridge"]["gift_messages"][message]
    clips = runtime_spoken_clips(robot)
    candidates = clips[spoken_key(text)]
    assert {Path(path).name for path in candidates} == {f"{code}_{index:02}.wav" for index in (1, 2, 3)}
    last, previous = {}, ""
    for _ in range(12):
        selected = choose_recorded_line(text, clips, last, exists=lambda _: True)
        assert selected in candidates and selected != previous
        previous = selected
