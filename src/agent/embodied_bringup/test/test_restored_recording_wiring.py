"""Regression coverage for restored manifest-to-runtime audio wiring."""

from pathlib import Path

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
