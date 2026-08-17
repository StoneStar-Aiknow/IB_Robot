from pathlib import Path

import yaml

from robot_navigation import save_lidar_map
from robot_navigation.save_lidar_map import _publish_map_atomically, _validate_saved_map


def _write_map(prefix: Path) -> None:
    prefix.with_suffix(".pgm").write_bytes(b"P5\n1 1\n255\n\0")
    prefix.with_suffix(".yaml").write_text(
        yaml.safe_dump({"image": prefix.with_suffix(".pgm").name, "resolution": 0.05}),
        encoding="utf-8",
    )


def test_validate_saved_map_requires_yaml_and_pgm_pair(tmp_path):
    prefix = tmp_path / "pending"
    _write_map(prefix)

    assert _validate_saved_map(prefix) == (prefix.with_suffix(".yaml"), prefix.with_suffix(".pgm"))


def test_publish_map_replaces_previous_valid_map(tmp_path):
    staged = tmp_path / "pending"
    output = tmp_path / "map"
    _write_map(staged)
    _write_map(output)

    _publish_map_atomically(staged, output)

    assert output.with_suffix(".yaml").is_file()
    assert output.with_suffix(".pgm").is_file()
    assert not output.with_suffix(".yaml.previous").exists()
    assert not output.with_suffix(".pgm.previous").exists()


def test_main_saves_map_using_staged_output(tmp_path, monkeypatch):
    output = tmp_path / "map"

    def fake_run(command, check):
        prefix = Path(command[command.index("-f") + 1])
        _write_map(prefix)

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(save_lidar_map.subprocess, "run", fake_run)

    assert save_lidar_map.main(["-f", str(output)]) == 0
    assert output.with_suffix(".yaml").is_file()
    assert output.with_suffix(".pgm").is_file()
