from pathlib import Path

import pytest
import yaml

from robot_navigation import save_lidar_map


def _write_map(prefix: Path) -> None:
    prefix.with_suffix(".pgm").write_bytes(b"P5\n1 1\n255\n\0")
    prefix.with_suffix(".yaml").write_text(
        yaml.safe_dump({"image": prefix.with_suffix(".pgm").name, "resolution": 0.05}),
        encoding="utf-8",
    )


def test_validate_saved_map_requires_yaml_and_pgm_pair(tmp_path):
    prefix = tmp_path / "pending"
    _write_map(prefix)

    assert save_lidar_map._validate_saved_map(prefix) == (prefix.with_suffix(".yaml"), prefix.with_suffix(".pgm"))


def test_publish_map_replaces_previous_valid_map(tmp_path):
    staged = tmp_path / "pending"
    output = tmp_path / "map"
    _write_map(staged)
    _write_map(output)

    save_lidar_map._publish_map_atomically(staged, output)

    assert output.with_suffix(".yaml").is_file()
    assert output.with_suffix(".pgm").is_file()
    assert not output.with_suffix(".yaml.previous").exists()
    assert not output.with_suffix(".pgm.previous").exists()


def test_publish_map_restores_old_pair_when_second_backup_fails(tmp_path, monkeypatch):
    staged = tmp_path / "pending"
    output = tmp_path / "map"
    _write_map(staged)
    _write_map(output)
    old_yaml = output.with_suffix(".yaml").read_bytes()
    old_image = output.with_suffix(".pgm").read_bytes()
    real_replace = save_lidar_map.os.replace
    calls = 0

    def fail_second_backup(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated image backup failure")
        real_replace(source, destination)

    monkeypatch.setattr(save_lidar_map.os, "replace", fail_second_backup)

    with pytest.raises(OSError, match="simulated image backup failure"):
        save_lidar_map._publish_map_atomically(staged, output)

    assert output.with_suffix(".yaml").read_bytes() == old_yaml
    assert output.with_suffix(".pgm").read_bytes() == old_image


def test_promote_saved_map_preserves_source_yaml_bytes_when_image_name_matches(tmp_path):
    source = tmp_path / "session" / "map"
    output = tmp_path / "maps" / "map"
    source.parent.mkdir()
    output.parent.mkdir()
    source_yaml = b"# preserve map metadata\nimage: map.pgm\nresolution: 0.05\n"
    source.with_suffix(".yaml").write_bytes(source_yaml)
    source.with_suffix(".pgm").write_bytes(b"new-map")

    save_lidar_map.promote_saved_map(source, output)

    assert output.with_suffix(".yaml").read_bytes() == source_yaml


def test_promote_saved_map_publishes_session_pair_to_navigation_path(tmp_path):
    source = tmp_path / "session" / "map"
    output = tmp_path / "maps" / "map"
    source.parent.mkdir()
    output.parent.mkdir()
    _write_map(source)
    _write_map(output)
    source.with_suffix(".pgm").write_bytes(b"new-map")

    save_lidar_map.promote_saved_map(source, output)

    assert output.with_suffix(".pgm").read_bytes() == b"new-map"
    metadata = yaml.safe_load(output.with_suffix(".yaml").read_text(encoding="utf-8"))
    assert metadata["image"] == "map.pgm"
    assert not output.with_suffix(".yaml.previous").exists()
    assert not output.with_suffix(".pgm.previous").exists()
    assert source.with_suffix(".yaml").is_file()
    assert source.with_suffix(".pgm").is_file()


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
