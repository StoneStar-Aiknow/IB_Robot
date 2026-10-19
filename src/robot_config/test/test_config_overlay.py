"""Tests for robot configuration inheritance and hand-path boundaries."""

from pathlib import Path

import pytest
import yaml

from robot_config.loader import _load_robot_section, load_robot_config_dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "robots"


def test_so101_rtp_overlay_keeps_only_application_deltas() -> None:
    # so101_rtp_distributed is the shipped real-world overlay: it layers the
    # RTP application deltas (observation transport, distributed inference,
    # disabled front camera, Gazebo pick_banana scene) on the runtime-provider
    # base instead of duplicating runtime-owned blocks (openspec D10).
    overlay_raw = yaml.safe_load((CONFIG_DIR / "so101_rtp_distributed.yaml").read_text(encoding="utf-8"))["robot"]

    assert overlay_raw["base_config"] == "so101_single_arm"
    assert overlay_raw["name"] == "so101_rtp_distributed"
    for key in ("joints", "moveit", "ros2_control"):
        assert key not in overlay_raw, f"runtime-owned robot.{key} belongs to the runtime profile, not the overlay"
    assert overlay_raw["simulation"] == {
        "platform": "gazebo",
        "scene": "pick_banana",
        "robot_spawn": {"x": 0.0, "y": 0.3, "z": -0.03},
        "sim_models_pkg": "sim_models",
    }
    assert {entry["name"]: entry for entry in overlay_raw["peripherals"]}["front"]["disabled"] is True

    base_path, base = _load_robot_section(CONFIG_DIR / "so101_single_arm.yaml")
    resolved_path, resolved = _load_robot_section(CONFIG_DIR / "so101_rtp_distributed.yaml")

    assert base_path.name == "so101_single_arm.yaml"
    assert resolved_path.name == "so101_rtp_distributed.yaml"
    assert resolved["name"] == "so101_rtp_distributed"
    # Sections the overlay does not override still resolve from the base.
    assert resolved["voice_tts"] == base["voice_tts"]


def test_overlay_append_requires_a_base_list(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("robot:\n  name: base\n  value: 1\n", encoding="utf-8")
    (tmp_path / "overlay.yaml").write_text(
        "robot:\n  name: overlay\n  base_config: base.yaml\n  value:\n    __append__: [2]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a list in the base configuration"):
        _load_robot_section(tmp_path / "overlay.yaml")


def test_overlay_append_rejects_ambiguous_mapping(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("robot:\n  name: base\n  values: [1]\n", encoding="utf-8")
    (tmp_path / "overlay.yaml").write_text(
        "robot:\n  name: overlay\n  base_config: base.yaml\n  values:\n    __append__: [2]\n    extra: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain only"):
        _load_robot_section(tmp_path / "overlay.yaml")


@pytest.mark.parametrize(
    ("base_value", "overlay_value"),
    [
        ("{enabled: true}", "[enabled]"),
        ("[base]", "{enabled: true}"),
        ("enabled", "{enabled: true}"),
    ],
)
def test_overlay_rejects_container_type_mismatch(tmp_path: Path, base_value: str, overlay_value: str) -> None:
    (tmp_path / "base.yaml").write_text(
        f"robot:\n  name: base\n  nested:\n    value: {base_value}\n",
        encoding="utf-8",
    )
    (tmp_path / "overlay.yaml").write_text(
        f"robot:\n  name: overlay\n  base_config: base.yaml\n  nested:\n    value: {overlay_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"overlay type mismatch at robot\.nested\.value"):
        _load_robot_section(tmp_path / "overlay.yaml")


def test_overlay_rejects_cross_directory_base_config(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    overlay_dir = tmp_path / "overlay"
    base_dir.mkdir()
    overlay_dir.mkdir()
    (base_dir / "base.yaml").write_text("robot:\n  name: base\n", encoding="utf-8")
    (overlay_dir / "overlay.yaml").write_text(
        "robot:\n  name: overlay\n  base_config: ../base/base.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must reference a sibling robot YAML"):
        _load_robot_section(overlay_dir / "overlay.yaml")


def test_overlay_cycle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("robot:\n  name: a\n  base_config: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("robot:\n  name: b\n  base_config: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base_config cycle detected"):
        _load_robot_section(tmp_path / "a.yaml")


def test_normalized_loader_reports_base_to_overlay_source_chain(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    middle = tmp_path / "middle.yaml"
    overlay = tmp_path / "overlay.yaml"
    base.write_text("robot:\n  name: base\n  values: [1]\n", encoding="utf-8")
    middle.write_text(
        "robot:\n  name: middle\n  base_config: base.yaml\n  values:\n    __append__: [2]\n",
        encoding="utf-8",
    )
    overlay.write_text("robot:\n  name: overlay\n  base_config: middle.yaml\n", encoding="utf-8")

    resolved = load_robot_config_dict(overlay)

    assert resolved["values"] == [1, 2]
    assert resolved["_config_path"] == str(overlay.resolve())
    assert resolved["_config_sources"] == [str(base.resolve()), str(middle.resolve()), str(overlay.resolve())]
