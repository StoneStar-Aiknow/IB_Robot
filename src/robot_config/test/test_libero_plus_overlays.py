"""LIBERO-plus variants preserve the baseline except for five provider deltas."""

import copy
from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "robots"


@pytest.mark.parametrize("suffix", ["", "_rtp"])
def test_libero_plus_overlay_has_only_five_deltas(suffix: str) -> None:
    base_path = CONFIG_DIR / f"franka_libero_evaluation{suffix}.yaml"
    plus_path = CONFIG_DIR / f"franka_libero_plus_evaluation{suffix}.yaml"
    name = plus_path.stem
    assert yaml.safe_load(plus_path.read_text())["robot"] == {
        "base_config": base_path.name,
        "name": name,
        "benchmark": {
            "instance_id": "libero_plus",
            "libero_provider": "libero-plus",
            "identity_packages": ["robosuite", "mujoco"],
            "evaluation": {"output": {"run_name": name}},
        },
    }

    base = load_robot_config_dict(base_path)
    resolved = load_robot_config_dict(plus_path)
    assert resolved.pop("_config_path") == str(plus_path)
    assert resolved.pop("_config_sources") == [str(base_path), str(plus_path)]
    expected = copy.deepcopy(base)
    expected.pop("_config_path")
    expected.pop("_config_sources")
    expected["name"] = name
    expected["benchmark"]["instance_id"] = "libero_plus"
    expected["benchmark"]["libero_provider"] = "libero-plus"
    expected["benchmark"]["identity_packages"] = ["robosuite", "mujoco"]
    expected["benchmark"]["evaluation"]["output"]["run_name"] = name
    assert resolved == expected
    # Replacement, not append: hf-libero must not leak into plus provenance.
    assert resolved["benchmark"]["identity_packages"] == ["robosuite", "mujoco"]


@pytest.mark.parametrize("suffix", ["", "_rtp"])
def test_libero_plus_user_overlay_replaces_lists_and_inherits_transport(tmp_path: Path, suffix: str) -> None:
    base_name = f"franka_libero_evaluation{suffix}.yaml"
    plus_name = f"franka_libero_plus_evaluation{suffix}.yaml"
    for name in (base_name, plus_name):
        (tmp_path / name).write_text((CONFIG_DIR / name).read_text())
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        f"robot:\n  base_config: {plus_name}\n  benchmark:\n"
        "    evaluation:\n      tasks: [1, 3]\n"
        "    identity_packages: [mujoco]\n"
    )
    inherited = load_robot_config_dict(tmp_path / plus_name)
    resolved = load_robot_config_dict(user_path)
    assert resolved["benchmark"]["evaluation"]["tasks"] == [1, 3]
    assert resolved["benchmark"]["identity_packages"] == ["mujoco"]
    assert resolved["contract"] == inherited["contract"]
    assert resolved["control_modes"] == inherited["control_modes"]
    assert resolved["_config_sources"] == [str(tmp_path / name) for name in (base_name, plus_name, "user.yaml")]
