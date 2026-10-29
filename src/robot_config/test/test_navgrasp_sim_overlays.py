"""Full-chain binding of the nav_grasp simulation overlays (D7, task 4.0).

Stages each overlay (navi in robot_config, lio in robot_navigation's test
assets), resolves the navigation stage, builds the deployment peripherals
fragment, projects the platform description the provider would publish, and
runs the strict ``bind_robot_interfaces`` — for both physics platforms. The
``wrist`` camera stays in the runtime inventory although the stage's
``peripheral_names`` does not list it (``peripheral_names`` only selects
fragment entries). A consumer left enabled for an excluded interface
(``voice_tts`` → ``audio.play``) must fail with ``unknown_interface``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.runtime import _write_peripherals_fragment
from robot_config.loader import load_robot_config_dict
from robot_runtime.interface_description import (
    build_description,
    merge_peripherals,
    platform_peripherals,
)
from robot_runtime.profile import load_profile
from test.navigation_sim_support import _package_dir, stage_navigation_config

PHYSICS_PLATFORMS = ("gazebo", "mujoco")


def _provider_profile(config: dict):
    from robot_runtime.launch_support import profile_share_path

    provider = str(config["runtime"]["provider"])
    profile_name = str(config["runtime"]["profile"])
    shipped = profile_share_path(provider, profile_name)
    assert shipped.is_file(), f"the provider ships no profile {profile_name!r}: {shipped}"
    return load_profile(shipped)


def _stage_binding_chain(overlay_path: Path, sim_platform: str):
    """The launch's offline chain: stage merge -> fragment -> description -> binding."""
    config = load_robot_config_dict(overlay_path, nav_stage="navigation", defer_interface_binding=True)
    fragment_path = _write_peripherals_fragment(config)
    fragment = yaml.safe_load(Path(fragment_path).read_text(encoding="utf-8")) if fragment_path else {}
    profile = _provider_profile(config)
    peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    descriptor = build_description(profile, peripherals=peripherals, use_sim=True, sim_platform=sim_platform)
    from robot_config.interface_binding import bind_robot_interfaces

    bound = bind_robot_interfaces(config, descriptor)
    return config, profile, peripherals, descriptor, bound


@pytest.mark.parametrize("sim_platform", PHYSICS_PLATFORMS)
def test_nav_grasp_navi_overlay_binds_on_both_physics_platforms(tmp_path, sim_platform):
    overlay = stage_navigation_config(tmp_path, sim_platform=sim_platform, base_config="lekiwi_nav_grasp")
    config, profile, peripherals, descriptor, bound = _stage_binding_chain(overlay, sim_platform)
    # the runtime under test is the mobile manipulator
    assert config["runtime"]["profile"] == "lekiwi_nav_grasp"
    # the excluded interfaces are gone from the description; the wrist camera
    # and the MID-360 stay
    keys = set(descriptor["interfaces"])
    assert not any(key.startswith("audio.") for key in keys)
    assert not any(key.startswith("camera.front.") for key in keys)
    assert any(key.startswith("camera.wrist.") for key in keys)
    assert any(key.startswith("lidar.mid360.") for key in keys)
    # the wrist camera stays in the runtime inventory although the stage's
    # peripheral_names does not list it (fragment selection only)
    assert "wrist" in {entry["name"] for entry in peripherals}
    # the consumers of the excluded interfaces are off in the bound stage
    assert bound["speech_direction"]["enabled"] is False
    assert bound["voice_tts"]["enabled"] is False
    # Nav2 runs the mobile-manipulator footprint
    params_file = bound["navigation"]["nav2_bringup"]["params_file"]
    assert "nav2_params_navgrasp_sim.yaml" in params_file


@pytest.mark.parametrize("sim_platform", PHYSICS_PLATFORMS)
def test_nav_grasp_lio_overlay_binds_on_both_physics_platforms(tmp_path, sim_platform):
    """robot_navigation's gate copy stages the same chain (sibling assets)."""
    package = Path(__file__).resolve().parents[2] / "robot_navigation"
    robot_config_package = _package_dir()
    destination = tmp_path / "staged"
    destination.mkdir(parents=True)
    # the loader resolves sibling bases only: stage the shipped base next to
    # the gate's overlay copy
    (destination / "lekiwi_nav_grasp.yaml").write_text(
        (robot_config_package / "config" / "robots" / "lekiwi_nav_grasp.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    document = yaml.safe_load(
        (package / "test" / "config" / "lekiwi_nav_grasp_lio_sim.yaml").read_text(encoding="utf-8")
    )
    document["robot"]["nav_stages"]["navigation"]["simulation"]["platform"] = sim_platform
    overlay = destination / "lekiwi_nav_grasp_lio_sim.yaml"
    overlay.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    config, _profile, _peripherals, descriptor, bound = _stage_binding_chain(overlay, sim_platform)
    assert config["runtime"]["profile"] == "lekiwi_nav_grasp"
    assert not any(key.startswith("audio.") for key in set(descriptor["interfaces"]))
    assert bound["voice_tts"]["enabled"] is False


@pytest.mark.parametrize("sim_platform", PHYSICS_PLATFORMS)
def test_a_consumer_left_enabled_for_an_excluded_interface_fails_the_binding(tmp_path, sim_platform):
    """The strict ``unknown_interface`` check: re-enabling voice_tts for the
    excluded audio.play interface must fail the launch instead of silently
    dropping the consumer."""
    overlay = stage_navigation_config(tmp_path, sim_platform=sim_platform, base_config="lekiwi_nav_grasp")
    config = load_robot_config_dict(overlay, nav_stage="navigation", defer_interface_binding=True)
    # flip the overlay's decision back on: the consumer is enabled while the
    # platform still excludes audio.play
    config["voice_tts"]["enabled"] = True
    fragment_path = _write_peripherals_fragment(config)
    fragment = yaml.safe_load(Path(fragment_path).read_text(encoding="utf-8")) if fragment_path else {}
    profile = _provider_profile(config)
    peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    assert not platform_peripherals(peripherals, sim_platform) or all(
        entry["name"] != "speaker" for entry in platform_peripherals(peripherals, sim_platform)
    )
    descriptor = build_description(profile, peripherals=peripherals, use_sim=True, sim_platform=sim_platform)
    from robot_config.interface_binding import InterfaceBindingError, bind_robot_interfaces

    with pytest.raises(InterfaceBindingError) as excinfo:
        bind_robot_interfaces(config, descriptor)
    assert "unknown_interface" in str(excinfo.value) and "audio" in str(excinfo.value)
