"""Launch-support unit tests: simulation selection (runtime-simulation-platform D1)."""

import logging

import pytest

from robot_runtime.launch_support import RuntimeProfileError, resolve_sim_selection
from robot_runtime.sim_platforms import SimPlatform, SimSensorTraits


@pytest.fixture(autouse=True)
def _fake_first_party_platforms(monkeypatch):
    """Register fake gazebo/mujoco platforms for every test in this module.

    resolve_sim_selection checks registration for an effective physics
    selection (sim-platform-registry D2); the fake platforms keep these
    contract tests independent of any real simulator plugin (D6).
    """
    import robot_runtime.sim_platforms as sim_platforms

    platforms = {}
    for name in ("gazebo", "mujoco"):
        platform = SimPlatform(
            name=name,
            package="test",
            embodiment="robot_runtime.sim_platforms:SDK_PLATFORM",  # never resolved here
            world=None,
            scene_task=None,
            sensor_traits=SimSensorTraits(),
        )
        platforms[name] = platform
    monkeypatch.setattr(sim_platforms, "discover_sim_platforms", lambda *, entry_points=None: platforms)


def _profile(**extra):
    profile = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "use_sim": False,
        "sim_platforms": ["sdk"],
        "sim_platform": "sdk",
    }
    profile.update(extra)
    return profile


def test_no_selection_uses_profile_default():
    assert resolve_sim_selection(_profile(), "", "", "") == (False, None)
    assert resolve_sim_selection(_profile(use_sim=True), "", "", "") == (True, "sdk")


def test_platform_alone_implies_simulation():
    assert resolve_sim_selection(_profile(sim_platforms=["sdk", "gazebo"]), "", "gazebo", "") == (True, "gazebo")
    # ... even when the profile itself defaults to physical execution
    assert resolve_sim_selection(_profile(), "", "sdk", "") == (True, "sdk")


def test_use_sim_alone_uses_profile_default_platform():
    assert resolve_sim_selection(_profile(), "true", "", "") == (True, "sdk")
    assert resolve_sim_selection(_profile(sim_platforms=["sdk", "gazebo"], sim_platform="gazebo"), "true", "", "") == (
        True,
        "gazebo",
    )


def test_simulated_alias_maps_to_use_sim_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="robot_runtime.launch_support"):
        assert resolve_sim_selection(_profile(), "", "", "true") == (True, "sdk")
        assert resolve_sim_selection(_profile(), "", "", "false") == (False, None)
        assert resolve_sim_selection(_profile(use_sim=True), "", "", "1") == (True, "sdk")
        assert resolve_sim_selection(_profile(), "", "", "no") == (False, None)
    assert any(
        "simulated:=true is deprecated" in record.message and "use_sim" in record.message for record in caplog.records
    )
    assert any(
        "simulated:=false is deprecated" in record.message and "use_sim" in record.message for record in caplog.records
    )


def test_agreeing_use_sim_and_simulated_selection_is_accepted():
    assert resolve_sim_selection(_profile(), "true", "", "true") == (True, "sdk")
    assert resolve_sim_selection(_profile(), "false", "", "false") == (False, None)


def test_conflicting_use_sim_and_simulated_selection_fails_naming_both():
    with pytest.raises(RuntimeProfileError, match=r"use_sim:=True.*simulated:=false"):
        resolve_sim_selection(_profile(), "true", "", "false")
    with pytest.raises(RuntimeProfileError, match="pass at most one"):
        resolve_sim_selection(_profile(), "false", "", "true")


def test_physical_execution_with_platform_fails_naming_both():
    with pytest.raises(RuntimeProfileError, match=r"use_sim:=false.*sim_platform:='gazebo'"):
        resolve_sim_selection(_profile(sim_platforms=["sdk", "gazebo"]), "false", "gazebo", "")


def test_malformed_platform_value_fails():
    # D2: a platform name is an identifier — anything else fails outright
    with pytest.raises(RuntimeProfileError, match="unknown sim_platform 'teleport-fast!'"):
        resolve_sim_selection(_profile(), "true", "teleport-fast!", "")


def test_unregistered_platform_value_fails_naming_the_package():
    # D2: an identifier the registry never heard of is valid syntax; it only
    # needs the plugin once it is the effective selection, and then it fails
    # naming the platform and the package to install
    with pytest.raises(RuntimeProfileError, match="simulation platform 'teleport' is not installed"):
        resolve_sim_selection(_profile(sim_platforms=["sdk", "teleport"]), "true", "teleport", "")


def test_unparseable_boolean_values_fail():
    with pytest.raises(RuntimeProfileError, match="cannot parse use_sim:='banana'"):
        resolve_sim_selection(_profile(), "banana", "", "")
    with pytest.raises(RuntimeProfileError, match="cannot parse simulated:='banana'"):
        resolve_sim_selection(_profile(), "", "", "banana")


def test_undeclared_platform_fails_naming_platform_set_and_profile():
    profile = _profile()
    with pytest.raises(
        RuntimeProfileError,
        match=r"sim_platform 'gazebo' is not supported.*supported: \['sdk'\].*profile: /tmp/p.yaml",
    ):
        resolve_sim_selection(profile, "true", "gazebo", "", source="/tmp/p.yaml")


def test_declared_gazebo_platform_resolves():
    profile = _profile(sim_platforms=["sdk", "gazebo"])
    assert resolve_sim_selection(profile, "true", "gazebo", "") == (True, "gazebo")
    assert resolve_sim_selection(profile, "", "gazebo", "") == (True, "gazebo")


def test_declared_mujoco_platform_resolves_and_undeclared_fails():
    profile = _profile(sim_platforms=["sdk", "gazebo", "mujoco"])
    assert resolve_sim_selection(profile, "true", "mujoco", "") == (True, "mujoco")
    assert resolve_sim_selection(profile, "", "mujoco", "") == (True, "mujoco")
    with pytest.raises(
        RuntimeProfileError,
        match=r"sim_platform 'mujoco' is not supported.*supported: \['sdk'\].*profile: /tmp/p.yaml",
    ):
        resolve_sim_selection(_profile(), "true", "mujoco", "", source="/tmp/p.yaml")


def test_profile_without_selection_keys_keeps_legacy_behaviour():
    profile = {"runtime": {"name": "test_robot", "version": "0.1.0"}, "simulated": True}
    assert resolve_sim_selection(profile, "", "", "") == (True, "sdk")
    assert resolve_sim_selection(profile, "", "", "false") == (False, None)


# ---------------------------------------------------------------------------
# D3.1: the spawner list follows the mode table's per-platform overrides
# ---------------------------------------------------------------------------


def _mode_profile() -> dict:
    return {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "ros2_control": {"broadcasters": ["joint_state_broadcaster"]},
        "modes": {
            "initial": "idle",
            "idle": {
                "controllers": [],
                "controllers_by_platform": {"mujoco": ["base_velocity_controller"]},
                "transitions": ["base_navigation"],
            },
            "base_navigation": {"controllers": ["base_velocity_controller"], "transitions": ["idle"]},
        },
    }


def test_profile_controllers_applies_the_platform_override():
    from robot_runtime.launch_support import profile_controllers

    broadcasters, active, inactive = profile_controllers(_mode_profile(), "mujoco")
    assert broadcasters == ["joint_state_broadcaster"]
    # idle (initial) carries the base controller on mujoco: spawned ACTIVE
    assert active == ["base_velocity_controller"]
    # base_navigation resolves to the same controller: no duplicate, nothing inactive
    assert inactive == []


def test_profile_controllers_without_the_platform_keep_the_plain_sets():
    from robot_runtime.launch_support import profile_controllers

    for selection in (None, "gazebo", "sdk"):
        _broadcasters, active, inactive = profile_controllers(_mode_profile(), selection)
        assert active == [] and inactive == ["base_velocity_controller"], selection


def test_profile_controllers_rejects_a_bad_platform_table():
    from robot_runtime.launch_support import profile_controllers

    profile = _mode_profile()
    # D2: the key must be a platform identifier (any registered platform may
    # override, not a fixed list); a reserved non-platform key is rejected too
    profile["modes"]["idle"]["controllers_by_platform"] = {"teleport-fast!": ["x"]}
    with pytest.raises(ValueError, match="not a simulation platform"):
        profile_controllers(profile, "mujoco")
    profile["modes"]["idle"]["controllers_by_platform"] = {"embedded_sensor": ["x"]}
    with pytest.raises(ValueError, match="not a simulation platform"):
        profile_controllers(profile, "mujoco")


@pytest.mark.parametrize("key", ["simulated", "port", "calib_file", "reset_positions"])
def test_hardware_xacro_args_cannot_override_a_derived_render_arg(tmp_path, key):
    from robot_runtime.launch_support import RuntimeProfileError, render_robot_description

    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(
        '<?xml version="1.0"?><robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="base"/></robot>',
        encoding="utf-8",
    )
    profile = {
        "description": {"xacro": str(xacro)},
        "hardware": {"port": "/dev/null", "xacro_args": {key: "override", "profile_acc": "254"}},
    }
    with pytest.raises(RuntimeProfileError, match=rf"'hardware.xacro_args' must not set \['{key}'\]"):
        render_robot_description(profile, tmp_path / "profile.yaml", True)
    # Other keys stay allowed, as the shipped SO-101 servo tuning uses them.
    profile["hardware"]["xacro_args"] = {"profile_acc": "254"}
    assert "<robot" in render_robot_description(profile, tmp_path / "profile.yaml", True)


@pytest.mark.parametrize(
    "name",
    [" calib_file", "calib_file ", "\tcalib_file", " port ", "\n simulated \n", "\freset_positions"],
)
def test_hardware_xacro_args_whitespace_cannot_smuggle_a_reserved_arg(tmp_path, name):
    # xacro strips argument names before matching (cli.py load_mappings), so
    # " calib_file" is the calib_file argument to the renderer. The check must
    # fire on the stripped name, or the plugin reads a calibration every check
    # reports on as a different one (#248).
    from robot_runtime.launch_support import RuntimeProfileError, render_robot_description

    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(
        '<?xml version="1.0"?><robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="base"/></robot>',
        encoding="utf-8",
    )
    profile = {
        "description": {"xacro": str(xacro)},
        "hardware": {"port": "/dev/null", "xacro_args": {name: "override"}},
    }
    with pytest.raises(RuntimeProfileError, match="'hardware.xacro_args' must not set"):
        render_robot_description(profile, tmp_path / "profile.yaml", True)


def test_hardware_xacro_args_empty_name_is_rejected(tmp_path):
    from robot_runtime.launch_support import RuntimeProfileError, render_robot_description

    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(
        '<?xml version="1.0"?><robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="base"/></robot>',
        encoding="utf-8",
    )
    profile = {
        "description": {"xacro": str(xacro)},
        "hardware": {"port": "/dev/null", "xacro_args": {"  ": "override"}},
    }
    with pytest.raises(RuntimeProfileError, match="empty argument name"):
        render_robot_description(profile, tmp_path / "profile.yaml", True)


def test_hardware_xacro_args_names_stripping_to_one_argument_are_rejected(tmp_path):
    # Two names that differ only in whitespace are one xacro argument; the
    # later one wins, which is how a reserved value slips past the check.
    from robot_runtime.launch_support import RuntimeProfileError, render_robot_description

    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(
        '<?xml version="1.0"?><robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="base"/></robot>',
        encoding="utf-8",
    )
    profile = {
        "description": {"xacro": str(xacro)},
        "hardware": {"port": "/dev/null", "xacro_args": {"profile_acc": "254", " profile_acc ": "120"}},
    }
    with pytest.raises(
        RuntimeProfileError,
        match=r"maps multiple names to one argument: \['profile_acc', ' profile_acc '\]",
    ):
        render_robot_description(profile, tmp_path / "profile.yaml", True)


def test_render_robot_description_reports_the_evil_calibration_end_to_end(tmp_path):
    # End to end over the real renderer: a whitespace-prefixed calib_file must
    # never reach xacro, or the rendered URDF hands the hardware plugin a
    # calibration the preflight never checked (the reported P2).
    from robot_runtime.launch_support import RuntimeProfileError, render_robot_description

    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(
        '<?xml version="1.0"?><robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro">'
        '<xacro:arg name="calib_file" default=""/>'
        '<link name="$(arg calib_file)"/></robot>',
        encoding="utf-8",
    )
    profile = {
        "description": {"xacro": str(xacro)},
        "hardware": {
            "port": "/dev/null",
            "calib_file": str(tmp_path / "checked.json"),
            "xacro_args": {" calib_file": str(tmp_path / "evil.json")},
        },
    }
    (tmp_path / "checked.json").write_text("{}", encoding="utf-8")
    (tmp_path / "evil.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeProfileError, match="'hardware.xacro_args' must not set"):
        render_robot_description(profile, tmp_path / "profile.yaml", False)
