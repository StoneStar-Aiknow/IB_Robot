"""runtime_preflight: offline check of a physical profile against the local calibration file.

Every case builds its own profile, xacro and calibration under ``tmp_path``;
nothing reads the calibration of an arm attached to the test machine.
"""

from __future__ import annotations

import io
import json
import math

import pytest
import yaml

from robot_runtime.preflight import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    checked_physical_description,
    main,
    run_preflight,
)

_TICKS_PER_RAD = 4096 / (2 * math.pi)

# Joint 1's URDF window (-3.0 .. 0.5) is narrower than its calibrated upper
# end, so the effective upper limit comes from the URDF; joint 6 is a gripper.
_XACRO = """<?xml version="1.0"?>
<robot name="preflight_test" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base"/>
  <link name="shoulder"/>
  <link name="tool"/>
  <joint name="1" type="revolute">
    <parent link="base"/><child link="shoulder"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-3.0" upper="0.5"/>
  </joint>
  <joint name="2" type="revolute">
    <parent link="shoulder"/><child link="tool"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-3.0" upper="3.0"/>
  </joint>
  <joint name="6" type="revolute">
    <parent link="tool"/><child link="base_gripper"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="0.0" upper="1.0"/>
  </joint>
  <link name="base_gripper"/>
</robot>
"""


def _ticks(rad: float) -> int:
    return round(2048 + rad * _TICKS_PER_RAD)


def _calibration(joint2_max_rad: float = 1.5) -> dict:
    return {
        "1": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": _ticks(-1.0), "range_max": _ticks(1.0)},
        "2": {
            "id": 2,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": _ticks(-1.0),
            "range_max": _ticks(joint2_max_rad),
        },
        "6": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 1560, "range_max": 3080},
    }


def _profile_path(tmp_path, calibration=None, *, calib_file="calib.json", home_joint2=1.2, **overrides):
    xacro = tmp_path / "robot.urdf.xacro"
    xacro.write_text(_XACRO, encoding="utf-8")
    hardware = {}
    if calib_file:
        hardware["calib_file"] = str(tmp_path / calib_file)
        if calibration is not None:
            (tmp_path / calib_file).write_text(json.dumps(calibration), encoding="utf-8")
    profile = {
        "runtime": {"name": "preflight_test", "version": "0.0.1"},
        "modes": {"initial": "idle", "idle": {"controllers": [], "transitions": []}},
        "capabilities": {"joint.state": {"joint_count": 3, "rate_hz": 10.0}},
        "joints": ["1", "2", "6"],
        "arm_joints": ["1", "2"],
        "gripper_joints": ["6"],
        "motion": {"base_link": "base", "ee_link": "tool", "shoulder_link": "shoulder"},
        "home_positions": {"1": 0.0, "2": home_joint2},
        "description": {"xacro": str(xacro)},
        "hardware": hardware,
    }
    profile.update(overrides)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return path


def _run(argv):
    out = io.StringIO()
    code = main(argv, stdout=out)
    return code, out.getvalue()


def test_consistent_calibration_passes_and_reports_every_position_joint(tmp_path):
    report = run_preflight(_profile_path(tmp_path, _calibration()))

    assert report.error is None
    rows = {row.name: row for row in report.rows}
    assert set(rows) == {"1", "2", "6"}
    # Joint 2: the calibrated range is inside the URDF window, so it is the limit.
    assert rows["2"].effective[1] == pytest.approx(rows["2"].calibrated[1])
    assert rows["2"].bound_by == "calib/calib"
    assert rows["2"].margin == pytest.approx(rows["2"].effective[1] - 1.2)
    # Joint 1: the URDF upper limit (0.5) is tighter than the calibration (1.0).
    assert rows["1"].effective[1] == pytest.approx(0.5)
    assert rows["1"].bound_by == "calib/urdf"
    # The gripper has no home pose in the profile: reported, but without a margin.
    assert rows["6"].home is None and rows["6"].margin is None


def test_cli_exit_code_and_output_for_a_consistent_profile(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, _calibration()))])

    assert code == EXIT_OK
    assert "PASS  runtime checks of a physical launch; home within limits for joints: 1, 2" in out
    assert "NOTE  no home pose in the profile, not checked: 6" in out
    assert "FAIL" not in out and "WARN" not in out


def test_home_outside_the_calibrated_range_fails_with_the_launch_error(tmp_path):
    # Calibrated upper end 1.0 rad, home 1.2 rad: the launch check rejects it.
    path = _profile_path(tmp_path, _calibration(joint2_max_rad=1.0))

    report = run_preflight(path)
    assert report.error == "ValueError: robot_model.home_positions.2 is outside physical limits"
    # The failing joint is still reported, with a negative margin.
    row = next(row for row in report.rows if row.name == "2")
    assert row.margin < 0

    code, out = _run(["--profile", str(path)])
    assert code == EXIT_FAILED
    assert "FAIL  ValueError: robot_model.home_positions.2 is outside physical limits" in out


def test_home_near_a_limit_passes_with_a_warning(tmp_path):
    # Upper end 1.23 rad, home 1.2 rad: inside the limit, 0.03 rad away.
    path = _profile_path(tmp_path, _calibration(joint2_max_rad=1.23))

    code, out = _run(["--profile", str(path)])
    assert code == EXIT_OK
    assert "WARN  joint 2: home +1.200 rad" in out
    assert "upper limit, set by calib" in out

    code, out = _run(["--profile", str(path), "--min-margin", "0.01"])
    assert code == EXIT_OK
    assert "WARN" not in out


def test_missing_calibration_file_fails(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, calibration=None))])

    assert code == EXIT_FAILED
    assert "FAIL  RuntimeProfileError: calibration file not found" in out


def test_non_object_calibration_fails(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, calibration=[]))])

    assert code == EXIT_FAILED
    assert "FAIL  ValueError: runtime calibration must be a JSON object" in out


def test_calibration_missing_a_profile_joint_fails(tmp_path):
    calibration = _calibration()
    del calibration["2"]

    code, out = _run(["--profile", str(_profile_path(tmp_path, calibration))])

    assert code == EXIT_FAILED
    assert "FAIL  KeyError: Joint '2' missing from calibration data" in out


def test_position_joints_without_a_calibration_file_fail(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, calib_file=""))])

    assert code == EXIT_FAILED
    assert "FAIL  ValueError: physical model requires effective calibrated conversion authority" in out


def test_profile_without_home_poses_passes_but_says_nothing_was_checked(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, _calibration(), home_positions={}))])

    assert code == EXIT_OK
    assert "home within limits for joints: none" in out
    assert "NOTE  no home pose in the profile, not checked: 1, 2, 6" in out


def test_wheel_only_profile_needs_no_calibration(tmp_path):
    xacro = tmp_path / "wheels.urdf.xacro"
    xacro.write_text(
        """<?xml version="1.0"?>
<robot name="wheels" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base"/><link name="wheel_link"/>
  <joint name="wheel" type="continuous">
    <parent link="base"/><child link="wheel_link"/><axis xyz="0 1 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    path = _profile_path(
        tmp_path,
        calib_file="",
        joints=["wheel"],
        arm_joints=[],
        gripper_joints=[],
        base={"wheel_joints": ["wheel"], "max_wheel_radps": 10.0},
        home_positions={},
        motion={},
        capabilities={"joint.state": {"joint_count": 1, "rate_hz": 10.0}},
        description={"xacro": str(xacro)},
    )

    code, out = _run(["--profile", str(path)])

    assert code == EXIT_OK, out
    assert "PASS" in out


def test_public_interface_errors_fail_like_the_launch(tmp_path):
    # build_description rejects this declaration; a physical launch fails on it too.
    path = _profile_path(tmp_path, _calibration(), interfaces={"runtime.get_status": {}})

    code, out = _run(["--profile", str(path)])

    assert code == EXIT_FAILED
    assert "FAIL  InterfaceDescriptionError" in out
    # The joint table is still printed.
    assert "calibrated [rad]" in out


def test_non_numeric_home_pose_fails_without_a_traceback(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, _calibration(), home_positions={"1": "bad"}))])

    assert code == EXIT_FAILED
    assert "FAIL  TypeError" in out


def test_unreadable_profile_is_a_usage_error(tmp_path):
    code, out = _run(["--profile", str(tmp_path / "absent.yaml")])

    assert code == EXIT_USAGE
    assert out.startswith("ERROR: ProfileError: runtime profile not found")


def test_malformed_profile_yaml_is_a_usage_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("runtime: [\n", encoding="utf-8")

    code, out = _run(["--profile", str(path)])

    assert code == EXIT_USAGE
    assert out.startswith("ERROR: ")


@pytest.mark.parametrize(
    ("section", "value"),
    [("ros2_control", ["bad"]), ("command_channels", None)],
)
def test_profile_section_of_the_wrong_type_is_a_usage_error(tmp_path, section, value):
    path = _profile_path(tmp_path, _calibration(), **{section: value})

    code, out = _run(["--profile", str(path)])

    assert code == EXIT_USAGE
    assert out.startswith("ERROR: ")


def test_unknown_package_is_a_usage_error():
    code, out = _run(["--package", "no_such_package_for_preflight", "--profile", "no_such_profile"])

    assert code == EXIT_USAGE
    assert out.startswith("ERROR: ")


def test_negative_min_margin_is_a_usage_error(tmp_path):
    code, out = _run(["--profile", str(_profile_path(tmp_path, _calibration())), "--min-margin", "-1"])

    assert code == EXIT_USAGE
    assert "--min-margin" in out


# --- checked_physical_description: the same check inside a physical launch ---


def _launch_calls(path):
    """The launch's own render and describe calls on ``path``, counting how often each runs."""
    from robot_runtime.interface_description import build_description
    from robot_runtime.launch_support import render_robot_description
    from robot_runtime.profile import load_profile

    profile = load_profile(path)
    calls = {"render": 0, "describe": 0}

    def render():
        calls["render"] += 1
        return render_robot_description(profile, path, False)

    def describe(urdf):
        calls["describe"] += 1
        return build_description(profile, use_sim=False, robot_description=urdf, sim_platform=None, profile_path=path)

    return profile, render, describe, calls


def _collect():
    logged = []
    return logged, lambda level, text: logged.append((level, text))


def test_launch_check_passes_with_one_summary_line_and_runs_each_step_once(tmp_path):
    path = _profile_path(tmp_path, _calibration())
    profile, render, describe, calls = _launch_calls(path)
    logged, log = _collect()

    urdf, description = checked_physical_description(profile, path, render, describe, log=log)

    assert calls == {"render": 1, "describe": 1}
    assert "<robot" in urdf and description["model"]["home_positions"] == {"1": 0.0, "2": 1.2}
    assert len(logged) == 1
    level, text = logged[0]
    assert level == "info"
    assert "\n" not in text
    # Joint 2 is the tightest: calibrated upper end 1.5 rad, home 1.2 rad.
    assert text.startswith("calibration preflight: PASS, 2 home poses within limits, min margin 0.300 rad at joint 2")


def test_launch_check_logs_the_table_and_continues_on_a_small_margin(tmp_path):
    path = _profile_path(tmp_path, _calibration(joint2_max_rad=1.23))
    profile, render, describe, _calls = _launch_calls(path)
    logged, log = _collect()

    checked_physical_description(profile, path, render, describe, log=log)

    assert [level for level, _ in logged] == ["warning"]
    assert "calibrated [rad]" in logged[0][1]
    assert "WARN  joint 2: home +1.200 rad" in logged[0][1]


def test_launch_check_logs_the_table_and_reraises_the_launch_error(tmp_path):
    path = _profile_path(tmp_path, _calibration(joint2_max_rad=1.0))
    profile, render, describe, calls = _launch_calls(path)
    logged, log = _collect()
    original = []

    def describe_recording(urdf):
        try:
            return describe(urdf)
        except ValueError as exc:
            original.append(exc)
            raise

    with pytest.raises(ValueError, match=r"robot_model\.home_positions\.2 is outside physical limits") as raised:
        checked_physical_description(profile, path, render, describe_recording, log=log)

    # The launch's own exception, unchanged, after one render and one describe.
    assert raised.value is original[0]
    assert calls == {"render": 1, "describe": 1}
    assert [level for level, _ in logged] == ["error"]
    text = logged[0][1]
    assert "before any hardware node starts" in text
    assert "calibrated [rad]" in text
    assert "FAIL  ValueError: robot_model.home_positions.2 is outside physical limits" in text


def test_launch_check_reraises_a_render_failure_without_describing(tmp_path):
    path = _profile_path(tmp_path, calibration=None)
    profile, render, describe, calls = _launch_calls(path)
    logged, log = _collect()

    with pytest.raises(ValueError, match="calibration file not found"):
        checked_physical_description(profile, path, render, describe, log=log)

    assert calls == {"render": 1, "describe": 0}
    assert [level for level, _ in logged] == ["error"]
    assert "calibrated [rad]" not in logged[0][1]
    assert "FAIL  RuntimeProfileError: calibration file not found" in logged[0][1]


def test_a_failing_report_projection_never_replaces_the_launch_error(tmp_path):
    # Two faults at once: build_description rejects the peripheral entry, and the
    # best-effort projection for the report then chokes on a home pose too large
    # for a float. The launch must still fail with build_description's error.
    path = _profile_path(
        tmp_path, _calibration(), home_positions={"1": 0.0, "2": 10**400}, peripherals=[{"type": "bogus"}]
    )
    profile, render, describe, _calls = _launch_calls(path)
    original = []

    def describe_recording(urdf):
        try:
            return describe(urdf)
        except Exception as exc:
            original.append(exc)
            raise

    logged, log = _collect()
    with pytest.raises(Exception) as raised:
        checked_physical_description(profile, path, render, describe_recording, log=log)

    assert raised.value is original[0]
    assert type(raised.value).__name__ == "InterfaceDescriptionError"
    assert [level for level, _ in logged] == ["error"]
    assert "FAIL  InterfaceDescriptionError" in logged[0][1]


# --- review round on !603: calibration path fork, home overflow, --package lookup ---


def test_hardware_xacro_args_cannot_hand_the_plugin_another_calibration(tmp_path):
    # Two valid calibrations: the checked one (hardware.calib_file) excludes
    # nothing, the override would reach the hardware plugin through the URDF.
    # The render refuses the override, so no check can report on a file the
    # hardware does not use.
    (tmp_path / "other.json").write_text(json.dumps(_calibration(joint2_max_rad=1.0)), encoding="utf-8")
    path = _profile_path(
        tmp_path,
        _calibration(),
        hardware={
            "calib_file": str(tmp_path / "calib.json"),
            "xacro_args": {"calib_file": str(tmp_path / "other.json")},
        },
    )
    (tmp_path / "calib.json").write_text(json.dumps(_calibration()), encoding="utf-8")

    code, out = _run(["--profile", str(path)])
    assert code == EXIT_FAILED
    assert "FAIL  RuntimeProfileError: runtime profile 'hardware.xacro_args' must not set ['calib_file']" in out

    profile, render, describe, calls = _launch_calls(path)
    logged, log = _collect()
    with pytest.raises(ValueError, match=r"must not set \['calib_file'\]"):
        checked_physical_description(profile, path, render, describe, log=log)
    assert calls == {"render": 1, "describe": 0}
    assert [level for level, _ in logged] == ["error"]


def test_home_too_large_for_a_float_fails_with_the_report(tmp_path):
    path = _profile_path(tmp_path, _calibration(), home_positions={"1": 0.0, "2": 10**400})

    code, out = _run(["--profile", str(path)])
    assert code == EXIT_FAILED
    assert "FAIL  ValueError: home_positions.2 is too large to be a joint position" in out

    profile, render, describe, _calls = _launch_calls(path)
    logged, log = _collect()
    original = []

    def describe_recording(urdf):
        try:
            return describe(urdf)
        except Exception as exc:
            original.append(exc)
            raise

    with pytest.raises(ValueError, match=r"home_positions\.2 is too large") as raised:
        checked_physical_description(profile, path, render, describe_recording, log=log)
    assert raised.value is original[0]
    assert [level for level, _ in logged] == ["error"]
    assert "FAIL  ValueError: home_positions.2 is too large to be a joint position" in logged[0][1]


def test_package_mode_ignores_a_same_named_file_in_the_working_directory(tmp_path, monkeypatch):
    # A valid profile named like the requested one sits in the working directory;
    # robot_runtime ships no such profile, so package mode must not use it.
    local = _profile_path(tmp_path, _calibration())
    (tmp_path / "demo_profile").write_text(local.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code, out = _run(["--package", "robot_runtime", "--profile", "demo_profile"])

    assert code == EXIT_USAGE
    assert out.startswith("ERROR: FileNotFoundError: no profile 'demo_profile' in robot_runtime/profiles")
    assert "PASS" not in out


@pytest.mark.parametrize("name", ["{abs}", "../escape", "sub/name", ".."])
def test_package_mode_takes_a_name_not_a_path(tmp_path, name):
    # A valid profile outside the package must not be reachable through the name.
    outside = _profile_path(tmp_path, _calibration())
    name = name.format(abs=str(outside.with_suffix("")))

    code, out = _run(["--package", "robot_runtime", "--profile", name])

    assert code == EXIT_USAGE
    assert out.startswith("ERROR: ValueError: with --package, --profile must be a profile name, not a path")
    assert "PASS" not in out


def _wheel_profile(tmp_path, max_wheel_radps):
    xacro = tmp_path / "wheels.urdf.xacro"
    xacro.write_text(
        """<?xml version="1.0"?>
<robot name="wheels" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base"/><link name="wheel_link"/>
  <joint name="wheel" type="continuous">
    <parent link="base"/><child link="wheel_link"/><axis xyz="0 1 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return _profile_path(
        tmp_path,
        calib_file="",
        joints=["wheel"],
        arm_joints=[],
        gripper_joints=[],
        base={"wheel_joints": ["wheel"], "max_wheel_radps": max_wheel_radps},
        home_positions={},
        motion={},
        capabilities={"joint.state": {"joint_count": 1, "rate_hz": 10.0}},
        description={"xacro": str(xacro)},
    )


def test_wheel_speed_too_large_for_a_float_fails_with_the_report(tmp_path):
    path = _wheel_profile(tmp_path, 10**400)

    code, out = _run(["--profile", str(path)])
    assert code == EXIT_FAILED
    assert "FAIL  ValueError: base.max_wheel_radps is too large to be a wheel speed" in out

    profile, render, describe, _calls = _launch_calls(path)
    logged, log = _collect()
    with pytest.raises(ValueError, match=r"base\.max_wheel_radps is too large"):
        checked_physical_description(profile, path, render, describe, log=log)
    assert [level for level, _ in logged] == ["error"]


def test_an_unnamed_overflow_still_gets_the_fail_report(tmp_path):
    # Any OverflowError a step raises is reported like the other check errors
    # and re-raised unchanged.
    path = _profile_path(tmp_path, _calibration())
    profile, render, _describe, _calls = _launch_calls(path)
    original = OverflowError("synthetic overflow")

    def describe(_urdf):
        raise original

    logged, log = _collect()
    with pytest.raises(OverflowError) as raised:
        checked_physical_description(profile, path, render, describe, log=log)
    assert raised.value is original
    assert [level for level, _ in logged] == ["error"]
    assert "FAIL  OverflowError: synthetic overflow" in logged[0][1]
