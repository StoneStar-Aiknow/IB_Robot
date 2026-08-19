import copy
import hashlib
import itertools
from pathlib import Path

import pytest
import yaml

from robot_navigation.cmd_vel_bridge_node import _body_to_wheel_radps

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "config"
ROBOT_CONFIG = WORKSPACE_ROOT / "src" / "robot_config" / "config" / "robots" / "lekiwi_realsense_navigation.yaml"

MPPI_PROFILES = [CONFIG_ROOT / "nav2_params.yaml", CONFIG_ROOT / "nav2_sim_params.yaml"]
DWB_PROFILES = [CONFIG_ROOT / "nav2_params_dwb.yaml", CONFIG_ROOT / "nav2_sim_params_dwb.yaml"]
REQUIRED_CRITICS = {
    "ConstraintCritic",
    "CostCritic",
    "GoalCritic",
    "GoalAngleCritic",
    "PathAlignCritic",
    "PathFollowCritic",
    "PathAngleCritic",
}


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _follow_path(profile):
    return profile["controller_server"]["ros__parameters"]["FollowPath"]


def _envelope():
    robot = _load(ROBOT_CONFIG)["robot"]
    return robot["navigation"]["motion_envelope"], robot["navigation"]["cmd_vel_bridge"]


def _strip_controller_choices(profile):
    result = copy.deepcopy(profile)
    params = result["controller_server"]["ros__parameters"]
    params.pop("FollowPath")
    smoother = result["velocity_smoother"]["ros__parameters"]
    for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
        smoother.pop(key)
    return result


@pytest.mark.parametrize("path", MPPI_PROFILES)
def test_mppi_profiles_use_humble_omni_contract(path):
    profile = _load(path)
    controller = _follow_path(profile)

    assert profile["controller_server"]["ros__parameters"]["controller_frequency"] == 20.0
    assert controller["plugin"] == "nav2_mppi_controller::MPPIController"
    assert controller["motion_model"] == "Omni"
    assert controller["model_dt"] == 0.05
    assert controller["iteration_count"] == 1
    assert controller["retry_attempt_limit"] == 1
    assert controller["visualize"] is False
    assert set(controller["critics"]) == REQUIRED_CRITICS
    assert "PreferForwardCritic" not in controller["critics"]
    assert controller["CostCritic"]["consider_footprint"] is False


@pytest.mark.parametrize("path", DWB_PROFILES)
def test_fallback_profiles_retain_dwb(path):
    assert _follow_path(_load(path))["plugin"] == "dwb_core::DWBLocalPlanner"


def test_dwb_fallback_hashes_match_recorded_baseline():
    regression = _load(CONFIG_ROOT / "nav2" / "controller_regression.yaml")
    for key, path in zip(("real", "simulation"), DWB_PROFILES, strict=True):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == regression["baseline"][key]["sha256"]


def test_real_profile_mirrors_robot_motion_envelope():
    envelope, _ = _envelope()
    controller = _follow_path(_load(MPPI_PROFILES[0]))
    smoother = _load(MPPI_PROFILES[0])["velocity_smoother"]["ros__parameters"]

    velocity = envelope["velocity"]
    acceleration = envelope["acceleration"]
    assert controller["vx_min"] == velocity["vx"]["min"]
    assert controller["vx_max"] == velocity["vx"]["max"]
    assert controller["vy_max"] == velocity["vy"]["max"]
    assert controller["wz_max"] == velocity["wz"]["max"]
    assert controller["ax_min"] == acceleration["vx"]["min"]
    assert controller["ax_max"] == acceleration["vx"]["max"]
    assert controller["ay_max"] == acceleration["vy"]["max"]
    assert controller["az_max"] == acceleration["wz"]["max"]
    assert smoother["min_velocity"] == [velocity[axis]["min"] for axis in ("vx", "vy", "wz")]
    assert smoother["max_velocity"] == [velocity[axis]["max"] for axis in ("vx", "vy", "wz")]
    assert smoother["max_decel"] == [acceleration[axis]["min"] for axis in ("vx", "vy", "wz")]
    assert smoother["max_accel"] == [acceleration[axis]["max"] for axis in ("vx", "vy", "wz")]


def test_simulation_limits_are_conservative_derivative_of_ssot():
    envelope, _ = _envelope()
    controller = _follow_path(_load(MPPI_PROFILES[1]))
    smoother = _load(MPPI_PROFILES[1])["velocity_smoother"]["ros__parameters"]

    for axis, key in (("vx", "vx_max"), ("vy", "vy_max"), ("wz", "wz_max")):
        assert 0 < controller[key] <= envelope["velocity"][axis]["max"]
    assert smoother["max_velocity"] == [controller["vx_max"], controller["vy_max"], controller["wz_max"]]
    assert smoother["min_velocity"] == [-value for value in smoother["max_velocity"]]


def _assert_wheel_feasible(envelope, bridge):
    velocity = envelope["velocity"]
    for vx, vy, wz in itertools.product(
        (velocity["vx"]["min"], velocity["vx"]["max"]),
        (velocity["vy"]["min"], velocity["vy"]["max"]),
        (velocity["wz"]["min"], velocity["wz"]["max"]),
    ):
        wheel_radps = _body_to_wheel_radps(
            vx,
            vy,
            wz,
            bridge["wheel_radius"],
            bridge["base_radius"],
            float("inf"),
        )
        assert max(abs(value) for value in wheel_radps) <= bridge["max_radps"] + 1e-9


def test_robot_motion_envelope_is_feasible_in_wheel_space():
    _assert_wheel_feasible(*_envelope())


def test_uncoordinated_angular_limit_increase_is_rejected():
    envelope, bridge = _envelope()
    unsafe = copy.deepcopy(envelope)
    unsafe["velocity"]["wz"] = {"min": -0.6, "max": 0.6}

    with pytest.raises(AssertionError):
        _assert_wheel_feasible(unsafe, bridge)


@pytest.mark.parametrize("active,fallback", list(zip(MPPI_PROFILES, DWB_PROFILES, strict=True)))
def test_controller_profiles_preserve_unrelated_nav2_contract(active, fallback):
    assert _strip_controller_choices(_load(active)) == _strip_controller_choices(_load(fallback))


def test_robot_config_selects_supported_production_profile():
    robot = _load(ROBOT_CONFIG)["robot"]
    params_file = robot["navigation"]["nav2_bringup"]["params_file"]
    assert params_file == "$(find robot_navigation)/config/nav2_params_dwb.yaml"
