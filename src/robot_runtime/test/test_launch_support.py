"""Launch-support unit tests: transport selection (runtime-simulation-transport D1)."""

import pytest

from robot_runtime.launch_support import RuntimeProfileError, resolve_transport


def _profile(**extra):
    profile = {
        "runtime": {"name": "test_robot", "version": "0.1.0"},
        "simulated": False,
        "transports": ["hardware", "sdk"],
        "transport": "hardware",
    }
    profile.update(extra)
    return profile


def test_no_selection_uses_profile_default():
    assert resolve_transport(_profile(), "", "") == "hardware"
    assert resolve_transport(_profile(simulated=True, transport="sdk"), "", "") == "sdk"


def test_simulated_alias_maps_to_sdk_and_hardware():
    profile = _profile()
    assert resolve_transport(profile, "", "true") == "sdk"
    assert resolve_transport(profile, "", "false") == "hardware"
    assert resolve_transport(profile, "", "1") == "sdk"
    assert resolve_transport(profile, "", "no") == "hardware"


def test_explicit_transport_overrides_profile_default():
    assert resolve_transport(_profile(), "sdk", "") == "sdk"
    assert resolve_transport(_profile(simulated=True, transport="sdk"), "hardware", "") == "hardware"


def test_agreeing_transport_and_simulated_selection_is_accepted():
    assert resolve_transport(_profile(), "sdk", "true") == "sdk"
    assert resolve_transport(_profile(), "hardware", "false") == "hardware"


def test_conflicting_transport_and_simulated_selection_fails_naming_both():
    with pytest.raises(RuntimeProfileError, match=r"transport:='gazebo'.*simulated:='false'"):
        resolve_transport(_profile(transports=["hardware", "sdk", "gazebo"]), "gazebo", "false")
    with pytest.raises(RuntimeProfileError, match="pass at most one"):
        resolve_transport(_profile(), "hardware", "true")


def test_unknown_transport_value_fails():
    with pytest.raises(RuntimeProfileError, match="unknown transport 'teleport'"):
        resolve_transport(_profile(), "teleport", "")


def test_unparseable_simulated_value_fails():
    with pytest.raises(RuntimeProfileError, match="cannot parse simulated:='banana'"):
        resolve_transport(_profile(), "", "banana")


def test_undeclared_transport_fails_naming_transport_set_and_profile():
    profile = _profile()
    with pytest.raises(
        RuntimeProfileError,
        match=r"transport 'gazebo' is not supported.*supported: \['hardware', 'sdk'\].*profile: /tmp/p.yaml",
    ):
        resolve_transport(profile, "gazebo", "", source="/tmp/p.yaml")


def test_declared_gazebo_transport_resolves():
    profile = _profile(transports=["hardware", "sdk", "gazebo"])
    assert resolve_transport(profile, "gazebo", "") == "gazebo"


def test_profile_without_transport_keys_keeps_legacy_behaviour():
    profile = {"runtime": {"name": "test_robot", "version": "0.1.0"}, "simulated": True}
    assert resolve_transport(profile, "", "") == "sdk"
    assert resolve_transport(profile, "", "false") == "hardware"
