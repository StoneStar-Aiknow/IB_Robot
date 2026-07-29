import pytest

from robot_teleop.phone.config_phone import PhoneConfig


def test_backend_defaults_to_webphone_and_ignores_legacy_phone_os():
    assert PhoneConfig.from_dict({"web": {}}).backend == "webphone"
    assert PhoneConfig.from_dict({"phone_os": "ios", "web": {}}).backend == "webphone"
    assert PhoneConfig.from_dict({"phone_os": "android", "web": {}}).backend == "webphone"


def test_explicit_webphone_backend_is_accepted_with_legacy_phone_os():
    config = PhoneConfig.from_dict({"backend": "webphone", "phone_os": "ios", "web": {}})
    assert config.backend == "webphone"


def test_removed_hebi_backend_is_rejected():
    with pytest.raises(ValueError, match="must be 'webphone'"):
        PhoneConfig.from_dict({"backend": "hebi", "web": {}})


def test_webphone_uses_a_fixed_cartesian_contract():
    config = PhoneConfig.from_dict({"backend": "webphone", "web": {}})
    assert config.cartesian_input_mode == "pose"
    assert config.web.ar_enabled is True
    assert config.optical_flow_fallback_enabled is True
    assert config.web.command_stale_s == 0.18
    assert config.end_effector_bounds["min"] == [-0.5, -0.5, -0.5]
    assert config.orientation_axis_mask.tolist() == [1.0, 1.0, 1.0]


def test_webphone_pose_fields_round_trip():
    config = PhoneConfig.from_dict(
        {
            "backend": "webphone",
            "position_scale": 0.8,
            "optical_flow_fallback_enabled": True,
            "orientation_axis_mask": [1.0, 0.5, 0.0],
            "orientation_deadzone_rad": 0.02,
            "orientation_filter_alpha": 0.2,
            "web": {},
        }
    )
    serialized = config.to_dict()
    assert config.cartesian_input_mode == "pose"
    assert "input_mode" not in serialized
    assert serialized["position_scale"] == 0.8
    assert serialized["optical_flow_fallback_enabled"] is True
    assert serialized["orientation_axis_mask"] == [1.0, 0.5, 0.0]
    assert serialized["orientation_deadzone_rad"] == 0.02
    assert serialized["orientation_filter_alpha"] == 0.2


def test_matching_legacy_pose_mode_is_accepted():
    config = PhoneConfig.from_dict({"backend": "webphone", "input_mode": "pose", "web": {}})

    assert config.cartesian_input_mode == "pose"


def test_webphone_velocity_mode_is_rejected():
    with pytest.raises(ValueError, match="no longer supported"):
        PhoneConfig.from_dict({"backend": "webphone", "input_mode": "velocity", "web": {}})


@pytest.mark.parametrize(
    "web_config",
    [
        {"http_port": 0},
        {"websocket_port": 65536},
        {"http_port": 9000, "websocket_port": 9000},
        {"command_stale_s": 0},
        {"command_stale_s": float("nan")},
        {"command_stale_s": float("inf")},
        {"command_stale_s": float("-inf")},
        {"tls": {"enabled": True, "cert_file": "cert.pem", "key_file": ""}},
    ],
)
def test_invalid_webphone_config_is_rejected(web_config):
    with pytest.raises(ValueError):
        PhoneConfig.from_dict({"backend": "webphone", "web": web_config})


def test_config_round_trip_keeps_public_web_fields():
    config = PhoneConfig.from_dict(
        {
            "backend": "webphone",
            "phone_os": "android",
            "angular_scale": 1.5,
            "web": {
                "bind_address": "127.0.0.1",
                "http_port": 18765,
                "websocket_port": 18766,
                "command_stale_s": 0.3,
                "tls": {"enabled": False},
            },
        }
    )

    serialized = config.to_dict()
    assert serialized["backend"] == "webphone"
    assert serialized["angular_scale"] == 1.5
    assert serialized["web"]["command_stale_s"] == 0.3
    assert serialized["web"]["tls"]["enabled"] is False


def test_string_false_configuration_is_not_treated_as_true():
    config = PhoneConfig.from_dict(
        {
            "backend": "webphone",
            "optical_flow_fallback_enabled": "false",
            "web": {
                "ar_enabled": "true",
                "binary_protocol_enabled": "false",
                "tls": {"enabled": "false", "allow_insecure_http": "false"},
            },
        }
    )

    assert config.optical_flow_fallback_enabled is False
    assert config.web.ar_enabled is True
    assert config.web.binary_protocol_enabled is False
    assert config.web.tls.enabled is False
    assert config.web.tls.allow_insecure_http is False


def test_invalid_boolean_configuration_is_rejected():
    with pytest.raises(ValueError, match="must be a boolean"):
        PhoneConfig.from_dict({"backend": "webphone", "optical_flow_fallback_enabled": "sometimes", "web": {}})


def test_webphone_rejects_configuration_without_tracking_source():
    with pytest.raises(ValueError, match="requires WebXR AR or optical-flow fallback"):
        PhoneConfig.from_dict(
            {
                "backend": "webphone",
                "optical_flow_fallback_enabled": False,
                "web": {"ar_enabled": False, "tls": {"enabled": False}},
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_ee_step_m", 0.0),
        ("max_ee_step_m", float("nan")),
        ("max_ee_step_m", float("inf")),
        ("max_angular_step_rad", float("-inf")),
    ],
)
def test_invalid_motion_limits_are_rejected(field, value):
    with pytest.raises(ValueError, match="step limits"):
        PhoneConfig.from_dict({field: value, "web": {"tls": {"enabled": False}}})


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_gripper_speed_factor_is_rejected(value):
    with pytest.raises(ValueError, match="gripper_speed_factor"):
        PhoneConfig.from_dict({"gripper_speed_factor": value, "web": {"tls": {"enabled": False}}})


@pytest.mark.parametrize(
    "gripper_range",
    [
        [1.0, 0.0],
        [float("nan"), 1.0],
        [0.0, float("inf")],
        [float("-inf"), 1.0],
    ],
)
def test_invalid_gripper_range_is_rejected(gripper_range):
    with pytest.raises(ValueError, match="gripper_range"):
        PhoneConfig.from_dict({"gripper_range": gripper_range, "web": {"tls": {"enabled": False}}})


@pytest.mark.parametrize(
    "bounds",
    [
        {"min": [0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        {"min": [0.0, 0.0, 0.0], "max": [0.0, 1.0, 1.0]},
        {"min": [0.0, 0.0, float("nan")], "max": [1.0, 1.0, 1.0]},
    ],
)
def test_invalid_end_effector_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="end_effector_bounds"):
        PhoneConfig.from_dict({"end_effector_bounds": bounds, "web": {"tls": {"enabled": False}}})


def test_phone_requires_zero_strictly_inside_each_relative_bound_axis():
    with pytest.raises(ValueError, match="contain zero strictly inside"):
        PhoneConfig.from_dict(
            {
                "backend": "webphone",
                "end_effector_bounds": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.5]},
                "web": {"tls": {"enabled": False}},
            }
        )


@pytest.mark.parametrize(
    "phone_config",
    [
        {"orientation_axis_mask": [1.0, 0.0]},
        {"orientation_axis_mask": [1.0, 1.1, 0.0]},
        {"orientation_deadzone_rad": -0.1},
        {"orientation_filter_alpha": 0.0},
        {"orientation_filter_alpha": 1.1},
    ],
)
def test_invalid_orientation_filter_config_is_rejected(phone_config):
    with pytest.raises(ValueError):
        PhoneConfig.from_dict({**phone_config, "web": {"tls": {"enabled": False}}})
