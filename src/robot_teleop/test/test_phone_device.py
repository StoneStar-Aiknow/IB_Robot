from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_teleop.phone.phone_device import (
    _WEB_WORLD_TO_BACKEND,
    PhoneDevice,
    compute_webphone_pose_delta,
)
from robot_teleop.phone.web_phone import WebPhone
from robot_teleop.vr_rotation import remap_base_rotation


class _Servo:
    def __init__(self):
        self.is_enabled = True
        self.max_linear_speed = 1.0
        self.max_angular_speed = 2.0
        self.disabled = 0
        self.pose_commands = []
        self.enable_calls = 0
        self.home_result = None
        self.stop_pending = False
        self.keepalive_calls = 0

    def enable(self):
        self.enable_calls += 1
        return True

    def disable(self):
        self.is_enabled = False
        self.disabled += 1
        return True

    def servo(self, linear, angular):
        self.last_twist = (linear, angular)

    def servo_pose(self, position, orientation):
        self.pose_commands.append((position, orientation))

    def home(self):
        self.is_enabled = False
        return True

    def consume_home_result(self):
        result = self.home_result
        self.home_result = None
        return result

    def keepalive(self):
        self.keepalive_calls += 1


def _device():
    device = PhoneDevice(
        {
            "control_frequency": 50.0,
            "phone_config": {
                "backend": "webphone",
                "web": {"tls": {"enabled": False}},
            },
        }
    )
    device.servo_client = _Servo()
    return device


def _pose_device():
    device = PhoneDevice(
        {
            "control_frequency": 50.0,
            "phone_config": {
                "backend": "webphone",
                "position_scale": 0.7,
                "web": {"tls": {"enabled": False}},
            },
        }
    )
    device.servo_client = _Servo()
    device.servo_client.is_enabled = False
    device._first_state_received = True
    return device


def _action(enabled, position=None, linear=None, angular=None, tracking_mode="ar_6dof"):
    return {
        "phone.enabled": enabled,
        "phone.pos": np.asarray(position or [0.0, 0.0, 0.0]),
        "phone.rot": Rotation.identity(),
        "phone.linear_vel": np.asarray(linear or [0.0, 0.0, 0.0]),
        "phone.angular_vel": np.asarray(angular or [0.0, 0.0, 0.0]),
        "phone.tracking_mode": tracking_mode,
        "phone.raw_inputs": {"move": enabled, "scale": 1.0},
    }


def test_phone_device_uses_ros_logger_when_available():
    ros_logger = object()
    device = PhoneDevice(
        {"phone_config": {"backend": "webphone", "web": {"tls": {"enabled": False}}}},
        node=SimpleNamespace(get_logger=lambda: ros_logger),
    )

    assert device.logger is ros_logger


def test_webphone_rejects_stop_request_latency_above_safety_bound():
    with pytest.raises(ValueError, match="before a stop request is issued"):
        PhoneDevice(
            {
                "control_frequency": 30.0,
                "phone_config": {
                    "backend": "webphone",
                    "web": {"command_stale_s": 0.2, "tls": {"enabled": False}},
                },
            }
        )


@pytest.mark.parametrize("control_frequency", [True, 0.0, float("nan"), float("inf"), float("-inf")])
def test_phone_device_rejects_invalid_control_frequency(control_frequency):
    with pytest.raises(ValueError, match="finite and positive"):
        PhoneDevice(
            {
                "control_frequency": control_frequency,
                "phone_config": {
                    "backend": "webphone",
                    "web": {"tls": {"enabled": False}},
                },
            }
        )


def test_enabled_action_preserves_absolute_pose_until_placo_latches_baseline():
    command = _device()._compute_cartesian_command(_action(True))
    assert command.enabled is True
    assert np.allclose(command.pose_position, np.zeros(3))
    assert np.allclose(command.pose_rotation.as_quat(), Rotation.identity().as_quat())


def test_webphone_pose_delta_uses_arcore_base_frame_rotation_contract():
    clutch = Rotation.from_euler("zyx", [0.6, -0.2, 0.1])
    world_delta = Rotation.from_euler("x", 0.25)
    current = world_delta * clutch
    position, rotation = compute_webphone_pose_delta(
        np.array([0.1, 0.2, 0.3]),
        current,
        np.zeros(3),
        clutch,
        position_scale=0.7,
        angular_scale=1.0,
        user_scale=1.0,
    )
    assert np.allclose(position, _WEB_WORLD_TO_BACKEND @ np.array([0.07, 0.14, 0.21]))
    expected = remap_base_rotation(world_delta, _WEB_WORLD_TO_BACKEND)
    assert np.allclose(rotation.as_matrix(), expected.as_matrix())


def test_webphone_mapping_uses_one_basis_for_position_and_rotation():
    position, rotation = compute_webphone_pose_delta(
        np.array([0.1, 0.2, -0.3]),
        Rotation.from_euler("x", 20.0, degrees=True),
        np.zeros(3),
        Rotation.identity(),
        position_scale=1.0,
        angular_scale=1.0,
        user_scale=1.0,
    )

    assert np.allclose(position, [0.3, -0.1, 0.2])
    assert np.allclose(rotation.as_rotvec(degrees=True), [0.0, -20.0, 0.0], atol=1e-7)


def test_webphone_translation_maps_control_axes_to_backend_axes():
    forward, _ = compute_webphone_pose_delta(
        np.array([0.0, 0.0, -0.1]),
        Rotation.identity(),
        np.zeros(3),
        Rotation.identity(),
        position_scale=1.0,
        angular_scale=1.0,
        user_scale=1.0,
    )
    right, _ = compute_webphone_pose_delta(
        np.array([0.1, 0.0, 0.0]),
        Rotation.identity(),
        np.zeros(3),
        Rotation.identity(),
        position_scale=1.0,
        angular_scale=1.0,
        user_scale=1.0,
    )

    assert np.allclose(forward, [0.1, 0.0, 0.0])
    assert np.allclose(right, [0.0, -0.1, 0.0])


def test_webphone_world_z_rotation_maps_to_base_x_after_rolled_reclutch():
    clutch = Rotation.from_euler("x", 90.0, degrees=True)
    world_yaw = Rotation.from_euler("z", 20.0, degrees=True)

    _, rotation = compute_webphone_pose_delta(
        np.zeros(3),
        world_yaw * clutch,
        np.zeros(3),
        clutch,
        position_scale=1.0,
        angular_scale=1.0,
        user_scale=1.0,
    )

    expected = remap_base_rotation(world_yaw, _WEB_WORLD_TO_BACKEND)
    assert np.allclose(rotation.as_matrix(), expected.as_matrix())
    assert np.allclose(rotation.as_rotvec(degrees=True), [-20.0, 0.0, 0.0], atol=1e-7)


def test_pose_waits_for_start_then_latches_zero_command():
    device = _pose_device()
    command = device._compute_cartesian_command(_action(True, position=[0.0, 0.0, 0.0]))
    device._get_cmd_internal = lambda: command

    device.get_joint_targets()
    assert device.servo_client.enable_calls == 1
    assert device.servo_client.pose_commands == []

    device.servo_client.is_enabled = True
    command = device._compute_cartesian_command(_action(True, position=[0.1, 0.0, 0.0]))
    device.get_joint_targets()
    assert device.servo_client.pose_commands == [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))]

    command = device._compute_cartesian_command(_action(True, position=[0.11, 0.0, 0.0]))
    device.get_joint_targets()
    position, orientation = device.servo_client.pose_commands[-1]
    assert np.allclose(position, [0.0, -0.007, 0.0])
    assert np.allclose(orientation, [0.0, 0.0, 0.0, 1.0])


def test_pose_release_stops_backend_and_clears_clutch():
    device = _pose_device()
    device._servo_enabled = True
    device.servo_client.is_enabled = True
    device._pose_clutch_pos = np.zeros(3)
    command = device._compute_cartesian_command(_action(False))
    device._get_cmd_internal = lambda: command

    device.get_joint_targets()

    assert device.servo_client.disabled == 1
    assert device._servo_enabled is False
    assert device._pose_clutch_pos is None


def test_pose_output_is_rate_limited_per_control_cycle():
    device = _pose_device()
    position, rotation = device._limit_pose_step(
        np.array([1.0, 0.0, 0.0]),
        Rotation.from_rotvec([0.0, 1.0, 0.0]),
    )
    assert np.isclose(np.linalg.norm(position), device.phone_config.max_ee_step_m)
    assert np.isclose(np.linalg.norm(rotation.as_rotvec()), device.phone_config.max_angular_step_rad)


def test_pose_target_is_clamped_to_configured_relative_workspace():
    device = PhoneDevice(
        {
            "control_frequency": 50.0,
            "phone_config": {
                "backend": "webphone",
                "position_scale": 1.0,
                "end_effector_bounds": {"min": [-0.1, -0.2, -0.3], "max": [0.1, 0.2, 0.3]},
                "max_ee_step_m": 1.0,
                "web": {"tls": {"enabled": False}},
            },
        }
    )
    device.servo_client = _Servo()
    device._servo_enabled = True
    device._first_state_received = True
    device._pose_clutch_pos = np.zeros(3)
    device._pose_clutch_rot = Rotation.identity()
    command = device._compute_cartesian_command(_action(True, position=[1.0, 1.0, 1.0]))
    device._get_cmd_internal = lambda: command

    device.get_joint_targets()

    position, _ = device.servo_client.pose_commands[-1]
    assert np.allclose(position, [-0.1, -0.2, 0.3])


def test_pose_orientation_filter_masks_yaw_and_smooths_roll_pitch():
    device = PhoneDevice(
        {
            "control_frequency": 50.0,
            "phone_config": {
                "backend": "webphone",
                "orientation_axis_mask": [1.0, 1.0, 0.0],
                "orientation_deadzone_rad": 0.05,
                "orientation_filter_alpha": 0.5,
                "web": {"tls": {"enabled": False}},
            },
        }
    )

    first = device._filter_pose_rotation(Rotation.from_rotvec([0.2, 0.1, 0.3])).as_rotvec()
    second = device._filter_pose_rotation(Rotation.from_rotvec([0.2, 0.1, 0.3])).as_rotvec()

    assert np.allclose(first, [0.1, 0.05, 0.0])
    assert np.allclose(second, [0.15, 0.075, 0.0])


def test_default_pose_orientation_filter_preserves_all_axes():
    device = _device()

    filtered = device._filter_pose_rotation(Rotation.from_rotvec([0.2, 0.1, 0.3])).as_rotvec()

    assert np.allclose(filtered, [0.2, 0.1, 0.3])


def test_webphone_rotation_axes_follow_position_basis():
    device = _device()

    mapped = []
    for axis in "xyz":
        _, rotation = compute_webphone_pose_delta(
            np.zeros(3),
            Rotation.from_euler(axis, 20.0, degrees=True),
            np.zeros(3),
            Rotation.identity(),
            position_scale=1.0,
            angular_scale=1.0,
            user_scale=1.0,
        )
        device._pose_filtered_rotvec = np.zeros(3)
        mapped.append(device._filter_pose_rotation(rotation).as_rotvec(degrees=True))

    assert np.allclose(mapped[0], [0.0, -20.0, 0.0], atol=1e-7)
    assert np.allclose(mapped[1], [0.0, 0.0, 20.0], atol=1e-7)
    assert np.allclose(mapped[2], [-20.0, 0.0, 0.0], atol=1e-7)


def test_pose_mode_uses_optical_flow_virtual_pose_without_reintegration():
    device = PhoneDevice(
        {
            "control_frequency": 50.0,
            "phone_config": {
                "backend": "webphone",
                "optical_flow_fallback_enabled": True,
                "web": {"tls": {"enabled": False}},
            },
        }
    )

    first = device._compute_cartesian_command(
        _action(True, position=[0.02, -0.04, 0.06], linear=[9.0, 9.0, 9.0], tracking_mode="optical_flow")
    )
    second = device._compute_cartesian_command(
        _action(True, position=[0.02, -0.04, 0.06], linear=[9.0, 9.0, 9.0], tracking_mode="optical_flow")
    )

    assert np.allclose(first.pose_position, [0.02, -0.04, 0.06])
    assert np.allclose(second.pose_position, first.pose_position)


def test_pose_motion_remains_active_while_closing_gripper():
    device = _pose_device()
    device._last_gripper_pos = 0.5
    action = _action(True, position=[0.02, -0.04, 0.06], tracking_mode="optical_flow")
    action["phone.raw_inputs"]["reservedButtonB"] = True

    command = device._compute_cartesian_command(action)

    assert command.enabled is True
    assert np.allclose(command.pose_position, [0.02, -0.04, 0.06])
    assert command.gripper_pos < 0.5


def test_webphone_ignores_legacy_velocity_fields():
    device = _device()
    command = device._compute_cartesian_command(
        _action(True, position=[0.01, 0.0, 0.0], linear=[0.1, 0.2, 0.3], angular=[0.4, 0.5, 0.6])
    )
    assert np.allclose(command.pose_position, [0.01, 0.0, 0.0])


def test_transport_stop_disables_servo_and_clears_device_state():
    device = _device()
    phone = WebPhone(device.phone_config)
    phone.require_release("network lost")
    device._phone_impl = phone
    device._pose_clutch_pos = np.ones(3)
    device._pose_clutch_rot = Rotation.identity()
    device._servo_enabled = True

    device._consume_transport_stop()

    assert device.servo_client.disabled == 1
    assert device._servo_enabled is False
    assert device._pose_clutch_pos is None
    assert device._pose_clutch_rot is None


def test_transport_stop_forces_disable_while_backend_reports_home_inactive():
    device = _device()
    phone = WebPhone(device.phone_config)
    phone.require_release("network lost")
    device._phone_impl = phone
    device._going_home = True
    device._servo_enabled = False
    device.servo_client.is_enabled = False

    device._consume_transport_stop()

    assert device.servo_client.disabled == 1
    assert device._going_home is False


def test_emergency_stop_forces_disable_during_home():
    device = _device()
    device._going_home = True
    device._servo_enabled = False
    device.servo_client.is_enabled = False

    device.emergency_stop()

    assert device.servo_client.disabled == 1
    assert device._going_home is False


def test_shutdown_complete_tracks_pending_placo_stop():
    device = _device()
    device.servo_client.stop_pending = True

    assert device.shutdown_complete is False

    device.servo_client.stop_pending = False
    assert device.shutdown_complete is True


def test_go_home_waits_for_backend_terminal_status():
    device = _device()
    device._going_home = True
    device._deadman_release_required = True

    targets = device._update_home_state()

    assert targets == {device.gripper_joint_names[0]: device._last_gripper_pos}
    assert device._going_home is True
    device.servo_client.home_result = True
    targets = device._update_home_state()

    assert targets == {device.gripper_joint_names[0]: device._last_gripper_pos}
    assert device._going_home is False
    assert device._servo_enabled is False
    assert device._deadman_release_required is True
    assert device.servo_client.disabled == 0


def test_go_home_abort_keeps_motion_released():
    device = _device()
    device._going_home = True
    device._deadman_release_required = True
    device.servo_client.home_result = False

    targets = device._update_home_state()

    assert targets == {device.gripper_joint_names[0]: device._last_gripper_pos}
    assert device._going_home is False
    assert device._deadman_release_required is True
    assert device.servo_client.disabled == 1


def test_phone_device_rejects_moveit_servo_when_launch_validation_is_bypassed():
    with pytest.raises(ValueError, match="requires cartesian_solver=placo_servo"):
        PhoneDevice(
            {
                "cartesian_solver": "moveit_servo",
                "control_frequency": 50.0,
                "phone_config": {
                    "backend": "webphone",
                    "web": {"tls": {"enabled": False}},
                },
            }
        )


def test_deadman_release_latch_requires_real_webphone_release():
    device = _device()
    device._require_deadman_release("home completed", request_transport_stop=False)

    held = device._compute_cartesian_command(_action(True))
    released = device._compute_cartesian_command(_action(False))
    pressed_again = device._compute_cartesian_command(_action(True))

    assert held.enabled is False
    assert released.enabled is False
    assert pressed_again.enabled is True


def test_keepalive_pauses_while_stop_is_pending():
    device = _device()
    command = device._compute_cartesian_command(_action(True))
    device._get_cmd_internal = lambda: command
    device.get_joint_targets()
    assert device.servo_client.keepalive_calls == 1

    device.servo_client.stop_pending = True
    device.get_joint_targets()

    assert device.servo_client.keepalive_calls == 1


def test_valid_phone_command_refreshes_command_lease():
    device = _device()
    command = device._compute_cartesian_command(_action(True))
    device._get_cmd_internal = lambda: command

    device.get_joint_targets()

    assert device.servo_client.keepalive_calls == 1
    assert device.servo_client.disabled == 0


def test_active_backend_fails_closed_when_phone_returns_empty_action():
    device = _device()
    device._is_connected = True
    device._phone_impl = WebPhone(device.phone_config)
    device._phone_impl.get_action = lambda: {}
    device._servo_enabled = True

    assert device.get_joint_targets() == {}

    assert device.servo_client.keepalive_calls == 0
    assert device.servo_client.disabled == 1
    assert device._servo_enabled is False
    assert device._deadman_release_required is True


def test_active_backend_fails_closed_when_phone_input_raises():
    device = _device()
    device._is_connected = True

    def raise_input_error():
        raise RuntimeError("transport failure")

    device._phone_impl = WebPhone(device.phone_config)
    device._phone_impl.get_action = raise_input_error
    device._servo_enabled = True

    assert device.get_joint_targets() == {}

    assert device.servo_client.keepalive_calls == 0
    assert device.servo_client.disabled == 1
    assert device._deadman_release_required is True


def test_inactive_backend_allows_missing_initial_phone_frame_without_disable():
    device = _device()
    device.servo_client.is_enabled = False
    device._is_connected = True
    device._phone_impl = WebPhone(device.phone_config)
    device._phone_impl.get_action = lambda: {}

    assert device.get_joint_targets() == {}

    assert device.servo_client.keepalive_calls == 0
    assert device.servo_client.disabled == 0
    assert device._deadman_release_required is False


def test_home_refreshes_command_lease_without_phone_frame():
    device = _device()
    device._going_home = True

    targets = device.get_joint_targets()

    assert targets == {device.gripper_joint_names[0]: device._last_gripper_pos}
    assert device.servo_client.keepalive_calls == 1
