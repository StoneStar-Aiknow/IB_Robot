"""Unit tests for the initial-position injection's fail-closed validation
(lekiwi-nav-grasp-sim-platforms D4, review rework).

The injection runs on the Gazebo/MuJoCo render paths only; these tests pin
the validation rules directly: finite values, every reset joint present in
the ros2_control block with a bare position state interface, conflicts
refused, no-reset profiles byte-identical.
"""

from __future__ import annotations

import math

import pytest

from robot_runtime.launch_support import RuntimeProfileError, _inject_initial_values

_URDF = """<robot name="demo">
  <transmission name="trans_1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
  </transmission>
  <ros2_control name="demo">
    <hardware>
      <plugin>demo/DemoSystem</plugin>
    </hardware>
    <joint name="1">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="2">
      <command_interface name="position"/>
      <state_interface name="position"/>
    </joint>
    <joint name="7">
      <command_interface name="velocity"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
  <link name="base"/>
</robot>
"""


def test_resets_are_injected_into_the_bare_position_state_interfaces():
    out = _inject_initial_values(_URDF, {"hardware": {"reset_positions": {"1": -1.5, "2": 0.25}}})
    assert '<param name="initial_value">-1.5</param>' in out
    assert '<param name="initial_value">0.25</param>' in out
    # the velocity state interface, joint 7 and the transmission's joint
    # REFERENCE (which carries a hardwareInterface, not state interfaces)
    # stay untouched — the rendered LeKiwi URDF carries both per joint
    assert '<state_interface name="velocity"/>' in out
    assert "<hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>" in out
    assert out.count('<param name="initial_value">') == 2


def test_a_profile_without_resets_renders_byte_identical():
    assert _inject_initial_values(_URDF, {"hardware": {}}) == _URDF
    assert _inject_initial_values(_URDF, {}) == _URDF


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "0.5", None])
def test_non_finite_or_non_numeric_resets_are_rejected_by_name(value):
    with pytest.raises(RuntimeProfileError, match=r"reset_positions\.1 must be a finite number"):
        _inject_initial_values(_URDF, {"hardware": {"reset_positions": {"1": value}}})


def test_a_reset_joint_missing_from_ros2_control_is_an_error():
    with pytest.raises(RuntimeProfileError, match="does not declare.*\\['9'\\]"):
        _inject_initial_values(_URDF, {"hardware": {"reset_positions": {"9": 0.0}}})


def test_a_reset_joint_without_a_position_state_interface_is_an_error():
    with pytest.raises(RuntimeProfileError, match="no position state interface"):
        _inject_initial_values(_URDF, {"hardware": {"reset_positions": {"7": 0.0}}})


def test_an_existing_initial_value_is_a_refused_conflict():
    urdf = _URDF.replace(
        """<joint name="2">
      <command_interface name="position"/>
      <state_interface name="position"/>
    </joint>""",
        """<joint name="2">
      <command_interface name="position"/>
      <state_interface name="position">
        <param name="initial_value">0.1</param>
      </state_interface>
    </joint>""",
    )
    with pytest.raises(RuntimeProfileError, match="conflicts with the initial_value"):
        _inject_initial_values(urdf, {"hardware": {"reset_positions": {"2": 0.25}}})


def test_a_position_state_interface_with_other_parameters_is_refused():
    urdf = _URDF.replace(
        """<joint name="2">
      <command_interface name="position"/>
      <state_interface name="position"/>
    </joint>""",
        """<joint name="2">
      <command_interface name="position"/>
      <state_interface name="position">
        <param name="custom">1</param>
      </state_interface>
    </joint>""",
    )
    with pytest.raises(RuntimeProfileError, match="already carries parameters"):
        _inject_initial_values(urdf, {"hardware": {"reset_positions": {"2": 0.25}}})
