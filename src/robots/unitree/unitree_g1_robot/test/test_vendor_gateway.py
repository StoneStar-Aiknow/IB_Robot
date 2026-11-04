"""Unit tests for the runtime's vendor boundary (``vendor_gateway``).

Since the SDK migration (Issue #171 PR 2) the gateway is a thin naming layer
over ``unitree_g1_adapter``; the hand-written RPC correlator it used to carry
is gone (its timeout/no-resend rules live with the official SDK's clients and
are covered by the adapter's own test suite). What is still THIS repo's
contract and therefore tested here:

* the lazy IDL re-exports resolve to the real vendor classes and refuse an
  SDK-less host with the runtime's guided error;
* the adapter's frame converters (which the runtime and the mock drive)
  project the projection's plain dict shapes onto the real vendor IDL —
  ``low_cmd`` header fields, the fixed 35-slot layout, the Dex3 7-slot hand
  frames with their RIS_Mode bytes;
* the sample readers flatten the real vendor samples back onto plain values.

These run without a ROS graph, a DDS participant or a robot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unitree_g1_robot import projection, vendor_gateway

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"


def _frame_latch() -> dict[str, float]:
    return {name: 0.01 * index for index, name in enumerate(projection.ARM_SDK_FRAME)}


def _vendor_available() -> bool:
    return vendor_gateway.VENDOR_AVAILABLE


requires_sdk = pytest.mark.skipif(not _vendor_available(), reason="unitree_sdk2py is not installed")


# --- lazy IDL re-exports ---------------------------------------------------------


def test_module_attributes_stay_lazy_without_importing_the_sdk():
    # The module imports cleanly on an SDK-less host; only touching a vendor
    # name resolves (and refuses) through the adapter.
    assert isinstance(vendor_gateway.VENDOR_AVAILABLE, bool)


@requires_sdk
def test_idl_names_resolve_to_the_real_vendor_classes():
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_

    assert vendor_gateway.LowState is LowState_
    assert vendor_gateway.LowCmd is LowCmd_
    assert vendor_gateway.HandState is HandState_
    assert vendor_gateway.HandCmd is HandCmd_
    assert vendor_gateway.Request is Request_
    assert vendor_gateway.Response is Response_
    # Resolution is cached: the same class object comes back.
    assert vendor_gateway.LowState is LowState_


# --- adapter frame builders driven by the projection ------------------------------


@requires_sdk
def test_lowcmd_frame_mapping_projects_every_idl_field():
    from unitree_g1_adapter import vendor as adapter_vendor

    d = projection.arm_sdk_lowcmd(_frame_latch(), weight=0.5, mode_pr=1, mode_machine=5)
    msg = adapter_vendor.lowcmd_from_slots(d["motor_cmd"], mode_pr=d["mode_pr"], mode_machine=d["mode_machine"])

    assert msg.mode_pr == 1
    assert msg.mode_machine == 5
    assert len(msg.motor_cmd) == projection.LOW_CMD_MOTOR_COUNT
    assert list(msg.reserve) == [0, 0, 0, 0]
    from unitree_sdk2py.utils.crc import CRC

    assert msg.crc == CRC().Crc(msg)
    old_crc = msg.crc
    msg.motor_cmd[29].q = 0.0
    assert CRC().Crc(msg) != old_crc
    msg.motor_cmd[29].q = 0.5

    for name, index in projection.LOW_STATE_JOINT_INDEX.items():
        slot = msg.motor_cmd[index]
        if name in _frame_latch():
            assert slot.q == pytest.approx(_frame_latch()[name])
            assert slot.kp == pytest.approx(projection.EXAMPLE_ARM_SDK_KP)
            assert slot.kd == pytest.approx(projection.EXAMPLE_ARM_SDK_KD)
        else:  # the 12 leg slots stay at vendor-default zeros
            assert slot.mode == 0 and slot.q == 0.0 and slot.kp == 0.0 and slot.kd == 0.0

    # The shared weight rides motor_cmd[29].q — the slot after JointIndex 0-28.
    assert msg.motor_cmd[projection.ARM_SDK_WEIGHT_INDEX].q == pytest.approx(0.5)
    for index in range(30, projection.LOW_CMD_MOTOR_COUNT):
        assert msg.motor_cmd[index].q == 0.0


@requires_sdk
def test_lowcmd_frame_builder_rejects_wrong_motor_count():
    from unitree_g1_adapter import vendor as adapter_vendor

    slots = [projection.motor_cmd() for _ in range(projection.LOW_CMD_MOTOR_COUNT - 1)]
    with pytest.raises(ValueError, match="35"):
        adapter_vendor.lowcmd_from_slots(slots, mode_pr=1, mode_machine=5, crc=0)


@requires_sdk
def test_handcmd_frame_builder_projects_seven_mode_bytes():
    from unitree_g1_adapter import vendor as adapter_vendor

    targets = {"left_hand_thumb_0_joint": 0.3, "left_hand_index_1_joint": -0.2}
    frame = projection.dex3_handcmd(targets, side="left")
    msg = adapter_vendor.handcmd_from_slots(frame["motor_cmd"])

    assert len(msg.motor_cmd) == projection.DEX3_MOTORS_PER_HAND
    for index in range(projection.DEX3_MOTORS_PER_HAND):
        slot = msg.motor_cmd[index]
        # The RIS_Mode byte addresses the hand's motors by position.
        assert slot.mode == projection.ris_mode_byte(index)
        commanded = projection.dex3_joint_names("left")[index] in targets
        if commanded:
            assert slot.q == pytest.approx(targets[projection.dex3_joint_names("left")[index]])
            assert slot.kp == pytest.approx(projection.EXAMPLE_DEX3_KP)
            assert slot.kd == pytest.approx(projection.EXAMPLE_DEX3_KD)
        else:  # slots without a latched target keep the vendor default zero
            assert slot.q == 0.0 and slot.kp == 0.0 and slot.kd == 0.0


@requires_sdk
def test_handcmd_frame_builder_timeout_release_sets_the_mode_msb_and_zeroes_gains():
    from unitree_g1_adapter import vendor as adapter_vendor

    targets = {"left_hand_thumb_0_joint": 0.3}
    frame = projection.dex3_handcmd(targets, side="left", timeout=True)
    msg = adapter_vendor.handcmd_from_slots(frame["motor_cmd"])

    for index in range(projection.DEX3_MOTORS_PER_HAND):
        slot = msg.motor_cmd[index]
        assert slot.mode == projection.ris_mode_byte(index, timeout=True)
        assert slot.kp == 0.0 and slot.kd == 0.0  # a released hand with kp still pulls


@requires_sdk
def test_handcmd_frame_builder_refuses_an_empty_slot_list():
    from unitree_g1_adapter import vendor as adapter_vendor

    with pytest.raises(ValueError, match="at least one"):
        adapter_vendor.handcmd_from_slots([])


# --- sample readers ---------------------------------------------------------------


def _lowstate():
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

    message = unitree_hg_msg_dds__LowState_()
    message.mode_machine = 5
    for index in range(projection.LOW_CMD_MOTOR_COUNT):
        message.motor_state[index].q = 0.1 * index
    message.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    message.imu_state.gyroscope = [0.1, 0.2, 0.3]
    message.imu_state.accelerometer = [0.0, 0.0, 9.81]
    return message


@requires_sdk
def test_lowstate_q_feeds_joint_position_projection():
    q = vendor_gateway.lowstate_q(_lowstate())

    assert len(q) == projection.LOW_CMD_MOTOR_COUNT
    joints = projection.low_state_joint_positions(q)
    assert joints["left_elbow_joint"] == pytest.approx(0.1 * 18)
    assert joints["right_wrist_yaw_joint"] == pytest.approx(0.1 * 28)


@requires_sdk
def test_lowstate_mode_machine_reads_the_generation_id():
    assert vendor_gateway.lowstate_mode_machine(_lowstate()) == 5


@requires_sdk
def test_lowstate_imu_feeds_imu_payload():
    payload = projection.imu_payload(vendor_gateway.lowstate_imu(_lowstate()))

    # The IDL fields are float32: values pass through at float32 precision.
    assert payload["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    assert payload["angular_velocity"] == {"x": pytest.approx(0.1), "y": pytest.approx(0.2), "z": pytest.approx(0.3)}
    assert payload["linear_acceleration"] == {"x": 0.0, "y": 0.0, "z": pytest.approx(9.81)}


@requires_sdk
def test_handstate_q_feeds_dex3_projection():
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_

    message = unitree_hg_msg_dds__HandState_()
    for index in range(projection.DEX3_MOTORS_PER_HAND):
        message.motor_state[index].q = 0.1 * index

    fingers = projection.dex3_joint_positions(vendor_gateway.handstate_q(message), side="left")

    assert fingers["left_hand_thumb_0_joint"] == pytest.approx(0.0)
    assert fingers["left_hand_middle_1_joint"] == pytest.approx(0.6)


# --- RPC frame converters (the mock's server side) ---------------------------------


@requires_sdk
def test_rpc_response_round_trips_through_the_converters():
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import (
        Request_,
        RequestHeader_,
        RequestIdentity_,
        RequestLease_,
        RequestPolicy_,
    )

    payload = {
        "header": {
            "identity": {"id": 7, "api_id": 7101},
            "status": {"code": 0},
        },
        "data": '{"ok":true}',
    }
    response = vendor_gateway.rpc_response_from_dict(payload)
    assert response.header.identity.id == 7
    assert response.header.identity.api_id == 7101
    assert response.header.status.code == 0
    assert response.data == '{"ok":true}'

    request = Request_(
        RequestHeader_(RequestIdentity_(3, 7101), RequestLease_(0), RequestPolicy_(0, False)),
        '{"data":4}',
        [],
    )
    flat = vendor_gateway.rpc_request_to_dict(request)
    assert flat["header"]["identity"] == {"id": 3, "api_id": 7101}
    assert flat["header"]["lease"] == {"id": 0}
    assert flat["header"]["policy"] == {"priority": 0, "noreply": False}
    assert flat["parameter"] == '{"data":4}'
    assert flat["binary"] == []
