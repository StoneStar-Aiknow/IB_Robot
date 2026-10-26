"""Unit tests for the G1 vendor gateway (the unitree_hg / unitree_api boundary).

These tests exercise the real generated vendor message classes — the exact
types the bridge will publish and receive — without a ROS graph, a DDS
participant or a robot (no ``rclpy.init``). ``RpcClient`` is driven through
injected ``publish``/``now`` callables, which is what makes the correlation,
timeout, no-resend and late-response rules testable at all.

Review-driven regression coverage:

* ``RpcClient.send`` must allocate the identity id before building the
  request — the projection rejects a placeholder 0 outright.
* a response arriving after the timeout window — whether the timeout was
  already polled or not — must land as a timeout result, never as a success.
* ``send`` with the vendor unavailable must leave no pending entry behind
  and burn no identity id.
* ``handcmd_from_dict`` appends into the unbounded HandCmd.motor_cmd and
  demands exactly 7 slots; ``handstate_q`` refuses a hand state that does not
  carry exactly 7 motor positions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rclpy.qos
import yaml
from unitree_api.msg import Response
from unitree_g1_robot import projection, vendor_gateway
from unitree_g1_robot.vendor_gateway import (
    RpcClient,
    VendorUnavailableError,
    handcmd_from_dict,
    handstate_q,
    lowcmd_from_dict,
    lowstate_imu,
    lowstate_mode_machine,
    lowstate_q,
    qos_profile,
    request_from_dict,
    response_to_dict,
)
from unitree_hg.msg import HandState, LowState, MotorState

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "g1_29dof_dex3.yaml"


def _frame_latch() -> dict[str, float]:
    return {name: 0.01 * index for index, name in enumerate(projection.ARM_SDK_FRAME)}


class _Clock:
    """Injectable ``now``: deterministic, advanced by hand."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _client() -> tuple[RpcClient, _Clock, list]:
    """An RpcClient wired to a fake clock and a publish recorder."""
    published: list = []
    clock = _Clock()
    return RpcClient(publish=published.append, now=clock), clock, published


def _response(identity_id: int, api_id: int, *, code: int = 0, data: str = "") -> Response:
    msg = Response()
    msg.header.identity.id = identity_id
    msg.header.identity.api_id = api_id
    msg.header.status.code = code
    msg.data = data
    return msg


# --- dict -> vendor messages ---------------------------------------------------


def test_lowcmd_from_dict_projects_every_idl_field():
    d = projection.arm_sdk_lowcmd(_frame_latch(), weight=0.5, mode_pr=1, mode_machine=5)
    msg = lowcmd_from_dict(d)

    assert msg.mode_pr == 1
    assert msg.mode_machine == 5
    assert len(msg.motor_cmd) == projection.LOW_CMD_MOTOR_COUNT
    assert list(msg.reserve) == [0, 0, 0, 0]
    assert msg.crc == 0  # the CRC-requiring rt/lowcmd tier stays disabled (D11)

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


def test_lowcmd_from_dict_rejects_wrong_motor_count():
    d = projection.arm_sdk_lowcmd(_frame_latch(), weight=0.0, mode_pr=1, mode_machine=5)
    d["motor_cmd"] = d["motor_cmd"][:34]
    with pytest.raises(ValueError, match="35"):
        lowcmd_from_dict(d)


def test_handcmd_from_dict_projects_seven_mode_bytes():
    targets = {"left_hand_thumb_0_joint": 0.3, "left_hand_index_1_joint": -0.2}
    msg = handcmd_from_dict(projection.dex3_handcmd(targets, side="left"))

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
        else:  # slots without a latched target keep the HandCmd default zero
            assert slot.q == 0.0 and slot.kp == 0.0 and slot.kd == 0.0


def test_handcmd_from_dict_timeout_release_sets_the_mode_msb_and_zeroes_gains():
    targets = {"left_hand_thumb_0_joint": 0.3}
    msg = handcmd_from_dict(projection.dex3_handcmd(targets, side="left", timeout=True))

    for index in range(projection.DEX3_MOTORS_PER_HAND):
        slot = msg.motor_cmd[index]
        assert slot.mode == projection.ris_mode_byte(index, timeout=True)
        assert slot.kp == 0.0 and slot.kd == 0.0  # a released hand with kp still pulls


def test_handcmd_from_dict_rejects_wrong_motor_count():
    d = projection.dex3_handcmd({}, side="right")
    d["motor_cmd"] = d["motor_cmd"][:6]
    with pytest.raises(ValueError, match="7"):
        handcmd_from_dict(d)


def test_request_from_dict_projects_header_and_payload():
    d = projection.rpc_request(7, 7101, {"data": 4}, binary=bytes([0, 127, 255]))
    msg = request_from_dict(d)

    assert msg.header.identity.id == 7
    assert msg.header.identity.api_id == 7101
    assert msg.header.lease.id == 0
    assert msg.header.policy.priority == 0
    assert msg.header.policy.noreply is False
    assert msg.parameter == '{"data":4}'
    # Request.binary is uint8[] — the full unsigned byte range survives.
    assert list(msg.binary) == [0, 127, 255]


# --- vendor messages -> pure values ---------------------------------------------


def _lowstate() -> LowState:
    msg = LowState()
    msg.mode_machine = 5
    for index in range(projection.LOW_CMD_MOTOR_COUNT):
        msg.motor_state[index].q = 0.1 * index
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    return msg


def test_lowstate_q_feeds_joint_position_projection():
    q = lowstate_q(_lowstate())

    assert len(q) == projection.LOW_CMD_MOTOR_COUNT
    joints = projection.low_state_joint_positions(q)
    assert joints["left_elbow_joint"] == pytest.approx(0.1 * 18)
    assert joints["right_wrist_yaw_joint"] == pytest.approx(0.1 * 28)


def test_lowstate_mode_machine_reads_the_generation_id():
    assert lowstate_mode_machine(_lowstate()) == 5


def test_lowstate_imu_feeds_imu_payload():
    payload = projection.imu_payload(lowstate_imu(_lowstate()))

    # The IDL fields are float32: values pass through at float32 precision.
    assert payload["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    assert payload["angular_velocity"] == {"x": pytest.approx(0.1), "y": pytest.approx(0.2), "z": pytest.approx(0.3)}
    assert payload["linear_acceleration"] == {"x": 0.0, "y": 0.0, "z": pytest.approx(9.81)}


def test_handstate_q_feeds_dex3_projection():
    msg = HandState()
    msg.motor_state = [MotorState() for _ in range(projection.DEX3_MOTORS_PER_HAND)]
    for index in range(projection.DEX3_MOTORS_PER_HAND):
        msg.motor_state[index].q = 0.1 * index

    fingers = projection.dex3_joint_positions(handstate_q(msg), side="left")

    assert fingers["left_hand_thumb_0_joint"] == pytest.approx(0.0)
    assert fingers["left_hand_middle_1_joint"] == pytest.approx(0.6)


def test_handstate_q_rejects_a_short_motor_state():
    msg = HandState()
    msg.motor_state = [MotorState() for _ in range(3)]
    with pytest.raises(ValueError, match="7"):
        handstate_q(msg)


def test_response_to_dict_feeds_projection_rpc_readers():
    msg = _response(7, 7101, data='{"fsm_id":4}')
    msg.binary = [1, -1]  # Response.binary is int8[]; negative bytes survive

    d = response_to_dict(msg)

    assert projection.rpc_response_matches(d, identity_id=7, api_id=7101)
    assert projection.rpc_response_code(d) == 0
    assert projection.rpc_response_data(d) == {"fsm_id": 4}
    assert d["binary"] == [1, -1]


# --- QoS -------------------------------------------------------------------------


def _vendor_qos() -> dict:
    with open(_PROFILE) as handle:
        return yaml.safe_load(handle)["vendor"]["qos"]


def test_qos_profile_resolves_profile_declared_endpoints():
    reliable = qos_profile(_vendor_qos(), "rpc_request")
    assert reliable.reliability == rclpy.qos.ReliabilityPolicy.RELIABLE
    assert reliable.durability == rclpy.qos.DurabilityPolicy.VOLATILE
    assert reliable.depth == 10

    best_effort = qos_profile(_vendor_qos(), "low_state")
    assert best_effort.reliability == rclpy.qos.ReliabilityPolicy.BEST_EFFORT


def test_qos_profile_falls_back_to_projection_defaults_for_undeclared_keys():
    profile = qos_profile({}, "dex3_state")
    assert profile.reliability == rclpy.qos.ReliabilityPolicy.BEST_EFFORT
    assert profile.durability == rclpy.qos.DurabilityPolicy.VOLATILE
    assert profile.depth == 10


# --- RpcClient --------------------------------------------------------------------


def test_send_allocates_monotonic_identity_ids_from_one():
    client, _, _ = _client()

    assert [client.send(7101, {"data": 1}, timeout_s=5.0) for _ in range(3)] == [1, 2, 3]


def test_send_publishes_request_with_allocated_identity_and_api():
    client, _, published = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)

    assert len(published) == 1
    request = published[0]
    assert request.header.identity.id == identity_id
    assert request.header.identity.api_id == 7101
    assert request.parameter == '{"data":4}'


def test_send_without_vendor_leaves_no_pending_and_burns_no_id(monkeypatch: pytest.MonkeyPatch):
    client, _, published = _client()
    monkeypatch.setattr(vendor_gateway, "VENDOR_AVAILABLE", False)

    with pytest.raises(VendorUnavailableError):
        client.send(7101, {"data": 4}, timeout_s=5.0)

    assert client._pending == {}
    assert published == []

    # The failed send burned no identity id: the next real call starts at 1.
    monkeypatch.setattr(vendor_gateway, "VENDOR_AVAILABLE", True)
    assert client.send(7101, {"data": 4}, timeout_s=5.0) == 1


def test_matching_response_completes_call():
    client, _, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    client.on_response(_response(identity_id, 7101, data='{"ok":true}'))

    result = client.poll(identity_id)
    assert result is not None
    assert result.ok is True
    assert result.code == 0
    assert result.data == {"ok": True}
    assert result.timed_out is False
    assert result.motion is False
    assert result.api_id == 7101


def test_vendor_error_code_is_reported_not_ok():
    client, _, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    client.on_response(_response(identity_id, 7101, code=7))

    result = client.poll(identity_id)
    assert result is not None
    assert result.ok is False
    assert result.code == 7


def test_unmatched_responses_are_dropped():
    client, clock, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    client.on_response(_response(99, 7101))  # unknown identity id
    client.on_response(_response(identity_id, 9999))  # right id, wrong api_id
    clock.advance(0.1)

    assert client.poll(identity_id) is None  # still pending, nothing recorded


def test_poll_returns_none_while_pending():
    client, clock, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    clock.advance(4.9)

    assert client.poll(identity_id) is None


def test_poll_reports_timeout_with_sentinel_code():
    client, clock, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    clock.advance(5.0)
    result = client.poll(identity_id)

    assert result is not None
    assert result.ok is False
    assert result.code == -1  # sentinel: no vendor code ever arrived, not a success code
    assert result.data is None
    assert result.timed_out is True
    assert result.api_id == 7101
    # Polling again reports the same terminal outcome.
    assert client.poll(identity_id) == result


def test_timeout_never_resends():
    client, clock, published = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    clock.advance(30.0)
    assert client.poll(identity_id) is not None
    assert client.poll(identity_id) is not None

    assert len(published) == 1  # a timed-out call is never re-issued here


def test_motion_calls_are_flagged_for_the_retry_decision():
    client, clock, _ = _client()

    timeout_id = client.send(7101, projection.fsm_id_parameter(4), timeout_s=5.0, motion=True)
    clock.advance(5.0)
    result = client.poll(timeout_id)
    assert result is not None and result.timed_out and result.motion is True

    answered_id = client.send(7106, projection.action_id_parameter(11), timeout_s=5.0, motion=True)
    client.on_response(_response(answered_id, 7106))
    result = client.poll(answered_id)
    assert result is not None and result.ok and result.motion is True


def test_late_response_after_a_polled_timeout_is_ignored():
    client, clock, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    clock.advance(5.0)
    timed_out = client.poll(identity_id)
    assert timed_out is not None and timed_out.timed_out

    client.on_response(_response(identity_id, 7101, data='{"late":true}'))

    result = client.poll(identity_id)
    assert result is not None
    assert result.timed_out and not result.ok and result.data is None


def test_late_response_after_window_without_poll_lands_as_timeout():
    client, clock, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    clock.advance(6.0)  # window elapsed, not polled yet
    client.on_response(_response(identity_id, 7101, data='{"late":true}'))

    result = client.poll(identity_id)
    assert result is not None
    assert result.timed_out and not result.ok
    assert result.code == -1
    assert result.data is None


def test_malformed_response_data_is_kept_as_string():
    client, _, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    client.on_response(_response(identity_id, 7101, data="not json"))

    result = client.poll(identity_id)
    assert result is not None
    assert result.ok and result.code == 0
    assert result.data == "not json"  # the raw string, not an escaped exception


def test_discard_releases_the_entry():
    client, _, _ = _client()

    identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
    client.discard(identity_id)

    assert client.poll(identity_id) is None  # the entry is gone, not "pending"
    client.discard(identity_id)  # discarding twice is a no-op
    client.on_response(_response(identity_id, 7101))  # a response for a discarded call is dropped


def test_discard_keeps_the_pending_table_bounded():
    client, _, _ = _client()

    for _ in range(3):
        identity_id = client.send(7101, {"data": 4}, timeout_s=5.0)
        client.discard(identity_id)

    assert len(client._pending) == 0
