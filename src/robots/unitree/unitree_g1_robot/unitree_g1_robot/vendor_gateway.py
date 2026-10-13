"""The only module that imports vendor (``unitree_hg`` / ``unitree_api``) types.

Keeping the vendor dependency here means the projection logic, the profile
and the tests that cover them run without the vendored message packages
built, and that no generic IB-Robot package ever links a vendor type. Every
builder is a pure function of vendor message classes, so it is exercised
against the real generated types without a robot and without a DDS
participant; the RPC correlation core (``RpcClient``) is likewise
constructor-injected with ``publish`` and ``now`` callables, so the
timeout/motion/no-resend rules are unit-testable without a ROS graph.

The vendored message classes are re-exported (``Request``, ``Response``,
``LowCmd``, ``LowState``, ``HandCmd``, ``HandState``, ``MotorState``,
``PressSensorState``) so the bridge and the mock wire the DDS graph without
importing a vendor package themselves — this module stays the single place
that names a vendor import.

No ROS node, subscription or timer is defined here: this module only maps
between the projection's plain values and the vendor IDL, and tracks pending
RPC calls. Wiring the DDS graph is the bridge's own job.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import rclpy.qos

from unitree_g1_robot import projection

try:  # pragma: no cover - import guard exercised by the missing-vendor test
    # The plain imports are used inside this module; the redundant-alias ones
    # (`import X as X`, PEP 484's explicit-re-export convention) are re-exports
    # for the bridge and the mock, which wire the DDS graph with them.
    from unitree_api.msg import Request
    from unitree_api.msg import Response as Response
    from unitree_hg.msg import HandCmd, HandState, LowCmd, LowState, MotorCmd
    from unitree_hg.msg import MotorState as MotorState
    from unitree_hg.msg import PressSensorState as PressSensorState

    VENDOR_AVAILABLE = True
    VENDOR_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - depends on the environment
    VENDOR_AVAILABLE = False
    VENDOR_IMPORT_ERROR = str(exc)


class VendorUnavailableError(RuntimeError):
    """Raised when the vendored unitree interfaces are not on the ROS 2 path."""

    def __init__(self) -> None:
        super().__init__(
            "unitree_hg/unitree_api are not available: build the vendored message packages "
            f"before launching the G1 runtime. Import error: {VENDOR_IMPORT_ERROR}"
        )


def require_vendor() -> None:
    if not VENDOR_AVAILABLE:
        raise VendorUnavailableError()


# --- dict -> vendor messages ---------------------------------------------------


def lowcmd_from_dict(d: Mapping[str, Any]) -> LowCmd:
    """Build a unitree_hg/LowCmd from :func:`projection.arm_sdk_lowcmd` output.

    Field names follow the vendored IDL exactly (unitree_hg/LowCmd.msg):
    ``mode_pr``/``mode_machine`` are uint8 and ``crc`` is carried verbatim —
    the projection emits 0 because the CRC-requiring rt/lowcmd tier stays
    disabled (design D11).
    """
    require_vendor()
    msg = LowCmd()
    msg.mode_pr = int(d["mode_pr"])
    msg.mode_machine = int(d["mode_machine"])
    slots = list(d["motor_cmd"])
    if len(slots) != projection.LOW_CMD_MOTOR_COUNT:
        raise ValueError(
            f"lowcmd carries {len(slots)} motor_cmd slots; the vendored LowCmd is fixed at "
            f"{projection.LOW_CMD_MOTOR_COUNT}"
        )
    for slot, cmd in enumerate(slots):
        target = msg.motor_cmd[slot]
        target.mode = int(cmd["mode"])
        target.q = float(cmd["q"])
        target.dq = float(cmd["dq"])
        target.tau = float(cmd["tau"])
        target.kp = float(cmd["kp"])
        target.kd = float(cmd["kd"])
    msg.reserve = [int(value) for value in d.get("reserve", (0, 0, 0, 0))]
    msg.crc = int(d.get("crc", 0))
    return msg


def handcmd_from_dict(d: Mapping[str, Any]) -> HandCmd:
    """Build a unitree_hg/HandCmd from :func:`projection.dex3_handcmd` output.

    HandCmd.motor_cmd is an unbounded sequence in the IDL (default empty),
    unlike LowCmd's fixed 35 — one MotorCmd is built per input slot and
    appended. The slot count must be exactly the Dex3 per-hand 7, because the
    mode byte's motor id addresses the hand's motors by position.
    """
    require_vendor()
    slots = list(d["motor_cmd"])
    if len(slots) != projection.DEX3_MOTORS_PER_HAND:
        raise ValueError(
            f"handcmd carries {len(slots)} motor_cmd slots; a Dex3 hand commands exactly "
            f"{projection.DEX3_MOTORS_PER_HAND}"
        )
    msg = HandCmd()
    for cmd in slots:
        target = MotorCmd()
        target.mode = int(cmd["mode"])
        target.q = float(cmd["q"])
        target.dq = float(cmd["dq"])
        target.tau = float(cmd["tau"])
        target.kp = float(cmd["kp"])
        target.kd = float(cmd["kd"])
        msg.motor_cmd.append(target)
    msg.reserve = [int(value) for value in d.get("reserve", (0, 0, 0, 0))]
    return msg


def request_from_dict(d: Mapping[str, Any]) -> Request:
    """Build a unitree_api/Request from :func:`projection.rpc_request` output.

    ``Request.binary`` is a uint8[] in the IDL, so Python bytes and the
    projection's plain int list both assign; the int list is used because it
    is what the projection emits and what round-trips through the message
    unchanged.
    """
    require_vendor()
    msg = Request()
    header = d["header"]
    identity = header["identity"]
    msg.header.identity.id = int(identity["id"])
    msg.header.identity.api_id = int(identity["api_id"])
    msg.header.lease.id = int(header["lease"]["id"])
    policy = header["policy"]
    msg.header.policy.priority = int(policy["priority"])
    msg.header.policy.noreply = bool(policy["noreply"])
    msg.parameter = str(d["parameter"])
    msg.binary = [int(value) for value in d.get("binary", ())]
    return msg


# --- vendor messages -> pure values ----------------------------------------------


def lowstate_q(msg: LowState) -> list[float]:
    """The 35-entry q vector of LowState.motor_state, as plain floats.

    The projection's :func:`projection.low_state_joint_positions` consumes
    indices 0-28 of exactly this vector.
    """
    return [float(state.q) for state in msg.motor_state]


def lowstate_mode_machine(msg: LowState) -> int:
    """LowState.mode_machine (design D10's startup generation check reads this)."""
    return int(msg.mode_machine)


def lowstate_imu(msg: LowState) -> dict[str, list[float]]:
    """LowState.imu_state as a plain mapping for :func:`projection.imu_payload`.

    The vendored IMUState carries quaternion/gyroscope/accelerometer/rpy;
    the projection reads the first three and assumes (w, x, y, z) order
    (unverified, gate 6.1), so no reordering happens here — neither side of
    the boundary is allowed to guess the vendor's frame.
    """
    imu = msg.imu_state
    return {
        "quaternion": [float(value) for value in imu.quaternion],
        "gyroscope": [float(value) for value in imu.gyroscope],
        "accelerometer": [float(value) for value in imu.accelerometer],
    }


def handstate_q(msg: HandState) -> list[float]:
    """One HandState's motor_state[].q, in vendor motor order.

    HandState.motor_state is an unbounded sequence too — a hand reporting a
    slot count other than the Dex3 7 has a different generation or a decoding
    fault, and a silently truncated or extended position list would be mapped
    onto the wrong joints by the projection. Refused instead.
    """
    if len(msg.motor_state) != projection.DEX3_MOTORS_PER_HAND:
        raise ValueError(
            f"hand state carries {len(msg.motor_state)} motor positions; a Dex3 hand reports exactly "
            f"{projection.DEX3_MOTORS_PER_HAND}"
        )
    return [float(state.q) for state in msg.motor_state]


def response_to_dict(msg: Any) -> dict[str, Any]:
    """Flatten a unitree_api/Response into what the projection's RPC readers take.

    The output shape is the Response.msg field tree: header.identity carries
    the correlated request's ``id`` and ``api_id``, header.status.code is the
    vendor result code (0 = success) and ``data`` is the JSON payload string.
    :func:`projection.rpc_response_matches`, :func:`projection.rpc_response_code`
    and :func:`projection.rpc_response_data` consume it directly.
    """
    return {
        "header": {
            "identity": {
                "id": int(msg.header.identity.id),
                "api_id": int(msg.header.identity.api_id),
            },
            "status": {"code": int(msg.header.status.code)},
        },
        "data": str(msg.data),
        "binary": [int(value) for value in msg.binary],
    }


# --- QoS --------------------------------------------------------------------------


def qos_profile(qos_config: Mapping[str, Any] | None, key: str) -> rclpy.qos.QoSProfile:
    """Resolve one vendor endpoint's QoS into an ``rclpy`` profile.

    The raw (reliability, durability, depth) triple comes from the projection,
    which owns the profile-declared defaults and every validation rule.
    """
    reliability, durability, depth = projection.vendor_qos(qos_config, key)
    return rclpy.qos.QoSProfile(
        reliability=rclpy.qos.ReliabilityPolicy.RELIABLE
        if reliability == "reliable"
        else rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
        durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
        if durability == "transient_local"
        else rclpy.qos.DurabilityPolicy.VOLATILE,
        depth=depth,
    )


# --- RPC client ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RpcResult:
    """Outcome of one RPC call.

    ``code`` is the vendor ResponseStatus.code for a call that was answered
    (0 = success). A timeout result carries ``code=-1`` instead — a sentinel
    meaning "no vendor code ever arrived", deliberately distinct from the
    vendor success code 0 so the two can never be confused.

    ``motion`` is echoed from the send call: a motion-changing request that
    timed out must NOT be transparently retried (the vendor may already be
    executing it); the layer above reads this flag to refuse the retry.
    """

    ok: bool
    code: int
    data: Any = None
    timed_out: bool = False
    motion: bool = False
    api_id: int = 0


@dataclass(slots=True)
class _PendingCall:
    api_id: int
    timeout_s: float
    motion: bool
    sent_at: float
    result: RpcResult | None = None


def _timeout_result(call: _PendingCall) -> RpcResult:
    """The timeout outcome of a pending call; ``code=-1`` is the sentinel, not a vendor code."""
    return RpcResult(ok=False, code=-1, data=None, timed_out=True, motion=call.motion, api_id=call.api_id)


class RpcClient:
    """Correlate unitree_api requests with their responses.

    The client is transport-agnostic: it is constructed with a ``publish``
    callable (the bridge hands it the request topic's publisher) and a ``now``
    callable returning seconds, so the correlation, timeout and no-resend
    rules run in unit tests without a ROS graph, a DDS participant or a
    robot. Identity ids are allocated strictly monotonically from 1, because
    the vendor RPC tier correlates on header.identity.id alone.

    A timed-out call is never re-sent here — re-issuing a motion request
    whose fate is unknown is a decision the bridge owns, and the ``motion``
    flag on the timeout result is what lets it refuse. Once the caller has
    taken a finished result it should :meth:`discard` the call so the pending
    table does not grow without bound.
    """

    def __init__(
        self,
        *,
        publish: Callable[[Any], None],
        now: Callable[[], float],
    ) -> None:
        self._publish = publish
        self._now = now
        self._lock = threading.Lock()
        self._next_identity_id = 1
        self._pending: dict[int, _PendingCall] = {}

    def send(self, api_id: int, parameter: Any, *, timeout_s: float, motion: bool = False) -> int:
        """Allocate an identity id, publish the request, track it as pending.

        The vendor check comes first so an unavailable overlay neither burns
        an identity id nor leaves a pending entry behind. The identity id is
        allocated under the lock before the request dict is built, because
        :func:`projection.rpc_request` rejects a non-positive id — the real id
        must already be known when the request is built.
        """
        require_vendor()
        with self._lock:
            identity_id = self._next_identity_id
            self._next_identity_id += 1
        request_dict = projection.rpc_request(identity_id, api_id, parameter)
        with self._lock:
            self._pending[identity_id] = _PendingCall(
                api_id=int(api_id),
                timeout_s=float(timeout_s),
                motion=bool(motion),
                sent_at=self._now(),
            )
        self._publish(request_from_dict(request_dict))
        return identity_id

    def on_response(self, msg: Any) -> None:
        """Record the response for its pending call; unmatched responses are dropped.

        Matching is by identity id AND api_id (the vendor echoes both in the
        response header). A response whose call has already timed out — in
        :meth:`poll` or merely because the timeout window has elapsed — lands
        as a timeout result and the response itself is discarded: a late
        answer is not evidence the call succeeded.
        """
        response = response_to_dict(msg)
        try:
            identity_id = int(response["header"]["identity"]["id"])
        except (KeyError, TypeError, ValueError):
            return
        with self._lock:
            pending = self._pending.get(identity_id)
            if pending is None or pending.result is not None:
                return
            if not projection.rpc_response_matches(response, identity_id=identity_id, api_id=pending.api_id):
                return
            if self._now() - pending.sent_at >= pending.timeout_s:
                pending.result = _timeout_result(pending)
                return
            code = projection.rpc_response_code(response)
            try:
                data: Any = projection.rpc_response_data(response) if response["data"] else None
            except ValueError:
                # The vendor's data field is declared a JSON string, but a
                # malformed payload must not escape a receive callback; the
                # caller gets the raw string to log instead of an exception.
                data = response["data"]
            pending.result = RpcResult(
                ok=code == 0,
                code=code,
                data=data,
                timed_out=False,
                motion=pending.motion,
                api_id=pending.api_id,
            )

    def poll(self, identity_id: int) -> RpcResult | None:
        """The call's outcome, a timeout result, or None while still pending.

        The timeout is resolved against the injected ``now``: a pending call
        whose ``timeout_s`` has elapsed becomes a ``timed_out`` result and is
        never re-sent by this client.
        """
        with self._lock:
            pending = self._pending.get(int(identity_id))
            if pending is None:
                return None
            if pending.result is not None:
                return pending.result
            if self._now() - pending.sent_at < pending.timeout_s:
                return None
            pending.result = _timeout_result(pending)
            return pending.result

    def discard(self, identity_id: int) -> None:
        """Release a call's bookkeeping entry once its result has been taken.

        Finished results are kept until discarded so the caller can poll
        repeatedly; without this the pending table would grow without bound
        over the life of the bridge. Discarding a still-pending call abandons
        it — a later response for it is then simply dropped.
        """
        with self._lock:
            self._pending.pop(int(identity_id), None)
