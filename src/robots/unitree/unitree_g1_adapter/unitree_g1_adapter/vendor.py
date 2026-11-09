"""The only module that imports a ``unitree_sdk2py`` symbol.

Everything else in this package (and, from PR 2 on, the G1 runtime) consumes
the injectable surface defined here, so the package imports — and its tests
run — on hosts where the vendor SDK is absent. The real implementation is
resolved lazily through :func:`default_vendor`; :func:`require_vendor` turns a
missing SDK into :class:`~unitree_g1_adapter.errors.UnitreeSdkUnavailableError`
with the setup guidance attached.

Every fact coded below was verified against the pinned upstream revision
(``814556d15970dd2ecf1c9984e845ca02ab07e206``), partly by executing it:

- ``ChannelPublisher(name, idl_type)`` + ``Init()`` / ``Write(sample, timeout)``
  / ``Close()``; ``ChannelSubscriber(name, idl_type)`` + ``Init(handler,
  queue_len)`` / ``Close()`` (``unitree_sdk2py/core/channel.py``).
- ``ChannelFactory`` is a process-wide singleton: ``Init`` returns early once
  initialized, silently ignoring new arguments.
- ``motor_cmd`` arrays are plain Python lists: element assignment, in-place
  mutation and whole-list assignment all work (executed against the pinned
  revision). ``LowCmd_`` is fixed at 35 slots; the ``HandCmd_`` default
  factory pre-fills 7 slots (Dex3) and accepts any list the caller assigns.
- RPC clients: ``ClientBase`` defaults to a 1.0 s timeout, does not retry, and
  returns plain status codes or ``(code, data)`` tuples; constants in
  ``unitree_sdk2py/rpc/internal.py`` are ``RPC_OK = 0`` and the client-side
  timeout sentinel ``RPC_ERR_CLIENT_API_TIMEOUT = 3104``.
- Known upstream quirk, documented not fixed here: ``AudioClient.TtsMaker``
  keeps ``tts_index`` at 0 forever (``self.tts_index += self.tts_index``), so
  every TTS request carries the same index. The adapter passes the vendor
  call through unchanged; PR 2 decides whether that matters for the speech
  contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from unitree_g1_adapter.errors import UnitreeSdkUnavailableError

VENDOR_IMPORT_ERROR = ""

try:  # pragma: no cover - exercised through require_vendor on hosts without the SDK
    from unitree_sdk2py.b2.robot_state.robot_state_api import (
        ROBOT_STATE_API_ID_SERVICE_LIST,
        ROBOT_STATE_API_VERSION,
        ROBOT_STATE_SERVICE_NAME,
    )
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.default import (
        unitree_api_msg_dds__Response_,
        unitree_hg_msg_dds__HandCmd_,
        unitree_hg_msg_dds__HandState_,
        unitree_hg_msg_dds__LowCmd_,
        unitree_hg_msg_dds__LowState_,
        unitree_hg_msg_dds__MotorCmd_,
        unitree_hg_msg_dds__MotorState_,
        unitree_hg_msg_dds__PressSensorState_,
    )
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_ as RpcRequest_
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import Response_ as RpcResponse_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
    from unitree_sdk2py.rpc import internal as _rpc_internal
    from unitree_sdk2py.rpc.client import Client as _SdkRpcClient
    from unitree_sdk2py.utils.crc import CRC

    RPC_OK = int(_rpc_internal.RPC_OK)
    RPC_ERR_CLIENT_API_TIMEOUT = int(_rpc_internal.RPC_ERR_CLIENT_API_TIMEOUT)
    RPC_ERR_CLIENT_SEND = int(_rpc_internal.RPC_ERR_CLIENT_SEND)

    class RobotStateClient(_SdkRpcClient):
        """Read-only B2 wire API over the official SDK RPC transport.

        The pinned B2 wrapper imports nonexistent rpc.client_internal. Do not
        mutate the installed SDK or silently substitute Go2's older API version.
        Register only the verified B2 service-list API, using its own constants.
        """

        def __init__(self):
            super().__init__(ROBOT_STATE_SERVICE_NAME, False)

        def Init(self):
            self._SetApiVerson(ROBOT_STATE_API_VERSION)
            self._RegistApi(ROBOT_STATE_API_ID_SERVICE_LIST, 0)

        def ServiceList(self):
            code, data = self._Call(ROBOT_STATE_API_ID_SERVICE_LIST, "{}")
            return code, json.loads(data) if code == 0 else None

    VENDOR_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on the host environment
    VENDOR_AVAILABLE = False
    VENDOR_IMPORT_ERROR = str(exc)

    # Bound to None so attribute access on an SDK-less host fails through
    # require_vendor's guided error instead of a bare NameError.
    ChannelFactoryInitialize = ChannelPublisher = ChannelSubscriber = None
    MotionSwitcherClient = G1ArmActionClient = AudioClient = LocoClient = None
    RobotStateClient = CRC = None
    HandCmd_ = HandState_ = LowCmd_ = LowState_ = None
    unitree_api_msg_dds__Response_ = None
    RpcRequest_ = RpcResponse_ = None
    unitree_hg_msg_dds__HandCmd_ = unitree_hg_msg_dds__HandState_ = None
    unitree_hg_msg_dds__LowCmd_ = unitree_hg_msg_dds__LowState_ = None
    unitree_hg_msg_dds__MotorCmd_ = None
    unitree_hg_msg_dds__MotorState_ = unitree_hg_msg_dds__PressSensorState_ = None
    RPC_OK = 0
    RPC_ERR_CLIENT_API_TIMEOUT = 3104
    RPC_ERR_CLIENT_SEND = 3102


def vendor_available() -> bool:
    """Whether the official vendor SDK imported successfully."""
    return VENDOR_AVAILABLE


def require_vendor() -> None:
    """Raise :class:`UnitreeSdkUnavailableError` when the SDK is absent."""
    if not VENDOR_AVAILABLE:
        raise UnitreeSdkUnavailableError(VENDOR_IMPORT_ERROR)


def _vendor_attr(name: str) -> Any:
    require_vendor()
    return globals()[name]


# --- injectable surface -------------------------------------------------------
#
# Consumers (channels/clients/runtime, and the tests' fakes) program against
# this mapping, never against ``import unitree_sdk2py`` directly. Fakes
# implement exactly these keys.


#: The ``unitree_hg`` IDL classes the G1 bridge exchanges, keyed by the neutral
#: names profiles will use from PR 2 on. ``LowCmd_``/``LowState_`` are the
#: whole-body tier; ``HandCmd_``/``HandState_`` are the DDS dexterous-hand tier
#: (Dex3). The Revo2 hand is a separate vendor adapter, not an ``unitree_hg``
#: type. ``rpc_request``/``rpc_response`` are the vendor RPC tier frames
#: (``unitree_api``), which the runtime-side RPC clients and the vendor mock's
#: RPC servers exchange over ``rt/api/<service>/{request,response}``.
IDL_CLASSES: Mapping[str, Callable[[], Any]] = {
    "low_cmd": lambda: _vendor_attr("LowCmd_"),
    "low_state": lambda: _vendor_attr("LowState_"),
    "hand_cmd": lambda: _vendor_attr("HandCmd_"),
    "hand_state": lambda: _vendor_attr("HandState_"),
    "rpc_request": lambda: _vendor_attr("RpcRequest_"),
    "rpc_response": lambda: _vendor_attr("RpcResponse_"),
}

_MOTOR_CMD_FACTORY = "unitree_hg_msg_dds__MotorCmd_"


def default_vendor() -> Mapping[str, Any]:
    """The real vendor surface, resolved lazily.

    Returns a plain mapping so tests can substitute a dict of fakes with the
    same keys; every value that touches the vendor still goes through
    :func:`require_vendor`, so an SDK-less host gets the guided error at first
    use rather than at import time.
    """
    return {
        "channel_factory_initialize": lambda domain_id, network_interface: _vendor_attr("ChannelFactoryInitialize")(
            domain_id, network_interface
        ),
        "channel_publisher": lambda name, type_key: _vendor_attr("ChannelPublisher")(name, idl_type(type_key)),
        "channel_subscriber": lambda name, type_key: _vendor_attr("ChannelSubscriber")(name, idl_type(type_key)),
        "loco_client": lambda: _vendor_attr("LocoClient")(),
        "arm_action_client": lambda: _vendor_attr("G1ArmActionClient")(),
        "audio_client": lambda: _vendor_attr("AudioClient")(),
        "motion_switcher_client": lambda: _vendor_attr("MotionSwitcherClient")(),
        "robot_state_client": lambda: _vendor_attr("RobotStateClient")(),
        "make_lowcmd": lowcmd_from_slots,
        "make_handcmd": handcmd_from_slots,
        "rpc_ok": RPC_OK,
        "rpc_timeout": RPC_ERR_CLIENT_API_TIMEOUT,
    }


def idl_type(type_key: str) -> Any:
    """The vendor IDL *class* for a neutral type key (channel constructor arg)."""
    if type_key not in IDL_CLASSES:
        raise KeyError(f"unknown unitree IDL type key {type_key!r}; known: {sorted(IDL_CLASSES)}")
    return IDL_CLASSES[type_key]()


# --- all-defaults IDL sample factories (consumed by the in-repo vendor mock) ---
#
# The mock builds and reads *real* vendor frames instead of translating to ROS
# types, so it needs the SDK's own default-constructed containers. They live
# here for the same reason every other vendor symbol does: this is the only
# module allowed to import ``unitree_sdk2py``, and the mock is a consumer of
# the adapter, not a second vendor import site.


def lowstate_defaults() -> Any:
    """A vendor ``LowState_`` sample from the SDK's all-defaults factory."""
    return _vendor_attr("unitree_hg_msg_dds__LowState_")()


def handstate_defaults() -> Any:
    """A vendor ``HandState_`` sample from the SDK's all-defaults factory."""
    return _vendor_attr("unitree_hg_msg_dds__HandState_")()


def motor_state_defaults() -> Any:
    """One vendor ``MotorState_`` slot from the SDK's all-defaults factory."""
    return _vendor_attr("unitree_hg_msg_dds__MotorState_")()


def motor_cmd_defaults() -> Any:
    """One vendor ``MotorCmd_`` slot from the SDK's all-defaults factory."""
    return _vendor_attr("unitree_hg_msg_dds__MotorCmd_")()


def press_sensor_defaults() -> Any:
    """One vendor ``PressSensorState_`` slot from the SDK's all-defaults factory."""
    return _vendor_attr("unitree_hg_msg_dds__PressSensorState_")()


# --- plain-data converters (transport-level; no robot semantics) --------------


_MOTOR_FLOAT_FIELDS = ("q", "dq", "tau", "kp", "kd")


def _apply_motor_command(target: Any, slot: Mapping[str, Any]) -> None:
    """Fill one vendor ``MotorCmd_`` from a plain slot dict.

    Unspecified fields keep the vendor factory default (0.0) — the adapter
    never invents gains or targets, it only refuses to *send* values the
    vendor could not mean (non-finite floats).
    """
    target.mode = int(slot.get("mode", 0))
    for field in _MOTOR_FLOAT_FIELDS:
        if field not in slot:
            continue
        value = float(slot[field])
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"motor slot field {field!r} must be finite, got {value!r}")
        setattr(target, field, value)


def lowcmd_from_slots(
    slots: Any,
    *,
    mode_pr: int = 0,
    mode_machine: int = 0,
    crc: int | None = None,
) -> Any:
    """Build a ``unitree_hg/LowCmd_`` from per-slot plain dicts.

    The vendor IDL fixes ``motor_cmd`` at 35 slots (29 joints + the arm_sdk
    weight slot + spares); a different count is a caller bug and is refused
    instead of silently mapping onto the wrong motors. Slot fields other than
    ``mode`` must be finite floats. By default compute CRC after populating the
    frame, matching the pinned official arm7 example. Explicit CRC overrides
    remain available for diagnostics and wire-format tests.
    """
    command = _vendor_attr("unitree_hg_msg_dds__LowCmd_")()
    slot_list = list(slots)
    if len(slot_list) != len(command.motor_cmd):
        raise ValueError(
            f"lowcmd carries {len(slot_list)} slots; the vendor LowCmd_ is fixed at {len(command.motor_cmd)}"
        )
    for index, slot in enumerate(slot_list):
        _apply_motor_command(command.motor_cmd[index], slot)
    command.mode_pr = int(mode_pr)
    command.mode_machine = int(mode_machine)
    command.crc = _vendor_attr("CRC")().Crc(command) if crc is None else int(crc)
    return command


def handcmd_from_slots(slots: Any) -> Any:
    """Build a ``unitree_hg/HandCmd_`` from per-slot plain dicts.

    ``HandCmd_.motor_cmd`` is an unbounded sequence: one ``MotorCmd_`` per
    input slot, in order, because the vendor addresses a hand's motors by
    position. An empty slot list is refused — a hand command frame with no
    motor command is never meaningful. The per-hand actuator count (Dex3: 7)
    is a robot-model fact owned by the runtime profile, not asserted here.
    """
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("handcmd needs at least one motor slot")
    motor_factory = _vendor_attr(_MOTOR_CMD_FACTORY)
    built = []
    for slot in slot_list:
        motor = motor_factory()
        _apply_motor_command(motor, slot)
        built.append(motor)
    command = _vendor_attr("unitree_hg_msg_dds__HandCmd_")()
    command.motor_cmd = built
    return command


def lowstate_q(message: Any) -> list[float]:
    """The per-motor ``q`` vector of a ``LowState_`` sample, as plain floats."""
    return [float(state.q) for state in message.motor_state]


def lowstate_mode_machine(message: Any) -> int:
    """``LowState_.mode_machine`` (the whole-machine configuration marker)."""
    return int(message.mode_machine)


def lowstate_imu(message: Any) -> dict[str, list[float]]:
    """``LowState_.imu_state`` as plain lists.

    The vendor IMUState carries quaternion/gyroscope/accelerometer/rpy; the
    adapter copies the arrays verbatim and never reorders axes — frame
    conventions are the runtime's concern (verified on hardware in PR 2).
    """
    imu = message.imu_state
    return {
        "quaternion": [float(value) for value in imu.quaternion],
        "gyroscope": [float(value) for value in imu.gyroscope],
        "accelerometer": [float(value) for value in imu.accelerometer],
    }


def handstate_motor_q(message: Any) -> list[float]:
    """The ``q`` vector of a ``HandState_`` sample, in vendor motor order.

    No slot-count assertion here on purpose: the per-hand actuator count is a
    robot-model fact (Dex3: 7) that belongs to the runtime profile, not to the
    transport adapter.
    """
    return [float(state.q) for state in message.motor_state]


# --- RPC frame converters (vendor mock server side) -----------------------------
#
# The runtime's RPC calls ride the adapter's typed RPC clients (the vendor SDK
# clients themselves). What also speaks the ``unitree_api`` wire is the vendor
# mock's RPC *servers*: they receive ``Request_`` frames and answer with
# ``Response_`` frames on ``rt/api/<service>/{request,response}``, exactly as
# the robot's own services do. These converters flatten those frames onto plain
# mappings so mock handlers stay free of vendor imports; the runtime side of
# this wire is exercised through the real SDK clients (see the coexistence
# test), which is the only honest way to prove the correlation contract.


def rpc_request_to_dict(message: Any) -> dict[str, Any]:
    """Flatten a ``unitree_api/Request_`` frame for a mock handler.

    The output mirrors the vendored IDL field tree: ``header.identity`` carries
    the correlation ``id`` and ``api_id``, ``header.lease``/``header.policy``
    the lease and policy, ``parameter`` the JSON string and ``binary`` the
    byte payload. Unknown api ids are not the converter's problem — the mock's
    handler table decides silence.
    """
    header = message.header
    return {
        "header": {
            "identity": {
                "id": int(header.identity.id),
                "api_id": int(header.identity.api_id),
            },
            "lease": {"id": int(header.lease.id)},
            "policy": {
                "priority": int(header.policy.priority),
                "noreply": bool(header.policy.noreply),
            },
        },
        "parameter": str(message.parameter),
        "binary": [int(value) for value in message.binary],
    }


def rpc_response_from_dict(payload: Mapping[str, Any]) -> Any:
    """Build a ``unitree_api/Response_`` from a plain mapping.

    ``payload`` needs ``header.identity.{id,api_id}`` (echoing the request's
    correlation pair — the vendor correlates on identity id alone) and
    ``header.status.code``; ``data`` is the JSON payload string, ``binary``
    the optional byte payload. The frame is built with all-args construction:
    the IDL types take required positional arguments (verified against the
    pinned revision), so the default factories are not reusable here.
    """
    require_vendor()
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import (  # noqa: PLC0415 - keeps the vendor import in this module
        RequestIdentity_,
        Response_,
        ResponseHeader_,
        ResponseStatus_,
    )

    header = payload.get("header") or {}
    identity = header.get("identity") or {}
    status = header.get("status") or {}
    frame_header = ResponseHeader_(
        RequestIdentity_(int(identity["id"]), int(identity["api_id"])),
        ResponseStatus_(int(status.get("code", 0))),
    )
    return Response_(
        frame_header,
        str(payload.get("data", "")),
        [int(value) for value in payload.get("binary", ())],
    )
