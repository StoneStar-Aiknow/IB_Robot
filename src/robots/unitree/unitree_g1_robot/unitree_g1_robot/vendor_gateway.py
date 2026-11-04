"""The runtime's vendor boundary: everything comes from the official SDK adapter.

Since PR 1 (AtomGit Issue #171), the vendor surface is the official
``unitree_sdk2py``, reached only through the ROS-free adapter package
``unitree_g1_adapter``; the hand-written hand-rolled RPC/transport layer this
module used to contain is gone. What remains here is the *naming and shape*
layer between the runtime and that adapter:

- the mock- and bridge-facing names of the vendor DDS frames
  (``LowState_``/``LowCmd_``/``HandState_``/``HandCmd_`` and the
  ``unitree_api`` RPC frames) come from the adapter's injectable surface, so
  the runtime and the mock keep importing vendor types from exactly one place
  (this module) while the adapter stays the only importer of vendor symbols;
- the QoS resolution the mock needs (the runtime's own RPC clients now run on
  the SDK's clients, which own their transport and need none);
- :class:`RpcResult`, the outcome type the runtime's call sites consume,
  mirrored from the adapter's :class:`~unitree_g1_adapter.clients.SdkCallResult`
  with the timeout sentinel mapped the same way (the vendor's 3104).

The legacy ``RpcClient`` (identity correlation over the raw RPC topics) was
deleted with the vendored ``unitree_api`` messages it served: the SDK clients
correlate internally and the mock answers on the same wire through the
adapter's ``rpc_request``/``rpc_response`` converters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unitree_g1_adapter import vendor
from unitree_g1_adapter.clients import SdkCallResult

#: The adapter is the vendor boundary; re-export its availability guard so the
#: runtime's startup check keeps a single import site.
VENDOR_AVAILABLE = vendor.vendor_available()
VENDOR_IMPORT_ERROR = "" if VENDOR_AVAILABLE else "unitree_sdk2py is not installed"


class VendorUnavailableError(RuntimeError):
    """Raised when the official vendor SDK is not on the Python path."""

    def __init__(self) -> None:
        super().__init__(
            "unitree_sdk2py is not available: run the workspace setup (scripts/setup.sh installs "
            "the pinned SDK via scripts/setup/install_unitree_sdk.sh) before launching the G1 "
            "runtime. Install error: " + VENDOR_IMPORT_ERROR
        )


def require_vendor() -> None:
    if not VENDOR_AVAILABLE:
        raise VendorUnavailableError()


# --- vendor frame types (via the adapter's injectable IDL registry) -------------


def _idl(type_key: str) -> Any:
    """One vendor IDL *class* through the adapter's registry.

    The adapter resolves lazily and refuses an SDK-less host with a guided
    error; this wrapper keeps that behavior but re-raises it as the runtime's
    own :class:`VendorUnavailableError` so launch-time failures name the
    runtime's setup path.
    """
    try:
        return vendor.idl_type(type_key)
    except Exception as exc:  # noqa: BLE001 - the adapter raises a bare error type per its contract
        if not vendor.vendor_available():
            raise VendorUnavailableError() from exc
        raise


def low_state_type() -> Any:
    return _idl("low_state")


def low_cmd_type() -> Any:
    return _idl("low_cmd")


def hand_state_type() -> Any:
    return _idl("hand_state")


def hand_cmd_type() -> Any:
    return _idl("hand_cmd")


def rpc_request_type() -> Any:
    return _idl("rpc_request")


def rpc_response_type() -> Any:
    return _idl("rpc_response")


#: Backwards-compatible names for the runtime modules and the mock: resolved
#: once on first attribute access through the adapter's IDL registry (module
#: __getattr__, PEP 562), so importing this module stays SDK-free and the
#: resolved value is the real vendor IDL class rclpy can wire.
_TYPE_KEY_BY_NAME = {
    "LowState": "low_state",
    "LowCmd": "low_cmd",
    "HandState": "hand_state",
    "HandCmd": "hand_cmd",
    "Request": "rpc_request",
    "Response": "rpc_response",
}


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attribute
    if name in _TYPE_KEY_BY_NAME:
        resolved = _idl(_TYPE_KEY_BY_NAME[name])
        globals()[name] = resolved  # cache: later accesses skip the registry
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def motor_state() -> Any:
    """One vendor ``MotorCmd_``-shaped motor state slot factory (mock aid)."""
    return vendor.motor_state_defaults()


def motor_cmd_slot() -> Any:
    """One vendor ``MotorCmd_`` slot factory (mock aid)."""
    return vendor.motor_cmd_defaults()


def press_sensor_state() -> Any:
    """One vendor ``PressSensorState_`` slot factory (mock aid)."""
    return vendor.press_sensor_defaults()


# --- vendor sample readers (delegating to the adapter's converters) -------------


def lowstate_q(message: Any) -> list[float]:
    """The 35-entry ``q`` vector of a ``LowState_`` sample, as plain floats."""
    return vendor.lowstate_q(message)


def lowstate_mode_machine(message: Any) -> int:
    """``LowState_.mode_machine`` (design D10's startup generation check reads this)."""
    return vendor.lowstate_mode_machine(message)


def lowstate_imu(message: Any) -> dict[str, list[float]]:
    """``LowState_.imu_state`` as plain lists (the adapter copies arrays verbatim)."""
    return vendor.lowstate_imu(message)


def handstate_q(message: Any) -> list[float]:
    """One ``HandState_`` sample's motor positions in vendor motor order."""
    return vendor.handstate_motor_q(message)


def rpc_request_to_dict(message: Any) -> dict[str, Any]:
    """Flatten a vendor RPC ``Request_`` frame for a mock handler."""
    return vendor.rpc_request_to_dict(message)


def rpc_response_from_dict(payload: Mapping[str, Any]) -> Any:
    """Build a vendor RPC ``Response_`` frame from a plain mapping."""
    return vendor.rpc_response_from_dict(payload)


# --- RPC outcome -----------------------------------------------------------------


def rpc_result_from_sdk(result: SdkCallResult, *, motion: bool = False, api_id: str = "") -> RpcResult:
    """Map the adapter's :class:`SdkCallResult` onto the runtime's outcome type.

    ``ok``/``code``/``data``/``timed_out`` carry over verbatim (the adapter
    already maps the vendor's timeout sentinel); ``motion`` is re-attached by
    the call site because the SDK clients do not tag requests as
    motion-changing — the runtime's stop contract needs the flag and the
    runtime is where the motion/no-motion classification lives.
    """
    return RpcResult(
        ok=result.ok,
        code=result.code,
        data=result.data,
        timed_out=result.timed_out,
        motion=motion,
        api_id=api_id,
    )


class RpcResult:
    """Outcome of one vendor RPC call (runtime-facing shape).

    ``code`` is the vendor status for a call that was answered (0 = success);
    ``timed_out`` marks the SDK's client-side timeout sentinel (3104) — the
    "fate unknown" outcome that a motion request must never be re-sent from
    (the adapter's clients never retry, and neither do the runtime's call
    sites). ``dropped`` is the opposite of unknown: the runtime refused to send
    this call (its guard saw a newer stop generation or a closed admission), so
    the vendor never saw it and there is nothing to retry. ``api_id`` is a
    symbolic name for logs, not the wire value.
    """

    __slots__ = ("ok", "code", "data", "timed_out", "motion", "api_id", "dropped")

    def __init__(
        self,
        *,
        ok: bool,
        code: int,
        data: Any = None,
        timed_out: bool = False,
        motion: bool = False,
        api_id: str = "",
        dropped: bool = False,
    ) -> None:
        self.ok = bool(ok)
        self.code = int(code)
        self.data = data
        self.timed_out = bool(timed_out)
        self.motion = bool(motion)
        self.api_id = str(api_id)
        self.dropped = bool(dropped)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RpcResult(ok={self.ok}, code={self.code}, timed_out={self.timed_out}, "
            f"dropped={self.dropped}, motion={self.motion}, api_id={self.api_id!r})"
        )
