"""Vendor high-level RPC client wrappers with one result type.

The vendor clients return either a bare status code (``SetVelocity``) or a
``(code, data)`` tuple (``GetFsmId``), and report their client-side timeout as
the sentinel code 3104. Every wrapper here normalizes that onto
:class:`SdkCallResult` so callers never compare raw vendor numbers.

Behavior contract (verified vendor sources, and PR 2's stop rule):

- The vendor client never retries; neither do these wrappers.
- ``timed_out=True`` means "fate unknown": a motion request in that state is
  never re-sent, and a successful return does not prove the robot finished
  the motion — it proves the vendor accepted the request.
- These calls block for up to the configured timeout on the calling thread.
  Which calls run on which thread is the runtime's decision (PR 2); the
  adapter deliberately owns no worker threads.
- The timeout is per *call*, not only per client: the vendor reads one
  ``ClientBase.__timeout`` for every request/response pair, so a short query
  uses :meth:`_ClientBase.call_timeout` to set it for its own call and restore
  the configured one afterwards. Serializing callers per client is what keeps
  one call's window from leaking into a concurrent one; that serialization
  belongs to the runtime's per-service dispatcher, not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from unitree_g1_adapter import vendor as vendor_module

#: The vendor client's documented default (``ClientBase.__timeout = 1.0``).
DEFAULT_CALL_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class SdkCallResult:
    """One vendor RPC outcome.

    ``code`` is the vendor status (0 = accepted) or the client-side error
    code; ``timed_out`` marks the vendor timeout sentinel so callers can
    distinguish "refused with a code" from "no answer at all".
    """

    ok: bool
    code: int
    data: Any = None
    timed_out: bool = False


class _ClientBase:
    """Shared lifecycle for the typed wrappers below."""

    api_name = "vendor"

    def __init__(self, *, vendor: Mapping[str, Any] | None = None, timeout_s: float = DEFAULT_CALL_TIMEOUT_S) -> None:
        self._vendor = vendor if vendor is not None else vendor_module.default_vendor()
        if float(timeout_s) <= 0.0:
            raise ValueError(f"{type(self).__name__} timeout_s must be positive, got {timeout_s!r}")
        self.timeout_s = float(timeout_s)
        self._client = self._make_client()
        self._client.Init()
        self._client.SetTimeout(self.timeout_s)
        self._closed = False

    def _make_client(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        """Mark the wrapper closed; the vendor client has no close method."""
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    @contextmanager
    def call_timeout(self, timeout_s: float | None = None):
        """Run the enclosed call under ``timeout_s``, then restore the default.

        The vendor keeps one timeout per client object and reads it for every
        request/response pair, so a single short query has to set it for its
        own call. ``None`` leaves the configured timeout untouched. The caller
        must serialize calls to one client (the runtime does, per service)
        between the set and the restore, otherwise two callers would race on
        the same field; this context manager only owns set/restore and the
        restore is in a ``finally`` so a raising call cannot leak its window.
        """
        if timeout_s is None:
            yield
            return
        value = float(timeout_s)
        if value <= 0.0:
            raise ValueError(f"call timeout_s must be positive, got {timeout_s!r}")
        previous = self.timeout_s
        self._client.SetTimeout(value)
        try:
            yield
        finally:
            self._client.SetTimeout(previous)

    def _map(self, api: str, outcome: Any) -> SdkCallResult:
        if isinstance(outcome, tuple):
            code, data = (outcome + (None,))[:2]
        else:
            code, data = outcome, None
        code = int(code)
        ok = code == int(self._vendor["rpc_ok"])
        timed_out = code == int(self._vendor["rpc_timeout"])
        return SdkCallResult(ok=ok, code=code, data=data, timed_out=timed_out)


class LocoClient(_ClientBase):
    """The vendor ``sport`` service: FSM modes and velocity commands."""

    api_name = "loco"

    def _make_client(self) -> Any:
        return self._vendor["loco_client"]()

    def get_fsm_id(self) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.GetFsmId())

    def get_fsm_mode(self) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.GetFsmMode())

    def set_fsm_id(self, fsm_id: int) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.SetFsmId(int(fsm_id)))

    def set_velocity(self, vx: float, vy: float, omega: float, duration: float = 1.0) -> SdkCallResult:
        """A motion request: a ``timed_out`` result must never be re-sent."""
        self._require_open()
        return self._map(self.api_name, self._client.SetVelocity(float(vx), float(vy), float(omega), float(duration)))

    def switch_to_user_ctrl(self) -> SdkCallResult:
        """Full-body user_lowcmd takeover; NOT an arm_sdk prerequisite."""
        self._require_open()
        return self._map(self.api_name, self._client.SwitchToUserCtrl())

    def switch_to_internal_ctrl(self, mode: Any) -> SdkCallResult:
        """A motion request; ``mode`` is the vendor ``InternalFsmMode``."""
        self._require_open()
        return self._map(self.api_name, self._client.SwitchToInternalCtrl(mode))


class RobotStateClient(_ClientBase):
    """Read-only service ownership discovery; never switches services."""

    api_name = "robot_state"

    def _make_client(self) -> Any:
        return self._vendor["robot_state_client"]()

    def service_list(self) -> SdkCallResult:
        self._require_open()
        result = self._map(self.api_name, self._client.ServiceList())
        if not result.ok:
            return result
        return SdkCallResult(
            ok=True,
            code=result.code,
            data=result.data,
        )


class ArmActionClient(_ClientBase):
    """The vendor ``arm`` service: preset arm actions."""

    api_name = "arm_action"

    def _make_client(self) -> Any:
        return self._vendor["arm_action_client"]()

    def execute_action(self, action_id: int) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.ExecuteAction(int(action_id)))

    def get_action_list(self) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.GetActionList())


class AudioClient(_ClientBase):
    """The vendor ``voice`` service: TTS and the RGB LED strip.

    Upstream quirk (pinned source): ``TtsMaker`` sends a constant ``index``
    (its ``tts_index += self.tts_index`` never leaves 0). The wrapper passes
    the vendor call through unchanged.
    """

    api_name = "audio"

    def _make_client(self) -> Any:
        return self._vendor["audio_client"]()

    def tts_maker(self, text: str, speaker_id: int = 0) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.TtsMaker(str(text), int(speaker_id)))

    def led_control(self, red: int, green: int, blue: int) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.LedControl(int(red), int(green), int(blue)))

    def get_volume(self) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.GetVolume())


class MotionSwitcherClient(_ClientBase):
    """The vendor ``motion_switcher`` service."""

    api_name = "motion_switcher"

    def _make_client(self) -> Any:
        return self._vendor["motion_switcher_client"]()

    def check_mode(self) -> SdkCallResult:
        self._require_open()
        return self._map(self.api_name, self._client.CheckMode())

    def select_mode(self, name_or_alias: str) -> SdkCallResult:
        """A motion request (switches the vendor motion service)."""
        self._require_open()
        return self._map(self.api_name, self._client.SelectMode(str(name_or_alias)))

    def release_mode(self) -> SdkCallResult:
        """A motion request (releases the vendor motion service)."""
        self._require_open()
        return self._map(self.api_name, self._client.ReleaseMode())
