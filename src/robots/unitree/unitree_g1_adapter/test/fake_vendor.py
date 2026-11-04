"""An injectable fake of ``unitree_g1_adapter.vendor``'s surface.

Mirrors the real vendor behavior that the adapter depends on (verified
against the pinned upstream revision): publishers/subscribers with
Init/Write/Close, RPC clients that return bare codes or ``(code, data)``
tuples, and the command-frame converters. No DDS participant, no network,
no device.
"""

from __future__ import annotations

from typing import Any

RPC_OK = 0
RPC_TIMEOUT = 3104
RPC_SEND_ERR = 3102


class FakePublisher:
    def __init__(self, name: str, type_key: str, *, write_ok: bool = True) -> None:
        self.name = name
        self.type_key = type_key
        self.write_ok = write_ok
        self.init_calls = 0
        self.written: list[Any] = []
        self.closed = 0

    def Init(self) -> None:
        self.init_calls += 1

    def Write(self, sample: Any, timeout: Any = None) -> bool:
        self.written.append((sample, timeout))
        return self.write_ok

    def Close(self) -> None:
        self.closed += 1


class FakeSubscriber:
    def __init__(self, name: str, type_key: str) -> None:
        self.name = name
        self.type_key = type_key
        self.init_calls = 0
        self.handler = None
        self.queue_len = None
        self.closed = 0

    def Init(self, handler: Any = None, queue_len: int = 0) -> None:
        self.init_calls += 1
        self.handler = handler
        self.queue_len = queue_len

    def deliver(self, sample: Any) -> None:
        assert self.handler is not None
        self.handler(sample)

    def Close(self) -> None:
        self.closed += 1


class FakeRpcClient:
    """Records lifecycle; returns scripted outcomes per method."""

    def __init__(self, api: str, outcomes: dict[str, Any] | None = None) -> None:
        self.api = api
        self.outcomes = outcomes or {}
        self.init_calls = 0
        self.timeout: float | None = None

    def Init(self) -> None:
        self.init_calls += 1

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def _outcome(self, method: str, default: Any) -> Any:
        if method in self.outcomes:
            outcome = self.outcomes[method]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return default

    def GetFsmId(self):
        return self._outcome("GetFsmId", (RPC_OK, {"data": 5}))

    def GetFsmMode(self):
        return self._outcome("GetFsmMode", (RPC_OK, {"data": 0}))

    def ServiceList(self):
        return self._outcome("ServiceList", (RPC_OK, [{"name": "g1_arm_example", "status": 1}]))

    def SetFsmId(self, fsm_id: int):
        self.set_fsm_id_arg = fsm_id
        return self._outcome("SetFsmId", RPC_OK)

    def SetVelocity(self, vx: float, vy: float, omega: float, duration: float = 1.0):
        self.velocity_arg = (vx, vy, omega, duration)
        return self._outcome("SetVelocity", RPC_OK)

    def SwitchToUserCtrl(self):
        return self._outcome("SwitchToUserCtrl", RPC_OK)

    def SwitchToInternalCtrl(self, mode: Any):
        self.internal_ctrl_arg = mode
        return self._outcome("SwitchToInternalCtrl", RPC_OK)

    def ExecuteAction(self, action_id: int):
        self.action_arg = action_id
        return self._outcome("ExecuteAction", RPC_OK)

    def GetActionList(self):
        return self._outcome("GetActionList", (RPC_OK, {"data": []}))

    def TtsMaker(self, text: str, speaker_id: int):
        self.tts_arg = (text, speaker_id)
        return self._outcome("TtsMaker", RPC_OK)

    def LedControl(self, red: int, green: int, blue: int):
        self.led_arg = (red, green, blue)
        return self._outcome("LedControl", RPC_OK)

    def CheckMode(self):
        return self._outcome("CheckMode", (RPC_OK, {"data": {"name": "ai_sport"}}))

    def SelectMode(self, name_or_alias: str):
        self.select_arg = name_or_alias
        return self._outcome("SelectMode", RPC_OK)

    def ReleaseMode(self):
        return self._outcome("ReleaseMode", RPC_OK)


def fake_lowcmd(slots: Any, **kwargs: Any) -> dict[str, Any]:
    return {"kind": "low_cmd", "slots": list(slots), **kwargs}


def fake_handcmd(slots: Any) -> dict[str, Any]:
    return {"kind": "hand_cmd", "slots": list(slots)}


def make_fake_vendor(*, publisher_write_ok: bool = True, **overrides: Any) -> dict[str, Any]:
    """A fresh fake vendor surface plus the created objects for assertions."""
    state: dict[str, Any] = {
        "factory_calls": [],
        "publishers": [],
        "subscribers": [],
        "clients": {},
    }

    def factory_initialize(domain_id: int, network_interface: Any) -> None:
        state["factory_calls"].append((domain_id, network_interface))

    def channel_publisher(name: str, type_key: str) -> FakePublisher:
        publisher = FakePublisher(name, type_key, write_ok=publisher_write_ok)
        state["publishers"].append(publisher)
        return publisher

    def channel_subscriber(name: str, type_key: str) -> FakeSubscriber:
        subscriber = FakeSubscriber(name, type_key)
        state["subscribers"].append(subscriber)
        return subscriber

    def client(key: str, outcomes: dict[str, Any] | None = None) -> FakeRpcClient:
        if key not in state["clients"]:
            state["clients"][key] = FakeRpcClient(key, outcomes)
        return state["clients"][key]

    surface: dict[str, Any] = {
        "channel_factory_initialize": factory_initialize,
        "channel_publisher": channel_publisher,
        "channel_subscriber": channel_subscriber,
        "loco_client": lambda: client("loco"),
        "arm_action_client": lambda: client("arm_action"),
        "audio_client": lambda: client("audio"),
        "motion_switcher_client": lambda: client("motion_switcher"),
        "robot_state_client": lambda: client("robot_state"),
        "make_lowcmd": fake_lowcmd,
        "make_handcmd": fake_handcmd,
        "rpc_ok": RPC_OK,
        "rpc_timeout": RPC_TIMEOUT,
    }
    surface.update(overrides)
    surface["_state"] = state
    return surface
