"""Host-step driver: the runtime calls a platform's ``reset`` / ``step`` itself.

The second clock mode of the port. A platform-pull adapter is driven by the
platform's own main loop; a host-step platform only exposes reset and step, so this loop
plays that main loop. Both feed the same ``SimulationCore``: idle means no ``step`` call, a
finishing session steps with unchanged targets, and the platform's result goes to the
engine's result listener only.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, Protocol

from robot_runtime_sim.direct_control.engine import SimulationCore
from robot_runtime_sim.direct_control.ports import IkReply, IkRequest, PlatformCodec, PlatformResult


class HostStepEnv(Protocol):
    def reset(self, seed: int | None) -> Mapping[str, Any]: ...

    def step(self, native_action: Any) -> Mapping[str, Any]: ...

    def done(self) -> bool: ...

    def ticks(self) -> int: ...

    def native_result(self) -> PlatformResult: ...

    def solve_ik(self, request: IkRequest) -> IkReply: ...


class HostStepLoop:
    """Runs sessions of a host-step platform through one core (the only writer)."""

    WRITER = "host_step"

    def __init__(self, core: SimulationCore, codec: PlatformCodec, env: HostStepEnv):
        self._core = core
        self._codec = codec
        self._env = env
        self._abort = threading.Event()
        core.claim_writer(self.WRITER)

    def abort(self) -> None:
        self._abort.set()

    def run_session(self, info: Mapping[str, Any], seed: int | None = None) -> PlatformResult:
        core, env = self._core, self._env
        ident = core.reset()
        session_id, generation = ident["session_id"], ident["generation"]
        core.begin_session(info)
        native = env.reset(seed)
        core.on_observation(self._codec.decode_observation(native, session_id=session_id, generation=generation))
        replies: list[IkReply] = []
        while not env.done() and not self._abort.is_set():
            decision = core.next_tick(self.WRITER, session_id, generation, replies)
            replies = [env.solve_ik(request) for request in decision.ik_requests]
            if decision.targets is None:
                continue  # idle: physics is not stepped
            native = env.step(self._codec.encode_action(decision.targets))
            frame = self._codec.decode_observation(native, session_id=session_id, generation=generation)
            core.on_observation(frame)
        result = env.native_result()
        core.session_end({"session_id": session_id, "generation": generation}, result)
        return result
