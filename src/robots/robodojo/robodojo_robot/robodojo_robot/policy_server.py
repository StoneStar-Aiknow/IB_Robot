"""Host RoboDojo's websocket policy server inside the runtime process.

The RoboDojo eval client connects to a policy server and calls ``reset`` /
``update_obs`` / ``get_action`` once per tick. Here that server's model is the
adapter's ``RoboDojoPolicyModel``, so the ROS runtime and RoboDojo's main loop meet
in one process without a relay. The server code is RoboDojo's own
(``XPolicyLab/client_server/ws``), imported from ``robodojo_root``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any


def import_policy_server(robodojo_root: str):
    root = Path(robodojo_root).expanduser().resolve()
    if not (root / "XPolicyLab" / "client_server").is_dir():
        raise FileNotFoundError(f"{root} has no XPolicyLab/client_server; is robodojo_root a RoboDojo checkout?")
    for path in (root / "XPolicyLab", root):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    return PolicyServer, PolicyServerConfig


class PolicyServerThread:
    """RoboDojo's ``PolicyServer`` on its own asyncio loop in a daemon thread."""

    def __init__(self, model: Any, *, robodojo_root: str, host: str, port: int, logger=None):
        policy_server, policy_server_config = import_policy_server(robodojo_root)
        # The eval client blocks in IK and physics between calls; keep pings tolerant.
        config = policy_server_config(host=host, port=int(port), ws_ping_interval_s=20.0, ws_ping_timeout_s=60.0)
        self._server = policy_server(model, config)
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._logger = logger
        self._thread = threading.Thread(target=self._run, name="robodojo_policy_server", daemon=True)

    @property
    def url(self) -> str:
        return self._server.url

    def start(self, timeout_s: float = 10.0) -> None:
        self._thread.start()
        if not self._started.wait(timeout_s):
            raise RuntimeError("RoboDojo policy server did not start")
        if self._error is not None:
            raise RuntimeError(f"RoboDojo policy server failed to start: {self._error!r}") from self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._server.start())
        except BaseException as exc:  # reported by start()
            self._error = exc
            self._started.set()
            return
        self._started.set()
        self._loop.run_forever()

    def stop(self) -> None:
        if not self._loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop)
        try:
            future.result(timeout=5.0)
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(f"policy server stop: {exc!r}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
