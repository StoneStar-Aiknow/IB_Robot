"""Child process for the executor teardown failure tests.

``TEARDOWN_CHILD_MODE`` selects the scenario:

- ``timeout`` (default): a timer callback sleeps 30 s, so the production 2 s
  wait deterministically expires; sentinel exits 97/98 fail the test if the
  timeout branch touches the SDK or the node;
- ``close_error``: the SDK close raises; the node must not be destroyed;
- ``destroy_error``: the SDK close succeeds and ``destroy_node`` raises.

The non-timeout modes announce READY from a timer, so the parent's SIGTERM
arrives while the executor is running.
"""

from __future__ import annotations

import os
import threading
import time

from rclpy.node import Node
from unitree_g1_robot.runtime_node import main

MODE = os.environ.get("TEARDOWN_CHILD_MODE", "timeout")


class _Sdk:
    def close(self) -> None:
        if MODE == "close_error":
            raise RuntimeError("fake SDK close failure")
        print("SDK-CLOSED", flush=True)
        if MODE == "timeout":
            raise SystemExit(97)


class _ChildNode(Node):
    def __init__(self) -> None:
        super().__init__("teardown_timeout_child")
        self._sdk = _Sdk()
        self.entered = threading.Event()
        self.announced = False
        if MODE == "timeout":
            self.create_timer(0.02, self._slow_callback)
        else:
            self.create_timer(0.05, self._announce_ready)

    def _slow_callback(self) -> None:
        if self.entered.is_set():
            return
        self.entered.set()
        print("CALLBACK-ENTERED", flush=True)
        time.sleep(30.0)

    def _announce_ready(self) -> None:
        if self.announced:
            return
        self.announced = True
        print("READY", flush=True)

    def destroy_node(self) -> bool:
        print("NODE-DESTROYED", flush=True)
        if MODE == "destroy_error":
            raise RuntimeError("fake node destroy failure")
        if MODE == "timeout":
            raise SystemExit(98)
        return super().destroy_node()


if __name__ == "__main__":
    raise SystemExit(main(node_factory=_ChildNode))
