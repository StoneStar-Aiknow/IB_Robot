"""Bounded ROS shutdown with explicit owner-drain evidence."""

import signal
import threading
import time

import rclpy

from .runtime_protocol import DEFAULT_CONTROL_RPC_TIMEOUT_NS


def spin_until_hosts_close(node, executor, *, timeout_sec=60.0):
    """Keep Global's ROS transport alive after the launch-wide stop signal.

    Hosts receive the same signal and stop serving before Close.
    Unknown owners remain pending until the bounded shutdown deadline; timeout
    is reported as failure, never interpreted as evidence of a clean drain.
    """
    stopping = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopping.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    deadline = None
    drained_at = None
    try:
        while rclpy.ok(context=node.context):
            if stopping.is_set():
                if deadline is None:
                    node.begin_shutdown()
                    deadline = time.monotonic() + timeout_sec
                pending = node.shutdown_pending()
                if not pending:
                    if drained_at is None:
                        drained_at = time.monotonic()
                    # Hosts retry unregister after a bounded reply wait. Keep
                    # serving for one control RPC budget even after the final
                    # owner is drained; a completed callback is not delivery.
                    if time.monotonic() - drained_at >= DEFAULT_CONTROL_RPC_TIMEOUT_NS / 1e9:
                        return
                else:
                    drained_at = None
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Global shutdown did not drain runtime owners: {pending}")
            executor.spin_once(timeout_sec=0.05)
            collect = getattr(node, "collect_retired_transport", None)
            if collect is not None:
                collect(executor)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
