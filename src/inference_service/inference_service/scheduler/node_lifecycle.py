"""Bounded ROS shutdown with explicit owner-drain evidence."""

import signal
import threading
import time

import rclpy


def spin_until_hosts_close(node, executor, *, timeout_sec=60.0):
    """Keep Global's ROS transport alive after the launch-wide stop signal.

    Hosts receive the same signal and stop serving before Close.
    Unknown owners remain pending until the bounded shutdown deadline; timeout
    is reported as failure, never interpreted as evidence of a clean drain.
    """
    stopping = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopping.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    deadline = None
    try:
        while rclpy.ok(context=node.context):
            if stopping.is_set():
                if deadline is None:
                    node.begin_shutdown()
                    deadline = time.monotonic() + timeout_sec
                pending = node.shutdown_pending()
                if not pending:
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Global shutdown did not drain runtime owners: {pending}")
            executor.spin_once(timeout_sec=0.05)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
