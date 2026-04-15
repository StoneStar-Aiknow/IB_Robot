"""Shared helpers for IB-Robot tracing loggers."""

from __future__ import annotations

import logging

try:
    import lttngust  # noqa: F401
    from lttngust import loghandler as _lttng_loghandler
except ImportError:
    _lttng_loghandler = None


def create_trace_logger(name: str) -> logging.Logger:
    """Create a logger wired to the LTTng Python agent when available."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if _lttng_loghandler is not None and not any(
        isinstance(handler, _lttng_loghandler._Handler) for handler in logger.handlers
    ):
        try:
            # lttngust exposes only the private _Handler entrypoint today.
            logger.addHandler(_lttng_loghandler._Handler())
            logger.propagate = False
        except OSError as exc:
            logging.getLogger(__name__).warning("LTTng Python log handler unavailable for %s: %s", name, exc)
    return logger
