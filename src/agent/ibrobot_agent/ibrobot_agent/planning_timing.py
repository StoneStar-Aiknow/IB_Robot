"""Request-local stage durations, available even when INFO tracing is disabled."""

import time
from contextlib import contextmanager
from contextvars import ContextVar

from embodied_common.tracing import trace_stage

_report = ContextVar("agent_planning_timing", default=None)


@contextmanager
def planning_report():
    stages = {
        name: {"status": "skipped", "sec": 0.0, "reason": "not selected"}
        for name in ("agent_catalog", "agent_lookup", "agent_model", "pose_preview")
    }
    token = _report.set(stages)
    try:
        yield stages
    finally:
        _report.reset(token)


def snapshot():
    return {k: dict(v) for k, v in (_report.get() or {}).items()}


@contextmanager
def planning_stage(logger, stage, **fields):
    name = {
        "agent.catalog": "agent_catalog",
        "agent.pose_lookup": "agent_lookup",
        "agent.planner": "agent_model",
        "agent.pose_preview": "pose_preview",
    }[stage]
    item = {"status": "executed", "sec": 0.0, "reason": ""}
    if _report.get() is not None:
        _report.get()[name] = item
    started = time.monotonic()
    try:
        with trace_stage(logger, stage, **fields):
            yield
    except Exception:
        item["status"] = "failed"
        raise
    finally:
        item["sec"] = max(0, time.monotonic() - started)
