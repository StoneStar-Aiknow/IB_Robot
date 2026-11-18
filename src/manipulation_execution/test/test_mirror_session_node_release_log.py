"""The node logs a release the tick made: the fault release shows in launch.log."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs")

from test_mirror_session_preset_core import make_rig  # noqa: E402

from manipulation_execution import mirror_session_node as module  # noqa: E402
from manipulation_execution.mirror_session.inputs import SessionInputs  # noqa: E402


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(text)


def node_for(session):
    logger = Logger()
    node = SimpleNamespace(
        session=session,
        _inputs=SimpleNamespace(take=lambda: SessionInputs()),
        _publish_enabled=True,
        _state_reporter=SimpleNamespace(messages=lambda transitions, snapshot, now: []),
        _state_publisher=SimpleNamespace(publish=lambda message: None),
        get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
    )
    return node, logger


def releases(logger):
    return [line for line in logger.lines if line.startswith("release (")]


def test_a_fault_release_made_by_the_tick_is_logged_once():
    rig = make_rig(with_preset=False)
    node, logger = node_for(rig.session)
    module.MirrorSessionNode._tick(node)
    assert releases(logger) == []
    rig.runtime.status = replace(rig.runtime.status, lifecycle="DEGRADED", faults=("joint feedback stale: arm",))
    rig.session.observe_status(rig.runtime.status)
    module.MirrorSessionNode._tick(node)
    module.MirrorSessionNode._tick(node)
    assert len(releases(logger)) == 1
    assert releases(logger)[0].startswith("release (fault): idle=True runtime returned to idle")
