"""The demo start/cancel rules (``DemoHooks``), with and without ``cancel_on_actor_loss``. No ROS."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from manipulation_execution.mirror_session.business import load_business_config
from manipulation_execution.mirror_session.demo import DemoHooks
from manipulation_execution.mirror_session.hooks import ActorSignal, PearWindow, SessionHooks, SessionInputs

SHIPPED_DEMO = Path(__file__).resolve().parents[2] / "robot_config" / "config" / "mirror_session" / "demo.yaml"
TICK = 0.1


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(**session):
    clock = Clock()
    return DemoHooks(session, None, inner=SessionHooks(), clock=clock), clock


def frame(source="yolox", height=400.0):
    return SimpleNamespace(source=source, bbox=[100.0, 100.0, 300.0, 100.0 + height])


def actor(present=True, age_s=0.0):
    return ActorSignal(present=present, actor_id=3 if present else None, stamp_s=0.0, age_s=age_s)


def inputs(actor_signal, *frames):
    return SessionInputs(window=PearWindow(frames=tuple(frames)), actor=actor_signal)


def cancel_over(hooks, clock, seconds, make_inputs):
    """Run cancel_criteria every tick for ``seconds``; return the set of reasons seen."""
    reasons = set()
    for _ in range(round(seconds / TICK)):
        clock.now += TICK
        hooks.cancel_criteria(make_inputs(), 0.0)
        reasons.add(hooks.cancel_reason)
    return reasons


def mirroring(hooks, clock):
    """A person locked and a yolox frame: what starts a round."""
    assert hooks.start_criteria(inputs(actor(), frame()))
    clock.now += TICK
    assert not hooks.cancel_criteria(inputs(actor(), frame()), 0.0)


# ---------------------------------------------------------------------------
# the switch unset: the rules as before
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session", [{}, {"cancel_on_actor_loss": True}])
def test_unset_or_true_cancels_on_a_lost_actor_as_before(session):
    hooks, clock = make(**session)
    mirroring(hooks, clock)
    assert hooks.cancel_criteria(inputs(actor(present=False), frame("center_fallback")), 0.0)
    assert hooks.cancel_reason == "no_actor"
    assert hooks.cancel_criteria(inputs(None), 0.0)
    assert hooks.cancel_reason == "no_actor"


def test_unset_center_fallback_frames_are_no_pose():
    hooks, clock = make()
    mirroring(hooks, clock)
    reasons = cancel_over(hooks, clock, 1.5, lambda: inputs(actor(), frame("center_fallback")))
    assert reasons == {"", "no_pose"}
    assert hooks.cancel_reason == "no_pose"


def test_unset_too_far_and_stale_as_before():
    hooks, clock = make(min_box_height_px=300)
    mirroring(hooks, clock)
    assert hooks.cancel_criteria(inputs(actor(), frame(height=200.0)), 0.0)
    assert hooks.cancel_reason == "too_far"
    hooks, clock = make()
    mirroring(hooks, clock)
    assert hooks.cancel_criteria(inputs(actor(age_s=1.5), frame()), 0.0)
    assert hooks.cancel_reason == "stale"


# ---------------------------------------------------------------------------
# cancel_on_actor_loss: false
# ---------------------------------------------------------------------------


def test_off_a_lost_actor_does_not_end_the_round_while_frames_come():
    hooks, clock = make(cancel_on_actor_loss=False)
    mirroring(hooks, clock)
    reasons = cancel_over(hooks, clock, 15.0, lambda: inputs(actor(present=False), frame("center_fallback")))
    assert reasons == {""}


def test_off_the_person_comes_back_and_the_round_goes_on():
    hooks, clock = make(cancel_on_actor_loss=False)
    mirroring(hooks, clock)
    assert cancel_over(hooks, clock, 12.0, lambda: inputs(actor(present=False), frame("center_fallback"))) == {""}
    assert cancel_over(hooks, clock, 3.0, lambda: inputs(actor(), frame())) == {""}


@pytest.mark.parametrize("source", ["yolox", "center_fallback"])
def test_off_no_frame_of_any_source_for_pear_max_age_s_ends_the_round(source):
    hooks, clock = make(cancel_on_actor_loss=False)
    mirroring(hooks, clock)
    assert cancel_over(hooks, clock, 0.5, lambda: inputs(actor(present=False), frame(source))) == {""}
    # the stream stops: ActorState still fresh (age 0), no PEAR frames
    reasons = cancel_over(hooks, clock, 0.9, lambda: inputs(actor(present=False)))
    assert reasons == {""}
    clock.now += 2 * TICK
    assert hooks.cancel_criteria(inputs(actor(present=False)), 0.0)
    assert hooks.cancel_reason == "no_pose"


def test_off_a_stale_or_never_received_actor_state_ends_the_round():
    hooks, clock = make(cancel_on_actor_loss=False)
    mirroring(hooks, clock)
    assert hooks.cancel_criteria(inputs(actor(present=False, age_s=1.5), frame("center_fallback")), 0.0)
    assert hooks.cancel_reason == "stale"
    assert hooks.cancel_criteria(inputs(None, frame("center_fallback")), 0.0)
    assert hooks.cancel_reason == "stale"
    assert not hooks.cancel_criteria(inputs(actor(present=False, age_s=0.9), frame("center_fallback")), 0.0)


def test_off_too_far_does_not_end_the_round():
    hooks, clock = make(cancel_on_actor_loss=False, min_box_height_px=300)
    assert hooks.start_criteria(inputs(actor(), frame(height=400.0)))
    assert cancel_over(hooks, clock, 5.0, lambda: inputs(actor(), frame(height=200.0))) == {""}


def test_off_start_is_unchanged():
    hooks, clock = make(cancel_on_actor_loss=False, min_box_height_px=300)
    assert not hooks.start_criteria(inputs(actor(present=False), frame("center_fallback")))
    assert not hooks.start_criteria(inputs(actor(), frame("center_fallback")))  # no yolox frame yet
    assert not hooks.start_criteria(inputs(None, frame()))
    assert not hooks.start_criteria(inputs(actor(present=False), frame()))
    assert not hooks.start_criteria(inputs(actor(age_s=1.5), frame()))
    assert not hooks.start_criteria(inputs(actor(), frame(height=200.0)))  # too far
    assert hooks.start_criteria(inputs(actor(), frame()))
    clock.now += 1.5
    assert not hooks.start_criteria(inputs(actor(), frame("center_fallback")))  # the yolox frame is too old


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, "false", None, 1.0])
def test_the_switch_must_be_a_bool(value):
    with pytest.raises(ValueError, match="cancel_on_actor_loss"):
        make(cancel_on_actor_loss=value)


def test_unknown_keys_are_still_refused():
    with pytest.raises(ValueError, match="cancel_on_actor_los"):
        make(cancel_on_actor_los=False)


def test_the_shipped_demo_config_turns_the_switch_off():
    session = load_business_config(str(SHIPPED_DEMO)).session
    assert session["cancel_on_actor_loss"] is False
    hooks = DemoHooks(session, None, inner=SessionHooks())
    assert hooks._cancel_on_actor_loss is False
