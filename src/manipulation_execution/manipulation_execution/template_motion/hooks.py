"""``TemplateBlendHooks``: the session hooks with the template blender around the mirror pipeline.

Wraps the hooks ``build_hooks`` made (demo rules around the mock playback)
and changes only what reaches the gate:

* ``retarget`` asks the inner hooks first, keeps the newest mirror command
  (not the blended command the mock limiter may hand back as "unreached"),
  and lets the blender replace it during A, the playback and C. While the
  blender is IDLE or cooling down the inner command passes unchanged, the
  same object, so the mirror is exactly the baseline;
* ``post_process`` is the inner one, for the mirror and the blended commands;
* ``end_round`` releases a playback fast, ``wrap_up`` waits for it to end.

A template is a nice-to-have: every hook catches its own errors, logs them,
turns templates off for the rest of the round and returns what the inner
hooks returned. The inner hooks are called outside those guards, so their
behaviour (and errors) are the baseline's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from manipulation_execution.mirror_session.endpoints import HAND_JOINT_COUNT
from manipulation_execution.mirror_session.hooks import (
    SessionHooks,
    SessionInputs,
    UpperBodyCommand,
    WrapUpContext,
    WrapUpStep,
)
from manipulation_execution.template_motion.blender import Blender

_ZERO_HAND = (0.0,) * HAND_JOINT_COUNT


class TemplateBlendHooks(SessionHooks):
    def __init__(
        self,
        inner: SessionHooks,
        blender: Blender,
        log_error: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._inner = inner
        self._blender = blender
        self._log_error = log_error
        self._clock = clock
        self._mirror: UpperBodyCommand | None = None
        self._last_blended: UpperBodyCommand | None = None
        self.errors = 0

    @property
    def cancel_reason(self) -> str:  # the core reads it from the hooks it holds
        return self._inner.cancel_reason

    @property
    def blender(self) -> Blender:
        return self._blender

    def _fail(self, where: str, exc: BaseException) -> None:
        self.errors += 1
        try:
            self._log_error(f"template {where} failed, templates off for this round: {exc!r}")
            self._blender.disable_round(type(exc).__name__, self._clock())
        except Exception:
            pass

    # -- passed through -------------------------------------------------------------

    def start_criteria(self, inputs: SessionInputs) -> bool:
        return self._inner.start_criteria(inputs)

    def cancel_criteria(self, inputs: SessionInputs, round_elapsed_s: float) -> bool:
        return self._inner.cancel_criteria(inputs, round_elapsed_s)

    def cooldown_over(self, elapsed_s: float) -> bool:
        return self._inner.cooldown_over(elapsed_s)

    def reset_pose(self, arm_at_takeover: Sequence[float] | None) -> UpperBodyCommand | None:
        return self._inner.reset_pose(arm_at_takeover)

    def post_process(self, command: UpperBodyCommand) -> UpperBodyCommand | None:
        return self._inner.post_process(command)

    # -- round --------------------------------------------------------------------

    def begin_round(self) -> None:
        self._inner.begin_round()
        self._mirror = self._last_blended = None
        try:
            self._blender.begin_round()
        except Exception as exc:
            self._fail("begin_round", exc)

    def end_round(self) -> None:
        try:
            self._blender.end_round(self._clock())
        except Exception as exc:
            self._fail("end_round", exc)
        self._inner.end_round()

    # -- the pipeline -------------------------------------------------------------

    def retarget(self, inputs: SessionInputs) -> UpperBodyCommand | None:
        raw = self._inner.retarget(inputs)
        try:
            if raw is not None and raw is not self._last_blended:
                self._mirror = raw
            mirror = self._mirror
            arm = self._blender.step(self._clock(), inputs, None if mirror is None else mirror.arm)
            if arm is None:
                return raw
            command = UpperBodyCommand(tuple(arm), _ZERO_HAND, None if mirror is None else mirror.source_stamp_s)
            self._last_blended = command
            return command
        except Exception as exc:
            self._fail("retarget", exc)
            return None if raw is self._last_blended else raw

    def wrap_up(self, context: WrapUpContext) -> WrapUpStep:
        try:
            step = self._blender.wrap(context, lambda arm: UpperBodyCommand(tuple(arm), _ZERO_HAND))
        except Exception as exc:
            self._fail("wrap_up", exc)
            step = None
        if step is not None:
            return step
        return self._inner.wrap_up(context)

    # -- the hand overlay (node, under the session lock) -----------------------------

    def hand_target(self, now: float) -> tuple[float, ...] | None:
        try:
            return self._blender.hand_target(now)
        except Exception as exc:
            self._fail("hand_target", exc)
            return None

    def observe_hand_output(self, values: Sequence[float] | None, held_state: int | None = None) -> None:
        try:
            self._blender.observe_hand_output(values, held_state)
        except Exception as exc:
            self._fail("observe_hand_output", exc)

    def snapshot(self) -> dict:
        try:
            return {**self._blender.snapshot(), "errors": self.errors}
        except Exception as exc:
            return {"errors": self.errors, "snapshot_error": repr(exc)}
