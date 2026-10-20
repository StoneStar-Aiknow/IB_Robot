"""Extension points of the visual interaction session. Every default is a no-op.

The core calls these at fixed points of one observation round and owns nothing
they decide. Three are ports to something outside the session:

``VisionLanguagePort``
    Looks at a window and answers what is worth saying. The default answers
    "nothing", so the skeleton runs and stays silent without a model. The
    board's InternVL loop replaces it once phase zero has measured it;
    :class:`~visual_interaction.session.fake_vl.ScriptedVision` replaces it in
    tests.

``SpeechPort``
    Delivers one reply. The default records it and reports an estimated end,
    which is all the first version does: the utterance is published and logged.

    TODO(speech-coordinator): speaking out loud needs arbitration this repo
    does not have. Three sources already call /voice_tts/synthesize and
    /voice_tts/play unaware of each other — the visual-game announcer, the
    agent's voice bridge and the skill CLI hook — with only per-process locks
    between them, and the playback node serialises requests FIFO with no
    priority, so a loser blocks for up to its whole timeout. There is no
    cross-process speaking lock, no is_speaking state, no budget, and
    PlayAudioFile has no request-level cancel at all. A coordinator owning the
    lease, the budget, the deadline and cancellation is a prerequisite, not an
    enhancement. It implements this port; the session does not change.

``ChangeGate``
    Decides whether a window is worth a model call. The default says yes to
    every window: a prefill is a fixed cost whatever the picture, so skipping
    one is the only real saving, but skipping the wrong one is a missed answer.
    Measuring that trade-off is phase zero's job.

Inputs are plain dataclasses the ROS adapter fills once per round. Nothing here
imports ROS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """The frames one round looks at, and when they were seen.

    ``stamps`` are observation times of the frames, oldest first; ``frames``
    are whatever the port understands (encoded bytes, arrays) and the session
    never inspects them. ``received_at`` is when the newest frame *arrived*,
    on the host's own clock.

    Both clocks are carried because they answer different questions. Whether
    the stream is alive is judged by arrival, which needs no agreement between
    two machines; how old the evidence is must use the camera stamp, because
    that is what the picture shows. The board demo learned this the hard way
    and judges a dropout by arrival time only.
    """

    frames: tuple[object, ...]
    stamps: tuple[float, ...]
    received_at: float

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.stamps):
            raise ValueError("every frame needs a stamp")
        if not self.frames:
            raise ValueError("a window needs at least one frame")

    @property
    def newest_stamp(self) -> float:
        return self.stamps[-1]

    @property
    def oldest_stamp(self) -> float:
        return self.stamps[0]

    @property
    def span(self) -> float:
        return self.newest_stamp - self.oldest_stamp


@dataclass(frozen=True)
class ActorSignal:
    """Who the perception is following, as the session needs it.

    Mirrors the fields of ``ibrobot_msgs/ActorState`` that matter here, as
    plain values: the ROS adapter converts, so the session stays testable.
    ``known`` is false when the detections are missing or stale — unknown, not
    absent.
    """

    present: bool
    known: bool
    bbox: tuple[float, float, float, float] | None
    stamp: float
    others: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class RoundInputs:
    """Everything one observation round is decided from."""

    window: Window
    actor: ActorSignal
    now: float


@dataclass(frozen=True)
class Delivery:
    """What a :class:`SpeechPort` reports back about one reply.

    ``accepted`` false means nothing could have been heard, so the topic goes
    back to unspoken. ``estimated_end`` is when the audio is expected to have
    finished on the host's clock — an estimate, never a hardware confirmation:
    the playback service acknowledges that PCM was published, which is not the
    same as a speaker having finished.
    """

    accepted: bool
    estimated_end: float
    partial: bool = False
    detail: str = ""


class VisionLanguagePort:
    """Looks at a window; answers what is worth saying, as model output.

    The return value is whatever :func:`~visual_interaction.session.semantics
    .parse_observation` accepts — a JSON string or a mapping — so a real model
    and a scripted one are interchangeable. ``None`` means the model could not
    be reached, which the session reports as MODEL_NOT_READY rather than as
    silence.
    """

    def observe(self, inputs: RoundInputs) -> str | dict[str, object] | None:
        """Default: nothing to say, so the skeleton runs silently without a model."""
        return {"kind": "none"}


class SpeechPort:
    """Delivers one reply.

    The default accepts it and estimates an immediate end, which is what
    publishing the utterance amounts to. A real implementation synthesises and
    plays, and reports an estimate derived from the audio duration.
    """

    def deliver(self, text: str, *, event_id: str, now: float) -> Delivery:
        return Delivery(accepted=True, estimated_end=now, detail="published")


class ChangeGate:
    """Decides whether a window deserves a model call.

    The default opens for every window. A cheap pixel or box difference inside
    the person box would save the fixed prefill cost of a quiet round, which is
    the only saving available — an early stop only skips the remaining decode.
    Whether the saving is worth the missed answers is a phase-zero measurement,
    so the skeleton does not guess.
    """

    def should_observe(self, inputs: RoundInputs) -> bool:
        return True


class SessionHooks:
    """Everything the session delegates. Each default is the conservative value."""

    def __init__(
        self,
        *,
        vision: VisionLanguagePort | None = None,
        speech: SpeechPort | None = None,
        gate: ChangeGate | None = None,
    ) -> None:
        self.vision = vision or VisionLanguagePort()
        self.speech = speech or SpeechPort()
        self.gate = gate or ChangeGate()

    def on_round(self, inputs: RoundInputs, outcome: str) -> None:
        """Called once per round with the reason it ended. Default: nothing.

        ``outcome`` is a suppression reason, or an empty string when something
        was said. A raise here is recorded, not acted on: the round is over.
        """

    def on_session_end(self, session_id: str, reason: str) -> None:
        """Called when a session ends, before the next one starts. Default: nothing."""


def frames_overlap(first: Window, second: Window) -> bool:
    """Whether two windows rest on the same newest frame.

    Confirmation needs two *different* looks, and the cheapest way to fake one
    is to re-read the same buffer.
    """
    return second.newest_stamp <= first.newest_stamp


def as_boxes(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float, float], ...]:
    """Normalise a sequence of boxes into the fixed-width tuples the session uses."""
    boxes = []
    for value in values:
        if len(value) != 4:
            raise ValueError("a box needs four values")
        boxes.append((float(value[0]), float(value[1]), float(value[2]), float(value[3])))
    return tuple(boxes)


__all__ = [
    "ActorSignal",
    "ChangeGate",
    "Delivery",
    "RoundInputs",
    "SessionHooks",
    "SpeechPort",
    "VisionLanguagePort",
    "Window",
    "as_boxes",
    "frames_overlap",
]
