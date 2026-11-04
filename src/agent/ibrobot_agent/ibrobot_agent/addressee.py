"""Decide who a voice utterance is addressed to: the voice bridge's addressee gate.

A platform's wake word opens a window, and inside it every utterance in the
room reaches the bridge whether or not it was meant for the robot. The gate
answers two questions per transcript: did this utterance prove on its own that
it is for the robot (it said the wake phrase, or under ``vendor_session`` it is
the utterance the source's wake event names), and if not, is it the same person
who did -- by the optional anchor checks, of which the voiceprint is the only
one so far. The transcript, the voiceprint and the wake event of an utterance
are paired by the speech segment each of them names, never by arrival time.

ROS-free and owned by ``ibrobot_agent.voice_bridge.VoiceBridgeCore``, which
keeps the protocol around it (stop keywords, confirmations, requests); see the
"寻址门控" section of this package's README for the measured behaviour.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from embodied_common.text_normalization import normalize_trigger_text

#: How the gate decides an utterance proves on its own that it is for the robot.
#:
#: ``wake_phrase``: the transcript must start with one of ``wake_phrases``. Needs
#: an audio source whose text still contains the wake word.
#:
#: ``vendor_session``: the source consumes its *own* wake word, so it never
#: reaches the text ("灵犀灵犀，挥个手" arrives as "挥个手" on the X2), and reports
#: it as a wake event naming the segment it opened. That utterance opens the
#: session and sets the anchors; the session lasts while the room keeps talking
#: (``session_gap_sec``) and nothing opens one but a wake event. Needs a source
#: that segments and reports its wake word (``SpeechTranscript`` and
#: ``SpeechWakeEvent``).
ADDRESSEE_TRIGGERS = ("wake_phrase", "vendor_session")


class AnchorCheck(Protocol):
    """One optional non-text signal that an utterance inside the window must still match.

    The utterance that says the wake phrase proves on its own that it is meant
    for the robot, so it does not get checked: it *sets* the anchor. Every later
    utterance the window lets through is then compared against that anchor, so
    the window stops being "whatever was said next" and becomes "whatever the
    same person said next".

    Checks are independent and opt-in; a deployment enables the ones its
    hardware can actually measure. The voiceprint is the only one so far: a
    direction check needs an azimuth source, and the X2 has none. A
    measurement that is missing accepts: a sensor that is down must not make
    the robot deaf, and ignoring its own user is a worse failure than
    answering someone else.
    """

    #: Measurement name. Also the key the node feeds to ``on_measurement``.
    name: str
    #: A measurement whose transcript has not come within this long is dropped
    #: (the ASR discarded that segment).
    max_age_sec: float

    def verdict(self, anchor: Any, measurement: Any) -> tuple[bool, str]:
        """True when ``measurement`` looks like the same source as ``anchor``, plus a log detail."""

    # A check may also define ``merge(anchor, measurement) -> anchor``. When every
    # check accepts an utterance, its measurement is folded into the anchor
    # through that hook, so the anchor stops depending on one utterance alone.
    # A check without it keeps the anchor it was given.


def cosine_similarity(left: Any, right: Any) -> float | None:
    """Cosine of two embeddings, or ``None`` when they are not comparable.

    Not comparable means an empty vector or two different lengths. That is not
    evidence about who spoke, so the caller treats it as a missing measurement
    rather than as a mismatch. A model change that keeps the length is caught
    by ``Voiceprint.model_id`` instead.
    """
    first = [float(item) for item in left]
    second = [float(item) for item in right]
    if not first or len(first) != len(second):
        return None
    norms = math.sqrt(sum(item * item for item in first)) * math.sqrt(sum(item * item for item in second))
    if not norms > 0.0:
        return None
    return sum(a * b for a, b in zip(first, second, strict=True)) / norms


@dataclass(frozen=True)
class Voiceprint:
    """One speaker vector and the vector space it lives in.

    ``model_id`` is what the producer publishes with every vector
    (``SpeakerEmbedding.model_id``). Two vectors are comparable exactly when it
    matches: a model swap that keeps the dimension changes every comparison
    while the numbers keep looking reasonable, so the length alone cannot tell.
    """

    vector: tuple[float, ...]
    model_id: str


@dataclass(frozen=True)
class SpeakerCheck:
    """Keep the window for the voice that said the wake phrase.

    The anchor is the voiceprint of the wake utterance and the measurement is
    the voiceprint of the utterance being judged, both ``Voiceprint``; they come
    from ``voice_asr_service``'s speaker_id node on its own topic. The cosine is
    computed here instead of trusting the producer to have normalized, and only
    between vectors of the same ``model_id``.

    ``min_similarity`` is deployment-specific: it depends on the model, the
    microphone, the room and how alike the people in it sound. It has to be
    calibrated against recordings of the actual users -- a value copied from a
    benchmark will either ignore the user or let the room through.
    """

    name: ClassVar[str] = "speaker"
    min_similarity: float
    max_age_sec: float
    #: How far above ``min_similarity`` an utterance must score to be merged.
    merge_margin: float = 0.0
    #: Reject an utterance the producer reported as too short for a
    #: voiceprint, instead of letting it through unchecked.
    reject_short: bool = False

    def verdict(self, anchor: Voiceprint, measurement: Voiceprint) -> tuple[bool, str]:
        similarity = _similarity(anchor, measurement)
        if similarity is None:
            return True, "not comparable"
        return similarity >= self.min_similarity, f"similarity {similarity:.2f} < {self.min_similarity:.2f}"

    def merge(self, anchor: Voiceprint, measurement: Voiceprint) -> Voiceprint:
        """Add an accepted utterance's voiceprint to the anchor, every utterance weighted equally.

        One utterance is a noisy estimate of a voice: the platform cuts the
        anchor sentence short at times, and a 3 s anchor scored the speaker's
        own follow-up at 0.59 on the X2 while another speaker reached 0.56.
        The anchor is kept as a sum of unit vectors; the cosine in ``verdict``
        normalizes it, so the sum stands for the mean direction.

        Only utterances every check accepted are merged, so a rejected voice
        never moves the anchor. One that slipped through moves it by 1/n.

        Accepted is not enough: an utterance just over the threshold is weak
        evidence that it is the same voice, and folding it in drags the anchor
        toward whoever sounds like it. Measured on the X2 (2026-10-25, two
        lecture recordings through the PA): a 0.66 follow-up merged into the
        anchor lifted another speaker from 0.59 to 0.66 and let him through.
        So only utterances ``merge_margin`` above the threshold are merged; the
        user's own follow-ups score 0.67-0.84 and still sharpen the anchor.
        """
        similarity = _similarity(anchor, measurement)
        if similarity is None or similarity < self.min_similarity + self.merge_margin:
            # Not comparable (another model, or nothing to compare) or too weak: nothing to learn.
            return anchor
        unit = _unit_vector(measurement.vector)
        return Voiceprint(tuple(float(a) + b for a, b in zip(anchor.vector, unit, strict=True)), anchor.model_id)


def _similarity(anchor: Voiceprint, measurement: Voiceprint) -> float | None:
    """Cosine of two voiceprints, or ``None`` when they come from different vector spaces."""
    if anchor.model_id != measurement.model_id:
        return None
    return cosine_similarity(anchor.vector, measurement.vector)


def _unit_vector(vector: Any) -> tuple[float, ...]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    return tuple(item / norm for item in values) if norm > 0.0 else ()


#: One speech segment: the ``source_id`` and the ``sequence`` of the BEGIN chunk
#: that opened it (``ibrobot_msgs/SpeechAudioChunk``). The transcript
#: (``SpeechTranscript``), the voiceprint (``SpeakerEmbedding``) and the wake
#: event (``SpeechWakeEvent``) of an utterance all name it, so the gate pairs
#: them by it instead of by arrival.
Segment = tuple[str, int]


@dataclass(frozen=True)
class _Measurement:
    """One signal's value for one segment, and when it arrived (for pruning)."""

    value: Any
    at: float


@dataclass(frozen=True)
class GateVerdict:
    """Whether an utterance is for the robot, with the wake phrase already stripped."""

    addressed: bool
    raw: str
    normalized: str
    #: Why it was rejected, for the log; empty when addressed.
    reason: str = ""


class AddresseeGate:
    """Decide whether a transcript is for the robot; not thread-safe.

    Owns everything the gate remembers between utterances: the window or
    session, the pending wake event, and per check the anchor and the
    measurements of segments not judged yet. The voice bridge keeps the
    protocol (stop keywords, confirmations, requests) and asks the gate twice
    per transcript: ``open_if_new_session`` before any branch can return, then
    ``judge`` for an utterance that reached addressing.

    Transcripts, measurements and wake events are paired by their speech
    segment. A measurement only ever judges the transcript of its own segment;
    one whose transcript never comes (the ASR dropped the segment) is pruned
    once it is older than its check's ``max_age_sec``, and a transcript whose
    measurement never comes is judged without it (fail-open). A transcript
    without a segment (a source that does not segment, such as the local VAD)
    has no measurements and can never be what a wake event opened.

    Times are the bridge's monotonic clock. A transcript held for its
    measurement is routed later than it was heard and passes ``heard_at``.
    """

    def __init__(
        self,
        *,
        trigger: str,
        wake_phrases: tuple[str, ...],
        wake_window_sec: float,
        session_gap_sec: float,
        checks: Iterable[AnchorCheck] = (),
    ) -> None:
        if trigger not in ADDRESSEE_TRIGGERS:
            raise ValueError(f"addressee_trigger must be one of {ADDRESSEE_TRIGGERS}")
        self.trigger = trigger
        self.wake_phrases = tuple(wake_phrases)
        self.wake_window_sec = float(wake_window_sec)
        self.session_gap_sec = float(session_gap_sec)
        self.checks: tuple[AnchorCheck, ...] = tuple(checks)
        self._max_age = {check.name: float(check.max_age_sec) for check in self.checks}
        #: ``wake_phrase``: monotonic deadline until which a follow-up needs no wake phrase.
        self.wake_until = 0.0
        #: ``vendor_session``: whether a wake event opened a session that is still alive.
        self.session_open = False
        #: ``vendor_session``: when the last transcript of the live session was heard.
        self.last_utterance_at = 0.0
        #: ``vendor_session``: the latest wake event's segment and arrival, until
        #: a transcript of it opens the session.
        self.pending_wake: tuple[Segment, float] | None = None
        #: ``vendor_session``: per source, the newest segment a transcript has
        #: already been routed for. A wake event for one of those came too late
        #: to open anything and must not pass to a later segment.
        self.transcribed: dict[str, int] = {}
        #: Check name -> the anchor captured from the utterance that proved itself.
        self.anchors: dict[str, Any] = {}
        #: Check name -> segment -> its measurement, until its transcript is judged.
        self.measurements: dict[str, dict[Segment, _Measurement]] = {}
        #: Check name -> the opener's segment, whose measurement becomes the
        #: anchor when it lands. The opener proves itself, so it is not held for it.
        self.awaiting_anchor: dict[str, Segment] = {}

    def required_measurements(self) -> tuple[str, ...]:
        """Measurement names the enabled checks need the node to feed."""
        return tuple(check.name for check in self.checks)

    def on_wake(self, now: float, segment: Segment | None) -> bool:
        """The audio source reports that its wake word opened ``segment``; False when it came too late.

        Under ``vendor_session`` the wake word never reaches the text, and this
        event is the only thing that can open a session: the transcript of that
        segment proves itself and sets the anchors. The source may first open a
        segment for the wake word's residue alone and drop it, so the first
        transcript of the same source from that segment on is the one the wake
        word preceded. One of an earlier segment -- a sentence that was still
        being transcribed when someone else woke the robot -- is not.

        The event and the transcript travel on separate topics, so either may
        land first. A transcript heard outside a session waits for its wake
        event (``needs_wait``). One already routed for this segment or a later
        one means the event came too late: it is refused rather than left
        pending, where it would hand the session to whoever spoke next.
        """
        if segment is None:
            return False
        if segment[1] <= self.transcribed.get(segment[0], -1):
            return False
        self.pending_wake = (segment, now)
        return True

    def on_measurement(self, name: str, value: Any, now: float, segment: Segment | None) -> None:
        """Record one non-text observation of ``segment``.

        ``None`` is a measurement too: the producer saw the segment end and had
        nothing usable for it (too short for a voiceprint), so its transcript
        need not wait. A measurement without a segment cannot be paired with any
        transcript and is ignored.
        """
        if segment is None:
            return
        name = str(name)
        if self.awaiting_anchor.get(name) == segment:
            del self.awaiting_anchor[name]
            if value is None:
                self.anchors.pop(name, None)
            else:
                self.anchors[name] = value
            return
        store = self.measurements.setdefault(name, {})
        store[segment] = _Measurement(value, now)
        max_age = self._max_age.get(name, 0.0)
        for stale in [key for key, item in store.items() if now - item.at > max_age]:
            del store[stale]

    def needs_wait(self, normalized: str, heard_at: float, segment: Segment | None) -> bool:
        """True when judging ``normalized`` now would miss something still on its way.

        Under ``vendor_session`` a transcript heard with no session alive waits
        for the wake event of its own segment, which travels on another topic
        and may land after it; judged now it would be rejected for want of one.
        Otherwise only an utterance that is compared against an anchor waits,
        for its own segment's measurement, and so does one that would be
        compared against an opener's anchor that has not landed yet. A session
        opener is never compared, and with no anchor there is nothing to compare
        against, so neither waits. The bridge answers for the utterances that
        never reach the gate (a stop, a confirmation answer). Measured on the
        X2, the voiceprint usually lands before its transcript and the wake event
        before both, so this rarely waits at all; ``measurement_grace_sec``
        bounds the wait.
        """
        if segment is None:
            return False
        if self.trigger == "vendor_session":
            if self._wakes(segment, heard_at):
                return False
            if not self._session_alive(heard_at):
                return not self._wake_cannot_come(segment)
        elif not self.wake_phrases or any(normalized.startswith(phrase) for phrase in self.wake_phrases):
            return False
        elif heard_at >= self.wake_until:
            # Outside the window it is rejected for want of a wake phrase anyway.
            return False
        for check in self.checks:
            if check.name in self.awaiting_anchor:
                return True
            if check.name in self.anchors and segment not in self.measurements.get(check.name, {}):
                return True
        return False

    def open_if_new_session(self, now: float, *, heard_at: float, segment: Segment | None) -> bool:
        """Note a transcript's arrival; True when it opens a ``vendor_session`` session.

        Called for *every* transcript, before the bridge can route it anywhere,
        including the ones the gate goes on to reject. Only a wake event opens
        a session: the transcript of the segment it names proves itself and
        re-anchors here, before the stop and confirmation branches can return,
        so the previous session's anchors never judge what follows it. A
        session stays alive while transcripts keep coming less than
        ``session_gap_sec`` apart -- any speech in the room, rejected or not,
        because the source's own session is kept alive the same way -- and
        once it lapses nothing is accepted until the next wake event. The first
        utterance after a silence is no evidence that it was meant for the
        robot: on the X2 the platform also releases speech nobody woke it for.
        """
        if self.trigger != "vendor_session":
            return False
        if segment is not None:
            self.transcribed[segment[0]] = max(segment[1], self.transcribed.get(segment[0], -1))
        if segment is not None and self._wakes(segment, heard_at):
            self.pending_wake = None
            self.session_open = True
            self.last_utterance_at = heard_at
            self._capture_anchors(now, segment)
            return True
        if self._session_alive(heard_at):
            self.last_utterance_at = heard_at
        else:
            self.session_open = False
        return False

    def judge(
        self, raw: str, normalized: str, now: float, *, session_opened: bool, heard_at: float, segment: Segment | None
    ) -> GateVerdict:
        """Decide whether this utterance is for the robot, by the configured trigger."""
        if self.trigger == "vendor_session":
            return self._session_gate(raw, normalized, session_opened=session_opened, segment=segment)
        return self._wake_phrase_gate(raw, normalized, now, heard_at=heard_at, segment=segment)

    def note_handled(self, heard_at: float) -> None:
        """An addressed utterance, heard at ``heard_at``, is being acted on: keep the ``wake_phrase`` window open.

        Extended only by an utterance that is actually being handled, so a
        window cannot be held open by speech the robot ignored. The
        vendor_session trigger has no such timer: its window is the session,
        tracked by ``open_if_new_session``.
        """
        if self.wake_phrases:
            self.wake_until = heard_at + self.wake_window_sec

    def _wakes(self, segment: Segment, heard_at: float) -> bool:
        """Whether the pending wake event opened ``segment``'s utterance.

        Same source, from the woken segment on, and while the wake is recent:
        a wake word whose own utterance never produced a transcript must not
        hand the session to whoever speaks long after it (``wake_window_sec``).
        """
        if self.pending_wake is None:
            return False
        (source_id, sequence), woken_at = self.pending_wake
        return segment[0] == source_id and segment[1] >= sequence and heard_at - woken_at <= self.wake_window_sec

    def _wake_cannot_come(self, segment: Segment) -> bool:
        """Whether a wake event for ``segment`` would be refused or already lost to a later one."""
        source_id, sequence = segment
        if self.transcribed.get(source_id, -1) > sequence:
            return True
        if self.pending_wake is None:
            return False
        (wake_source, wake_sequence), _woken_at = self.pending_wake
        return wake_source == source_id and wake_sequence > sequence

    def _session_alive(self, heard_at: float) -> bool:
        return self.session_open and heard_at - self.last_utterance_at <= self.session_gap_sec

    def _capture_anchors(self, now: float, segment: Segment | None) -> None:
        """Anchor every check on the utterance that just proved itself.

        The opener is routed at once; a measurement of its segment that has not
        arrived yet becomes the anchor when it lands (``on_measurement``). An
        opener that had none (too short for a voiceprint) leaves the check
        without an anchor, so the window accepts every voice until the next
        wake. Anchoring on some later utterance instead could hand the session
        to a bystander who happened to speak first, and the robot would then
        ignore its own user -- the worse of the two failures.
        """
        for check in self.checks:
            self.anchors.pop(check.name, None)
            self.awaiting_anchor.pop(check.name, None)
            measurement = self._take(check, segment)
            if measurement is not None:
                if measurement.value is not None:
                    self.anchors[check.name] = measurement.value
            elif segment is not None:
                self.awaiting_anchor[check.name] = segment

    def _take(self, check: AnchorCheck, segment: Segment | None) -> _Measurement | None:
        """Consume this check's measurement of ``segment``, if it arrived."""
        if segment is None:
            return None
        return self.measurements.get(check.name, {}).pop(segment, None)

    def _rejecting_check(self, segment: Segment | None) -> str:
        """Describe the first check that says this utterance is someone else's; empty if none does.

        When none does, the utterance is the anchored source by every enabled
        check, and its measurements are folded into the anchors (see
        ``AnchorCheck``). A rejected utterance leaves every anchor untouched.
        """
        rejections = []
        judged: list[tuple[AnchorCheck, Any]] = []
        for check in self.checks:
            anchor = self.anchors.get(check.name)
            measurement = self._take(check, segment)
            if anchor is None or measurement is None:
                continue
            if measurement.value is None:
                # The producer saw this utterance and had nothing to compare.
                if getattr(check, "reject_short", False):
                    rejections.append(f"{check.name}: too short to compare")
                continue
            agrees, detail = check.verdict(anchor, measurement.value)
            if not agrees:
                rejections.append(f"{check.name}: {detail}")
            judged.append((check, measurement.value))
        if rejections:
            return rejections[0]
        for check, value in judged:
            merge = getattr(check, "merge", None)
            if merge is not None:
                self.anchors[check.name] = merge(self.anchors[check.name], value)
        return ""

    def _session_gate(self, raw: str, normalized: str, *, session_opened: bool, segment: Segment | None) -> GateVerdict:
        """The source only released this after its own wake word; the question is whose turn it is.

        The transcript a wake event names proves itself and set the anchors in
        ``open_if_new_session``; everything later in the same session is
        checked against them. With no live session nothing is accepted: the
        session is the window, and only a wake event opens one.
        """
        if session_opened:
            return GateVerdict(True, raw, normalized)
        if not self.session_open:
            return GateVerdict(False, raw, normalized, reason="no wake event opened a session")
        rejected = self._rejecting_check(segment)
        return GateVerdict(not rejected, raw, normalized, reason=rejected)

    def _wake_phrase_gate(
        self, raw: str, normalized: str, now: float, *, heard_at: float, segment: Segment | None
    ) -> GateVerdict:
        """Decide whether this utterance is for the robot, and strip the wake phrase.

        The platform's own wake word only arms its VAD once (``only_voice`` mode
        keeps it armed afterwards), so every later utterance in the room reaches
        us whether or not it was addressed to the robot. A wake phrase at the
        start of the utterance is the one deterministic signal available here;
        it opens a short window so a follow-up does not have to repeat it, and
        it anchors the optional checks that then decide who the window is for.

        Matching is on the normalized text, but the request carries ``raw``, so
        the phrase is removed from both: the shortest prefix of ``raw`` whose
        normalization is the phrase. ``normalize_trigger_text`` only drops and
        case-folds characters, so such a prefix always exists.
        """
        if not self.wake_phrases:
            return GateVerdict(True, raw, normalized)
        for phrase in self.wake_phrases:
            if not normalized.startswith(phrase):
                continue
            # The ASR drops the comma, so the wake phrase arrives glued to the
            # command ("灵犀灵犀挥个手"); the planner must not see it.
            cut = next((at for at in range(1, len(raw) + 1) if normalize_trigger_text(raw[:at]) == phrase), 0)
            spare = " ，,。.、!?！？"
            # This utterance named the robot, so it is not checked against
            # anything: it is what the checks are anchored on.
            self._capture_anchors(now, segment)
            # Naming the robot is itself a request for attention, so the window
            # opens even when nothing follows the phrase ("灵犀灵犀" then "挥个手").
            self.wake_until = heard_at + self.wake_window_sec
            return GateVerdict(True, raw[cut:].lstrip(spare), normalized[len(phrase) :].lstrip(spare))
        if heard_at < self.wake_until:
            rejected = self._rejecting_check(segment)
            return GateVerdict(not rejected, raw, normalized, reason=rejected)
        return GateVerdict(False, raw, normalized, reason="no wake phrase")
