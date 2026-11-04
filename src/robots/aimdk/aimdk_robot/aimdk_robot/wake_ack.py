"""Turn the platform's own wake-word acknowledgement into a project wake event.

The X2 consumes its wake word before it releases speech, and it exposes no wake
event of its own. What it does do -- on every wake word and only then -- is have
its native assistant start a reply on the playback topic: a new ``token_id``
under ``pkg_name`` ``tts``. Measured on the robot (2026-10-24): 14 of 14
utterances classified correctly, 7 with the wake word and 7 without (lecture
audio, a sentence that merely mentions the robot's name, plain follow-ups), and
every acknowledgement fell between 1.2 s before and 1.0 s after the BEGIN of the
vendor speech segment it belonged to.

An acknowledgement is only reported when it pairs with a segment BEGIN and its
token has the acknowledgement's form (``token_pattern``), and it is reported as
that segment: the sequence of the BEGIN chunk it paired with. The platform also
speaks unprompted answers from its dialog subsystem; one of those, landing
within the pairing window of the next utterance's BEGIN, made a bystander look
like a new wake (2026-10-24). Every acknowledgement seen so far (9 of 9) had a
token of the form ``_<epoch ms>_<n><random>``; the unprompted answer's token was
``hds_tts_<n><random>``. The form is undocumented vendor detail, so it is
configuration, and required: a pairer that accepted any token would bring that
false wake back. If the form ever changes, the symptom is that no wake events
arrive and the consumer falls back to its silence-gap rule, never that a
bystander is let in. The speaker on the robot does not have to work for
any of this: the platform publishes the playback message whether or not
anything comes out.

Kept free of ROS so the timing rule can be tested on its own.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

#: Tokens remembered so a re-sent playback chunk of the same reply is not a new one.
_REMEMBERED_TOKENS = 64


@dataclass
class WakeAckPairer:
    """Pair a fresh acknowledgement token with a vendor speech BEGIN."""

    pkg_names: frozenset[str]
    #: Regular expression an acknowledgement's token must match. Required: the
    #: platform's other voices share the pkg_name and differ only in the token.
    token_pattern: str
    #: Largest gap, either way, between the acknowledgement and the BEGIN.
    window_s: float = 1.5
    _seen: deque[str] = field(default_factory=lambda: deque(maxlen=_REMEMBERED_TOKENS))
    _ack_at: float | None = None
    _begin_at: float | None = None
    _begin_segment: int | None = None

    def __post_init__(self) -> None:
        if not self.pkg_names:
            raise ValueError("wake acknowledgement needs at least one playback pkg_name")
        if not self.window_s > 0:
            raise ValueError("wake acknowledgement pairing window must be positive")
        if not self.token_pattern:
            raise ValueError("wake acknowledgement needs a token_pattern; any token would pair dialog answers too")
        try:
            self._token_re = re.compile(self.token_pattern)
        except re.error as error:
            raise ValueError(f"wake acknowledgement token_pattern is not a regular expression: {error}") from error

    def on_playback(self, pkg_name: str, token_id: str, now: float) -> int | None:
        """Note one playback chunk; the woken segment when it completes a wake, else None."""
        if pkg_name not in self.pkg_names or not token_id or token_id in self._seen:
            # Every chunk of a reply carries the same token; only its first one counts.
            return None
        self._seen.append(token_id)
        if not self._token_re.match(token_id):
            # Another of the platform's voices (its dialog answers), not the
            # acknowledgement; it must not be paired with anyone's BEGIN.
            return None
        self._ack_at = now
        return self._pair()

    def on_speech_begin(self, now: float, segment: int) -> int | None:
        """Note a vendor speech BEGIN, ``segment`` being its chunk sequence; as ``on_playback``."""
        self._begin_at = now
        self._begin_segment = segment
        return self._pair()

    def _pair(self) -> int | None:
        if self._ack_at is None or self._begin_at is None:
            return None
        if abs(self._ack_at - self._begin_at) > self.window_s:
            return None
        segment = self._begin_segment
        # One wake per pair: neither half may complete a second one.
        self._ack_at = self._begin_at = self._begin_segment = None
        return segment
