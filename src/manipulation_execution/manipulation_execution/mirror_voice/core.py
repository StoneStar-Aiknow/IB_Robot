"""Session state messages to spoken events, the phrase queue and the TTS service wait. No ROS.

The voice node only listens to ``/mirror_session/state``. Whatever started or
stopped the session (CLI, cover gestures, the voice agent), it ends in the
same transition, so the transition alone decides what is said:

=============  =========================================================
event          transition (``last_transition`` of a state message)
=============  =========================================================
``starting``   ``WAIT/none -> STANDBY/approach``
``ready``      ``-> STANDBY/ready``, from the approach or a cooldown
``mirroring``  ``-> MIRRORING/running``, every round
``goodbye``    ``-> WAIT/none`` from anywhere else, whatever the reason
=============  =========================================================

Everything else (``-> STANDBY/cooldown``, ``-> */wrap_up``) is silent. The
state strings are built from the session's own ``SessionState`` and ``Phase``,
so renaming a state breaks here at import instead of going quiet.

The WAV cache, synthesis and playback order are in ``mirror_voice.speaker``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence

from embodied_common.mirror_voice_contract import SERVICE_WAIT_ERROR_SEC, SERVICE_WAIT_REPEAT_SEC
from manipulation_execution.mirror_session.core import Phase, SessionState

WAIT_NONE = f"{SessionState.WAIT.value}/{Phase.NONE.value}"
STANDBY_APPROACH = f"{SessionState.STANDBY.value}/{Phase.APPROACH.value}"
STANDBY_READY = f"{SessionState.STANDBY.value}/{Phase.READY.value}"
MIRRORING_RUNNING = f"{SessionState.MIRRORING.value}/{Phase.RUNNING.value}"

GOODBYE = "goodbye"
TRANSITION_KEYS = ("from", "to", "t")


def event_for(transition: Mapping) -> str | None:
    """The event a transition is spoken as, or ``None`` for a silent one."""
    source, target = transition["from"], transition["to"]
    if source == WAIT_NONE and target == STANDBY_APPROACH:
        return "starting"
    if target == STANDBY_READY:
        return "ready"
    if target == MIRRORING_RUNNING:
        return "mirroring"
    if target == WAIT_NONE and source != WAIT_NONE:
        return GOODBYE
    return None


class TransitionFilter:
    """State messages to events, each transition spoken at most once.

    The first message only records its transition: on a transient_local
    subscription it is the replay of a transition that happened before this
    node listened. After that, a transition is spoken when its
    ``(from, to, t)`` key is new; heartbeats repeat the last transition and
    are dropped by the key. The last ``memory`` keys are kept.

    A restarted session starts again from ``WAIT/none`` with no transition,
    and its new transitions have new keys, so each is spoken once.
    """

    def __init__(self, memory: int = 64):
        if isinstance(memory, bool) or not isinstance(memory, int) or memory < 1:
            raise ValueError("memory must be a positive integer")
        self._memory = memory
        self._keys: deque[tuple] = deque()
        self._seen: set[tuple] = set()
        self._first = True
        self.received = 0
        self.events = 0
        self.malformed = 0

    def feed(self, message: object) -> str | None:
        """The event this message is spoken as; ``None`` when nothing is said."""
        self.received += 1
        first, self._first = self._first, False
        if not isinstance(message, Mapping) or "last_transition" not in message:
            self.malformed += 1
            return None
        transition = message["last_transition"]
        if transition is None:
            return None
        key = _key(transition)
        if key is None:
            self.malformed += 1
            return None
        if key in self._seen:
            return None
        self._remember(key)
        if first:
            return None
        event = event_for(transition)
        if event is not None:
            self.events += 1
        return event

    def _remember(self, key: tuple) -> None:
        self._keys.append(key)
        self._seen.add(key)
        if len(self._keys) > self._memory:
            self._seen.discard(self._keys.popleft())


def _key(transition: object) -> tuple | None:
    if not isinstance(transition, Mapping) or not all(name in transition for name in TRANSITION_KEYS):
        return None
    source, target, t = (transition[name] for name in TRANSITION_KEYS)
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    if isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t):
        return None
    return source, target, t


class PhraseQueue:
    """Phrases waiting to be spoken, oldest first, at most ``maxlen``.

    A push into a full queue drops the oldest phrase that is not a goodbye;
    with only goodbyes queued, the oldest goodbye.
    """

    def __init__(self, maxlen: int):
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen < 1:
            raise ValueError("maxlen must be a positive integer")
        self._maxlen = maxlen
        self._items: deque[tuple[str, str]] = deque()
        self.dropped = 0

    def push(self, event: str, text: str) -> tuple[str, str] | None:
        """Queue a phrase; returns the ``(event, text)`` dropped to make room, if any."""
        dropped = None
        if len(self._items) >= self._maxlen:
            index = next((i for i, item in enumerate(self._items) if item[0] != GOODBYE), 0)
            dropped = self._items[index]
            del self._items[index]
            self.dropped += 1
        self._items.append((event, text))
        return dropped

    def pop(self) -> tuple[str, str] | None:
        return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        return len(self._items)


def pick_phrase(phrases: Mapping[str, Sequence[str]], event: str) -> str:
    """The phrase spoken for an event: its first one.

    Varied phrasing (say, a random choice) only needs this function replaced.
    """
    return phrases[event][0]


class ServiceWatch:
    """When to report a missing TTS service.

    From the first poll that finds it missing: the first report after
    ``error_after`` seconds, then one every ``repeat`` seconds. Finding the
    service clears it. The caller passes the time; at most one report per poll.
    """

    def __init__(self, error_after: float = SERVICE_WAIT_ERROR_SEC, repeat: float = SERVICE_WAIT_REPEAT_SEC):
        for name, value in (("error_after", error_after), ("repeat", repeat)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than 0")
        self._error_after = float(error_after)
        self._repeat = float(repeat)
        self._due: float | None = None

    def poll(self, now: float, available: bool) -> bool:
        """True when an ERROR is due now."""
        if available:
            self._due = None
            return False
        if self._due is None:
            self._due = now + self._error_after
        if now < self._due:
            return False
        while self._due <= now:
            self._due += self._repeat
        return True
