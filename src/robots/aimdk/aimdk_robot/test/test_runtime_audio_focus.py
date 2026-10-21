"""Unit tests for how long the runtime holds the platform's audio focus.

Releasing focus makes ``hal_audio`` abandon and flush whatever it still holds,
so the bridge must keep focus until the audio it forwarded has played out, and
must not take the speaker when it has nothing to play. The tests drive the
node's own focus methods with fake vendor clients and a virtual clock, so they
run without the AimDK overlay.
"""

from __future__ import annotations

import sys
import threading
import types
from collections import deque
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from aimdk_robot import runtime_node  # noqa: E402
from aimdk_robot.runtime_node import AimdkRuntimeNode  # noqa: E402

#: 16 kHz mono S16LE, as both X2 profiles declare their playback.
_BYTES_PER_MS = 32


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(runtime_node, "time", types.SimpleNamespace(monotonic=fake.monotonic))
    monkeypatch.setattr(
        runtime_node,
        "vg",
        types.SimpleNamespace(
            audio_focus_request=lambda _node, **fields: fields,
            audio_focus_release=lambda _node, **fields: fields,
            focus_granted=lambda response: response,
        ),
    )
    return fake


def bridge(clock, *, granted=True, release_idle_s=1.0):
    """A runtime wired just enough to forward playback and hold focus for it."""
    node = AimdkRuntimeNode.__new__(AimdkRuntimeNode)
    node._audio_lock = threading.Lock()
    node._playback_buffer = deque(maxlen=50)
    node._focus_held = False
    node._focus_attempt_s = -10.0
    node._render_until = 0.0
    node._focus_priority, node._focus_weight = 6, 0
    node._focus_release_idle_s = release_idle_s
    node._source = "ibrobot.aimdk_robot"
    node._vendor = {"audio": {"playback_sample_rate": 16000, "playback_channels": 1, "playback_sample_format": "S16LE"}}
    node.requests, node.releases, node.published = [], [], []
    node._request_focus = types.SimpleNamespace(
        call=lambda _request: node.requests.append(clock.now) or types.SimpleNamespace(ok=True, response=granted)
    )
    node._abandon_focus = types.SimpleNamespace(call=lambda _request: node.releases.append(clock.now))
    node._playback_type = lambda: types.SimpleNamespace(info=types.SimpleNamespace(), data=types.SimpleNamespace())
    node._playback_pub = types.SimpleNamespace(publish=node.published.append)
    node.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: None))
    node.get_logger = lambda: types.SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    return node


def speech(ms):
    return types.SimpleNamespace(audio=types.SimpleNamespace(data=[0] * (ms * _BYTES_PER_MS)))


def test_focus_outlasts_the_audio_it_forwarded_not_the_stream_that_carried_it(clock):
    node = bridge(clock)
    # A publisher writing 1 s ahead hands over 3 s of speech before the first second has played.
    for _ in range(15):
        node._on_audio_play(speech(200))
    node._audio_focus_tick()
    assert node.requests == [0.0] and len(node.published) == 15

    for now in (2.5, 3.5):  # still playing, then inside the grace after the last sample
        clock.now = now
        node._audio_focus_tick()
    assert node.releases == []

    clock.now = 4.01
    node._audio_focus_tick()
    assert node.releases == [4.01]


def test_a_refused_speaker_is_not_asked_for_again_with_nothing_to_play(clock):
    node = bridge(clock, granted=False, release_idle_s=3.0)
    node._on_audio_play(speech(200))
    node._audio_focus_tick()
    assert node.requests == [0.0] and not node._playback_buffer

    # Past the retry interval, still inside the idle window: there is nothing to play.
    for now in (1.1, 2.2):
        clock.now = now
        node._audio_focus_tick()
    assert node.requests == [0.0]

    node._on_audio_play(speech(200))
    clock.now = 2.3
    node._audio_focus_tick()
    assert node.requests == [0.0, 2.3]
