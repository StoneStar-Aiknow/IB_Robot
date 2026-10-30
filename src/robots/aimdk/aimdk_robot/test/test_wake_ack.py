"""The timing rule that turns the platform's wake acknowledgement into a wake event."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from aimdk_robot.wake_ack import WakeAckPairer  # noqa: E402

#: The X2's acknowledgement token form, as the shipped profiles configure it.
ACK_PATTERN = "^_[0-9]+_"


def _ack(n: int) -> str:
    """An acknowledgement token in the form the X2 uses: ``_<epoch ms>_<n><random>``."""
    return f"_179283543039{n}_{n}ack"


def _pairer():
    return WakeAckPairer(frozenset({"tts"}), token_pattern=ACK_PATTERN, window_s=1.5)


def _segment(at: float) -> int:
    """The sequence of the BEGIN chunk the test pretends arrived at ``at``."""
    return int(at * 10)


def _begin(pairer, at: float):
    return pairer.on_speech_begin(at, _segment(at))


@pytest.mark.parametrize(
    ("ack_offset", "label"),
    [
        pytest.param(-0.5, "ack before begin", id="ack 0.5 s before begin (measured)"),
        pytest.param(1.0, "ack after begin", id="ack 1.0 s after begin (measured)"),
    ],
)
def test_an_acknowledgement_near_a_speech_begin_is_a_wake_of_that_segment(ack_offset, label):
    """Both orders were measured on the X2; the event fires on whichever half arrives
    second, and either way it names the segment whose BEGIN it paired with."""
    pairer = _pairer()
    if ack_offset < 0:
        assert pairer.on_playback("tts", _ack(1), 10.0 + ack_offset) is None
        assert _begin(pairer, 10.0) == _segment(10.0)
    else:
        assert _begin(pairer, 10.0) is None
        assert pairer.on_playback("tts", _ack(1), 10.0 + ack_offset) == _segment(10.0)


def test_speech_without_an_acknowledgement_is_not_a_wake():
    pairer = _pairer()

    assert _begin(pairer, 10.0) is None
    assert _begin(pairer, 20.0) is None


def test_a_later_spoken_answer_is_not_a_wake():
    """The assistant's own answer is a new token too, but long after the utterance began."""
    pairer = _pairer()
    _begin(pairer, 10.0)

    assert pairer.on_playback("tts", _ack(3), 14.0) is None


def test_every_chunk_of_one_reply_counts_once():
    pairer = _pairer()
    assert pairer.on_playback("tts", _ack(1), 9.5) is None
    assert _begin(pairer, 10.0) == _segment(10.0)

    # More chunks of the same reply, then the next utterance's BEGIN: no second wake.
    assert pairer.on_playback("tts", _ack(1), 10.2) is None
    assert _begin(pairer, 10.5) is None


def test_one_pair_makes_one_wake():
    """A second BEGIN inside the window (the platform re-segmented) must not fire again."""
    pairer = _pairer()
    _begin(pairer, 10.0)
    assert pairer.on_playback("tts", _ack(1), 10.7) == _segment(10.0)

    assert _begin(pairer, 11.2) is None


def test_a_late_acknowledgement_names_the_latest_begin():
    """Two BEGINs before the acknowledgement: the wake belongs to the one it pairs with."""
    pairer = _pairer()
    _begin(pairer, 8.0)
    _begin(pairer, 10.0)

    assert pairer.on_playback("tts", _ack(1), 10.4) == _segment(10.0)


def test_playback_from_another_source_is_ignored():
    """The runtime's own playback goes out on the same topic under its own pkg_name."""
    pairer = _pairer()
    _begin(pairer, 10.0)

    assert pairer.on_playback("ibrobot.aimdk_robot", _ack(1), 10.2) is None


def test_each_wake_needs_its_own_token():
    pairer = _pairer()
    _begin(pairer, 10.0)
    assert pairer.on_playback("tts", _ack(1), 10.2) == _segment(10.0)

    _begin(pairer, 30.0)
    assert pairer.on_playback("tts", _ack(2), 30.4) == _segment(30.0)


@pytest.mark.parametrize(
    ("pkg_names", "pattern", "window", "match"),
    [
        pytest.param(frozenset(), ACK_PATTERN, 1.5, "pkg_name", id="no pkg names"),
        pytest.param(frozenset({"tts"}), ACK_PATTERN, 0.0, "window", id="zero window"),
        pytest.param(frozenset({"tts"}), "", 1.5, "token_pattern", id="no token pattern"),
        pytest.param(frozenset({"tts"}), "_(", 1.5, "regular expression", id="invalid token pattern"),
    ],
)
def test_the_pairer_configuration_is_validated(pkg_names, pattern, window, match):
    with pytest.raises(ValueError, match=match):
        WakeAckPairer(pkg_names, token_pattern=pattern, window_s=window)


def test_an_unprompted_dialog_answer_is_not_a_wake_even_at_a_begin():
    """Seen on the X2: the dialog subsystem spoke on its own (token hds_tts_...)
    just as a bystander started talking, and that paired into a false wake."""
    pairer = _pairer()
    _begin(pairer, 10.0)

    assert pairer.on_playback("tts", "hds_tts_36IUknWjEw0NVKL7L4RGgK3", 10.3) is None


def test_a_rejected_token_does_not_spoil_a_later_acknowledgement():
    pairer = _pairer()
    pairer.on_playback("tts", "hds_tts_answer", 9.6)

    assert pairer.on_playback("tts", "_1792835430393_ack", 9.8) is None
    assert _begin(pairer, 10.0) == _segment(10.0)
