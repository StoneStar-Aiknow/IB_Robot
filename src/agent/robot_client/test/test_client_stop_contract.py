"""The client stop contract must be the shared vocabulary, not a copy.

An import-time ``except ImportError`` made the source depend on the import
environment. This suite pins the single source and the whole-utterance
behaviour in both languages.
"""

from __future__ import annotations

import pytest

from embodied_common.control_phrases import STOP_PHRASES as CANONICAL_STOP_PHRASES
from robot_client.stop_contract import STOP_PHRASES, is_stop


def test_the_client_reexports_the_canonical_tuple_itself():
    assert STOP_PHRASES is CANONICAL_STOP_PHRASES


@pytest.mark.parametrize("text", ["别动了", "别动", "停止", "停下", "停", "stop", "STOP", "  halt  "])
def test_known_stop_phrases_are_recognized(text):
    assert is_stop(text)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "别动一下", "停下来看看", "stop now", "nonstop", "停止服务", "继续", "go on", None],
)
def test_substrings_and_non_stop_utterances_are_not_stop(text):
    assert not is_stop(text)
