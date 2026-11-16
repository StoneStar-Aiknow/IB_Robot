"""The CLI stop path must re-export the same words the client and Hermes use.

The historical ``robot_skill_cli.stop_contract`` path stays importable, but it
now resolves one sibling module; the installed Hermes copy receives that
sibling from ``embodied_common`` instead of trying a second source.
"""

from __future__ import annotations

import importlib

import pytest

from embodied_common.control_phrases import STOP_PHRASES as CANONICAL_STOP_PHRASES
from robot_skill_cli.control_phrases import STOP_PHRASES as CLI_STOP_PHRASES
from robot_skill_cli.stop_contract import STOP_PHRASES, is_stop


def test_the_cli_stop_path_reexports_one_canonical_object():
    assert CLI_STOP_PHRASES is CANONICAL_STOP_PHRASES
    assert STOP_PHRASES is CANONICAL_STOP_PHRASES
    historical = importlib.import_module("robot_skill_cli.stop_contract")
    assert historical.STOP_PHRASES is CANONICAL_STOP_PHRASES


@pytest.mark.parametrize("text", ["别动了", "别动", "停止", "停下", "停", "stop", "HALT", "  halt  "])
def test_known_stop_phrases_are_recognized(text):
    assert is_stop(text)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "别动一下", "停下来看看", "stop now", "nonstop", "停止服务", "继续", "go on", None],
)
def test_substrings_and_non_stop_utterances_are_not_stop(text):
    assert not is_stop(text)
