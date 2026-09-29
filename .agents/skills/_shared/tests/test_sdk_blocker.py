"""SDK-free regression must apply in pytest and in script subprocesses."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["atomgit_sdk", "atomgit_sdk.client"])
def test_sdk_import_is_blocked_in_parent_and_child(name):
    marker = "blocked by the collaboration test import blocker"
    with pytest.raises(ImportError, match=marker):
        importlib.import_module(name)
    result = subprocess.run(
        [sys.executable, "-c", f"import json, argparse; import {name}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and marker in result.stderr
