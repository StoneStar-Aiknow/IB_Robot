import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from verification_gate import extract_verified_commit, file_triggers_dual_docker_gate, validate_verified_commit


def test_gate_detects_global_and_package_dependency_changes():
    assert file_triggers_dual_docker_gate("scripts/setup.sh")
    assert file_triggers_dual_docker_gate("requirements/ubuntu.txt")
    assert file_triggers_dual_docker_gate("src/demo/package.xml", "+  <exec_depend>demo</exec_depend>")
    assert not file_triggers_dual_docker_gate("src/demo/package.xml", "+  <description>Updated</description>")
    assert file_triggers_dual_docker_gate("src/demo/package.xml")
    assert not file_triggers_dual_docker_gate("src/demo/setup.py")


def test_verified_commit_requires_one_full_matching_sha():
    head = "a" * 40
    body = f"## Verification\n\n**Verified commit:** `{head}`"

    assert extract_verified_commit(body) == head
    assert validate_verified_commit(body, head.upper()) == head

    with pytest.raises(ValueError, match="exactly one"):
        validate_verified_commit("**Verified commit:** `abc123`", head)
    with pytest.raises(ValueError, match="does not match"):
        validate_verified_commit(f"**Verified commit:** `{'b' * 40}`", head)
    with pytest.raises(ValueError, match="exactly one"):
        validate_verified_commit(f"**Verified commit:** `{head}`\n**Verified commit:** `{head}`", head)
