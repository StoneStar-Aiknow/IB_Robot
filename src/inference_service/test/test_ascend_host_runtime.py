"""Host CANN detection and ABI matching.

The three detection cases mirror ``scripts/setup/tests/test_cann_torch_versions.sh``
one for one: setup and the load path must agree about the host's release, so
they are tested against the same layouts.
"""

from __future__ import annotations

import pytest

from inference_service.backends.ascend.host_runtime import (
    detect_cann_release,
    normalize_runtime_abi,
    runtime_abi_matches,
    toolkit_root,
)


def test_a_merged_latest_directory_reports_its_public_release(tmp_path):
    """``latest`` is a tree of component symlinks, so only version.cfg names the release."""
    root = tmp_path / "latest"
    root.mkdir()
    (root / "version.cfg").write_text(
        "# version: 1.0\n"
        "runtime_running_version=[7.7.0.1.238:8.1.RC1]\n"
        "compiler_running_version=[7.7.0.1.238:8.1.RC1]\n"
        "runtime_installed_version=[7.7.0.1.238:8.1.RC1]\n",
        encoding="utf-8",
    )

    assert detect_cann_release(root) == "8.1.RC1"


def test_a_versioned_root_reports_its_install_metadata(tmp_path):
    root = tmp_path / "8.1.RC1"
    (root / "aarch64-linux").mkdir(parents=True)
    (root / "aarch64-linux" / "ascend_toolkit_install.info").write_text(
        "package_name=Ascend-cann-toolkit\nversion=8.1.RC1\ninnerversion=V100R001C21SPC001B238\narch=aarch64\n",
        encoding="utf-8",
    )

    assert detect_cann_release(root) == "8.1.RC1"


def test_component_metadata_ignores_the_internal_build_number(tmp_path):
    """``Version=7.7.0.1.238`` is the component build, not the release users name."""
    root = tmp_path / "component-layout"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "version.info").write_text(
        "Version=7.7.0.1.238\nversion_dir=8.1.RC1\ntimestamp=20250428_203017761\n",
        encoding="utf-8",
    )

    assert detect_cann_release(root) == "8.1.RC1"


def test_a_versioned_directory_name_is_the_last_resort(tmp_path):
    root = tmp_path / "8.3.RC1"
    root.mkdir()

    assert detect_cann_release(root) == "8.3.RC1"


def test_no_toolkit_reports_nothing_rather_than_guessing(tmp_path):
    """A host without an NPU has no release; that is an answer, not a failure."""
    assert detect_cann_release(tmp_path / "absent") is None


def test_the_toolkit_root_follows_the_environment_setup_uses(monkeypatch, tmp_path):
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.setenv("IBR_CANN_TOOLKIT_ROOT", str(tmp_path / "chosen"))

    assert toolkit_root() == tmp_path / "chosen"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cann-8.1.RC1", "8.1.RC1"),
        ("cann_8.1.RC1", "8.1.RC1"),
        ("8.1.RC1", "8.1.RC1"),
        ("", None),
        (None, None),
    ],
)
def test_an_abi_string_reduces_to_its_release(value, expected):
    assert normalize_runtime_abi(value) == expected


def test_the_same_release_matches_however_it_is_written():
    assert runtime_abi_matches("cann-8.1.RC1", "8.1.RC1")


def test_a_different_minor_release_does_not_match():
    """The case that cost a boot: an 8.3-built OM fails to load on 8.1."""
    assert not runtime_abi_matches("cann-8.3.RC1", "8.1.RC1")


def test_a_patch_difference_still_matches():
    assert runtime_abi_matches("cann-8.1.RC1", "8.1.RC2")


def test_an_unknown_side_matches_so_the_caller_decides():
    """Refusing on an unreadable host would be worse than not checking at all."""
    assert runtime_abi_matches(None, "8.1.RC1")
    assert runtime_abi_matches("cann-8.1.RC1", None)
