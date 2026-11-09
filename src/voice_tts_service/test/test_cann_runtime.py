"""Selecting the CANN toolkit an Ascend artifact declares, not whatever the caller sourced."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_tts_service.cann_runtime import ascend_node_environment, cann_version_from_abi, resolve_cann_root


@pytest.mark.parametrize(
    ("abi", "expected"),
    [
        ("cann-8.1.RC1", "8.1.RC1"),
        ("cann-8.0", "8.0"),
        (" cann-8.1.RC1 ", "8.1.RC1"),
        ("acl", None),
        ("", None),
        (None, None),
    ],
)
def test_cann_version_is_parsed_from_the_declared_abi(abi, expected):
    assert cann_version_from_abi(abi) == expected


def test_declared_abi_selects_its_installed_toolkit(tmp_path):
    (tmp_path / "8.1.RC1").mkdir()
    (tmp_path / "latest").mkdir()

    assert resolve_cann_root("cann-8.1.RC1", install_root=tmp_path) == tmp_path / "8.1.RC1"


def test_uninstalled_abi_resolves_to_nothing_rather_than_a_substitute(tmp_path):
    """Substituting another CANN can abort the node inside ACL, hiding why it died."""
    (tmp_path / "latest").mkdir()

    assert resolve_cann_root("cann-9.9.RC9", install_root=tmp_path) is None


def test_missing_abi_resolves_to_nothing(tmp_path):
    assert resolve_cann_root(None, install_root=tmp_path) is None
    assert resolve_cann_root("acl", install_root=tmp_path) is None


def test_unresolved_toolkit_contributes_no_environment():
    # Naming a directory that does not exist would be worse than leaving the caller's env.
    assert ascend_node_environment(None) == {}


def test_nonexistent_root_contributes_no_environment(tmp_path):
    assert ascend_node_environment(tmp_path / "absent") == {}


def test_environment_exposes_only_existing_paths(tmp_path, monkeypatch):
    root = tmp_path / "8.1.RC1"
    (root / "pyACL" / "python" / "site-packages").mkdir(parents=True)
    (root / "lib64").mkdir()
    (root / "opp").mkdir()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    environment = ascend_node_environment(root)

    assert environment["ASCEND_TOOLKIT_HOME"] == str(root)
    assert environment["PYTHONPATH"] == str(root / "pyACL" / "python" / "site-packages")
    assert environment["LD_LIBRARY_PATH"] == str(root / "lib64")
    assert environment["ASCEND_OPP_PATH"] == str(root / "opp")
    # A toolkit without this layout must not contribute a nonexistent path.
    assert "runtime/lib64" not in environment["LD_LIBRARY_PATH"]


def test_environment_keeps_the_callers_entries_behind_the_toolkit(tmp_path, monkeypatch):
    root = tmp_path / "8.1.RC1"
    (root / "lib64").mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/caller/lib")

    assert ascend_node_environment(root)["LD_LIBRARY_PATH"] == f"{root / 'lib64'}:/caller/lib"


def test_foreign_toolkit_entries_are_dropped_from_every_path_variable(tmp_path, monkeypatch):
    """A caller that sourced another toolkit must not leave that ABI behind this one.

    ld.so and Python resolve in order, so the selected root wins for what it ships -- but a library
    or module that only the other version has would still load from it.
    """
    selected = tmp_path / "8.1.RC1"
    foreign = tmp_path / "8.0.RC2"
    for root in (selected, foreign):
        (root / "bin").mkdir(parents=True)
        (root / "lib64").mkdir()
        (root / "pyACL" / "python" / "site-packages").mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{foreign / 'bin'}:/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(foreign / "lib64"))
    monkeypatch.setenv("PYTHONPATH", str(foreign / "pyACL" / "python" / "site-packages"))

    environment = ascend_node_environment(selected)

    assert environment["PATH"] == f"{selected / 'bin'}:/usr/bin"
    assert environment["LD_LIBRARY_PATH"] == str(selected / "lib64")
    assert environment["PYTHONPATH"] == str(selected / "pyACL" / "python" / "site-packages")


def test_inherited_entries_of_the_selected_toolkit_are_not_duplicated(tmp_path, monkeypatch):
    root = tmp_path / "8.1.RC1"
    site = root / "pyACL" / "python" / "site-packages"
    site.mkdir(parents=True)
    # An empty entry (a trailing or doubled separator) names nothing and must not survive.
    monkeypatch.setenv("PYTHONPATH", f"{site}::")

    assert ascend_node_environment(root)["PYTHONPATH"] == str(site)


@pytest.mark.model_bundle("zipvoice")
def test_real_zipvoice_manifest_declares_the_abi_the_builder_reads():
    """Pin the manifest attribute path: a rename there would silently stop selecting a toolkit."""
    from inference_manifest import load_inference_manifest

    bundle = Path(__file__).resolve().parents[3] / "models" / "zipvoice"

    validated = load_inference_manifest(str(bundle), "ascend_310p")
    profiles = dict(validated.deployment.role_runtime_profiles or {})
    if not profiles and validated.deployment.runtime_profile is not None:
        profiles = {"": validated.deployment.runtime_profile}
    ascend = [profile for profile in profiles.values() if profile.backend == "ascend"]

    assert ascend, "the ascend_310p deployment must expose an ascend runtime profile"
    abis = {profile.target.runtime_abi for profile in ascend if profile.target is not None}
    assert abis == {"cann-8.1.RC1"}
    # The declared ABI must be one resolve_cann_root can turn into a toolkit directory name.
    assert cann_version_from_abi(next(iter(abis))) == "8.1.RC1"
