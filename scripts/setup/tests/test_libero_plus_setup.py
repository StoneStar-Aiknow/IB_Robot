"""Setup regressions without downloading assets or installing system/Python packages."""

import hashlib
import importlib.util
import shlex
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_DIR = REPO_ROOT / "scripts" / "setup"
RULE = "libero/libero/assets"
PINNED_ZIP_PREFIX = (
    "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/"
)


@pytest.fixture
def config_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("libero_plus_config", SETUP_DIR / "libero_plus_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("IBROBOT_LIBERO_PLUS_ASSETS_DIR", raising=False)
    return module


@pytest.fixture
def small_asset_payload(config_module, monkeypatch):
    payload = {"scenes/arena.xml": b'<mesh file="arena.msh"/>', "scenes/arena.msh": b"pinned mesh"}
    digest = hashlib.sha256()
    for name in sorted(payload, key=Path):
        digest.update(name.encode() + b"\0" + hashlib.sha256(payload[name]).digest())
    expected = len(payload), digest.hexdigest()
    integrity = config_module._integrity
    monkeypatch.setitem(integrity.EXPECTED_ASSET_FILE_COUNT, "libero-plus", expected[0])
    monkeypatch.setitem(integrity.EXPECTED_ASSET_TREE_SHA256, "libero-plus", expected[1])
    return payload, expected


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Setup Test", "-c", "user.email=setup@example.invalid", "commit", "-qm", "fixture")


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q")
    (seed / "libero" / "libero" / "benchmark").mkdir(parents=True)
    (seed / "libero" / "libero" / "benchmark" / ".gitkeep").touch()
    (seed / "tracked.txt").write_text("original\n")
    commit(seed)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git(workspace, "init", "-q")
    git(workspace, "-c", "protocol.file.allow=always", "submodule", "add", "--quiet", str(seed), "libs/libero_plus")
    commit(workspace)
    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.delenv("IBROBOT_LIBERO_PLUS_ROOT", raising=False)
    monkeypatch.delenv("IBROBOT_LIBERO_PLUS_ASSETS_DIR", raising=False)
    return workspace, workspace / "libs" / "libero_plus", workspace / "venv" / "ibrobot_libero_plus"


def exclude_path(provider):
    path = Path(git(provider, "rev-parse", "--git-path", "info/exclude"))
    return path if path.is_absolute() else provider / path


def run_shell(workspace, script):
    # Source the real environment first, then isolate only the mock workspace.
    return subprocess.run(
        [
            "bash",
            "-c",
            f"source {shlex.quote(str(REPO_ROOT / '.shrc_local'))}\n"
            f"set -euo pipefail\nexport WORKSPACE={shlex.quote(str(workspace))}\n{script}",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("existing_link", [False, True])
def test_prepare_assets_reuses_cache_and_repairs_exclude(config_module, checkout, monkeypatch, existing_link):
    _, provider, config = checkout
    assets = config / "assets"
    assets.mkdir(parents=True)
    package_assets = provider / RULE
    if existing_link:
        package_assets.symlink_to(assets, target_is_directory=True)
    exclude = exclude_path(provider)
    exclude.write_text("# keep this comment without a trailing newline")
    validate = Mock(return_value=(5, "tree-sha256"))
    download = Mock(side_effect=AssertionError("valid assets must not be downloaded"))
    monkeypatch.setattr(config_module, "_validate_assets", validate)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))

    for _ in range(2):
        assert config_module._prepare_assets(config, provider) == (assets.resolve(), 5, "tree-sha256")
    assert package_assets.is_symlink()
    assert package_assets.resolve() == assets
    assert exclude.read_text() == f"# keep this comment without a trailing newline\n{RULE}\n"
    assert git(provider, "status", "--porcelain") == ""
    assert validate.call_count == 2
    download.assert_not_called()


def test_prepare_assets_pending_manual_download(config_module, checkout, monkeypatch, capsys):
    _, provider, config = checkout
    download = Mock(side_effect=AssertionError("assets must never be downloaded by setup"))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download, snapshot_download=download)
    )
    assets = config / "assets"

    # Two runs: the pending state must be idempotent and never download.
    for _ in range(2):
        assert config_module._prepare_assets(config, provider) == (assets.resolve(), 0, "pending")
    assert assets.is_dir()
    package_assets = provider / RULE
    assert package_assets.is_symlink()
    assert package_assets.resolve() == assets
    out = capsys.readouterr().out
    assert "WARN: LIBERO-plus assets are missing; setup does not download them." in out
    assert "HINT:   huggingface-cli download Sylvest/LIBERO-plus" in out
    assert "IBROBOT_LIBERO_PLUS_ASSETS_DIR" in out
    assert git(provider, "status", "--porcelain") == ""
    download.assert_not_called()


def test_config_override_is_used_for_generation(config_module, checkout, monkeypatch, tmp_path):
    workspace, _, default_config = checkout
    config = tmp_path / "external-config"
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_CONFIG_DIR", str(config))
    result = config_module.generate_config(str(workspace))
    assert Path(result[0]) == config / "config.yaml"
    assert (config / "provider" / "libero" / "__init__.py").is_file()
    assert not default_config.exists()


def test_prepare_assets_extracts_user_provided_zip(config_module, checkout, monkeypatch):
    _, provider, config = checkout
    config.mkdir(parents=True, exist_ok=True)
    assets = config / "assets"
    with zipfile.ZipFile(config / "assets.zip", "w") as archive:
        archive.writestr("assets/scenes/arena.xml", "<xml/>")

    def validate(path):
        assert path != assets
        assert (path / "scenes" / "arena.xml").read_text() == "<xml/>"
        assert not assets.exists()
        return 5, "tree-sha256"

    monkeypatch.setattr(config_module, "_validate_assets", validate)
    assert config_module._prepare_assets(config, provider) == (assets.resolve(), 5, "tree-sha256")
    assert (assets / "scenes" / "arena.xml").read_text() == "<xml/>"
    assert (config / "assets.zip").is_file()


@pytest.mark.parametrize("existing_link", [False, True])
@pytest.mark.parametrize("with_zip", [False, True])
def test_populated_invalid_cache_is_preserved(config_module, checkout, existing_link, with_zip):
    _, provider, config = checkout
    assets = config / "assets"
    (assets / "scenes").mkdir(parents=True)
    partial = assets / "scenes" / "partial.xml"
    partial.write_bytes(b"partial")
    if existing_link:
        (provider / RULE).symlink_to(assets, target_is_directory=True)
    if with_zip:
        (config / "assets.zip").write_bytes(b"not even opened")
    with pytest.raises(config_module.LiberoPlusConfigError, match="Move this path aside or repair it"):
        config_module._prepare_assets(config, provider)
    assert partial.read_bytes() == b"partial"
    assert (provider / RULE).is_symlink() == existing_link


@pytest.mark.parametrize(
    "member",
    [
        "../escape.xml",
        "assets/../escape.xml",
        "/escape.xml",
        "assets//escape.xml",
        "assets/./escape.xml",
        "C:/escape.xml",
        "assets/C:/escape.xml",
        "assets\\..\\escape.xml",
    ],
)
def test_zip_rejects_unsafe_paths(config_module, tmp_path, member):
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(member, b"unsafe")
    target = tmp_path / "assets"
    with pytest.raises(config_module.LiberoPlusConfigError, match="Unsafe ZIP member"):
        config_module._extract_assets_zip(zip_path, target)
    assert not target.exists()
    assert zip_path.is_file()
    assert set(tmp_path.iterdir()) == {zip_path}


def test_zip_rejects_symlink(config_module, tmp_path):
    zip_path = tmp_path / "assets.zip"
    entry = zipfile.ZipInfo("assets/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(entry, "../outside")
    with pytest.raises(config_module.LiberoPlusConfigError, match="Unsafe ZIP member"):
        config_module._extract_assets_zip(zip_path, tmp_path / "assets")
    assert set(tmp_path.iterdir()) == {zip_path}


@pytest.mark.parametrize("member", ["../escape.xml", "scenes/../../escape.xml", "scenes/C:/escape.xml"])
def test_pinned_zip_wrapper_does_not_hide_unsafe_paths(config_module, tmp_path, member):
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(PINNED_ZIP_PREFIX + member, b"unsafe")
    with pytest.raises(config_module.LiberoPlusConfigError, match="Unsafe ZIP member"):
        config_module._extract_assets_zip(zip_path, tmp_path / "assets")
    assert set(tmp_path.iterdir()) == {zip_path}


@pytest.mark.parametrize("failure", ["mutation", "deletion"])
def test_mesh_integrity_is_required(config_module, small_asset_payload, tmp_path, failure):
    payload, expected = small_asset_payload
    for name in config_module._integrity.EXPECTED_ASSET_DIRS["libero-plus"]:
        (tmp_path / name).mkdir()
    for name, content in payload.items():
        (tmp_path / name).write_bytes(content)
    assert config_module._validate_assets(tmp_path) == expected
    mesh = tmp_path / "scenes" / "arena.msh"
    if failure == "mutation":
        mesh.write_bytes(b"modified mesh")
    else:
        mesh.unlink()
    with pytest.raises(
        config_module.LiberoPlusConfigError, match="SHA-256" if failure == "mutation" else "business files"
    ):
        config_module._validate_assets(tmp_path)


@pytest.mark.parametrize("prefix", ["", "assets/", PINNED_ZIP_PREFIX], ids=["flat", "assets", "pinned-upstream"])
@pytest.mark.parametrize("wrapper_entry", [False, True], ids=["implicit-wrapper", "explicit-wrapper"])
def test_zip_layout_validates_mesh_and_relative_paths(
    config_module, small_asset_payload, tmp_path, prefix, wrapper_entry
):
    payload, expected = small_asset_payload
    zip_path = tmp_path / "assets.zip"
    target = tmp_path / "assets"
    target.mkdir()
    with zipfile.ZipFile(zip_path, "w") as archive:
        if prefix and wrapper_entry:
            archive.writestr(prefix, b"")
        for name in config_module._integrity.EXPECTED_ASSET_DIRS["libero-plus"]:
            archive.writestr(f"{prefix}{name}/", b"")
        for name, content in payload.items():
            archive.writestr(prefix + name, content)
    assert config_module._extract_assets_zip(zip_path, target) == expected
    assert config_module._validate_assets(target) == expected
    assert (target / "scenes" / "arena.msh").read_bytes() == payload["scenes/arena.msh"]
    assert set(tmp_path.iterdir()) == {target, zip_path}


@pytest.mark.parametrize("layout", ["unknown-wrapper", "mixed-wrappers"])
def test_zip_does_not_guess_payload_prefix(config_module, small_asset_payload, tmp_path, layout):
    payload, _ = small_asset_payload
    zip_path = tmp_path / "assets.zip"
    target = tmp_path / "assets"
    target.mkdir()
    prefix = "unknown/assets/" if layout == "unknown-wrapper" else PINNED_ZIP_PREFIX
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in config_module._integrity.EXPECTED_ASSET_DIRS["libero-plus"]:
            archive.writestr(f"{prefix}{name}/", b"")
        for name, content in payload.items():
            archive.writestr(prefix + name, content)
        if layout == "mixed-wrappers":
            archive.writestr("assets/scenes/unexpected.xml", b"unrelated")
    with pytest.raises(config_module.LiberoPlusConfigError, match="missing required directories"):
        config_module._extract_assets_zip(zip_path, target)
    assert list(target.iterdir()) == []
    assert set(tmp_path.iterdir()) == {target, zip_path}


@pytest.mark.parametrize("empty_target", [False, True])
def test_zip_invalid_tree_never_published(config_module, tmp_path, empty_target):
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("assets/scenes/arena.xml", "<xml/>")
    target = tmp_path / "assets"
    if empty_target:
        target.mkdir()
    with pytest.raises(config_module.LiberoPlusConfigError, match="missing required directories"):
        config_module._extract_assets_zip(zip_path, target)
    assert target.exists() == empty_target
    if empty_target:
        assert list(target.iterdir()) == []
    assert zip_path.is_file()
    assert not list(tmp_path.glob("libero-plus-extract-*"))


def test_zip_refuses_merge(config_module, tmp_path):
    target = tmp_path / "assets"
    target.mkdir()
    (target / "user.xml").write_bytes(b"preserve")
    with pytest.raises(config_module.LiberoPlusConfigError, match="Refusing to merge"):
        config_module._extract_assets_zip(tmp_path / "not-opened.zip", target)
    assert (target / "user.xml").read_bytes() == b"preserve"


def test_zip_hash_failure_preserves_placeholder(config_module, tmp_path, monkeypatch):
    integrity = config_module._integrity
    monkeypatch.setitem(integrity.EXPECTED_ASSET_FILE_COUNT, "libero-plus", 1)
    target = tmp_path / "assets"
    target.mkdir()
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in integrity.EXPECTED_ASSET_DIRS["libero-plus"]:
            archive.writestr(f"assets/{name}/", "")
        archive.writestr("assets/scenes/arena.xml", "tampered")
    with pytest.raises(config_module.LiberoPlusConfigError, match="SHA-256"):
        config_module._extract_assets_zip(zip_path, target)
    assert list(target.iterdir()) == []
    assert set(tmp_path.iterdir()) == {target, zip_path}


def test_zip_cannot_replace_target_populated_during_validation(config_module, tmp_path, monkeypatch):
    target = tmp_path / "assets"
    target.mkdir()
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("scenes/arena.xml", "<xml/>")

    def validate(stage):
        (target / "user.xml").write_bytes(b"preserve")
        return 1, "sha"

    monkeypatch.setattr(config_module, "_validate_assets", validate)
    with pytest.raises(config_module.LiberoPlusConfigError, match="target changed during extraction"):
        config_module._extract_assets_zip(zip_path, target)
    assert (target / "user.xml").read_bytes() == b"preserve"
    assert not (target / "scenes").exists()
    assert set(tmp_path.iterdir()) == {target, zip_path}


@pytest.mark.parametrize("dangling", [False, True])
def test_invalid_cache_symlink_is_preserved(config_module, checkout, dangling):
    workspace, provider, config = checkout
    config.mkdir(parents=True)
    source = workspace / "old-source"
    if not dangling:
        source.mkdir()
    (config / "assets").symlink_to(source, target_is_directory=True)
    (provider / RULE).symlink_to(config / "assets", target_is_directory=True)
    with pytest.raises(config_module.LiberoPlusConfigError, match="no files were removed"):
        config_module._prepare_assets(config, provider)
    assert (config / "assets").is_symlink()
    assert (config / "assets").resolve() == source


def test_plus_override_replaces_only_empty_placeholder(config_module, checkout, monkeypatch):
    workspace, provider, config = checkout
    assets = config / "assets"
    assets.mkdir(parents=True)
    (provider / RULE).symlink_to(assets, target_is_directory=True)
    source = workspace / "selected-assets"
    source.mkdir()
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ASSETS_DIR", str(source))

    def validate(path):
        if path.resolve() == source:
            return 1, "sha256"
        raise config_module.LiberoPlusConfigError("placeholder")

    monkeypatch.setattr(config_module, "_validate_assets", validate)
    assert config_module._prepare_assets(config, provider) == (source, 1, "sha256")
    assert assets.is_symlink()


@pytest.mark.parametrize("wrapped", [False, True])
def test_zip_publishes_validated_tree_and_preserves_archive(config_module, tmp_path, monkeypatch, capsys, wrapped):
    target = tmp_path / "assets"
    target.mkdir()
    zip_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(("assets/" if wrapped else "") + "scenes/arena.xml", "<xml/>")

    def validate(stage):
        assert stage.parent.parent == target.parent
        assert stage != target
        assert list(target.iterdir()) == []
        assert (stage / "scenes" / "arena.xml").read_text() == "<xml/>"
        return 1, "validated-sha"

    monkeypatch.setattr(config_module, "_validate_assets", validate)
    assert config_module._extract_assets_zip(zip_path, target) == (1, "validated-sha")
    assert (target / "scenes" / "arena.xml").read_text() == "<xml/>"
    assert zip_path.is_file()
    assert "remove the ZIP manually" in capsys.readouterr().out


def test_plus_assets_revision_is_immutable(config_module, checkout, capsys):
    _, provider, config = checkout
    assert config_module.HF_ASSETS_REVISION == "b548dd25ee0401c46217ba7e3614a598e8979e48"
    config_module._prepare_assets(config, provider)
    assert f"--revision {config_module.HF_ASSETS_REVISION}" in capsys.readouterr().out


def test_prepare_assets_env_var_invalid_fails(config_module, checkout, monkeypatch):
    workspace, provider, config = checkout
    bogus = workspace / "bogus-assets"
    bogus.mkdir()
    monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ASSETS_DIR", str(bogus))
    with pytest.raises(config_module.LiberoPlusConfigError, match="Invalid IBROBOT_LIBERO_PLUS_ASSETS_DIR"):
        config_module._prepare_assets(config, provider)


def test_generate_config_pending_writes_config_shim_and_hint(config_module, checkout, capsys):
    workspace, _, config = checkout
    config_path, submodule_root, assets_dir, count, tree_sha256 = config_module.generate_config(str(workspace))
    assert count == 0
    assert tree_sha256 == "pending"
    assert Path(config_path) == config / "config.yaml"
    assert Path(submodule_root) == workspace / "libs" / "libero_plus"
    assert assets_dir == str((config / "assets").resolve())
    assert (config / "provider" / "libero" / "__init__.py").is_file()
    out = capsys.readouterr().out
    assert "HINT:   huggingface-cli download Sylvest/LIBERO-plus" in out


@pytest.mark.parametrize("kind", ["foreign_link", "directory", "file", "missing"])
def test_exclude_does_not_hide_unmanaged_assets(config_module, checkout, kind):
    workspace, provider, config = checkout
    package_assets = provider / RULE
    if kind == "foreign_link":
        package_assets.symlink_to(workspace / "user-assets", target_is_directory=True)
    elif kind == "directory":
        package_assets.mkdir()
        (package_assets / "user.xml").touch()
    elif kind == "file":
        package_assets.touch()
    exclude = exclude_path(provider)
    before = exclude.read_bytes()
    config_module.ensure_managed_assets_exclude(config, provider)
    assert exclude.read_bytes() == before
    if kind != "missing":
        assert RULE in git(provider, "status", "--porcelain")


@pytest.mark.parametrize("explicit_override", [False, True])
def test_explicit_managed_paths(config_module, checkout, monkeypatch, explicit_override):
    workspace, provider, config = checkout
    selected = workspace / "selected-assets"
    selected.mkdir()
    if explicit_override:
        monkeypatch.setenv("IBROBOT_LIBERO_PLUS_ASSETS_DIR", str(selected))
    else:
        # A custom config directory is also explicitly owned by the caller.
        config = workspace / "custom-config"
        config.mkdir()
        (config / "assets").symlink_to(selected, target_is_directory=True)
    (provider / RULE).symlink_to(selected, target_is_directory=True)
    config_module.ensure_managed_assets_exclude(config, provider)
    config_module.ensure_managed_assets_exclude(config, provider)
    assert exclude_path(provider).read_text().splitlines().count(RULE) == 1


def test_relative_git_path_is_relative_to_provider(config_module, tmp_path):
    provider = tmp_path / "provider"
    (provider / "libero" / "libero").mkdir(parents=True)
    git(provider, "init", "-q")
    assert git(provider, "rev-parse", "--git-path", "info/exclude") == ".git/info/exclude"
    config = tmp_path / "config"
    (provider / RULE).symlink_to(config / "assets", target_is_directory=True)
    config_module.ensure_managed_assets_exclude(config, provider)
    assert RULE in (provider / ".git" / "info" / "exclude").read_text().splitlines()


def test_prepare_cli_needs_no_yaml_or_assets_validation(checkout):
    workspace, provider, config = checkout
    # A dangling managed link can be repaired before venv/dependencies exist.
    (provider / RULE).symlink_to(config / "assets", target_is_directory=True)
    result = run_shell(
        workspace,
        f"{shlex.quote(sys.executable)} -I -S {shlex.quote(str(SETUP_DIR / 'libero_plus_config.py'))} "
        f"--prepare-submodule {shlex.quote(str(provider))}",
    )
    assert result.returncode == 0, result.stderr
    assert RULE in exclude_path(provider).read_text().splitlines()
    assert not config.exists()


@pytest.mark.parametrize("state", ["managed", "tracked", "untracked", "foreign", "file", "missing"])
def test_first_sync_repairs_old_setup_but_preserves_dirty_protection(checkout, state):
    workspace, provider, config = checkout
    package_assets = provider / RULE
    if state not in {"missing", "file"}:
        target = workspace / "user-assets" if state == "foreign" else config / "assets"
        package_assets.symlink_to(target, target_is_directory=True)
    if state == "tracked":
        (provider / "tracked.txt").write_text("user edit\n")
    elif state == "untracked":
        (provider / "user.txt").touch()
    elif state == "file":
        package_assets.touch()
    elif state == "missing":
        git(workspace, "submodule", "deinit", "--force", "libs/libero_plus")
    result = run_shell(
        workspace,
        f"""
source {shlex.quote(str(SETUP_DIR / "submodules.sh"))}
SCRIPT_DIR={shlex.quote(str(REPO_ROOT / "scripts"))}
SETUP_BOOTSTRAP_PYTHON_BIN={shlex.quote(sys.executable)}
INSTALL_BENCHMARK_DEPS=true
SETUP_PROFILE=full
log_info() {{ :; }}
log_done() {{ :; }}
log_warn() {{ printf '%s\\n' "$*"; }}
log_error() {{ printf '%s\\n' "$*" >&2; }}
ask_yn() {{ return 0; }}
run_cmd() {{
    if [[ "$1 $2 $3" == 'git submodule update' ]]; then
        printf 'UPDATE: %s\\n' "$*"
    else
        "$@"
    fi
}}
update_submodules
""",
    )
    assert result.returncode == 0, result.stderr
    if state in {"managed", "missing"}:
        assert "UPDATE: git submodule update --init libs/libero_plus" in result.stdout
        assert "has uncommitted changes" not in result.stdout
    else:
        assert "UPDATE:" not in result.stdout
        assert "has uncommitted changes" in result.stdout
    if state in {"managed", "tracked", "untracked"}:
        assert RULE in exclude_path(provider).read_text().splitlines()


def test_failed_exclude_repair_stops_sync(checkout):
    workspace, _, _ = checkout
    result = run_shell(
        workspace,
        f"""
source {shlex.quote(str(SETUP_DIR / "submodules.sh"))}
SCRIPT_DIR={shlex.quote(str(REPO_ROOT / "scripts"))}
SETUP_BOOTSTRAP_PYTHON_BIN=false
INSTALL_BENCHMARK_DEPS=true
log_info() {{ :; }}
log_error() {{ printf '%s\\n' "$*" >&2; }}
run_cmd() {{ "$@"; }}
submodule_update_block_reason() {{ printf 'DIRTY_CHECK_REACHED\\n'; return 1; }}
update_submodules
printf 'CONTINUED\\n'
""",
    )
    assert result.returncode == 1
    assert "Failed to prepare LIBERO-plus submodule excludes" in result.stderr
    assert "DIRTY_CHECK_REACHED" not in result.stdout
    assert "CONTINUED" not in result.stdout


@pytest.mark.parametrize("use_sudo", [True, False])
@pytest.mark.parametrize("enabled,install_status", [(True, 0), (True, 17), (False, 0)])
def test_system_dependency_hook_uses_run_sudo_and_propagates_failure(tmp_path, use_sudo, enabled, install_status):
    result = run_shell(
        tmp_path,
        f"""
source {shlex.quote(str(SETUP_DIR / "common.sh"))}
source {shlex.quote(str(SETUP_DIR / "platforms" / "ubuntu-22.04.sh"))}
USE_SUDO={str(use_sudo).lower()}
INSTALL_BENCHMARK_DEPS={str(enabled).lower()}
sudo() {{ printf 'SUDO\\n'; "$@"; }}
apt-get() {{ printf 'APT: %s\\n' "$*"; return {install_status}; }}
platform_install_benchmark_system_deps
printf 'CONTINUED\\n'
""",
    )
    assert result.returncode == (install_status if enabled else 0), result.stderr
    if enabled:
        assert (
            "APT: install -y --no-install-recommends libosmesa6-dev ffmpeg libmagickwand-6.q16-6 -qq" in result.stdout
        )
        assert ("SUDO" in result.stdout) == use_sudo
        assert ("CONTINUED" in result.stdout) == (install_status == 0)
    else:
        assert "APT:" not in result.stdout
        assert "SUDO" not in result.stdout
    assert "remove" not in result.stdout


@pytest.mark.parametrize("wand_status", [0, 1])
def test_wheels_are_followed_by_required_wand_import(tmp_path, wand_status):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").touch()
    result = run_shell(
        tmp_path,
        f"""
source {shlex.quote(str(SETUP_DIR / "python_venv.sh"))}
SCRIPT_DIR={shlex.quote(str(REPO_ROOT / "scripts"))}
VENV_PYTHON=mock_python
SETUP_BOOTSTRAP_PYTHON_BIN=mock_python
INSTALL_BENCHMARK_DEPS=true
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_ACCEL_ID=cpu
SETUP_PROFILE=inference
BENCHMARK_PIP_CONSTRAINTS=/dev/null
CALLS="${{WORKSPACE}}/calls"
log_info() {{ :; }}
log_warn() {{ :; }}
log_error() {{ printf '%s\\n' "$*" >&2; }}
log_done() {{ :; }}
platform_supports_local_workspace_build() {{ return 0; }}
benchmark_prepare_torch_profile() {{ :; }}
benchmark_install_runtime_abi() {{ :; }}
install_accel_requirements() {{ :; }}
accel_post_requirements() {{ :; }}
accel_pip_constraints() {{ :; }}
accel_verify_runtime() {{ :; }}
sudo() {{ printf 'FORBIDDEN sudo\\n' >> "${{CALLS}}"; exit 99; }}
apt-get() {{ printf 'FORBIDDEN apt-get\\n' >> "${{CALLS}}"; exit 99; }}
run_cmd() {{ printf 'RUN: %s\\n' "$*" >> "${{CALLS}}"; }}
mock_python() {{
    if [[ "$*" == '-c import wand.image' ]]; then
        printf 'WAND: %s\\n' "${{PYTHONNOUSERSITE}}" >> "${{CALLS}}"
        return {wand_status}
    fi
    case "$*" in
        *'print(sys.version_info[0])'*) printf '3\\n' ;;
        *'print(sys.version_info[1])'*) printf '10\\n' ;;
        *'sys.version_info.major'*) printf '3.10\\n' ;;
        *'sys.version_info[:3]'*) printf '3.10.12\\n' ;;
        *'numpy.__version__'*) printf '1.26.4\\n' ;;
    esac
    return 0
}}
setup_python_venv
printf 'COMPLETED\\n'
""",
    )
    assert result.returncode == wand_status, result.stderr
    calls = (tmp_path / "calls").read_text()
    wheels = f"-r {tmp_path}/requirements/libero_plus.txt --quiet"
    assert wheels in calls
    assert "WAND: 1\n" in calls
    assert calls.index(wheels) < calls.index("WAND:")
    assert "FORBIDDEN" not in calls
    assert ("COMPLETED" in result.stdout) == (wand_status == 0)
    if wand_status:
        assert "LIBERO-plus requires importable wand.image" in result.stderr
        assert not (venv / "benchmark_libero_plus_config_output.txt").exists()
