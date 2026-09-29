"""Project workflow regressions; CLI responses are supplied by the shared fake."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import atomgit_cli
from atomgit_cli import (
    AtomgitURLError,
    CLIAuthError,
    CLIAuthStatus,
    CLIConflictError,
    CLIExecutionError,
    CLIJSONError,
    CLINotFoundError,
    CLISelectionError,
    CLITimeoutError,
    MissingTargetError,
    RepoTarget,
    TargetConflictError,
    TargetResolutionError,
    check_auth_status,
    check_cli_identity,
    config_repo_defaults,
    load_config,
    parse_atomgit_url,
    read_git_remotes,
    resolve_cli,
    resolve_source,
    resolve_target,
    run,
    run_json,
    run_paginated,
    validate_config_base_url,
)
from entrypoint_fixture import EntrypointFakeCLI

SPECIAL_BODY = (
    "摘要 line1\nline2 `backtick` \"double quotes\" 'single' $HOME $(id:fake) back\\\\slash 中文 ünïcödé 🤖 tab\tend\n"
)


class FakeCLI(EntrypointFakeCLI):
    """Shared executable fixture with a hostile PATH for selection tests."""

    def env(self, **extra):
        return super().env(**({"PATH": str(self.bin_dir)} | extra))

    def write_decoy(self, name, marker, *, stdout="gc: graphviz graph count tool\n", exit_code=1):
        path = self.bin_dir / name
        path.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"with open({str(marker)!r}, 'a') as handle: handle.write('invoked\\n')\n"
            f"sys.stdout.write({stdout!r})\nsys.exit({exit_code})\n"
        )
        path.chmod(0o755)
        return path


@pytest.fixture
def fake(tmp_path: Path) -> FakeCLI:
    fake = FakeCLI(tmp_path)
    fake.configure([], default={"stdout": "", "stderr": "no rule", "exit": 64})
    return fake


@pytest.fixture
def fake_env(fake: FakeCLI) -> dict[str, str]:
    return fake.env(IBROBOT_ATOMGIT_CLI=str(fake.exe))


# CLI resolution (task 2.1)


def test_resolve_prefers_explicit_absolute_path_over_path(fake: FakeCLI, tmp_path: Path):
    other_dir = tmp_path / "other-bin"
    other_dir.mkdir()
    decoy = other_dir / "gitcode"
    decoy.write_text("#!/bin/sh\nexit 99\n")
    decoy.chmod(0o755)
    env = fake.env(IBROBOT_ATOMGIT_CLI=str(fake.exe), PATH=f"{other_dir}:{fake.bin_dir}")

    assert resolve_cli(env) == str(fake.exe)


def test_resolve_rejects_relative_explicit_path(fake: FakeCLI):
    env = fake.env(IBROBOT_ATOMGIT_CLI="bin/gitcode")

    with pytest.raises(CLISelectionError, match="absolute"):
        resolve_cli(env)


def test_resolve_invalid_explicit_path_never_falls_back_to_path(fake: FakeCLI, tmp_path: Path):
    marker = tmp_path / "decoy-marker"
    fake.write_decoy("gitcode", marker)  # PATH 'gitcode' exists, but must not be used
    env = fake.env(IBROBOT_ATOMGIT_CLI=str(fake.bin_dir / "missing-gitcode"))

    with pytest.raises(CLISelectionError, match="missing file"):
        resolve_cli(env)
    assert marker.is_file() is False  # the PATH installation was never even probed as fallback


def test_resolve_rejects_non_executable_explicit_path(fake: FakeCLI):
    plain = fake.exe.with_name("gitcode-plain")
    plain.write_text("#!/bin/sh\nexit 0\n")
    plain.chmod(0o644)
    env = fake.env(IBROBOT_ATOMGIT_CLI=str(plain))

    with pytest.raises(CLISelectionError, match="non-executable"):
        resolve_cli(env)


def test_resolve_only_graphviz_installed_reports_missing_cli(fake: FakeCLI, tmp_path: Path):
    fake.exe.unlink()
    marker = tmp_path / "gc-marker"
    fake.write_decoy("gc", marker)
    env = fake.env()

    with pytest.raises(CLISelectionError, match="gitcode"):
        resolve_cli(env)
    assert marker.is_file() is False  # Graphviz gc was never invoked


def test_resolve_gc_and_gitcode_coexistence_uses_gitcode(fake: FakeCLI, tmp_path: Path):
    marker = tmp_path / "gc-marker"
    fake.write_decoy("gc", marker)
    env = fake.env()

    assert resolve_cli(env) == str(fake.exe.resolve())
    assert marker.is_file() is False


# Identity and version gate (task 2.1)


def test_identity_rejects_same_name_graphviz_binary(fake: FakeCLI, tmp_path: Path):
    fake.exe.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write('-1 nodes -1 edges\\n')\nsys.exit(0)\n")
    fake.exe.chmod(0o755)
    env = fake.env(IBROBOT_ATOMGIT_CLI=str(fake.exe))

    with pytest.raises(CLISelectionError, match="not the official AtomGit CLI"):
        check_cli_identity(env=env)


def test_identity_rejects_version_below_verified_floor(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["version"], "stdout": "gitcode version 0.13.0\n", "exit": 0}])

    with pytest.raises(CLISelectionError, match="older than the verified minimum"):
        check_cli_identity(env=fake_env)


def test_identity_rejects_failing_version_probe(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["version"], "stdout": "", "stderr": "boom", "exit": 1}])

    with pytest.raises(CLISelectionError, match="failed 'version'"):
        check_cli_identity(env=fake_env)


# Authentication diagnostics (task 2.1)


def test_auth_status_not_logged_in_is_diagnostic_not_crash(fake: FakeCLI, fake_env):
    fake.configure(
        [
            {
                "argv_contains": ["auth", "status"],
                "stdout": "atomgit.com\n  ✗ Not logged in\n",
                "stderr": "Error: not logged in to atomgit.com\n",
                "exit": 4,
            }
        ]
    )

    status = check_auth_status(env=fake_env)

    assert isinstance(status, CLIAuthStatus)
    assert status.logged_in is False
    assert status.returncode == 4
    assert "auth login" in status.guidance
    argv = fake.invocations()[0]["argv"]
    assert "--show-token" not in argv  # credentials are never printed (E3)


def test_auth_status_unexpected_exit_raises(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["auth", "status"], "stderr": "garbled", "exit": 1}])

    with pytest.raises(CLIExecutionError, match="unexpected 'auth status' exit code 1"):
        check_auth_status(env=fake_env)


# Execution boundary: argv, stdin, host pinning (task 2.3)


def test_run_preserves_multiline_and_special_characters_in_argv(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["pr", "reply"], "exit": 0}])
    args = ["pr", "reply", "123", "--discussion", "d-1", "--body", SPECIAL_BODY]

    result = run(args, env=fake_env)

    assert result.ok
    invocation = fake.invocations()[0]
    assert invocation["argv"] == args + ["--no-interactive"]  # argv-level fidelity, no shell


def test_run_pins_gc_host_and_no_interactive(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["repo", "view"], "stdout": "{}", "exit": 0}])
    env = fake.env(IBROBOT_ATOMGIT_CLI=str(fake.exe), GC_HOST="evil.example.com")

    run(["repo", "view", "openEuler/IB_Robot", "--json"], env=env)

    invocation = fake.invocations()[0]
    assert invocation["env"]["GC_HOST"] == "atomgit.com"
    assert invocation["argv"][-1] == "--no-interactive"


def test_require_data_raises_on_nonzero_exit_even_when_json_decoded(fake: FakeCLI, fake_env):
    fake.configure(
        [
            {
                "argv_contains": ["pr", "view"],
                "stdout": '{"number": 7, "title": "partial"}',
                "stderr": "enrichment failed",
                "exit": 3,
            }
        ]
    )

    result = run_json(["pr", "view", "7", "-R", "o/r", "--json"], env=fake_env)

    assert result.data == {"number": 7, "title": "partial"}  # partial data stays inspectable
    with pytest.raises(CLINotFoundError):
        result.require_data()  # but is never accepted as complete success


def test_run_json_invalid_output_sets_json_error(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["pr", "view"], "stdout": "not-json{{", "exit": 0}])

    result = run_json(["pr", "view", "7", "-R", "o/r", "--json"], env=fake_env)

    assert result.ok is True
    assert result.data is None
    assert result.json_error
    with pytest.raises(CLIJSONError, match="no usable JSON"):
        result.require_data()


def test_run_json_empty_stdout_sets_json_error(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["issue", "close"], "stdout": "", "exit": 0}])

    result = run_json(["issue", "close", "5", "-R", "o/r", "--yes"], env=fake_env)

    assert result.ok is True
    assert result.data is None
    with pytest.raises(CLIJSONError, match="empty stdout"):
        result.require_data()


@pytest.mark.parametrize(
    "exit_code,expected_class",
    [
        (1, CLIExecutionError),
        (3, CLINotFoundError),
        (4, CLIAuthError),
        (5, CLIConflictError),
    ],
)
def test_failure_classification_matrix(fake: FakeCLI, fake_env, exit_code, expected_class):
    fake.configure([{"argv_contains": ["pr", "view"], "stderr": "boom", "exit": exit_code}])

    result = run(["pr", "view", "7", "-R", "o/r", "--json"], env=fake_env)
    failure = result.failure()

    assert isinstance(failure, expected_class)
    assert failure.returncode == exit_code
    assert failure.argv[-2] == "--json"
    if exit_code == 4:
        assert atomgit_cli.AUTH_LOGIN_GUIDANCE in str(failure)


def test_failure_message_omits_bodies_flags_and_raw_cli_output(fake: FakeCLI, fake_env):
    secret_body = "SECRET reply body $(id:fake)"
    fake.configure(
        [
            {
                "argv_contains": ["pr", "comment"],
                "stderr": "Error: raw-cli-stderr-marker",
                "exit": 1,
            }
        ]
    )

    result = run(["pr", "comment", "123", "-R", "o/r", "--body", secret_body], env=fake_env)
    failure = result.failure()
    message = str(failure)

    assert secret_body not in message  # body never leaks via message text
    assert "--body" not in message  # flags and their values are omitted
    assert "raw-cli-stderr-marker" not in message  # raw CLI output is omitted
    assert "[arguments omitted]" in message
    assert failure.argv[-2] == secret_body  # full argv stays in the structured attribute
    assert failure.stderr == "Error: raw-cli-stderr-marker"  # raw output kept for deliberate logging


def test_write_timeout_raises_once_with_reconciliation_context(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["pr", "comment"], "sleep": 2, "exit": 0}])

    with pytest.raises(CLITimeoutError) as excinfo:
        run(["pr", "comment", "123", "-R", "o/r", "--body", "reply"], env=fake_env, timeout=0.5)

    error = excinfo.value
    assert error.timeout == 0.5
    assert "reconcile" in str(error)
    assert error.argv[:3] == [str(fake.exe), "pr", "comment"]
    assert len(fake.invocations()) == 1  # uncertain write recorded once, never blindly retried


def test_timeout_message_sanitized_but_argv_preserved(fake: FakeCLI, fake_env):
    secret_body = "SECRET timeout body"
    fake.configure([{"argv_contains": ["pr", "comment"], "sleep": 2, "exit": 0}])

    with pytest.raises(CLITimeoutError) as excinfo:
        run(["pr", "comment", "123", "-R", "o/r", "--body", secret_body], env=fake_env, timeout=0.5)

    error = excinfo.value
    message = str(error)
    assert secret_body not in message
    assert "--body" not in message
    assert "[arguments omitted]" in message
    assert error.argv[-2] == secret_body  # full argv preserved structurally for reconciliation


# Pagination and completeness (task 2.3)


def _items(start: int, count: int) -> str:
    return json.dumps([{"id": i} for i in range(start, start + count)])


def test_paginate_collects_all_pages(fake: FakeCLI, fake_env):
    fake.configure(
        [
            {"argv_contains": ["--page", "1"], "stdout": _items(0, 100), "exit": 0},
            {"argv_contains": ["--page", "2"], "stdout": _items(100, 37), "exit": 0},
        ]
    )

    page = run_paginated(
        lambda n: ["issue", "list", "-R", "o/r", "--page", str(n), "--per-page", "100", "--json"],
        env=fake_env,
    )

    assert page.complete and page.stop_reason == "complete"
    assert len(page.items) == 137
    assert page.pages == 2
    assert len(fake.invocations()) == 2


def test_paginate_repeated_page_requires_independent_completeness_proof(fake: FakeCLI, fake_env):
    # A repeated first page can also conceal a capped/incomplete collection.
    fake.configure(
        [
            {"argv_contains": ["repos/o/r/pulls/7/files?page=1&per_page=3"], "stdout": _items(0, 7), "exit": 0},
            {"argv_contains": ["repos/o/r/pulls/7/files?page=2&per_page=3"], "stdout": _items(0, 7), "exit": 0},
        ]
    )

    page = run_paginated(
        lambda n: ["api", f"repos/o/r/pulls/7/files?page={n}&per_page=3"],
        per_page=3,
        max_pages=5,
        env=fake_env,
    )

    assert page.truncated and page.stop_reason == "repeated-page"
    assert len(page.items) == 7  # the repeated page extended nothing
    assert page.pages == 2
    assert len(fake.invocations()) == 2


def test_paginate_window_shift_partial_overlap_still_collects_new_items(fake: FakeCLI, fake_env):
    # A genuinely paging endpoint with window-shift duplicates (some old,
    # some new) keeps collecting: only a page with NO new items stops.
    fake.configure(
        [
            {"argv_contains": ["--page", "1"], "stdout": _items(0, 3), "exit": 0},
            {"argv_contains": ["--page", "2"], "stdout": _items(2, 3), "exit": 0},
            {"argv_contains": ["--page", "3"], "stdout": _items(5, 1), "exit": 0},
        ]
    )

    page = run_paginated(
        lambda n: ["issue", "list", "-R", "o/r", "--page", str(n), "--per-page", "3", "--json"],
        per_page=3,
        env=fake_env,
    )

    assert page.complete
    assert len(page.items) == 7  # overlapping second page plus a genuinely new final item
    assert page.pages == 3


def test_paginate_empty_collection_is_complete_not_unavailable(fake: FakeCLI, fake_env):
    fake.configure([{"argv_contains": ["--page", "1"], "stdout": "[]", "exit": 0}])

    page = run_paginated(lambda n: ["issue", "list", "-R", "o/r", "--page", str(n), "--json"], env=fake_env)

    assert page.complete and page.items == []
    assert page.pages == 1


def test_paginate_page_failure_preserves_partial_items_as_truncated(fake: FakeCLI, fake_env):
    fake.configure(
        [
            {"argv_contains": ["--page", "1"], "stdout": _items(0, 100), "exit": 0},
            {"argv_contains": ["--page", "2"], "stdout": "", "stderr": "Error: not found", "exit": 3},
        ]
    )

    page = run_paginated(
        lambda n: ["issue", "list", "-R", "o/r", "--page", str(n), "--per-page", "100", "--json"],
        env=fake_env,
    )

    assert page.truncated and page.stop_reason == "failed-page"
    assert len(page.items) == 100  # partial data kept; caller must not report "no comments"
    assert page.last_result.returncode == 3


def test_paginate_max_pages_guard_stops_runaway_loops(fake: FakeCLI, fake_env):
    fake.configure(
        [
            {"argv_contains": ["--page", "1"], "stdout": _items(0, 2), "exit": 0},
            {"argv_contains": ["--page", "2"], "stdout": _items(2, 2), "exit": 0},
        ]
    )

    page = run_paginated(
        lambda n: ["issue", "list", "-R", "o/r", "--page", str(n), "--per-page", "2", "--json"],
        per_page=2,
        max_pages=2,
        env=fake_env,
    )

    assert page.truncated and page.stop_reason == "max-pages"
    assert len(page.items) == 4
    assert len(fake.invocations()) == 2


# URL and host mapping (task 2.2)


def test_parse_url_pr_issue_and_branch_forms():
    pr = parse_atomgit_url("https://atomgit.com/openEuler/IB_Robot/pull/123")
    assert pr == {"owner": "openEuler", "repo": "IB_Robot", "host": "atomgit.com", "pr_number": 123}
    assert parse_atomgit_url("https://www.atomgit.com/o/r/pulls/9")["pr_number"] == 9
    assert parse_atomgit_url("https://atomgit.com/o/r/merge_requests/789")["pr_number"] == 789
    assert parse_atomgit_url("https://atomgit.com/o/r/issues/456")["issue_number"] == 456
    assert parse_atomgit_url("https://atomgit.com/o/r.git")["repo"] == "r"
    assert parse_atomgit_url("https://atomgit.com/o/r/tree/master/src")["branch"] == "master/src"


def test_parse_url_rejects_unregistered_service_hosts():
    with pytest.raises(AtomgitURLError, match="unsupported URL host"):
        parse_atomgit_url("https://gitcode.com/o/r/pull/1")
    with pytest.raises(AtomgitURLError, match="unsupported URL host"):
        parse_atomgit_url("https://github.com/o/r/pull/1")


def test_parse_url_rejects_missing_or_malformed_parts():
    with pytest.raises(AtomgitURLError, match="numeric PR number"):
        parse_atomgit_url("https://atomgit.com/o/r/pull/abc")
    with pytest.raises(AtomgitURLError, match="owner/repository"):
        parse_atomgit_url("https://atomgit.com/onlyowner")
    with pytest.raises(AtomgitURLError, match="empty URL"):
        parse_atomgit_url("")


# Local Git remote reading (task 2.2, upstream targeting tier)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "--quiet"], repo)
    return repo


def test_read_git_remotes_deduplicates_identical_fetch_push_urls(tmp_path: Path):
    repo = _init_repo(tmp_path)
    _git(["remote", "add", "origin", "https://atomgit.com/BreezeWu/IB_Robot.git"], repo)
    _git(["remote", "set-url", "--push", "origin", "https://atomgit.com/BreezeWu/IB_Robot.git"], repo)

    remotes = read_git_remotes(cwd=repo)

    assert remotes == {"origin": ["https://atomgit.com/BreezeWu/IB_Robot.git"]}  # fetch==push collapses


def test_read_git_remotes_keeps_distinct_fetch_push_urls_verbatim(tmp_path: Path):
    repo = _init_repo(tmp_path)
    _git(["remote", "add", "upstream", "git@atomgit.com:openEuler/IB_Robot.git"], repo)
    _git(["remote", "set-url", "--push", "upstream", "https://atomgit.com/openEuler/IB_Robot.git"], repo)

    remotes = read_git_remotes(cwd=repo)

    # URLs stay verbatim (ssh and https forms); interpretation belongs to resolve_target
    assert remotes == {
        "upstream": ["git@atomgit.com:openEuler/IB_Robot.git", "https://atomgit.com/openEuler/IB_Robot.git"]
    }


def test_resolve_target_consumes_read_git_remotes_upstream_fallback(tmp_path: Path):
    repo = _init_repo(tmp_path)
    _git(["remote", "add", "origin", "https://atomgit.com/BreezeWu/IB_Robot.git"], repo)
    _git(["remote", "add", "upstream", "git@atomgit.com:openEuler/IB_Robot.git"], repo)

    target = resolve_target(config_path=tmp_path / "absent.json", remotes=read_git_remotes(cwd=repo))

    assert target.slug == "openEuler/IB_Robot"  # upstream wins; origin fork is never the target
    assert target.origin == "upstream"


# Target resolution (task 2.2)


def test_resolve_target_explicit_overrides_config_defaults_without_conflict(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"atomgit": {"owner": "cfg-owner", "repo": "cfg-repo"}}), encoding="utf-8")

    target = resolve_target(owner="A", repo="B", config_path=config)

    assert target.slug == "A/B"  # explicit overrides config defaults; that is not a conflict
    assert target.origin == "explicit"


def test_resolve_target_rejects_explicit_vs_url_conflict(tmp_path: Path):
    with pytest.raises(TargetConflictError, match="conflicts with URL"):
        resolve_target(owner="o1", repo="r1", url="https://atomgit.com/o2/r2/pull/1")


def test_resolve_target_rejects_case_only_difference_as_unresolved_ambiguity():
    with pytest.raises(TargetConflictError, match="canonical identity"):
        resolve_target(owner="openeuler", url="https://atomgit.com/openEuler/IB_Robot/pull/1")


def test_resolve_target_partial_explicit_complements_from_url():
    target = resolve_target(repo="IB_Robot", url="https://atomgit.com/openEuler/IB_Robot/pull/5")

    assert target.slug == "openEuler/IB_Robot"
    assert target.origin == "explicit"


def test_resolve_target_config_without_token_key_is_sufficient(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"atomgit": {"owner": "openEuler", "repo": "IB_Robot"}}), encoding="utf-8")

    config_data = load_config(config)
    assert config_repo_defaults(config_data) == ("openEuler", "IB_Robot")  # token never validated (D2)


def _write_config(tmp_path: Path, atomgit: dict) -> Path:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"atomgit": atomgit}), encoding="utf-8")
    return config


def test_config_base_url_missing_null_or_empty_accepted(tmp_path: Path):
    no_key = _write_config(tmp_path, {"owner": "o", "repo": "r"})
    validate_config_base_url(load_config(no_key))

    null_url = _write_config(tmp_path, {"owner": "o", "repo": "r", "baseUrl": None})
    validate_config_base_url(load_config(null_url))

    empty_url = _write_config(tmp_path, {"owner": "o", "repo": "r", "baseUrl": "  "})
    validate_config_base_url(load_config(empty_url))


@pytest.mark.parametrize(
    "bad_base_url",
    [
        "https://api.gitcode.com",
        "https://github.com",
        "https://api.atomgit.com.evil.example.com",  # suffix attack: exact-host matching
        "api.gitcode.com",  # no scheme
    ],
)
def test_config_base_url_unregistered_host_rejected_before_request(tmp_path: Path, bad_base_url):
    config = _write_config(tmp_path, {"owner": "openEuler", "repo": "IB_Robot", "baseUrl": bad_base_url})

    with pytest.raises(AtomgitURLError, match="not a registered AtomGit service"):
        resolve_target(config_path=config)


def test_resolve_target_validates_base_url_even_when_explicit_target_wins(tmp_path: Path):
    config = _write_config(tmp_path, {"owner": "cfg-owner", "repo": "cfg-repo", "baseUrl": "https://api.gitcode.com"})

    with pytest.raises(AtomgitURLError, match="not a registered AtomGit service"):
        resolve_target(owner="openEuler", repo="IB_Robot", config_path=config)


def test_resolve_target_explicit_config_path_missing_errors(tmp_path: Path):
    with pytest.raises(TargetResolutionError, match="config file not found"):
        resolve_target(config_path=tmp_path / "nope.json", explicit_config=True)


def test_resolve_target_default_config_missing_falls_through_to_upstream(tmp_path: Path):
    remotes = {
        "upstream": ["git@atomgit.com:openEuler/IB_Robot.git"],
        "origin": ["https://atomgit.com/BreezeWu/IB_Robot.git"],
    }

    target = resolve_target(config_path=tmp_path / "absent.json", remotes=remotes)

    assert target.slug == "openEuler/IB_Robot"
    assert target.origin == "upstream"


def test_resolve_target_origin_is_never_the_default_target(tmp_path: Path):
    remotes = {"origin": ["https://atomgit.com/BreezeWu/IB_Robot.git"]}

    with pytest.raises(MissingTargetError, match="origin"):
        resolve_target(config_path=tmp_path / "absent.json", remotes=remotes)


def test_resolve_target_ambiguous_upstream_errors(tmp_path: Path):
    remotes = {"upstream": ["https://atomgit.com/a/b.git", "https://atomgit.com/c/d.git"]}

    with pytest.raises(TargetResolutionError, match="unambiguous"):
        resolve_target(config_path=tmp_path / "absent.json", remotes=remotes)


def test_resolve_target_upstream_unregistered_host_rejected(tmp_path: Path):
    remotes = {"upstream": ["https://github.com/o/r.git"]}

    with pytest.raises(AtomgitURLError, match="not a registered AtomGit web host"):
        resolve_target(config_path=tmp_path / "absent.json", remotes=remotes)


def test_resolve_target_partial_config_owner_without_repo_errors(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"atomgit": {"owner": "openEuler"}}), encoding="utf-8")

    with pytest.raises(TargetResolutionError, match="both owner and repo"):
        resolve_target(config_path=config)


def test_resolve_target_invalid_config_json_errors(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text("{not json", encoding="utf-8")

    with pytest.raises(TargetResolutionError, match="invalid JSON"):
        resolve_target(config_path=config)


def test_resolve_source_keeps_cross_fork_source_separate_from_target(tmp_path: Path):
    target = resolve_target(owner="openEuler", repo="IB_Robot", config_path=tmp_path / "absent.json")

    source = resolve_source("BreezeWu", target)

    assert source.slug == "BreezeWu/IB_Robot"
    assert source.origin == "source"
    assert target.slug == "openEuler/IB_Robot"  # target untouched for -R/--repo args


def test_resolve_source_rejects_same_owner_as_target():
    with pytest.raises(TargetConflictError, match="distinct source/target"):
        resolve_source("openEuler", RepoTarget("openEuler", "IB_Robot", "explicit"))


def test_resolve_source_rejects_empty_fork_owner():
    with pytest.raises(TargetResolutionError, match="non-empty"):
        resolve_source("  ", RepoTarget("openEuler", "IB_Robot", "explicit"))
