import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "review_resolution.py"
SKILL_DOC = Path(__file__).parents[1] / "SKILL.md"
SPEC = importlib.util.spec_from_file_location("review_resolution", SCRIPT)
assert SPEC and SPEC.loader
review_resolution = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review_resolution
SPEC.loader.exec_module(review_resolution)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def commit_file(repo: Path, path: str, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, "--literal-pathspecs", "add", "--", path)
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-b", "feature")
    git(path, "config", "user.name", "Test User")
    git(path, "config", "user.email", "test@example.com")
    return path


def make_args(repo: Path, **overrides):
    values = {
        "work_dir": str(repo),
        "pr": 1,
        "fixup_target": None,
        "base_branch": None,
        "dry_run": False,
        "push": False,
        "ai_model": "test-model",
        "reply_mode": "threaded",
        "output_dir": str(repo.parent / "state"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_api(*comment_ids: int) -> Mock:
    api = Mock()
    api.client.config = SimpleNamespace(owner="test", repo="repo")
    api.get_pr_comments.return_value = [
        {
            "id": comment_id,
            "discussion_id": f"discussion-{comment_id}",
            "comment_type": "diff_comment",
            "path": "tracked.txt",
        }
        for comment_id in comment_ids
    ]
    api.reply_to_comment.return_value = {"id": 1000}
    api.get_pr_url.return_value = "https://atomgit.com/test/repo/pull/1"
    return api


def test_snapshot_restores_file_mode(tmp_path: Path):
    path = tmp_path / "tool.sh"
    path.write_bytes(b"#!/bin/sh\n")
    path.chmod(0o755)
    snapshots = {}

    review_resolution.snapshot_file(snapshots, str(path))
    path.write_bytes(b"broken\n")
    path.chmod(0o644)
    review_resolution.restore_snapshot(snapshots)

    assert path.read_bytes() == b"#!/bin/sh\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o755


def test_reply_only_batch_ignores_dirty_index(repo: Path):
    commit_file(repo, "tracked.txt", "base\n", "base")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    git(repo, "add", "staged.txt")
    api = make_api(7)

    status = review_resolution.execute_fix_batch(
        make_args(repo),
        api,
        [{"type": "reply_only", "comment_id": 7, "reply": "No code change needed."}],
    )

    assert status == review_resolution.STATUS_SUCCESS
    api.reply_to_comment.assert_called_once()
    assert git(repo, "diff", "--cached", "--name-only") == "staged.txt"


def test_code_fix_requires_explicit_target_before_pr_metadata(repo: Path):
    commit_file(repo, "tracked.txt", "base\n", "base")
    api = Mock()

    status = review_resolution.execute_fix_batch(
        make_args(repo),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 8,
                "file_path": "tracked.txt",
                "original_code": "base",
                "fixed_code": "fixed",
            }
        ],
    )

    assert status == review_resolution.STATUS_FAILED
    api.client.get_pull_request.assert_not_called()


def test_pr_metadata_base_must_be_head_ancestor(repo: Path):
    root_sha = commit_file(repo, "root.txt", "root\n", "root")
    feature_sha = commit_file(repo, "feature.txt", "feature\n", "feature")
    git(repo, "switch", "-c", "sibling", root_sha)
    sibling_sha = commit_file(repo, "sibling.txt", "sibling\n", "sibling")
    git(repo, "switch", "feature")
    api = Mock()
    api.client.get_pull_request.return_value = {
        "base": {"sha": sibling_sha},
        "head": {
            "sha": feature_sha,
            "ref": "feature",
            "repo": {"html_url": "https://atomgit.com/test/repo.git"},
        },
    }

    with pytest.raises(RuntimeError, match="is not an ancestor"):
        review_resolution.load_pr_git_context(str(repo), api, 1)


def test_pr_metadata_selects_exact_source_push_url_and_branch(repo: Path):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    head_sha = commit_file(repo, "feature.txt", "feature\n", "feature")
    git(repo, "remote", "add", "fork", "git@atomgit.com:source/repo.git")
    api = Mock()
    api.client.get_pull_request.return_value = {
        "base": {"sha": base_sha},
        "head": {
            "sha": head_sha,
            "ref": "feature/review-fix",
            "repo": {
                "full_name": "source/repo",
                "html_url": "https://atomgit.com/source/repo.git",
            },
        },
    }

    context = review_resolution.load_pr_git_context(str(repo), api, 1)

    assert context == review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=head_sha,
        head_branch="feature/review-fix",
        source_push_url="git@atomgit.com:source/repo.git",
        source_ref="refs/heads/feature/review-fix",
    )


def test_code_fix_rejects_existing_tracked_wip(repo: Path):
    commit_file(repo, "tracked.txt", "clean\n", "base")
    (repo / "tracked.txt").write_text("user wip\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tracked worktree already has unstaged changes"):
        review_resolution.ensure_clean_worktree(str(repo))


def test_fix_transaction_rolls_back_keyboard_interrupt_and_mode(repo: Path):
    base_sha = commit_file(repo, "first.sh", "#!/bin/sh\nold-one\n", "base")
    first_path = repo / "first.sh"
    first_path.write_text("#!/bin/sh\none\n", encoding="utf-8")
    first_path.chmod(0o755)
    git(repo, "add", "first.sh")
    git(repo, "commit", "-m", "first change")
    target_sha = git(repo, "rev-parse", "HEAD")
    commit_file(repo, "second.sh", "two\n", "second change")
    original_head = git(repo, "rev-parse", "HEAD")
    plan = review_resolution.PlannedFix(
        payload={
            "type": "code_fix",
            "comment_id": 9,
            "file_path": "first.sh",
            "original_code": "one",
            "fixed_code": "fixed",
        },
        fix_type="code_fix",
        rel_path="first.sh",
        abs_path=str(first_path),
        target_sha=target_sha,
    )

    with (
        pytest.raises(KeyboardInterrupt),
        review_resolution.FixTransaction(str(repo), base_sha, [plan]) as transaction,
    ):
        review_resolution.apply_planned_fix(plan, str(repo), base_sha)
        transaction.commit_group(target_sha, [plan])
        raise KeyboardInterrupt("stop")

    assert git(repo, "rev-parse", "HEAD") == original_head
    assert first_path.read_text(encoding="utf-8") == "#!/bin/sh\none\n"
    assert stat.S_IMODE(first_path.stat().st_mode) == 0o755
    assert git(repo, "status", "--porcelain") == ""


def test_sigterm_uses_transaction_rollback(repo: Path):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    target_sha = commit_file(repo, "target.txt", "old\n", "target")
    target_path = repo / "target.txt"
    plan = review_resolution.PlannedFix(
        payload={
            "type": "code_fix",
            "comment_id": 15,
            "file_path": "target.txt",
            "original_code": "old",
            "fixed_code": "fixed",
        },
        fix_type="code_fix",
        rel_path="target.txt",
        abs_path=str(target_path),
        target_sha=target_sha,
    )

    with (
        pytest.raises(review_resolution.WorkflowInterrupted),
        review_resolution.handle_termination_signals(),
        review_resolution.FixTransaction(str(repo), base_sha, [plan]) as transaction,
    ):
        review_resolution.apply_planned_fix(plan, str(repo), base_sha)
        transaction.commit_group(target_sha, [plan])
        os.kill(os.getpid(), signal.SIGTERM)

    assert target_path.read_text(encoding="utf-8") == "old\n"
    assert git(repo, "rev-parse", "HEAD") == target_sha
    assert git(repo, "status", "--porcelain") == ""


def test_multi_target_fixups_autosquash_once(repo: Path):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    first_sha = commit_file(repo, "first.txt", "one\n", "first change")
    second_sha = commit_file(repo, "second.txt", "two\n", "second change")
    plans = [
        review_resolution.PlannedFix(
            payload={
                "type": "code_fix",
                "comment_id": 10,
                "file_path": "first.txt",
                "original_code": "one",
                "fixed_code": "ONE",
            },
            fix_type="code_fix",
            rel_path="first.txt",
            abs_path=str(repo / "first.txt"),
            target_sha=first_sha,
        ),
        review_resolution.PlannedFix(
            payload={
                "type": "code_fix",
                "comment_id": 11,
                "file_path": "second.txt",
                "original_code": "two",
                "fixed_code": "TWO",
            },
            fix_type="code_fix",
            rel_path="second.txt",
            abs_path=str(repo / "second.txt"),
            target_sha=second_sha,
        ),
    ]

    with review_resolution.FixTransaction(str(repo), base_sha, plans) as transaction:
        for plan in plans:
            review_resolution.apply_planned_fix(plan, str(repo), base_sha)
            transaction.commit_group(plan.target_sha, [plan])
        transaction.autosquash()

    subjects = git(repo, "log", "--format=%s", f"{base_sha}..HEAD").splitlines()
    assert subjects == ["second change", "first change"]
    assert (repo / "first.txt").read_text(encoding="utf-8") == "ONE\n"
    assert (repo / "second.txt").read_text(encoding="utf-8") == "TWO\n"


def test_push_command_binds_source_ref_to_old_oid():
    context = review_resolution.PRGitContext(
        base_sha="a" * 40,
        head_sha="b" * 40,
        head_branch="feature/review-fix",
        source_push_url="git@atomgit.com:source/repo.git",
        source_ref="refs/heads/feature/review-fix",
    )

    assert review_resolution.push_command(context) == [
        "git",
        "push",
        f"--force-with-lease=refs/heads/feature/review-fix:{'b' * 40}",
        "git@atomgit.com:source/repo.git",
        "HEAD:refs/heads/feature/review-fix",
    ]


def test_code_replies_are_sent_only_after_verified_push(repo: Path, monkeypatch):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    target_sha = commit_file(repo, "target.txt", "old\n", "target")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=target_sha,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    events = []
    api = make_api(14)

    monkeypatch.setattr(review_resolution, "load_pr_git_context", lambda *_args: context)
    monkeypatch.setattr(review_resolution, "ensure_paths_tracked", lambda *_args: None)

    class FakeTransaction:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            events.append("transaction")
            return self

        def __exit__(self, *_args):
            return False

        def commit_group(self, *_args):
            pass

        def autosquash(self):
            pass

    monkeypatch.setattr(review_resolution, "FixTransaction", FakeTransaction)
    monkeypatch.setattr(review_resolution, "apply_planned_fix", lambda *_args: None)

    def verified_push(work_dir, push_context):
        assert push_context == context
        events.append("push_verified")
        return git(Path(work_dir), "rev-parse", "HEAD")

    def reply(*_args, **_kwargs):
        assert events == ["transaction", "push_verified"]
        events.append("reply")

    monkeypatch.setattr(review_resolution, "push_and_verify", verified_push)
    api.reply_to_comment.side_effect = reply
    status = review_resolution.execute_fix_batch(
        make_args(repo, push=True),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 14,
                "file_path": "target.txt",
                "fixup_target": target_sha,
                "original_code": "old",
                "fixed_code": "fixed",
            }
        ],
    )

    assert status == review_resolution.STATUS_SUCCESS
    assert events == ["transaction", "push_verified", "reply"]


@pytest.mark.parametrize("push", [False, True])
def test_code_replies_stay_pending_without_verified_push(repo: Path, monkeypatch, push: bool):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    target_sha = commit_file(repo, "target.txt", "old\n", "target")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=target_sha,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    api = make_api(16)

    monkeypatch.setattr(review_resolution, "load_pr_git_context", lambda *_args: context)
    monkeypatch.setattr(review_resolution, "ensure_paths_tracked", lambda *_args: None)

    class FakeTransaction:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def commit_group(self, *_args):
            pass

        def autosquash(self):
            pass

    monkeypatch.setattr(review_resolution, "FixTransaction", FakeTransaction)
    monkeypatch.setattr(review_resolution, "apply_planned_fix", lambda *_args: None)

    push_attempt = Mock(side_effect=RuntimeError("lease rejected"))
    monkeypatch.setattr(review_resolution, "push_and_verify", push_attempt)
    status = review_resolution.execute_fix_batch(
        make_args(repo, push=push),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 16,
                "file_path": "target.txt",
                "fixup_target": target_sha,
                "original_code": "old",
                "fixed_code": "fixed",
            }
        ],
    )

    assert status == review_resolution.STATUS_PENDING_PUSH
    api.reply_to_comment.assert_not_called()
    assert push_attempt.call_count == int(push)


def test_build_plan_rejects_one_file_across_targets(repo: Path):
    base_sha = commit_file(repo, "same.txt", "base\n", "base")
    first_sha = commit_file(repo, "same.txt", "one\n", "first")
    second_sha = commit_file(repo, "other.txt", "two\n", "second")
    fixes = [
        {
            "type": "code_fix",
            "comment_id": 12,
            "file_path": "same.txt",
            "original_code": "one",
            "fixed_code": "ONE",
            "fixup_target": first_sha,
        },
        {
            "type": "code_fix",
            "comment_id": 13,
            "file_path": "./same.txt",
            "original_code": "ONE",
            "fixed_code": "final",
            "fixup_target": second_sha,
        },
    ]

    with pytest.raises(ValueError, match="one file cannot be staged into multiple fixup targets"):
        review_resolution.build_fix_plan(fixes, str(repo), base_sha, None)


@pytest.mark.parametrize("content", ["abc\n", "abc abc\n"])
def test_apply_code_fix_rejects_empty_or_non_unique_match(tmp_path: Path, content: str):
    target = tmp_path / "target.txt"
    target.write_text(content, encoding="utf-8")
    original = "" if content == "abc\n" else "abc"

    with pytest.raises(ValueError, match="non-empty|exactly once"):
        review_resolution.apply_code_fix(str(target), original, "fixed")

    assert target.read_text(encoding="utf-8") == content


def test_build_plan_merges_same_target_delete_lines_from_one_snapshot(repo: Path):
    base_sha = commit_file(repo, "same.txt", "base\n", "base")
    target_sha = commit_file(repo, "same.txt", "one\ntwo\nthree\nfour\n", "target")
    fixes = [
        {
            "type": "delete_lines",
            "comment_id": 21,
            "file_path": "same.txt",
            "delete_lines": [2],
            "fixup_target": target_sha,
        },
        {
            "type": "delete_lines",
            "comment_id": 22,
            "file_path": "./same.txt",
            "delete_lines": [4, 2],
            "fixup_target": target_sha,
        },
    ]

    plans = review_resolution.build_fix_plan(fixes, str(repo), base_sha, None)

    assert len(plans) == 1
    assert plans[0].payload["delete_lines"] == [4, 2]
    assert [payload["comment_id"] for payload in plans[0].reply_payloads] == [21, 22]
    review_resolution.apply_planned_fix(plans[0], str(repo), base_sha)
    assert (repo / "same.txt").read_text(encoding="utf-8") == "one\nthree\n"


def test_build_plan_rejects_multiple_non_delete_fixes_for_one_file(repo: Path):
    base_sha = commit_file(repo, "same.txt", "base\n", "base")
    target_sha = commit_file(repo, "same.txt", "one\ntwo\n", "target")
    fixes = [
        {
            "type": "code_fix",
            "comment_id": 23,
            "file_path": "same.txt",
            "original_code": "one",
            "fixed_code": "ONE",
            "fixup_target": target_sha,
        },
        {
            "type": "delete_lines",
            "comment_id": 24,
            "file_path": "./same.txt",
            "delete_lines": [2],
            "fixup_target": target_sha,
        },
    ]

    with pytest.raises(ValueError, match="multiple fixes for one file are unsafe"):
        review_resolution.build_fix_plan(fixes, str(repo), base_sha, None)


def test_cli_help_does_not_expose_unsupported_auto_mode():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--auto" not in result.stdout
    assert "--resume" in result.stdout
    assert not hasattr(review_resolution, "mode_auto")


def test_skill_quick_start_references_existing_script():
    skill_doc = SKILL_DOC.read_text(encoding="utf-8")

    assert "repair_pr.py" not in skill_doc
    assert "review_resolution.py --url" in skill_doc


def test_autosquash_rejects_merge_commits_before_rebase(repo: Path, monkeypatch):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    commit_file(repo, "feature.txt", "feature\n", "feature")
    git(repo, "switch", "-c", "side", base_sha)
    commit_file(repo, "side.txt", "side\n", "side")
    git(repo, "switch", "feature")
    git(repo, "merge", "--no-ff", "side", "-m", "merge side")
    rebase = Mock()
    real_run = subprocess.run

    def record_rebase(command, *args, **kwargs):
        if command[:3] == ["git", "rebase", "-i"]:
            rebase(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(review_resolution.subprocess, "run", record_rebase)
    transaction = review_resolution.FixTransaction(str(repo), base_sha, [])

    with pytest.raises(RuntimeError, match="contains merge commits"):
        transaction.autosquash()

    rebase.assert_not_called()


def test_source_push_url_rejects_multiple_pushurls(repo: Path):
    git(repo, "remote", "add", "fork", "https://atomgit.com/source/repo.git")
    git(repo, "remote", "set-url", "--add", "--push", "fork", "git@atomgit.com:source/repo.git")
    git(repo, "remote", "set-url", "--add", "--push", "fork", "https://mirror.example/source/repo.git")

    with pytest.raises(RuntimeError, match="push URLs"):
        review_resolution._find_source_push_url(
            str(repo),
            {"full_name": "source/repo", "html_url": "https://atomgit.com/source/repo.git"},
        )


def test_source_push_url_rejects_multiple_matching_remotes(repo: Path):
    git(repo, "remote", "add", "fork-a", "git@atomgit.com:source/repo.git")
    git(repo, "remote", "add", "fork-b", "https://atomgit.com/source/repo.git")

    with pytest.raises(RuntimeError, match="multiple push targets"):
        review_resolution._find_source_push_url(
            str(repo),
            {"full_name": "source/repo", "html_url": "https://atomgit.com/source/repo.git"},
        )


def test_push_and_verify_uses_exact_push_url_for_both_commands(repo: Path, monkeypatch):
    commit_file(repo, "tracked.txt", "value\n", "value")
    new_head = git(repo, "rev-parse", "HEAD")
    context = review_resolution.PRGitContext(
        base_sha="a" * 40,
        head_sha="b" * 40,
        head_branch="feature",
        source_push_url="git@atomgit.com:source/repo.git",
        source_ref="refs/heads/feature",
    )
    commands = []
    monkeypatch.setattr(review_resolution, "_resolve_commit", lambda *_args: new_head)

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, f"{new_head}\trefs/heads/feature\n", "")

    monkeypatch.setattr(review_resolution.subprocess, "run", fake_run)

    assert review_resolution.push_and_verify(str(repo), context) == new_head
    assert commands[0][-2] == context.source_push_url
    assert commands[1][4] == context.source_push_url


def test_invalid_comment_id_is_rejected_before_transaction(repo: Path, monkeypatch):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    target_sha = commit_file(repo, "target.txt", "old\n", "target")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=target_sha,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    api = make_api(99)
    transaction = Mock(side_effect=AssertionError("transaction must not start"))
    monkeypatch.setattr(review_resolution, "load_pr_git_context", lambda *_args: context)
    monkeypatch.setattr(review_resolution, "FixTransaction", transaction)

    status = review_resolution.execute_fix_batch(
        make_args(repo),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 98,
                "file_path": "target.txt",
                "fixup_target": target_sha,
                "original_code": "old",
                "fixed_code": "fixed",
            }
        ],
    )

    assert status == review_resolution.STATUS_FAILED
    transaction.assert_not_called()
    assert git(repo, "rev-parse", "HEAD") == target_sha


def test_valid_comment_metadata_is_retained_for_reply_state(repo: Path):
    api = make_api(30)

    metadata = review_resolution.validate_pr_comment_ids(api, 1, [30])

    assert metadata["30"]["discussion_id"] == "discussion-30"
    assert metadata["30"]["comment_type"] == "diff_comment"


def test_comment_validation_falls_back_to_sdk_client_method():
    api = SimpleNamespace(
        client=SimpleNamespace(get_all_pr_comments=lambda pr_number: [{"id": pr_number, "discussion_id": "d"}])
    )

    metadata = review_resolution.validate_pr_comment_ids(api, 30, [30])

    assert metadata["30"]["discussion_id"] == "d"


def test_comment_validation_rejects_explicit_mismatched_pr_metadata():
    api = make_api()
    api.get_pr_comments.return_value = [{"id": 31, "pull_request_number": 2}]

    with pytest.raises(ValueError, match="do not belong to target PR"):
        review_resolution.validate_pr_comment_ids(api, 1, [31])


def test_literal_pathspec_prevents_magic_file_from_staging_other_changes(repo: Path):
    magic_path = ":(top)"
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    target_sha = commit_file(repo, magic_path, "old\n", "target")
    (repo / magic_path).write_text("fixed\n", encoding="utf-8")
    (repo / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    plan = review_resolution.PlannedFix(
        payload={"type": "code_fix", "comment_id": 31},
        fix_type="code_fix",
        rel_path=magic_path,
        abs_path=str(repo / magic_path),
        target_sha=target_sha,
    )

    review_resolution.ensure_paths_tracked(str(repo), [magic_path])
    transaction = review_resolution.FixTransaction(str(repo), base_sha, [plan])
    transaction.original_head = git(repo, "rev-parse", "HEAD")
    transaction.commit_group(target_sha, [plan])

    assert git(repo, "show", "--format=", "--name-only", "HEAD").strip() == magic_path
    assert "unrelated.txt" not in git(repo, "ls-files").splitlines()


def test_revert_file_treats_magic_filename_as_literal(repo: Path):
    magic_path = ":(top)"
    base_sha = commit_file(repo, magic_path, "base\n", "base")
    commit_file(repo, magic_path, "changed\n", "change magic")
    commit_file(repo, "other.txt", "keep\n", "add other")

    review_resolution.revert_file(str(repo / magic_path), str(repo), base_sha)

    assert (repo / magic_path).read_text(encoding="utf-8") == "base\n"
    assert (repo / "other.txt").read_text(encoding="utf-8") == "keep\n"


def test_pending_push_state_contains_resumable_git_and_reply_data(repo: Path, monkeypatch):
    base_sha = commit_file(repo, "base.txt", "base\n", "base")
    target_sha = commit_file(repo, "target.txt", "old\n", "target")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=target_sha,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    api = make_api(40)
    monkeypatch.setattr(review_resolution, "load_pr_git_context", lambda *_args: context)

    status = review_resolution.execute_fix_batch(
        make_args(repo),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 40,
                "file_path": "target.txt",
                "fixup_target": target_sha,
                "original_code": "old",
                "fixed_code": "fixed",
                "fix_description": "replace old",
            }
        ],
    )

    assert status == review_resolution.STATUS_PENDING_PUSH
    state_files = list((repo.parent / "state").glob("*.json"))
    assert len(state_files) == 1
    state = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert state["git"]["old_head"] == target_sha
    assert state["git"]["new_head"] == git(repo, "rev-parse", "HEAD")
    assert state["git"]["push_url"] == context.source_push_url
    assert state["git"]["source_ref"] == context.source_ref
    assert state["replies"][0]["status"] == "pending"
    assert state["replies"][0]["discussion_id"] == "discussion-40"
    assert state["replies"][0]["body"]
    assert state["reply_counts"] == {"pending": 1, "sent": 0, "total": 1}


def test_resume_push_accepts_old_local_head_and_pushes_recorded_new_oid(repo: Path, monkeypatch):
    old_head = commit_file(repo, "old.txt", "old\n", "old")
    new_head = commit_file(repo, "new.txt", "new\n", "new")
    git(repo, "reset", "--hard", old_head)
    state_path = str(repo.parent / "resume.json")
    state = {
        "work_dir": str(repo),
        "status": review_resolution.STATUS_PENDING_PUSH,
        "last_error": None,
        "replies": [],
        "git": {
            "old_head": old_head,
            "new_head": new_head,
            "push_url": "git@atomgit.com:test/repo.git",
            "source_ref": "refs/heads/feature",
            "push_status": "pending",
        },
    }
    remote_heads = iter([old_head, new_head])
    monkeypatch.setattr(review_resolution, "read_remote_ref", lambda *_args: next(remote_heads))
    real_resolve_commit = review_resolution._resolve_commit
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    assert real_resolve_commit(str(repo), "HEAD", "local HEAD") == old_head
    assert real_resolve_commit(str(repo), new_head, "rewritten HEAD") == new_head
    monkeypatch.setattr(
        review_resolution,
        "_resolve_commit",
        lambda _work_dir, ref, _description: old_head if ref == "HEAD" else new_head,
    )
    monkeypatch.setattr(review_resolution.subprocess, "run", fake_run)
    monkeypatch.setattr(review_resolution, "write_workflow_state", lambda *_args: None)

    review_resolution.resume_push(state_path, state)

    assert commands == [
        [
            "git",
            "push",
            f"--force-with-lease=refs/heads/feature:{old_head}",
            "git@atomgit.com:test/repo.git",
            f"{new_head}:refs/heads/feature",
        ]
    ]
    assert state["git"]["push_status"] == "verified"


def test_resume_push_revalidates_exact_remote_even_when_state_says_verified(repo: Path, monkeypatch):
    old_head = commit_file(repo, "old.txt", "old\n", "old")
    new_head = commit_file(repo, "new.txt", "new\n", "new")
    state = {
        "work_dir": str(repo),
        "status": review_resolution.STATUS_PENDING_REPLIES,
        "last_error": None,
        "replies": [],
        "git": {
            "old_head": old_head,
            "new_head": new_head,
            "push_url": "git@atomgit.com:test/repo.git",
            "source_ref": "refs/heads/feature",
            "push_status": "verified",
        },
    }
    read_remote = Mock(return_value=new_head)
    monkeypatch.setattr(review_resolution, "read_remote_ref", read_remote)
    monkeypatch.setattr(review_resolution, "write_workflow_state", lambda *_args: None)
    monkeypatch.setattr(
        review_resolution,
        "_resolve_commit",
        lambda _work_dir, ref, _description: new_head if ref in {"HEAD", new_head} else old_head,
    )
    push = Mock(side_effect=AssertionError("verified remote must not be pushed again"))
    monkeypatch.setattr(review_resolution.subprocess, "run", push)

    review_resolution.resume_push(str(repo.parent / "resume.json"), state)

    read_remote.assert_called_once_with(
        str(repo),
        "git@atomgit.com:test/repo.git",
        "refs/heads/feature",
    )
    push.assert_not_called()


def test_resume_push_failure_clears_previous_verified_status(repo: Path, monkeypatch):
    old_head = commit_file(repo, "old.txt", "old\n", "old")
    new_head = commit_file(repo, "new.txt", "new\n", "new")
    state = {
        "work_dir": str(repo),
        "status": review_resolution.STATUS_PENDING_REPLIES,
        "last_error": None,
        "replies": [],
        "git": {
            "old_head": old_head,
            "new_head": new_head,
            "push_url": "git@atomgit.com:test/repo.git",
            "source_ref": "refs/heads/feature",
            "push_status": "verified",
        },
    }
    monkeypatch.setattr(review_resolution, "read_remote_ref", Mock(return_value=old_head))
    monkeypatch.setattr(review_resolution, "write_workflow_state", lambda *_args: None)

    def fail_push(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, "", "lease rejected")

    monkeypatch.setattr(review_resolution.subprocess, "run", fail_push)
    monkeypatch.setattr(
        review_resolution,
        "_resolve_commit",
        lambda _work_dir, ref, _description: new_head if ref in {"HEAD", new_head} else old_head,
    )

    with pytest.raises(RuntimeError, match="resume push failed"):
        review_resolution.resume_push(str(repo.parent / "resume.json"), state)

    assert state["git"]["push_status"] == "pending"
    assert state["status"] == review_resolution.STATUS_PENDING_PUSH


def test_completed_code_state_is_revalidated_on_resume(repo: Path, monkeypatch):
    api = make_api(90)
    args = make_args(repo, resume=None)
    context = review_resolution.PRGitContext(
        base_sha="a" * 40,
        head_sha="b" * 40,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 90, "discussion_id": "d90", "body": "reply"}],
        context,
    )
    state["git"]["new_head"] = "c" * 40
    state["git"]["push_status"] = "verified"
    state["replies"][0]["status"] = "sent"
    state["status"] = review_resolution.STATUS_SUCCESS
    review_resolution.write_workflow_state(state_path, state)
    args.resume = state_path
    monkeypatch.setattr(review_resolution, "reconcile_state_replies", lambda *_args: None)
    monkeypatch.setattr(review_resolution, "validate_resume_pr_metadata", lambda *_args: None)
    resume_push = Mock()
    monkeypatch.setattr(review_resolution, "resume_push", resume_push)
    monkeypatch.setattr(review_resolution, "send_state_replies", lambda *_args: True)

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS
    resume_push.assert_called_once()


def test_send_state_replies_checkpoints_and_resume_only_sends_pending(repo: Path):
    api = make_api(50, 51)
    args = make_args(repo)
    specs = [
        {"comment_id": 50, "discussion_id": "d50", "body": "first"},
        {"comment_id": 51, "discussion_id": "d51", "body": "second"},
    ]
    state_path, state = review_resolution.create_workflow_state(args, api, specs, None)
    api.reply_to_comment.side_effect = [{"id": 500}, RuntimeError("network")]

    assert not review_resolution.send_state_replies(api, state_path, state)
    checkpoint = review_resolution.load_workflow_state(state_path)
    assert [entry["status"] for entry in checkpoint["replies"]] == ["sent", "pending"]
    assert checkpoint["replies"][0]["reply_id"] == 500
    api.reply_to_comment.reset_mock()
    api.reply_to_comment.side_effect = None
    api.reply_to_comment.return_value = {"id": 501}
    api.get_pr_comments.return_value = make_api(50, 51).get_pr_comments.return_value

    assert review_resolution.send_state_replies(api, state_path, checkpoint)
    api.reply_to_comment.assert_called_once()
    assert api.reply_to_comment.call_args.args[1] == 51
    completed = review_resolution.load_workflow_state(state_path)
    assert completed["status"] == review_resolution.STATUS_SUCCESS
    assert completed["reply_counts"] == {"pending": 0, "sent": 2, "total": 2}


def test_resume_reconciles_marker_without_resending_after_ambiguous_failure(repo: Path):
    api = make_api(60)
    args = make_args(repo)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 60, "discussion_id": "d60", "body": "reply"}],
        None,
    )
    marker_body = state["replies"][0]["body"]
    api.get_pr_comments.return_value = [{"id": 600, "body": marker_body, "discussion_id": "d60"}]

    assert review_resolution.send_state_replies(api, state_path, state)
    api.reply_to_comment.assert_not_called()
    completed = review_resolution.load_workflow_state(state_path)
    assert completed["replies"][0]["reply_id"] == 600


def test_ambiguous_reply_failure_is_reconciled_on_next_resume(repo: Path):
    api = make_api(61)
    args = make_args(repo)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 61, "discussion_id": "d61", "body": "reply"}],
        None,
    )
    api.reply_to_comment.side_effect = RuntimeError("timeout after server accepted reply")

    assert not review_resolution.send_state_replies(api, state_path, state)
    pending = review_resolution.load_workflow_state(state_path)
    assert pending["replies"][0]["status"] == "pending"
    api.reply_to_comment.reset_mock()
    api.reply_to_comment.side_effect = None
    api.get_pr_comments.return_value = [{"id": 610, "body": pending["replies"][0]["body"], "discussion_id": "d61"}]

    assert review_resolution.send_state_replies(api, state_path, pending)
    api.reply_to_comment.assert_not_called()


def test_mode_resume_reconciles_before_validating_pending_comment(repo: Path, monkeypatch):
    api = make_api(80)
    args = make_args(repo, resume=None)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 80, "discussion_id": "d80", "body": "reply"}],
        None,
    )
    args.resume = state_path
    api.get_pr_comments.return_value = [{"id": 800, "body": state["replies"][0]["body"], "discussion_id": "d80"}]
    validator = Mock(side_effect=AssertionError("no pending comment should be validated"))
    monkeypatch.setattr(review_resolution, "validate_pr_comment_ids", validator)

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS
    validator.assert_not_called()
    api.reply_to_comment.assert_not_called()


def test_load_state_rejects_pending_push_without_new_head(repo: Path):
    api = make_api(70)
    args = make_args(repo)
    context = review_resolution.PRGitContext(
        base_sha="a" * 40,
        head_sha="b" * 40,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 70, "discussion_id": "d70", "body": "reply"}],
        context,
    )
    state["status"] = review_resolution.STATUS_PENDING_PUSH
    review_resolution.write_workflow_state(state_path, state)

    with pytest.raises(ValueError, match="requires git.new_head"):
        review_resolution.load_workflow_state(state_path)
