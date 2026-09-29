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

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

import atomgit_cli

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
        "include_self_comments": False,
        "include_bot_comments": False,
        "output_dir": str(repo.parent / "state"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_api(*comment_ids: int) -> Mock:
    api = Mock()
    api.repository = {"owner": "test", "repo": "repo"}
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
    api.get_pr_record.assert_not_called()


def test_pr_metadata_base_must_be_head_ancestor(repo: Path):
    root_sha = commit_file(repo, "root.txt", "root\n", "root")
    feature_sha = commit_file(repo, "feature.txt", "feature\n", "feature")
    git(repo, "switch", "-c", "sibling", root_sha)
    sibling_sha = commit_file(repo, "sibling.txt", "sibling\n", "sibling")
    git(repo, "switch", "feature")
    api = Mock()
    api.get_pr_record.return_value = {
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
    api.get_pr_record.return_value = {
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
    assert state["reply_counts"] == {"pending": 1, "sent": 0, "total": 1, "uncertain": 0}


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
    assert completed["reply_counts"] == {"pending": 0, "sent": 2, "total": 2, "uncertain": 0}


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
    api.reply_to_comment.side_effect = review_resolution.UncertainReplyOutcomeError(
        "timeout after server accepted reply"
    )

    assert not review_resolution.send_state_replies(api, state_path, state)
    pending = review_resolution.load_workflow_state(state_path)
    assert pending["replies"][0]["status"] == "pending"
    # The attempted-but-unproven reply is flagged uncertain on disk.
    assert pending["replies"][0]["uncertain_outcome"] is True
    assert pending["reply_counts"]["uncertain"] == 1
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


# --------------------------------------------------------------------------- #
# RepairCli gateway (task 5.1): comment/discussion association, reply modes
# and resolve/unresolve argv shapes, with ownership re-proved from the
# PR-scoped comment list before every write. Platform access is faked by
# monkeypatching the shared boundary; no real CLI, no credentials.
# --------------------------------------------------------------------------- #

GATEWAY_TARGET = atomgit_cli.RepoTarget("test", "repo", "test", None)
MUTATION_PREFIXES = {("pr", "reply"), ("pr", "comment")}


class FakeBoundary:
    """Fake shared boundary recording every argv/stdin and serving controlled payloads."""

    def __init__(self):
        self.calls = []  # (kind, argv, kwargs) per invocation
        self.pr_comments: list[dict] = []
        self.pr_files: list[dict] = []
        self.pr_record: dict = {}
        self.user: dict = {"login": "alice"}
        self.failures: dict[tuple[str, ...], int] = {}  # argv prefix -> exit code

    def _exit_for(self, args: list[str]) -> int:
        for prefix, code in self.failures.items():
            if tuple(args[: len(prefix)]) == prefix:
                return code
        return 0

    def run(self, args, **kwargs):
        code = self._exit_for(args)
        self.calls.append(("run", list(args), kwargs))
        return atomgit_cli.CLIResult(
            argv=["gitcode", *args],
            returncode=code,
            stdout="",
            stderr="" if code == 0 else "boom",
        )

    def run_json(self, args, **kwargs):
        code = self._exit_for(args)
        self.calls.append(("run_json", list(args), kwargs))
        if args[:2] == ["user", "view"]:
            payload = self.user
        elif args[:2] == ["pr", "comments"]:
            payload = self.pr_comments
        elif args[:2] == ["pr", "view"]:
            payload = self.pr_record
        elif args[:1] == ["api"] and len(args) > 1 and "/files?" in args[1]:
            payload = self.pr_files
        else:
            payload = {"id": 123}
        return atomgit_cli.CLIResult(
            argv=["gitcode", *args],
            returncode=code,
            stdout=json.dumps(payload) if code == 0 else "",
            stderr="" if code == 0 else "boom",
            data=payload if code == 0 else None,
        )

    def install(self, monkeypatch):
        monkeypatch.setattr(atomgit_cli, "run", self.run)
        monkeypatch.setattr(atomgit_cli, "run_json", self.run_json)

    def mutation_calls(self):
        return [call for call in self.calls if tuple(call[1][:2]) in MUTATION_PREFIXES]


def make_gateway(monkeypatch, *, comments=None, files=None) -> tuple:
    boundary = FakeBoundary()
    if comments is not None:
        boundary.pr_comments = comments
    if files is not None:
        boundary.pr_files = files
    boundary.install(monkeypatch)
    return review_resolution.RepairCli("gitcode", GATEWAY_TARGET), boundary


def test_gateway_threaded_reply_targets_discussion_and_reports_no_reply_id(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "path": "a.py"}])

    result = gateway.reply_to_comment(7, 7, "fixed\nwith `code`")

    writes = boundary.mutation_calls()
    assert len(writes) == 1
    assert writes[0][0] == "run"
    assert writes[0][1] == [
        "pr",
        "reply",
        "7",
        "--discussion",
        "d7",
        "--body",
        "fixed\nwith `code`",
        "-R",
        "test/repo",
    ]
    assert result["reply_mode"] == "threaded"
    assert result["discussion_id"] == "d7"
    assert result.get("id") is None  # pr reply has no --json (matrix A11); resume reconciles via marker
    assert result["thread_url"].endswith("#discussion-d7")


def test_gateway_threaded_reply_on_outdated_thread_reports_visibility_hint(monkeypatch):
    gateway, _ = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "is_outdated": True}])

    result = gateway.reply_to_comment(7, 7, "fixed")

    assert result["visibility_hint"]


def test_gateway_threaded_reply_without_discussion_id_is_explicit_dead_end(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "path": "a.py"}])

    with pytest.raises(ValueError, match="reply-mode visible"):
        gateway.reply_to_comment(7, 7, "fixed")

    assert not boundary.mutation_calls()


def test_gateway_visible_reply_with_new_side_line_posts_inline_comment(monkeypatch):
    # New-side line 11 is visible in this patch (hunk starts at new line 10:
    # context line 10, added line 11); the anchor is re-proved against the
    # current PR head diff before the write.
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "path": "src/a.py", "new_line": 11}],
        files=[{"filename": "src/a.py", "patch": "@@ -10,1 +10,2 @@\n context\n+added line"}],
    )

    result = gateway.reply_to_comment(7, 7, "fixed", reply_mode="visible")

    writes = boundary.mutation_calls()
    assert len(writes) == 1
    assert writes[0][0] == "run_json"
    assert writes[0][1] == [
        "pr",
        "comment",
        "7",
        "-R",
        "test/repo",
        "--body-file",
        "-",
        "--path",
        "src/a.py",
        "--position",
        "11",
        "--json",
    ]
    assert "Reply to review comment #7" in writes[0][2]["stdin_text"]
    assert result["id"] == 123
    assert result["reply_mode"] == "visible"
    assert result["thread_url"].endswith("#discussion-d7")
    assert result["visibility_hint"]


def test_gateway_visible_reply_on_pr_level_comment_posts_top_level(monkeypatch):
    # A genuine PR-level original comment (no file/line/position anchoring at
    # all) is the only case that may degrade to a top-level visible reply.
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 8, "discussion_id": "d8", "body": "general remark"}],
    )

    result = gateway.reply_to_comment(7, 8, "answer", reply_mode="visible")

    writes = boundary.mutation_calls()
    assert writes[0][1] == ["pr", "comment", "7", "-R", "test/repo", "--body-file", "-", "--json"]
    assert result["reply_mode"] == "visible"
    assert result["visibility_hint"]


def test_gateway_visible_reply_int_position_does_not_crash_and_refuses_inline(monkeypatch):
    # CLI comment JSON may carry ``position`` as a bare int; it must never be
    # treated as a dict (AttributeError) nor used as a new-side line number.
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "position": 42}],
    )

    with pytest.raises(ValueError, match="cannot be located on the new side"):
        gateway.reply_to_comment(7, 7, "fixed", reply_mode="visible")

    assert not boundary.mutation_calls()


def test_gateway_visible_reply_old_side_only_is_unsupported_without_write(monkeypatch):
    # --position is the NEW-side line (pr-comment help evidence): an old-side
    # comment must not anchor the reply anywhere, and spec D4 forbids silently
    # degrading an inline comment to a PR-level reply.
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "path": "src/a.py", "old_line": 30}],
    )

    with pytest.raises(ValueError, match="NOT degraded to a PR-level comment"):
        gateway.reply_to_comment(7, 7, "fixed", reply_mode="visible")

    assert not boundary.mutation_calls()


def test_gateway_visible_reply_outdated_without_new_line_is_unsupported(monkeypatch):
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "path": "src/a.py", "old_line": 4, "is_outdated": True}],
    )

    with pytest.raises(ValueError, match="NOT degraded to a PR-level comment"):
        gateway.reply_to_comment(7, 7, "fixed", reply_mode="visible")

    assert not boundary.mutation_calls()


def test_gateway_visible_reply_line_outside_current_diff_is_unsupported(monkeypatch):
    # The comment's new_line exists but is not a new-side diff line in the
    # current PR head diff (comment outdated / line moved): zero writes.
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "path": "src/a.py", "new_line": 99, "is_outdated": True}],
        files=[{"filename": "src/a.py", "patch": "@@ -10,1 +10,2 @@\n context\n+added line"}],
    )

    with pytest.raises(ValueError, match="not visible on the new side of the current PR head diff"):
        gateway.reply_to_comment(7, 7, "fixed", reply_mode="visible")

    assert not boundary.mutation_calls()


def test_gateway_visible_reply_file_missing_from_current_diff_is_unsupported(monkeypatch):
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "path": "gone.py", "new_line": 3}],
        files=[{"filename": "other.py", "patch": "@@ -1,1 +1,2 @@\n a\n+b"}],
    )

    with pytest.raises(ValueError, match="not part of the current PR head diff"):
        gateway.reply_to_comment(7, 7, "fixed", reply_mode="visible")

    assert not boundary.mutation_calls()


def test_gateway_reply_rejects_comment_from_another_pr_without_write(monkeypatch):
    gateway, boundary = make_gateway(
        monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "pull_request_number": 99}]
    )

    with pytest.raises(ValueError, match="not found in PR 7"):
        gateway.reply_to_comment(7, 7, "fixed")

    assert not boundary.mutation_calls()


def test_gateway_resolve_comment_proves_ownership_then_targets_discussion(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "path": "a.py"}])

    result = gateway.resolve_comment(7, 7, resolved=True)

    writes = boundary.mutation_calls()
    assert len(writes) == 1
    assert writes[0][0] == "run"
    assert writes[0][1] == ["pr", "comment", "resolve", "7", "d7", "-R", "test/repo"]
    assert result == {"discussion_id": "d7", "resolved": True}


def test_gateway_resolve_comment_unresolve_uses_unresolve_action(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "path": "a.py"}])

    result = gateway.resolve_comment(7, 7, resolved=False)

    writes = boundary.mutation_calls()
    assert writes[0][1] == ["pr", "comment", "unresolve", "7", "d7", "-R", "test/repo"]
    assert result == {"discussion_id": "d7", "resolved": False}


def test_gateway_resolve_discussion_direct_argv(monkeypatch):
    # Even the direct --resolve-discussion path re-proves ownership from the
    # PR-scoped comment list: d9 must belong to PR 7 as an inline discussion.
    gateway, boundary = make_gateway(
        monkeypatch, comments=[{"id": 21, "discussion_id": "d9", "path": "src/a.py", "new_line": 3}]
    )

    result = gateway.resolve_discussion(7, "d9", resolved=False)

    writes = boundary.mutation_calls()
    assert writes[0][1] == ["pr", "comment", "unresolve", "7", "d9", "-R", "test/repo"]
    assert result == {"discussion_id": "d9", "resolved": False}


def test_gateway_resolve_discussion_unknown_id_is_refused_without_write(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "path": "a.py"}])

    with pytest.raises(ValueError, match="cannot be proved to belong to this PR"):
        gateway.resolve_discussion(7, "missing-discussion", True)

    assert not boundary.mutation_calls()


def test_gateway_resolve_discussion_cross_pr_comment_is_refused_without_write(monkeypatch):
    # A comment of another PR carrying the same discussion id is not PR-scoped
    # evidence: the direct resolve never reaches the platform.
    gateway, boundary = make_gateway(
        monkeypatch,
        comments=[{"id": 7, "discussion_id": "d7", "path": "a.py", "pull_request_number": 99}],
    )

    with pytest.raises(ValueError, match="cannot be proved to belong to this PR"):
        gateway.resolve_discussion(7, "d7", True)

    assert not boundary.mutation_calls()


def test_gateway_resolve_discussion_general_comment_thread_is_refused(monkeypatch):
    # A general PR comment thread (no file/line anchoring) is not a resolvable
    # inline discussion: explain instead of marking an unrelated discussion.
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "body": "general remark"}])

    with pytest.raises(ValueError, match="general PR comment thread"):
        gateway.resolve_discussion(7, "d7", True)

    assert not boundary.mutation_calls()


def test_gateway_resolve_comment_on_general_comment_is_refused(monkeypatch):
    # --resolve-comment routes through the same inline-discussion gate.
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "body": "general remark"}])

    with pytest.raises(ValueError, match="general PR comment thread"):
        gateway.resolve_comment(7, 7, True)

    assert not boundary.mutation_calls()


def test_gateway_resolve_comment_without_discussion_id_is_rejected(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7}])

    with pytest.raises(ValueError, match="does not include discussion_id"):
        gateway.resolve_comment(7, 7)

    assert not boundary.mutation_calls()


def test_gateway_resolve_failure_raises_and_never_retries(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "path": "a.py"}])
    boundary.failures[("pr", "comment", "resolve")] = 1

    with pytest.raises(atomgit_cli.CLIExecutionError):
        gateway.resolve_discussion(7, "d7", True)

    resolve_calls = [call for call in boundary.calls if call[1][:3] == ["pr", "comment", "resolve"]]
    assert len(resolve_calls) == 1


def test_gateway_current_user_login_read_and_cached(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch)
    boundary.user = {"login": "alice"}

    assert gateway.get_current_user_login() == "alice"
    assert gateway.get_current_user_login() == "alice"

    user_views = [call for call in boundary.calls if call[1][:2] == ["user", "view"]]
    assert len(user_views) == 1


def test_gateway_current_user_login_degrades_to_none_on_auth_failure(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch)
    boundary.failures[("user", "view")] = 4

    assert gateway.get_current_user_login() is None
    # The degradation is explicit, not silent: the filter status reports that
    # the requested self filtering could NOT be applied.
    status = gateway.self_filter_status(include_self_comments=False)
    assert status["self_filter"] == "unavailable"
    assert status["reason"]


def test_gateway_self_filter_status_applied_and_skipped(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch)

    applied = gateway.self_filter_status(include_self_comments=False)
    assert applied == {"self_filter": "applied", "current_user": "alice"}

    user_views = [call for call in boundary.calls if call[1][:2] == ["user", "view"]]
    assert len(user_views) == 1

    skipped = gateway.self_filter_status(include_self_comments=True)
    assert skipped["self_filter"] == "skipped"
    # include_self_comments=True never needs an additional identity read.
    assert len([call for call in boundary.calls if call[1][:2] == ["user", "view"]]) == 1


def test_gateway_review_threads_group_filter_and_annotate(monkeypatch):
    comments = [
        {
            "id": 1,
            "discussion_id": "d1",
            "body": "self note",
            "author": {"username": "alice"},
            "created_at": "2024-01-01T00:00:00",
        },
        {
            "id": 2,
            "discussion_id": "d2",
            "body": "issue",
            "author": {"login": "bob"},
            "created_at": "2024-01-02T00:00:00",
        },
        {
            "id": 3,
            "discussion_id": "d2",
            "body": "re",
            "author": {"username": "alice"},
            "created_at": "2024-01-03T00:00:00",
        },
        {
            "id": 4,
            "discussion_id": "d4",
            "body": "bot says hi",
            "author": {"username": "ci-bot"},
            "created_at": "2024-01-04T00:00:00",
        },
    ]
    gateway, _ = make_gateway(monkeypatch, comments=comments)

    threads = gateway.get_review_threads(7)

    # d1 is self-authored and d4 is bot-only: both filtered by default.
    assert [thread["discussion_id"] for thread in threads] == ["d2"]
    thread = threads[0]
    assert thread["id"] == 2
    assert thread["user"] == "bob"
    assert thread["thread_size"] == 2
    assert thread["reply_count"] == 1
    assert thread["participants"] == ["alice", "bob"]
    assert thread["latest_comment_id"] == 3
    assert thread["latest_comment_user"] == "alice"
    assert thread["url"].endswith("#discussion-d2")

    included = gateway.get_review_threads(7, include_self_comments=True, include_bot_comments=True)
    assert {thread["discussion_id"] for thread in included} == {"d1", "d2", "d4"}


# --------------------------------------------------------------------------- #
# Task 5.2 regression: the file transaction, fixup/autosquash, exception
# rollback, exact push URL/OID lease and push-verify-reply ordering all
# survived the CLI migration. These tests regress the mechanics against real
# Git state — a genuine mid-rebase autosquash conflict, and a real push with
# --force-with-lease against a local bare remote driven end-to-end through
# execute_fix_batch and the RepairCli gateway (only the shared CLI boundary
# is faked; no real platform, no credentials).
# --------------------------------------------------------------------------- #


def test_autosquash_conflict_aborts_rebase_and_restores_pre_transaction_state(repo: Path):
    base_sha = commit_file(repo, "conflict.txt", "line\n", "base")
    target_sha = commit_file(repo, "conflict.txt", "A\n", "introduce A")
    original_head = commit_file(repo, "conflict.txt", "B\n", "follow-up B")
    plan = review_resolution.PlannedFix(
        payload={
            "type": "code_fix",
            "comment_id": 17,
            "file_path": "conflict.txt",
            "original_code": "B",
            "fixed_code": "FIXED",
        },
        fix_type="code_fix",
        rel_path="conflict.txt",
        abs_path=str(repo / "conflict.txt"),
        target_sha=target_sha,
    )

    # The fixup patch was recorded against B's file state, so replaying it
    # directly after A (autosquash reorders it there) conflicts: the rebase
    # dies mid-way with .git/rebase-merge state left behind.
    with (
        pytest.raises(RuntimeError, match="autosquash failed"),
        review_resolution.FixTransaction(str(repo), base_sha, [plan]) as transaction,
    ):
        review_resolution.apply_planned_fix(plan, str(repo), base_sha)
        transaction.commit_group(target_sha, [plan])
        transaction.autosquash()

    assert review_resolution.rebase_in_progress(str(repo)) is False
    assert git(repo, "rev-parse", "HEAD") == original_head
    assert (repo / "conflict.txt").read_text(encoding="utf-8") == "B\n"
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "log", "--format=%s", f"{base_sha}..HEAD").splitlines() == ["follow-up B", "introduce A"]


def _work_repo_with_bare_remote(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-b", "feature")
    git(work, "config", "user.name", "Test User")
    git(work, "config", "user.email", "test@example.com")
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "feature", str(bare))
    return work, bare


def _install_real_gateway(monkeypatch, comments: list[dict]) -> tuple:
    boundary = FakeBoundary()
    boundary.pr_comments = comments
    boundary.install(monkeypatch)
    return review_resolution.RepairCli("gitcode", GATEWAY_TARGET), boundary


def test_verified_push_binds_real_oid_lease_then_replies_exactly_once(tmp_path: Path, monkeypatch):
    work, bare = _work_repo_with_bare_remote(tmp_path)
    base_sha = commit_file(work, "base.txt", "base\n", "base")
    old_head = commit_file(work, "target.txt", "old\n", "target")
    git(work, "push", str(bare), f"{old_head}:refs/heads/feature")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=old_head,
        head_branch="feature",
        source_push_url=str(bare),
        source_ref="refs/heads/feature",
    )
    monkeypatch.setattr(review_resolution, "load_pr_git_context", lambda *_args: context)
    api, boundary = _install_real_gateway(
        monkeypatch,
        [{"id": 41, "discussion_id": "discussion-41", "comment_type": "diff_comment", "path": "target.txt"}],
    )

    status = review_resolution.execute_fix_batch(
        make_args(work, push=True),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 41,
                "file_path": "target.txt",
                "fixup_target": old_head,
                "original_code": "old",
                "fixed_code": "fixed",
                "fix_description": "replace old",
            }
        ],
    )

    assert status == review_resolution.STATUS_SUCCESS
    new_head = git(work, "rev-parse", "HEAD")
    assert new_head != old_head
    # The lease accepted the push only because the remote still held the
    # pre-transaction OID, and verification re-read the exact push URL.
    assert git(work, "ls-remote", str(bare), "refs/heads/feature").startswith(new_head)
    # Exactly one platform write happened: the reply, after the verified push.
    writes = boundary.mutation_calls()
    assert len(writes) == 1
    reply_argv = writes[0][1]
    assert reply_argv[:2] == ["pr", "reply"]
    assert reply_argv[2] == "1"
    assert reply_argv[3:5] == ["--discussion", "discussion-41"]
    assert reply_argv[5] == "--body"
    reply_body = reply_argv[6]
    assert reply_argv[7:9] == ["-R", "test/repo"]
    assert "fixed" in reply_body
    assert "ai@test-model" in reply_body
    assert review_resolution.REPLY_MARKER_PREFIX in reply_body
    # The threaded CLI reply returns no reply id: the checkpoint keeps the
    # marker for resume reconciliation instead of a fabricated id.
    state_files = list((tmp_path / "state").glob("*.json"))
    assert len(state_files) == 1
    state = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert state["status"] == review_resolution.STATUS_SUCCESS
    assert state["git"]["old_head"] == old_head
    assert state["git"]["new_head"] == new_head
    assert state["git"]["push_status"] == "verified"
    assert state["replies"][0]["status"] == "sent"
    assert state["replies"][0]["reply_id"] is None
    assert state["reply_counts"] == {"pending": 0, "sent": 1, "total": 1, "uncertain": 0}


def test_stale_push_lease_keeps_remote_and_replies_untouched(tmp_path: Path, monkeypatch):
    work, bare = _work_repo_with_bare_remote(tmp_path)
    base_sha = commit_file(work, "base.txt", "base\n", "base")
    old_head = commit_file(work, "target.txt", "old\n", "target")
    git(work, "push", str(bare), f"{old_head}:refs/heads/feature")
    # A concurrent writer moves the remote ref after the PR context was read.
    git(work, "switch", "-c", "concurrent", base_sha)
    concurrent_sha = commit_file(work, "concurrent.txt", "concurrent\n", "concurrent")
    git(work, "push", "--force", str(bare), f"{concurrent_sha}:refs/heads/feature")
    git(work, "switch", "feature")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=old_head,
        head_branch="feature",
        source_push_url=str(bare),
        source_ref="refs/heads/feature",
    )
    monkeypatch.setattr(review_resolution, "load_pr_git_context", lambda *_args: context)
    api, boundary = _install_real_gateway(
        monkeypatch,
        [{"id": 42, "discussion_id": "discussion-42", "comment_type": "diff_comment", "path": "target.txt"}],
    )

    status = review_resolution.execute_fix_batch(
        make_args(work, push=True),
        api,
        [
            {
                "type": "code_fix",
                "comment_id": 42,
                "file_path": "target.txt",
                "fixup_target": old_head,
                "original_code": "old",
                "fixed_code": "fixed",
                "fix_description": "replace old",
            }
        ],
    )

    assert status == review_resolution.STATUS_PENDING_PUSH
    # The lease rejected the overwrite: the concurrent remote work survives.
    assert git(work, "ls-remote", str(bare), "refs/heads/feature").startswith(concurrent_sha)
    # No platform write happened — replies wait for the resume flow.
    assert boundary.mutation_calls() == []
    # The completed local autosquash is preserved for the resume push.
    assert (work / "target.txt").read_text(encoding="utf-8") == "fixed\n"
    state_files = list((tmp_path / "state").glob("*.json"))
    assert len(state_files) == 1
    state = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert state["status"] == review_resolution.STATUS_PENDING_PUSH
    assert state["git"]["push_status"] == "pending"
    assert state["git"]["old_head"] == old_head
    assert state["git"]["new_head"] == git(work, "rev-parse", "HEAD")
    assert state["replies"][0]["status"] == "pending"
    assert "explicit lease preserved" in state["last_error"]


@pytest.mark.parametrize("mode", ["unreadable", "mismatch"])
def test_push_and_verify_rejects_unverified_remote_before_any_reply(repo: Path, monkeypatch, mode: str):
    commit_file(repo, "tracked.txt", "value\n", "value")
    new_head = git(repo, "rev-parse", "HEAD")
    context = review_resolution.PRGitContext(
        base_sha="a" * 40,
        head_sha="b" * 40,
        head_branch="feature",
        source_push_url="git@atomgit.com:source/repo.git",
        source_ref="refs/heads/feature",
    )

    def fake_run(command, **_kwargs):
        if command[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if mode == "unreadable":
            return subprocess.CompletedProcess(command, 2, "", "remote ref unreadable")
        return subprocess.CompletedProcess(command, 0, f"{'c' * 40}\trefs/heads/feature\n", "")

    monkeypatch.setattr(review_resolution, "_resolve_commit", lambda *_args: new_head)
    monkeypatch.setattr(review_resolution.subprocess, "run", fake_run)

    if mode == "unreadable":
        with pytest.raises(RuntimeError, match="no review replies were sent"):
            review_resolution.push_and_verify(str(repo), context)
    else:
        with pytest.raises(RuntimeError, match="verification mismatch: expected"):
            review_resolution.push_and_verify(str(repo), context)


# --------------------------------------------------------------------------- #
# Task 5.3 regression: state schema/version compatibility gate. The state
# format is v1-only (unchanged by the CLI migration), the policy is explicit
# rejection — never a silent migration — and every incompatible state must be
# refused before any platform call, Git mutation or state-file rewrite.
# --------------------------------------------------------------------------- #


def _resume_args_with_state(repo: Path, monkeypatch, *, context=None, comments=None, reply_body="reply"):
    api, boundary = _install_real_gateway(
        monkeypatch,
        comments
        or [
            {
                "id": 44,
                "discussion_id": "discussion-44",
                "comment_type": "diff_comment",
                "path": "tracked.txt",
            }
        ],
    )
    args = make_args(repo)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 44, "discussion_id": "discussion-44", "body": reply_body}],
        context,
    )
    args.resume = state_path
    return args, api, boundary, state_path, state


def _tamper_state(state_path: str, mutate) -> bytes:
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    mutate(state)
    Path(state_path).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return Path(state_path).read_bytes()


@pytest.mark.parametrize(
    "tamper",
    [
        {"schema_version": 3},
        {"schema_version": "2"},
        {"schema_version": 1},
        {"schema_version": None},
    ],
    ids=["newer-version", "string-version", "legacy-sdk-v1", "missing-version"],
)
def test_resume_rejects_incompatible_schema_version_with_no_side_effects(repo: Path, monkeypatch, tamper: dict):
    args, api, boundary, state_path, _state = _resume_args_with_state(repo, monkeypatch)

    def mutate(state: dict) -> None:
        if tamper["schema_version"] is None:
            state.pop("schema_version")
        else:
            state["schema_version"] = tamper["schema_version"]

    tampered_bytes = _tamper_state(state_path, mutate)

    with pytest.raises(ValueError, match="unsupported resume state schema_version"):
        review_resolution.mode_resume(args, api)

    # Rejected before any platform read/write, Git mutation or state rewrite.
    assert boundary.calls == []
    assert Path(state_path).read_bytes() == tampered_bytes


@pytest.mark.parametrize(
    "tamper",
    [
        lambda state: state.pop("state_provenance"),
        lambda state: state.__setitem__("state_provenance", {"producer": "atomgit-sdk"}),
        lambda state: state.__setitem__("state_provenance", "atomgit-cli-review-resolution"),
    ],
    ids=["missing-provenance", "wrong-producer", "string-provenance"],
)
def test_resume_rejects_state_without_cli_provenance_no_side_effects(repo: Path, monkeypatch, tamper):
    args, api, boundary, state_path, _state = _resume_args_with_state(repo, monkeypatch)

    tampered_bytes = _tamper_state(state_path, tamper)

    with pytest.raises(ValueError, match="provenance"):
        review_resolution.mode_resume(args, api)

    assert boundary.calls == []
    assert Path(state_path).read_bytes() == tampered_bytes


def test_created_state_carries_schema_v2_and_cli_provenance(repo: Path):
    api = make_api(90)
    args = make_args(repo)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 90, "discussion_id": "d90", "body": "reply"}],
        None,
    )

    assert state["schema_version"] == 2
    assert state["state_provenance"] == {"producer": review_resolution.STATE_PRODUCER}
    assert all(entry["uncertain_outcome"] is False for entry in state["replies"])
    reloaded = review_resolution.load_workflow_state(state_path)
    assert reloaded["state_provenance"] == {"producer": review_resolution.STATE_PRODUCER}
    assert reloaded["reply_counts"]["uncertain"] == 0


def test_resume_rejects_repository_context_mismatch_without_side_effects(repo: Path, monkeypatch):
    args, api, boundary, state_path, _state = _resume_args_with_state(repo, monkeypatch)

    def mutate(state: dict) -> None:
        state["repository"] = {"owner": "someone-else", "repo": "repo"}

    tampered_bytes = _tamper_state(state_path, mutate)

    with pytest.raises(ValueError, match="does not match the configured CLI target"):
        review_resolution.mode_resume(args, api)

    assert boundary.calls == []
    assert Path(state_path).read_bytes() == tampered_bytes


@pytest.mark.parametrize("status", ["applying", review_resolution.STATUS_FAILED], ids=["applying", "failed"])
def test_resume_rejects_unresumable_status_without_side_effects(repo: Path, monkeypatch, status: str):
    args, api, boundary, state_path, _state = _resume_args_with_state(repo, monkeypatch)

    def mutate(state: dict) -> None:
        state["status"] = status

    tampered_bytes = _tamper_state(state_path, mutate)

    with pytest.raises(RuntimeError, match="not resumable"):
        review_resolution.mode_resume(args, api)

    assert boundary.calls == []
    assert Path(state_path).read_bytes() == tampered_bytes


def test_mode_resume_pending_push_pushes_recorded_lease_then_replies_once(tmp_path: Path, monkeypatch):
    work, _bare = _work_repo_with_bare_remote(tmp_path)
    base_sha = commit_file(work, "base.txt", "base\n", "base")
    old_head = commit_file(work, "target.txt", "old\n", "old")
    new_head = commit_file(work, "target.txt", "fixed\n", "fold review fix")
    context = review_resolution.PRGitContext(
        base_sha=base_sha,
        head_sha=old_head,
        head_branch="feature",
        source_push_url="git@atomgit.com:test/repo.git",
        source_ref="refs/heads/feature",
    )
    api, boundary = _install_real_gateway(
        monkeypatch,
        comments=[
            {
                "id": 43,
                "discussion_id": "discussion-43",
                "comment_type": "diff_comment",
                "path": "target.txt",
            }
        ],
    )
    boundary.pr_record = {
        "head": {
            "sha": new_head,
            "ref": "feature",
            "repo": {"full_name": "test/repo", "html_url": "https://atomgit.com/test/repo.git"},
        }
    }
    args = make_args(work, push=True)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 43, "discussion_id": "discussion-43", "body": "reply"}],
        context,
    )
    state["git"]["new_head"] = new_head
    state["status"] = review_resolution.STATUS_PENDING_PUSH
    review_resolution.write_workflow_state(state_path, state)
    args.resume = state_path

    remote_heads = iter([old_head, new_head])
    monkeypatch.setattr(review_resolution, "read_remote_ref", lambda *_args: next(remote_heads))
    commands = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[:2] == ["git", "push"]:
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr(review_resolution.subprocess, "run", fake_run)

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS

    # The lease still binds the pre-transaction OID recorded in the state.
    assert commands == [
        [
            "git",
            "push",
            f"--force-with-lease=refs/heads/feature:{old_head}",
            "git@atomgit.com:test/repo.git",
            f"{new_head}:refs/heads/feature",
        ]
    ]
    writes = boundary.mutation_calls()
    assert len(writes) == 1
    assert writes[0][1][:2] == ["pr", "reply"]
    assert writes[0][1][3:5] == ["--discussion", "discussion-43"]
    resumed = review_resolution.load_workflow_state(state_path)
    assert resumed["status"] == review_resolution.STATUS_SUCCESS
    assert resumed["git"]["push_status"] == "verified"
    assert resumed["replies"][0]["status"] == "sent"


def test_mode_resume_reply_only_state_sends_pending_reply_without_git(repo: Path, monkeypatch):
    api, boundary = _install_real_gateway(
        monkeypatch,
        comments=[
            {
                "id": 45,
                "discussion_id": "discussion-45",
                "comment_type": "diff_comment",
                "path": "tracked.txt",
            }
        ],
    )
    args = make_args(repo)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": 45, "discussion_id": "discussion-45", "body": "reply"}],
        None,
    )
    assert state["git"] is None
    assert state["status"] == review_resolution.STATUS_PENDING_REPLIES
    args.resume = state_path

    pushes = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[:2] == ["git", "push"]:
            pushes.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(review_resolution.subprocess, "run", fake_run)

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS

    # reply_only has no git section: nothing is pushed, only the reply fires.
    assert pushes == []
    writes = boundary.mutation_calls()
    assert len(writes) == 1
    assert writes[0][1][:2] == ["pr", "reply"]
    assert writes[0][1][3:5] == ["--discussion", "discussion-45"]
    resumed = review_resolution.load_workflow_state(state_path)
    assert resumed["git"] is None
    assert resumed["replies"][0]["status"] == "sent"
    assert resumed["status"] == review_resolution.STATUS_SUCCESS


# --------------------------------------------------------------------------- #
# Task 5.4 regression: reply-accepted-but-not-checkpointed faults. The
# threaded CLI reply never returns a reply id (capability matrix A11), so the
# only proof of acceptance is the marker/body-hash reconciliation against the
# PR comment list. Each test injects the fault at a different window —
# process death before the post-reply checkpoint, a CLI error raised after the
# platform already accepted the reply, and a checkpoint write failure — and
# proves the resume converges without ever re-sending the reply.
# --------------------------------------------------------------------------- #


def _reply_only_state_and_gateway(repo: Path, monkeypatch, *, comment_id=44):
    api, boundary = _install_real_gateway(
        monkeypatch,
        comments=[
            {
                "id": comment_id,
                "discussion_id": f"discussion-{comment_id}",
                "comment_type": "diff_comment",
                "path": "tracked.txt",
            }
        ],
    )
    args = make_args(repo)
    state_path, state = review_resolution.create_workflow_state(
        args,
        api,
        [{"comment_id": comment_id, "discussion_id": f"discussion-{comment_id}", "body": "reply"}],
        None,
    )
    args.resume = state_path
    return args, api, boundary, state_path, state


def _platform_accepted_reply(boundary, comment_id: int, body: str) -> None:
    """The platform now shows the accepted reply alongside the review comment."""
    boundary.pr_comments.append(
        {"id": 900, "discussion_id": f"discussion-{comment_id}", "body": body, "author": {"login": "alice"}}
    )


def test_resume_after_crash_before_reply_checkpoint_reconciles_without_resend(repo: Path, monkeypatch):
    args, api, boundary, state_path, state = _reply_only_state_and_gateway(repo, monkeypatch)

    # The process died right after the platform accepted the reply and before
    # any post-acceptance checkpoint: on disk only the pre-send checkpoint
    # (attempts=1, outcome flagged uncertain before the mutation ran) survives,
    # while the platform already holds the reply body.
    crashed = json.loads(Path(state_path).read_text(encoding="utf-8"))
    crashed["replies"][0]["attempts"] = 1
    crashed["replies"][0]["uncertain_outcome"] = True
    Path(state_path).write_text(json.dumps(crashed, ensure_ascii=False), encoding="utf-8")
    _platform_accepted_reply(boundary, comment_id=44, body=crashed["replies"][0]["body"])

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS

    # Reconciliation recovered the platform reply id and re-sent nothing.
    assert boundary.mutation_calls() == []
    resumed = review_resolution.load_workflow_state(state_path)
    assert resumed["status"] == review_resolution.STATUS_SUCCESS
    assert resumed["replies"][0]["status"] == "sent"
    assert resumed["replies"][0]["reply_id"] == 900


def test_reply_accepted_despite_cli_error_reconciles_on_resume_exactly_once(repo: Path, monkeypatch):
    args, api, boundary, state_path, state = _reply_only_state_and_gateway(repo, monkeypatch)
    # The CLI invocation is the fault point: the platform accepted the reply,
    # but the response was lost and the boundary surfaced a nonzero exit.
    boundary.failures[("pr", "reply")] = 1

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_PENDING_REPLIES
    assert len(boundary.mutation_calls()) == 1

    # The failed attempt actually landed: the platform comment list now
    # contains the exact reply body recorded in the pending state.
    pending = review_resolution.load_workflow_state(state_path)
    assert pending["replies"][0]["status"] == "pending"
    boundary.failures.clear()
    _platform_accepted_reply(boundary, comment_id=44, body=pending["replies"][0]["body"])

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS

    # Still exactly one reply attempt overall: the recovery proved the reply
    # is on the platform instead of blindly re-sending it.
    assert len(boundary.mutation_calls()) == 1
    resumed = review_resolution.load_workflow_state(state_path)
    assert resumed["replies"][0]["status"] == "sent"
    assert resumed["replies"][0]["reply_id"] == 900
    assert resumed["status"] == review_resolution.STATUS_SUCCESS


def test_checkpoint_failure_after_accepted_reply_converges_without_duplicate(repo: Path, monkeypatch):
    args, api, boundary, state_path, state = _reply_only_state_and_gateway(repo, monkeypatch)
    fault = {"armed": True}
    real_write = review_resolution.write_workflow_state

    def faulty_write(path, write_state, *args_, **kwargs_):
        # Fail the first checkpoint that runs after the reply mutation —
        # the write that would persist the accepted reply as sent.
        if fault["armed"] and boundary.mutation_calls():
            fault["armed"] = False
            raise OSError("checkpoint volume went read-only")
        return real_write(path, write_state, *args_, **kwargs_)

    monkeypatch.setattr(review_resolution, "write_workflow_state", faulty_write)

    # The reply was accepted (rc=0) but the checkpoint write failed; the
    # resume handler still records a safe non-duplicating state.
    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_PENDING_REPLIES
    assert len(boundary.mutation_calls()) == 1
    assert fault["armed"] is False

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS

    # No second reply was ever sent and the ledger converged.
    assert len(boundary.mutation_calls()) == 1
    resumed = review_resolution.load_workflow_state(state_path)
    assert resumed["status"] == review_resolution.STATUS_SUCCESS
    assert resumed["replies"][0]["status"] == "sent"
    # Threaded replies carry no CLI reply id (A11): the accepted state is the
    # authoritative record, exactly as documented for the marker policy.
    assert resumed["replies"][0]["reply_id"] is None


# --------------------------------------------------------------------------- #
# Task 5.4 gap regression: an attempted-but-uncertain reply must never be
# auto-resent just because one comment-list read shows no marker (the marker
# may be stripped, paginated away or not yet visible). Uncertain entries are
# only cleared by marker/body-hash reconciliation or by a human flipping the
# flag after verifying the PR page; auth failures provably accepted nothing
# and stay retryable.
# --------------------------------------------------------------------------- #


def test_uncertain_reply_without_marker_evidence_is_never_resent(repo: Path, monkeypatch):
    args, api, boundary, state_path, state = _reply_only_state_and_gateway(repo, monkeypatch)
    # The CLI invocation fails nonzero *after* the platform accepted the reply,
    # but the marker never becomes readable (stripped / paginated away / not
    # yet visible): one marker-less list read is not resend evidence.
    boundary.failures[("pr", "reply")] = 1

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_PENDING_REPLIES
    assert len(boundary.mutation_calls()) == 1

    boundary.failures.clear()
    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_PENDING_REPLIES

    # Still exactly one reply attempt overall: the uncertain entry was skipped.
    assert len(boundary.mutation_calls()) == 1
    pending = review_resolution.load_workflow_state(state_path)
    assert pending["replies"][0]["status"] == "pending"
    assert pending["replies"][0]["uncertain_outcome"] is True
    assert pending["reply_counts"]["uncertain"] == 1
    assert pending["last_error"]


def test_uncertain_flag_blocks_resend_even_after_manual_flag_reset_requires_edit(repo: Path, monkeypatch):
    args, api, boundary, state_path, state = _reply_only_state_and_gateway(repo, monkeypatch)
    boundary.failures[("pr", "reply")] = 1

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_PENDING_REPLIES

    # A human verified on the PR page that the reply never landed and flipped
    # the flag in the state file: the resend is explicitly re-authorized.
    boundary.failures.clear()
    cleared = json.loads(Path(state_path).read_text(encoding="utf-8"))
    cleared["replies"][0]["uncertain_outcome"] = False
    Path(state_path).write_text(json.dumps(cleared, ensure_ascii=False), encoding="utf-8")

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS
    assert len(boundary.mutation_calls()) == 2


def test_auth_failed_reply_is_retryable_after_login(repo: Path, monkeypatch):
    args, api, boundary, state_path, state = _reply_only_state_and_gateway(repo, monkeypatch)
    # Exit 4 (auth) provably accepted nothing: the reply stays retryable.
    boundary.failures[("pr", "reply")] = 4

    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_PENDING_REPLIES
    pending = review_resolution.load_workflow_state(state_path)
    assert pending["replies"][0]["status"] == "pending"
    assert pending["replies"][0]["uncertain_outcome"] is False

    boundary.failures.clear()
    assert review_resolution.mode_resume(args, api) == review_resolution.STATUS_SUCCESS
    assert len(boundary.mutation_calls()) == 2
    resumed = review_resolution.load_workflow_state(state_path)
    assert resumed["replies"][0]["status"] == "sent"
    assert resumed["replies"][0]["uncertain_outcome"] is False


def test_reply_timeout_maps_to_uncertain_outcome(monkeypatch):
    gateway, boundary = make_gateway(monkeypatch, comments=[{"id": 7, "discussion_id": "d7", "path": "a.py"}])

    def raise_timeout(*_args, **_kwargs):
        raise atomgit_cli.CLITimeoutError("timed out", argv=["gitcode", "pr", "reply"], timeout=1)

    monkeypatch.setattr(atomgit_cli, "run", raise_timeout)

    with pytest.raises(review_resolution.UncertainReplyOutcomeError, match="outcome is unknown"):
        gateway.reply_to_comment(7, 7, "fixed")

    assert not boundary.mutation_calls()


def test_fetch_comments_reports_unavailable_self_filter_in_metadata(repo: Path, capsys):
    api = make_api(30)
    api.get_review_threads.return_value = [
        {"id": 30, "discussion_id": "d30", "path": "a.py", "body": "issue", "url": "https://example"}
    ]
    api.self_filter_status.return_value = {
        "self_filter": "unavailable",
        "reason": "not authenticated or lacking permission",
    }
    args = make_args(repo)

    review_resolution.mode_fetch_comments(args, api)

    captured = capsys.readouterr()
    assert "未能应用 self 评论过滤" in captured.out
    output_file = Path(args.output_dir) / "repo_pr_1_unresolved_comments.json"
    metadata = json.loads(output_file.read_text(encoding="utf-8"))["metadata"]
    assert metadata["self_filter"]["self_filter"] == "unavailable"
    assert metadata["self_filter"]["reason"]


# --------------------------------------------------------------------------- #
# Remote-smoke fixtures (task 6.5, 2026-09-29): the REAL inline diff_comment
# shape from evidence/remote-smoke/25-pr-comments-after-submit.json — the
# platform readback anchors inline comments exclusively through a
# ``diff_position`` dict (start_new_line/end_new_line/position_type) and
# exposes NO file path (diff_file is always an empty string).
# --------------------------------------------------------------------------- #

REAL_DIFF_COMMENT = {
    "id": 192075138,
    "discussion_id": "cc0427b5c0ed9f02c4938956caec3ff35e3e5970",
    "comment_type": "diff_comment",
    "resolved": False,
    "diff_file": "",
    "diff_position": {"end_new_line": 8, "position_type": "text", "start_new_line": 8},
    "user": {"login": "BreezeWu"},
    "body": "inline positioning probe",
}


def test_real_diff_comment_shape_anchors_via_diff_position():
    anchor = review_resolution.comment_anchor_fields(REAL_DIFF_COMMENT)

    assert anchor["new_line"] == 8
    assert anchor["file_path"] is None  # platform readback exposes no path
    assert anchor["diff_position"] == REAL_DIFF_COMMENT["diff_position"]
    assert review_resolution.comment_is_inline_discussion(REAL_DIFF_COMMENT) is True


def test_gateway_resolves_real_inline_comment_with_diff_position_only(monkeypatch):
    # Real shape: no path/new_line fields, only the diff_position dict — the
    # resolve gate must recognize it as an inline discussion and write.
    gateway, boundary = make_gateway(monkeypatch, comments=[dict(REAL_DIFF_COMMENT)])

    result = gateway.resolve_comment(7, 192075138, resolved=True)

    writes = boundary.mutation_calls()
    assert writes[0][1] == [
        "pr",
        "comment",
        "resolve",
        "7",
        "cc0427b5c0ed9f02c4938956caec3ff35e3e5970",
        "-R",
        "test/repo",
    ]
    assert result["resolved"] is True


def test_gateway_visible_reply_on_pathless_inline_comment_is_unsupported(monkeypatch):
    # Without a file path the visible inline reply cannot be anchored; the
    # platform readback provides none, so this is an explicit zero-write
    # unsupported outcome (never a PR-level fallback, D4).
    gateway, boundary = make_gateway(monkeypatch, comments=[dict(REAL_DIFF_COMMENT)])

    with pytest.raises(ValueError, match="does not expose one for inline comments"):
        gateway.reply_to_comment(7, 192075138, "fixed", reply_mode="visible")

    assert not boundary.mutation_calls()


def test_thread_serialization_carries_line_for_real_inline_shape():
    thread = review_resolution.serialize_review_comment("https://atomgit.com/test/repo/pull/7", dict(REAL_DIFF_COMMENT))

    assert thread["line"] == 8
    assert thread["path"] is None  # documented platform readback limitation
    assert thread["comment_type"] == "diff_comment"
