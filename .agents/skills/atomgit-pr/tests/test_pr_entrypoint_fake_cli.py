"""Entry-point behavior tests for the CLI-migrated PR skill (task 3.2).

These tests execute the real script entry points — ``atomgit-pr/scripts/
pr_creation.py`` and ``atomgit-pr/scripts/pr_management.py`` — as subprocesses
against the executable fake CLI (task 2.4), so the full path an Agent drives is
exercised for real: argument parsing → shared CLI boundary → execve
argv/stdin → exit-code classification → context/report output.

Scenario coverage required by task 3.2: cross-fork ``pr create`` argv shaping
(``--fork owner/repo --head branch``), the body via ``--body-file -`` stdin,
auth exit 4, conflict exit 5 without retry, WIP stage deferral, the
dual-Docker gate with the v2 fingerprint, and the written-but-validation-
pending state (the mutation outcome is preserved, never reported as "not
created"). The management side covers the fetch-info context JSON with the
v1/v2 fingerprint policy fields and placeholder-tainted platform inputs
blocking the update write before any platform mutation. D5 reconciliation
(create timeout → list open PRs and match by full identity plus body
equality) is exercised in-process because the boundary timeout is 120s — too
slow to trigger through the fake CLI's sleep. No real platform access, no
credentials.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = TESTS_DIR.parents[1]
SHARED_DIR = SKILLS_DIR / "_shared"
SHARED_TESTS_DIR = SHARED_DIR / "tests"
SCRIPTS_DIR = TESTS_DIR.parent / "scripts"

sys.path.insert(0, str(SHARED_TESTS_DIR))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import atomgit_cli  # noqa: E402
import pr_creation  # noqa: E402
import pr_management  # noqa: E402
from entrypoint_fixture import AUTH_LOGIN_GUIDANCE, EntrypointFakeCLI  # noqa: E402
from verification_gate import (  # noqa: E402
    compute_legacy_verification_inputs,
    compute_verification_inputs,
    format_verification_metadata,
)

PR_CREATION_SCRIPT = SCRIPTS_DIR / "pr_creation.py"
PR_MANAGEMENT_SCRIPT = SCRIPTS_DIR / "pr_management.py"
PR_TARGET_ARGS = ["--owner", "acme", "--repo", "widget"]
AGENT_TOOL_ARGS = [
    "--agent-tool",
    "OpenCode 1.17.20",
    "--ai-model",
    "gpt-5.6-sol",
    "--prompt-summary",
    "Implement the requested feature",
    "--third-party-materials",
    "无",
]
DISCLOSURE_START = "<!-- openEuler-ai-disclosure:start -->"
AI_COMMIT_MESSAGE = "feat: change things\n\nCo-Authored-By: gpt-5.6-sol"
# Page-1 api endpoints as they appear as a single argv element (argv_contains
# matches whole elements, and the boundary folds the query into the endpoint).
COMMITS_ENDPOINT = "repos/acme/widget/pulls/7/commits?page=1&per_page=100"
FILES_ENDPOINT = "repos/acme/widget/pulls/7/files?page=1&per_page=100"


@pytest.fixture
def fake(tmp_path: Path) -> EntrypointFakeCLI:
    return EntrypointFakeCLI(tmp_path)


def run_script(script: Path, args: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )


class GitRepo:
    """Local git repository with a bare ``origin`` remote (offline push/read)."""

    def __init__(self, root: Path) -> None:
        self.path = root / "repo"
        self.path.mkdir()
        self.git("init", "-b", "master")
        self.git("config", "user.name", "Test User")
        self.git("config", "user.email", "test@example.com")
        (self.path / "README.md").write_text("base\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-m", "chore: base")
        bare = root / "origin.git"
        self.git("init", "--bare", "-b", "master", str(bare))
        self.git("remote", "add", "origin", str(bare))
        self.git("push", "origin", "master")

    def git(self, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=self.path, capture_output=True, text=True)
        assert result.returncode == 0, f"git {args} failed: {result.stderr}"
        return result.stdout.strip()

    def commit_branch(self, branch: str, files: dict[str, str], message: str, *, push: bool) -> None:
        self.git("checkout", "-b", branch)
        for name, content in files.items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-m", message)
        if push:
            self.git("push", "origin", branch)

    def file_infos(self, branch: str, *filenames: str) -> list[dict]:
        infos = []
        for filename in filenames:
            patch = self.git("diff", "--unified=0", f"origin/master...{branch}", "--", filename)
            infos.append({"filename": filename, "patch": patch})
        return infos


def write_description(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pr_description.md"
    path.write_text(body, encoding="utf-8")
    return path


def create_args(
    *,
    branch: str,
    description_file: Path,
    title: str,
    extra: list[str] | None = None,
) -> list[str]:
    return [
        "--branch",
        branch,
        "--fork-owner",
        "acme-dev",
        "--title",
        title,
        "--description-file",
        str(description_file),
        *AGENT_TOOL_ARGS,
        "--human-reviewed",
        "-y",
        *PR_TARGET_ARGS,
        *(extra or []),
    ]


def management_cwd(tmp_path: Path) -> Path:
    workdir = tmp_path / "pr-cwd"
    workdir.mkdir()
    return workdir


# --------------------------------------------------------------------------- #
# pr_creation.py entry point
# --------------------------------------------------------------------------- #


def test_pr_create_success_maps_cross_fork_and_sends_body(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/simple", {"src/foo.py": "print('hello')\n"}, AI_COMMIT_MESSAGE, push=False)
    description = write_description(tmp_path, "## Summary\n\nAdd the feature.\n")
    fake.configure([{"argv_contains": ["pr", "create"], "stdout": '{"number": 9}', "exit": 0}])

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(branch="feat/simple", description_file=description, title="feat: change things"),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PR 创建成功" in result.stdout
    assert "PR 编号: #9" in result.stdout
    assert "https://atomgit.com/acme/widget/pull/9" in result.stdout
    assert fake.argv_of(["pr", "create"]) == [
        "pr",
        "create",
        "-R",
        "acme/widget",
        "--fork",
        "acme-dev/widget",
        "--head",
        "feat/simple",
        "--base",
        "master",
        "--title",
        "feat: change things",
        "--body-file",
        "-",
        "--json",
        "--no-interactive",
    ]
    stdin = fake.invocation_of(["pr", "create"])["stdin"]
    assert DISCLOSURE_START in stdin
    assert "## Summary" in stdin
    assert "Add the feature." in stdin


def test_pr_create_auth_exit4_prints_login_guidance(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/simple", {"src/foo.py": "print('hello')\n"}, AI_COMMIT_MESSAGE, push=False)
    description = write_description(tmp_path, "## Summary\n\nAdd the feature.\n")
    fake.configure([{"argv_contains": ["pr", "create"], "stderr": "not logged in to atomgit.com", "exit": 4}])

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(branch="feat/simple", description_file=description, title="feat: change things"),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 1
    assert AUTH_LOGIN_GUIDANCE in result.stdout
    assert len([argv for argv in fake.argvs() if argv[:2] == ["pr", "create"]]) == 1


def test_pr_create_conflict_exit5_reports_conflict_without_retry(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/simple", {"src/foo.py": "print('hello')\n"}, AI_COMMIT_MESSAGE, push=False)
    description = write_description(tmp_path, "## Summary\n\nAdd the feature.\n")
    fake.configure([{"argv_contains": ["pr", "create"], "stderr": "conflict", "exit": 5}])

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(branch="feat/simple", description_file=description, title="feat: change things"),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 1
    assert "平台状态冲突" in result.stdout
    assert "不要重复创建" in result.stdout
    assert len([argv for argv in fake.argvs() if argv[:2] == ["pr", "create"]]) == 1


def test_pr_create_wip_stage_defers_docker_gate(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/gated", {"scripts/setup.sh": "echo setup\n"}, AI_COMMIT_MESSAGE, push=False)
    description = write_description(tmp_path, "## Summary\n\nGated change.\n")
    fake.configure([{"argv_contains": ["pr", "create"], "stdout": '{"number": 10}', "exit": 0}])

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(
            branch="feat/gated",
            description_file=description,
            title="feat: change things",
            extra=["--pr-stage", "wip"],
        ),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "跳过双平台 Docker 验证" in result.stdout
    assert "[WIP] feat: change things" in fake.argv_of(["pr", "create"])


def test_pr_create_gated_review_validates_tree(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/gated", {"scripts/setup.sh": "echo setup-v2\n"}, AI_COMMIT_MESSAGE, push=True)
    file_infos = repo.file_infos("feat/gated", "scripts/setup.sh")
    inputs = compute_verification_inputs(file_infos)
    tree = repo.git("rev-parse", "feat/gated^{tree}")
    remote_head = repo.git("rev-parse", "feat/gated")
    description = write_description(
        tmp_path,
        "## Summary\n\nGated change.\n\n"
        + format_verification_metadata("full", inputs, tree, "ubuntu|openeuler")
        + "\n",
    )
    fake.configure(
        [
            {"argv_contains": ["pr", "create"], "stdout": '{"number": 11}', "exit": 0},
            {
                "argv_contains": ["pr", "view"],
                "stdout": json.dumps({"number": 11, "head": {"sha": remote_head}}),
                "exit": 0,
            },
        ]
    )

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(
            branch="feat/gated",
            description_file=description,
            title="feat: change things",
            extra=["--pr-stage", "review"],
        ),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PR 创建成功" in result.stdout
    assert fake.argv_of(["pr", "view"]) == ["pr", "view", "11", "-R", "acme/widget", "--json", "--no-interactive"]


def test_pr_create_post_write_validation_pending_preserves_mutation(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/gated", {"scripts/setup.sh": "echo setup-v2\n"}, AI_COMMIT_MESSAGE, push=True)
    file_infos = repo.file_infos("feat/gated", "scripts/setup.sh")
    inputs = compute_verification_inputs(file_infos)
    tree = repo.git("rev-parse", "feat/gated^{tree}")
    description = write_description(
        tmp_path,
        "## Summary\n\nGated change.\n\n"
        + format_verification_metadata("full", inputs, tree, "ubuntu|openeuler")
        + "\n",
    )
    # The platform already accepted the create, but the read-back reports a
    # different head SHA: the outcome must stay "created, validation pending".
    fake.configure(
        [
            {"argv_contains": ["pr", "create"], "stdout": '{"number": 12}', "exit": 0},
            {
                "argv_contains": ["pr", "view"],
                "stdout": json.dumps({"number": 12, "head": {"sha": "f" * 40}}),
                "exit": 0,
            },
        ]
    )

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(
            branch="feat/gated",
            description_file=description,
            title="feat: change things",
            extra=["--pr-stage", "review"],
        ),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 1
    assert "PR #12 已创建，但写后校验未完成" in result.stdout
    assert "https://atomgit.com/acme/widget/pull/12" in result.stdout
    assert "不要重复创建" in result.stdout
    assert len([argv for argv in fake.argvs() if argv[:2] == ["pr", "create"]]) == 1


def test_pr_create_dry_run_writes_nothing(fake, tmp_path):
    repo = GitRepo(tmp_path)
    repo.commit_branch("feat/simple", {"src/foo.py": "print('hello')\n"}, AI_COMMIT_MESSAGE, push=False)
    description = write_description(tmp_path, "## Summary\n\nDraft.\n")
    fake.configure([])

    result = run_script(
        PR_CREATION_SCRIPT,
        create_args(
            branch="feat/simple", description_file=description, title="feat: change things", extra=["--dry-run"]
        ),
        env=fake.env(),
        cwd=repo.path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run" in result.stdout
    assert fake.mutation_argv() == []
    assert fake.argvs() == [["version", "--no-interactive"]]


# --------------------------------------------------------------------------- #
# pr_management.py entry point
# --------------------------------------------------------------------------- #


def test_pr_fetch_info_writes_context(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    files = [{"filename": "docs/guide.md", "status": "modified", "additions": 1, "deletions": 0, "patch": "+text"}]
    commits = [{"sha": "a" * 40, "commit": {"author": {"name": "Dev"}, "message": "docs: update guide"}}]
    pr_view = {
        "number": 7,
        "title": "docs: update guide",
        "state": "open",
        "user": {"login": "dev"},
        "head": {"ref": "feat/docs", "sha": "b" * 40, "repo": {}},
        "base": {"ref": "master", "sha": "c" * 40, "repo": {}},
        "body": "old body",
    }
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view), "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": json.dumps(commits), "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(files), "exit": 0},
            {"argv_contains": ["pr", "comments"], "stdout": json.dumps([{"id": 1, "body": "nice"}]), "exit": 0},
        ]
    )
    out_dir = workdir / "out"

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        ["--fetch-info", "--pr", "7", "--output-dir", str(out_dir), *PR_TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    context = json.loads((out_dir / "widget_pr_7_context.json").read_text(encoding="utf-8"))
    assert context["metadata"]["title"] == "docs: update guide"
    assert context["metadata"]["branch"] == "feat/docs -> master"
    assert context["metadata"]["verification_inputs"] is None
    assert context["metadata"]["verification_policy"] is None
    assert context["comments"] == [{"id": 1, "body": "nice"}]
    assert context["changed_files"][0]["patch"] == "+text"
    assert "1 文件修改" in result.stdout
    assert fake.argv_of(["pr", "view"]) == ["pr", "view", "7", "-R", "acme/widget", "--json", "--no-interactive"]
    api_argv = [argv for argv in fake.argvs() if argv[:1] == ["api"]]
    assert len(api_argv) == 2
    assert any("pulls/7/commits?page=1&per_page=100" in argv[1] for argv in api_argv)
    assert any("pulls/7/files?page=1&per_page=100" in argv[1] for argv in api_argv)
    assert fake.argv_of(["pr", "comments"]) == [
        "pr",
        "comments",
        "7",
        "-R",
        "acme/widget",
        "--json",
        "--limit",
        "0",
        "--no-interactive",
    ]


def test_pr_fetch_info_reports_v1_policy_from_legacy_record(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    files = [
        {
            "filename": "scripts/setup.sh",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": "@@ -1 +1 @@\n-echo setup\n+echo setup-v2",
        }
    ]
    legacy_inputs = compute_legacy_verification_inputs(files)
    assert legacy_inputs != compute_verification_inputs(files)
    pr_view = {
        "number": 7,
        "title": "feat: change things",
        "state": "open",
        "user": {"login": "dev"},
        # CLI Repository objects expose the web URL under web_url; a non-https
        # value keeps the head-tree resolution offline and unresolvable.
        "head": {"ref": "feat/gated", "sha": "b" * 40, "repo": {"web_url": "file:///tmp/not-a-repo"}},
        "base": {"ref": "master", "sha": "c" * 40, "repo": {}},
        "body": format_verification_metadata("full", legacy_inputs, "d" * 40, "ubuntu|openeuler"),
    }
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view), "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": "[]", "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(files), "exit": 0},
            {"argv_contains": ["pr", "comments"], "stdout": "[]", "exit": 0},
        ]
    )
    out_dir = workdir / "out"

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        ["--fetch-info", "--pr", "7", "--output-dir", str(out_dir), *PR_TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    context = json.loads((out_dir / "widget_pr_7_context.json").read_text(encoding="utf-8"))
    metadata = context["metadata"]
    assert metadata["verification_policy"] == "1"
    assert metadata["legacy_verification_inputs"] == legacy_inputs
    assert metadata["verification"]["verified_inputs"] == legacy_inputs
    assert metadata["head_tree"] is None


def test_pr_fetch_info_comment_failure_writes_no_context(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    files = [{"filename": "docs/guide.md", "status": "modified", "additions": 1, "deletions": 0, "patch": "+text"}]
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": '{"number": 7, "title": "docs"}', "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": "[]", "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(files), "exit": 0},
            {"argv_contains": ["pr", "comments"], "stderr": "transient failure", "exit": 1},
        ]
    )
    out_dir = workdir / "out"

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        ["--fetch-info", "--pr", "7", "--output-dir", str(out_dir), *PR_TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 1
    assert "获取数据失败" in result.stdout
    assert not out_dir.exists()


def _update_pr_args(description_file: Path, extra: list[str] | None = None) -> list[str]:
    return [
        "--update-pr",
        str(description_file),
        "--pr",
        "7",
        *AGENT_TOOL_ARGS,
        *(extra or []),
        *PR_TARGET_ARGS,
    ]


def _configure_update_scenarios(fake, *, files: list[dict], pr_view: dict, commits: list[dict] | None = None) -> None:
    fake.configure(
        [
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": json.dumps(commits or []), "exit": 0},
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view), "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(files), "exit": 0},
            {"argv_contains": ["pr", "edit"], "stdout": '{"number": 7}', "exit": 0},
        ]
    )


def test_pr_update_pr_edits_via_temp_body_file(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    files = [{"filename": "src/foo.py", "status": "modified", "additions": 2, "deletions": 0, "patch": "+x"}]
    commits = [{"sha": "a" * 40, "commit": {"message": AI_COMMIT_MESSAGE}}]
    pr_view = {
        "number": 7,
        "title": "feat: change things",
        "body": "old body",
        "head": {"ref": "feat/x", "sha": "b" * 40, "repo": {}},
        "base": {"ref": "master", "repo": {}},
    }
    description_file = tmp_path / "description.json"
    description_file.write_text(
        json.dumps({"title": "feat: better title", "description": "## Summary\n\nUpdated.\n"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _configure_update_scenarios(fake, files=files, pr_view=pr_view, commits=commits)

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        _update_pr_args(description_file, extra=["--human-reviewed"]),
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PR #7 描述已更新" in result.stdout
    assert "https://atomgit.com/acme/widget/pull/7" in result.stdout
    edit_argv = fake.argv_of(["pr", "edit"])
    assert edit_argv[:7] == ["pr", "edit", "7", "-R", "acme/widget", "--title", "feat: better title"]
    body_path = edit_argv[edit_argv.index("--body-file") + 1]
    # pr edit has no stdin '-' mode (ReadTextFile): the body goes through a
    # temporary file that is cleaned up afterwards.
    assert body_path != "-"
    assert not Path(body_path).exists()
    assert edit_argv[-2:] == ["--json", "--no-interactive"]


def test_pr_update_pr_wip_stage_defers_gate(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    files = [
        {
            "filename": "scripts/setup.sh",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": "@@ -1 +1 @@\n-echo setup\n+echo setup-v2",
        }
    ]
    commits = [{"sha": "a" * 40, "commit": {"message": AI_COMMIT_MESSAGE}}]
    pr_view = {
        "number": 7,
        "title": "feat: change things",
        "body": "old",
        "head": {"ref": "feat/x", "repo": {}},
        "base": {"ref": "master"},
    }
    description_file = tmp_path / "description.json"
    description_file.write_text(
        json.dumps({"title": "feat: change things", "description": "## Summary\n\nUpdate.\n"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _configure_update_scenarios(fake, files=files, pr_view=pr_view, commits=commits)

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        _update_pr_args(description_file, extra=["--human-reviewed", "--pr-stage", "wip"]),
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "跳过双平台 Docker 验证" in result.stdout
    assert "[WIP] feat: change things" in fake.argv_of(["pr", "edit"])


def test_pr_update_pr_placeholder_patches_block_write(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    # A gate-relevant file returned without a patch: fingerprint inputs are
    # placeholder-tainted and the update must stop before any platform write.
    files = [{"filename": "scripts/setup.sh", "status": "modified", "additions": 1, "deletions": 1}]
    commits = [{"sha": "a" * 40, "commit": {"message": AI_COMMIT_MESSAGE}}]
    pr_view = {
        "number": 7,
        "title": "feat: change things",
        "body": "old",
        "head": {"ref": "feat/x", "repo": {}},
        "base": {"ref": "master"},
    }
    description_file = tmp_path / "description.json"
    description_file.write_text(
        json.dumps({"title": "feat: change things", "description": "## Summary\n\nUpdate.\n"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _configure_update_scenarios(fake, files=files, pr_view=pr_view, commits=commits)

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        _update_pr_args(description_file, extra=["--human-reviewed", "--pr-stage", "review"]),
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 1
    assert "更新失败" in result.stdout
    assert "patches are incomplete" in result.stdout
    assert all(argv[:2] != ["pr", "edit"] for argv in fake.argvs())


def test_pr_update_pr_dry_run_writes_nothing(fake, tmp_path):
    workdir = management_cwd(tmp_path)
    files = [{"filename": "src/foo.py", "status": "modified", "additions": 2, "deletions": 0, "patch": "+x"}]
    commits = [{"sha": "a" * 40, "commit": {"message": AI_COMMIT_MESSAGE}}]
    pr_view = {
        "number": 7,
        "title": "feat: change things",
        "body": "old",
        "head": {"ref": "feat/x", "repo": {}},
        "base": {"ref": "master"},
    }
    description_file = tmp_path / "description.json"
    description_file.write_text(
        json.dumps({"title": "feat: change things", "description": "## Summary\n\nDraft.\n"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _configure_update_scenarios(fake, files=files, pr_view=pr_view, commits=commits)

    result = run_script(
        PR_MANAGEMENT_SCRIPT,
        _update_pr_args(description_file, extra=["--dry-run"]),
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry Run" in result.stdout
    assert fake.mutation_argv() == []


# --------------------------------------------------------------------------- #
# D5 reconciliation (in-process: the boundary timeout is 120s, too slow to
# trigger through the fake CLI's sleep)
# --------------------------------------------------------------------------- #


def _cli_result(data) -> atomgit_cli.CLIResult:
    result = atomgit_cli.CLIResult(argv=["gitcode"], returncode=0, stdout=json.dumps(data), stderr="")
    result.data = data
    return result


def _patched_run_json(monkeypatch, *, create_error: Exception, pr_list: list[dict]) -> list[list[str]]:
    calls = []

    def fake_run_json(args, cli=None, stdin_text=None, **kwargs):
        calls.append(list(args))
        if args[:2] == ["pr", "create"]:
            raise create_error
        if args[:2] == ["pr", "list"]:
            return _cli_result(pr_list)
        raise AssertionError(f"unexpected CLI call: {args}")

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    return calls


def _d5_targets() -> tuple[atomgit_cli.RepoTarget, atomgit_cli.RepoTarget]:
    return (
        atomgit_cli.RepoTarget("acme", "widget", "explicit"),
        atomgit_cli.RepoTarget("acme-dev", "widget", "source"),
    )


def _open_pr(
    number: int,
    body: str,
    *,
    ref: str = "feat/x",
    base: str = "master",
    owner: str = "acme-dev",
    repo_name: str | None = "widget",
) -> dict:
    repo = {"owner": {"login": owner}, "name": repo_name} if repo_name else {}
    return {
        "number": number,
        "body": body,
        "head": {"ref": ref, "repo": repo},
        "base": {"ref": base},
    }


def test_create_timeout_reconciles_by_full_identity_and_body(monkeypatch, capsys):
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    calls = _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        # An older PR from the same branch must not be mistaken for this write:
        # same head/base/owner but a different body.
        pr_list=[_open_pr(8, "an older unrelated PR body"), _open_pr(9, body)],
    )

    number = pr_creation.create_pr_or_reconcile(
        cli="gitcode",
        target=target,
        source=source,
        branch="feat/x",
        base="master",
        description=body,
        create_argv=["pr", "create", "-R", "acme/widget"],
    )

    assert number == 9
    assert [
        "pr",
        "list",
        "-R",
        "acme/widget",
        "--state",
        "open",
        "--json",
        "--paginate",
        "--per-page",
        "100",
    ] in calls
    assert "对账确认" in capsys.readouterr().out


def test_create_timeout_without_body_match_is_uncertain(monkeypatch, capsys):
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=[_open_pr(8, "an older unrelated PR body")],
    )

    with pytest.raises(SystemExit) as excinfo:
        pr_creation.create_pr_or_reconcile(
            cli="gitcode",
            target=target,
            source=source,
            branch="feat/x",
            base="master",
            description=body,
            create_argv=["pr", "create", "-R", "acme/widget"],
        )
    assert excinfo.value.code == 1
    assert "uncertain" in capsys.readouterr().out


def test_create_timeout_multiple_matches_is_uncertain(monkeypatch, capsys):
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=[_open_pr(9, body), _open_pr(10, body)],
    )

    with pytest.raises(SystemExit) as excinfo:
        pr_creation.create_pr_or_reconcile(
            cli="gitcode",
            target=target,
            source=source,
            branch="feat/x",
            base="master",
            description=body,
            create_argv=["pr", "create", "-R", "acme/widget"],
        )
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "uncertain" in out
    assert "无法唯一确认" in out


def test_create_timeout_reconciles_excludes_same_owner_other_repo(monkeypatch, capsys):
    """Task 3.2 review: matching only the owner is not full identity — a
    different fork repository under the same owner with the same branch and
    body must be excluded instead of being mistaken for this write."""
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=[_open_pr(8, body, repo_name="other-repo"), _open_pr(9, body)],
    )

    number = pr_creation.create_pr_or_reconcile(
        cli="gitcode",
        target=target,
        source=source,
        branch="feat/x",
        base="master",
        description=body,
        create_argv=["pr", "create", "-R", "acme/widget"],
    )

    assert number == 9
    assert "对账确认" in capsys.readouterr().out


def test_create_timeout_reconciles_same_owner_other_repo_alone_is_uncertain(monkeypatch, capsys):
    """The same-owner other-repo entry alone cannot prove this write was
    accepted; the result stays uncertain instead of matching a foreign PR."""
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=[_open_pr(8, body, repo_name="other-repo")],
    )

    with pytest.raises(SystemExit) as excinfo:
        pr_creation.create_pr_or_reconcile(
            cli="gitcode",
            target=target,
            source=source,
            branch="feat/x",
            base="master",
            description=body,
            create_argv=["pr", "create", "-R", "acme/widget"],
        )
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "uncertain" in out
    assert "未找到可证明为本次写入" in out


def test_create_timeout_reconciles_malformed_entry_is_uncertain(monkeypatch, capsys):
    """A non-object entry could be this write; silently skipping it could
    turn a stale match into a false unique confirmation (task 3.2 review)."""
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=["not-a-dict", _open_pr(9, body)],
    )

    with pytest.raises(SystemExit) as excinfo:
        pr_creation.create_pr_or_reconcile(
            cli="gitcode",
            target=target,
            source=source,
            branch="feat/x",
            base="master",
            description=body,
            create_argv=["pr", "create", "-R", "acme/widget"],
        )
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "uncertain" in out
    assert "无法核实身份" in out


def test_create_timeout_reconciles_missing_head_identity_is_uncertain(monkeypatch, capsys):
    """head/base/body match but the entry cannot prove its head repository
    identity — it can be neither confirmed nor excluded as this write."""
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=[_open_pr(9, body, repo_name=None)],
    )

    with pytest.raises(SystemExit) as excinfo:
        pr_creation.create_pr_or_reconcile(
            cli="gitcode",
            target=target,
            source=source,
            branch="feat/x",
            base="master",
            description=body,
            create_argv=["pr", "create", "-R", "acme/widget"],
        )
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "uncertain" in out
    assert "owner/repo 身份" in out


def test_create_timeout_reconciles_case_ambiguous_identity_is_uncertain(monkeypatch, capsys):
    """design.md D2: a case-only identity difference proves neither same nor
    different repository, so the outcome stays uncertain instead of guessing."""
    target, source = _d5_targets()
    body = "## Summary\n\nThe created body.\n"
    _patched_run_json(
        monkeypatch,
        create_error=atomgit_cli.CLITimeoutError("timed out"),
        pr_list=[_open_pr(9, body, owner="Acme-Dev")],
    )

    with pytest.raises(SystemExit) as excinfo:
        pr_creation.create_pr_or_reconcile(
            cli="gitcode",
            target=target,
            source=source,
            branch="feat/x",
            base="master",
            description=body,
            create_argv=["pr", "create", "-R", "acme/widget"],
        )
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "uncertain" in out
    assert "大小写不一致" in out


# --------------------------------------------------------------------------- #
# CLI field-shape mapping (Repository.web_url → project html_url)
# --------------------------------------------------------------------------- #


def test_fetch_pr_record_maps_cli_web_url_to_html_url(monkeypatch):
    payload = {
        "number": 7,
        "head": {"ref": "feat/x", "sha": "a" * 40, "repo": {"web_url": "https://atomgit.com/acme-dev/widget"}},
        "base": {"ref": "master", "repo": {"web_url": "https://atomgit.com/acme/widget"}},
    }
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, cli=None, **kwargs: _cli_result(payload))

    pr = pr_management.fetch_pr_record("gitcode", atomgit_cli.RepoTarget("acme", "widget", "explicit"), 7)

    assert pr["head"]["repo"]["html_url"] == "https://atomgit.com/acme-dev/widget"
    assert pr["base"]["repo"]["html_url"] == "https://atomgit.com/acme/widget"
