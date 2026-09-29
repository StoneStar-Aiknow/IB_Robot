"""Behavior tests for the offline pr_body_gate entry point.

Each test builds a throwaway local Git repository (bare origin + clone, no
network) and runs ``pr_body_gate.py`` as a subprocess against it. The tests
cover this project's own gate behavior only — not the official CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

from pr_gates import compute_verification_inputs, format_verification_metadata

SCRIPT = Path(__file__).parents[1] / "scripts" / "pr_body_gate.py"
ENV_TEXT = "ubuntu:ros-humble-desktop-full-jammy|openeuler:ibrobot-dev-env"

AI_ARGS = [
    "--agent-tool",
    "OpenCode 1.17.20",
    "--ai-model",
    "glm-5.3",
    "--prompt-summary",
    "Implement the requested feature",
    "--third-party-materials",
    "无",
]


class GateRepo:
    """Local-only repository: a bare origin plus a working clone."""

    def __init__(self, tmp_path: Path):
        self.origin = tmp_path / "origin.git"
        self.work = tmp_path / "work"
        self._git(["-c", "init.defaultBranch=master", "init", "--bare", str(self.origin)], cwd=tmp_path)
        self._git(["-c", "init.defaultBranch=master", "clone", str(self.origin), str(self.work)], cwd=tmp_path)
        self._git(["symbolic-ref", "HEAD", "refs/heads/master"])
        self._git(["config", "user.name", "Tester"])
        self._git(["config", "user.email", "tester@example.com"])

    def _git(self, args: list[str], cwd: Path | None = None) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.work),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"git {args}: {result.stderr}"
        return result.stdout.strip()

    def git(self, args: list[str]) -> str:
        return self._git(args)

    def commit(self, message: str, files: dict[str, str], branch: str | None = None) -> str:
        if branch:
            self._git(["checkout", "-B", branch])
        for path, content in files.items():
            target = self.work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self._git(["add", "--", path])
        self._git(["commit", "-m", message])
        return self._git(["rev-parse", "HEAD"])

    def push(self, branch: str) -> None:
        self._git(["push", "-u", "origin", branch])

    def tree(self, ref: str) -> str:
        return self._git(["rev-parse", f"{ref}^{{tree}}"]).lower()


@pytest.fixture
def repo(tmp_path: Path) -> GateRepo:
    gate_repo = GateRepo(tmp_path)
    gate_repo.commit("chore: initialize repository\n\nBase commit for gate tests.", {"README.md": "base\n"})
    gate_repo.push("master")
    gate_repo.commit(
        "feat: add feature module\n\nAdd a small feature for gate testing.\n\nCo-Authored-By: glm-5.3",
        {"src/feature.py": "VALUE = 1\n"},
        branch="feat/change",
    )
    gate_repo.push("feat/change")
    return gate_repo


def run_gate(
    repo: GateRepo,
    draft: str,
    *extra: str,
    tmp_path: Path,
    branch: str = "feat/change",
    publication_authorized: bool = True,
) -> subprocess.CompletedProcess:
    draft_file = tmp_path / "draft.md"
    draft_file.write_text(draft, encoding="utf-8")
    output_file = tmp_path / "pr_body.md"
    argv = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(repo.work),
        "--branch",
        branch,
        "--description-file",
        str(draft_file),
        "--output",
        str(output_file),
        *AI_ARGS,
    ]
    if publication_authorized:
        argv.append("--publication-authorized")
    return subprocess.run(
        [*argv, *extra],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_plain_pr_passes_and_injects_disclosure(repo, tmp_path):
    result = run_gate(repo, "## 方案\n\n新增 feature 模块。", tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    body = (tmp_path / "pr_body.md").read_text(encoding="utf-8")
    assert body.startswith("<!-- openEuler-ai-disclosure:start -->")
    assert "Agent平台信息（Tool）: OpenCode 1.17.20" in body
    assert "模型信息 (Model): glm-5.3" in body
    assert "待开发者完成逐项实质审查" in body
    assert "已逐项审查" not in body
    assert "## 方案" in body
    assert "✓ 标题: feat: add feature module" in result.stdout


def test_missing_publication_authorized_refuses(repo, tmp_path):
    result = run_gate(repo, "## 方案\n\n正文。", tmp_path=tmp_path, publication_authorized=False)
    assert result.returncode == 1
    assert "--publication-authorized" in result.stdout


def test_handwritten_disclosure_rejected(repo, tmp_path):
    draft = "### 当前PR是否有AI参与:\n- [x] 是\n\n手写披露，应当被拒绝。"
    result = run_gate(repo, draft, tmp_path=tmp_path)
    assert result.returncode == 1
    assert "hand-written AI disclosure" in result.stdout


def test_model_coverage_enforced_over_all_commits(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    repo.commit(
        "feat: second module\n\nAnother change.\n\nCo-Authored-By: gpt-9.1",
        {"src/second.py": "VALUE = 2\n"},
    )
    repo.push("feat/change")
    result = run_gate(repo, "## 方案\n\n两个模块。", tmp_path=tmp_path)
    assert result.returncode == 1
    assert "gpt-9.1" in result.stdout
    assert "does not include commit models" in result.stdout


def test_wip_stage_defers_docker_gate(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    repo.commit(
        "feat: touch setup\n\nGate-relevant change.\n\nCo-Authored-By: glm-5.3", {"scripts/setup.sh": "set -e\n"}
    )
    repo.push("feat/change")
    result = run_gate(repo, "## 方案\n\n改 setup。", "--pr-stage", "wip", tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    # git log lists the newest commit first, so the default title comes from
    # the branch's latest commit subject.
    assert "✓ 标题: [WIP] feat: touch setup" in result.stdout
    assert "跳过双平台 Docker 验证" in result.stdout


def test_review_stage_requires_docker_block_and_prints_fingerprint(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    repo.commit(
        "feat: touch setup\n\nGate-relevant change.\n\nCo-Authored-By: glm-5.3", {"scripts/setup.sh": "set -e\n"}
    )
    repo.push("feat/change")
    result = run_gate(repo, "## 方案\n\n改 setup。", "--pr-stage", "review", tmp_path=tmp_path)
    assert result.returncode == 1
    expected = compute_verification_inputs([{"filename": "scripts/setup.sh", "patch": "+set -e"}])
    assert expected is not None
    assert expected in result.stdout


def test_review_stage_accepts_matching_full_block(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    repo.commit(
        "feat: touch setup\n\nGate-relevant change.\n\nCo-Authored-By: glm-5.3", {"scripts/setup.sh": "set -e\n"}
    )
    repo.push("feat/change")
    inputs = compute_verification_inputs([{"filename": "scripts/setup.sh", "patch": "+set -e"}])
    block = format_verification_metadata("full", inputs, repo.tree("feat/change"), ENV_TEXT)
    draft = f"## 方案\n\n改 setup。\n\n{block}\n"
    result = run_gate(repo, draft, "--pr-stage", "review", tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Docker 验证块: full" in result.stdout


def test_update_reuses_previous_evidence(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    repo.commit(
        "feat: touch setup\n\nGate-relevant change.\n\nCo-Authored-By: glm-5.3", {"scripts/setup.sh": "set -e\n"}
    )
    repo.push("feat/change")
    inputs = compute_verification_inputs([{"filename": "scripts/setup.sh", "patch": "+set -e"}])
    previous_tree = "e" * 40
    previous_body = "旧 PR 描述\n\n" + format_verification_metadata("full", inputs, previous_tree, ENV_TEXT) + "\n"
    previous_file = tmp_path / "previous.md"
    previous_file.write_text(previous_body, encoding="utf-8")
    result = run_gate(
        repo,
        "## 方案\n\n改 setup（更新）。",
        "--pr-stage",
        "review",
        "--previous-body-file",
        str(previous_file),
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    body = (tmp_path / "pr_body.md").read_text(encoding="utf-8")
    assert "reused-environment" in body
    assert "Docker 验证块: reused-environment" in result.stdout


def test_unpushed_local_head_refuses(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    repo.commit("feat: unpushed\n\nNot pushed yet.\n\nCo-Authored-By: glm-5.3", {"scripts/setup.sh": "set -e\n"})
    result = run_gate(repo, "## 方案\n\n改 setup。", "--pr-stage", "review", tmp_path=tmp_path)
    assert result.returncode == 1
    assert "does not match origin/feat/change" in result.stdout


def test_missing_tracking_ref_requires_fetch(tmp_path):
    gate_repo = GateRepo(tmp_path)
    gate_repo.commit("chore: initialize repository\n\nBase commit.", {"README.md": "base\n"})
    gate_repo.push("master")
    gate_repo.commit(
        "feat: never pushed\n\nLocal-only branch with a gated file.\n\nCo-Authored-By: glm-5.3",
        {"scripts/setup.sh": "set -e\n"},
        branch="feat/local-only",
    )
    result = run_gate(
        gate_repo,
        "## 方案\n\n本地分支。",
        "--pr-stage",
        "review",
        tmp_path=tmp_path,
        branch="feat/local-only",
    )
    assert result.returncode == 1
    assert "git fetch origin" in result.stdout


def test_explicit_missing_base_ref_errors(repo, tmp_path):
    result = run_gate(repo, "## 方案\n\n正文。", "--base", "upstream/master", tmp_path=tmp_path)
    assert result.returncode == 1
    assert "git rev-parse --verify upstream/master failed" in result.stdout


def test_large_change_requires_reuse_self_check(repo, tmp_path):
    repo.git(["checkout", "feat/change"])
    big_file = "\n".join(f"LINE_{i} = {i}" for i in range(1, 2002)) + "\n"
    repo.commit("feat: large module\n\nBig change.\n\nCo-Authored-By: glm-5.3", {"src/big.py": big_file})
    repo.push("feat/change")
    result = run_gate(repo, "## 方案\n\n大变更。", tmp_path=tmp_path)
    assert result.returncode == 1
    assert "Reuse Self-Check" in result.stdout
    assert "2000" in result.stdout


def test_renamed_gate_file_expands_to_full_delete_and_add(tmp_path):
    """Rename divergence regression (author side): the fingerprint must cover
    a full delete of the old gate path plus a full add of the new path — the
    same fixed records the reviewer derives — never a rename-paired,
    content-only diff."""
    gate_repo = GateRepo(tmp_path)
    gate_repo.commit(
        "chore: initialize\n\nBase with a requirements file.\n\nCo-Authored-By: glm-5.3",
        {"README.md": "base\n", "requirements/ubuntu.txt": "old-a\nold-b\n"},
    )
    gate_repo.push("master")
    gate_repo.git(["checkout", "-B", "feat/rename"])
    gate_repo.git(["mv", "requirements/ubuntu.txt", "requirements/server.txt"])
    gate_repo.commit(
        "feat: rename requirement\n\nGate-relevant rename.\n\nCo-Authored-By: glm-5.3",
        {"requirements/server.txt": "old-a\nold-b\nserver-c\n"},
    )
    gate_repo.push("feat/rename")

    deterministic = compute_verification_inputs(
        [
            {"filename": "requirements/server.txt", "patch": "+old-a\n+old-b\n+server-c"},
            {"filename": "requirements/ubuntu.txt", "patch": "-old-a\n-old-b"},
        ]
    )
    platform_renamed = compute_verification_inputs(
        [{"filename": "requirements/server.txt", "previous_filename": "", "patch": " old-a\n old-b\n+server-c"}]
    )
    assert deterministic != platform_renamed  # rename pairing would fork the fingerprint

    result = run_gate(
        gate_repo, "## 方案\n\n重命名需求文件。", "--pr-stage", "review", tmp_path=tmp_path, branch="feat/rename"
    )
    assert result.returncode == 1
    assert deterministic in result.stdout  # the gate reports the D+A fingerprint
    assert platform_renamed not in result.stdout
