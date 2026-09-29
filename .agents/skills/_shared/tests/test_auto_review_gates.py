"""Both auto-review workflows must finish reading before publishing any result."""

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import atomgit_cli

TARGET = atomgit_cli.RepoTarget("acme", "widget", "test")
PATCH = "@@ -1,1 +1,2 @@\n ctx\n+added line"
CONFIG = {"anthropic": {"apiKey": "test-key"}}


@pytest.fixture(params=["code", "architecture"])
def review(request, monkeypatch):
    arch = request.param == "architecture"
    skill = Path(__file__).parents[2] / ("atomgit-pr-architecture-review" if arch else "atomgit-pr-review")
    script = "architecture_review" if arch else "pr_review"
    llm_name = "llm_architecture_reviewer" if arch else "llm_reviewer"
    collision_names = ("comment_formatter", llm_name)
    saved = {name: sys.modules[name] for name in collision_names if name in sys.modules}
    for name in collision_names:
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(skill / "lib"))
    monkeypatch.syspath_prepend(str(skill / "scripts"))
    spec = importlib.util.spec_from_file_location(f"auto_gate_{script}", skill / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
        formatter = module.CommentFormatter(ai_model="test-model")
        reviewer = (module.ArchitectureReviewer if arch else module.CodeReviewer)("gitcode", TARGET, formatter)
        yield SimpleNamespace(module=module, reviewer=reviewer, arch=arch)
    finally:
        for name in collision_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.mark.parametrize(
    "scenario",
    ["findings", "no-findings", "missing-head", "partial-read", "all-reads", "incomplete-dry-run", "missing-key"],
)
def test_auto_review_publication_requires_complete_reads(review, scenario, monkeypatch, tmp_path, capsys):
    module = review.module
    calls, reviewed = [], []
    broken_head = scenario in {"missing-head", "incomplete-dry-run"}
    filenames = ["src/a.py", "src/b.py"] if scenario == "partial-read" else ["src/a.py"]
    unreadable = {"src/b.py"} if scenario == "partial-read" else {"src/a.py"} if scenario == "all-reads" else set()
    record = {
        "number": 7,
        "title": "change",
        "head": {
            "sha": "b" * 40,
            "ref": "feature",
            "repo": {} if broken_head else {"full_name": TARGET.slug},
        },
    }

    def run_json(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["pr", "view"]:
            data = record
        elif args[0] == "api" and "/files?" in args[1]:
            data = [
                {"filename": name, "status": "modified", "additions": 1, "deletions": 0, "patch": PATCH}
                for name in filenames
            ]
        elif args[0] == "api" and "/contents/" in args[1]:
            name = unquote(args[1].split("/contents/", 1)[1].split("?", 1)[0])
            if name in unreadable:
                raise atomgit_cli.CLIExecutionError(f"cannot read {name}")
            content = b"ctx\nadded line\n"
            data = {"content": base64.b64encode(content).decode(), "size": len(content)}
        elif args[:2] == ["pr", "comment"]:
            data = {"id": 9}
        else:
            raise AssertionError(f"unexpected CLI call: {args}")
        return atomgit_cli.CLIResult(argv=args, returncode=0, stdout=json.dumps(data), stderr="", data=data)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("unexpected unstructured CLI access")

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def review_file(self, file_path, *args, **kwargs):
            reviewed.append(file_path)
            if scenario not in {"findings", "partial-read"} or file_path != "src/a.py":
                return []
            values = dict(file=file_path, line=2, title="Wrong default", description="Breaks the contract.")
            return [
                module.ArchitectureIssue(**values, pillar="ssot")
                if review.arch
                else module.CodeIssue(**values, confidence=100)
            ]

    monkeypatch.setattr(atomgit_cli, "run_json", run_json)
    monkeypatch.setattr(atomgit_cli, "run", unexpected_run)
    monkeypatch.setattr(module, "LLMArchitectureReviewer" if review.arch else "LLMCodeReviewer", FakeLLM)
    args = SimpleNamespace(
        pr=7,
        llm_provider="anthropic",
        llm_model="test-model",
        threshold=80,
        dry_run=scenario == "incomplete-dry-run",
        output_dir=str(tmp_path),
    )
    ok = module.mode_auto(args, "gitcode", TARGET, review.reviewer, {} if scenario == "missing-key" else CONFIG)
    writes = [argv for argv in calls if argv[:2] == ["pr", "comment"]]
    if scenario in {"findings", "no-findings"}:
        assert ok is True
        assert reviewed == filenames
        if scenario == "findings":
            assert any("--position" in argv for argv in writes)
        elif review.arch:
            assert len(writes) == 1 and "--path" not in writes[0]
    else:
        assert ok is False and writes == []
        if scenario == "missing-key":
            assert calls == []
        else:
            assert "已阻止任何发布" in capsys.readouterr().out
            if scenario != "partial-read":
                assert reviewed == []
            else:
                assert reviewed == ["src/a.py"]  # partial findings must not escape
