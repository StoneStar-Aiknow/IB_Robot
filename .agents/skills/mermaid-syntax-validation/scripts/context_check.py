"""Preflight context check for mermaid-syntax-validation skill (skill tool).

Automates the 5 mandatory checks from references/context-checks.md so they run
as one command instead of relying on the agent to follow prose instructions:

1. docs stack       - locate conf.py, report extensions / myst fence / mermaid_* opts / html_js_files
2. local runtime    - _static/js mermaid runtime assets (UMD or ESM + chunks/)
3. fence count      - number of ```mermaid blocks in *.md under docs root
4. mermaid pages    - generated HTML pages containing a mermaid container
5. script tags      - mermaid runtime references and CDN leakage in generated HTML

Usage:
    python -X utf8 context_check.py --docs-root docs [--build-dir docs/build/html]

Output is the evidence base required by the skill's report format. Missing
conf.py / build dir are reported, not treated as errors (repo may feed an
external Sphinx project).

Exit code: 0 = checks executed (findings may still require judgment),
           2 = usage / path error.
"""

import argparse
import re
import sys
from pathlib import Path

CONF_KEYS = [
    "extensions",
    "myst_fence_as_directive",
    "html_js_files",
    "mermaid_version",
    "mermaid_output_format",
    "mermaid_init_config",
    "mermaid_include_elk",
    "mermaid_fullscreen",
    "mermaid_use_local",
    "mermaid_elk_use_local",
    "mermaid_zenuml_use_local",
    "d3_use_local",
]

RUNTIME_FILES = ["mermaid.min.js", "mermaid-run.js", "mermaid.esm.min.mjs", "mermaid-init.js"]
CDN_RE = re.compile(r"cdn\.jsdelivr\.net|unpkg\.com")
MERMAID_CLASS_RE = re.compile(r'class="mermaid"|class="[^" ]*mermaid')
SCRIPT_RUNTIME_RE = re.compile(r"mermaid\.min\.js|mermaid-run\.js|mermaid\.esm\.min\.mjs")


def check_conf(docs_root: Path) -> list[str]:
    print("== [1/5] docs stack ==")
    lines = []
    confs = sorted(docs_root.rglob("conf.py"))
    if not confs:
        print("  conf.py: NOT FOUND under docs root (external Sphinx project?)")
        return lines
    for conf in confs:
        print(f"  conf.py: {conf}")
        text = conf.read_text(encoding="utf-8", errors="ignore")
        for key in CONF_KEYS:
            for m in re.finditer(rf"^\s*{re.escape(key)}\s*=\s*(.+)$", text, re.MULTILINE):
                val = m.group(1).strip()
                print(f"    {key} = {val[:120]}")
                lines.append(f"{conf}: {key} = {val[:120]}")
        if "sphinxcontrib.mermaid" not in text and "myst_parser" not in text:
            print("    WARNING: neither sphinxcontrib.mermaid nor myst_parser found in extensions")
    return lines


def check_runtime(docs_root: Path) -> list[str]:
    print("== [2/5] local mermaid runtime assets ==")
    found = []
    for static in sorted(docs_root.rglob("_static")):
        js_dirs = [static / "js", static]
        seen = set()
        for js_dir in js_dirs:
            if not js_dir.is_dir():
                continue
            for name in RUNTIME_FILES:
                f = js_dir / name
                if f.is_file() and f not in seen:
                    seen.add(f)
                    found.append(str(f))
                    print(f"  {f}")
            chunks = js_dir / "chunks"
            if chunks.is_dir() and chunks not in seen:
                seen.add(chunks)
                found.append(str(chunks) + "/")
                print(f"  {chunks}/ (ESM chunks)")
    if not found:
        print("  no local mermaid runtime files found under any _static/")
    return found


def check_fences(docs_root: Path) -> int:
    print("== [3/5] mermaid fence count ==")
    total = 0
    for path in sorted(docs_root.rglob("*.md")):
        total += sum(
            1 for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines() if line.strip() == "```mermaid"
        )
    print(f"  total ```mermaid fences in {docs_root}: {total}")
    return total


def check_html(build_dir: Path) -> tuple[list[str], list[str], list[str]]:
    print("== [4/5] generated HTML pages containing mermaid ==")
    pages, runtime_pages, cdn_pages = [], [], []
    if not build_dir.is_dir():
        print(f"  build dir not found, skipped: {build_dir}")
        return pages, runtime_pages, cdn_pages
    for path in sorted(build_dir.rglob("*.html")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if MERMAID_CLASS_RE.search(text):
            pages.append(str(path))
        if SCRIPT_RUNTIME_RE.search(text):
            runtime_pages.append(str(path))
        if CDN_RE.search(text):
            cdn_pages.append(str(path))
    print(f"  pages with mermaid container: {len(pages)}")
    print("== [5/5] script tags / CDN references in generated HTML ==")
    print(f"  pages referencing local mermaid runtime: {len(runtime_pages)}")
    print(f"  pages referencing CDN (cdn.jsdelivr.net / unpkg.com): {len(cdn_pages)}")
    for p in cdn_pages:
        print(f"    CDN: {p}")
    if pages and not runtime_pages:
        print("  WARNING: mermaid containers exist but no local runtime script referenced in HTML")
    return pages, runtime_pages, cdn_pages


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs-root", required=True, help="docs root directory to scan")
    ap.add_argument("--build-dir", default=None, help="generated HTML root (default: <docs-root>/../build/html)")
    args = ap.parse_args()

    docs_root = Path(args.docs_root)
    if not docs_root.is_dir():
        print(f"error: docs root not found: {docs_root}", file=sys.stderr)
        return 2
    build_dir = Path(args.build_dir) if args.build_dir else docs_root.parent / "build" / "html"

    print(f"context check: docs_root={docs_root} build_dir={build_dir}")
    check_conf(docs_root)
    check_runtime(docs_root)
    check_fences(docs_root)
    check_html(build_dir)
    print("context check done. Use these findings in the skill report (runtime files, fence count, page stats).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
