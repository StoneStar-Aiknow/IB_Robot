"""Post-build HTML scanner for mermaid-syntax-validation skill (skill tool).

Replaces the inline bash heredocs and POSIX-only `test -f` checks of the
build-verification step so the same gate runs on bash and Windows PowerShell
alike:

1. errors  - Mermaid error text in generated HTML
             (Syntax error in text / Parse error / mermaid version / Diagram error)
2. cdn     - online runtime dependencies accidentally introduced
             (cdn.jsdelivr.net / unpkg.com)
3. runtime - local mermaid runtime assets copied into the build output
             (UMD: mermaid.min.js + runner; ESM: mermaid.esm.min.mjs + chunks/)

Usage:
    python -X utf8 html_scan.py --build-dir docs/build/html --scan errors
    python -X utf8 html_scan.py --build-dir docs/build/html --scan cdn
    python -X utf8 html_scan.py --build-dir docs/build/html --scan runtime

For errors/cdn scans, matching file paths are printed one per line to stdout
and a summary goes to stderr so stdout stays empty when clean. For the runtime
scan, found assets are printed to stdout (evidence for the report's runtime
field); an ESM entry without chunks/ beside it is reported as a WARNING.

Whether a result is a failure is policy and stays in SKILL.md: error text is
always a failure; CDN references are a failure only for docs required to work
offline; a missing browser runtime is normal for server-side rendering
(png/svg direct output) and must be explained in the report.

Exit code: 0 = errors/cdn: no matches; runtime: assets found and complete
           1 = errors/cdn: matches found; runtime: no assets found, or an
               ESM entry without chunks/ beside it
           2 = scan could not execute (build dir missing / no HTML files);
               must NOT be treated as a passed gate.
"""

import argparse
import re
import sys
from pathlib import Path

ERROR_TEXT_RE = re.compile(r"Syntax error in text|Parse error|mermaid version|Diagram error")
CDN_RE = re.compile(r"cdn\.jsdelivr\.net|unpkg\.com")

# Keep in sync with context_check.py RUNTIME_FILES.
RUNTIME_FILES = ["mermaid.min.js", "mermaid-run.js", "mermaid.esm.min.mjs", "mermaid-init.js"]
ESM_ENTRY = "mermaid.esm.min.mjs"

SCANS = {
    "errors": ERROR_TEXT_RE,
    "cdn": CDN_RE,
}


def scan_runtime(build_dir: Path) -> int:
    """List local mermaid runtime assets under _static/ in the build output.

    Mirrors the asset layout accepted by context_check.py: runtime files may
    sit directly in _static/ or in _static/js/, and an ESM entry requires a
    chunks/ directory beside it.
    """
    found = []
    for static in sorted(build_dir.rglob("_static")):
        for js_dir in (static / "js", static):
            if not js_dir.is_dir():
                continue
            for name in RUNTIME_FILES:
                f = js_dir / name
                if f.is_file():
                    found.append(f)
            chunks = js_dir / "chunks"
            if chunks.is_dir():
                found.append(chunks)
    for f in found:
        print(f)
    if not found:
        print(f"runtime scan: no local mermaid runtime assets found under _static/ in {build_dir}", file=sys.stderr)
        return 1
    incomplete = [f for f in found if f.name == ESM_ENTRY and (f.parent / "chunks") not in found]
    for f in incomplete:
        print(f"WARNING: ESM entry {f} has no chunks/ directory beside it; ESM runtime incomplete", file=sys.stderr)
    status = "ESM runtime incomplete" if incomplete else "complete"
    print(f"runtime scan: {len(found)} asset(s) under _static/ in {build_dir} ({status})", file=sys.stderr)
    return 1 if incomplete else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-dir", required=True, help="generated HTML root to scan")
    ap.add_argument("--scan", required=True, choices=sorted(SCANS) + ["runtime"], help="scan type: cdn, errors, or runtime")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.is_dir():
        print(f"error: build dir not found: {build_dir}", file=sys.stderr)
        return 2
    pages = sorted(build_dir.rglob("*.html"))
    if not pages:
        print(f"error: no HTML files found under {build_dir}", file=sys.stderr)
        return 2

    if args.scan == "runtime":
        return scan_runtime(build_dir)

    pattern = SCANS[args.scan]
    matches = [path for path in pages if pattern.search(path.read_text(encoding="utf-8", errors="ignore"))]
    for path in matches:
        print(path)
    print(f"{args.scan} scan: {len(matches)} matching page(s) in {len(pages)} HTML files", file=sys.stderr)
    return 1 if matches else 0


if __name__ == "__main__":
    sys.exit(main())
