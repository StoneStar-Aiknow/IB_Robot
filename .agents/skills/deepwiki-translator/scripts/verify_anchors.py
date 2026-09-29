"""Post-generation anchor and relative link verifier for DeepWiki translated output.

Mandatory final step of the deepwiki-translator pipeline: run it against the
generated output directory after every full or incremental generation.

Checks every generated Markdown file for:
  1. numeric page-id anchors (e.g. #16.1) that survived generation (always a defect);
  2. in-page anchors (#section) that match no heading in the same file;
  3. relative links (./x.md, ../y/x.md) whose target file does not exist;
  4. relative Markdown links with an #anchor suffix whose anchor matches no
     heading in the target file.

Fenced code blocks are ignored. External http(s)/mailto links are not checked.
Inline links are expected to stay on a single line, matching the DeepWiki export format.

Exit code: 0 when clean, 1 when issues are found, 2 on bad usage.

Usage:
    python verify_anchors.py --output-dir <output_dir>
"""

import argparse
import re
import sys
from pathlib import Path

from deepwiki_links import extract_heading_slugs, protect_blocks, replace_markdown_links

NUMERIC_PAGE_ID = re.compile(r"^\d+(?:\.\d+)*$")


def load_target_slugs(target, cache):
    cached = cache.get(target)
    if cached is None:
        protected, _ = protect_blocks(target.read_text(encoding="utf-8-sig"))
        cached = extract_heading_slugs(protected)
        cache[target] = cached
    return cached


def scan_file(md_path, root_dir, target_slug_cache):
    content = md_path.read_text(encoding="utf-8-sig")
    protected, _ = protect_blocks(content)
    own_slugs = extract_heading_slugs(protected)
    rel = md_path.relative_to(root_dir).as_posix()
    issues = []

    def check_link_at(line_no):
        def check_link(m):
            original = m.group(0)
            text = m.group(2).strip()
            url = m.group(3).strip()
            if url.startswith(("http://", "https://", "mailto:")) or not url:
                return original

            if url.startswith("#"):
                anchor = url[1:]
                if NUMERIC_PAGE_ID.match(anchor):
                    issues.append((rel, line_no, f"numeric page-id anchor survived generation: {original}"))
                elif anchor not in own_slugs:
                    issues.append((rel, line_no, f"dangling in-page anchor: {original}"))
                return original

            path_part, _, anchor = url.partition("#")
            target = (md_path.parent / path_part).resolve()
            if not target.exists():
                issues.append((rel, line_no, f"broken relative link: {original}"))
                return original
            if anchor and target.suffix == ".md":
                target_slugs = load_target_slugs(target, target_slug_cache)
                if anchor not in target_slugs:
                    issues.append((rel, line_no, f"dangling anchor in target {path_part}: {original}"))
            return original

        return check_link

    for line_no, line in enumerate(protected.splitlines(), start=1):
        replace_markdown_links(line, check_link_at(line_no))
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify anchors and relative links in generated DeepWiki output")
    parser.add_argument("--output-dir", default="ib_robot", help="Generated output directory (default: ib_robot/ in current working directory)")
    args = parser.parse_args(argv)

    root_dir = Path(args.output_dir)
    if not root_dir.is_dir():
        print(f"error: output directory not found: {root_dir}", file=sys.stderr)
        return 2

    target_slug_cache = {}
    md_files = sorted(root_dir.rglob("*.md"))
    issues = []
    for md_path in md_files:
        issues.extend(scan_file(md_path, root_dir, target_slug_cache))

    if not issues:
        print(f"OK: {len(md_files)} files scanned, no dangling anchors or broken relative links.")
        return 0

    print(f"{len(issues)} issue(s) found in {root_dir}:")
    for rel, line_no, message in issues:
        print(f"  {rel}:{line_no}: {message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
