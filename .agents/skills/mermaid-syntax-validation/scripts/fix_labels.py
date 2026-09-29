"""Syntax-only Mermaid label fixer for markdown sources (skill tool).

Fixes (graph/flowchart diagrams only; sequence/state/class diagrams are skipped):
1. doubled label delimiter quotes  A[""x""]           -> A["x"]
2. inner quotes in labels           A[a "b"]            -> A["a #quot;b#quot;"]
3. unquoted labels with specials    A[profile: {d: 0}]  -> A["profile: {d: 0}"]
4. anonymous node declarations      ["label"]           -> mmd_anon_N["label"]
5. edge labels with specials        -->|Returns X[]|     -->|"Returns X[]"|

Only ```mermaid fenced blocks are touched. Visible label text is preserved.

Usage:
    python -X utf8 fix_labels.py <target_dir>          # apply fixes
    python -X utf8 fix_labels.py <target_dir> --dry-run

Idempotent: running twice reports 0 changes on already-fixed sources.
"""

import argparse
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"(```mermaid\r?\n)(.*?)(```)", re.DOTALL)
DIAGRAM_HEAD_RE = re.compile(r"^\s*(graph|flowchart)\b", re.IGNORECASE)
SPECIAL_CHARS = set('()[]{}"')

STATMENT_KEYWORDS = ("classDef", "class ", "style ", "click ", "linkStyle", "%%", "subgraph", "end")

CHANGED = []


def transform_label(content: str) -> str:
    """Return fixed label content (text between brackets), preserving visible text."""
    # protect inner shape wrappers: (X) cylinder/stadium, {X} hexagon, /X/ \X\ parallelogram
    if len(content) >= 2:
        first, last = content[0], content[-1]
        if (first == "(" and last == ")") or (first == "{" and last == "}") or (
            first in "/\\" and last in "/\\"
        ):
            return first + transform_label(content[1:-1]) + last
    if content.startswith('"') and content.endswith('"') and len(content) >= 2:
        core = content[1:-1]
        # strip doubled delimiter quotes: outer "" "" -> " "
        if len(core) >= 2 and core.startswith('"') and core.endswith('"') and '"' not in core[1:-1]:
            core = core[1:-1]
        core = core.replace('"', "#quot;")
        return f'"{core}"'
    if any(c in content for c in SPECIAL_CHARS):
        core = content.replace('"', "#quot;")
        return f'"{core}"'
    return content


def fix_edge_labels(line: str, ctx: dict) -> str:
    """Quote |edge label| segments whose content contains shape-triggering characters.

    Applies only to lines containing an arrow (-- or -.), i.e. edge statements.
    Labels already quoted are left untouched.
    """
    if not re.search(r"--|-\.", line):
        return line
    if "|" not in line:
        return line

    def repl(m):
        label = m.group(1)
        if label.startswith('"') and label.endswith('"'):
            return m.group(0)
        if any(c in label for c in SPECIAL_CHARS):
            fixed = '"' + label.replace('"', "#quot;") + '"'
            CHANGED.append((ctx["file"], ctx["block"], line.strip()[:80], label[:60], fixed[:60]))
            return f"|{fixed}|"
        return m.group(0)

    return re.sub(r"\|([^|\n]*)\|", repl, line)


def fix_line(line: str, ctx: dict) -> str:
    stripped = line.lstrip()
    if stripped.startswith(STATMENT_KEYWORDS):
        return line
    line = fix_edge_labels(line, ctx)
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in "[{":
            # walk back over the ORIGINAL line to find an adjacent node id prefix
            id_prefix = ""
            k = i
            while k > 0 and (line[k - 1].isalnum() or line[k - 1] in "_-" or "\u4e00" <= line[k - 1] <= "\u9fff"):
                k -= 1
            if k < i:
                id_prefix = line[k:i]
            close = "]" if ch == "[" else "}"
            depth = 0
            j2 = i
            end = -1
            while j2 < n:
                c = line[j2]
                if c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        end = j2
                        break
                j2 += 1
            if end == -1:
                out.append(ch)
                i += 1
                continue
            content = line[i + 1 : end]
            if content.strip() == "":
                out.append(line[i : end + 1])
                i = end + 1
                continue
            fixed = transform_label(content)
            if id_prefix:
                # id chars were already emitted to out; append shape bracket + fixed label
                if fixed != content:
                    CHANGED.append((ctx["file"], ctx["block"], line.strip()[:80], content[:60], fixed[:60]))
                out.append(f"{ch}{fixed}{close}")
            elif line[:i].strip() == "":
                # anonymous node at line start -> assign stable id
                prefix = f"mmd_anon_{ctx['anon']}"
                ctx["anon"] += 1
                CHANGED.append((ctx["file"], ctx["block"], "ANON->id " + prefix, content[:40], prefix))
                out.append(f"{prefix}{ch}{fixed}{close}")
            else:
                # bracket mid-line without adjacent id (e.g. inside edge label) -> leave as-is
                out.append(line[i : end + 1])
            i = end + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def fix_block(block: str, fname: str, bidx: int) -> str:
    lines = block.splitlines(keepends=True)
    if not lines or not DIAGRAM_HEAD_RE.match(lines[0]):
        # only graph/flowchart labels are auto-fixed; other diagram types need manual rules
        return block
    ctx = {"file": fname, "block": bidx, "anon": 1}
    return "".join(fix_line(ln, ctx) for ln in lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="directory containing .md files to fix")
    ap.add_argument("--dry-run", action="store_true", help="report fixes without writing")
    args = ap.parse_args()

    target = Path(args.target)
    if not target.is_dir():
        print(f"error: not a directory: {target}", file=sys.stderr)
        return 2

    n_files = 0
    n_blocks = 0
    for path in sorted(target.rglob("*.md")):
        raw = path.read_bytes()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")

        def repl(m):
            nonlocal n_blocks
            n_blocks += 1
            return m.group(1) + fix_block(m.group(2), path.name, n_blocks) + m.group(3)

        new_text = BLOCK_RE.sub(repl, text)
        if new_text != text:
            n_files += 1
            if not args.dry_run:
                data = new_text.encode("utf-8")
                if bom:
                    data = b"\xef\xbb\xbf" + data
                path.write_bytes(data)
    mode = "DRY-RUN, would change" if args.dry_run else "files changed"
    print(f"{mode}: {n_files}, mermaid blocks seen: {n_blocks}, label fixes: {len(CHANGED)}")
    for c in CHANGED:
        print(f"  {c[0]} [#{c[1]}] {c[2]!r}\n      {c[3]!r} -> {c[4]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
