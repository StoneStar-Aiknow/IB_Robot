"""Generate doc_config.json from DeepWiki wiki structure text (skill tool).

Automates steps 2-7 of the deepwiki-config skill so they run as one command
instead of relying on the agent to follow prose instructions:

2. parse           - extract section id / title / level / parent per line
3. labels          - slug labels (strip "(xxx)", drop stop words, underscore);
                     update scenario reuses labels from an existing config;
                     conflicts get "_2"/"_3" suffixes
4. hierarchy       - hybrid mode: top sections without sub pages become
                     "<label>.md" leaf nodes, sections with sub pages become
                     directory nodes with "subs"
5. assemble        - build id_to_label / title_to_label / hierarchy
6. validate        - three-way consistency check; on failure print a
                     structured error report and do NOT write the file
7. write           - dump JSON (indent=4, ensure_ascii=False)

Usage:
    python -X utf8 generate_config.py <input_file> --output doc_config.json [--repo owner/repo]

Input file is the page list returned by the DeepWiki MCP
read_wiki_structure tool, e.g.:

    - 1 IB-Robot Overview
    - 2 Getting Started
      - 2.1 Environment Setup

Exit code: 0 = success (summary on stdout),
           1 = consistency validation failed (file not written),
           2 = input / usage error.
"""

import argparse
import json
import re
import sys
from pathlib import Path

STOP_WORDS = {"and", "or", "of", "the", "for", "with", "a", "an", "in", "on", "to"}
SECTION_RE = re.compile(r"^\s*[-*]?\s*(\d+(?:\.\d+)?)\s+(.+?)\s*$")


def sort_key(section_id):
    return tuple(int(part) for part in section_id.split("."))


def parse_wiki_structure(text):
    """Parse wiki structure lines into page records.

    Returns (pages, skipped): pages are dicts with id / title / level / parent,
    skipped are (lineno, line) tuples for lines that matched no section.
    """
    pages = []
    skipped = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        match = SECTION_RE.match(raw)
        if not match:
            skipped.append((lineno, raw.strip()))
            continue
        section_id, title = match.groups()
        pages.append(
            {
                "id": section_id,
                "title": title,
                "level": 1 if "." in section_id else 0,
                "parent": section_id.split(".")[0],
            }
        )
    return pages, skipped


def make_label(title):
    """Slug a title: strip "(xxx)" groups, lowercase, drop stop words, underscore."""
    stripped = re.sub(r"\([^)]*\)", " ", title)
    words = [w for w in re.split(r"[^a-z0-9]+", stripped.lower()) if w and w not in STOP_WORDS]
    return "_".join(words)


def assign_labels(pages, existing_id_to_label):
    """Assign labels to pages, reusing existing ones in update scenarios.

    Returns (id_to_label, conflicts) where conflicts are
    (section_id, title, original_label) tuples that required a numeric suffix.
    """
    id_to_label = {}
    conflicts = []
    used = set()
    for page in sorted(pages, key=lambda p: sort_key(p["id"])):
        section_id = page["id"]
        if section_id in existing_id_to_label:
            label = existing_id_to_label[section_id]
        else:
            label = make_label(page["title"])
            if label in used:
                conflicts.append((section_id, page["title"], label))
                suffix = 2
                while f"{label}_{suffix}" in used:
                    suffix += 1
                label = f"{label}_{suffix}"
        used.add(label)
        id_to_label[section_id] = label
    return id_to_label, conflicts


def build_hierarchy(pages, id_to_label):
    """Build the hybrid-mode hierarchy.

    Top sections without sub pages become "<label>.md" leaf nodes; sections
    with sub pages become directory nodes holding "subs".
    Returns (hierarchy, leaf_nodes, dir_nodes) with node tuples of
    (label, title[, sub_count]).
    """
    hierarchy = {}
    leaf_nodes = []
    dir_nodes = []
    top_pages = sorted((p for p in pages if p["level"] == 0), key=lambda p: sort_key(p["id"]))
    sub_pages = sorted((p for p in pages if p["level"] == 1), key=lambda p: sort_key(p["id"]))
    for page in top_pages:
        label = id_to_label[page["id"]]
        subs = [s for s in sub_pages if s["parent"] == page["id"]]
        if subs:
            hierarchy[label] = {
                "title": page["title"],
                "subs": {id_to_label[s["id"]] + ".md": s["title"] for s in subs},
            }
            dir_nodes.append((label, page["title"], len(subs)))
        else:
            hierarchy[label + ".md"] = {"title": page["title"]}
            leaf_nodes.append((label, page["title"]))
    return hierarchy, leaf_nodes, dir_nodes


def validate(id_to_label, title_to_label, hierarchy, pages):
    """Three-way consistency check (6a/6b/6c). Returns a list of errors."""
    errors = []

    # 6a: id_to_label <-> title_to_label via the page table
    for page in sorted(pages, key=lambda p: sort_key(p["id"])):
        label = id_to_label[page["id"]]
        mapped = title_to_label.get(page["title"])
        if mapped != label:
            errors.append(f"id_to_label[{page['id']}] label={label!r} 但 title_to_label[{page['title']!r}] label={mapped!r}")
    page_titles = {p["title"] for p in pages}
    for title in sorted(set(title_to_label) - page_titles):
        errors.append(f"title_to_label 存在多余条目（无对应页面）: {title!r}")

    # 6b: id_to_label <-> hierarchy label sets
    label_set = set(id_to_label.values())
    hierarchy_labels = set()
    for key, cfg in hierarchy.items():
        hierarchy_labels.add(key[:-3] if key.endswith(".md") else key)
        for sub_key in cfg.get("subs", {}):
            hierarchy_labels.add(sub_key[:-3])
    for label in sorted(label_set - hierarchy_labels):
        errors.append(f"孤立 label（id_to_label 中存在但 hierarchy 中缺失）: {label!r}")
    for label in sorted(hierarchy_labels - label_set):
        errors.append(f"缺失 label（hierarchy 中存在但 id_to_label 中缺失）: {label!r}")

    # 6c: title_to_label <-> hierarchy titles
    hierarchy_titles = set()
    for cfg in hierarchy.values():
        hierarchy_titles.add(cfg["title"])
        hierarchy_titles.update(cfg.get("subs", {}).values())
    for title in sorted(hierarchy_titles - set(title_to_label)):
        errors.append(f"hierarchy 标题不在 title_to_label 中: {title!r}")
    for title in sorted(set(title_to_label) - hierarchy_titles):
        errors.append(f"title_to_label 标题不在 hierarchy 中: {title!r}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="从 DeepWiki wiki 结构文本生成 doc_config.json")
    parser.add_argument("input_file", help="wiki 结构文本文件（mcp_deepwiki_read_wiki_structure 的返回内容）")
    parser.add_argument("--output", "-o", default="doc_config.json", help="输出路径（默认 doc_config.json）")
    parser.add_argument("--repo", default=None, help="仓库名，仅用于摘要显示")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"错误: 输入文件不存在: {args.input_file}", file=sys.stderr)
        return 2
    text = input_path.read_text(encoding="utf-8-sig")

    pages, skipped = parse_wiki_structure(text)
    if not pages:
        print("错误: 输入中未解析到任何页面", file=sys.stderr)
        return 2

    # Reject duplicate titles early: they would collapse title_to_label entries.
    seen_titles = {}
    for page in sorted(pages, key=lambda p: sort_key(p["id"])):
        if page["title"] in seen_titles:
            print(
                f"错误: 重复标题 {page['title']!r}（{seen_titles[page['title']]} 与 {page['id']}）",
                file=sys.stderr,
            )
            return 2
        seen_titles[page["title"]] = page["id"]

    # Sub pages whose parent section is missing from the input.
    top_ids = {p["id"] for p in pages if p["level"] == 0}
    for page in sorted((p for p in pages if p["level"] == 1 and p["parent"] not in top_ids), key=lambda p: sort_key(p["id"])):
        print(f"错误: 二级子页面 {page['id']} {page['title']!r} 的父章节 {page['parent']} 不存在", file=sys.stderr)
        return 2

    # Update scenario: reuse labels from an existing valid config.
    existing = {}
    output_path = Path(args.output)
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8-sig") as f:
                old = json.load(f)
            existing = dict(old["id_to_label"])
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            print(f"错误: 已存在的 {args.output} 无法解析（{exc}）；请修复或删除后重试", file=sys.stderr)
            return 2

    id_to_label, conflicts = assign_labels(pages, existing)
    ordered_pages = sorted(pages, key=lambda p: sort_key(p["id"]))
    title_to_label = {p["title"]: id_to_label[p["id"]] for p in ordered_pages}
    hierarchy, leaf_nodes, dir_nodes = build_hierarchy(pages, id_to_label)

    errors = validate(id_to_label, title_to_label, hierarchy, pages)
    if errors:
        print("验证失败:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(f"未写入 {args.output}", file=sys.stderr)
        return 1

    config = {"id_to_label": id_to_label, "title_to_label": title_to_label, "hierarchy": hierarchy}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        f.write("\n")

    # Summary (step 8 of the skill workflow).
    reused = sorted(set(existing) & set(id_to_label), key=sort_key)
    print("=== doc_config.json 生成摘要 ===")
    print(f"仓库: {args.repo or '(未指定)'}")
    print(f"页面总数: {len(pages)}（一级章节 {len(top_ids)}，二级子页面 {len(pages) - len(top_ids)}）")
    print(f"叶子节点: {len(leaf_nodes)} 个 — " + (", ".join(label for label, _ in leaf_nodes) or "无"))
    print(
        f"目录节点: {len(dir_nodes)} 个 — "
        + (", ".join(f"{label} ({count} 个子页面)" for label, _, count in dir_nodes) or "无")
    )
    print(f"验证: id_to_label / title_to_label / hierarchy 三者一致")
    if reused:
        print(f"更新场景: 复用已有 label {len(reused)} 个")
    if conflicts:
        for section_id, title, original in conflicts:
            print(f"label 冲突: {section_id} {title!r} 生成 {original!r} 已撞车，已追加数字后缀")
    if skipped:
        print(f"跳过 {len(skipped)} 行无法识别:")
        for lineno, line in skipped:
            print(f"  第 {lineno} 行: {line!r}")
    print("label 对照表:")
    for page in ordered_pages:
        marker = "（已有）" if page["id"] in existing else ""
        print(f"  {page['id']:<6} {page['title']} → {id_to_label[page['id']]}{marker}")
    print(f"已写入: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
