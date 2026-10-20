import argparse
import re
import shutil
from pathlib import Path


def _slugify(title: str) -> str:
    """Slug aligned with DeepWiki official labels: drop parenthesized segments,
    treat remaining non-word chars as separators, collapse runs into a single "_"."""
    without_parens = re.sub(r"[（(][^（）()]*[)）]", " ", title)
    replaced = re.sub(r"[^\w\s]", " ", without_parens)
    return re.sub(r"[\s_]+", "_", replaced).strip("_").lower()


def _normalize_heading(text: str) -> str:
    """Strip leading numbering ("1.1. ") and trailing section suffix ("(Section 8.3)" / "（8.3 节）")."""
    t = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text.strip())
    t = re.sub(r"\s*[（(]\s*(?:Section\s*)?[\d.]+\s*节?\s*[)）]$", "", t)
    return t.strip()


def _page_content(full_title: str, body: str) -> str:
    title_heading = f"# {full_title}"
    body = body.lstrip("\n")
    lines = body.split("\n")
    # Drop a leading H1 that is a numbering/suffix variant of the page title,
    # e.g. "# 1.1. Getting Started & Environment" for "Getting Started & Environment".
    heading = re.match(r"^#\s+(.+)$", lines[0].strip()) if lines else None
    if heading and _normalize_heading(heading.group(1)) == full_title.strip():
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
        body = "\n".join(lines).lstrip("\n")
    if body.split("\n", 1)[0].strip() == title_heading:
        return body.rstrip() + "\n"
    return f"{title_heading}\n\n{body}".rstrip() + "\n"


def split_raw_md(input_file: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    content = input_file.read_text(encoding="utf-8")
    pages = re.split(r"# Page: ", content)

    for page in pages[1:]:
        lines = page.strip().split("\n")
        full_title = lines[0].strip()
        body = "\n".join(lines[1:]).lstrip("\n")
        filename = f"{_slugify(full_title)}.md"

        out_path = output_dir / filename
        print(f"Saving: {full_title} -> {out_path}")
        out_path.write_text(_page_content(full_title, body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DeepWiki '# Page: <Title>' markdown into flat page files.")
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("docs/migration/IB_Robot_doc_raw.md"),
        help="DeepWiki raw markdown file. Default: docs/migration/IB_Robot_doc_raw.md",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("docs/migration/raw_md"),
        help="Directory for split markdown pages. Default: docs/migration/raw_md",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_raw_md(args.input_file, args.output_dir)
