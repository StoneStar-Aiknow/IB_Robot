import argparse
import re
import shutil
from pathlib import Path


def _slugify(title: str) -> str:
    return re.sub(r"[^\w\s]", "", title).strip().replace(" ", "_").lower()


def _page_content(full_title: str, body: str) -> str:
    title_heading = f"# {full_title}"
    body = body.lstrip("\n")
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
