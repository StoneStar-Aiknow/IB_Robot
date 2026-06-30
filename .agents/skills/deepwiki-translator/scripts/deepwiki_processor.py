import argparse
from pathlib import Path

from deepwiki_config import (
    build_label_to_filepath,
    configured_titles,
    label_to_title,
    load_config,
    merged_title_to_label,
    validate_config,
)
from deepwiki_generator import run_generation
from deepwiki_links import (
    extract_page_id,
    fix_links,
    get_relative_path,
    get_url_from_text,
    is_line_noise_link,
    looks_like_repo_path,
    looks_like_source_ref,
    protect_blocks,
    repo_url,
    report_link_conversions,
    resolve_page_link,
    resolve_title_link,
    restore_blocks,
    strip_page_marker,
)
from deepwiki_pages import clean_content, extract_title, load_pages

DEFAULT_WORKDIR = Path.cwd()


class DeepWikiProcessor:
    def __init__(self, input_dir, output_dir, branch, config_file, source_config_file=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.branch = branch
        self.base_url = f"https://atomgit.com/openeuler/IB_Robot/{self.branch}/"
        self.repo_root = Path.cwd()
        self.warnings = []

        self.id_to_label, self.title_to_label, self.hierarchy = load_config(config_file)
        self.link_title_to_label = merged_title_to_label(self.title_to_label, source_config_file)
        self.label_to_title = label_to_title(self.title_to_label)
        self.label_to_filepath = {}
        self.link_conversions = []
        self._build_label_to_filepath()

    def warn(self, message):
        self.warnings.append(message)

    def _build_label_to_filepath(self):
        self.label_to_filepath = build_label_to_filepath(self.hierarchy, self.title_to_label)

    def get_url_from_text(self, text):
        return get_url_from_text(text, self.base_url, self.repo_root)

    def _looks_like_source_ref(self, text):
        return looks_like_source_ref(text)

    def _looks_like_repo_path(self, text):
        return looks_like_repo_path(text)

    def _is_line_noise_link(self, text):
        return is_line_noise_link(text)

    def _strip_page_marker(self, text):
        return strip_page_marker(text)

    def _repo_url(self, path):
        return repo_url(path, self.base_url, self.repo_root)

    def _extract_page_id(self, text, url):
        return extract_page_id(text, url)

    def _resolve_page_link(self, text, url, current_filepath):
        return resolve_page_link(text, url, current_filepath, self.id_to_label, self.link_title_to_label, self.label_to_filepath, self.warn, self.label_to_title)

    def _resolve_title_link(self, text, current_filepath):
        return resolve_title_link(text, current_filepath, self.link_title_to_label, self.label_to_filepath, self.warn, self.label_to_title)

    def _protect_blocks(self, content):
        return protect_blocks(content)

    def _restore_blocks(self, content, placeholders):
        return restore_blocks(content, placeholders)

    def _get_relative_path(self, from_file, to_file):
        return get_relative_path(from_file, to_file)

    def _record_conversion(self, filepath, original, converted):
        self.link_conversions.append({
            "file": filepath,
            "original": original,
            "converted": converted,
        })

    def report_link_conversions(self):
        report_link_conversions(self.link_conversions, self.output_dir)

    def fix_links(self, content, current_filepath):
        return fix_links(
            content,
            current_filepath,
            self.base_url,
            self.id_to_label,
            self.link_title_to_label,
            self.label_to_filepath,
            self.link_conversions,
            self.warn,
            self.repo_root,
            self.label_to_title,
        )

    def _clean_content(self, md_content, page_title):
        return clean_content(md_content, page_title)

    def _load_pages(self):
        return load_pages(self.input_dir, self.warn)

    def _extract_title(self, text, file_path):
        return extract_title(text, file_path, self.warn)

    def _configured_titles(self):
        return configured_titles(self.hierarchy)

    def _validate_config(self, title_to_content):
        return validate_config(self.hierarchy, self.id_to_label, self.label_to_filepath, title_to_content, self.warn)

    def run(self):
        run_generation(
            self.input_dir,
            self.output_dir,
            self.branch,
            self.id_to_label,
            self.link_title_to_label,
            self.hierarchy,
            self.label_to_filepath,
            self.warnings,
            self.link_conversions,
            self.label_to_title,
        )


def main():
    parser = argparse.ArgumentParser(description="DeepWiki processor: convert raw markdown to Sphinx-ready output")
    parser.add_argument("--input-dir", default=str(DEFAULT_WORKDIR / "raw_md"), help="Input markdown directory (default: raw_md/ in current working directory)")
    parser.add_argument("--output-dir", default=str(DEFAULT_WORKDIR / "ib_robot"), help="Output directory (default: ib_robot/ in current working directory)")
    parser.add_argument("--branch", default="master", help="AtomGit branch for repo URLs (default: master)")
    parser.add_argument("--config-file", default=str(DEFAULT_WORKDIR / "doc_config.json"), help="doc_config.json path (default: doc_config.json in current working directory)")
    parser.add_argument("--source-config-file", default=None, help="Optional source-language doc_config.json used only as title aliases for empty link resolution")
    args = parser.parse_args()

    source_config_file = args.source_config_file
    if source_config_file is None:
        default_source_config = Path(args.config_file).with_name("doc_config.json")
        if default_source_config != Path(args.config_file):
            source_config_file = str(default_source_config)

    processor = DeepWikiProcessor(args.input_dir, args.output_dir, args.branch, args.config_file, source_config_file)
    processor.run()


if __name__ == "__main__":
    main()
