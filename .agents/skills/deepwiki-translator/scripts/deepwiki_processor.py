import argparse
from pathlib import Path

from deepwiki_config import build_label_to_filepath, label_to_title, load_config, merged_title_to_label
from deepwiki_generator import run_generation

DEFAULT_WORKDIR = Path.cwd()


class DeepWikiProcessor:
    def __init__(self, input_dir, output_dir, branch, config_file, source_config_file=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.branch = branch

        self.id_to_label, self.title_to_label, self.hierarchy = load_config(config_file)
        self.link_title_to_label = merged_title_to_label(self.title_to_label, source_config_file)
        self.label_to_title = label_to_title(self.title_to_label)
        self.label_to_filepath = build_label_to_filepath(self.hierarchy, self.title_to_label)
        self.warnings = []
        self.link_conversions = []

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
