"""Emit the L3 gate's skill-word lexicon from the skill catalog.

The gate lets a skill word through even when the model scores the sentence low
("帮我抓取橘子" is rare in the training corpus, but it is unmistakably addressed
to the robot). The word list must therefore follow the catalog rather than a
hand copy that drifts when a skill is renamed.

    python3 tools/addressee/emit_lexicon.py            # JSON list on stdout
    python3 tools/addressee/emit_lexicon.py --yaml      # YAML for a robot profile
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2] / "src" / "agent" / "skill_catalog" / "config" / "skills"

#: Aliases that are also ordinary dialogue. As an override the lexicon can only
#: ever accept, so an everyday "是的" or "同意" here would defeat the gate for
#: the most common back-channels in a meeting.
DIALOGUE_WORDS = frozenset(
    {"是的", "对", "对的", "好", "好的", "行", "可以", "同意", "嗯", "嗯嗯", "没错", "了解", "明白"}
)


def _aliases_from_text(text: str) -> list[str]:
    """Read ``description.aliases_zh`` without requiring PyYAML."""
    lines = text.splitlines()
    aliases: list[str] = []
    inside = False
    for line in lines:
        if re.match(r"^\s{2}aliases_zh:\s*$", line):
            inside = True
            continue
        if inside:
            match = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if match:
                aliases.append(_unquote(match.group(1)))
                continue
            if line.strip() and not line.startswith((" ", "\t")):
                break
            if line.strip() and not line.lstrip().startswith("-") and re.match(r"^\s{2}\w", line):
                break
    return aliases


def _unquote(value: str) -> str:
    return value.strip().strip("\"'")


def catalog_lexicon(catalog: Path = CATALOG) -> tuple[str, ...]:
    """Every ``aliases_zh`` string in the catalog, de-duplicated and ordered."""
    words: list[str] = []
    for manifest in sorted(catalog.glob("*/manifest.yaml")):
        for alias in _aliases_from_text(manifest.read_text(encoding="utf-8")):
            alias = alias.strip()
            if alias and alias not in words and alias not in DIALOGUE_WORDS:
                words.append(alias)
    return tuple(words)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(CATALOG))
    parser.add_argument("--yaml", action="store_true", help="print as a YAML list block")
    args = parser.parse_args()

    words = catalog_lexicon(Path(args.catalog))
    if not words:
        print(f"no aliases_zh found under {args.catalog}", file=sys.stderr)
        return 1
    if args.yaml:
        for word in words:
            print(f"          - {word}")
    else:
        print(json.dumps(list(words), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
