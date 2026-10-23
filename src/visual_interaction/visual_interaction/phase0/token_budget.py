"""Count prompt tokens against the fixed OM shapes. No model is loaded.

Usage::

    source .shrc_local
    python3 -m visual_interaction.phase0.token_budget \\
        --tokenizer-dir <dir with vocab.json/merges.txt/added_tokens.json> \\
        --output measurement.json

Runs anywhere — the tokenizer is data, so this needs no NPU and no CANN. Exit
status 1 when any scenario does not fit, so an overflow is caught here rather
than on the board mid-round.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visual_interaction.phase0 import prompts
from visual_interaction.phase0.budget import (
    BudgetError,
    OmShape,
    assess,
    check_image_slots,
    expand_images,
)
from visual_interaction.phase0.report import build_report, digests, write_report
from visual_interaction.phase0.tokenizer import PromptTokenizer, TokenizerDataError

TOOL = "token_budget"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="checkout whose commit is recorded")
    parser.add_argument("--prefill-len", type=int, default=1280, help="OM prefill length P")
    parser.add_argument("--context-len", type=int, default=1344, help="OM KV cache length L")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-tokens", type=int, default=256, help="image tokens per frame")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    try:
        shape = OmShape(
            prefill_len=args.prefill_len,
            context_len=args.context_len,
            frames=args.frames,
            image_tokens_per_frame=args.image_tokens,
        )
        tokenizer = PromptTokenizer(args.tokenizer_dir)
    except (TokenizerDataError, ValueError) as error:
        print(f"{TOOL}: {error}", file=sys.stderr)
        return 2

    assessments = []
    for scenario in prompts.default_scenarios(frames=shape.frames).scenarios:
        try:
            check_image_slots(scenario.name, scenario.text, with_images=scenario.with_images, shape=shape)
        except BudgetError as error:
            print(f"{TOOL}: {error}", file=sys.stderr)
            return 2
        parts = [
            (name, tokenizer.count(expand_images(text, shape) if scenario.with_images else text))
            for name, text in scenario.parts
        ]
        assessments.append(
            assess(
                scenario.name,
                prompt_tokens=tokenizer.count(expand_images(scenario.text, shape)),
                max_new_tokens=args.max_new_tokens or scenario.max_new_tokens,
                shape=shape,
                with_images=scenario.with_images,
                parts=parts,
            )
        )

    results = {
        "shape": {
            "prefill_len": shape.prefill_len,
            "context_len": shape.context_len,
            "frames": shape.frames,
            "image_tokens_per_frame": shape.image_tokens_per_frame,
            "image_positions": shape.image_positions,
            "text_capacity_with_images": shape.text_capacity_with_images,
        },
        "scenarios": [item.as_dict() for item in assessments],
        "all_fit": all(item.fits for item in assessments),
    }
    report = build_report(
        tool=TOOL,
        conditions={"max_new_tokens": args.max_new_tokens},
        results=results,
        repo=args.repo,
        assets=digests(tokenizer.data_files()),
        notes=["Tokenizer only; no model was loaded, so this measures the prompt budget and nothing else."],
    )
    write_report(args.output, report)

    for item in assessments:
        verdict = "ok " if item.fits else "OVER"
        print(
            f"  {verdict} {item.name:28s} {item.prompt_tokens:5d} tokens "
            f"({item.text_tokens:4d} text) headroom {shape.prefill_len - item.prompt_tokens:5d}"
        )
    print(f"{TOOL}: {'all scenarios fit' if results['all_fit'] else 'SOME DO NOT FIT'} -> {args.output}")
    return 0 if results["all_fit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
