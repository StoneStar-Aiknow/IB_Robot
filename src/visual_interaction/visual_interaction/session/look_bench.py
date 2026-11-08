"""Phase-zero bench for look_say's model calls, run on the board by hand.

Measures what plan §4.7 and §8 leave open: how long the three OMs take to
load, what one ``prepare`` (crop + vision OM) and one ``ask`` per question
cost at p50/p95 with the crop and with the whole frame, what the NPU and the
process hold before and after loading, and whether the yes/no questions say
"yes" to a picture with no gesture in it (informational only).

Run it only while no launch is active: it loads its own copy of the OMs onto
the NPU the live nodes share. It refuses when it sees a ``ros2 launch``
process unless ``--allow-launch`` is given. Environment as for the node: the
CANN ``set_env.sh`` sourced and the system site-packages on PYTHONPATH (see
internvl_vision's ImportError text).

    python3 -m visual_interaction.session.look_bench --demo-root /root/vlx_internvl_310p \\
        --image a.jpg --bbox 560,220,840,900 --image b.jpg [--runs 5] \\
        [--no-gesture-image c.jpg --no-gesture-bbox 630,180,955,960]

``--bbox`` pairs with ``--image`` by position, in the image's pixels; an image
without one gets a synthetic centred box (marked ``synthetic`` in the output).
Output is a plain-text table followed by one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

YES_NO_PREFIX = "q3."


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(value) for value in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bbox must be x1,y1,x2,y2, got {text!r}")
    return parts[0], parts[1], parts[2], parts[3]


def _synthetic_bbox(width: int, height: int) -> tuple[float, float, float, float]:
    """A standing person in the middle of the frame, for when no box is given."""
    return 0.35 * width, 0.15 * height, 0.65 * width, 0.95 * height


def _launch_running() -> list[str]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        words = [Path(word.decode(errors="replace")).name for word in argv if word]
        if "launch" in words and any(word.startswith("ros2") for word in words):
            found.append(f"{entry.name}: {' '.join(words)[:120]}")
    return found


def _npu_smi() -> str | None:
    """``npu-smi info`` as text, or None where the tool is absent. Read only."""
    tool = shutil.which("npu-smi")
    if tool is None:
        return None
    try:
        done = subprocess.run([tool, "info"], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"(npu-smi failed: {type(error).__name__}: {error})"
    return (done.stdout + done.stderr).rstrip()


def _rss_kb() -> dict[str, int]:
    values = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            name, _, rest = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                values[name] = int(rest.split()[0])
    except OSError:
        pass
    return values


def _stats(samples: list[float]) -> dict[str, float]:
    import numpy as np

    if not samples:
        return {"n": 0}
    data = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(data.size),
        "p50": round(float(np.percentile(data, 50)), 4),
        "p95": round(float(np.percentile(data, 95)), 4),
        "max": round(float(data.max()), 4),
    }


def _says_yes(text: str) -> bool:
    """Plan §4.1's reading of a yes/no answer: any negation is no; else it must start with 是."""
    stripped = text.strip()
    if any(mark in stripped for mark in ("不", "否", "没")):
        return False
    return stripped.startswith("是")


def _read(path: str):
    import cv2

    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read image {path}")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo-root", required=True)
    parser.add_argument("--image", action="append", required=True, help="repeatable")
    parser.add_argument(
        "--bbox", action="append", type=_parse_bbox, default=[], help="x1,y1,x2,y2 for the n-th --image"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="untimed passes before the measured runs")
    parser.add_argument("--no-gesture-image")
    parser.add_argument("--no-gesture-bbox", type=_parse_bbox)
    parser.add_argument("--allow-launch", action="store_true", help="run even if a ros2 launch is active")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup >= 0")
    if len(args.bbox) > len(args.image):
        parser.error("more --bbox than --image")

    running = _launch_running()
    if running and not args.allow_launch:
        print("a ros2 launch is running; this bench loads its own OMs onto the shared NPU:", file=sys.stderr)
        for line in running:
            print(f"  {line}", file=sys.stderr)
        return 2

    try:
        from visual_interaction.session.look_say import QUESTIONS
    except ImportError as error:
        print(f"look_say.QUESTIONS is not available: {error}", file=sys.stderr)
        return 2

    from visual_interaction.session.internvl_vision import InternVlVision

    images = []
    for index, path in enumerate(args.image):
        frame = _read(path)
        bbox = args.bbox[index] if index < len(args.bbox) else None
        synthetic = bbox is None
        if synthetic:
            bbox = _synthetic_bbox(frame.shape[1], frame.shape[0])
        images.append({"path": path, "frame": frame, "bbox": bbox, "synthetic": synthetic})

    report: dict[str, object] = {"demo_root": args.demo_root, "runs": args.runs, "warmup": args.warmup}
    report["npu_smi_before"] = _npu_smi()
    report["rss_kb_before"] = _rss_kb()

    vision = InternVlVision(demo_root=args.demo_root, frames=1, prompt_mode="chatter")
    vision.register_questions(QUESTIONS)
    started = time.perf_counter()
    vision.load()
    report["load_s"] = round(time.perf_counter() - started, 3)
    report["npu_smi_after"] = _npu_smi()
    report["rss_kb_after_load"] = _rss_kb()
    report["question_failures"] = dict(vision.question_failures)
    keys = [key for key in QUESTIONS if key not in vision.question_failures]

    prepare_s: dict[str, list[float]] = {"crop": [], "whole": []}
    ask_s: dict[str, list[float]] = {}
    ask_tokens: dict[str, list[int]] = {}
    fallbacks = 0
    answers: dict[str, dict[str, str]] = {}
    for run in range(args.warmup + args.runs):
        timed = run >= args.warmup
        for item in images:
            seen = {
                "crop": vision.prepare(item["frame"], item["bbox"], crop=True),
                "whole": vision.prepare(item["frame"], None, crop=False),
            }
            for view, picture in seen.items():
                if timed:
                    prepare_s[view].append(picture.seconds)
                    fallbacks += int(view == "crop" and picture.crop_fallback)
                for key in keys:
                    answer = vision.ask(picture, key)
                    if timed:
                        ask_s.setdefault(f"{key}/{view}", []).append(answer.seconds)
                        ask_tokens.setdefault(f"{key}/{view}", []).append(answer.tokens)
                    answers.setdefault(f"{Path(item['path']).name}/{view}", {})[key] = answer.text
    report["rss_kb_after_runs"] = _rss_kb()
    report["prepare"] = {view: _stats(values) for view, values in prepare_s.items()}
    report["crop_fallbacks"] = fallbacks
    report["ask"] = {
        name: {**_stats(values), "tokens_mean": round(sum(ask_tokens[name]) / len(ask_tokens[name]), 1)}
        for name, values in ask_s.items()
    }
    report["images"] = [
        {
            "path": item["path"],
            "size": list(item["frame"].shape[1::-1]),
            "bbox": list(item["bbox"]),
            "synthetic": item["synthetic"],
        }
        for item in images
    ]
    report["answers_last_run"] = answers

    yes_bias = None
    if args.no_gesture_image:
        frame = _read(args.no_gesture_image)
        bbox = args.no_gesture_bbox or _synthetic_bbox(frame.shape[1], frame.shape[0])
        yes_bias = {"image": args.no_gesture_image, "bbox": list(bbox), "synthetic": args.no_gesture_bbox is None}
        for view, crop in (("crop", True), ("whole", False)):
            picture = vision.prepare(frame, bbox if crop else None, crop=crop)
            asked = [key for key in keys if key.startswith(YES_NO_PREFIX) or key == "q1"]
            replies = {key: vision.ask(picture, key).text for key in asked}
            yes_bias[view] = {
                key: {"text": text, "yes": _says_yes(text)} if key.startswith(YES_NO_PREFIX) else {"text": text}
                for key, text in replies.items()
            }
    report["q3_no_gesture"] = yes_bias

    _print_text(report)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


def _print_text(report: dict) -> None:
    out = sys.stdout
    for name in ("npu_smi_before", "npu_smi_after"):
        text = report[name]
        print(f"== {name} ==", file=out)
        print(text if text is not None else "(npu-smi not found, skipped)", file=out)
    rss = {name: report[name] for name in ("rss_kb_before", "rss_kb_after_load", "rss_kb_after_runs")}
    print(f"== rss_kb == {rss}", file=out)
    print(f"== load == {report['load_s']:.2f} s   question failures: {report['question_failures'] or 'none'}", file=out)
    for item in report["images"]:
        mark = " (synthetic)" if item["synthetic"] else ""
        print(f"== image == {item['path']} {item['size']} bbox={item['bbox']}{mark}", file=out)
    print(f"{'step':<22}{'n':>5}{'p50 s':>9}{'p95 s':>9}{'max s':>9}{'tok':>6}", file=out)
    rows = [(f"prepare/{view}", stats, "") for view, stats in report["prepare"].items()]
    rows += [(f"ask {name}", stats, stats.get("tokens_mean", "")) for name, stats in report["ask"].items()]
    for name, stats, tokens in rows:
        if not stats.get("n"):
            print(f"{name:<22}{0:>5}", file=out)
            continue
        timing = f"{stats['p50']:>9.3f}{stats['p95']:>9.3f}{stats['max']:>9.3f}"
        print(f"{name:<22}{stats['n']:>5}{timing}{tokens!s:>6}", file=out)
    print(f"crop fallbacks: {report['crop_fallbacks']}", file=out)
    for picture, replies in report["answers_last_run"].items():
        print(f"-- {picture}: " + " | ".join(f"{key}={text}" for key, text in replies.items()), file=out)
    bias = report["q3_no_gesture"]
    if bias is not None:
        print(f"== q3 on no-gesture image (informational) == {bias['image']}", file=out)
        for view in ("crop", "whole"):
            print(
                f"-- {view}: "
                + " | ".join(
                    f"{key}={reply['text']}{' [YES]' if reply.get('yes') else ''}" for key, reply in bias[view].items()
                ),
                file=out,
            )


if __name__ == "__main__":
    sys.exit(main())
