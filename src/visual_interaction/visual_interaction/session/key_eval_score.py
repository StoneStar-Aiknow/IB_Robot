"""Scoring the key-action evaluation: alignment, labels, the gate, the numbers.

The second half of key_eval (plan v11 §4). Pure Python and numpy, no model and
no video: it reads what ``key_eval run`` wrote for each video — a features file
(motion numbers only) and a ticks file (one line per look) — so it also runs on
a workstation from the statistics alone.

**Alignment.** A video's labels follow from where in it the cue track started
(and, should the mp4's clock be off, how fast it ran). Two numbers per 0.2 s
bin describe the video: ``motion``, how much changed since the previous frame,
and ``dev``, how far the frame is from the person's usual stance (a running
median). Holding a gesture keeps ``dev`` up and moves little, so ``dev`` is
matched against the hold windows and ``motion`` against the moments the hands
go up and come down. The best offset is kept only if it clearly beats the best
offset more than 2 s away from it (``runner_up`` is that one's score as a
fraction of the best); otherwise the video is refused and reported, and an
offset can be given by hand.

**Cadence.** ``run`` looks every ``tick`` seconds of video, whatever a look
costs. Online the looks run back to back, so a look that takes 1.4 s is
followed by one 1.4 s later. The replay here walks the ticks that way: each
look starts when the previous one finished, uses the tick nearest that moment,
and its scores reach the gate when it finishes.

**Counting.** A trigger is judged by the frame its look saw:

* inside a key slot's window (``HIT_EARLY`` before the high beep to
  ``HIT_LATE`` after the low one), it is a hit if it names that slot's
  action, else a wrong label;
* anywhere else in the cue track it is a false trigger;
* before or after the cue track it is not counted.

A slot is recalled when a hit's look *finished* within ``RECALL_SEC`` of the
high beep; ``recall_window`` counts any hit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from visual_interaction.session import key_eval_schedule as schedule
from visual_interaction.session.key_actions import KEY_ACTIONS, GateConfig, KeyGate

HIT_EARLY = 0.5
HIT_LATE = 1.0
RECALL_SEC = 2.5

# Plan v11 §4's bar for keeping an action.
PASS_RECALL = 0.8
PASS_FALSE_PER_MIN = 0.5
PASS_WRONG = 0.02
PASS_LOOK_P95_SEC = 0.8

MAX_OFFSET_SEC = 40.0
RATES = tuple(round(0.97 + 0.005 * i, 3) for i in range(13))
MAX_RUNNER_UP = 0.85
# Neighbours of the best offset closer than this are the same peak.
PEAK_HALF_WIDTH_SEC = 2.0


# ---------------------------------------------------------------------------
# Alignment


@dataclass(frozen=True)
class Alignment:
    """Where the cue track sits in one video: video = offset + rate × cue time."""

    version: int
    offset: float
    rate: float = 1.0
    score: float = 0.0
    runner_up: float = 0.0
    manual: bool = False

    def video(self, cue_sec: float) -> float:
        return self.offset + self.rate * cue_sec

    def label(self, video_sec: float) -> tuple[str, schedule.Slot | None]:
        cue = (video_sec - self.offset) / self.rate
        return schedule.label_at(cue, 0.0, self.version)

    def windows(self) -> list[tuple[float, float, schedule.Slot]]:
        """``(go, down, slot)`` in video seconds."""
        return [(self.video(go), self.video(down), s) for go, down, s in schedule.move_windows(0.0, self.version)]

    def span(self) -> tuple[float, float]:
        return self.video(schedule.LEAD_IN_SEC), self.video(schedule.total_sec(self.version))


def _robust_z(values):
    import numpy as np

    x = np.asarray(values, np.float64)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return np.clip((x - med) / (mad + 1e-9), -3.0, 6.0)


def _templates(version: int, rate: float, bin_sec: float, n: int):
    """Hold and transition indicators for a cue track starting at bin 0, ``n`` bins long."""
    import numpy as np

    hold, moves = np.zeros(n), np.zeros(n)

    def mark(array, a, b):
        i, j = int(round(a / bin_sec)), int(round(b / bin_sec))
        array[max(i, 0) : max(min(j, n), 0)] = 1.0

    for go, down, _ in schedule.move_windows(0.0, version):
        go, down = go * rate, down * rate
        mark(hold, go + 0.5, down + 0.3)
        mark(moves, go, go + 1.0)
        mark(moves, down, down + 1.0)
    return hold, moves


def _pearson(feature, template) -> float:
    import numpy as np

    t = template - template.mean()
    denom = np.sqrt((t * t).sum()) * np.sqrt((feature * feature).sum())
    return float((feature * t).sum() / denom) if denom > 0 else 0.0


def align(
    features: Mapping[str, object],
    versions: Sequence[int] = schedule.VERSIONS,
    rates: Sequence[float] = RATES,
    max_offset: float = MAX_OFFSET_SEC,
) -> tuple[Alignment, list[Alignment]]:
    """The best alignment, and each version's best (for the report)."""
    import numpy as np

    bin_sec = float(features["bin_sec"])
    dev = _robust_z(features["dev"])
    motion = _robust_z(features["motion"])
    dev, motion = dev - dev.mean(), motion - motion.mean()
    n = len(dev)
    shifts = int(round(max_offset / bin_sec)) + 1
    per_version = []
    for version in versions:
        best = None
        for rate in rates:
            hold, moves = _templates(version, rate, bin_sec, n)
            scores = np.empty(shifts)
            for k in range(shifts):
                if k:
                    h, m = np.zeros(n), np.zeros(n)
                    h[k:], m[k:] = hold[: n - k], moves[: n - k]
                else:
                    h, m = hold, moves
                scores[k] = _pearson(dev, h) + _pearson(motion, m)
            k = int(scores.argmax())
            if best is None or scores[k] > best[0]:
                width = int(round(PEAK_HALF_WIDTH_SEC / bin_sec))
                others = np.concatenate((scores[: max(k - width, 0)], scores[k + width + 1 :]))
                second = float(others.max()) if others.size else 0.0
                best = (float(scores[k]), k, rate, second)
        score, k, rate, second = best
        runner_up = second / score if score > 0 else 1.0
        per_version.append(Alignment(version, k * bin_sec, rate, round(score, 4), round(runner_up, 4)))
    return max(per_version, key=lambda a: a.score), per_version


# ---------------------------------------------------------------------------
# What each method's look said, from a tick record

# look_say's label tags for the key actions; the rest are not key actions.
LABEL_ACTIONS = {
    "hand/heart": "heart",
    "hand/thumb": "thumb",
    "hand/yeah": "yeah",
    "big/wave": "wave",
    "big/open": "open",
}


def _baseline(verify: bool) -> Callable[[Mapping], tuple[dict, float] | None]:
    """Method 0: look_say's Q1 (and Q2) label, after Q3's yes with ``verify``."""

    def view(tick: Mapping) -> tuple[dict, float] | None:
        rec = tick.get("methods", {}).get("0")
        if rec is None or "error" in rec:
            return None
        seconds = rec["prepare_s"] + rec["q1_s"] + rec.get("q2_s", 0.0)
        action = LABEL_ACTIONS.get(rec.get("label") or "")
        if verify:
            seconds += rec.get("q3_s", 0.0)
            if not rec.get("yes"):
                action = None
        return ({action: 1.0} if action else {}), seconds

    return view


VIEWS: dict[str, Callable[[Mapping], tuple[dict, float] | None]] = {
    "0": _baseline(verify=True),
    "0nv": _baseline(verify=False),
}


@dataclass(frozen=True)
class Look:
    frame_t: float
    decided: float
    scores: Mapping[str, float]
    seconds: float


def replay(ticks: Sequence[Mapping], view: Callable, tick_sec: float) -> list[Look]:
    """The looks the node would have made, back to back (module docstring, cadence)."""
    looks: list[Look] = []
    ticks = sorted(ticks, key=lambda tick: tick["t"])
    if not ticks:
        return looks
    cursor, i = ticks[0]["t"], 0
    while i < len(ticks):
        while i < len(ticks) and ticks[i]["t"] < cursor - tick_sec / 2:
            i += 1
        if i == len(ticks):
            break
        out = view(ticks[i])
        if out is not None:
            scores, seconds = out
            decided = max(cursor, ticks[i]["t"]) + seconds
            looks.append(Look(ticks[i]["t"], decided, scores, seconds))
            cursor = decided
        i += 1
    return looks


# ---------------------------------------------------------------------------
# Counting


@dataclass
class SlotOutcome:
    variant: str
    label: str
    go: float
    first: str | None = None  # first action triggered in the window, if any
    latency: float | None = None  # of the first hit, from the high beep to the look's end
    wrong: list = field(default_factory=list)


@dataclass
class VideoResult:
    stem: str
    person: str
    slots: list
    false: list  # (action, frame_t, variant)
    none_minutes: float
    look_seconds: list
    triggers: int


def person_of(stem: str) -> str:
    return stem.split("_")[0]


def evaluate_video(stem: str, alignment: Alignment, looks: Sequence[Look], gate: GateConfig) -> VideoResult:
    keeper = KeyGate(gate)
    fired = []
    for look in looks:
        trigger = keeper.update(look.decided, look.scores)
        if trigger is not None:
            fired.append((trigger.action, look.frame_t, look.decided))
    start, end = alignment.span()
    windows = [(go, down, s) for go, down, s in alignment.windows() if s.label in KEY_ACTIONS]
    slots = [SlotOutcome(s.variant, s.label, go) for go, down, s in windows]
    false = []
    for action, frame_t, decided in fired:
        hit = None
        for index, (go, down, _) in enumerate(windows):
            if go - HIT_EARLY <= frame_t <= down + HIT_LATE:
                hit = index
                break
        if hit is not None:
            outcome = slots[hit]
            outcome.first = outcome.first or action
            if action == outcome.label:
                if outcome.latency is None:
                    outcome.latency = round(decided - outcome.go, 3)
            else:
                outcome.wrong.append(action)
            continue
        label, slot = alignment.label(frame_t)
        if label == "" or not (start <= frame_t < end):
            continue
        variant = slot.variant if slot.label == schedule.NONE else f"{slot.variant}/idle"
        false.append((action, round(frame_t, 2), variant))
    keyed = sum((down + HIT_LATE) - (go - HIT_EARLY) for go, down, _ in windows)
    return VideoResult(
        stem=stem,
        person=person_of(stem),
        slots=slots,
        false=false,
        none_minutes=max(end - start - keyed, 0.0) / 60.0,
        look_seconds=[look.seconds for look in looks],
        triggers=len(fired),
    )


def _pct(values: Iterable[float], q: float) -> float | None:
    import numpy as np

    data = list(values)
    return round(float(np.percentile(data, q)), 3) if data else None


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def summarize(results: Sequence[VideoResult]) -> dict:
    """The plan's numbers over ``results``: per action, per variant, per person, confusion."""
    none_min = sum(r.none_minutes for r in results)
    slots = [slot for r in results for slot in r.slots]
    false = [item for r in results for item in r.false]
    looks = [s for r in results for s in r.look_seconds]
    actions = {}
    for action in KEY_ACTIONS:
        own = [s for s in slots if s.label == action]
        if not own:
            continue
        others = [s for s in slots if s.label != action]
        hits = [s for s in own if s.latency is not None]
        quick = [s for s in hits if s.latency <= RECALL_SEC]
        wrong_into = sum(1 for s in others if action in s.wrong)
        false_n = sum(1 for item in false if item[0] == action)
        fired = len(hits) + wrong_into + false_n
        precision = _ratio(len(hits), fired)
        recall = _ratio(len(quick), len(own))
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
        false_rate = round(false_n / none_min, 3) if none_min else None
        wrong_rate = _ratio(wrong_into, len(others)) or 0.0
        actions[action] = {
            "slots": len(own),
            "recall": recall,
            "recall_window": _ratio(len(hits), len(own)),
            "latency_p50": _pct((s.latency for s in hits), 50),
            "latency_p95": _pct((s.latency for s in hits), 95),
            "false": false_n,
            "false_per_min": false_rate,
            "wrong_into": wrong_into,
            "wrong_rate": wrong_rate,
            "precision": precision,
            "f1": f1,
            "pass": bool(
                recall is not None
                and recall >= PASS_RECALL
                and false_rate is not None
                and false_rate <= PASS_FALSE_PER_MIN
                and wrong_rate <= PASS_WRONG
            ),
        }
    variants = defaultdict(lambda: [0, 0, 0])
    for s in slots:
        row = variants[s.variant]
        row[0] += 1
        row[1] += int(s.latency is not None and s.latency <= RECALL_SEC)
        row[2] += int(s.latency is not None)
    people = {}
    for person in sorted({r.person for r in results}):
        mine = [r for r in results if r.person == person]
        minutes = sum(r.none_minutes for r in mine)
        own = [s for r in mine for s in r.slots]
        people[person] = {
            "videos": len(mine),
            "recall": {
                a: _ratio(
                    sum(1 for s in own if s.label == a and s.latency is not None and s.latency <= RECALL_SEC),
                    sum(1 for s in own if s.label == a),
                )
                for a in KEY_ACTIONS
                if any(s.label == a for s in own)
            },
            "false_per_min": round(sum(len(r.false) for r in mine) / minutes, 3) if minutes else None,
        }
    confusion = {a: Counter(s.first or "-" for s in slots if s.label == a) for a in KEY_ACTIONS}
    return {
        "videos": len(results),
        "none_minutes": round(none_min, 2),
        "looks": len(looks),
        "look_p50": _pct(looks, 50),
        "look_p95": _pct(looks, 95),
        "look_pass": bool(looks) and _pct(looks, 95) <= PASS_LOOK_P95_SEC,
        "false_per_min": round(len(false) / none_min, 3) if none_min else None,
        "wrong_slots": _ratio(sum(1 for s in slots if s.wrong), len(slots)),
        "actions": actions,
        "variants": {
            name: {"slots": v[0], "recall": _ratio(v[1], v[0]), "recall_window": _ratio(v[2], v[0])}
            for name, v in sorted(variants.items())
        },
        "false_by_variant": dict(Counter(item[2] for item in false).most_common()),
        "false_by_action": dict(Counter(item[0] for item in false).most_common()),
        "people": people,
        "confusion": {a: dict(c) for a, c in confusion.items() if c},
    }


# ---------------------------------------------------------------------------
# Files and the command line


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    header, ticks = {}, []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "header":
                header = row
            elif row.get("kind") == "tick":
                ticks.append(row)
    return header, ticks


def _parse_offsets(items: Sequence[str]) -> dict[str, tuple[float, int | None]]:
    out = {}
    for item in items:
        stem, _, value = item.partition("=")
        seconds, _, version = value.partition(":")
        if not stem or not seconds:
            raise argparse.ArgumentTypeError(f"--offset wants STEM=SECONDS[:VERSION], got {item!r}")
        out[stem] = (float(seconds), int(version) if version else None)
    return out


def _floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def _md(report: dict) -> str:
    lines = ["# 关键动作离线评测", ""]
    lines += [
        "## 对齐",
        "",
        "| 录像 | 版本 | 起点 s | 速率 | 峰值 | 次峰/峰 | 用了吗 |",
        "|---|---|---|---|---|---|---|",
    ]
    for stem, a in report["alignment"].items():
        used = "手给" if a["manual"] else ("是" if a["accepted"] else "**否**")
        lines.append(
            f"| {stem} | v{a['version']} | {a['offset']:.1f} | {a['rate']:.3f} | {a['score']:.3f} "
            f"| {a['runner_up']:.2f} | {used} |"
        )
    for run in report["runs"]:
        s = run["summary"]
        lines += ["", f"## 方法 {run['view']}，确认 {run['confirm']} 次，阈值 {run['threshold']}", ""]
        lines.append(
            f"录像 {s['videos']} 条，负样本时间 {s['none_minutes']} 分钟，look {s['looks']} 次，"
            f"耗时 p50 {s['look_p50']} s / p95 {s['look_p95']} s；"
            f"误报合计 {s['false_per_min']} 次/分钟；说错的格 {s['wrong_slots']}。"
        )
        lines += [
            "",
            "| 动作 | 格数 | 召回(2.5s) | 召回(整格) | 延迟p50 | 延迟p95 | 误报/分 | 错认别的格 | 精确 | F | 达标 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for action, a in s["actions"].items():
            lines.append(
                f"| {action} | {a['slots']} | {a['recall']} | {a['recall_window']} | {a['latency_p50']} "
                f"| {a['latency_p95']} | {a['false_per_min']} | {a['wrong_into']} ({a['wrong_rate']}) "
                f"| {a['precision']} | {a['f1']} | {'是' if a['pass'] else '否'} |"
            )
        lines += [
            "",
            "按变体（召回 2.5s / 整格）："
            + "；".join(f"{name} {v['recall']}/{v['recall_window']}" for name, v in s["variants"].items()),
        ]
        if s["false_by_variant"]:
            lines.append("误报出在：" + "；".join(f"{k} {v}" for k, v in s["false_by_variant"].items()))
        lines.append(
            "混淆（格的动作 → 第一次触发）："
            + "；".join(f"{a}: " + ", ".join(f"{k} {v}" for k, v in c.items()) for a, c in s["confusion"].items())
        )
        lines.append(
            "按人："
            + "；".join(
                f"{p} 误报 {v['false_per_min']}/分，召回 " + ", ".join(f"{a} {r}" for a, r in v["recall"].items())
                for p, v in s["people"].items()
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="key_eval score", description="Score key_eval run output (plan v11 §4).")
    parser.add_argument("dir", type=Path, help="key_eval run's --out directory")
    parser.add_argument("--name", default="report", help="writes <dir>/<name>.json and .md")
    parser.add_argument("--version", default="auto", choices=("auto", *map(str, schedule.VERSIONS)))
    parser.add_argument("--offset", action="append", default=[], help="STEM=SECONDS[:VERSION], repeatable")
    parser.add_argument("--max-runner-up", type=float, default=MAX_RUNNER_UP)
    parser.add_argument("--views", default=",".join(VIEWS), help=f"comma list of {sorted(VIEWS)}")
    parser.add_argument("--confirm", default="1,2")
    parser.add_argument("--threshold", default="0.5", help="comma list, one gate per value (all actions)")
    parser.add_argument("--actions", default=",".join(KEY_ACTIONS), help="the actions the gate may trigger")
    parser.add_argument("--release", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=8.0)
    args = parser.parse_args(argv)
    views = [view for view in args.views.split(",") if view]
    unknown = [view for view in views if view not in VIEWS]
    if unknown:
        parser.error(f"unknown views {unknown}")
    offsets = _parse_offsets(args.offset)
    versions = schedule.VERSIONS if args.version == "auto" else (int(args.version),)

    report: dict = {"alignment": {}, "runs": [], "refused": []}
    videos = []
    for path in sorted(args.dir.glob("*.features.json")):
        stem = path.name[: -len(".features.json")]
        features = json.loads(path.read_text(encoding="utf-8"))
        if stem in offsets:
            seconds, version = offsets[stem]
            alignment = Alignment(version or versions[-1], seconds, 1.0, manual=True)
            accepted = True
        else:
            alignment, _ = align(features, versions)
            accepted = alignment.runner_up <= args.max_runner_up
        report["alignment"][stem] = {**alignment.__dict__, "accepted": accepted, "duration": features.get("duration")}
        if not accepted:
            report["refused"].append(stem)
            continue
        ticks_path = args.dir / f"{stem}.ticks.jsonl"
        if ticks_path.exists():
            header, ticks = read_jsonl(ticks_path)
            videos.append((stem, alignment, ticks, float(header.get("tick_sec", 0.5))))

    allowed = [a for a in args.actions.split(",") if a]
    for view in views:
        for confirm in (int(c) for c in args.confirm.split(",") if c):
            for threshold in _floats(args.threshold):
                gate = GateConfig({a: threshold for a in allowed}, confirm, args.release, args.cooldown)
                results = [
                    evaluate_video(stem, alignment, replay(ticks, VIEWS[view], tick_sec), gate)
                    for stem, alignment, ticks, tick_sec in videos
                ]
                if results:
                    report["runs"].append(
                        {"view": view, "confirm": confirm, "threshold": threshold, "summary": summarize(results)}
                    )
    (args.dir / f"{args.name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    text = _md(report)
    (args.dir / f"{args.name}.md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    if report["refused"]:
        print(f"refused (alignment unclear, give --offset): {', '.join(report['refused'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
