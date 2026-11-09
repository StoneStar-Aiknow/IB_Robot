"""Offline key-action evaluation on recorded videos (plan v11 §4), run on the board.

Each video is one person following the cue track of key_eval_schedule. This
tool walks the video in time order and writes, per video, into ``--out``:

* ``<stem>.features.json``: duration, frame rate, and two numbers per 0.2 s
  (how much moved, and how far from the person's usual stance) that
  key_eval_score aligns the cue track with. Numbers only.
* ``<stem>.ticks.jsonl`` (``run`` only): one line every ``--tick`` seconds of
  video, with the person box used and what each method answered and how long
  it took. Text and numbers only.
* ``<stem>.embeds.npz`` (method ``A`` only): per tick, the vision features of
  the newest crop pooled to a 4×4 grid, for method C's nearest neighbours in
  key_eval_score. Derived from pictures of people: it stays on the board.
* ``<stem>/crops/`` (``--dump-every N`` only): every N-th tick's 448² crop, to
  check the box by eye. These are pictures of people: they stay on the board.

The person box comes from the same YOLOX OM the mirror's perception runs
(``--yolox``), taken as the largest person; where it finds nobody the last box
is kept for up to 2 s, then a fixed box in the middle of the frame is used
(``box_src`` says which). ``--no-yolox`` uses the fixed box throughout.

Methods (``--methods``):

* ``0``: look_say as it is — Q1 on the newest frame, Q2 when Q1 names a vague
  hand gesture, Q3's yes/no on a named key action. key_eval_score reads it
  both with and without Q3.
* ``A``: key_choice's multiple-choice question, scored from the first token.
  One vision OM run on four crops of the person, ``MULTI_OFFSETS`` apart
  (each cut with this tick's box), then the four-picture question (A) and the
  one-picture question on the newest crop alone (A1), each in both option
  orders. The header carries the same questions' answers on blank grey
  pictures, for calibration. Its pooled features feed method C.

``--stride METHOD=N`` runs a method on every N-th tick only (the baseline's
looks take ~1.5 s, so the replay never uses more than one tick in three).

Subcommands::

    python3 -m visual_interaction.session.key_eval features VIDEO... --out DIR
    python3 -m visual_interaction.session.key_eval run VIDEO... --out DIR \\
        --demo-root /root/vlx_internvl_310p --yolox YOLOX.om [--tick 0.5] [--limit-sec S] [--dump-every N]
    python3 -m visual_interaction.session.key_eval score DIR [...]   # see key_eval_score

``features`` needs no NPU and is the quick check of a fresh recording. ``run``
loads its own copy of the OMs onto the shared NPU, so like look_bench it
refuses while a ``ros2 launch`` is running unless ``--allow-launch``, and it
needs the node's environment (CANN ``set_env.sh``, the workspace's
``setup.bash`` for perception_service, the system site-packages on
PYTHONPATH).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

BIN_SEC = 0.2
THUMB_WIDTH = 80
# The usual stance: the median of one thumbnail a second over this window.
STANCE_WINDOW_SEC = 60.0
HOLD_BOX_SEC = 2.0
MIN_PERSON_CONFIDENCE = 0.3
DEFAULT_TICK_SEC = 0.5
# Method A's pictures, seconds before the tick, oldest first.
MULTI_OFFSETS = (0.75, 0.5, 0.25, 0.0)
HISTORY_SEC = 1.0
POOL_GRID = 4


def frames(path: Path) -> Iterator[tuple[int, float, object]]:
    """``(index, seconds, bgr)`` for each frame, in order, by the container's clock."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    index, last = 0, -1.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if seconds <= last:  # a container without usable timestamps
                seconds = index / fps if fps > 0 else last + 1 / 30
            last = seconds
            yield index, seconds, frame
            index += 1
    finally:
        capture.release()


class Features:
    """Collects the alignment numbers while the frames go by."""

    def __init__(self) -> None:
        self.motion: dict[int, list[float]] = {}
        self.thumbs: dict[int, object] = {}
        self._previous = None
        self.count = 0
        self.last_t = 0.0
        self.size: tuple[int, int] | None = None

    def add(self, seconds: float, frame) -> None:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        self.size = (width, height)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(
            gray, (THUMB_WIDTH, max(1, round(THUMB_WIDTH * height / width))), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        k = int(seconds / BIN_SEC)
        if self._previous is not None:
            self.motion.setdefault(k, []).append(float(np.abs(thumb - self._previous).mean()))
        self._previous = thumb
        self.thumbs.setdefault(k, thumb)
        self.count += 1
        self.last_t = seconds

    def result(self, fps: float) -> dict:
        import numpy as np

        n = int(self.last_t / BIN_SEC) + 1
        motion = [float(np.mean(self.motion[k])) if k in self.motion else 0.0 for k in range(n)]
        # Fill bins without a frame (a slow stream) from the one before.
        thumbs, held = [], None
        for k in range(n):
            held = self.thumbs.get(k, held)
            thumbs.append(held)
        per_sec = round(1.0 / BIN_SEC)
        half = int(STANCE_WINDOW_SEC / 2 / BIN_SEC)
        dev, stance, stance_at = [], None, None
        for k in range(n):
            centre = (k // per_sec) * per_sec
            if centre != stance_at:
                picks = [
                    thumbs[j]
                    for j in range(max(0, centre - half), min(n, centre + half), per_sec)
                    if thumbs[j] is not None
                ]
                stance = np.median(np.stack(picks), axis=0) if picks else None
                stance_at = centre
            dev.append(
                float(np.abs(thumbs[k] - stance).mean()) if stance is not None and thumbs[k] is not None else 0.0
            )
        return {
            "bin_sec": BIN_SEC,
            "frames": self.count,
            "fps": round(fps, 3),
            "duration": round(self.last_t, 3),
            "size": list(self.size) if self.size else None,
            "motion": [round(v, 3) for v in motion],
            "dev": [round(v, 3) for v in dev],
        }


def _fps(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# The person box


class YoloxPeople:
    """The mirror's YOLOX OM, run directly: frame in, person boxes out (largest first)."""

    def __init__(self, om_path: str) -> None:
        # The demo's session helpers, on sys.path once InternVlVision has loaded.
        from run_llm_om import om_backend, session

        from perception_service.yolox_adapter import YoloXAdapter

        sess = session(om_path)
        inputs = [item.name for item in sess.get_inputs()]
        if len(inputs) != 1:
            raise SystemExit(f"YOLOX OM has inputs {inputs}, expected one")
        self._input = inputs[0]
        self._run = om_backend(sess, ["output"])
        self._adapter = YoloXAdapter()

    def __call__(self, frame_bgr) -> list[tuple[float, tuple[float, float, float, float]]]:
        height, width = frame_bgr.shape[:2]
        feed = self._adapter.preprocess(frame_bgr[:, :, ::-1])  # it takes RGB
        raw = self._run({self._input: feed["observation.image"]})[0]
        found = self._adapter._decode(raw, (height, width), min(640.0 / height, 640.0 / width))
        people = [
            (d.confidence, tuple(float(v) for v in d.bbox_xyxy)) for d in found if d.confidence >= MIN_PERSON_CONFIDENCE
        ]
        return sorted(people, key=lambda p: -(p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))


def fixed_box(width: int, height: int) -> tuple[float, float, float, float]:
    """look_bench's stand-in: a standing person in the middle of the frame."""
    return 0.35 * width, 0.15 * height, 0.65 * width, 0.95 * height


class Boxes:
    """The box for each tick: detected, held from a recent tick, or fixed."""

    def __init__(self, detect: Callable | None) -> None:
        self._detect = detect
        self._last: tuple[float, tuple] | None = None

    def __call__(self, seconds: float, frame) -> tuple[tuple[float, float, float, float], str, float | None]:
        height, width = frame.shape[:2]
        if self._detect is not None:
            people = self._detect(frame)
            if people:
                confidence, box = people[0]
                self._last = (seconds, box)
                return box, "yolox", round(confidence, 3)
            if self._last is not None and seconds - self._last[0] <= HOLD_BOX_SEC:
                return self._last[1], "held", None
        return fixed_box(width, height), "fixed", None


# ---------------------------------------------------------------------------
# Methods


class Baseline:
    """Method 0: look_say's three questions, as the node asks them."""

    name = "0"

    def __init__(self, vision) -> None:
        from visual_interaction.session.look_say import QUESTIONS

        self._vision = vision
        vision.register_questions(QUESTIONS)

    def __call__(self, frame, box, history=(), source_size=None) -> tuple[dict, object]:
        from visual_interaction.session import look_say

        vision = self._vision
        seen = vision.prepare(frame, box, source_size=source_size)
        q1 = vision.ask(seen, look_say.Q1)
        label = look_say.classify(q1.text)
        rec = {
            "prepare_s": round(seen.seconds, 4),
            "fallback": seen.crop_fallback,
            "q1": q1.text,
            "q1_s": round(q1.seconds, 4),
        }
        if look_say.needs_refine(label):
            q2 = vision.ask(seen, look_say.Q2)
            label = look_say.refine(label, q2.text)
            rec.update(q2=q2.text, q2_s=round(q2.seconds, 4))
        rec["label"] = label.tag if label is not None else None
        if label is not None and label.named:
            q3 = vision.ask(seen, label.verify_key)
            rec.update(q3=q3.text, q3_s=round(q3.seconds, 4), yes=look_say.parse_yes(q3.text))
        return rec, seen.image


def nearest(history: Sequence[tuple[float, object]], seconds: float) -> tuple[float, object]:
    """The ``(t, frame)`` of ``history`` closest to ``seconds``."""
    return min(history, key=lambda item: abs(item[0] - seconds))


def pooled(embeds) -> object:
    """One picture's ``(IMAGE_TOKENS, hidden)`` features as a POOL_GRID² grid, float16."""
    import numpy as np

    side = int(round(len(embeds) ** 0.5))
    grid = np.asarray(embeds, np.float32).reshape(side, side, -1)
    step = side // POOL_GRID
    blocks = grid.reshape(POOL_GRID, step, POOL_GRID, step, -1).mean(axis=(1, 3))
    return blocks.reshape(POOL_GRID * POOL_GRID, -1).astype(np.float16)


class Choices:
    """Method A (and A1): key_choice's question, scored, see the module docstring."""

    name = "A"

    def __init__(self, vision) -> None:
        from visual_interaction.session import key_choice

        self._vision = vision
        self._questions = key_choice.questions(frames=(1, len(MULTI_OFFSETS)))
        vision.register_choices(self._questions)

    def _ask(self, seen, frames: int) -> dict:
        from visual_interaction.session import key_choice

        out = {}
        for order in key_choice.ORDERS:
            choice = self._vision.score(seen, key_choice.key(frames, order))
            out[order] = {
                "p": {a: round(p, 5) for a, p in key_choice.by_answer(choice.probs, order).items()},
                "mass": round(choice.mass, 4),
                "top": choice.top,
                "s": round(choice.seconds, 4),
            }
        return out

    def header(self) -> dict:
        """The answers on blank grey pictures: what the question says about nothing."""
        import numpy as np

        from visual_interaction.session.internvl_vision import Seen

        grey = np.full((448, 448, 3), 128, np.uint8)
        seen = self._vision.prepare_many([grey] * len(MULTI_OFFSETS))
        one = Seen(embeds=seen.embeds[-1:], image=grey, crop_fallback=False, seconds=0.0)
        return {"blank": {"4": self._ask(seen, len(MULTI_OFFSETS)), "1": self._ask(one, 1)}}

    def __call__(self, frame, box, history=(), source_size=None) -> tuple[dict, object]:
        from visual_interaction.session import look_crop
        from visual_interaction.session.internvl_vision import Seen

        now = history[-1][0] if history else 0.0
        picks = (
            [nearest(history, now - offset) for offset in MULTI_OFFSETS]
            if history
            else [(now, frame)] * len(MULTI_OFFSETS)
        )
        started = time.perf_counter()
        crops = [look_crop.crop_upper_body(picked, box) for _, picked in picks]
        seen = self._vision.prepare_many([crop.image for crop in crops])
        vision_s = time.perf_counter() - started
        newest = Seen(embeds=seen.embeds[-1:], image=seen.image, crop_fallback=crops[-1].fallback, seconds=0.0)
        rec = {
            "vision_s": round(vision_s, 4),
            "fallback": sum(crop.fallback for crop in crops),
            "dt": [round(t - now, 3) for t, _ in picks],
            "4": self._ask(seen, len(MULTI_OFFSETS)),
            "1": self._ask(newest, 1),
            "_pooled": pooled(seen.embeds[-1]),
        }
        return rec, seen.image


METHODS = {"0": Baseline, "A": Choices}


# ---------------------------------------------------------------------------
# One video


def process(
    path: Path,
    out: Path,
    methods: Sequence | None = None,
    boxes: Boxes | None = None,
    tick_sec: float = DEFAULT_TICK_SEC,
    limit_sec: float | None = None,
    dump_every: int = 0,
    log=print,
) -> dict:
    """Features for ``path``, and with ``methods`` its ticks file. Returns the features."""
    from collections import deque

    import cv2
    import numpy as np

    stem = path.stem
    features = Features()
    methods = list(methods or [])
    ticks_file = None
    started = time.perf_counter()
    if methods:
        ticks_file = (out / f"{stem}.ticks.jsonl").open("w", encoding="utf-8")
        header = {
            "kind": "header",
            "video": path.name,
            "tick_sec": tick_sec,
            "methods": [m.name for m in methods],
            "strides": {m.name: getattr(m, "stride", 1) for m in methods},
            "multi_offsets": list(MULTI_OFFSETS),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "calibration": {m.name: m.header() for m in methods if hasattr(m, "header")},
        }
        ticks_file.write(json.dumps(header, ensure_ascii=False) + "\n")
    next_tick, count = 0.0, 0
    history: deque = deque()
    embeds_t, embeds = [], []
    try:
        for _, seconds, frame in frames(path):
            if limit_sec is not None and seconds > limit_sec:
                break
            features.add(seconds, frame)
            if not methods:
                continue
            history.append((seconds, frame))
            while seconds - history[0][0] > HISTORY_SEC + 0.1:
                history.popleft()
            if seconds + 1e-6 < next_tick:
                continue
            box, source, confidence = boxes(seconds, frame) if boxes else (None, "none", None)
            row = {
                "kind": "tick",
                "t": round(seconds, 3),
                "box": [round(v, 1) for v in box] if box else None,
                "box_src": source,
                "box_conf": confidence,
                "methods": {},
            }
            for method in methods:
                if count % getattr(method, "stride", 1):
                    continue
                try:
                    rec, image = method(frame, box, tuple(history))
                except Exception as error:  # noqa: BLE001 - recorded per tick, the run goes on
                    rec, image = {"error": f"{type(error).__name__}: {error}"}, None
                grid = rec.pop("_pooled", None)
                if grid is not None:
                    embeds_t.append(seconds)
                    embeds.append(grid)
                row["methods"][method.name] = rec
                if dump_every and count % dump_every == 0 and image is not None:
                    crops = out / stem / "crops"
                    crops.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crops / f"{seconds:08.2f}_{method.name}.jpg"), image)
            ticks_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            next_tick = (int(seconds / tick_sec) + 1) * tick_sec
            if count % 100 == 0:
                ticks_file.flush()
                log(f"{stem}: {seconds:.0f} s of video, {count} ticks, {time.perf_counter() - started:.0f} s")
    finally:
        if ticks_file is not None:
            ticks_file.write(
                json.dumps({"kind": "end", "ticks": count, "seconds": round(time.perf_counter() - started, 1)}) + "\n"
            )
            ticks_file.close()
        if embeds:
            np.savez(out / f"{stem}.embeds.npz", t=np.array(embeds_t), grid=np.stack(embeds))
    result = features.result(_fps(path))
    (out / f"{stem}.features.json").write_text(json.dumps(result), encoding="utf-8")
    log(
        f"{stem}: {result['frames']} frames, {result['duration']:.1f} s, {result['fps']} fps, size {result['size']}"
        + (f", {count} ticks" if methods else "")
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["score"]:
        from visual_interaction.session import key_eval_score

        return key_eval_score.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("features", "run"):
        p = sub.add_parser(name)
        p.add_argument("videos", nargs="+", type=Path)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--limit-sec", type=float, help="stop each video after this many seconds")
    run = sub.choices["run"]
    run.add_argument("--demo-root", required=True)
    run.add_argument("--yolox", help="the YOLOX person OM")
    run.add_argument("--no-yolox", action="store_true", help="use the fixed middle box throughout")
    run.add_argument("--methods", default="0", help=f"comma list of {sorted(METHODS)}")
    run.add_argument("--stride", action="append", default=[], help="METHOD=N: run it on every N-th tick only")
    run.add_argument("--tick", type=float, default=DEFAULT_TICK_SEC)
    run.add_argument("--dump-every", type=int, default=0, help="save every N-th tick's crop (stays on the board)")
    run.add_argument("--allow-launch", action="store_true")
    sub.add_parser("score", help="see key_eval_score --help")
    args = parser.parse_args(argv)
    missing = [str(v) for v in args.videos if not v.is_file()]
    if missing:
        parser.error(f"no such video: {', '.join(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "features":
        for video in args.videos:
            process(video, args.out, limit_sec=args.limit_sec)
        return 0

    if not args.yolox and not args.no_yolox:
        parser.error("give --yolox OM or --no-yolox")
    names = [m for m in args.methods.split(",") if m]
    unknown = [m for m in names if m not in METHODS]
    if unknown:
        parser.error(f"unknown methods {unknown}")
    strides = {}
    for item in args.stride:
        name, _, value = item.partition("=")
        if name not in names or not value.isdigit() or int(value) < 1:
            parser.error(f"--stride wants METHOD=N for one of {names}, got {item!r}")
        strides[name] = int(value)
    from visual_interaction.session.look_bench import _launch_running

    running = _launch_running()
    if running and not args.allow_launch:
        print("a ros2 launch is running; key_eval loads its own OMs onto the shared NPU:", file=sys.stderr)
        for line in running:
            print(f"  {line}", file=sys.stderr)
        return 2

    from visual_interaction.session.internvl_vision import InternVlVision

    vision = InternVlVision(demo_root=args.demo_root, frames=1, prompt_mode="chatter")
    methods = [METHODS[name](vision) for name in names]
    for method in methods:
        method.stride = strides.get(method.name, 1)
    vision.load()
    if vision.question_failures:
        print(f"questions refused at load: {vision.question_failures}", file=sys.stderr)
        return 2
    boxes = Boxes(None if args.no_yolox else YoloxPeople(args.yolox))
    for video in args.videos:
        process(video, args.out, methods, boxes, args.tick, args.limit_sec, args.dump_every)
        boxes = Boxes(boxes._detect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
