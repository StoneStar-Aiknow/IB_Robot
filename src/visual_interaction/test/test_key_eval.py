"""The offline key-action evaluation: schedule, gate, alignment, replay, counting, files."""

import json

import numpy as np
import pytest

from visual_interaction.session import key_eval, key_eval_score
from visual_interaction.session import key_eval_schedule as schedule
from visual_interaction.session.key_actions import KEY_ACTIONS, GateConfig, KeyGate
from visual_interaction.session.key_eval_score import Alignment, Look, align, evaluate_video, replay, summarize


class TestSchedule:
    def test_v2_is_v1_with_the_standard_slots_retimed(self):
        v1, v2 = schedule.slots(1), schedule.slots(2)
        assert [(s.variant, s.cue, s.go, s.down) for s in v1] == [(s.variant, s.cue, s.go, s.down) for s in v2]
        assert {s.length for s in v1 if s.beeps and s.down == 7.0} == {9.5}
        lengths = [s.length for s in v2 if s.beeps and s.down == 7.0]
        assert len(lengths) == 41 and min(lengths) == 9.0 and max(lengths) == 12.5
        assert schedule.total_sec(1) == pytest.approx(437.5)
        assert schedule.total_sec(2) == pytest.approx(489.0)

    def test_each_key_action_and_heart_variant_is_cued_four_times(self):
        labels = [s.label for s in schedule.slots()]
        assert {a: labels.count(a) for a in KEY_ACTIONS} == {"yeah": 4, "thumb": 4, "wave": 4, "heart": 12, "open": 4}
        hearts = [s.variant for s in schedule.slots() if s.label == "heart"]
        assert {v: hearts.count(v) for v in set(hearts)} == {"heart_finger": 4, "heart_twohand": 4, "heart_overhead": 4}

    def test_labels_follow_the_offset(self):
        start = schedule.starts(2)[1]  # the first heart
        assert schedule.label_at(10.0 + start + 1.0, 10.0, 2) == ("none", schedule.slots(2)[1])
        assert schedule.label_at(10.0 + start + 4.0, 10.0, 2)[0] == "heart"
        assert schedule.label_at(10.0 + start + 7.5, 10.0, 2)[0] == "none"
        assert schedule.label_at(11.0, 10.0, 2) == ("", None)  # lead-in
        assert schedule.label_at(10.0 + schedule.total_sec(2) + 1, 10.0, 2) == ("", None)

    def test_an_unknown_version_is_refused(self):
        with pytest.raises(ValueError):
            schedule.slots(9)

    def test_the_sheet_and_json_name_the_version(self, capsys):
        assert schedule.main(["--json", "--version", "1"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["version"] == 1 and len(payload["slots"]) == len(schedule.slots(1))
        assert schedule.main(["--sheet"]) == 0
        assert "schedule v2" in capsys.readouterr().out


class TestGate:
    def gate(self, **kw):
        return KeyGate(GateConfig({"heart": 0.5, "thumb": 0.5}, **kw))

    def test_two_looks_in_a_row_trigger_once_while_held(self):
        gate = self.gate()
        fired = [gate.update(t, {"heart": 0.9}) for t in range(6)]
        assert [f.action if f else None for f in fired] == [None, "heart", None, None, None, None]

    def test_a_break_in_the_streak_starts_it_over(self):
        gate = self.gate()
        out = [gate.update(0, {"heart": 0.9}), gate.update(1, {"thumb": 0.9}), gate.update(2, {"heart": 0.9})]
        assert out == [None, None, None]

    def test_released_and_cooled_down_it_triggers_again(self):
        gate = self.gate(cooldown_sec=8.0)
        assert gate.update(0, {"heart": 0.9}) is None and gate.update(1, {"heart": 0.9})
        gate.update(2, {})
        # Released, but inside the cooldown.
        assert gate.update(3, {"heart": 0.9}) is None and gate.update(4, {"heart": 0.9}) is None
        gate.update(5, {})
        assert gate.update(9, {"heart": 0.9}) is None
        assert gate.update(10, {"heart": 0.9}).action == "heart"

    def test_below_threshold_or_without_one_is_no_candidate(self):
        gate = self.gate(confirm=1)
        assert gate.update(0, {"heart": 0.4}) is None
        assert gate.update(1, {"wave": 0.99}) is None
        assert gate.update(2, {"heart": 0.6, "thumb": 0.8}).action == "thumb"

    def test_bad_configs_are_refused(self):
        with pytest.raises(ValueError):
            GateConfig({"dance": 0.5})
        with pytest.raises(ValueError):
            GateConfig({"heart": 0.5}, confirm=0)


def synthetic_features(version, offset, seconds=None, seed=0, rate=1.0):
    """What a person following the cue track looks like to key_eval's two numbers."""
    rng = np.random.default_rng(seed)
    bin_sec = key_eval.BIN_SEC
    seconds = seconds or offset + schedule.total_sec(version) * rate + 10
    n = int(seconds / bin_sec)
    t = np.arange(n) * bin_sec
    dev, motion = rng.normal(1.0, 0.3, n), rng.normal(1.0, 0.3, n)
    for go, down, _ in schedule.move_windows(0.0, version):
        a = offset + rate * go + rng.uniform(0.3, 0.9)
        b = offset + rate * down + rng.uniform(0.2, 0.8)
        dev[(t >= a) & (t < b)] += 3.0
        motion[(t >= a) & (t < a + 0.6)] += 4.0
        motion[(t >= b) & (t < b + 0.6)] += 4.0
    return {"bin_sec": bin_sec, "dev": dev.tolist(), "motion": motion.tolist(), "duration": seconds}


class TestAlignment:
    def test_v2_is_found_clearly(self):
        best, per = align(synthetic_features(2, 12.4), rates=(1.0,))
        assert best.version == 2 and best.offset == pytest.approx(12.4 + 0.3, abs=0.6)
        assert best.runner_up < 0.7

    def test_a_drifting_clock_is_found(self):
        best, _ = align(synthetic_features(2, 6.0, rate=1.02, seed=3), versions=(2,))
        assert best.rate == pytest.approx(1.02, abs=0.006) and best.offset == pytest.approx(6.3, abs=0.6)

    def test_v1_finds_the_offset_but_cannot_rule_out_a_slot_shift(self):
        best, _ = align(synthetic_features(1, 20.0, seed=1), versions=(1,), rates=(1.0,))
        assert best.offset == pytest.approx(20.3, abs=0.6)
        assert best.runner_up > key_eval_score.MAX_RUNNER_UP


class TestReplay:
    def test_looks_run_back_to_back_at_their_own_cost(self):
        ticks = [{"t": 0.5 * k, "methods": {"0": {"prepare_s": 0.3, "q1_s": 0.9, "label": None}}} for k in range(20)]
        looks = replay(ticks, key_eval_score.VIEWS["0"], 0.5)
        assert [look.frame_t for look in looks[:4]] == [0.0, 1.0, 2.5, 3.5]
        assert looks[0].decided == pytest.approx(1.2) and looks[1].decided == pytest.approx(2.4)

    def test_baseline_views_read_q3(self):
        rec = {"prepare_s": 0.3, "q1_s": 0.6, "q3_s": 0.1, "label": "hand/heart", "yes": False}
        tick = {"t": 0, "methods": {"0": rec}}
        assert key_eval_score.VIEWS["0"](tick, {}) == ({}, pytest.approx(1.0))
        assert key_eval_score.VIEWS["0nv"](tick, {}) == ({"heart": 1.0}, pytest.approx(0.9))
        assert key_eval_score.VIEWS["0"]({"t": 0, "methods": {"0": {"error": "x"}}}, {}) is None


class TestCounting:
    alignment = Alignment(version=2, offset=10.0)

    def slot_go(self, variant_index):
        go, down, slot = self.alignment.windows()[variant_index]
        return go, down, slot

    def test_hits_wrong_labels_false_and_outside(self):
        go1, _, heart = self.slot_go(1)  # heart_finger
        go0, _, armsup = self.slot_go(0)  # none slot
        go19, _, thumb = self.slot_go(19)  # thumb_right
        assert heart.label == "heart" and armsup.label == "none" and thumb.label == "thumb"
        looks = [
            Look(1.0, 2.0, {"heart": 1.0}, 1.0),  # before the cue track: not counted
            Look(1.5, 3.0, {"heart": 1.0}, 1.0),
            Look(go0 + 0.5, go0 + 1.5, {"wave": 1.0}, 1.0),  # a none slot: false
            Look(go0 + 1.5, go0 + 2.5, {"wave": 1.0}, 1.0),
            Look(go1 + 0.5, go1 + 1.4, {"heart": 1.0}, 0.9),
            Look(go1 + 1.4, go1 + 2.3, {"heart": 1.0}, 0.9),  # hit, 2.3 s after the beep
            Look(go19 + 0.5, go19 + 1.5, {"yeah": 1.0}, 1.0),
            Look(go19 + 1.5, go19 + 2.5, {"yeah": 1.0}, 1.0),  # wrong label in a thumb slot
        ]
        result = evaluate_video("p01_1", self.alignment, looks, GateConfig(dict.fromkeys(KEY_ACTIONS, 0.5)))
        assert result.triggers == 4 and result.person == "p01"
        assert [f[0] for f in result.false] == ["wave"] and result.false[0][2] == "none_armsup"
        heart_slot = result.slots[0]
        assert heart_slot.variant == "heart_finger" and heart_slot.latency == pytest.approx(2.3)
        thumb_slot = next(s for s in result.slots if s.variant == "thumb_right")
        assert thumb_slot.wrong == ["yeah"] and thumb_slot.latency is None
        summary = summarize([result])
        assert summary["actions"]["heart"]["recall"] == pytest.approx(1 / 12, abs=1e-3)
        assert summary["actions"]["yeah"]["wrong_into"] == 1
        assert summary["actions"]["wave"]["false"] == 1 and summary["false_by_variant"] == {"none_armsup": 1}
        assert summary["confusion"]["thumb"]["yeah"] == 1
        assert 0 < summary["none_minutes"] < schedule.total_sec(2) / 60

    def test_a_late_hit_counts_for_the_window_only(self):
        go, down, _ = self.slot_go(1)
        looks = [Look(down - 0.5, down + 0.5, {"heart": 1.0}, 1.0), Look(down + 0.5, down + 1.5, {"heart": 1.0}, 1.0)]
        summary = summarize([evaluate_video("p02_1", self.alignment, looks, GateConfig({"heart": 0.5}))])
        assert summary["actions"]["heart"]["recall"] == 0.0
        assert summary["actions"]["heart"]["recall_window"] == pytest.approx(1 / 12, abs=1e-3)


class FakeVision:
    def __init__(self, replies):
        self.replies = replies
        self.asked = []
        self.question_failures = {}

    def register_questions(self, questions):
        self.questions = dict(questions)

    def load(self):
        pass

    def prepare(self, frame, box, crop=True, *, source_size=None):
        from visual_interaction.session.internvl_vision import Seen

        return Seen(embeds=None, image=np.zeros((448, 448, 3), np.uint8), crop_fallback=box is None, seconds=0.3)

    def ask(self, seen, key):
        from visual_interaction.session.internvl_vision import Answer

        self.asked.append(key)
        return Answer(text=self.replies.get(key, ""), seconds=0.2, tokens=3)


class TestBaseline:
    def test_a_named_answer_is_verified(self):
        vision = FakeVision({"q1": "比心", "q3.heart": "是"})
        rec, image = key_eval.Baseline(vision)(np.zeros((10, 10, 3), np.uint8), (0, 0, 5, 5))
        assert vision.asked == ["q1", "q3.heart"]
        assert rec["label"] == "hand/heart" and rec["yes"] is True and image.shape == (448, 448, 3)

    def test_a_vague_gesture_is_refined_first(self):
        vision = FakeVision({"q1": "做手势", "q2": "竖大拇指", "q3.thumb": "不是"})
        rec, _ = key_eval.Baseline(vision)(np.zeros((10, 10, 3), np.uint8), None)
        assert vision.asked == ["q1", "q2", "q3.thumb"]
        assert rec["label"] == "hand/thumb" and rec["yes"] is False and rec["fallback"] is True


class FakeChoiceVision:
    """register_choices / prepare_many / score: each picture's features are its mean grey level."""

    HIDDEN = 8

    def __init__(self):
        self.scored = []
        self.batches = []

    def register_choices(self, choices):
        self.choices = dict(choices)

    def prepare_many(self, images):
        from visual_interaction.session.internvl_vision import Seen

        self.batches.append([float(image.mean()) for image in images])
        embeds = np.stack([np.full((256, self.HIDDEN), image.mean(), np.float32) for image in images])
        return Seen(embeds=embeds, image=images[-1], crop_fallback=False, seconds=0.3, images=tuple(images))

    def score(self, seen, key):
        from visual_interaction.session.internvl_vision import Choice

        assert key in self.choices
        frames = int(key.split(".")[1])
        assert len(seen.embeds) == frames
        self.scored.append(key)
        # Letter A is likely; the rest share what is left of 0.9.
        probs = {"A": 0.5, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.05, "F": 0.05}
        return Choice(probs=probs, mass=0.9, top="A", seconds=0.15)


class TestChoices:
    def test_one_vision_batch_feeds_both_picture_counts_and_orders(self):
        vision = FakeChoiceVision()
        method = key_eval.Choices(vision)
        assert set(vision.choices) == {"key.1.fwd", "key.1.rev", "key.4.fwd", "key.4.rev"}
        history = [(round(0.05 * k, 2), np.full((48, 64, 3), k, np.uint8)) for k in range(21)]
        rec, image = method(history[-1][1], None, tuple(history))
        assert len(vision.batches) == 1 and vision.batches[0] == pytest.approx([5, 10, 15, 20])
        assert rec["dt"] == pytest.approx([-0.75, -0.5, -0.25, 0.0]) and rec["fallback"] == 4
        assert vision.scored == ["key.4.fwd", "key.4.rev", "key.1.fwd", "key.1.rev"]
        fwd, rev = rec["4"]["fwd"], rec["4"]["rev"]
        assert fwd["p"][KEY_ACTIONS[0]] == 0.5 and rev["p"][KEY_ACTIONS[-1]] == 0.5
        assert fwd["p"]["none"] == rev["p"]["none"] == 0.05 and fwd["mass"] == 0.9 and fwd["top"] == "A"
        assert rec["_pooled"].shape == (16, FakeChoiceVision.HIDDEN) and rec["_pooled"].dtype == np.float16
        assert float(rec["_pooled"][0, 0]) == 20.0 and image.shape == (448, 448, 3)

    def test_without_history_the_frame_fills_every_slot(self):
        vision = FakeChoiceVision()
        rec, _ = key_eval.Choices(vision)(np.zeros((48, 64, 3), np.uint8), None)
        assert rec["dt"] == [0.0] * 4 and len(vision.batches[0]) == 4

    def test_the_header_scores_blank_pictures(self):
        vision = FakeChoiceVision()
        header = key_eval.Choices(vision).header()
        assert vision.batches == [[128.0] * 4]
        assert set(header["blank"]) == {"4", "1"} and set(header["blank"]["1"]) == {"fwd", "rev"}

    def test_pooling_averages_blocks(self):
        embeds = np.arange(256, dtype=np.float32).reshape(256, 1).repeat(2, axis=1)
        grid = key_eval.pooled(embeds)
        # Block (0, 0) holds rows 0-3 of the 16x16 grid, columns 0-3.
        assert grid.shape == (16, 2) and float(grid[0, 0]) == pytest.approx(
            np.mean([r * 16 + c for r in range(4) for c in range(4)])
        )

    def test_the_views_read_the_shares(self):
        p = {"heart": 0.4, "thumb": 0.1, "yeah": 0.05, "wave": 0.05, "open": 0.05, "none": 0.15}
        rev = {**p, "heart": 0.1, "thumb": 0.4}
        asked = {"fwd": {"p": p, "s": 0.15}, "rev": {"p": rev, "s": 0.16}}
        tick = {"t": 0, "methods": {"A": {"vision_s": 0.3, "4": asked, "1": asked}}}
        scores, seconds = key_eval_score.VIEWS["A"](tick, {})
        assert seconds == pytest.approx(0.45) and set(scores) == {"heart"}
        assert scores["heart"] == pytest.approx(0.5)
        scores, seconds = key_eval_score.VIEWS["A1.cal"](tick, {})
        assert seconds == pytest.approx(0.61) and scores == pytest.approx({"heart": 0.3125, "thumb": 0.3125})
        # A blank picture already says heart: calibrated, thumb is what stands out.
        blank = {**p, "heart": 0.8, "thumb": 0.05, "none": 0.1}
        header = {"calibration": {"A": {"blank": {"4": {"fwd": {"p": blank}}}}}}
        scores, _ = key_eval_score.VIEWS["A.blank"](tick, header)
        assert set(scores) == {"thumb"}
        assert key_eval_score.VIEWS["A"]({"t": 0, "methods": {"A": {"error": "x"}}}, {}) is None


class TestNearestNeighbours:
    def test_each_person_is_matched_against_the_others_only(self):
        alignment = Alignment(version=2, offset=0.0)
        hold = {}
        for go, _, slot in alignment.windows():
            if slot.label in KEY_ACTIONS:
                hold.setdefault(slot.label, go + 2.0)
        none_t = next(go + 1.0 for go, _, slot in alignment.windows() if slot.label == "none")
        times = [hold["heart"], hold["thumb"], none_t]
        directions = {hold["heart"]: 0, hold["thumb"]: 1, none_t: 2}

        def features(seed):
            rng = np.random.default_rng(seed)
            grids = [np.eye(3, 8)[directions[t]] + rng.normal(0, 0.05, (16, 8)) for t in times]
            return {"t": np.array(times), "grid": np.stack(grids).astype(np.float16)}

        def ticks():
            return [{"t": t, "methods": {"A": {"vision_s": 0.3}}} for t in times]

        videos = [(stem, alignment, ticks(), 0.5) for stem in ("p01_1", "p02_1", "p03_1")]
        embeds = {"p01_1": features(1), "p02_1": features(2)}
        assert key_eval_score.knn_ticks(videos, embeds, k=1) == 6
        heart, thumb, none = videos[0][2]
        assert heart["methods"]["C.grid"] == {"scores": {"heart": 1.0}, "s": 0.3}
        assert thumb["methods"]["C.mean"]["scores"] == {"thumb": 1.0}
        assert none["methods"]["C.grid"]["scores"] == {}
        assert "C.mean" not in videos[2][2][0]["methods"]  # no features for p03
        # Alone, a person has nobody to be compared with.
        assert key_eval_score.knn_ticks(videos[:1], {"p01_1": features(1)}, k=1) == 0

    def test_reference_labels_skip_the_edges_of_a_key_window(self):
        alignment = Alignment(version=2, offset=0.0)
        go, down, slot = next(w for w in alignment.windows() if w[2].label in KEY_ACTIONS)
        assert key_eval_score.reference_label(alignment, go + 0.5) is None
        assert key_eval_score.reference_label(alignment, go + 1.5) == slot.label
        assert key_eval_score.reference_label(alignment, down + 0.2) is None


def write_video(path, seconds=3.0, fps=10):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), (i * 7) % 255, np.uint8))
    writer.release()


class TestFiles:
    def test_process_writes_features_and_ticks_and_score_reads_them(self, tmp_path):
        video = tmp_path / "p03_1.mp4"
        write_video(video)
        out = tmp_path / "out"
        out.mkdir()
        vision = FakeVision({"q1": "竖大拇指", "q3.thumb": "是"})
        seen_boxes = []

        def detect(frame):
            seen_boxes.append(frame.shape)
            return [] if len(seen_boxes) % 2 else [(0.9, (10.0, 5.0, 40.0, 45.0))]

        features = key_eval.process(
            video,
            out,
            [key_eval.Baseline(vision)],
            key_eval.Boxes(detect),
            tick_sec=0.5,
            dump_every=2,
            log=lambda *a: None,
        )
        assert features["frames"] == 30 and features["size"] == [64, 48]
        assert len(features["dev"]) == len(features["motion"]) >= 14
        rows = [json.loads(line) for line in (out / "p03_1.ticks.jsonl").read_text().splitlines()]
        assert rows[0]["kind"] == "header" and rows[-1]["kind"] == "end"
        ticks = [row for row in rows if row["kind"] == "tick"]
        assert [tick["t"] for tick in ticks] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        assert [tick["box_src"] for tick in ticks[:3]] == ["fixed", "yolox", "held"]
        assert ticks[1]["methods"]["0"]["label"] == "hand/thumb"
        assert len(list((out / "p03_1" / "crops").glob("*.jpg"))) == 3

        # A 3 s video ends long before the last key slot: refused, however it is aligned.
        for extra in ([], ["--offset", "p03_1=0:2"]):
            assert key_eval.main(["score", str(out), "--confirm", "1", *extra]) == 0
            report = json.loads((out / "report.json").read_text())
            assert report["refused"] == ["p03_1"] and report["runs"] == []

        path = out / "p03_1.features.json"
        path.write_text(json.dumps({**features, "duration": schedule.total_sec(2)}))
        code = key_eval.main(["score", str(out), "--offset", "p03_1=0:2", "--confirm", "1"])
        assert code == 0
        report = json.loads((out / "report.json").read_text())
        assert report["alignment"]["p03_1"]["manual"] is True
        assert {(run["view"], run["confirm"]) for run in report["runs"]} == {("0", 1), ("0nv", 1)}
        assert (out / "report.md").read_text().startswith("# 关键动作离线评测")

    def test_process_with_method_a_keeps_features_and_strides(self, tmp_path):
        video = tmp_path / "p06_1.mp4"
        write_video(video, seconds=2.0)
        out = tmp_path / "out"
        out.mkdir()
        baseline = key_eval.Baseline(FakeVision({"q1": "站着"}))
        baseline.stride = 2
        vision = FakeChoiceVision()
        key_eval.process(video, out, [baseline, key_eval.Choices(vision)], None, tick_sec=0.5, log=lambda *a: None)
        rows = [json.loads(line) for line in (out / "p06_1.ticks.jsonl").read_text().splitlines()]
        header, ticks = rows[0], [row for row in rows if row["kind"] == "tick"]
        assert header["strides"] == {"0": 2, "A": 1} and header["multi_offsets"] == list(key_eval.MULTI_OFFSETS)
        assert set(header["calibration"]["A"]["blank"]) == {"4", "1"}
        assert [("0" in tick["methods"], "A" in tick["methods"]) for tick in ticks] == [(True, True), (False, True)] * 2
        assert all("_pooled" not in tick["methods"]["A"] for tick in ticks)
        with np.load(out / "p06_1.embeds.npz") as data:
            assert data["t"].tolist() == pytest.approx([tick["t"] for tick in ticks])
            assert data["grid"].shape == (4, 16, FakeChoiceVision.HIDDEN)

    def test_features_command_needs_no_model(self, tmp_path, capsys):
        video = tmp_path / "p04_1.mp4"
        write_video(video, seconds=2.0)
        assert key_eval.main(["features", str(video), "--out", str(tmp_path / "f")]) == 0
        assert (tmp_path / "f" / "p04_1.features.json").exists()
        assert not (tmp_path / "f" / "p04_1.ticks.jsonl").exists()

    def test_run_refuses_while_a_launch_is_up(self, tmp_path, monkeypatch, capsys):
        from visual_interaction.session import look_bench

        video = tmp_path / "p05_1.mp4"
        write_video(video, seconds=1.0)
        monkeypatch.setattr(look_bench, "_launch_running", lambda: ["123: ros2 launch x"])
        code = key_eval.main(
            ["run", str(video), "--out", str(tmp_path / "o"), "--demo-root", str(tmp_path), "--no-yolox"]
        )
        assert code == 2 and "ros2 launch" in capsys.readouterr().err
