"""The InternVL adapter against a fake board deployment: no NPU, no OMs.

The fixture writes stand-ins for the board's demo modules (``aclruntime``,
``live_chat``, ``llm_host``, ``qwen_tok``, ``qwen_detok``, ``run_llm_om``) into
a temporary ``demo_root``, so ``load`` runs its real import path and the tests
count vision and generator calls.
"""

import sys
import textwrap

import numpy as np
import pytest

from visual_interaction.session.hooks import ActorSignal, RoundInputs, Window
from visual_interaction.session.internvl_vision import (
    IMAGE_TOKENS,
    PREFILL_LEN,
    VISION_BATCH,
    Answer,
    Choice,
    InternVlVision,
    QuestionError,
    Seen,
    VisionBackendError,
    question_prompt,
)

HIDDEN = 8
IMG_CONTEXT = 900001
IM_END = 900002

_FAKES = {
    "scripts/aclruntime.py": "",
    "scripts/run_llm_om.py": f"""
        import numpy as np

        OM = {{name: name for name in ("internvl3_1b_vision_b4_fp16", "llm_prefill_fp16", "llm_decode_fp16")}}
        VISION_CALLS = []

        def session(path):
            return path

        def om_backend(sess, want_outputs):
            if sess != "internvl3_1b_vision_b4_fp16":
                return lambda feed: None

            def run(feed):
                pixels = feed["pixel_values"]
                VISION_CALLS.append(pixels.copy())
                # Row b of every token carries the mean of picture b, so the
                # features say which picture they came from.
                out = np.repeat(pixels.reshape(pixels.shape[0], -1).mean(axis=1)[:, None, None],
                                {IMAGE_TOKENS} * {HIDDEN}, axis=2).reshape(-1, {HIDDEN})
                return [out.astype(np.float32)]
            return run
    """,
    "scripts/qwen_tok.py": f"""
        class Tokenizer:
            added = {{"<IMG_CONTEXT>": {IMG_CONTEXT}, "<|im_end|>": {IM_END}}}

            def __init__(self, model_dir):
                pass

            def encode(self, text):
                ids = []
                for piece in text.split("<IMG_CONTEXT>"):
                    ids.extend(ord(ch) for ch in piece)
                    ids.append({IMG_CONTEXT})
                return ids[:-1]
    """,
    "scripts/qwen_detok.py": f"""
        class Detokenizer:
            def __init__(self, model_dir):
                pass

            def decode(self, ids):
                return "".join(chr(i) for i in ids if i != {IM_END})
    """,
    "scripts/llm_host.py": f"""
        import numpy as np

        RUNS = []
        REPLY = {{"text": "比心"}}
        # score: the first token's logits, zero but for these characters.
        BOOST = {{"A": 2.0, "B": 1.0}}

        class Generator:
            def __init__(self, embed, prefill, decode, P, L, img_ctx, eos):
                self.P, self.L, self.img_ctx, self.eos = P, L, img_ctx, eos

            def run(self, input_ids, vit_embeds, max_new_tokens=64, keep_logits=True):
                # What the board's generator does with the features: read them
                # into the image slots of a fresh prompt array.
                slots = np.flatnonzero(np.asarray(input_ids) == self.img_ctx)
                vit = vit_embeds.reshape(-1, {HIDDEN})
                if len(slots) != len(vit):
                    raise ValueError(f"{{len(slots)}} slots, {{len(vit)}} feature rows")
                if len(input_ids) > self.P or len(input_ids) + max_new_tokens - 1 > self.L:
                    raise ValueError("prompt does not fit")
                RUNS.append(dict(ids=np.asarray(input_ids), embeds=vit_embeds, max_new_tokens=max_new_tokens,
                                 first=float(vit[0, 0])))
                tokens = [ord(ch) for ch in REPLY["text"]][: max_new_tokens - 1] + [self.eos]
                logits = []
                if keep_logits:
                    first = np.zeros(65536, np.float32)
                    for ch, value in BOOST.items():
                        first[ord(ch)] = value
                    logits = [first]
                return dict(tokens=tokens, logits=logits, prefill_s=0.0, decode_s=[])
    """,
    "live/live_chat.py": """
        import numpy as np

        MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        PREPROCESSED = []

        def quiet_native_output(sink_path):
            pass

        def preprocess(jpegs):
            PREPROCESSED.append(list(jpegs))
            return np.full((len(jpegs), 3, 448, 448), 0.5, np.float32)
    """,
}
_MODULES = ("aclruntime", "run_llm_om", "qwen_tok", "qwen_detok", "llm_host", "live_chat")


@pytest.fixture
def demo_root(tmp_path):
    root = tmp_path / "vlx"
    for name, body in _FAKES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    (root / "tokenizer").mkdir()
    (root / "om").mkdir()
    (root / "onnx").mkdir()
    np.save(root / "onnx" / "embed_tokens_fp32.npy", np.zeros((4, HIDDEN), np.float32))
    saved_path = list(sys.path)
    saved = {name: sys.modules.pop(name) for name in _MODULES if name in sys.modules}
    yield root
    sys.path[:] = saved_path
    for name in _MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def fakes():
    return sys.modules["run_llm_om"], sys.modules["llm_host"]


QUESTIONS = {
    "q1": ("画面里离镜头最近的人正在做什么动作或手势？", 16),
    "q2": ("这个人的手指比的是什么手势？用不超过五个字回答。", 6),
    "q3.heart": ("这个人是在用手比爱心吗？只回答是或不是。", 3),
}


def frame(value=120, width=640, height=480):
    return np.full((height, width, 3), value, np.uint8)


class TestPrepareAndAsk:
    def test_prepare_runs_vision_once_and_asks_reuse_its_features(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        vision.load()
        oms, host = fakes()

        seen = vision.prepare(frame(), (200.0, 100.0, 400.0, 460.0))
        first = vision.ask(seen, "q1")
        second = vision.ask(seen, "q3.heart")
        again = vision.ask(seen, "q1")

        assert len(oms.VISION_CALLS) == 1
        assert isinstance(seen, Seen) and isinstance(first, Answer)
        assert seen.embeds.shape == (1, IMAGE_TOKENS, HIDDEN)
        assert not seen.embeds.flags.writeable
        assert seen.image.shape == (448, 448, 3) and seen.image.dtype == np.uint8
        assert seen.crop_fallback is False and seen.seconds >= 0.0
        assert [run["embeds"] is seen.embeds for run in host.RUNS] == [True, True, True]
        # Each question's own prompt and budget, not the observe prompt's.
        assert [run["max_new_tokens"] for run in host.RUNS] == [16, 3, 16]
        assert not np.array_equal(host.RUNS[0]["ids"], host.RUNS[1]["ids"])
        assert first.text == "比心" and first.tokens == 3 and first.seconds >= 0.0
        assert again.text == first.text and second.tokens == 3

    def test_padding_rows_stay_zero_and_only_the_picture_is_spliced(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        bright = vision.prepare(frame(250), None, crop=False)
        dark = vision.prepare(frame(10), None, crop=False)
        oms, host = fakes()
        assert len(oms.VISION_CALLS) == 2
        for pixels in oms.VISION_CALLS:
            assert pixels.shape == (VISION_BATCH, 3, 448, 448)
            assert not pixels[1:].any()
        vision.ask(bright, "q1")
        vision.ask(dark, "q1")
        vision.ask(bright, "q2")
        firsts = [run["first"] for run in host.RUNS]
        assert firsts[0] == firsts[2] != firsts[1]

    def test_preprocessing_is_the_demos_normalisation(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        picture = np.zeros((448, 448, 3), np.uint8)
        picture[..., 0] = 255  # blue in BGR
        vision.prepare(picture, None, crop=False)
        oms, _ = fakes()
        pixels = oms.VISION_CALLS[0][0]
        chat = sys.modules["live_chat"]
        expected_rgb = (np.array([0.0, 0.0, 1.0], np.float32)[:, None, None] - chat.MEAN) / chat.STD
        assert np.allclose(pixels[:, 0, 0], expected_rgb[:, 0, 0])

    def test_crop_fallback_is_reported(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        assert vision.prepare(frame(), None).crop_fallback is True
        assert vision.prepare(frame(), (10.0, 10.0, 20.0, 20.0)).crop_fallback is True
        assert vision.prepare(frame(), None, crop=False).crop_fallback is False

    def test_questions_registered_before_load_are_checked_by_prepare(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        seen = vision.prepare(frame(), None)
        assert vision.ask(seen, "q2").text == "比心"
        assert vision.question_failures == {}

    def test_questions_registered_after_load_are_checked_at_once(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.load()
        vision.register_questions({"late": ("这个人在挥手吗？", 3), "huge": ("x" * (PREFILL_LEN + 10), 3)})
        seen = vision.prepare(frame(), None)
        assert vision.ask(seen, "late").tokens == 3
        assert "huge" in vision.question_failures

    def test_a_vision_failure_raises_the_backend_error(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.load()
        vision._vision = lambda feed: (_ for _ in ()).throw(RuntimeError("device lost"))
        with pytest.raises(VisionBackendError, match="device lost"):
            vision.prepare(frame(), None)
        assert "device lost" in vision.last_error

    def test_a_generation_failure_raises_the_backend_error(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        seen = vision.prepare(frame(), None)
        _, host = fakes()
        host.REPLY["text"] = None  # makes the fake raise TypeError
        with pytest.raises(VisionBackendError, match="TypeError"):
            vision.ask(seen, "q1")
        host.REPLY["text"] = "比心"

    def test_load_failure_surfaces_from_prepare(self, demo_root):
        (demo_root / "live" / "live_chat.py").write_text("raise ImportError('no cv2 here')\n")
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        with pytest.raises(VisionBackendError, match="no cv2 here"):
            vision.prepare(frame(), None)


class TestQuestionBudget:
    def test_a_question_over_the_prefill_is_refused_and_named(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions({**QUESTIONS, "long": ("长" * PREFILL_LEN, 4)})
        vision.load()
        assert set(vision.question_failures) == {"long"}
        assert "prefill" in vision.question_failures["long"]
        seen = vision.prepare(frame(), None)
        with pytest.raises(QuestionError, match="'long' was refused"):
            vision.ask(seen, "long")
        assert vision.ask(seen, "q1").text == "比心"

    def test_a_decode_budget_over_the_cache_is_refused(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions({"greedy": ("说吧", 4000)})
        vision.load()
        assert "cache" in vision.question_failures["greedy"]

    def test_a_chat_prompt_without_exactly_one_image_slot_is_refused(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        two = "<|im_start|>user\n<image><image>你好<|im_end|>\n<|im_start|>assistant\n"
        none = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        vision.register_questions({"two": (two, 3), "none": (none, 3)})
        vision.load()
        assert set(vision.question_failures) == {"two", "none"}

    def test_an_unknown_key_is_refused(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        seen = vision.prepare(frame(), None)
        with pytest.raises(QuestionError, match="not registered"):
            vision.ask(seen, "q9")

    @pytest.mark.parametrize(
        "questions",
        [{"q": "text only"}, {"q": ("", 3)}, {"q": ("text", 0)}, {"q": ("text", 2.5)}, {"": ("text", 3)}],
    )
    def test_a_malformed_registration_raises_at_once(self, demo_root, questions):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        with pytest.raises(ValueError):
            vision.register_questions(questions)

    def test_a_plain_question_is_wrapped_like_chatter_with_one_image(self):
        from visual_interaction.session.chatter import CHATTER_SYSTEM

        prompt = question_prompt("这个人在挥手吗？")
        assert prompt.count("<image>") == 1
        assert CHATTER_SYSTEM in prompt and prompt.endswith("<|im_start|>assistant\n")


def choice_prompt(images):
    from visual_interaction.phase0.prompts import chat, frame_header

    return chat(frame_header(images) + "这个人在做哪个动作？A 比心 B 挥手 C 都不是", system="只回答一个字母。")


def picture(value):
    return np.full((448, 448, 3), value, np.uint8)


class TestPrepareManyAndScore:
    def test_four_pictures_run_in_one_batch_and_score_reads_the_options(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_choices({"pick": (choice_prompt(4), ("A", "B", "C"))})
        seen = vision.prepare_many([picture(v) for v in (10, 20, 30, 40)])
        choice = vision.score(seen, "pick")
        oms, host = fakes()

        assert len(oms.VISION_CALLS) == 1 and oms.VISION_CALLS[0][1:].any()
        assert seen.embeds.shape == (4, IMAGE_TOKENS, HIDDEN) and not seen.embeds.flags.writeable
        assert len(seen.images) == 4 and seen.image is seen.images[-1]
        assert isinstance(choice, Choice) and set(choice.probs) == {"A", "B", "C"}
        # Softmax over the whole fake vocabulary: A and B boosted, C at zero like the rest.
        expected = np.exp([2.0, 1.0, 0.0]) / (np.exp(2.0) + np.exp(1.0) + 65534)
        assert np.allclose([choice.probs[o] for o in "ABC"], expected)
        assert choice.mass == pytest.approx(expected.sum())
        assert choice.top == "A" and choice.seconds >= 0.0
        assert host.RUNS[-1]["max_new_tokens"] == 1 and host.RUNS[-1]["embeds"] is seen.embeds

    def test_several_seen_fill_the_slots_in_order(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_choices({"pick": (choice_prompt(3), ("A", "B", "C"))})
        references = vision.prepare_many([picture(200), picture(100)])
        live = vision.prepare(picture(50), None, crop=False)
        vision.score([references, live], "pick")
        _, host = fakes()
        embeds = host.RUNS[-1]["embeds"]
        assert embeds.shape == (3, IMAGE_TOKENS, HIDDEN)
        assert embeds[0, 0, 0] > embeds[1, 0, 0] > embeds[2, 0, 0]

    def test_prepare_after_prepare_many_still_pads_with_zeros(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.prepare_many([picture(200)] * 3)
        vision.prepare(frame(), None)
        oms, _ = fakes()
        assert oms.VISION_CALLS[0][1:3].any() and not oms.VISION_CALLS[1][1:].any()

    def test_the_wrong_number_of_pictures_is_refused(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_choices({"pick": (choice_prompt(2), ("A", "B"))})
        seen = vision.prepare_many([picture(1)] * 3)
        with pytest.raises(QuestionError, match="shows 2 picture"):
            vision.score(seen, "pick")

    def test_a_plain_choice_question_shows_one_picture(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_choices({"one": ("这个人在比心吗？A 是 B 不是", ("A", "B"))})
        seen = vision.prepare(frame(), None)
        assert vision.score(seen, "one").probs["A"] > vision.score(seen, "one").probs["B"]

    def test_bad_choice_questions_are_refused_at_load(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_choices(
            {
                "five": (choice_prompt(5), ("A", "B")),
                "none": ("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", ("A", "B")),
                "long": (choice_prompt(4) + "长" * 300, ("A", "B")),
                "words": (choice_prompt(1), ("A", "比心")),
                "ok": (choice_prompt(1), ("A", "B")),
            }
        )
        vision.load()
        assert set(vision.question_failures) == {"five", "none", "long", "words"}
        assert "prefill" in vision.question_failures["long"]
        assert "2 tokens" in vision.question_failures["words"]
        seen = vision.prepare(frame(), None)
        with pytest.raises(QuestionError, match="refused"):
            vision.score(seen, "words")
        assert vision.score(seen, "ok").top == "A"

    def test_ask_and_score_keep_to_their_own_questions(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        vision.register_choices({"pick": (choice_prompt(1), ("A", "B"))})
        with pytest.raises(ValueError, match="already registered"):
            vision.register_choices({"q1": (choice_prompt(1), ("A",))})
        with pytest.raises(ValueError, match="already registered"):
            vision.register_questions({"pick": ("这个人在挥手吗？", 3)})
        seen = vision.prepare(frame(), None)
        with pytest.raises(QuestionError, match="not registered"):
            vision.ask(seen, "pick")
        with pytest.raises(QuestionError, match="not a registered choice"):
            vision.score(seen, "q1")

    @pytest.mark.parametrize(
        "choices",
        [{"q": ("text", "AB")}, {"q": ("text", ())}, {"q": ("text", ("A", "A"))}, {"q": ("", ("A",))}, {"q": "x"}],
    )
    def test_a_malformed_choice_registration_raises_at_once(self, demo_root, choices):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        with pytest.raises(ValueError):
            vision.register_choices(choices)

    @pytest.mark.parametrize("count", [0, 5])
    def test_prepare_many_takes_one_to_four_pictures(self, demo_root, count):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        with pytest.raises(ValueError, match="1 to 4"):
            vision.prepare_many([picture(1)] * count)

    def test_prepare_many_wants_cut_pictures(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        with pytest.raises(ValueError, match="448"):
            vision.prepare_many([frame()])


class TestObserveUnchanged:
    def window(self, frames):
        return RoundInputs(
            window=Window(frames=frames, stamps=tuple(float(i) for i in range(len(frames))), received_at=1.0),
            actor=ActorSignal(present=True, known=True, bbox=(0.0, 0.0, 10.0, 10.0), stamp=1.0),
            now=1.0,
        )

    def test_observe_answers_from_the_newest_frames(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=2, prompt_mode="chatter", max_new_tokens=16)
        text = vision.observe(self.window((b"a", b"b", b"c")))
        oms, host = fakes()
        assert text == "比心"
        assert sys.modules["live_chat"].PREPROCESSED == [[b"b", b"c"]]
        assert len(oms.VISION_CALLS) == 1 and host.RUNS[0]["embeds"].shape == (2, IMAGE_TOKENS, HIDDEN)
        assert host.RUNS[0]["max_new_tokens"] == 16
        assert vision.prompt_tokens > 2 * IMAGE_TOKENS

    def test_observe_with_too_few_frames_is_none(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=2, prompt_mode="chatter")
        assert vision.observe(self.window((b"a",))) is None
        assert "need 2" in vision.last_error

    def test_observe_and_ask_share_one_loaded_model(self, demo_root):
        vision = InternVlVision(demo_root=str(demo_root), frames=1, prompt_mode="chatter")
        vision.register_questions(QUESTIONS)
        assert vision.observe(self.window((b"a",))) == "比心"
        seen = vision.prepare(frame(), None)
        assert vision.ask(seen, "q1").text == "比心"
        oms, _ = fakes()
        assert len(oms.VISION_CALLS) == 2


class TestLookBench:
    def test_the_bench_runs_end_to_end_on_the_fake_deployment(self, demo_root, tmp_path, monkeypatch, capsys):
        import json
        import types

        import cv2

        from visual_interaction.session import look_bench

        questions = {**QUESTIONS, "q3.wave": ("这个人是在挥手打招呼吗？只回答是或不是。", 3)}
        monkeypatch.setitem(
            sys.modules, "visual_interaction.session.look_say", types.SimpleNamespace(QUESTIONS=questions)
        )
        # Same file name in two folders: the frames of a live run are named like this.
        gesture, still = tmp_path / "a" / "0.jpg", tmp_path / "b" / "0.jpg"
        gesture.parent.mkdir()
        still.parent.mkdir()
        cv2.imwrite(str(gesture), frame(150, 1344, 972))
        cv2.imwrite(str(still), frame(90, 1344, 972))
        code = look_bench.main(
            [
                "--demo-root",
                str(demo_root),
                "--image",
                str(gesture),
                "--bbox",
                "560,220,840,900",
                "--image",
                str(still),
                "--runs",
                "2",
                "--no-gesture-image",
                str(still),
                "--allow-launch",
            ]
        )
        out = capsys.readouterr().out.strip().splitlines()
        assert code == 0
        report = json.loads(out[-1])
        assert report["prepare"]["crop"]["n"] == 4 and report["prepare"]["whole"]["n"] == 4
        assert set(report["ask"]) == {f"{key}/{view}" for key in questions for view in ("crop", "whole")}
        assert report["ask"]["q1/crop"]["n"] == 4
        assert [item["synthetic"] for item in report["images"]] == [False, True]
        assert len(report["answers_last_run"]) == 4
        assert set(report["q3_no_gesture"]["crop"]) == {"q1", "q3.heart", "q3.wave"}
        assert report["load_s"] >= 0 and report["question_failures"] == {}
        assert any(line.startswith("prepare/crop") for line in out)

    def test_the_bench_reads_yes_the_way_the_plan_does(self):
        from visual_interaction.session.look_bench import _says_yes

        assert _says_yes("是")
        assert _says_yes("是的。")
        assert not _says_yes("不是")
        assert not _says_yes("是，没有")
        assert not _says_yes("好像是")
