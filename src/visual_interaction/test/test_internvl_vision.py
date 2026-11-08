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
                return dict(tokens=tokens, prefill_s=0.0, decode_s=[])
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
        gesture, still = tmp_path / "gesture.jpg", tmp_path / "still.jpg"
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
