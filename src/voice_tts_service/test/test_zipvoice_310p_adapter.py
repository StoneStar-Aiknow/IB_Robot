import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends.errors import BackendInferenceError
from inference_service.unified_runtime import ExecutionContext, ModelRequest
from voice_tts_service.errors import BackendLoadError
from voice_tts_service.zipvoice_310p_adapter import ZipVoiceAscendSession, _EmiliaTokenizer
from voice_tts_service.zipvoice_shared import TokenGroup, duration_model, plan_chunks


class _Tokenizer:
    pad_id = 0
    tail_sentinel = ()

    @staticmethod
    def text_to_groups(text):
        return [TokenGroup((char,), "zh") for char in text]

    @staticmethod
    def tokens_to_ids(tokens):
        return [1] * len(tokens)


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((1, 240), dtype=np.float32)


class _Torch:
    @staticmethod
    def from_numpy(value):
        return _Tensor(value)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _Vocos:
    def __call__(self, value):
        return value


def _session():
    session = object.__new__(ZipVoiceAscendSession)
    session._config = {
        "text_capacity": 256,
        "flow_frames": 1537,
        "num_steps": 4,
        "sample_rate": 24000,
        "cross_fade_sec": 0.1,
    }
    session._text_role = "text_encoder"
    session._flow_role = "flow_decoder_1537"
    session._tokenizer = _Tokenizer()
    session._prompt = SimpleNamespace(
        tokens=np.zeros((1, 29), dtype=np.int64),
        features=np.zeros((1, 302, 100), dtype=np.float32),
        frame_count=302,
    )
    session._torch = _Torch()
    session._vocos = _Vocos()
    session.roles = []

    def run_role(role_index, role, values):
        session.roles.append((role_index, role))
        if role == "text_encoder":
            tokens_len = int(np.asarray(values["host.zipvoice.tokens_len"]))
            features_len = 302 + tokens_len
            return {
                "host.zipvoice.text_condition": np.zeros((1, 3072, 100), dtype=np.float32),
                "host.zipvoice.features_len": np.asarray(features_len, dtype=np.int64),
                "host.zipvoice.padding_mask": np.arange(3072)[None, :] >= features_len,
            }
        if role == "flow_decoder_1537":
            return {"host.zipvoice.velocity": np.zeros_like(values["host.zipvoice.flow_x"])}
        raise AssertionError(f"unexpected role: {role}")

    session._run_role = run_role
    return session


def _request(text="机器人", *, prompt=False):
    return ModelRequest(
        {
            "tts.text": np.frombuffer(text.encode(), dtype=np.uint8),
            "tts.prompt_audio": np.ones(1, dtype=np.float32) if prompt else np.empty(0, dtype=np.float32),
            "tts.prompt_sample_rate": np.asarray(24000 if prompt else 0, dtype=np.int64),
            "tts.prompt_text": np.frombuffer("参考".encode(), dtype=np.uint8)
            if prompt
            else np.empty(0, dtype=np.uint8),
        }
    )


def _context():
    return ExecutionContext("tts-test")


def test_verified_310p_session_runs_four_flow_steps_and_returns_pcm():
    session = _session()

    outputs = session._execute(_request(), _context())

    # 240 vocoded samples plus the default 100 ms of trailing silence at 24 kHz.
    assert outputs["tts.audio"].shape == (240 + 2400,)
    assert session.roles == [(0, "text_encoder")] + [(1, "flow_decoder_1537")] * 4


def test_310p_session_lays_out_planned_frames_after_the_encoder_prompt_layout():
    session = _session()
    session._tokenizer.text_to_groups = lambda text: [
        TokenGroup(("a1",), "zh"),
        TokenGroup((",",), "punct"),
        TokenGroup(("b1",), "zh"),
        TokenGroup((".",), "punct"),
    ]
    inputs = {}
    run_role = session._run_role

    def capture(role_index, role, values):
        inputs.setdefault(role, dict(values))
        return run_role(role_index, role, values)

    session._run_role = capture
    session._execute(_request("a,b."), _context())

    (plan,) = plan_chunks(
        session._tokenizer.text_to_groups(""),
        duration_model(session._prompt, session._config),
        max_tokens=256,
        max_target_frames=1537 - 302,
    )
    flow = inputs["flow_decoder_1537"]
    # The encoder runs at speed 1 and only supplies embeddings; 29 prompt tokens keep 302 // 29 = 10
    # frames each, and the target tokens follow with their planned frames.
    assert float(inputs["text_encoder"]["host.zipvoice.speed"]) == 1.0
    assert int((~flow["host.zipvoice.flow_padding_mask"]).sum()) == 290 + sum(plan.frames)


def test_verified_310p_deployment_explicitly_rejects_request_prompt():
    with pytest.raises(BackendInferenceError, match="fixed prompt") as error:
        _session()._execute(_request(prompt=True), _context())
    assert error.value.code == "unsupported_prompt"


def _install_fake_frontend(monkeypatch, *, english=True):
    fake_cn2an = SimpleNamespace(transform=lambda value, _mode: value)
    fake_jieba = SimpleNamespace(
        default_logger=SimpleNamespace(setLevel=lambda _level: None), initialize=lambda: None, cut=lambda value: [value]
    )
    fake_pypinyin = SimpleNamespace(
        Style=SimpleNamespace(TONE3="tone3"),
        lazy_pinyin=lambda words, **kwargs: ["a1" for word in words for _char in word],
    )
    fake_tone = SimpleNamespace(
        to_finals_tone3=lambda syllable, **kwargs: syllable, to_initials=lambda *args, **kwargs: ""
    )
    modules = {
        "cn2an": fake_cn2an,
        "jieba": fake_jieba,
        "pypinyin": fake_pypinyin,
        "pypinyin.contrib.tone_convert": fake_tone,
        "piper_phonemize": SimpleNamespace(phonemize_espeak=lambda text, voice: [list(text.lower())])
        if english
        else None,
    }
    monkeypatch.setitem(sys.modules, "pypinyin.contrib", SimpleNamespace(tone_convert=fake_tone))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_frontend_tokenizes_mixed_chinese_and_english_words(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("_\t0\n.\t1\n \t2\na1\t3\nr\t4\no\t5\ns\t6\n", encoding="utf-8")
    _install_fake_frontend(monkeypatch)
    tokenizer = _EmiliaTokenizer(token_file)

    groups = tokenizer.text_to_groups("打开ROS")

    assert [(group.kind, group.tokens) for group in groups] == [
        ("zh", ("a1", "a1")),
        ("en", ("r", "o", "s")),
        ("punct", (".",)),
    ]


def test_frontend_reports_english_without_the_espeak_dependency(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("_\t0\n.\t1\na1\t3\n", encoding="utf-8")
    _install_fake_frontend(monkeypatch, english=False)
    tokenizer = _EmiliaTokenizer(token_file)

    assert tokenizer.text_to_tokens("打开") == ["a1", "a1", "."]
    with pytest.raises(Exception, match="requires piper_phonemize"):
        tokenizer.text_to_tokens("hello")


def test_frontend_import_failure_is_reported(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("_\t0\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "cn2an", None)

    with pytest.raises(BackendLoadError, match="dependency is unavailable"):
        _EmiliaTokenizer(token_file)


def test_ellipsis_is_treated_as_sentence_punctuation():
    assert _EmiliaTokenizer._map_punctuation("等等……") == "等等."
