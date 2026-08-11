import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends.errors import BackendInferenceError
from inference_service.generic_runtime import NamedTensorRequest
from voice_tts_service.errors import BackendLoadError
from voice_tts_service.zipvoice_310p_adapter import ZipVoiceAscendSession, _ChineseTokenizer


class _Tokenizer:
    pad_id = 0

    @staticmethod
    def text_to_tokens(text):
        return list(text)

    @staticmethod
    def tokens_to_ids(tokens):
        return [1] * len(tokens)

    @staticmethod
    def chunk_tokens(tokens, max_tokens):
        return [tokens[index : index + max_tokens] for index in range(0, len(tokens), max_tokens)]


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
    return NamedTensorRequest(
        "tts-test",
        {
            "tts.text": np.frombuffer(text.encode(), dtype=np.uint8),
            "tts.prompt_audio": np.ones(1, dtype=np.float32) if prompt else np.empty(0, dtype=np.float32),
            "tts.prompt_sample_rate": np.asarray(24000 if prompt else 0, dtype=np.int64),
            "tts.prompt_text": np.frombuffer("参考".encode(), dtype=np.uint8)
            if prompt
            else np.empty(0, dtype=np.uint8),
        },
    )


def test_verified_310p_session_runs_four_flow_steps_and_returns_pcm():
    session = _session()

    outputs = session._execute(_request())

    assert outputs["tts.audio"].shape == (240,)
    assert session.roles == [(0, "text_encoder")] + [(1, "flow_decoder_1537")] * 4


def test_verified_310p_deployment_explicitly_rejects_request_prompt():
    with pytest.raises(BackendInferenceError, match="fixed prompt") as error:
        _session()._execute(_request(prompt=True))
    assert error.value.code == "unsupported_prompt"


def test_chinese_frontend_rejects_ascii_before_loading_optional_dependencies(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("_\t0\n.\t1\n", encoding="utf-8")
    fake_cn2an = SimpleNamespace(transform=lambda value, _mode: value)
    fake_jieba = SimpleNamespace(
        default_logger=SimpleNamespace(setLevel=lambda _level: None), initialize=lambda: None, cut=lambda value: [value]
    )
    fake_pypinyin = SimpleNamespace(Style=SimpleNamespace(TONE3="tone3"), lazy_pinyin=lambda words, **kwargs: words)
    fake_tone = SimpleNamespace(to_finals_tone3=lambda *args, **kwargs: "", to_initials=lambda *args, **kwargs: "")
    modules = {
        "cn2an": fake_cn2an,
        "jieba": fake_jieba,
        "pypinyin": fake_pypinyin,
        "pypinyin.contrib.tone_convert": fake_tone,
    }
    monkeypatch.setitem(sys.modules, "pypinyin.contrib", SimpleNamespace(tone_convert=fake_tone))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    tokenizer = _ChineseTokenizer(token_file)

    with pytest.raises(Exception, match="not English words"):
        tokenizer.text_to_tokens("hello")


def test_frontend_import_failure_is_reported(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("_\t0\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "cn2an", None)

    with pytest.raises(BackendLoadError, match="dependency is unavailable"):
        _ChineseTokenizer(token_file)


def test_ellipsis_is_treated_as_sentence_punctuation():
    assert _ChineseTokenizer._map_punctuation("等等...") == "等等…"
    assert _ChineseTokenizer.chunk_tokens(["deng3", "…", "hao3", "."], 3) == [
        ["deng3", "…"],
        ["hao3", "."],
    ]
