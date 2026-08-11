import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from voice_tts_service.errors import BackendInferenceError, BackendLoadError, TTSError
from voice_tts_service.zipvoice_310p_adapter import ZipVoice310PAdapter, _ChineseTokenizer


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


class _TextSession:
    def infer(self, inputs):
        tokens_len = int(np.asarray(inputs["tts.tokens_len"]))
        features_len = 302 + tokens_len
        condition = np.zeros((1, 3072, 100), dtype=np.float32)
        mask = np.arange(3072)[None, :] >= features_len
        return {
            "internal.text_condition": condition,
            "internal.features_len": np.asarray(features_len, dtype=np.int64),
            "internal.padding_mask": mask,
        }


class _FlowSession:
    def __init__(self):
        self.calls = 0

    def infer(self, inputs):
        self.calls += 1
        return {"tts.velocity": np.zeros_like(inputs["tts.flow_x"])}


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


def _adapter():
    adapter = object.__new__(ZipVoice310PAdapter)
    adapter._config = {
        "text_capacity": 256,
        "flow_frames": 1537,
        "num_steps": 4,
        "sample_rate": 24000,
        "cross_fade_sec": 0.1,
    }
    adapter._text_session = _TextSession()
    adapter._flow_session = _FlowSession()
    adapter._tokenizer = _Tokenizer()
    adapter._prompt = SimpleNamespace(
        tokens=np.zeros((1, 29), dtype=np.int64),
        features=np.zeros((1, 302, 100), dtype=np.float32),
        frame_count=302,
    )
    adapter._torch = _Torch()
    adapter._vocos = _Vocos()
    return adapter


def test_verified_310p_schedule_runs_four_flow_steps_and_returns_pcm():
    adapter = _adapter()

    result = adapter.synthesize("机器人", None, "")

    assert result.sample_rate == 24000
    assert len(result.samples) == 240
    assert adapter._flow_session.calls == 4


def test_verified_310p_deployment_explicitly_rejects_request_prompt():
    adapter = _adapter()

    with pytest.raises(TTSError, match="fixed prompt") as error:
        adapter.synthesize("机器人", object(), "参考文本")
    assert error.value.code == "UNSUPPORTED_PROMPT"


def test_chinese_frontend_rejects_ascii_before_loading_optional_dependencies(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("_\t0\n.\t1\n", encoding="utf-8")
    fake_cn2an = SimpleNamespace(transform=lambda value, _mode: value)
    fake_jieba = SimpleNamespace(
        default_logger=SimpleNamespace(setLevel=lambda _level: None),
        initialize=lambda: None,
        cut=lambda value: [value],
    )
    fake_pypinyin = SimpleNamespace(
        Style=SimpleNamespace(TONE3="tone3"),
        lazy_pinyin=lambda words, **kwargs: words,
    )
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

    with pytest.raises(BackendInferenceError, match="not English words"):
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
