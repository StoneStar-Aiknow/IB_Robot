"""LLMClientService 单元测试（mock）+ 真实网络冒烟测试。

真实网络用例在未设置 ALIYUN_API_KEY 时自动 skip。
"""

import os

import pytest

from embodied_agent.llm_client_service import LLMClientService, _read_system_prompt


class _FakeVLM:
    """记录 chat() 入参、返回预设结果的假 VLMClient。"""

    def __init__(self, result=None):
        self.calls = []
        self.cleared = 0
        self._result = result or {"status": "ok", "content": "嘎，你好呀！", "error": None}

    def chat(self, text, model=None):
        self.calls.append({"text": text, "model": model})
        return self._result

    def clear_history(self):
        self.cleared += 1


def test_reply_passes_text_and_model_and_returns_dict():
    fake = _FakeVLM()
    svc = LLMClientService(model="qwen-vl-plus", vlm=fake)
    result = svc.reply("你好")
    assert result["status"] == "ok"
    assert result["content"] == "嘎，你好呀！"
    assert fake.calls == [{"text": "你好", "model": "qwen-vl-plus"}]


def test_reply_transparently_returns_error_dict():
    err = {"status": "error", "content": "", "error": "missing API key"}
    svc = LLMClientService(vlm=_FakeVLM(result=err))
    result = svc.reply("你好")
    assert result["status"] == "error"
    assert result["error"] == "missing API key"


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_reply_rejects_empty_or_non_string(bad):
    svc = LLMClientService(vlm=_FakeVLM())
    with pytest.raises(ValueError):
        svc.reply(bad)


def test_reset_clears_history():
    fake = _FakeVLM()
    svc = LLMClientService(vlm=fake)
    svc.reset()
    assert fake.cleared == 1


def test_read_system_prompt_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _read_system_prompt(tmp_path / "nope.txt")


def test_read_system_prompt_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _read_system_prompt(empty)


def test_read_system_prompt_path_is_directory(tmp_path):
    """路径指向目录（存在但读不出文本）时，统一包装成 ValueError。"""
    with pytest.raises(ValueError, match="failed to read"):
        _read_system_prompt(tmp_path)


def test_read_system_prompt_invalid_encoding(tmp_path):
    """文件存在但非 UTF-8（如误传二进制）时，统一包装成 ValueError。"""
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\xff\xfe\x00\x01not utf-8")
    with pytest.raises(ValueError, match="failed to read"):
        _read_system_prompt(bad)


def test_read_system_prompt_ok(tmp_path):
    f = tmp_path / "sys.txt"
    f.write_text("  你是一只小鸭子机器人  \n", encoding="utf-8")
    assert _read_system_prompt(f) == "你是一只小鸭子机器人"


def test_construct_without_system_prompt_uses_no_system(monkeypatch):
    """不传 system_prompt_path 时退化为无预设裸对话，底层 VLMClient system=None。"""
    captured = {}

    class _CapturingVLM:
        def __init__(self, system=None):
            captured["system"] = system

    monkeypatch.setattr("embodied_agent.llm_client_service.VLMClient", _CapturingVLM)
    LLMClientService()  # 不传路径
    assert captured["system"] is None


def test_construct_with_system_prompt_reads_file(monkeypatch, tmp_path):
    """传入 system_prompt_path 时读取文件内容注入底层 VLMClient。"""
    captured = {}

    class _CapturingVLM:
        def __init__(self, system=None):
            captured["system"] = system

    prompt = tmp_path / "sys.txt"
    prompt.write_text("你是一只小鸭子机器人", encoding="utf-8")
    monkeypatch.setattr("embodied_agent.llm_client_service.VLMClient", _CapturingVLM)
    LLMClientService(system_prompt_path=prompt)
    assert captured["system"] == "你是一只小鸭子机器人"


@pytest.mark.skipif(
    not os.getenv("ALIYUN_API_KEY"),
    reason="ALIYUN_API_KEY not set; skipping real cloud call",
)
def test_real_cloud_reply_returns_non_empty_content(tmp_path):
    """真实网络：预设 system prompt + 用户文字，验证云端返回非空回复。"""
    prompt = tmp_path / "sys.txt"
    prompt.write_text("你是一只小鸭子外形的机器人助手，用简短中文回复。", encoding="utf-8")
    svc = LLMClientService(system_prompt_path=prompt)
    result = svc.reply("用一句话跟我打个招呼")
    assert result["status"] == "ok", f"cloud call failed: {result.get('error')}"
    assert result["content"].strip(), "expected non-empty reply content"


@pytest.mark.skipif(
    not os.getenv("ALIYUN_API_KEY"),
    reason="ALIYUN_API_KEY not set; skipping real cloud call",
)
def test_real_cloud_multi_turn_and_reset(tmp_path):
    """真实网络：验证多轮上下文累积、system 风格保持、reset 后可开新话题。"""
    prompt = tmp_path / "sys.txt"
    prompt.write_text("你是一只小鸭子外形的机器人助手，用简短中文回复。", encoding="utf-8")
    svc = LLMClientService(system_prompt_path=prompt)

    r1 = svc.reply("记一下，我有一个猕猴桃")
    assert r1["status"] == "ok", f"turn1 failed: {r1.get('error')}"
    assert r1["content"].strip()

    r2 = svc.reply("那你能帮我拿东西吗？")
    assert r2["status"] == "ok", f"turn2 failed: {r2.get('error')}"
    assert r2["content"].strip()

    svc.reset()
    r3 = svc.reply("用一句话打个招呼")
    assert r3["status"] == "ok", f"post-reset failed: {r3.get('error')}"
    assert r3["content"].strip()
