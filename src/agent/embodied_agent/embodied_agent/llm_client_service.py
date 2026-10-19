"""业务层对话接口：预设 system prompt + 用户文字 → 云端对话模型回复。

对底层 embodied_common.VLMClient 的轻量封装：可选地从文件读入预设 prompt
作为 system 指令（每轮请求都带上、不进对话历史），多轮上下文完全交给
VLMClient 管理。返回底层 chat() 的结构化 dict，错误如实透传，不吞异常、
不返回空串。system prompt 的内容与来源由业务方决定，本接口不内置默认 prompt。
"""

from __future__ import annotations

import pathlib
from typing import Any

from embodied_common.vlm_api_client import VLMClient


def _read_system_prompt(prompt_path: pathlib.Path) -> str:
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ValueError(f"system prompt file not found: {prompt_path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        # 文件存在但读不出来：权限不足 / 路径是目录 / 非 UTF-8 编码等。
        # 统一包装成 ValueError，让调用方只需处理一种"prompt 文件有问题"的异常。
        raise ValueError(f"failed to read system prompt file {prompt_path}: {exc}") from exc
    if not text:
        raise ValueError(f"system prompt file is empty: {prompt_path}")
    return text


class LLMClientService:
    """有状态对话服务：持有一个 VLMClient，跨多次 reply() 复用其上下文。"""

    def __init__(
        self,
        system_prompt_path: str | pathlib.Path | None = None,
        model: str | None = None,
        vlm: VLMClient | None = None,
    ) -> None:
        """构造对话服务。

        Args:
            system_prompt_path: 预设 system prompt 文件路径；None 时不设 system
                prompt，退化为无预设的裸对话。
            model: 指定 vlm_models.yaml 中的模型名；None 时使用 defaults.model。
            vlm: 注入自定义 VLMClient（主要用于测试）；给定时忽略 system_prompt_path。
        """
        self._model = model
        if vlm is not None:
            self._vlm = vlm
            return
        system = _read_system_prompt(pathlib.Path(system_prompt_path)) if system_prompt_path else None
        self._vlm = VLMClient(system=system)

    def reply(self, user_text: str) -> dict[str, Any]:
        """发送一轮用户文字，返回底层结构化 dict（status/content/error/usage/...）。

        上下文由 VLMClient 自动累积；system prompt 每轮自动带上。
        """
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        return self._vlm.chat(user_text, model=self._model)

    def reset(self) -> None:
        """清空多轮对话历史；预设 system prompt 不受影响，后续仍会带上。"""
        self._vlm.clear_history()
