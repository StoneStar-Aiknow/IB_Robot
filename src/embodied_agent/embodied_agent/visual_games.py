"""Entry-layer data and request construction for VLM visual games.

This module is intentionally a *pure* helper: it holds the trigger aliases,
STT normalization, role prompt, allowed output values, and the request-context
construction for lightweight visual games (the Sorting Hat being the first). It
creates no ROS node, subscribes to no camera, and calls no VLM —
``task_entry_node`` uses it to translate a matched voice command into a
``SceneAnalysisRequest`` that the existing ``perception_service_node`` executes.

Prompt text and STT alias tables are carried over verbatim from the retired
``interaction_skills`` package (``skills/sorting_hat.py`` and ``stt_fix.py``).
"""

from __future__ import annotations

import json
import time
from typing import Any

from ibrobot_msgs.msg import SceneAnalysisRequest

# --- Sorting Hat -----------------------------------------------------------
# House order is intentional: Slytherin first so a "first match" scan is
# deterministic regardless of which houses the model mentions.
HOUSES: list[str] = ["斯莱特林", "格兰芬多", "拉文克劳", "赫奇帕奇"]

SORTING_HAT_PROMPT: str = (
    "这是一个纯娱乐性质的分院帽互动。你是霍格沃茨的分院帽。\n"
    "本次任务只关注画面中的这个人，请忽略桌面物品、电脑、设备、机械臂等一切背景物体，"
    "只根据这个人的外貌、表情、神态、发型、穿着、姿态和整体气质来判断他/她最适合哪个学院。\n\n"
    "四个学院的特点：\n"
    "- 斯莱特林：有野心，清楚自己想干什么，精明果断\n"
    "- 格兰芬多：勇敢，开朗，外向，富有冒险精神\n"
    "- 拉文克劳：聪明，富有智慧，喜欢思考，有创造力\n"
    "- 赫奇帕奇：忠诚，朴实，热爱自然和动物，勤劳肯干\n\n"
    "判断要求：\n"
    "1. 请根据这个人真实的外貌与气质独立判断，不要默认选择格兰芬多；四个学院都应有被选中的可能。\n"
    "2. 若画面中没有清晰可见的人，则依据可见的气质线索谨慎判断，不要硬凑。\n\n"
    "请返回一个 JSON 对象，包含以下字段：\n"
    "- scene_summary：只能填写四个学院之一（斯莱特林、格兰芬多、拉文克劳、赫奇帕奇），"
    "只写学院名，不要添加任何解释或理由。\n"
    "- visible_objects：字符串数组，只填写你从这个人身上观察到的线索"
    "（如表情、神态、发型、穿着、姿态、气质等），不要填写桌面物品或设备。\n"
    "- robot_state_summary、ee_pose_interpretation：本互动不涉及机械臂状态，可填“本互动不涉及机械臂状态”。\n"
    "- risks：字符串数组，无风险时填空数组。\n"
    "- confidence：0.0 到 1.0 之间的置信度。"
)


# STT misrecognition correction for the "分院帽" trigger word. The local
# paraformer-zh ASR frequently mishears it as "奔月帽"/"风月帽"/... and may
# insert spaces between characters. Applied before trigger-alias matching as a
# best-effort fallback; the primary mechanism is the ASR SetHotwords service.
_SPACED_STT_MAP: dict[str, str] = {
    "奔 月 帽": "分院帽",
    "风 月 帽": "分院帽",
    "丰 元 茂": "分院帽",
    "分 月 帽": "分院帽",
    "芬 院 冒": "分院帽",
    "芬 远 茂": "分院帽",
    "分 远 帽": "分院帽",
    "纷 院 帽": "分院帽",
}
_STT_MAP: dict[str, str] = {
    "奔月帽": "分院帽",
    "风月帽": "分院帽",
    "丰元茂": "分院帽",
    "分月帽": "分院帽",
    "芬院冒": "分院帽",
    "芬远茂": "分院帽",
    "分远帽": "分院帽",
    "纷院帽": "分院帽",
}


def fix_stt_errors(text: str) -> str:
    """Correct common ASR misrecognitions of game trigger words."""
    for wrong, correct in _SPACED_STT_MAP.items():
        text = text.replace(wrong, correct)
    for wrong, correct in _STT_MAP.items():
        text = text.replace(wrong, correct)
    return text


# Per-game static request data. Adding a new visual game = one entry here plus
# its enable flag / trigger aliases in robot_config. There is only one game
# today, so this is deliberately not a large registry.
_GAMES: dict[str, dict[str, Any]] = {
    "sorting_hat": {
        "prompt": SORTING_HAT_PROMPT,
        "allowed_values": HOUSES,
    },
}


def match_game(text: str, enabled_games: dict[str, Any]) -> str | None:
    """Return the first enabled game whose trigger alias is in ``text``.

    ``enabled_games`` maps game name -> policy dict with keys ``enabled`` (bool)
    and ``trigger_aliases`` (list[str]), sourced from ``robot_config``
    ``embodied.entry.visual_games``. The ASR text is STT-normalized before
    substring matching. Returns None when nothing matches so the caller falls
    back to normal task routing.
    """
    normalized = fix_stt_errors(text)
    for name, policy in enabled_games.items():
        if not isinstance(policy, dict) or not policy.get("enabled", False):
            continue
        if name not in _GAMES:
            continue
        for alias in policy.get("trigger_aliases", []):
            if alias and alias in normalized:
                return name
    return None


def build_game_request(game_name: str) -> SceneAnalysisRequest:
    """Build a SceneAnalysisRequest for a matched visual game.

    The request carries a strong role prompt in ``user_text`` plus a
    machine-readable ``response_contract`` and ``required_inputs`` in
    ``context_json``. ``perception_service_node`` executes it as an ordinary
    scene-analysis request and honors ``required_inputs`` (this game declares
    only ``primary_image``, so it succeeds with EE pose / joint state offline);
    it also enforces ``response_contract`` (``scene_summary`` must be exactly one
    of ``allowed_values``), publishing ``success=false`` with
    ``error_code=INVALID_RESPONSE_CONTRACT`` when the VLM answer falls outside the
    allowed houses.
    """
    spec = _GAMES[game_name]
    request_id = f"game-{time.time_ns()}"

    request = SceneAnalysisRequest()
    request.request_id = request_id
    request.source = f"game.{game_name}"
    request.session_id = f"{game_name.replace('_', '-')}-{request_id}"
    request.user_text = spec["prompt"]
    request.context_json = json.dumps(
        {
            "intent": "visual_game",
            "game_name": game_name,
            "entertainment_only": True,
            "response_contract": {
                "field": "scene_summary",
                "kind": "enum",
                "allowed_values": spec["allowed_values"],
            },
            "required_inputs": ["primary_image"],
        },
        ensure_ascii=False,
    )
    request.timeout_sec = 0.0  # use perception-configured default timeout
    return request
