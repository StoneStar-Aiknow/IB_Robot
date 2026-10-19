"""Request construction for VLM visual games.

This module is intentionally a *pure* helper: it retrieves registered runtime
prompts and constructs requests for lightweight visual games (the Sorting Hat
being the first). The visual-game Gateway uses it to build the
``SceneAnalysisRequest`` executed by ``perception_service_node``.
"""

from __future__ import annotations

import json
from typing import Any

from embodied_common.visual_game_contracts import (
    get_default_visual_game_handler,
    get_visual_game_handler,
    get_visual_game_prompt,
)
from ibrobot_msgs.msg import SceneAnalysisRequest


def _resolve_handler(game_name: str, handler: str | None) -> tuple[str, dict[str, Any]]:
    handler_name = handler or get_default_visual_game_handler(game_name)
    contract = get_visual_game_handler(handler_name)
    if contract["game_name"] != game_name:
        raise ValueError(f"visual game handler '{handler_name}' is not registered for '{game_name}'")
    contract["prompt"] = get_visual_game_prompt(handler_name)
    return handler_name, contract


def build_game_request(
    game_name: str,
    *,
    handler: str | None = None,
    request_id: str,
    timeout_sec: float = 0.0,
) -> SceneAnalysisRequest:
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
    handler_name, contract = _resolve_handler(game_name, handler)
    resolved_request_id = request_id.strip()
    if not resolved_request_id:
        raise ValueError("visual game request_id must be non-empty")

    request = SceneAnalysisRequest()
    request.request_id = resolved_request_id
    request.source = f"game.{game_name}"
    request.session_id = f"{game_name.replace('_', '-')}-{resolved_request_id}"
    request.user_text = contract["prompt"]
    request.context_json = json.dumps(
        {
            "intent": "visual_game",
            "game_name": game_name,
            "entertainment_only": True,
            "handler": handler_name,
            "response_contract": contract["result_schema"],
            "required_inputs": contract["required_inputs"],
        },
        ensure_ascii=False,
    )
    request.timeout_sec = float(timeout_sec)
    return request
