"""Cross-package visual-game request/response contract regression tests."""

import json

from embodied_agent.visual_games import build_game_request
from embodied_common.scene_analysis import SceneAnalysis
from perception_service.perception_service_node import PerceptionServiceNode


def test_game_request_contract_matches_perception_enforcer():
    request = build_game_request("sorting_hat", request_id="contract-test-1")
    context = json.loads(request.context_json)

    valid = SceneAnalysis("赫奇帕奇", [], "", "", [], 1.0)
    invalid = SceneAnalysis("不存在的学院", [], "", "", [], 1.0)
    assert PerceptionServiceNode._check_response_contract(context, valid) is None
    assert PerceptionServiceNode._check_response_contract(context, invalid) is not None
