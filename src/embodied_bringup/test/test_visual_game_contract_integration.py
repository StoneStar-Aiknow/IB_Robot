"""Cross-package visual-game contract integration tests.

This lives in embodied_bringup (the orchestration package) rather than
perception_service so the generic perception package does not test-depend on the
entry-layer business package.
"""

import json

from embodied_agent.visual_games import build_game_request
from embodied_common.scene_analysis import SceneAnalysis
from perception_service.perception_service_node import PerceptionServiceNode


def _analysis(scene_summary: str) -> SceneAnalysis:
    return SceneAnalysis(
        scene_summary=scene_summary,
        visible_objects=[],
        robot_state_summary="",
        ee_pose_interpretation="",
        risks=[],
        confidence=1.0,
    )


def test_sorting_hat_request_contract_matches_perception_enforcer():
    request = build_game_request("sorting_hat")
    context = json.loads(request.context_json)

    assert PerceptionServiceNode._check_response_contract(context, _analysis("赫奇帕奇")) is None
    assert PerceptionServiceNode._check_response_contract(context, _analysis("不存在的学院")) is not None
