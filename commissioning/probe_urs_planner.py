"""Exercise the configured model with the site catalog; no ROS node or execution client is created."""

import dataclasses
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from ibrobot_agent.contracts import AgentRequest
from ibrobot_agent.node import _planner_from_config

from robot_config.interface_binding import bind_robot_interfaces
from robot_config.loader import load_robot_config_dict
from robot_runtime.interface_description import build_description
from robot_runtime.profile import load_profile
from robot_skill_cli.catalog import _snapshot_capability_view, compile_local_snapshot

root = Path(__file__).resolve().parents[1]
path = root / "commissioning/robots/hz_site_live.yaml"
profile = load_profile(root / "src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml")
config = load_robot_config_dict(path, defer_interface_binding=True)
config = bind_robot_interfaces(config, build_description(profile, use_sim=False))
catalog = _snapshot_capability_view(compile_local_snapshot(config, path))
planner = _planner_from_config(json.dumps(config["embodied"]["agent"]["planner"]))
history = [
    {"role": "user", "content": "我想跟你握一握手"},
    {"role": "assistant", "content": "执行失败，请查看执行记录或联系操作员。"},
    {"role": "user", "content": "抓取苹果"},
    {"role": "assistant", "content": "目前只能做挥手、鼓掌这类动作，还抓不了东西，需要先切换到机械臂规划模式。"},
]
cases = [
    ("named_motion", "抓取苹果", [], ["pick_object"]),
    ("named_motion", "抓取苹果", history, ["pick_object"]),
    ("moveit_planning", "先挥手，然后抓取苹果", history, ["wave_hand", "pick_object"]),
    ("named_motion", "不要抓取苹果", [], []),
]
results = []
for index, (mode, text, messages, expected) in enumerate(cases):
    request = AgentRequest(
        schema_version=1,
        request_id=f"urs-planning-only-{index}",
        session_id="urs-planning-only",
        channel_id="offline_probe",
        principal_id="local_operator",
        robot_scope=config["name"],
        text=text,
        received_at=datetime.now(timezone.utc),
    )
    view = {**catalog, "active_control_mode": mode, "control_plane_ready": True, "motion_authorized": False}
    outcome = planner.plan(request, {"messages": messages, "clarification": None}, view, threading.Event())
    actual = [step.skill_name for step in outcome.steps]
    result = {"mode": mode, "text": text, "history": bool(messages), "outcome": dataclasses.asdict(outcome)}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert actual == expected, (actual, expected)
    if expected:
        assert outcome.kind == "workflow", outcome.kind
        for step in outcome.steps:
            if step.skill_name == "pick_object":
                assert "apple" in step.target_name.lower(), step.target_name
    else:
        assert outcome.kind != "workflow", outcome.kind
    results.append(result)
(root / "commissioning/logs/urs_planner_probe_20261105.json").write_text(
    json.dumps({"result": "PASS", "execution_client_created": False, "cases": results}, ensure_ascii=False, indent=2)
)
print("RESULT=PASS; planning only; no execution submitted", flush=True)
