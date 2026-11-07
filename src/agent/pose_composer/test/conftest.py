"""Configure the composer once per session with the official model and the OmniHand profile."""

from pathlib import Path

import pytest
import yaml

MJCF = Path.home() / ".cache/ibrobot/agibot_x2_urdf/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.3.0/scene.xml"
PROFILE = Path(__file__).resolve().parents[3] / "robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"


@pytest.fixture(scope="session", autouse=True)
def configured():
    if not MJCF.exists():
        pytest.skip("official X2 MuJoCo model not fetched (ros2 run aimdk_robot aimdk_fetch_x2_mjcf)")
    from pose_composer import model

    anchors = yaml.safe_load((Path(__file__).parent / "fixtures/x2_home_task_poses.yaml").read_text())
    return model.configure(MJCF, PROFILE, task_poses=anchors)


@pytest.fixture
def agent_session(tmp_path):
    """Shared Agent/SQLite wiring; each scenario owns and closes its service."""
    import time
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from ibrobot_agent.contracts import AgentRequest
    from ibrobot_agent.request_store import SQLiteRequestStore
    from ibrobot_agent.service import AgentService

    events = []
    store = SQLiteRequestStore(tmp_path / "requests.sqlite3")

    def create(**options):
        return AgentService(
            catalog=SimpleNamespace(
                get_status=lambda: dict(
                    control_plane_ready=True, registry_epoch="epoch", registry_generation=1, registry_digest="catalog"
                ),
                get_catalog=lambda _: dict(
                    skills=[dict(name="free_pose", planner_visible=True, semantic_level="skill")]
                ),
            ),
            store=store,
            event_sink=SimpleNamespace(publish=events.append),
            **options,
        )

    def send(service, text, expected, robot_scope="mock_x2"):
        request = AgentRequest(
            1, str(len(events)), "session", "test", "operator", robot_scope, text, datetime.now(timezone.utc)
        )
        assert service.send_message(request).accepted
        deadline = time.monotonic() + 5
        while not any(event.request_key == request.to_key() and event.state == expected for event in events):
            assert time.monotonic() < deadline, events
            time.sleep(0.005)
        return request

    return SimpleNamespace(create=create, send=send, events=events, store=store)
