"""Configure the composer once per session with the official model and the OmniHand profile."""

from pathlib import Path

import pytest

MJCF = Path.home() / ".cache/ibrobot/agibot_x2_urdf/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.3.0/scene.xml"
PROFILE = Path(__file__).resolve().parents[3] / "robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"


@pytest.fixture(scope="session", autouse=True)
def configured():
    if not MJCF.exists():
        pytest.skip("official X2 MuJoCo model not fetched (ros2 run aimdk_robot aimdk_fetch_x2_mjcf)")
    from pose_composer import model

    return model.configure(MJCF, PROFILE)
