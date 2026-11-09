"""Node boundary keeps complete hands in a verified seed and rejects stale targets."""

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs")

from manipulation_execution import mirror_session_node as module  # noqa: E402
from manipulation_execution.mirror_session.hooks import UpperBodyCommand  # noqa: E402


def test_preset_target_uses_fresh_arm_preview_and_hand_follow_without_publishing(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    raw = UpperBodyCommand((0.1,) * 14, (0.0,) * 20)
    node = SimpleNamespace(
        _preset_last_preview_at=9.9,
        _preset_last_raw=None,
        _hands=SimpleNamespace(follow_target=lambda now: (0.2,) * 20),
        _clock_s=lambda: 20.0,
    )
    target = module.MirrorSessionNode._preset_target(node, raw)
    assert target.arm == raw.arm and target.hand == (0.2,) * 20
    assert module.MirrorSessionNode._preset_target(node, None) == target
    node._preset_last_preview_at = 9.0
    assert module.MirrorSessionNode._preset_target(node, raw) is None
    node._preset_last_preview_at = 9.9
    node._hands = SimpleNamespace(follow_target=lambda now: None)
    assert module.MirrorSessionNode._preset_target(node, raw) is None


def test_verified_seed_bypasses_ordinary_hand_overlay_once():
    sent, reseeded = [], []

    def unexpected_apply(*args):
        raise AssertionError("ordinary hand tracking must not overwrite the completed named target")

    publisher = SimpleNamespace(publish=lambda value: sent.append(value.data))
    node = SimpleNamespace(
        _preset=object(),
        session=SimpleNamespace(preset_seed_emitting=True),
        _hands=SimpleNamespace(reseed=lambda values, now: reseeded.append((values, now)), apply=unexpected_apply),
        _hands_last=None,
        _clock_s=lambda: 20.0,
        _publish_enabled=True,
        _arm_publisher=publisher,
        _hand_publisher=publisher,
        _arm_preview=publisher,
        _hand_preview=publisher,
    )
    seed = UpperBodyCommand((0.1,) * 14, (0.2,) * 20)
    module.MirrorSessionNode._publish(node, seed)
    assert reseeded == [(seed.hand, 20.0)]
    assert sent[1] == sent[3] == list(seed.hand)
