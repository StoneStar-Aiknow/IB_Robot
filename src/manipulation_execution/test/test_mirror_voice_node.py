"""The voice node's ROS adapter: what it subscribes to, and the utterance door.

The speaker and the transition filter are tested without ROS
(test_mirror_voice_speaker.py). What is left here is the plumbing the door
gets wrong: whether the utterance subscription exists at all, which of the
message's fields admit a reply, and whether the mirror state the gate reads
is the current one rather than a transition.
"""

from __future__ import annotations

import json

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs.msg")

from embodied_common.mirror_session_contract import (  # noqa: E402
    STATE_MIRRORING,
    STATE_PERIOD_SEC,
    STATE_STALE_SEC,
    STATE_STANDBY,
    STATE_TOPIC,
    STATES,
)
from embodied_common.mirror_voice_contract import parse_voice_section, to_json  # noqa: E402
from embodied_common.visual_interaction_contract import UTTERANCE_TOPIC  # noqa: E402


def voice_json(utterance_topic=UTTERANCE_TOPIC):
    return to_json(parse_voice_section({"enabled": True, "utterance_topic": utterance_topic}))


def voice_node(utterance_topic=UTTERANCE_TOPIC):
    """A live node, built without a launch file; the caller shuts ROS down."""
    from rclpy.parameter import Parameter

    from manipulation_execution.mirror_voice_node import MirrorVoiceNode

    rclpy.init()
    try:
        node = MirrorVoiceNode(parameter_overrides=[Parameter("voice_json", value=voice_json(utterance_topic))])
        assert node.speaker is not None, "the test section is enabled; the node must start"
        return node, rclpy
    except Exception:
        rclpy.shutdown()
        raise


def utterance(text="这个玩偶真有趣", *, kind="object", suppress_reason="", latest=(0, 0)):
    from ibrobot_msgs.msg import VisualInteractionUtterance

    message = VisualInteractionUtterance()
    message.text = text
    message.kind = kind
    message.suppress_reason = suppress_reason
    message.latest_start_at.sec = latest[0]
    message.latest_start_at.nanosec = latest[1]
    return message


def state_message(state, transition=None):
    from std_msgs.msg import String

    return String(data=json.dumps({"state": state, "last_transition": transition}))


class TestSubscriptions:
    def test_it_subscribes_the_state_and_the_utterance_topic(self):
        node, rclpy = voice_node()
        try:
            topics = {sub.topic_name for sub in node.subscriptions}

            assert STATE_TOPIC in topics
            assert UTTERANCE_TOPIC in topics
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_without_an_utterance_topic_it_subscribes_only_the_state(self):
        """Empty is the default: today's behaviour, transition phrases only."""
        node, rclpy = voice_node(utterance_topic="")
        try:
            topics = {sub.topic_name for sub in node.subscriptions}

            assert STATE_TOPIC in topics
            assert UTTERANCE_TOPIC not in topics
        finally:
            node.destroy_node()
            rclpy.shutdown()


class TestStateRecording:
    def test_the_current_state_and_its_arrival_are_recorded(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            assert node._state == STATE_MIRRORING
            assert node._state_at is not None
            assert node._mirroring_now() is True
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_every_other_state_closes_the_gate(self):
        node, rclpy = voice_node()
        try:
            for state in STATES:
                if state == STATE_MIRRORING:
                    continue
                node._on_state(state_message(state))

                assert node._mirroring_now() is False, state
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_state_that_went_stale_closes_the_gate(self):
        """Two missed heartbeats: unknown, which gates off — the conservative side."""
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))
            node._state_at -= STATE_STALE_SEC + 1.0

            assert node._mirroring_now() is False
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_malformed_message_leaves_the_last_state_aging(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))
            heard_at = node._state_at

            from std_msgs.msg import String

            node._on_state(String(data="not json"))

            assert node._state == STATE_MIRRORING
            assert node._state_at == heard_at
        finally:
            node.destroy_node()
            rclpy.shutdown()


class TestTheUtteranceDoor:
    """One message per visual interaction round; the door keeps the ones worth
    saying and counts the rest, so a silent robot can be told from a muted one."""

    def admitted(self, node):
        status = node.speaker.status()
        return status["utterances"], status["queue"]

    def test_a_speakable_untimed_reply_is_admitted(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))
            node._on_utterance(utterance())

            # The synthesis service does not exist in this test, so a live
            # utterance waits at the head of the queue — which is itself the
            # proof it was admitted.
            assert self.admitted(node) == (1, 1)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_quiet_kind_is_refused(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            node._on_utterance(utterance(kind="none"))
            node._on_utterance(utterance(kind="uncertain"))

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["kind"] == 2
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_suppressed_round_is_refused(self):
        """The session already decided against it; a quiet round is not a reply."""
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            node._on_utterance(utterance(suppress_reason="MIRROR_INACTIVE"))

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["suppressed"] == 1
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_an_empty_text_is_refused(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            node._on_utterance(utterance(text="   "))

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["empty"] == 1
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_text_zipvoice_cannot_speak_is_refused(self):
        """The phrase check refuses such text at launch; an utterance is checked
        here, with the same rule."""
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            node._on_utterance(utterance(text="这个 toy 很有趣"))

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["ascii"] == 1
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_round_from_before_the_mirror_started_is_refused(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_STANDBY))

            node._on_utterance(utterance())

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["mirror"] == 1
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_round_from_a_stale_mirror_state_is_refused(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))
            node._state_at -= STATE_STALE_SEC + 1.0

            node._on_utterance(utterance())

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["mirror"] == 1
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_reply_past_its_latest_start_is_refused(self):
        """The evidence went stale while the reply waited; saying it would be
        commenting on a picture from ages ago."""
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            node._on_utterance(utterance(latest=(1, 0)))

            assert self.admitted(node) == (0, 0)
            assert node._utterance_refused["late"] == 1
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_a_reply_with_its_start_still_ahead_is_admitted(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_MIRRORING))

            node._on_utterance(utterance(latest=(int(node._now() + 60.0), 0)))

            assert self.admitted(node) == (1, 1)
        finally:
            node.destroy_node()
            rclpy.shutdown()


class TestStatus:
    def test_it_carries_the_mirror_state_and_the_refusal_counts(self):
        node, rclpy = voice_node()
        try:
            node._on_state(state_message(STATE_STANDBY))
            node._on_utterance(utterance(kind="none"))
            captured = {}
            node._status_publisher.publish = lambda message: captured.setdefault("data", message.data)
            node._publish_status()

            status = json.loads(captured["data"])

            assert status["mirror_state"] == STATE_STANDBY
            assert status["utterance_refused"] == {
                "kind": 1,
                "suppressed": 0,
                "empty": 0,
                "ascii": 0,
                "mirror": 0,
                "late": 0,
            }
        finally:
            node.destroy_node()
            rclpy.shutdown()


class TestContractAgreement:
    """The node reads the state through the contract's constants; they must say
    what the publisher actually sends."""

    def test_the_states_match_the_session_enum(self):
        from manipulation_execution.mirror_session.core import SessionState

        assert tuple(state.value for state in SessionState) == STATES

    def test_staleness_means_at_least_two_missed_heartbeats(self):
        assert STATE_STALE_SEC >= 2 * STATE_PERIOD_SEC
