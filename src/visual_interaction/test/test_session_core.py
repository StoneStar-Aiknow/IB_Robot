"""One round at a time: when the session speaks, and when it stays quiet."""

import pytest

from embodied_common.visual_interaction_contract import (
    COVERED_ESTIMATED,
    KIND_UNCERTAIN,
    PARTIAL_COVERED,
    SUPPRESS_AMBIGUOUS,
    SUPPRESS_COOLDOWN,
    SUPPRESS_DUPLICATE,
    SUPPRESS_MODEL_NOT_READY,
    SUPPRESS_MUTED,
    SUPPRESS_NO_NEW_TOPIC,
    SUPPRESS_PRIVACY,
    SUPPRESS_SCHEMA_INVALID,
    SUPPRESS_STALE,
    SUPPRESS_UNCERTAIN,
    SUPPRESS_UNCONFIRMED,
    SUPPRESS_WAITING_NEXT_VISITOR,
    SUPPRESSED,
)
from visual_interaction.session.core import SessionConfig, VisualInteractionSession
from visual_interaction.session.fake_vl import ScriptedVision, answer, quiet
from visual_interaction.session.hooks import (
    ActorSignal,
    ChangeGate,
    Delivery,
    RoundInputs,
    SessionHooks,
    SpeechPort,
    Window,
)
from visual_interaction.session.topics import TopicKey

HERE = (100.0, 100.0, 200.0, 400.0)


class RecordingSpeech(SpeechPort):
    """Accepts everything and remembers it, with a configurable audio length."""

    def __init__(self, *, duration: float = 2.0, accepted: bool = True, partial: bool = False, detail: str = ""):
        self.duration = duration
        self.accepted = accepted
        self.partial = partial
        self.detail = detail
        self.spoken: list[tuple[str, str, float]] = []

    def deliver(self, text, *, event_id, now):
        self.spoken.append((text, event_id, now))
        return Delivery(
            accepted=self.accepted,
            estimated_end=now + self.duration,
            partial=self.partial,
            detail=self.detail,
        )


class ClosedGate(ChangeGate):
    def should_observe(self, inputs):
        return False


def inputs(*, now=10.0, stamp=None, present=True, known=True, bbox=HERE, others=()):
    """One round's inputs, with the window stamped just before ``now``."""
    stamp = now - 0.1 if stamp is None else stamp
    return RoundInputs(
        window=Window(
            frames=(b"f1", b"f2", b"f3", b"f4"), stamps=(stamp - 0.3, stamp - 0.2, stamp - 0.1, stamp), received_at=now
        ),
        actor=ActorSignal(present=present, known=known, bbox=bbox, stamp=stamp, others=others),
        now=now,
    )


def session(*, vision=None, speech=None, gate=None, **config):
    hooks = SessionHooks(vision=vision, speech=speech, gate=gate)
    return VisualInteractionSession(hooks=hooks, config=SessionConfig(**config))


def speak_once(sess, *, start=10.0, step=1.0):
    """Drive rounds until one is spoken, returning every result."""
    results = []
    for index in range(6):
        results.append(sess.step(inputs(now=start + index * step)))
        if results[-1].spoken:
            break
    return results


class TestAFullRound:
    def test_two_windows_then_it_speaks(self):
        speech = RecordingSpeech()
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=speech)

        first = sess.step(inputs(now=10.0))
        second = sess.step(inputs(now=11.0))

        assert not first.spoken
        assert first.reason == SUPPRESS_UNCONFIRMED
        assert second.spoken
        assert second.text == "你举起的玩偶真有趣。"
        assert speech.spoken[0][0] == "你举起的玩偶真有趣。"

    def test_the_result_carries_what_the_adapter_publishes(self):
        sess = session(vision=ScriptedVision.of(answer("hat", region="head")))

        result = speak_once(sess)[-1]

        assert result.spoken
        assert result.topic == TopicKey.of("hat", "head")
        assert result.observation.category == "hat"
        assert result.session_id == sess.session_id
        assert result.actor_epoch == sess.actor_epoch
        assert result.event_id.startswith(sess.session_id)
        assert result.confirmations == 2
        assert result.model_called
        assert result.evidence_stamp > 0

    def test_a_spoken_topic_settles_and_never_repeats(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=0.0))
        speak_once(sess)

        later = sess.step(inputs(now=100.0))

        assert not later.spoken
        assert later.reason == SUPPRESS_DUPLICATE
        assert sess.ledger().coverage_of(TopicKey.of("toy", "hand")) == COVERED_ESTIMATED

    def test_a_second_topic_is_still_answerable(self):
        vision = ScriptedVision.of(answer("toy"), answer("toy"), answer("hat", region="head"))
        sess = session(vision=vision, speech=RecordingSpeech(duration=0.0), min_gap_sec=0.0)
        speak_once(sess)

        third = sess.step(inputs(now=30.0))
        fourth = sess.step(inputs(now=31.0))

        assert not third.spoken
        assert fourth.spoken
        assert fourth.topic == TopicKey.of("hat", "head")


class TestQuietRounds:
    def test_a_none_answer_reports_a_working_silence(self):
        sess = session(vision=ScriptedVision.of(quiet()))

        result = sess.step(inputs())

        assert result.quiet
        assert result.reason == SUPPRESS_NO_NEW_TOPIC
        assert result.model_called

    def test_an_uncertain_answer_reports_its_own_reason(self):
        sess = session(vision=ScriptedVision.of({"kind": KIND_UNCERTAIN}))

        assert sess.step(inputs()).reason == SUPPRESS_UNCERTAIN

    def test_an_unreachable_model_is_not_a_silence(self):
        """MODEL_NOT_READY is the difference a bystander cannot hear."""
        sess = session(vision=ScriptedVision.unreachable())

        result = sess.step(inputs())

        assert result.reason == SUPPRESS_MODEL_NOT_READY
        assert result.model_called

    def test_a_closed_gate_skips_the_model_entirely(self):
        vision = ScriptedVision.of(answer("toy"))
        sess = session(vision=vision, gate=ClosedGate())

        result = sess.step(inputs())

        assert result.reason == SUPPRESS_NO_NEW_TOPIC
        assert not result.model_called
        assert vision.call_count == 0

    def test_a_malformed_answer_becomes_silence_with_its_reason(self):
        sess = session(vision=ScriptedVision.of('{"kind": "display"'))

        assert sess.step(inputs()).reason == SUPPRESS_SCHEMA_INVALID

    def test_a_disallowed_category_is_refused(self):
        sess = session(
            vision=ScriptedVision.of(
                {"kind": "display", "category": "pendant", "region": "chest", "reply": "你的挂坠很特别。"}
            )
        )

        assert sess.step(inputs()).reason == SUPPRESS_PRIVACY

    def test_a_deployment_may_narrow_the_categories(self):
        sess = session(vision=ScriptedVision.of(answer("hat", region="head")), categories=("toy",))

        assert sess.step(inputs()).reason == SUPPRESS_PRIVACY

    def test_an_empty_category_set_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="can never speak"):
            SessionConfig(categories=())


class TestTargetFirst:
    def test_no_target_means_no_model_call(self):
        vision = ScriptedVision.of(answer("toy"))
        sess = session(vision=vision)

        result = sess.step(inputs(present=False, bbox=None))

        assert result.reason == SUPPRESS_STALE
        assert vision.call_count == 0

    def test_two_people_close_together_stay_quiet(self):
        vision = ScriptedVision.of(answer("toy"))
        sess = session(vision=vision)
        sess.step(inputs(now=10.0))

        result = sess.step(inputs(now=10.5, others=(HERE, (210.0, 100.0, 310.0, 400.0))))

        assert result.reason == SUPPRESS_AMBIGUOUS

    def test_an_unknown_frame_is_not_a_target_loss_for_the_session(self):
        """It reports stale for this round but the lock and session survive."""
        sess = session(vision=ScriptedVision.of(answer("toy")))
        sess.step(inputs(now=10.0))
        before = sess.session_id

        result = sess.step(inputs(now=10.2, present=False, known=False, bbox=None))

        assert result.reason == SUPPRESS_STALE
        assert sess.session_id == before
        assert sess.tracker.locked

    def test_a_target_jump_voids_collected_evidence(self):
        """A relock means the previous window rested on someone else."""
        sess = session(vision=ScriptedVision.of(answer("toy")))
        sess.step(inputs(now=10.0))

        jumped = sess.step(inputs(now=10.2, bbox=(600.0, 100.0, 700.0, 400.0)))

        assert not jumped.spoken
        assert jumped.confirmations == 1


class TestCooldown:
    def test_a_fresh_remark_waits_out_the_gap(self):
        vision = ScriptedVision.of(answer("toy"), answer("toy"), answer("hat", region="head"))
        sess = session(vision=vision, speech=RecordingSpeech(duration=2.0), min_gap_sec=10.0)
        spoken = speak_once(sess)[-1]

        assert spoken.spoken
        # Spoken at 11.0, two seconds of audio, ten seconds of quiet.
        assert sess.step(inputs(now=20.0)).reason == SUPPRESS_COOLDOWN
        assert sess.step(inputs(now=23.1)).reason != SUPPRESS_COOLDOWN

    def test_the_gap_is_measured_from_the_estimated_end_not_the_request(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=5.0), min_gap_sec=1.0)
        speak_once(sess)

        # Spoken at 11.0 + 5s audio + 1s gap = 17.0.
        assert sess.step(inputs(now=16.0)).reason == SUPPRESS_COOLDOWN
        assert sess.step(inputs(now=17.1)).reason != SUPPRESS_COOLDOWN

    def test_the_cooldown_check_precedes_the_model(self):
        vision = ScriptedVision.of(answer("toy"))
        sess = session(vision=vision, speech=RecordingSpeech(duration=2.0), min_gap_sec=10.0)
        speak_once(sess)
        calls_after_speaking = vision.call_count

        sess.step(inputs(now=14.0))

        assert vision.call_count == calls_after_speaking


class TestStaleEvidence:
    def test_an_old_picture_is_not_spoken(self):
        """Checked before delivery, because that is when it would be heard."""
        sess = session(vision=ScriptedVision.of(answer("toy")), max_evidence_age_sec=1.0)

        sess.step(inputs(now=10.0, stamp=9.9))
        result = sess.step(inputs(now=11.0, stamp=9.95))

        assert not result.spoken
        assert result.reason == SUPPRESS_STALE

    def test_a_fresh_picture_within_the_age_is_spoken(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), max_evidence_age_sec=5.0)

        results = speak_once(sess)

        assert results[-1].spoken


class TestDeliveryOutcomes:
    def test_a_refused_delivery_leaves_the_topic_unspoken(self):
        speech = RecordingSpeech(accepted=False)
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=speech, min_gap_sec=0.0)

        results = speak_once(sess)

        assert not any(result.spoken for result in results)
        assert sess.ledger().coverage_of(TopicKey.of("toy", "hand")) == ""

    def test_a_partial_delivery_blocks_a_repeat(self):
        """Some of it may have been heard, so it is not re-said from the top."""
        sess = session(
            vision=ScriptedVision.of(answer("toy")),
            speech=RecordingSpeech(partial=True, duration=0.0),
            min_gap_sec=0.0,
        )
        speak_once(sess)

        assert sess.ledger().coverage_of(TopicKey.of("toy", "hand")) == PARTIAL_COVERED
        assert sess.step(inputs(now=50.0)).reason == SUPPRESS_DUPLICATE

    def test_a_topic_is_reserved_while_it_is_being_delivered(self):
        seen = {}

        class SlowSpeech(SpeechPort):
            def deliver(self, text, *, event_id, now):
                seen["coverage"] = sess.ledger().coverage_of(TopicKey.of("toy", "hand"))
                return Delivery(accepted=True, estimated_end=now)

        sess = session(vision=ScriptedVision.of(answer("toy")), speech=SlowSpeech())
        speak_once(sess)

        assert seen["coverage"] == "in_flight"


class TestInterrupt:
    def test_a_visitor_stop_suppresses_the_topic_and_mutes(self):
        """Leaving it unspoken would bring the same remark back next round."""
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=0.0))
        speak_once(sess)
        topic = TopicKey.of("toy", "hand")

        sess.interrupt(topic, by_user=True)

        assert sess.ledger().coverage_of(topic) == SUPPRESSED
        assert sess.muted
        assert sess.step(inputs(now=60.0)).reason == SUPPRESS_MUTED

    def test_a_content_invalid_cancel_leaves_the_topic_unspoken(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=0.0))
        speak_once(sess)
        topic = TopicKey.of("toy", "hand")

        sess.interrupt(topic, by_user=False)

        assert sess.ledger().coverage_of(topic) == ""
        assert not sess.muted

    def test_only_resume_clears_a_mute(self):
        """A timer must not undo a request for quiet."""
        sess = session(vision=ScriptedVision.of(answer("toy")))
        sess.mute()

        assert sess.step(inputs(now=10_000.0)).reason == SUPPRESS_MUTED

        sess.resume()
        assert sess.step(inputs(now=10_001.0)).reason != SUPPRESS_MUTED

    def test_muting_precedes_the_model(self):
        vision = ScriptedVision.of(answer("toy"))
        sess = session(vision=vision)
        sess.mute()

        sess.step(inputs())

        assert vision.call_count == 0


class TestSessionBoundaries:
    def test_a_hand_over_forgets_covered_topics(self):
        """A queue rarely produces a confirmed absence, so this is the normal reset."""
        sess = session(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=0.0), min_gap_sec=0.0)
        speak_once(sess)
        first_session = sess.session_id

        sess.next_visitor(now=20.0)

        assert sess.session_id != first_session
        assert sess.ledger().covered_keys() == ()
        assert speak_once(sess, start=21.0)[-1].spoken

    def test_a_confirmed_absence_starts_a_new_session(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), absent_reset_sec=30.0)
        sess.step(inputs(now=10.0))
        first = sess.session_id

        sess.step(inputs(now=15.0, present=False, bbox=None))
        sess.step(inputs(now=50.0, present=False, bbox=None))

        assert sess.session_id != first

    def test_a_dropout_does_not_start_a_new_session(self):
        """Resetting coverage under a visitor who never moved is worse than waiting."""
        sess = session(vision=ScriptedVision.of(answer("toy")), absent_reset_sec=30.0)
        sess.step(inputs(now=10.0))
        first = sess.session_id

        for now in (15.0, 40.0, 90.0):
            sess.step(inputs(now=now, present=False, known=False, bbox=None))

        assert sess.session_id == first

    def test_the_duration_limit_pauses_a_busy_stand(self):
        """It is a missed-hand-over signal, not a reset.

        An earlier revision asserted the opposite — that the limit started a
        new session by itself — which repeated every remark to the same
        visitor on a timer. See TestMaxDurationWaitsForAHandOver.
        """
        sess = session(vision=ScriptedVision.of(answer("toy")), max_session_sec=60.0)
        sess.step(inputs(now=10.0))
        first = sess.session_id

        result = sess.step(inputs(now=75.0))

        assert sess.session_id == first
        assert result.reason == SUPPRESS_WAITING_NEXT_VISITOR

    def test_a_mute_survives_a_new_session(self):
        """It is about the robot's behaviour, not about who is standing there."""
        sess = session(vision=ScriptedVision.of(answer("toy")))
        sess.mute()

        sess.next_visitor(now=20.0)

        assert sess.muted


class TestHookNotifications:
    def test_every_round_reports_its_outcome(self):
        seen = []

        class Recording(SessionHooks):
            def on_round(self, round_inputs, outcome):
                seen.append(outcome)

        hooks = Recording(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=0.0))
        sess = VisualInteractionSession(hooks=hooks, config=SessionConfig())

        sess.step(inputs(now=10.0))
        sess.step(inputs(now=11.0))

        assert seen == [SUPPRESS_UNCONFIRMED, ""]

    def test_a_raising_hook_does_not_lose_the_decision(self):
        class Breaking(SessionHooks):
            def on_round(self, round_inputs, outcome):
                raise RuntimeError("diagnostics are down")

        hooks = Breaking(vision=ScriptedVision.of(answer("toy")), speech=RecordingSpeech(duration=0.0))
        sess = VisualInteractionSession(hooks=hooks, config=SessionConfig())

        sess.step(inputs(now=10.0))
        result = sess.step(inputs(now=11.0))

        assert result.spoken

    def test_a_session_end_is_reported_with_its_reason(self):
        seen = []

        class Recording(SessionHooks):
            def on_session_end(self, session_id, reason):
                seen.append((session_id, reason))

        hooks = Recording(vision=ScriptedVision.of(quiet()))
        sess = VisualInteractionSession(hooks=hooks, config=SessionConfig())
        sess.step(inputs(now=10.0))

        sess.next_visitor(now=20.0)

        assert seen == [("s1", "handover")]


class TestDefaults:
    def test_the_skeleton_runs_silently_with_no_model(self):
        """The default vision port answers "nothing", so nothing is ever said."""
        sess = VisualInteractionSession()

        for index in range(5):
            result = sess.step(inputs(now=10.0 + index))

        assert result.quiet
        assert result.reason == SUPPRESS_NO_NEW_TOPIC

    def test_the_default_speech_port_accepts_and_estimates_now(self):
        sess = session(vision=ScriptedVision.of(answer("toy")))

        result = speak_once(sess)[-1]

        assert result.spoken


class TestWindow:
    def test_a_window_needs_a_stamp_per_frame(self):
        with pytest.raises(ValueError, match="every frame needs a stamp"):
            Window(frames=(b"a", b"b"), stamps=(1.0,), received_at=1.0)

    def test_an_empty_window_is_refused(self):
        with pytest.raises(ValueError, match="at least one frame"):
            Window(frames=(), stamps=(), received_at=1.0)

    def test_a_window_reports_its_span(self):
        window = Window(frames=(b"a", b"b"), stamps=(1.0, 3.0), received_at=3.5)

        assert window.oldest_stamp == 1.0
        assert window.newest_stamp == 3.0
        assert window.span == 2.0


class TestConfirmationChainBreaks:
    """A round that saw nothing it can vouch for must not bridge two looks.

    Review finding: "seen twice at some point" was passing for "confirmed by
    two windows", so an occlusion or a model fault let an old look and a new
    one add up.
    """

    def test_uncertain_breaks_the_chain(self):
        vision = ScriptedVision.of(answer("toy"), {"kind": KIND_UNCERTAIN}, answer("toy"))
        sess = session(vision=vision, min_gap_sec=0.0)

        first = sess.step(inputs(now=10.0))
        middle = sess.step(inputs(now=11.0))
        third = sess.step(inputs(now=12.0))

        assert first.confirmations == 1
        assert middle.reason == SUPPRESS_UNCERTAIN
        assert not third.spoken
        assert third.confirmations == 1

    def test_an_unreachable_model_breaks_the_chain(self):
        vision = ScriptedVision.of(answer("toy"), None, answer("toy"))
        sess = session(vision=vision, min_gap_sec=0.0)

        sess.step(inputs(now=10.0))
        sess.step(inputs(now=11.0))

        assert not sess.step(inputs(now=12.0)).spoken

    def test_an_unreadable_answer_breaks_the_chain(self):
        vision = ScriptedVision.of(answer("toy"), '{"kind": "display"', answer("toy"))
        sess = session(vision=vision, min_gap_sec=0.0)

        sess.step(inputs(now=10.0))
        sess.step(inputs(now=11.0))

        assert not sess.step(inputs(now=12.0)).spoken

    def test_a_lost_target_breaks_the_chain(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), min_gap_sec=0.0)

        sess.step(inputs(now=10.0))
        sess.step(inputs(now=11.0, present=False, bbox=None))

        assert not sess.step(inputs(now=12.0)).spoken

    def test_an_uninterrupted_pair_still_confirms(self):
        """The break must not make confirmation impossible."""
        sess = session(vision=ScriptedVision.of(answer("toy")), min_gap_sec=0.0)

        sess.step(inputs(now=10.0))

        assert sess.step(inputs(now=11.0)).spoken


class TestMaxDurationWaitsForAHandOver:
    """Review finding: the duration limit was resetting coverage by itself.

    With the same visitor still standing there that repeats every remark on a
    timer, which is the behaviour the session exists to prevent.
    """

    def test_it_pauses_instead_of_resetting(self):
        sess = session(
            vision=ScriptedVision.of(answer("toy")),
            speech=RecordingSpeech(duration=0.0),
            max_session_sec=120.0,
            min_gap_sec=0.0,
        )
        speak_once(sess)
        first = sess.session_id

        later = sess.step(inputs(now=200.0))

        assert sess.session_id == first
        assert later.reason == SUPPRESS_WAITING_NEXT_VISITOR

    def test_it_keeps_waiting_until_an_attendant_acts(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), max_session_sec=60.0, min_gap_sec=0.0)
        sess.step(inputs(now=10.0))
        sess.step(inputs(now=100.0))

        for now in (110.0, 200.0, 500.0):
            assert sess.step(inputs(now=now)).reason == SUPPRESS_WAITING_NEXT_VISITOR

    def test_a_hand_over_ends_the_wait_and_clears_coverage(self):
        sess = session(
            vision=ScriptedVision.of(answer("toy")),
            speech=RecordingSpeech(duration=0.0),
            max_session_sec=120.0,
            min_gap_sec=0.0,
        )
        speak_once(sess)
        sess.step(inputs(now=200.0))

        sess.next_visitor(now=210.0)

        assert speak_once(sess, start=211.0)[-1].spoken

    def test_a_confirmed_absence_still_resets_on_its_own(self):
        """Nobody is there to inherit the covered topics."""
        sess = session(vision=ScriptedVision.of(answer("toy")), absent_reset_sec=30.0, max_session_sec=0.0)
        sess.step(inputs(now=10.0))
        first = sess.session_id

        sess.step(inputs(now=15.0, present=False, bbox=None))
        sess.step(inputs(now=60.0, present=False, bbox=None))

        assert sess.session_id != first


class TestValidityUsesTheDeliveryClock:
    """Review finding: generation time was invisible to the validity check.

    A fake model returns instantly, so only a clock port exposes it.
    """

    def test_a_slow_generation_makes_the_evidence_stale(self):
        from visual_interaction.session.hooks import Clock

        class ThirtySecondModel(Clock):
            def now(self, round_inputs):
                return round_inputs.now + 30.0

        sess = VisualInteractionSession(
            hooks=SessionHooks(vision=ScriptedVision.of(answer("toy")), clock=ThirtySecondModel()),
            config=SessionConfig(max_evidence_age_sec=8.0, min_gap_sec=0.0),
        )

        sess.step(inputs(now=10.0))
        result = sess.step(inputs(now=11.0))

        assert not result.spoken
        assert result.reason == SUPPRESS_STALE

    def test_a_fast_generation_still_speaks(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), max_evidence_age_sec=8.0)

        assert speak_once(sess)[-1].spoken


class TestResultCarriesTheMessageFields:
    def test_a_spoken_round_reports_its_episode_and_deadline(self):
        sess = session(vision=ScriptedVision.of(answer("toy")), max_evidence_age_sec=8.0)

        result = speak_once(sess)[-1]

        assert result.episode_seq == 1
        assert result.latest_start_at == pytest.approx(result.evidence_stamp + 8.0)

    def test_a_quiet_round_carries_neither(self):
        sess = session(vision=ScriptedVision.of(quiet()))

        result = sess.step(inputs())

        assert result.episode_seq == 0
        assert result.latest_start_at == 0.0
