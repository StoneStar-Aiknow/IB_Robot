"""Association, session boundaries and validity: is this still the same visitor."""

import pytest

from embodied_common.visual_interaction_contract import (
    SUPPRESS_AMBIGUOUS,
    SUPPRESS_DUPLICATE,
    SUPPRESS_STALE,
)
from visual_interaction.session.validity import (
    END_ABSENT,
    END_MAX_DURATION,
    ActorTracker,
    Box,
    Candidate,
    SessionWindow,
    candidate_validity,
)

# A person box and the same person a frame later, barely moved.
HERE = (100.0, 100.0, 200.0, 400.0)
NUDGED = (105.0, 100.0, 205.0, 400.0)
# Far enough that no continuity check can join them.
ELSEWHERE = (600.0, 100.0, 700.0, 400.0)


def seen(tracker, box=HERE, stamp=1.0, **kwargs):
    return tracker.update(present=True, known=True, box=box, stamp=stamp, **kwargs)


class TestBox:
    def test_geometry(self):
        box = Box.of(HERE)

        assert box.width == 100.0
        assert box.height == 300.0
        assert box.area == 30000.0
        assert box.centre == (150.0, 250.0)

    def test_identical_boxes_fully_overlap(self):
        assert Box.of(HERE).iou(Box.of(HERE)) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        assert Box.of(HERE).iou(Box.of(ELSEWHERE)) == 0.0

    def test_a_nudged_box_mostly_overlaps(self):
        assert Box.of(HERE).iou(Box.of(NUDGED)) > 0.9

    @pytest.mark.parametrize(
        "values",
        [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0, 5.0), (10.0, 0.0, 5.0, 10.0), (0.0, 10.0, 10.0, 5.0)],
        ids=["too-few", "too-many", "inverted-x", "inverted-y"],
    )
    def test_a_malformed_box_is_refused(self, values):
        with pytest.raises(ValueError):
            Box.of(values)


class TestAssociation:
    def test_the_first_sighting_locks_and_bumps_the_epoch(self):
        tracker = ActorTracker()

        result = seen(tracker)

        assert result.associated
        assert result.epoch_bumped
        assert tracker.locked
        assert tracker.epoch == 1

    def test_a_small_move_continues_without_a_bump(self):
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = seen(tracker, box=NUDGED, stamp=1.2)

        assert result.associated
        assert not result.epoch_bumped
        assert tracker.epoch == 1

    def test_a_jump_relocks_and_voids_everything_before_it(self):
        """A box that cannot be the same person starts a new epoch."""
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = seen(tracker, box=ELSEWHERE, stamp=1.2)

        assert result.associated
        assert result.epoch_bumped
        assert result.reason == "moved"
        assert tracker.epoch == 2

    def test_a_long_gap_relocks_however_alike_the_boxes_are(self):
        """Beyond the gap the boxes are not comparable at all."""
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = seen(tracker, box=HERE, stamp=5.0)

        assert result.epoch_bumped
        assert result.reason == "gap"

    def test_a_backwards_stamp_relocks_rather_than_continuing(self):
        tracker = ActorTracker()
        seen(tracker, stamp=5.0)

        assert seen(tracker, stamp=4.0).epoch_bumped

    def test_a_sudden_size_change_breaks_continuity(self):
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        # Same centre, four times the area: not one person one frame later.
        result = seen(tracker, box=(50.0, 100.0, 250.0, 700.0), stamp=1.2)

        assert result.epoch_bumped
        assert result.reason == "moved"

    def test_losing_the_person_drops_the_lock_and_bumps(self):
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = tracker.update(present=False, known=True, box=None, stamp=2.0)

        assert not result.associated
        assert result.epoch_bumped
        assert not tracker.locked
        assert tracker.epoch == 2

    def test_absence_without_a_lock_changes_nothing(self):
        tracker = ActorTracker()

        result = tracker.update(present=False, known=True, box=None, stamp=1.0)

        assert not result.epoch_bumped
        assert tracker.epoch == 0


class TestUnknownIsNotAbsent:
    def test_an_unknown_frame_neither_extends_nor_breaks_the_lock(self):
        """Stale or missing detections tell us nothing; they are not an exit."""
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = tracker.update(present=False, known=False, box=None, stamp=1.2)

        assert not result.associated
        assert not result.epoch_bumped
        assert result.reason == "unknown"
        assert tracker.locked
        assert tracker.epoch == 1

    def test_the_lock_survives_a_brief_dropout(self):
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)
        tracker.update(present=False, known=False, box=None, stamp=1.2)

        result = seen(tracker, box=NUDGED, stamp=1.4)

        assert result.associated
        assert not result.epoch_bumped

    def test_a_long_dropout_still_ends_in_a_relock_through_the_gap_check(self):
        """Unknown frames do not extend the lock, so the gap eventually wins."""
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)
        for stamp in (1.5, 2.0, 2.5):
            tracker.update(present=False, known=False, box=None, stamp=stamp)

        assert seen(tracker, box=HERE, stamp=3.0).reason == "gap"


class TestAmbiguity:
    def test_two_people_close_together_refuse_rather_than_pick(self):
        """A wrong pick attributes one visitor's object to another."""
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = seen(tracker, stamp=1.2, others=[HERE, (210.0, 100.0, 310.0, 400.0)])

        assert not result.associated
        assert result.ambiguous
        assert result.reason == SUPPRESS_AMBIGUOUS

    def test_ambiguity_does_not_bump_the_epoch(self):
        """Nothing was decided, so nothing collected so far is void."""
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = seen(tracker, stamp=1.2, others=[HERE, (210.0, 100.0, 310.0, 400.0)])

        assert not result.epoch_bumped
        assert tracker.epoch == 1

    def test_a_well_separated_second_person_is_not_ambiguous(self):
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        result = seen(tracker, stamp=1.2, others=[HERE, ELSEWHERE])

        assert result.associated
        assert not result.ambiguous

    def test_the_target_itself_is_not_a_rival(self):
        tracker = ActorTracker()
        seen(tracker, stamp=1.0)

        assert seen(tracker, stamp=1.2, others=[HERE]).associated


class TestSessionWindow:
    def test_a_present_visitor_keeps_the_session_open(self):
        window = SessionWindow(session_id="s1", started_at=0.0)

        window.observe(present=True, known=True, now=10.0)

        assert window.expired_reason(10.0) == ""
        assert window.absent_for(10.0) == 0.0

    def test_a_confirmed_absence_ends_the_session(self):
        window = SessionWindow(session_id="s1", started_at=0.0, absent_reset_sec=30.0)

        window.observe(present=False, known=True, now=5.0)

        assert window.absent_for(20.0) == 15.0
        assert window.expired_reason(20.0) == ""
        assert window.expired_reason(35.0) == END_ABSENT

    def test_a_dropout_is_unknown_and_does_not_start_the_timer(self):
        """Resetting coverage under a visitor who never moved is worse than waiting."""
        window = SessionWindow(session_id="s1", started_at=0.0, absent_reset_sec=30.0)

        for stamp in (5.0, 20.0, 50.0):
            window.observe(present=False, known=False, now=stamp)

        assert window.absent_for(100.0) == 0.0
        assert window.expired_reason(100.0) != END_ABSENT

    def test_a_returning_visitor_clears_the_absence_timer(self):
        window = SessionWindow(session_id="s1", started_at=0.0, absent_reset_sec=30.0)
        window.observe(present=False, known=True, now=5.0)

        window.observe(present=True, known=True, now=10.0)

        assert window.absent_for(40.0) == 0.0
        assert window.expired_reason(40.0) == ""

    def test_the_maximum_duration_ends_a_busy_session(self):
        """Without it a queue never resets and every visitor inherits the first."""
        window = SessionWindow(session_id="s1", started_at=0.0, max_session_sec=120.0)
        window.observe(present=True, known=True, now=119.0)

        assert window.expired_reason(119.0) == ""
        assert window.expired_reason(120.0) == END_MAX_DURATION

    def test_absence_outranks_the_duration_when_both_apply(self):
        window = SessionWindow(session_id="s1", started_at=0.0, absent_reset_sec=10.0, max_session_sec=20.0)
        window.observe(present=False, known=True, now=5.0)

        assert window.expired_reason(30.0) == END_ABSENT

    def test_a_zero_absence_timer_disables_that_ending(self):
        window = SessionWindow(session_id="s1", started_at=0.0, absent_reset_sec=0.0)
        window.observe(present=False, known=True, now=1.0)

        assert window.expired_reason(10_000.0) == END_MAX_DURATION

    def test_a_zero_duration_limit_disables_that_ending(self):
        window = SessionWindow(session_id="s1", started_at=0.0, max_session_sec=0.0)
        window.observe(present=True, known=True, now=1.0)

        assert window.expired_reason(10_000.0) == ""


class TestCandidateValidity:
    def candidate(self, **overrides):
        values = {
            "event_id": "e1",
            "session_id": "s1",
            "actor_epoch": 3,
            "evidence_stamp": 100.0,
            "text": "你举起的玩偶真有趣。",
        }
        values.update(overrides)
        return Candidate(**values)

    def check(self, candidate=None, **overrides):
        values = {
            "session_id": "s1",
            "actor_epoch": 3,
            "target_present": True,
            "topic_blocked": False,
            "now": 101.0,
            "max_age_sec": 8.0,
        }
        values.update(overrides)
        return candidate_validity(candidate or self.candidate(), **values)

    def test_a_fresh_candidate_for_the_same_visitor_is_valid(self):
        assert self.check() == ""

    def test_a_candidate_from_another_session_is_refused(self):
        assert self.check(session_id="s2") == SUPPRESS_AMBIGUOUS

    def test_a_candidate_from_another_epoch_is_refused(self):
        """Its evidence belongs to somebody else by then."""
        assert self.check(actor_epoch=4) == SUPPRESS_AMBIGUOUS

    def test_a_candidate_is_dropped_once_the_target_left(self):
        assert self.check(target_present=False) == SUPPRESS_STALE

    def test_a_candidate_is_dropped_when_its_topic_got_settled_meanwhile(self):
        assert self.check(topic_blocked=True) == SUPPRESS_DUPLICATE

    def test_an_old_picture_is_refused_even_with_the_visitor_still_there(self):
        """ "Somebody is still present" never makes an old event valid forever."""
        assert self.check(now=120.0) == SUPPRESS_STALE

    def test_the_age_is_measured_against_the_evidence_not_the_decision(self):
        assert self.check(now=107.9) == ""
        assert self.check(now=108.1) == SUPPRESS_STALE

    def test_identity_is_checked_before_age(self):
        """A wrong-visitor candidate is reported as such, not merely as stale."""
        assert self.check(session_id="s2", now=10_000.0) == SUPPRESS_AMBIGUOUS
