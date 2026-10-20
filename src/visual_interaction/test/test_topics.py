"""Coverage and confirmation: what counts as a repeat, and what counts as evidence."""

import pytest

from embodied_common.visual_interaction_contract import (
    COVERAGE_STATES,
    COVERED_ESTIMATED,
    IN_FLIGHT,
    KIND_DISPLAY,
    KIND_OBJECT,
    PARTIAL_COVERED,
    QUARANTINED,
    SUPPRESS_DUPLICATE,
    SUPPRESS_IN_FLIGHT,
    SUPPRESS_UNCONFIRMED,
    SUPPRESSED,
)
from visual_interaction.session.topics import TopicKey, TopicLedger, canonical_region

TOY_HAND = TopicKey.of("toy", "hand")


def confirm(ledger, key=TOY_HAND, kind=KIND_DISPLAY, start=1.0):
    """Admit a topic until it is allowed, returning the final decision."""
    decision = None
    for index in range(ledger.confirmation_count):
        decision = ledger.admit(key, kind, window_stamp=start + index)
    return decision


class TestTopicKey:
    def test_the_key_is_the_closed_pair_and_never_the_reply(self):
        assert TopicKey.of("toy", "hand") == TopicKey.of("toy", "hand")
        assert TopicKey.of("toy", "hand") != TopicKey.of("hat", "hand")

    def test_different_categories_in_one_place_stay_separate(self):
        assert TopicKey.of("toy", "hand") != TopicKey.of("sign", "hand")

    @pytest.mark.parametrize("region", ["hand", "chest", "upper_body"])
    def test_upper_body_regions_collapse_together(self, region):
        """A held-up toy reads as hand in one window and upper_body in the next."""
        assert canonical_region(region) == "upper_body"
        assert TopicKey.of("toy", region) == TopicKey.of("toy", "upper_body")

    def test_head_stays_its_own_region(self):
        assert TopicKey.of("hat", "head") != TopicKey.of("hat", "upper_body")

    def test_unknown_collapses_into_nothing(self):
        """Matching everything would silence unrelated topics."""
        assert canonical_region("unknown") == "unknown"
        assert TopicKey.of("toy", "unknown") != TopicKey.of("toy", "hand")

    def test_an_unmapped_region_passes_through(self):
        assert canonical_region("elbow") == "elbow"

    def test_keys_sort_for_reproducible_diagnostics(self):
        keys = sorted([TopicKey.of("toy", "hand"), TopicKey.of("badge", "chest")])

        assert keys[0].category == "badge"


class TestConfirmation:
    def test_one_window_is_not_enough(self):
        ledger = TopicLedger()

        decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=1.0)

        assert not decision.allowed
        assert decision.reason == SUPPRESS_UNCONFIRMED
        assert decision.confirmations == 1

    def test_two_different_windows_confirm(self):
        ledger = TopicLedger()

        ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=1.0)
        decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=2.0)

        assert decision.allowed
        assert decision.confirmations == 2

    def test_the_same_window_cannot_confirm_itself(self):
        """Re-reading one frame set is not a second look at the subject."""
        ledger = TopicLedger()

        ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=5.0)
        for _ in range(4):
            decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=5.0)

        assert not decision.allowed
        assert decision.confirmations == 1

    def test_an_older_window_does_not_count_either(self):
        ledger = TopicLedger()

        ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=5.0)
        decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=4.0)

        assert not decision.allowed
        assert decision.confirmations == 1

    def test_a_changed_kind_restarts_the_chain(self):
        """An object becoming a display is a different claim, not more evidence."""
        ledger = TopicLedger()

        ledger.admit(TOY_HAND, KIND_OBJECT, window_stamp=1.0)
        decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=2.0)

        assert not decision.allowed
        assert decision.confirmations == 1

    def test_a_broken_chain_starts_over(self):
        ledger = TopicLedger()
        ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=1.0)

        ledger.break_chain(TOY_HAND)
        decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=2.0)

        assert not decision.allowed
        assert decision.confirmations == 1

    def test_breaking_all_chains_keeps_coverage(self):
        """A target change invalidates evidence; it does not unsay a remark."""
        ledger = TopicLedger()
        confirm(ledger)
        ledger.mark(TOY_HAND, COVERED_ESTIMATED)

        ledger.break_all_chains()

        assert ledger.confirmations(TOY_HAND) == 0
        assert ledger.is_blocked(TOY_HAND)

    def test_the_count_is_configurable_but_never_below_two(self):
        assert TopicLedger(confirmation_count=3).confirmation_count == 3

        with pytest.raises(ValueError, match="at least 2"):
            TopicLedger(confirmation_count=1)

    def test_a_third_window_is_required_when_configured(self):
        ledger = TopicLedger(confirmation_count=3)

        ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=1.0)
        assert not ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=2.0).allowed
        assert ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=3.0).allowed


class TestCoverage:
    @pytest.mark.parametrize("state", COVERAGE_STATES)
    def test_every_coverage_state_blocks_a_second_remark(self, state):
        ledger = TopicLedger()
        ledger.mark(TOY_HAND, state)

        decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=9.0)

        assert not decision.allowed
        assert ledger.is_blocked(TOY_HAND)

    def test_in_flight_reports_its_own_reason(self):
        """A topic mid-delivery is different from one already said."""
        ledger = TopicLedger()
        ledger.reserve(TOY_HAND)

        assert ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=9.0).reason == SUPPRESS_IN_FLIGHT

    @pytest.mark.parametrize("state", [COVERED_ESTIMATED, SUPPRESSED, PARTIAL_COVERED, QUARANTINED])
    def test_a_settled_topic_reports_duplicate(self, state):
        ledger = TopicLedger()
        ledger.mark(TOY_HAND, state)

        assert ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=9.0).reason == SUPPRESS_DUPLICATE

    def test_coverage_is_checked_before_evidence_is_counted(self):
        """A long look must not rebuild a chain that can never be used."""
        ledger = TopicLedger()
        ledger.mark(TOY_HAND, COVERED_ESTIMATED)

        for stamp in (1.0, 2.0, 3.0):
            ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=stamp)

        assert ledger.confirmations(TOY_HAND) == 0

    def test_an_unrelated_topic_is_unaffected(self):
        ledger = TopicLedger()
        ledger.mark(TOY_HAND, COVERED_ESTIMATED)
        hat = TopicKey.of("hat", "head")

        ledger.admit(hat, KIND_OBJECT, window_stamp=1.0)
        assert ledger.admit(hat, KIND_OBJECT, window_stamp=2.0).allowed

    def test_an_invalid_coverage_state_is_refused(self):
        ledger = TopicLedger()

        with pytest.raises(ValueError, match="is not one of"):
            ledger.mark(TOY_HAND, "spoken")


class TestCoverageOutranksEpisodeNumbering:
    def test_putting_an_object_down_and_raising_it_again_stays_silent(self):
        """The first version cannot tell a second toy from the first returning.

        Letting a fresh episode speak again turns one ambiguous object into
        repeated remarks, which is the failure a visitor notices.
        """
        ledger = TopicLedger()
        assert confirm(ledger).allowed
        ledger.reserve(TOY_HAND)
        ledger.release(TOY_HAND, COVERED_ESTIMATED)

        # The object leaves the frame, so the evidence chain breaks, and comes
        # back: a new display episode by any numbering.
        ledger.break_chain(TOY_HAND)
        for stamp in (10.0, 11.0, 12.0):
            decision = ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=stamp)
            assert not decision.allowed
            assert decision.reason == SUPPRESS_DUPLICATE

    def test_a_region_move_does_not_restore_eligibility(self):
        """Moving the same object from hand to chest is not a new topic."""
        ledger = TopicLedger()
        ledger.mark(TopicKey.of("toy", "hand"), COVERED_ESTIMATED)

        moved = ledger.admit(TopicKey.of("toy", "chest"), KIND_DISPLAY, window_stamp=20.0)

        assert not moved.allowed
        assert moved.reason == SUPPRESS_DUPLICATE

    def test_a_kind_change_does_not_bypass_coverage(self):
        """It restarts the chain, but a settled topic is not re-opened."""
        ledger = TopicLedger()
        ledger.mark(TOY_HAND, COVERED_ESTIMATED)

        assert ledger.admit(TOY_HAND, KIND_OBJECT, window_stamp=20.0).reason == SUPPRESS_DUPLICATE


class TestReservation:
    def test_a_reservation_blocks_a_concurrent_round(self):
        ledger = TopicLedger()
        confirm(ledger)

        ledger.reserve(TOY_HAND)

        assert ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=3.0).reason == SUPPRESS_IN_FLIGHT

    def test_releasing_with_a_state_settles_the_topic(self):
        ledger = TopicLedger()
        ledger.reserve(TOY_HAND)

        ledger.release(TOY_HAND, COVERED_ESTIMATED)

        assert ledger.coverage_of(TOY_HAND) == COVERED_ESTIMATED
        assert ledger.spoken_count() == 1

    def test_releasing_empty_leaves_the_topic_unspoken(self):
        """Cancellation before the first audio: nothing could have been heard."""
        ledger = TopicLedger()
        confirm(ledger)
        ledger.reserve(TOY_HAND)

        ledger.release(TOY_HAND, "")

        assert ledger.coverage_of(TOY_HAND) == ""
        assert not ledger.is_blocked(TOY_HAND)

    def test_giving_a_topic_back_also_drops_its_evidence(self):
        """The evidence it rested on is what just turned out not to hold."""
        ledger = TopicLedger()
        confirm(ledger)
        ledger.reserve(TOY_HAND)

        ledger.release(TOY_HAND, "")

        assert ledger.confirmations(TOY_HAND) == 0
        assert not ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=30.0).allowed

    def test_a_partial_delivery_still_blocks_a_repeat(self):
        """Some of it may have been heard, so it is not re-said from the top."""
        ledger = TopicLedger()
        ledger.reserve(TOY_HAND)

        ledger.release(TOY_HAND, PARTIAL_COVERED)

        assert ledger.admit(TOY_HAND, KIND_DISPLAY, window_stamp=30.0).reason == SUPPRESS_DUPLICATE

    def test_a_release_state_must_be_a_coverage_state(self):
        """A failed delivery must never silently become a covered one."""
        ledger = TopicLedger()
        ledger.reserve(TOY_HAND)

        with pytest.raises(ValueError, match="is not one of"):
            ledger.release(TOY_HAND, "done")


class TestBookkeeping:
    def test_covered_keys_lists_settled_topics_in_order(self):
        ledger = TopicLedger()
        ledger.mark(TopicKey.of("toy", "hand"), COVERED_ESTIMATED)
        ledger.mark(TopicKey.of("badge", "chest"), SUPPRESSED)

        keys = ledger.covered_keys()

        assert [key.category for key in keys] == ["badge", "toy"]

    def test_covered_keys_includes_in_flight_so_nothing_races_in(self):
        ledger = TopicLedger()
        ledger.reserve(TOY_HAND)

        assert ledger.covered_keys() == (TOY_HAND,)
        assert ledger.coverage_of(TOY_HAND) == IN_FLIGHT

    def test_spoken_count_counts_only_delivered_remarks(self):
        ledger = TopicLedger()
        ledger.mark(TopicKey.of("toy", "hand"), COVERED_ESTIMATED)
        ledger.mark(TopicKey.of("hat", "head"), SUPPRESSED)
        ledger.mark(TopicKey.of("badge", "chest"), QUARANTINED)

        assert ledger.spoken_count() == 1

    def test_a_fresh_ledger_knows_nothing(self):
        ledger = TopicLedger()

        assert ledger.covered_keys() == ()
        assert ledger.spoken_count() == 0
        assert ledger.confirmations(TOY_HAND) == 0
        assert not ledger.is_blocked(TOY_HAND)
