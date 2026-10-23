"""PCM timing: the two views, and what each one catches that the other misses."""

import pytest

from visual_interaction.phase0.pcm import Chunk, PcmFormat, analyse, analyse_utterance, split_utterances


def board() -> PcmFormat:
    """What the playback node is configured for: 24 kHz mono signed 16-bit."""
    return PcmFormat()


def steady(count: int, pcm: PcmFormat, *, start: float = 100.0) -> list[Chunk]:
    """Chunks arriving exactly on the sample clock."""
    payload = pcm.chunk_frames * pcm.bytes_per_frame
    return [Chunk(arrived=start + index * pcm.chunk_period, payload_bytes=payload) for index in range(count)]


class TestPcmFormat:
    def test_the_chunk_is_fifty_milliseconds_of_audio(self):
        """The playback node publishes rate // 20 frames per message."""
        pcm = board()

        assert pcm.chunk_frames == 1200
        assert pcm.chunk_period == pytest.approx(0.05)

    def test_frames_are_derived_from_the_payload_size(self):
        assert board().frames_in(2400) == 1200

    def test_a_partial_frame_is_refused(self):
        """Signed 16-bit mono means an odd byte count is a corrupt message."""
        with pytest.raises(ValueError, match="whole number"):
            board().frames_in(2401)

    def test_stereo_halves_the_frame_count(self):
        assert PcmFormat(channels=2).frames_in(2400) == 600

    @pytest.mark.parametrize("field", ["sample_rate", "channels", "sample_width"])
    def test_every_field_must_be_positive(self, field):
        with pytest.raises(ValueError, match="positive"):
            PcmFormat(**{field: 0})

    def test_a_very_low_rate_still_publishes_whole_frames(self):
        assert PcmFormat(sample_rate=10).chunk_frames == 1


class TestSplitUtterances:
    def test_a_steady_stream_is_one_utterance(self):
        pcm = board()

        assert len(split_utterances(steady(20, pcm), gap=1.0)) == 1

    def test_a_pause_longer_than_the_gap_starts_a_new_one(self):
        pcm = board()
        first = steady(5, pcm, start=100.0)
        second = steady(5, pcm, start=105.0)

        assert [len(group) for group in split_utterances(first + second, gap=1.0)] == [5, 5]

    def test_chunks_arriving_out_of_order_are_sorted_first(self):
        pcm = board()
        ordered = steady(5, pcm)

        assert split_utterances(list(reversed(ordered)), gap=1.0) == [ordered]

    def test_no_chunks_is_no_utterance(self):
        assert split_utterances([], gap=1.0) == []

    @pytest.mark.parametrize("gap", [0.0, -1.0])
    def test_a_non_positive_gap_is_refused(self, gap):
        with pytest.raises(ValueError, match="positive"):
            split_utterances([], gap=gap)


class TestAnalyseUtterance:
    def test_a_stream_on_the_sample_clock_is_never_late(self):
        pcm = board()
        result = analyse_utterance(steady(30, pcm), pcm)

        assert result["interval_ms"]["p50"] == pytest.approx(50.0)
        assert result["max_lateness_ms"] == pytest.approx(0.0, abs=1e-6)
        assert result["intervals_over_period"]["1x"]["count"] == 0

    def test_float_representation_is_not_counted_as_an_overrun(self):
        """Arrivals computed as start + i*period drift by ~1e-14 s.

        A strict comparison reported six overruns in a perfectly on-time
        30-chunk stream, which would have put spurious violations in every
        report. A real stall is many orders of magnitude larger.
        """
        pcm = board()
        payload = pcm.chunk_frames * pcm.bytes_per_frame
        chunks = [Chunk(arrived=100.0 + index * pcm.chunk_period, payload_bytes=payload) for index in range(30)]
        result = analyse_utterance(chunks, pcm)

        assert result["intervals_over_period"]["1x"]["count"] == 0
        assert result["late_by_over_period"]["1x"]["count"] == 0

    def test_a_real_overrun_just_over_the_period_is_still_counted(self):
        """The slack must not swallow a genuine millisecond-scale stall."""
        pcm = board()
        payload = pcm.chunk_frames * pcm.bytes_per_frame
        chunks = [
            Chunk(arrived=100.0, payload_bytes=payload),
            Chunk(arrived=100.0 + pcm.chunk_period + 0.001, payload_bytes=payload),
        ]

        assert analyse_utterance(chunks, pcm)["intervals_over_period"]["1x"]["count"] == 1

    def test_audio_duration_comes_from_the_frames_not_the_wall_clock(self):
        pcm = board()
        result = analyse_utterance(steady(20, pcm), pcm)

        assert result["audio_ms"] == pytest.approx(1000.0)

    def test_one_long_gap_shows_in_the_intervals(self):
        """An average publish rate would hide it; the distribution must not."""
        pcm = board()
        chunks = steady(10, pcm)
        payload = pcm.chunk_frames * pcm.bytes_per_frame
        chunks.append(Chunk(arrived=chunks[-1].arrived + 0.4, payload_bytes=payload))
        result = analyse_utterance(chunks, pcm)

        assert result["interval_ms"]["max"] == pytest.approx(400.0)
        assert result["intervals_over_period"]["2x"]["count"] == 1

    def test_many_slightly_long_intervals_accumulate_into_lateness(self):
        """The case the interval view alone misses: no single gap looks bad.

        Each interval is 55 ms against a 50 ms period — never twice the
        period — yet by the twentieth chunk the sink is 95 ms behind.
        """
        pcm = board()
        payload = pcm.chunk_frames * pcm.bytes_per_frame
        chunks = [Chunk(arrived=100.0 + index * 0.055, payload_bytes=payload) for index in range(20)]
        result = analyse_utterance(chunks, pcm)

        assert result["intervals_over_period"]["1.5x"]["count"] == 0
        assert result["max_lateness_ms"] == pytest.approx(95.0)
        assert result["late_by_over_period"]["1x"]["count"] > 0

    def test_arriving_early_is_negative_lateness_rather_than_zero(self):
        """A publisher running ahead is a different condition from one on time."""
        pcm = board()
        payload = pcm.chunk_frames * pcm.bytes_per_frame
        chunks = [Chunk(arrived=100.0 + index * 0.04, payload_bytes=payload) for index in range(5)]

        assert analyse_utterance(chunks, pcm)["lateness_ms"]["min"] < 0.0

    def test_a_single_chunk_has_no_intervals_but_one_lateness(self):
        pcm = board()
        result = analyse_utterance(steady(1, pcm), pcm)

        assert result["interval_ms"] == {"n": 0}
        assert result["lateness_ms"]["n"] == 1

    def test_an_empty_utterance_is_refused(self):
        with pytest.raises(ValueError, match="at least one chunk"):
            analyse_utterance([], board())


class TestAnalyse:
    def test_it_reports_per_utterance_and_pooled(self):
        pcm = board()
        chunks = steady(5, pcm, start=100.0) + steady(5, pcm, start=105.0)
        result = analyse(chunks, pcm, gap=1.0)

        assert result["utterances"] == 2
        assert len(result["per_utterance"]) == 2
        assert result["pooled"]["interval_ms"]["n"] == 8

    def test_the_pause_between_utterances_is_not_an_interval(self):
        """Otherwise every silence would read as a five-second stall."""
        pcm = board()
        chunks = steady(5, pcm, start=100.0) + steady(5, pcm, start=105.0)

        assert analyse(chunks, pcm, gap=1.0)["pooled"]["interval_ms"]["max"] == pytest.approx(50.0)

    def test_the_report_records_the_format_and_the_gap_it_used(self):
        pcm = board()
        result = analyse(steady(5, pcm), pcm, gap=0.75)

        assert result["format"]["chunk_frames"] == 1200
        assert result["format"]["sample_rate"] == 24000
        assert result["utterance_gap_sec"] == 0.75

    def test_it_says_where_the_observation_was_taken(self):
        """A subscriber cannot see the speaker, and the report must not imply it did."""
        assert analyse(steady(3, board()), board(), gap=1.0)["observed_at"] == "ros_subscriber"
