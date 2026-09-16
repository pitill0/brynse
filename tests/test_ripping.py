from pathlib import Path

import pytest

from fluxtuner_ripper.ripping import (
    ContentKind,
    EncodedAudioRingBuffer,
    MetadataEvent,
    SplitDecision,
    SplitKind,
)


def test_content_kind_keeps_future_semantic_categories_explicit() -> None:
    assert ContentKind.MUSIC == "music"
    assert ContentKind.ADVERTISEMENT == "advertisement"
    assert ContentKind.JINGLE == "jingle"
    assert ContentKind.STATION_ID == "station_id"
    assert ContentKind.TALK == "talk"
    assert ContentKind.UNKNOWN == "unknown"


def test_metadata_event_rejects_negative_audio_offset() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MetadataEvent(title="Artist - Track", audio_offset=-1)


def test_no_boundary_has_no_offsets() -> None:
    decision = SplitDecision(kind=SplitKind.NO_BOUNDARY)

    assert decision.incoming_start is None
    assert decision.outgoing_end is None


def test_no_boundary_rejects_offsets() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        SplitDecision(
            kind=SplitKind.NO_BOUNDARY,
            incoming_start=10,
            outgoing_end=10,
        )


def test_hard_cut_requires_identical_offsets() -> None:
    decision = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=123,
        outgoing_end=123,
    )

    assert decision.incoming_start == decision.outgoing_end

    with pytest.raises(ValueError, match="identical"):
        SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=123,
            outgoing_end=124,
        )


def test_crossfade_requires_positive_overlap() -> None:
    decision = SplitDecision(
        kind=SplitKind.CROSSFADE,
        incoming_start=100,
        outgoing_end=140,
    )

    assert decision.outgoing_end - decision.incoming_start == 40

    with pytest.raises(ValueError, match="incoming_start < outgoing_end"):
        SplitDecision(
            kind=SplitKind.CROSSFADE,
            incoming_start=140,
            outgoing_end=140,
        )


def test_exclusion_requires_positive_gap() -> None:
    decision = SplitDecision(
        kind=SplitKind.EXCLUSION,
        incoming_start=140,
        outgoing_end=100,
    )

    assert decision.incoming_start - decision.outgoing_end == 40

    with pytest.raises(ValueError, match="outgoing_end < incoming_start"):
        SplitDecision(
            kind=SplitKind.EXCLUSION,
            incoming_start=100,
            outgoing_end=100,
        )


def test_ring_buffer_tracks_absolute_offsets() -> None:
    buffer = EncodedAudioRingBuffer(max_bytes=10)

    assert buffer.append(b"abc") == (0, 3)
    assert buffer.append(b"defg") == (3, 7)
    assert buffer.start_offset == 0
    assert buffer.end_offset == 7
    assert buffer.retained_bytes == 7
    assert buffer.read(2, 6) == b"cdef"


def test_ring_buffer_discards_oldest_bytes_without_rebasing_offsets() -> None:
    buffer = EncodedAudioRingBuffer(max_bytes=5)

    buffer.append(b"abc")
    assert buffer.append(b"defg") == (3, 7)

    assert buffer.start_offset == 2
    assert buffer.end_offset == 7
    assert buffer.retained_bytes == 5
    assert buffer.read(2, 7) == b"cdefg"


def test_ring_buffer_handles_chunk_larger_than_capacity() -> None:
    buffer = EncodedAudioRingBuffer(max_bytes=4)

    assert buffer.append(b"abcdefgh") == (0, 8)

    assert buffer.start_offset == 4
    assert buffer.end_offset == 8
    assert buffer.read(4, 8) == b"efgh"


def test_ring_buffer_rejects_ranges_that_are_no_longer_retained() -> None:
    buffer = EncodedAudioRingBuffer(max_bytes=4)
    buffer.append(b"abcdef")

    assert not buffer.contains(0, 2)
    assert buffer.contains(2, 6)

    with pytest.raises(ValueError, match="outside retained range"):
        buffer.read(0, 2)


def test_ring_buffer_empty_append_does_not_move_offsets() -> None:
    buffer = EncodedAudioRingBuffer(max_bytes=4)

    assert buffer.append(b"") == (0, 0)
    assert buffer.start_offset == 0
    assert buffer.end_offset == 0


def _mp3_frame(*, bitrate_index: int, sample_rate_index: int = 0, payload_byte: int = 0) -> bytes:
    header = 0
    header |= 0x7FF << 21
    header |= 0b11 << 19
    header |= 0b01 << 17
    header |= 0b1 << 16
    header |= bitrate_index << 12
    header |= sample_rate_index << 10

    bitrate_kbps = {
        1: 32,
        5: 64,
        9: 128,
        11: 192,
        14: 320,
    }[bitrate_index]
    sample_rate = (44100, 48000, 32000)[sample_rate_index]
    frame_length = (144000 * bitrate_kbps) // sample_rate
    return header.to_bytes(4, "big") + bytes([payload_byte]) * (frame_length - 4)


def _adts_frame(*, frame_length: int, sample_rate_index: int = 4, payload_byte: int = 0) -> bytes:
    if frame_length < 7:
        raise ValueError("frame_length must be at least 7")

    profile = 1
    channel_config = 2

    b0 = 0xFF
    b1 = 0xF1
    b2 = (profile << 6) | (sample_rate_index << 2) | (channel_config >> 2)
    b3 = ((channel_config & 0x03) << 6) | ((frame_length >> 11) & 0x03)
    b4 = (frame_length >> 3) & 0xFF
    b5 = ((frame_length & 0x07) << 5) | 0x1F
    b6 = 0xFC

    header = bytes((b0, b1, b2, b3, b4, b5, b6))
    return header + bytes([payload_byte]) * (frame_length - 7)


def test_parse_mp3_frames_uses_each_frame_bitrate_for_vbr_timeline() -> None:
    from fluxtuner_ripper.ripping import parse_mp3_frames

    first = _mp3_frame(bitrate_index=9, payload_byte=1)
    second = _mp3_frame(bitrate_index=14, payload_byte=2)
    third = _mp3_frame(bitrate_index=5, payload_byte=3)

    frames = parse_mp3_frames(first + second + third)

    assert [frame.length for frame in frames] == [len(first), len(second), len(third)]
    assert [frame.offset for frame in frames] == [0, len(first), len(first) + len(second)]
    assert frames[0].time_seconds == 0.0
    assert frames[1].time_seconds == pytest.approx(1152 / 44100)
    assert frames[2].time_seconds == pytest.approx(2304 / 44100)


def test_parse_mp3_frames_resynchronizes_after_leading_junk() -> None:
    from fluxtuner_ripper.ripping import parse_mp3_frames

    frame = _mp3_frame(bitrate_index=9)
    frames = parse_mp3_frames(b"junk" + frame)

    assert len(frames) == 1
    assert frames[0].offset == 4


def test_parse_adts_frames_handles_variable_frame_lengths() -> None:
    from fluxtuner_ripper.ripping import parse_adts_frames

    first = _adts_frame(frame_length=900, payload_byte=1)
    second = _adts_frame(frame_length=931, payload_byte=2)
    third = _adts_frame(frame_length=875, payload_byte=3)

    frames = parse_adts_frames(first + second + third)

    assert [frame.length for frame in frames] == [900, 931, 875]
    assert [frame.offset for frame in frames] == [0, 900, 1831]
    assert frames[0].sample_rate == 44100
    assert frames[1].time_seconds == pytest.approx(1024 / 44100)
    assert frames[2].time_seconds == pytest.approx(2048 / 44100)


def test_parse_adts_frames_resynchronizes_after_leading_junk() -> None:
    from fluxtuner_ripper.ripping import parse_adts_frames

    frame = _adts_frame(frame_length=900)
    frames = parse_adts_frames(b"abc" + frame)

    assert len(frames) == 1
    assert frames[0].offset == 3


def test_frame_at_or_before_offset_uses_frame_start_offsets() -> None:
    from fluxtuner_ripper.ripping import frame_at_or_before_offset, parse_adts_frames

    first = _adts_frame(frame_length=900)
    second = _adts_frame(frame_length=931)
    frames = parse_adts_frames(first + second)

    assert frame_at_or_before_offset(frames, 0) is frames[0]
    assert frame_at_or_before_offset(frames, 899) is frames[0]
    assert frame_at_or_before_offset(frames, 900) is frames[1]


def test_frame_nearest_time_returns_closest_frame_start() -> None:
    from fluxtuner_ripper.ripping import frame_nearest_time, parse_adts_frames

    data = b"".join(_adts_frame(frame_length=900) for _ in range(4))
    frames = parse_adts_frames(data)

    target = frames[2].time_seconds + 0.001
    assert frame_nearest_time(frames, target) is frames[2]


def _icy_block(metadata: str) -> bytes:
    raw = metadata.encode("utf-8")
    blocks = (len(raw) + 15) // 16
    if blocks > 255:
        raise ValueError("metadata too large for ICY")
    padded = raw + b"\x00" * (blocks * 16 - len(raw))
    return bytes([blocks]) + padded


def test_icy_parser_strips_metadata_and_reports_clean_audio_offset() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)
    raw = b"abcd" + _icy_block("StreamTitle='Artist - Track';") + b"efgh"

    result = parser.feed(raw)

    assert result.audio == b"abcdefgh"
    assert len(result.events) == 1
    assert result.events[0].title == "Artist - Track"
    assert result.events[0].audio_offset == 4
    assert parser.audio_offset == 8


def test_icy_parser_handles_zero_length_metadata_block() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=3)

    result = parser.feed(b"abc\x00def")

    assert result.audio == b"abcdef"
    assert result.events == ()
    assert parser.audio_offset == 6


def test_icy_parser_handles_metadata_split_across_chunks() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)
    block = _icy_block("StreamTitle='Split Metadata';")
    raw = b"abcd" + block + b"efgh"

    parts = [
        raw[:2],
        raw[2:5],
        raw[5:9],
        raw[9:17],
        raw[17:],
    ]

    audio = bytearray()
    events = []

    for part in parts:
        result = parser.feed(part)
        audio.extend(result.audio)
        events.extend(result.events)

    assert bytes(audio) == b"abcdefgh"
    assert len(events) == 1
    assert events[0].title == "Split Metadata"
    assert events[0].audio_offset == 4


def test_icy_parser_handles_length_byte_as_its_own_chunk() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)
    block = _icy_block("StreamTitle='Chunky';")

    first = parser.feed(b"abcd")
    second = parser.feed(block[:1])
    third = parser.feed(block[1:] + b"efgh")

    assert first.audio == b"abcd"
    assert first.events == ()
    assert second.audio == b""
    assert second.events == ()
    assert third.audio == b"efgh"
    assert len(third.events) == 1
    assert third.events[0].audio_offset == 4


def test_icy_parser_emits_multiple_events_with_absolute_audio_offsets() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)
    raw = (
        b"aaaa"
        + _icy_block("StreamTitle='One';")
        + b"bbbb"
        + _icy_block("StreamTitle='Two';")
        + b"cccc"
    )

    result = parser.feed(raw)

    assert result.audio == b"aaaabbbbcccc"
    assert [(event.title, event.audio_offset) for event in result.events] == [
        ("One", 4),
        ("Two", 8),
    ]
    assert parser.audio_offset == 12


def test_icy_parser_ignores_metadata_without_stream_title() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)
    raw = b"abcd" + _icy_block("StreamUrl='https://example.invalid';") + b"efgh"

    result = parser.feed(raw)

    assert result.audio == b"abcdefgh"
    assert result.events == ()


def test_icy_parser_ignores_empty_stream_title() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)
    raw = b"abcd" + _icy_block("StreamTitle='';") + b"efgh"

    result = parser.feed(raw)

    assert result.audio == b"abcdefgh"
    assert result.events == ()


def test_icy_parser_empty_feed_preserves_state() -> None:
    from fluxtuner_ripper.ripping import IcyStreamParser

    parser = IcyStreamParser(metaint=4)

    assert parser.feed(b"ab").audio == b"ab"
    assert parser.feed(b"").audio == b""
    assert parser.feed(b"cd\x00efgh").audio == b"cdefgh"
    assert parser.audio_offset == 8


def test_incremental_frame_timeline_handles_mp3_frame_split_across_feeds() -> None:
    from fluxtuner_ripper.ripping import IncrementalFrameTimeline

    first = _mp3_frame(bitrate_index=9)
    second = _mp3_frame(bitrate_index=14)
    timeline = IncrementalFrameTimeline("mp3")

    assert timeline.feed(first[:100]) == ()
    created = timeline.feed(first[100:] + second[:50])

    assert len(created) == 1
    assert created[0].offset == 0
    assert created[0].time_seconds == 0.0

    created = timeline.feed(second[50:])

    assert len(created) == 1
    assert created[0].offset == len(first)
    assert created[0].time_seconds == pytest.approx(1152 / 44100)


def test_incremental_frame_timeline_handles_adts_frame_split_across_feeds() -> None:
    from fluxtuner_ripper.ripping import IncrementalFrameTimeline

    first = _adts_frame(frame_length=900)
    second = _adts_frame(frame_length=931)
    timeline = IncrementalFrameTimeline("aac")

    assert timeline.feed(first[:6]) == ()
    created = timeline.feed(first[6:] + second[:200])

    assert len(created) == 1
    assert created[0].offset == 0

    created = timeline.feed(second[200:])

    assert len(created) == 1
    assert created[0].offset == 900
    assert created[0].time_seconds == pytest.approx(1024 / 44100)


def test_incremental_frame_timeline_resynchronizes_and_keeps_absolute_offsets() -> None:
    from fluxtuner_ripper.ripping import IncrementalFrameTimeline

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")

    created = timeline.feed(b"junk" + frame)

    assert len(created) == 1
    assert created[0].offset == 4


def test_ripping_stream_ingestor_resolves_icy_metadata_to_exact_aac_time() -> None:
    from fluxtuner_ripper.ripping import RippingStreamIngestor

    first = _adts_frame(frame_length=900)
    second = _adts_frame(frame_length=900)
    third = _adts_frame(frame_length=900)

    parser = RippingStreamIngestor(
        metaint=len(first) + len(second),
        codec="aac",
        ring_max_bytes=10000,
    )
    raw = first + second + _icy_block("StreamTitle='Artist - Track';") + third

    result = parser.feed(raw)

    assert result.audio == first + second + third
    assert len(result.metadata_events) == 1
    assert result.metadata_events[0].audio_offset == 1800
    assert len(result.timed_metadata_events) == 1
    assert result.timed_metadata_events[0].title == "Artist - Track"
    assert result.timed_metadata_events[0].audio_time_seconds == pytest.approx(2048 / 44100)


def test_ripping_stream_ingestor_survives_arbitrary_raw_chunk_boundaries() -> None:
    from fluxtuner_ripper.ripping import RippingStreamIngestor

    first = _mp3_frame(bitrate_index=9)
    second = _mp3_frame(bitrate_index=14)
    metaint = len(first)

    ingestor = RippingStreamIngestor(
        metaint=metaint,
        codec="mp3",
        ring_max_bytes=10000,
    )

    clean = first + second
    raw = (
        clean[:metaint]
        + _icy_block("StreamTitle='Next';")
        + clean[metaint : 2 * metaint]
        + b"\x00"
        + clean[2 * metaint : 3 * metaint]
        + b"\x00"
        + clean[3 * metaint :]
    )

    outputs = []
    for start in range(0, len(raw), 37):
        outputs.append(ingestor.feed(raw[start : start + 37]))

    clean_audio = b"".join(output.audio for output in outputs)
    timed = [event for output in outputs for event in output.timed_metadata_events]

    assert clean_audio == clean
    assert len(timed) == 1
    assert timed[0].audio_offset == len(first)
    assert timed[0].audio_time_seconds == pytest.approx(1152 / 44100)


def test_ripping_stream_ingestor_delays_boundary_metadata_until_next_frame_arrives() -> None:
    from fluxtuner_ripper.ripping import RippingStreamIngestor

    first = _adts_frame(frame_length=900)
    second = _adts_frame(frame_length=900)
    block = _icy_block("StreamTitle='Next';")

    ingestor = RippingStreamIngestor(
        metaint=900,
        codec="aac",
        ring_max_bytes=10000,
    )

    first_result = ingestor.feed(first + block)
    assert len(first_result.metadata_events) == 1
    assert first_result.timed_metadata_events == ()

    second_result = ingestor.feed(second)
    assert len(second_result.timed_metadata_events) == 1
    assert second_result.timed_metadata_events[0].audio_offset == 900
    assert second_result.timed_metadata_events[0].audio_time_seconds == pytest.approx(1024 / 44100)


def test_ripping_stream_ingestor_prunes_frame_records_with_ring_buffer() -> None:
    from fluxtuner_ripper.ripping import RippingStreamIngestor

    frame = _adts_frame(frame_length=900)
    ingestor = RippingStreamIngestor(
        metaint=900,
        codec="aac",
        ring_max_bytes=1000,
    )

    raw = frame + b"\x00" + frame + b"\x00" + frame
    ingestor.feed(raw)

    assert ingestor.ring_buffer.start_offset == 1700
    assert all(
        item.offset + item.length > ingestor.ring_buffer.start_offset
        for item in ingestor.timeline.frames
    )


def _timed_metadata(
    title: str,
    *,
    offset: int,
    time_seconds: float,
):
    from fluxtuner_ripper.ripping import TimedMetadataEvent

    return TimedMetadataEvent(
        title=title,
        audio_offset=offset,
        audio_time_seconds=time_seconds,
    )


def test_metadata_semantic_tracker_marks_short_lifetime_no_boundary() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker, SplitKind

    tracker = MetadataSemanticTracker(transient_threshold_seconds=8.0)

    assert tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=0.0)) == (
        (),
        (),
    )

    decisions, candidates = tracker.feed(
        _timed_metadata("Commercial-free", offset=100, time_seconds=5.0)
    )

    assert len(decisions) == 1
    assert decisions[0].title == "Song A"
    assert decisions[0].kind is SplitKind.NO_BOUNDARY
    assert decisions[0].lifetime_seconds == pytest.approx(5.0)
    assert candidates == ()


def test_metadata_semantic_tracker_confirms_durable_title_on_repeated_metadata() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker

    tracker = MetadataSemanticTracker(transient_threshold_seconds=8.0)

    tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=0.0))
    decisions, candidates = tracker.feed(_timed_metadata("Song A", offset=800, time_seconds=8.1))

    assert decisions == ()
    assert len(candidates) == 1
    assert candidates[0].title == "Song A"
    assert candidates[0].start_offset == 0
    assert candidates[0].confirmed_at_offset == 800


def test_metadata_semantic_tracker_confirms_durable_title_when_replaced() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker, SplitKind

    tracker = MetadataSemanticTracker(transient_threshold_seconds=8.0)

    tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=0.0))
    decisions, candidates = tracker.feed(_timed_metadata("Song B", offset=1000, time_seconds=10.0))

    assert len(decisions) == 1
    assert decisions[0].title == "Song A"
    assert decisions[0].kind is SplitKind.HARD_CUT
    assert decisions[0].lifetime_seconds == pytest.approx(10.0)

    assert len(candidates) == 1
    assert candidates[0].title == "Song A"
    assert candidates[0].confirmed_at_time_seconds == pytest.approx(10.0)


def test_metadata_semantic_tracker_does_not_duplicate_confirmed_candidate() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker

    tracker = MetadataSemanticTracker(transient_threshold_seconds=8.0)

    tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=0.0))
    _, first = tracker.feed(_timed_metadata("Song A", offset=800, time_seconds=8.0))
    _, second = tracker.feed(_timed_metadata("Song A", offset=900, time_seconds=9.0))

    assert len(first) == 1
    assert second == ()


def test_metadata_semantic_tracker_confirm_current_without_repeat_packet() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker

    tracker = MetadataSemanticTracker(transient_threshold_seconds=8.0)

    tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=0.0))

    assert tracker.confirm_current(audio_offset=700, audio_time_seconds=7.9) == ()

    candidates = tracker.confirm_current(audio_offset=900, audio_time_seconds=8.1)

    assert len(candidates) == 1
    assert candidates[0].title == "Song A"
    assert candidates[0].confirmed_at_offset == 900


def test_metadata_semantic_tracker_transient_bridge_sequence() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker, SplitKind

    tracker = MetadataSemanticTracker(transient_threshold_seconds=8.0)

    tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=0.0))
    tracker.confirm_current(audio_offset=1000, audio_time_seconds=10.0)

    decisions, candidates = tracker.feed(
        _timed_metadata("Commercial-free", offset=8000, time_seconds=80.0)
    )

    assert len(decisions) == 1
    assert decisions[0].title == "Song A"
    assert decisions[0].kind is SplitKind.HARD_CUT
    assert candidates == ()

    decisions, candidates = tracker.feed(_timed_metadata("Song B", offset=8250, time_seconds=82.5))

    assert len(decisions) == 1
    assert decisions[0].title == "Commercial-free"
    assert decisions[0].kind is SplitKind.NO_BOUNDARY
    assert decisions[0].lifetime_seconds == pytest.approx(2.5)
    assert candidates == ()


def test_metadata_semantic_tracker_rejects_non_monotonic_time() -> None:
    from fluxtuner_ripper.ripping import MetadataSemanticTracker

    tracker = MetadataSemanticTracker()

    tracker.feed(_timed_metadata("Song A", offset=0, time_seconds=10.0))

    with pytest.raises(ValueError):
        tracker.feed(_timed_metadata("Song B", offset=100, time_seconds=9.0))


def test_acoustic_window_extractor_returns_frame_aligned_aac_window() -> None:
    from fluxtuner_ripper.ripping import (
        AcousticWindowExtractor,
        EncodedAudioRingBuffer,
        IncrementalFrameTimeline,
    )

    frames = [_adts_frame(frame_length=900) for _ in range(10)]
    clean = b"".join(frames)

    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(clean)

    ring = EncodedAudioRingBuffer(max_bytes=len(clean))
    ring.append(clean)

    extractor = AcousticWindowExtractor(search_radius_seconds=0.05)
    candidate_time = timeline.frames[5].time_seconds

    window = extractor.extract(
        candidate_time_seconds=candidate_time,
        timeline=timeline,
        ring_buffer=ring,
    )

    assert window is not None
    assert window.start_offset % 900 == 0
    assert window.end_offset % 900 == 0
    assert window.data == clean[window.start_offset : window.end_offset]


def test_acoustic_window_extractor_clamps_to_missing_preroll() -> None:
    from fluxtuner_ripper.ripping import (
        AcousticWindowExtractor,
        EncodedAudioRingBuffer,
        IncrementalFrameTimeline,
    )

    frame = _adts_frame(frame_length=900)
    clean = frame * 12

    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(clean)

    ring = EncodedAudioRingBuffer(max_bytes=3600)
    ring.append(clean)

    extractor = AcousticWindowExtractor(search_radius_seconds=0.20)
    candidate_time = timeline.frames[-2].time_seconds

    window = extractor.extract(
        candidate_time_seconds=candidate_time,
        timeline=timeline,
        ring_buffer=ring,
    )

    assert window is not None
    assert window.start_offset >= ring.start_offset
    assert ring.contains(window.start_offset, window.end_offset)


def test_acoustic_window_extractor_clamps_to_missing_postroll() -> None:
    from fluxtuner_ripper.ripping import (
        AcousticWindowExtractor,
        EncodedAudioRingBuffer,
        IncrementalFrameTimeline,
    )

    frame = _mp3_frame(bitrate_index=9)
    clean = frame * 6

    timeline = IncrementalFrameTimeline("mp3")
    timeline.feed(clean)

    ring = EncodedAudioRingBuffer(max_bytes=len(clean))
    ring.append(clean)

    extractor = AcousticWindowExtractor(search_radius_seconds=1.0)
    candidate_time = timeline.frames[-1].time_seconds

    window = extractor.extract(
        candidate_time_seconds=candidate_time,
        timeline=timeline,
        ring_buffer=ring,
    )

    assert window is not None
    assert window.end_offset == ring.end_offset


def test_acoustic_window_extractor_returns_none_without_retained_frames() -> None:
    from fluxtuner_ripper.ripping import (
        AcousticWindowExtractor,
        EncodedAudioRingBuffer,
        IncrementalFrameTimeline,
    )

    frame = _adts_frame(frame_length=900)

    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame)

    ring = EncodedAudioRingBuffer(max_bytes=100)
    ring.append(b"x" * 1000)

    assert ring.start_offset == 900

    extractor = AcousticWindowExtractor()

    window = extractor.extract(
        candidate_time_seconds=0.0,
        timeline=timeline,
        ring_buffer=ring,
    )

    assert window is None


def test_acoustic_window_extractor_handles_small_retained_tail() -> None:
    from fluxtuner_ripper.ripping import (
        AcousticWindowExtractor,
        EncodedAudioRingBuffer,
        IncrementalFrameTimeline,
    )

    frame = _adts_frame(frame_length=900)
    clean = frame * 5

    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(clean)

    ring = EncodedAudioRingBuffer(max_bytes=1000)
    ring.append(clean)

    extractor = AcousticWindowExtractor(search_radius_seconds=0.5)

    window = extractor.extract(
        candidate_time_seconds=timeline.frames[-1].time_seconds,
        timeline=timeline,
        ring_buffer=ring,
    )

    assert window is not None
    assert ring.contains(window.start_offset, window.end_offset)
    assert len(window.data) == window.end_offset - window.start_offset


def _make_acoustic_window(data: bytes):
    from fluxtuner_ripper.ripping import AcousticWindow

    return AcousticWindow(
        start_offset=0,
        end_offset=len(data),
        start_time_seconds=0.0,
        end_time_seconds=1.0,
        data=data,
    )


def test_decoded_pcm_reports_sample_count() -> None:
    from fluxtuner_ripper.ripping import DecodedPcm

    pcm = DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=b"\x00\x00" * 123,
    )

    assert pcm.sample_count == 123


def test_ffmpeg_acoustic_decoder_rejects_missing_binary() -> None:
    from fluxtuner_ripper.ripping import AcousticDecodeError, FfmpegAcousticDecoder

    decoder = FfmpegAcousticDecoder(ffmpeg_binary="/definitely/missing/fluxtuner-ffmpeg")

    with pytest.raises(AcousticDecodeError, match="FFmpeg binary not found"):
        decoder.decode(_make_acoustic_window(b"not-a-stream"))


def test_ffmpeg_acoustic_decoder_rejects_invalid_audio_when_ffmpeg_available() -> None:
    import shutil

    from fluxtuner_ripper.ripping import AcousticDecodeError, FfmpegAcousticDecoder

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

    decoder = FfmpegAcousticDecoder()

    with pytest.raises(AcousticDecodeError, match="FFmpeg decode failed"):
        decoder.decode(_make_acoustic_window(b"not-a-valid-audio-stream"))


def test_ffmpeg_acoustic_decoder_decodes_mp3_window_to_mono_8khz() -> None:
    import shutil
    import subprocess

    from fluxtuner_ripper.ripping import FfmpegAcousticDecoder

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "mp3",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )

    decoder = FfmpegAcousticDecoder(ffmpeg_binary=ffmpeg)
    pcm = decoder.decode(_make_acoustic_window(generated.stdout))

    assert pcm.sample_rate == 8000
    assert pcm.channels == 1
    assert pcm.sample_width_bytes == 2
    assert pcm.sample_count > 1000


def test_ffmpeg_acoustic_decoder_decodes_adts_aac_window_to_mono_8khz() -> None:
    import shutil
    import subprocess

    from fluxtuner_ripper.ripping import FfmpegAcousticDecoder

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.25",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "aac",
            "-f",
            "adts",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )

    decoder = FfmpegAcousticDecoder(ffmpeg_binary=ffmpeg)
    pcm = decoder.decode(_make_acoustic_window(generated.stdout))

    assert pcm.sample_rate == 8000
    assert pcm.channels == 1
    assert pcm.sample_width_bytes == 2
    assert pcm.sample_count > 1000


def _pcm_from_samples(samples: list[int], *, sample_rate: int = 8):
    from array import array

    from fluxtuner_ripper.ripping import DecodedPcm

    data = array("h", samples).tobytes()
    return DecodedPcm(
        sample_rate=sample_rate,
        channels=1,
        sample_width_bytes=2,
        data=data,
    )


def test_rms_acoustic_analyzer_reports_zero_for_silence() -> None:
    from fluxtuner_ripper.ripping import RmsAcousticAnalyzer

    analyzer = RmsAcousticAnalyzer(window_seconds=0.5)
    profile = analyzer.analyze(_pcm_from_samples([0] * 8, sample_rate=8))

    assert len(profile.levels) == 2
    assert all(level.rms == 0.0 for level in profile.levels)


def test_rms_acoustic_analyzer_reports_constant_signal_level() -> None:
    from fluxtuner_ripper.ripping import RmsAcousticAnalyzer

    analyzer = RmsAcousticAnalyzer(window_seconds=0.5)
    profile = analyzer.analyze(_pcm_from_samples([1000] * 8, sample_rate=8))

    assert len(profile.levels) == 2
    assert all(level.rms == pytest.approx(1000.0) for level in profile.levels)


def test_rms_acoustic_analyzer_finds_known_energy_valley() -> None:
    from fluxtuner_ripper.ripping import RmsAcousticAnalyzer

    analyzer = RmsAcousticAnalyzer(window_seconds=0.5)
    pcm = _pcm_from_samples(
        [2000] * 4 + [100] * 4 + [1500] * 4,
        sample_rate=8,
    )

    profile = analyzer.analyze(pcm)
    minimum = profile.minimum_level()

    assert minimum is not None
    assert minimum.start_time_seconds == pytest.approx(0.5)
    assert minimum.end_time_seconds == pytest.approx(1.0)
    assert minimum.rms == pytest.approx(100.0)


def test_rms_acoustic_analyzer_handles_partial_final_window() -> None:
    from fluxtuner_ripper.ripping import RmsAcousticAnalyzer

    analyzer = RmsAcousticAnalyzer(window_seconds=0.5)
    profile = analyzer.analyze(_pcm_from_samples([1000] * 10, sample_rate=8))

    assert len(profile.levels) == 3
    assert profile.levels[-1].start_time_seconds == pytest.approx(1.0)
    assert profile.levels[-1].end_time_seconds == pytest.approx(1.25)
    assert profile.levels[-1].rms == pytest.approx(1000.0)


def test_acoustic_profile_minimum_level_returns_none_when_empty() -> None:
    from fluxtuner_ripper.ripping import AcousticProfile

    profile = AcousticProfile(levels=())

    assert profile.minimum_level() is None


def test_rms_acoustic_analyzer_rejects_non_positive_window() -> None:
    from fluxtuner_ripper.ripping import RmsAcousticAnalyzer

    with pytest.raises(ValueError):
        RmsAcousticAnalyzer(window_seconds=0.0)


def _profile_from_rms(values: list[float], *, window_seconds: float = 0.5):
    from fluxtuner_ripper.ripping import AcousticLevel, AcousticProfile

    levels = []
    for index, rms in enumerate(values):
        start = index * window_seconds
        end = start + window_seconds
        levels.append(
            AcousticLevel(
                start_time_seconds=start,
                end_time_seconds=end,
                rms=rms,
            )
        )
    return AcousticProfile(levels=tuple(levels))


def test_acoustic_candidate_finder_finds_local_minima() -> None:
    from fluxtuner_ripper.ripping import AcousticCandidateFinder

    profile = _profile_from_rms([10.0, 2.0, 9.0, 1.0, 8.0])
    window = _make_acoustic_window(b"x" * 100)

    candidates = AcousticCandidateFinder().find(
        profile=profile,
        window=window,
    )

    assert [candidate.rms for candidate in candidates] == [2.0, 1.0]
    assert [candidate.relative_time_seconds for candidate in candidates] == pytest.approx(
        [0.75, 1.75]
    )


def test_acoustic_candidate_finder_maps_relative_to_absolute_time() -> None:
    from fluxtuner_ripper.ripping import AcousticCandidateFinder, AcousticWindow

    profile = _profile_from_rms([5.0, 1.0, 4.0])
    window = AcousticWindow(
        start_offset=100,
        end_offset=200,
        start_time_seconds=50.0,
        end_time_seconds=51.5,
        data=b"x" * 100,
    )

    candidates = AcousticCandidateFinder().find(
        profile=profile,
        window=window,
    )

    assert len(candidates) == 1
    assert candidates[0].relative_time_seconds == pytest.approx(0.75)
    assert candidates[0].time_seconds == pytest.approx(50.75)


def test_acoustic_candidate_finder_filters_by_center_radius() -> None:
    from fluxtuner_ripper.ripping import AcousticCandidateFinder, AcousticWindow

    profile = _profile_from_rms([9.0, 1.0, 8.0, 2.0, 7.0])
    window = AcousticWindow(
        start_offset=0,
        end_offset=100,
        start_time_seconds=100.0,
        end_time_seconds=102.5,
        data=b"x" * 100,
    )

    candidates = AcousticCandidateFinder().find(
        profile=profile,
        window=window,
        center_time_seconds=101.8,
        radius_seconds=0.4,
    )

    assert len(candidates) == 1
    assert candidates[0].time_seconds == pytest.approx(101.75)
    assert candidates[0].rms == pytest.approx(2.0)


def test_acoustic_candidate_finder_returns_empty_for_empty_profile() -> None:
    from fluxtuner_ripper.ripping import (
        AcousticCandidateFinder,
        AcousticProfile,
    )

    candidates = AcousticCandidateFinder().find(
        profile=AcousticProfile(levels=()),
        window=_make_acoustic_window(b"x" * 10),
    )

    assert candidates == ()


def test_acoustic_candidate_finder_accepts_edge_minima() -> None:
    from fluxtuner_ripper.ripping import AcousticCandidateFinder

    profile = _profile_from_rms([1.0, 5.0, 2.0])
    candidates = AcousticCandidateFinder().find(
        profile=profile,
        window=_make_acoustic_window(b"x" * 10),
    )

    assert [candidate.rms for candidate in candidates] == [1.0, 2.0]


def test_acoustic_candidate_finder_requires_center_for_radius() -> None:
    from fluxtuner_ripper.ripping import AcousticCandidateFinder

    with pytest.raises(ValueError):
        AcousticCandidateFinder().find(
            profile=_profile_from_rms([1.0]),
            window=_make_acoustic_window(b"x" * 10),
            radius_seconds=1.0,
        )


def _track_candidate(
    *,
    title: str = "Song",
    start_time_seconds: float,
    start_offset: int = 0,
):
    from fluxtuner_ripper.ripping import TrackCandidate

    return TrackCandidate(
        title=title,
        start_offset=start_offset,
        start_time_seconds=start_time_seconds,
        confirmed_at_offset=start_offset + 100,
        confirmed_at_time_seconds=start_time_seconds + 10.0,
    )


def _acoustic_candidate(*, time_seconds: float, rms: float):
    from fluxtuner_ripper.ripping import AcousticBoundaryCandidate

    return AcousticBoundaryCandidate(
        time_seconds=time_seconds,
        rms=rms,
        relative_time_seconds=time_seconds,
    )


def test_nearest_boundary_matcher_selects_closest_candidate() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher(search_radius_seconds=8.0)
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=94.0, rms=1.0),
            _acoustic_candidate(time_seconds=99.5, rms=5.0),
            _acoustic_candidate(time_seconds=102.0, rms=0.5),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(99.5)
    assert match.delta_seconds == pytest.approx(0.5)


def test_nearest_boundary_matcher_uses_rms_as_tie_breaker() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher(search_radius_seconds=8.0)
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=99.0, rms=3.0),
            _acoustic_candidate(time_seconds=101.0, rms=1.0),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(101.0)
    assert match.acoustic.rms == pytest.approx(1.0)


def test_nearest_boundary_matcher_returns_none_outside_radius() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher(search_radius_seconds=2.0)
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=97.5, rms=1.0),
            _acoustic_candidate(time_seconds=102.5, rms=1.0),
        ),
    )

    assert match is None


def test_nearest_boundary_matcher_returns_none_without_candidates() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher()
    track = _track_candidate(start_time_seconds=100.0)

    assert matcher.match(track=track, acoustic_candidates=()) is None


def test_nearest_boundary_matcher_prefers_earlier_time_after_full_tie() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher()
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=99.0, rms=1.0),
            _acoustic_candidate(time_seconds=101.0, rms=1.0),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(99.0)


def test_nearest_boundary_matcher_overrides_late_candidate_for_much_quieter_earlier_boundary() -> (
    None
):
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher()
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=98.5, rms=100.0),
            _acoustic_candidate(time_seconds=100.25, rms=2000.0),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(98.5)
    assert match.acoustic.rms == pytest.approx(100.0)


def test_nearest_boundary_matcher_keeps_late_candidate_when_quiet_ratio_is_too_small() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher()
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=98.5, rms=100.0),
            _acoustic_candidate(time_seconds=100.25, rms=500.0),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(100.25)


def test_nearest_boundary_matcher_does_not_override_candidate_already_before_metadata() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher()
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=98.5, rms=1.0),
            _acoustic_candidate(time_seconds=99.75, rms=1000.0),
            _acoustic_candidate(time_seconds=101.0, rms=2000.0),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(99.75)


def test_nearest_boundary_matcher_ignores_quieter_boundary_outside_override_radius() -> None:
    from fluxtuner_ripper.ripping import NearestBoundaryMatcher

    matcher = NearestBoundaryMatcher()
    track = _track_candidate(start_time_seconds=100.0)

    match = matcher.match(
        track=track,
        acoustic_candidates=(
            _acoustic_candidate(time_seconds=97.5, rms=0.1),
            _acoustic_candidate(time_seconds=100.25, rms=2000.0),
        ),
    )

    assert match is not None
    assert match.acoustic.time_seconds == pytest.approx(100.25)


def test_boundary_relation_classifier_marks_close_candidates_as_agreement() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationClassifier,
    )

    classifier = BoundaryRelationClassifier(divergence_threshold_seconds=3.0)

    result = classifier.classify(
        semantic_time_seconds=100.0,
        acoustic_time_seconds=98.5,
    )

    assert result.relation is BoundaryRelation.AGREEMENT
    assert result.signed_delta_seconds == pytest.approx(1.5)


def test_boundary_relation_classifier_marks_acoustic_candidate_earlier() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationClassifier,
    )

    classifier = BoundaryRelationClassifier(divergence_threshold_seconds=3.0)

    result = classifier.classify(
        semantic_time_seconds=110.0,
        acoustic_time_seconds=105.0,
    )

    assert result.relation is BoundaryRelation.ACOUSTIC_EARLIER
    assert result.signed_delta_seconds == pytest.approx(5.0)


def test_boundary_relation_classifier_marks_semantic_candidate_earlier() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationClassifier,
    )

    classifier = BoundaryRelationClassifier(divergence_threshold_seconds=3.0)

    result = classifier.classify(
        semantic_time_seconds=100.0,
        acoustic_time_seconds=105.0,
    )

    assert result.relation is BoundaryRelation.SEMANTIC_EARLIER
    assert result.signed_delta_seconds == pytest.approx(-5.0)


def test_boundary_relation_classifier_treats_exact_threshold_as_agreement() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationClassifier,
    )

    classifier = BoundaryRelationClassifier(divergence_threshold_seconds=3.0)

    early = classifier.classify(
        semantic_time_seconds=103.0,
        acoustic_time_seconds=100.0,
    )
    late = classifier.classify(
        semantic_time_seconds=100.0,
        acoustic_time_seconds=103.0,
    )

    assert early.relation is BoundaryRelation.AGREEMENT
    assert late.relation is BoundaryRelation.AGREEMENT


def test_boundary_relation_classifier_rejects_invalid_threshold() -> None:
    from fluxtuner_ripper.ripping import BoundaryRelationClassifier

    with pytest.raises(ValueError):
        BoundaryRelationClassifier(divergence_threshold_seconds=0.0)


def test_boundary_relation_classifier_rejects_negative_times() -> None:
    from fluxtuner_ripper.ripping import BoundaryRelationClassifier

    classifier = BoundaryRelationClassifier()

    with pytest.raises(ValueError):
        classifier.classify(
            semantic_time_seconds=-1.0,
            acoustic_time_seconds=0.0,
        )

    with pytest.raises(ValueError):
        classifier.classify(
            semantic_time_seconds=0.0,
            acoustic_time_seconds=-1.0,
        )


def test_temporal_split_policy_uses_acoustic_boundary_on_agreement() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationResult,
        TemporalSplitKind,
        TemporalSplitPolicy,
    )

    relation = BoundaryRelationResult(
        relation=BoundaryRelation.AGREEMENT,
        semantic_time_seconds=100.0,
        acoustic_time_seconds=99.2,
        signed_delta_seconds=0.8,
    )

    decision = TemporalSplitPolicy().decide(relation)

    assert decision.kind is TemporalSplitKind.HARD_CUT
    assert decision.incoming_start_seconds == pytest.approx(99.2)
    assert decision.outgoing_end_seconds == pytest.approx(99.2)


def test_temporal_split_policy_uses_semantic_boundary_when_it_is_earlier() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationResult,
        TemporalSplitKind,
        TemporalSplitPolicy,
    )

    relation = BoundaryRelationResult(
        relation=BoundaryRelation.AGREEMENT,
        semantic_time_seconds=100.0,
        acoustic_time_seconds=100.8,
        signed_delta_seconds=-0.8,
    )

    decision = TemporalSplitPolicy().decide(relation)

    assert decision.kind is TemporalSplitKind.HARD_CUT
    assert decision.incoming_start_seconds == pytest.approx(100.0)
    assert decision.outgoing_end_seconds == pytest.approx(100.0)


def test_temporal_split_policy_creates_crossfade_when_acoustic_is_earlier() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationResult,
        TemporalSplitKind,
        TemporalSplitPolicy,
    )

    relation = BoundaryRelationResult(
        relation=BoundaryRelation.ACOUSTIC_EARLIER,
        semantic_time_seconds=110.0,
        acoustic_time_seconds=105.0,
        signed_delta_seconds=5.0,
    )

    decision = TemporalSplitPolicy().decide(relation)

    assert decision.kind is TemporalSplitKind.CROSSFADE
    assert decision.incoming_start_seconds == pytest.approx(105.0)
    assert decision.outgoing_end_seconds == pytest.approx(110.0)


def test_temporal_split_policy_uses_acoustic_boundary_when_semantic_is_earlier() -> None:
    from fluxtuner_ripper.ripping import (
        BoundaryRelation,
        BoundaryRelationResult,
        TemporalSplitKind,
        TemporalSplitPolicy,
    )

    relation = BoundaryRelationResult(
        relation=BoundaryRelation.SEMANTIC_EARLIER,
        semantic_time_seconds=100.0,
        acoustic_time_seconds=105.0,
        signed_delta_seconds=-5.0,
    )

    decision = TemporalSplitPolicy().decide(relation)

    assert decision.kind is TemporalSplitKind.HARD_CUT
    assert decision.incoming_start_seconds == pytest.approx(105.0)
    assert decision.outgoing_end_seconds == pytest.approx(105.0)


def test_temporal_split_decision_rejects_invalid_hard_cut() -> None:
    from fluxtuner_ripper.ripping import TemporalSplitDecision, TemporalSplitKind

    with pytest.raises(ValueError):
        TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=10.0,
            outgoing_end_seconds=11.0,
        )


def test_temporal_split_decision_rejects_invalid_crossfade() -> None:
    from fluxtuner_ripper.ripping import TemporalSplitDecision, TemporalSplitKind

    with pytest.raises(ValueError):
        TemporalSplitDecision(
            kind=TemporalSplitKind.CROSSFADE,
            incoming_start_seconds=10.0,
            outgoing_end_seconds=10.0,
        )


def test_temporal_split_decision_accepts_exclusion_gap() -> None:
    from fluxtuner_ripper.ripping import TemporalSplitDecision, TemporalSplitKind

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        incoming_start_seconds=12.0,
        outgoing_end_seconds=10.0,
    )

    assert decision.outgoing_end_seconds < decision.incoming_start_seconds


def test_temporal_split_decision_rejects_collapsed_exclusion() -> None:
    from fluxtuner_ripper.ripping import TemporalSplitDecision, TemporalSplitKind

    with pytest.raises(ValueError, match="outgoing_end_seconds < incoming_start_seconds"):
        TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            incoming_start_seconds=10.0,
            outgoing_end_seconds=10.0,
        )


def test_temporal_split_aligner_maps_hard_cut_to_single_frame_offset() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 5)

    target = timeline.frames[2].time_seconds + 0.001

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=target,
        outgoing_end_seconds=target,
    )

    aligned = TemporalSplitAligner().align(
        decision=decision,
        timeline=timeline,
    )

    assert aligned.kind is SplitKind.HARD_CUT
    assert aligned.incoming_start == timeline.frames[2].offset
    assert aligned.outgoing_end == timeline.frames[2].offset


def test_temporal_split_aligner_maps_crossfade_edges_independently() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 8)

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=timeline.frames[2].time_seconds + 0.001,
        outgoing_end_seconds=timeline.frames[6].time_seconds - 0.001,
    )

    aligned = TemporalSplitAligner().align(
        decision=decision,
        timeline=timeline,
    )

    assert aligned.kind is SplitKind.CROSSFADE
    assert aligned.incoming_start == timeline.frames[2].offset
    assert aligned.outgoing_end == timeline.frames[6].offset


def test_temporal_split_aligner_maps_exclusion_edges_independently() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 8)

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=timeline.frames[2].time_seconds + 0.001,
        incoming_start_seconds=timeline.frames[6].time_seconds - 0.001,
    )

    aligned = TemporalSplitAligner().align(
        decision=decision,
        timeline=timeline,
    )

    assert aligned.kind is SplitKind.EXCLUSION
    assert aligned.outgoing_end == timeline.frames[2].offset
    assert aligned.incoming_start == timeline.frames[6].offset
    assert aligned.outgoing_end < aligned.incoming_start


def test_temporal_split_aligner_prefers_earlier_frame_on_exact_tie() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    first = _mp3_frame(bitrate_index=9)
    second = _mp3_frame(bitrate_index=9)

    timeline = IncrementalFrameTimeline("mp3")
    timeline.feed(first + second)

    midpoint = (timeline.frames[0].time_seconds + timeline.frames[1].time_seconds) / 2.0

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=midpoint,
        outgoing_end_seconds=midpoint,
    )

    aligned = TemporalSplitAligner().align(
        decision=decision,
        timeline=timeline,
    )

    assert aligned.incoming_start == timeline.frames[0].offset


def test_temporal_split_aligner_rejects_time_before_retained_timeline() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        SplitAlignmentError,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 3)
    timeline.discard_before(900)

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=0.0,
        outgoing_end_seconds=0.0,
    )

    with pytest.raises(SplitAlignmentError):
        TemporalSplitAligner().align(
            decision=decision,
            timeline=timeline,
        )


def test_temporal_split_aligner_rejects_time_after_retained_timeline() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        SplitAlignmentError,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 2)

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=10.0,
        outgoing_end_seconds=10.0,
    )

    with pytest.raises(SplitAlignmentError):
        TemporalSplitAligner().align(
            decision=decision,
            timeline=timeline,
        )


def test_temporal_split_aligner_rejects_collapsed_crossfade_after_alignment() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        SplitAlignmentError,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 3)

    base = timeline.frames[1].time_seconds

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=base,
        outgoing_end_seconds=base + 0.001,
    )

    with pytest.raises(SplitAlignmentError):
        TemporalSplitAligner().align(
            decision=decision,
            timeline=timeline,
        )


def test_temporal_split_aligner_rejects_collapsed_exclusion_after_alignment() -> None:
    from fluxtuner_ripper.ripping import (
        IncrementalFrameTimeline,
        SplitAlignmentError,
        TemporalSplitAligner,
        TemporalSplitDecision,
        TemporalSplitKind,
    )

    frame = _adts_frame(frame_length=900)
    timeline = IncrementalFrameTimeline("aac")
    timeline.feed(frame * 3)

    base = timeline.frames[1].time_seconds

    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=base,
        incoming_start_seconds=base + 0.001,
    )

    with pytest.raises(SplitAlignmentError):
        TemporalSplitAligner().align(
            decision=decision,
            timeline=timeline,
        )


def test_track_range_planner_builds_hard_cut_ranges() -> None:
    from fluxtuner_ripper.ripping import SplitDecision, SplitKind, TrackRangePlanner

    decision = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=1000,
        outgoing_end=1000,
    )

    plan = TrackRangePlanner().plan(
        previous_start_offset=100,
        next_end_offset=2000,
        decision=decision,
    )

    assert plan.outgoing.start_offset == 100
    assert plan.outgoing.end_offset == 1000
    assert plan.incoming.start_offset == 1000
    assert plan.incoming.end_offset == 2000


def test_track_range_planner_preserves_crossfade_overlap() -> None:
    from fluxtuner_ripper.ripping import SplitDecision, SplitKind, TrackRangePlanner

    decision = SplitDecision(
        kind=SplitKind.CROSSFADE,
        incoming_start=900,
        outgoing_end=1200,
    )

    plan = TrackRangePlanner().plan(
        previous_start_offset=100,
        next_end_offset=2000,
        decision=decision,
    )

    assert plan.outgoing.start_offset == 100
    assert plan.outgoing.end_offset == 1200
    assert plan.incoming.start_offset == 900
    assert plan.incoming.end_offset == 2000

    overlap = plan.outgoing.end_offset - plan.incoming.start_offset
    assert overlap == 300


def test_track_range_planner_preserves_exclusion_gap() -> None:
    from fluxtuner_ripper.ripping import SplitDecision, SplitKind, TrackRangePlanner

    decision = SplitDecision(
        kind=SplitKind.EXCLUSION,
        incoming_start=1200,
        outgoing_end=900,
    )

    plan = TrackRangePlanner().plan(
        previous_start_offset=100,
        next_end_offset=2000,
        decision=decision,
    )

    assert plan.outgoing.start_offset == 100
    assert plan.outgoing.end_offset == 900
    assert plan.incoming.start_offset == 1200
    assert plan.incoming.end_offset == 2000
    assert plan.incoming.start_offset - plan.outgoing.end_offset == 300


def test_track_range_planner_rejects_no_boundary() -> None:
    from fluxtuner_ripper.ripping import SplitDecision, SplitKind, TrackRangePlanner

    decision = SplitDecision(kind=SplitKind.NO_BOUNDARY)

    with pytest.raises(ValueError):
        TrackRangePlanner().plan(
            previous_start_offset=0,
            next_end_offset=1000,
            decision=decision,
        )


def test_encoded_track_writer_returns_exact_hard_cut_bytes() -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        EncodedTrackWriter,
        TrackByteRange,
    )

    source = EncodedAudioRingBuffer(max_bytes=20)
    source.append(b"abcdefghijklmnopqrst")

    written = EncodedTrackWriter().write_range(
        source=source,
        byte_range=TrackByteRange(start_offset=4, end_offset=10),
    )

    assert written == b"efghij"


def test_encoded_track_writer_can_materialize_overlapping_tracks() -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        EncodedTrackWriter,
        TrackByteRange,
    )

    source = EncodedAudioRingBuffer(max_bytes=20)
    source.append(b"abcdefghijklmnopqrst")
    writer = EncodedTrackWriter()

    outgoing = writer.write_range(
        source=source,
        byte_range=TrackByteRange(start_offset=0, end_offset=12),
    )
    incoming = writer.write_range(
        source=source,
        byte_range=TrackByteRange(start_offset=8, end_offset=20),
    )

    assert outgoing == b"abcdefghijkl"
    assert incoming == b"ijklmnopqrst"
    assert outgoing[8:12] == incoming[:4]


def test_encoded_track_writer_rejects_evicted_range() -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        EncodedTrackWriter,
        TrackByteRange,
    )

    source = EncodedAudioRingBuffer(max_bytes=5)
    source.append(b"abcdefghij")

    with pytest.raises(ValueError):
        EncodedTrackWriter().write_range(
            source=source,
            byte_range=TrackByteRange(start_offset=0, end_offset=5),
        )


def test_track_file_writer_uses_codec_extension() -> None:
    from fluxtuner_ripper.ripping import TrackFileWriter

    writer = TrackFileWriter()

    assert writer.extension_for_codec("mp3") == ".mp3"
    assert writer.extension_for_codec("aac") == ".aac"
    assert writer.extension_for_codec(" AAC ") == ".aac"


def test_track_file_writer_rejects_unsupported_codec() -> None:
    from fluxtuner_ripper.ripping import TrackFileWriter

    with pytest.raises(ValueError):
        TrackFileWriter().extension_for_codec("flac")


def test_track_file_writer_persists_exact_bytes(tmp_path) -> None:
    from fluxtuner_ripper.ripping import TrackFileWriter

    payload = b"encoded-track-bytes"
    writer = TrackFileWriter()

    path = writer.write(
        directory=tmp_path,
        stem="Artist - Track",
        codec="mp3",
        data=payload,
    )

    assert path == tmp_path / "Artist - Track.mp3"
    assert path.read_bytes() == payload


def test_track_file_writer_replaces_existing_target_atomically(tmp_path) -> None:
    from fluxtuner_ripper.ripping import TrackFileWriter

    target = tmp_path / "Track.aac"
    target.write_bytes(b"old")

    path = TrackFileWriter().write(
        directory=tmp_path,
        stem="Track",
        codec="aac",
        data=b"new-audio",
    )

    assert path == target
    assert path.read_bytes() == b"new-audio"


def test_track_file_writer_leaves_no_temp_file_after_success(tmp_path) -> None:
    from fluxtuner_ripper.ripping import TrackFileWriter

    TrackFileWriter().write(
        directory=tmp_path,
        stem="Track",
        codec="mp3",
        data=b"abc",
    )

    leftovers = [
        item
        for item in tmp_path.iterdir()
        if item.name.startswith(".Track.") and item.suffix == ".tmp"
    ]
    assert leftovers == []


def test_track_file_writer_rejects_empty_stem_and_data(tmp_path) -> None:
    from fluxtuner_ripper.ripping import TrackFileWriter

    writer = TrackFileWriter()

    with pytest.raises(ValueError):
        writer.write(
            directory=tmp_path,
            stem="",
            codec="mp3",
            data=b"abc",
        )

    with pytest.raises(ValueError):
        writer.write(
            directory=tmp_path,
            stem="Track",
            codec="mp3",
            data=b"",
        )


def test_mp3_track_finalizer_rejects_missing_binary() -> None:
    from fluxtuner_ripper.ripping import Mp3TrackFinalizer, TrackFinalizeError

    finalizer = Mp3TrackFinalizer(ffmpeg_binary="/definitely/missing/fluxtuner-ffmpeg")

    with pytest.raises(TrackFinalizeError, match="FFmpeg binary not found"):
        finalizer.finalize(b"not-an-mp3")


def test_mp3_track_finalizer_rejects_empty_input() -> None:
    from fluxtuner_ripper.ripping import Mp3TrackFinalizer

    with pytest.raises(ValueError):
        Mp3TrackFinalizer().finalize(b"")


def test_mp3_track_finalizer_rejects_invalid_mp3_when_ffmpeg_available() -> None:
    import shutil

    from fluxtuner_ripper.ripping import Mp3TrackFinalizer, TrackFinalizeError

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    with pytest.raises(TrackFinalizeError, match="MP3 finalization failed"):
        Mp3TrackFinalizer(ffmpeg_binary=ffmpeg).finalize(b"not-a-valid-mp3")


def test_mp3_track_finalizer_remuxes_mp3_without_transcoding() -> None:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    from fluxtuner_ripper.ripping import Mp3TrackFinalizer

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.0",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "mp3",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout

    finalized = Mp3TrackFinalizer(ffmpeg_binary=ffmpeg).finalize(generated)

    assert finalized

    probe = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "null",
            "-",
        ],
        input=finalized,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0


def test_mp3_track_finalizer_preserves_decodable_duration() -> None:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    from fluxtuner_ripper.ripping import Mp3TrackFinalizer

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=0.5",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "mp3",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout

    finalized = Mp3TrackFinalizer(ffmpeg_binary=ffmpeg).finalize(generated)

    before = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "pipe:1",
        ],
        input=generated,
        capture_output=True,
        check=True,
    ).stdout

    after = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "pipe:1",
        ],
        input=finalized,
        capture_output=True,
        check=True,
    ).stdout

    assert abs(len(before) - len(after)) <= 320


def test_aac_track_finalizer_rejects_missing_binary() -> None:
    from fluxtuner_ripper.ripping import AacTrackFinalizer, TrackFinalizeError

    finalizer = AacTrackFinalizer(ffmpeg_binary="/definitely/missing/fluxtuner-ffmpeg")

    with pytest.raises(TrackFinalizeError, match="FFmpeg binary not found"):
        finalizer.finalize(b"not-aac")


def test_aac_track_finalizer_rejects_empty_input() -> None:
    from fluxtuner_ripper.ripping import AacTrackFinalizer

    with pytest.raises(ValueError):
        AacTrackFinalizer().finalize(b"")


def test_aac_track_finalizer_rejects_invalid_aac_when_ffmpeg_available() -> None:
    import shutil

    from fluxtuner_ripper.ripping import AacTrackFinalizer, TrackFinalizeError

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    with pytest.raises(TrackFinalizeError, match="AAC finalization failed"):
        AacTrackFinalizer(ffmpeg_binary=ffmpeg).finalize(b"not-valid-aac")


def test_aac_track_finalizer_remuxes_adts_to_m4a_without_transcoding() -> None:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    from fluxtuner_ripper.ripping import AacTrackFinalizer

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:duration=1.0",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "aac",
            "-f",
            "adts",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout

    finalized = AacTrackFinalizer(ffmpeg_binary=ffmpeg).finalize(generated)

    assert finalized
    assert b"ftyp" in finalized[:64]

    probe = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "null",
            "-",
        ],
        input=finalized,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0


def test_aac_track_finalizer_preserves_decodable_duration() -> None:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    from fluxtuner_ripper.ripping import AacTrackFinalizer

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=550:duration=0.5",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "aac",
            "-f",
            "adts",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout

    finalized = AacTrackFinalizer(ffmpeg_binary=ffmpeg).finalize(generated)

    before = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "aac",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "pipe:1",
        ],
        input=generated,
        capture_output=True,
        check=True,
    ).stdout

    after = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "pipe:1",
        ],
        input=finalized,
        capture_output=True,
        check=True,
    ).stdout

    assert abs(len(before) - len(after)) <= 320


class _FakeFinalizer:
    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix
        self.calls: list[bytes] = []

    def finalize(self, data: bytes) -> bytes:
        self.calls.append(data)
        return self.prefix + data


def test_track_output_service_composes_mp3_pipeline(tmp_path) -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        TrackByteRange,
        TrackOutputService,
    )

    source = EncodedAudioRingBuffer(max_bytes=20)
    source.append(b"abcdefghijklmnopqrst")
    finalizer = _FakeFinalizer(b"MP3:")

    service = TrackOutputService(mp3_finalizer=finalizer)

    path = service.write_track(
        source=source,
        byte_range=TrackByteRange(start_offset=4, end_offset=10),
        directory=tmp_path,
        stem="Track",
        codec="mp3",
    )

    assert finalizer.calls == [b"efghij"]
    assert path == tmp_path / "Track.mp3"
    assert path.read_bytes() == b"MP3:efghij"


def test_track_output_service_composes_aac_to_m4a_pipeline(tmp_path) -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        TrackByteRange,
        TrackOutputService,
    )

    source = EncodedAudioRingBuffer(max_bytes=20)
    source.append(b"abcdefghijklmnopqrst")
    finalizer = _FakeFinalizer(b"M4A:")

    service = TrackOutputService(aac_finalizer=finalizer)

    path = service.write_track(
        source=source,
        byte_range=TrackByteRange(start_offset=8, end_offset=14),
        directory=tmp_path,
        stem="Track",
        codec="aac",
    )

    assert finalizer.calls == [b"ijklmn"]
    assert path == tmp_path / "Track.m4a"
    assert path.read_bytes() == b"M4A:ijklmn"


def test_track_output_service_rejects_unsupported_codec(tmp_path) -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        TrackByteRange,
        TrackOutputService,
    )

    source = EncodedAudioRingBuffer(max_bytes=10)
    source.append(b"abcdefghij")

    with pytest.raises(ValueError):
        TrackOutputService().write_track(
            source=source,
            byte_range=TrackByteRange(start_offset=0, end_offset=5),
            directory=tmp_path,
            stem="Track",
            codec="flac",
        )


def test_track_output_service_rejects_evicted_range(tmp_path) -> None:
    from fluxtuner_ripper.ripping import (
        EncodedAudioRingBuffer,
        TrackByteRange,
        TrackOutputService,
    )

    source = EncodedAudioRingBuffer(max_bytes=5)
    source.append(b"abcdefghij")

    with pytest.raises(ValueError):
        TrackOutputService(mp3_finalizer=_FakeFinalizer(b"MP3:")).write_track(
            source=source,
            byte_range=TrackByteRange(start_offset=0, end_offset=5),
            directory=tmp_path,
            stem="Track",
            codec="mp3",
        )


def test_public_package_api_exports_expected_symbols() -> None:
    import fluxtuner_ripper

    expected = {
        "AacTrackFinalizer",
        "AcousticCandidateFinder",
        "AcousticWindowExtractor",
        "BoundaryRelationClassifier",
        "EncodedAudioRingBuffer",
        "FfmpegAcousticDecoder",
        "IcyStreamParser",
        "IncrementalFrameTimeline",
        "MetadataSemanticTracker",
        "Mp3TrackFinalizer",
        "NearestBoundaryMatcher",
        "RippingStreamIngestor",
        "RmsAcousticAnalyzer",
        "TemporalSplitAligner",
        "TemporalSplitPolicy",
        "TrackOutputService",
        "TrackRangePlanner",
        "parse_adts_frames",
        "parse_mp3_frames",
    }

    assert expected <= set(fluxtuner_ripper.__all__)

    for name in expected:
        assert getattr(fluxtuner_ripper, name) is not None


def test_public_package_api_version_matches_project_bootstrap() -> None:
    import fluxtuner_ripper

    assert fluxtuner_ripper.__version__ == "0.1.0.dev0"


def test_encoded_track_writer_iterates_range_in_bounded_chunks() -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import EncodedTrackWriter

    class _BoundedSource:
        def __init__(self, data: bytes, max_read_size: int) -> None:
            self.data = data
            self.max_read_size = max_read_size
            self.reads: list[tuple[int, int]] = []

        @property
        def end_offset(self) -> int:
            return len(self.data)

        def contains(self, start: int, end: int) -> bool:
            return 0 <= start <= end <= len(self.data)

        def read(self, start: int, end: int) -> bytes:
            size = end - start
            if size > self.max_read_size:
                raise RuntimeError(f"unbounded read attempted: {size} > {self.max_read_size}")

            self.reads.append((start, end))
            return self.data[start:end]

    source = _BoundedSource(
        b"abcdefghijklmnopqrst",
        max_read_size=4,
    )

    chunks = tuple(
        EncodedTrackWriter().iter_range(
            source=source,
            byte_range=TrackByteRange(
                start_offset=2,
                end_offset=18,
            ),
            chunk_size=4,
        )
    )

    assert chunks == (
        b"cdef",
        b"ghij",
        b"klmn",
        b"opqr",
    )

    assert source.reads == [
        (2, 6),
        (6, 10),
        (10, 14),
        (14, 18),
    ]


def test_mp3_finalizer_streams_chunks_directly_to_output_file(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.output import Mp3TrackFinalizer

    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        """#!/usr/bin/env python3
import shutil
import sys

shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)
""",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)

    output_path = tmp_path / "segment.mp3"

    chunks = iter(
        (
            b"abcd",
            b"efgh",
            b"ijkl",
            b"mnop",
        )
    )

    finalizer = Mp3TrackFinalizer(
        ffmpeg_binary=str(fake_ffmpeg),
    )

    finalizer.finalize_stream(
        chunks=chunks,
        output_path=output_path,
    )

    assert output_path.read_bytes() == b"abcdefghijklmnop"


def test_aac_finalizer_streams_chunks_directly_to_output_file(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.output import AacTrackFinalizer

    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

output_path = pathlib.Path(sys.argv[-1])
output_path.write_bytes(sys.stdin.buffer.read())
""",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)

    output_path = tmp_path / "segment.m4a"

    chunks = iter(
        (
            b"abcd",
            b"efgh",
            b"ijkl",
            b"mnop",
        )
    )

    finalizer = AacTrackFinalizer(
        ffmpeg_binary=str(fake_ffmpeg),
    )

    finalizer.finalize_stream(
        chunks=chunks,
        output_path=output_path,
    )

    assert output_path.read_bytes() == b"abcdefghijklmnop"


def test_track_output_service_streams_mp3_without_full_range_read(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _BoundedSource:
        def __init__(self, data: bytes, max_read_size: int) -> None:
            self.data = data
            self.max_read_size = max_read_size
            self.reads: list[tuple[int, int]] = []

        @property
        def end_offset(self) -> int:
            return len(self.data)

        def contains(self, start: int, end: int) -> bool:
            return 0 <= start <= end <= len(self.data)

        def read(self, start: int, end: int) -> bytes:
            size = end - start
            if size > self.max_read_size:
                raise RuntimeError(f"unbounded read attempted: {size} > {self.max_read_size}")

            self.reads.append((start, end))
            return self.data[start:end]

    class _StreamingFinalizer:
        def __init__(self) -> None:
            self.received: list[bytes] = []

        def finalize(self, data: bytes) -> bytes:
            raise AssertionError("legacy whole-segment finalization must not be used")

        def finalize_stream(
            self,
            *,
            chunks,
            output_path: Path,
        ) -> None:
            with output_path.open("wb") as handle:
                for chunk in chunks:
                    self.received.append(chunk)
                    handle.write(chunk)

    source = _BoundedSource(
        b"abcdefghijklmnopqrst",
        max_read_size=4,
    )
    finalizer = _StreamingFinalizer()

    service = TrackOutputService(
        mp3_finalizer=finalizer,
    )

    path = service.write_track(
        source=source,
        byte_range=TrackByteRange(
            start_offset=2,
            end_offset=18,
        ),
        directory=tmp_path,
        stem="Track",
        codec="mp3",
        chunk_size=4,
    )

    assert path == tmp_path / "Track.mp3"
    assert path.read_bytes() == b"cdefghijklmnopqr"

    assert source.reads == [
        (2, 6),
        (6, 10),
        (10, 14),
        (14, 18),
    ]

    assert finalizer.received == [
        b"cdef",
        b"ghij",
        b"klmn",
        b"opqr",
    ]


def test_track_output_service_streams_aac_without_full_range_read(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _BoundedSource:
        def __init__(self, data: bytes, max_read_size: int) -> None:
            self.data = data
            self.max_read_size = max_read_size
            self.reads: list[tuple[int, int]] = []

        @property
        def end_offset(self) -> int:
            return len(self.data)

        def contains(self, start: int, end: int) -> bool:
            return 0 <= start <= end <= len(self.data)

        def read(self, start: int, end: int) -> bytes:
            size = end - start
            if size > self.max_read_size:
                raise RuntimeError(f"unbounded read attempted: {size} > {self.max_read_size}")

            self.reads.append((start, end))
            return self.data[start:end]

    class _StreamingFinalizer:
        def __init__(self) -> None:
            self.received: list[bytes] = []

        def finalize(self, data: bytes) -> bytes:
            raise AssertionError("legacy whole-segment finalization must not be used")

        def finalize_stream(
            self,
            *,
            chunks,
            output_path: Path,
        ) -> None:
            with output_path.open("wb") as handle:
                for chunk in chunks:
                    self.received.append(chunk)
                    handle.write(chunk)

    source = _BoundedSource(
        b"abcdefghijklmnopqrst",
        max_read_size=4,
    )
    finalizer = _StreamingFinalizer()

    service = TrackOutputService(
        aac_finalizer=finalizer,
    )

    path = service.write_track(
        source=source,
        byte_range=TrackByteRange(
            start_offset=2,
            end_offset=18,
        ),
        directory=tmp_path,
        stem="Track",
        codec="aac",
        chunk_size=4,
    )

    assert path == tmp_path / "Track.m4a"
    assert path.read_bytes() == b"cdefghijklmnopqr"

    assert source.reads == [
        (2, 6),
        (6, 10),
        (10, 14),
        (14, 18),
    ]

    assert finalizer.received == [
        b"cdef",
        b"ghij",
        b"klmn",
        b"opqr",
    ]


def test_track_output_service_rejects_segment_over_configured_size_limit(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _SourceThatMustNotBeRead:
        @property
        def end_offset(self) -> int:
            return 100

        def contains(self, start: int, end: int) -> bool:
            return True

        def read(self, start: int, end: int) -> bytes:
            raise AssertionError("oversized segment must be rejected before reading source bytes")

    service = TrackOutputService(
        max_segment_bytes=8,
    )

    with pytest.raises(
        ValueError,
        match="segment exceeds maximum allowed size",
    ):
        service.write_track(
            source=_SourceThatMustNotBeRead(),
            byte_range=TrackByteRange(
                start_offset=10,
                end_offset=19,
            ),
            directory=tmp_path,
            stem="oversized",
            codec="mp3",
            chunk_size=4,
        )

    assert not (tmp_path / "oversized.mp3").exists()


def test_track_output_service_rejects_when_free_disk_space_is_too_low(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _SourceThatMustNotBeRead:
        @property
        def end_offset(self) -> int:
            return 100

        def contains(self, start: int, end: int) -> bool:
            return True

        def read(self, start: int, end: int) -> bytes:
            raise AssertionError("disk-space rejection must happen before reading source bytes")

    class _DiskUsage:
        total = 1_000_000
        used = 950_000
        free = 50_000

    import shutil

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: _DiskUsage(),
    )

    service = TrackOutputService(
        min_free_output_bytes=100_000,
    )

    with pytest.raises(
        RuntimeError,
        match="insufficient free disk space",
    ):
        service.write_track(
            source=_SourceThatMustNotBeRead(),
            byte_range=TrackByteRange(
                start_offset=10,
                end_offset=20,
            ),
            directory=tmp_path,
            stem="disk-full",
            codec="mp3",
            chunk_size=4,
        )

    assert not (tmp_path / "disk-full.mp3").exists()


def test_track_output_service_requires_space_for_segment_plus_reserve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import shutil

    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _SourceThatMustNotBeRead:
        @property
        def end_offset(self) -> int:
            return 10_000

        def contains(self, start: int, end: int) -> bool:
            return True

        def read(self, start: int, end: int) -> bytes:
            raise AssertionError(
                "insufficient proportional disk space must be rejected before reading source bytes"
            )

    class _DiskUsage:
        total = 1_000_000
        used = 849_999
        free = 150_001

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: _DiskUsage(),
    )

    service = TrackOutputService(
        min_free_output_bytes=100_000,
        output_space_factor=1.0,
    )

    with pytest.raises(
        RuntimeError,
        match="insufficient free disk space",
    ):
        service.write_track(
            source=_SourceThatMustNotBeRead(),
            byte_range=TrackByteRange(
                start_offset=0,
                end_offset=100_000,
            ),
            directory=tmp_path,
            stem="large-segment",
            codec="mp3",
            chunk_size=4,
        )

    assert not (tmp_path / "large-segment.mp3").exists()


def test_track_output_service_rejects_chunk_size_above_hard_limit(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _SourceThatMustNotBeRead:
        @property
        def end_offset(self) -> int:
            return 100

        def contains(self, start: int, end: int) -> bool:
            return True

        def read(self, start: int, end: int) -> bytes:
            raise AssertionError(
                "oversized chunk_size must be rejected before reading source bytes"
            )

    service = TrackOutputService()

    with pytest.raises(
        ValueError,
        match="chunk_size exceeds maximum allowed size",
    ):
        service.write_track(
            source=_SourceThatMustNotBeRead(),
            byte_range=TrackByteRange(
                start_offset=0,
                end_offset=10,
            ),
            directory=tmp_path,
            stem="chunk-too-large",
            codec="mp3",
            chunk_size=16 * 1024 * 1024,
        )

    assert not (tmp_path / "chunk-too-large.mp3").exists()


def test_encoded_track_writer_rejects_source_returning_wrong_chunk_size() -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import EncodedTrackWriter

    class _BrokenSource:
        @property
        def end_offset(self) -> int:
            return 20

        def contains(self, start: int, end: int) -> bool:
            return True

        def read(self, start: int, end: int) -> bytes:
            requested = end - start
            return b"x" * (requested + 1)

    writer = EncodedTrackWriter()

    with pytest.raises(
        RuntimeError,
        match="encoded byte source returned unexpected chunk size",
    ):
        tuple(
            writer.iter_range(
                source=_BrokenSource(),
                byte_range=TrackByteRange(
                    start_offset=0,
                    end_offset=8,
                ),
                chunk_size=4,
            )
        )


@pytest.mark.parametrize(
    "stem",
    (
        "../escape",
        "nested/track",
        r"nested\track",
        ".",
        "..",
    ),
)
def test_track_output_service_rejects_unsafe_output_stem(
    tmp_path: Path,
    stem: str,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _SourceThatMustNotBeRead:
        @property
        def end_offset(self) -> int:
            return 100

        def contains(self, start: int, end: int) -> bool:
            return True

        def read(self, start: int, end: int) -> bytes:
            raise AssertionError("unsafe output stem must be rejected before reading source bytes")

    service = TrackOutputService()

    with pytest.raises(
        ValueError,
        match="unsafe output stem",
    ):
        service.write_track(
            source=_SourceThatMustNotBeRead(),
            byte_range=TrackByteRange(
                start_offset=0,
                end_offset=10,
            ),
            directory=tmp_path,
            stem=stem,
            codec="mp3",
            chunk_size=4,
        )
