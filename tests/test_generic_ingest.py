from fluxtuner_ripper.ingest import EncodedStreamIngestor


def _adts_frame(
    *,
    frame_length: int,
    sample_rate_index: int = 4,
    payload_byte: int = 0,
) -> bytes:
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

    return bytes((b0, b1, b2, b3, b4, b5, b6)) + bytes([payload_byte]) * (frame_length - 7)


def test_encoded_stream_ingestor_builds_ring_and_timeline_without_metadata() -> None:
    first = _adts_frame(frame_length=100, payload_byte=1)
    second = _adts_frame(frame_length=120, payload_byte=2)

    ingestor = EncodedStreamIngestor(
        codec="aac",
        ring_max_bytes=10_000,
    )

    ingestor.feed(first + second)

    assert ingestor.ring_buffer.start_offset == 0
    assert ingestor.ring_buffer.end_offset == len(first) + len(second)
    assert (
        ingestor.ring_buffer.read(
            0,
            len(first) + len(second),
        )
        == first + second
    )

    frames = ingestor.timeline.frames

    assert len(frames) == 2
    assert frames[0].offset == 0
    assert frames[0].length == len(first)
    assert frames[0].time_seconds == 0.0

    assert frames[1].offset == len(first)
    assert frames[1].length == len(second)
    assert frames[1].time_seconds > 0.0


def test_encoded_stream_ingestor_accepts_incremental_chunks() -> None:
    first = _adts_frame(frame_length=100, payload_byte=1)
    second = _adts_frame(frame_length=120, payload_byte=2)

    ingestor = EncodedStreamIngestor(
        codec="aac",
        ring_max_bytes=10_000,
    )

    ingestor.feed(first)
    ingestor.feed(second)

    assert len(ingestor.timeline.frames) == 2
    assert ingestor.ring_buffer.end_offset == len(first) + len(second)


def test_encoded_stream_ingestor_ignores_empty_chunks() -> None:
    ingestor = EncodedStreamIngestor(
        codec="aac",
        ring_max_bytes=10_000,
    )

    ingestor.feed(b"")

    assert ingestor.ring_buffer.start_offset == 0
    assert ingestor.ring_buffer.end_offset == 0
    assert ingestor.timeline.frames == ()
