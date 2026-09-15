from pathlib import Path

from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.spooling_ingest import SpoolingEncodedStreamIngestor
from fluxtuner_ripper.streaming_spool import StreamingSpool


def _mp3_frame() -> bytes:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz.
    header = bytes.fromhex("fffb9000")
    frame_length = 417
    return header + bytes(frame_length - len(header))


def test_spooling_ingestor_fans_out_identical_bytes(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        ingestor = EncodedStreamIngestor(
            codec="mp3",
            ring_max_bytes=4096,
        )
        spooling = SpoolingEncodedStreamIngestor(
            ingestor=ingestor,
            spool=spool,
        )

        payload = _mp3_frame() + _mp3_frame()

        spooling.feed(payload)

        assert spool.start_offset == 0
        assert spool.end_offset == len(payload)
        assert spool.read(0, len(payload)) == payload

        assert ingestor.ring_buffer.end_offset == len(payload)
        assert len(ingestor.timeline.frames) == 2
    finally:
        spool.close()


def test_spooling_ingestor_preserves_absolute_offsets_across_feeds(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        ingestor = EncodedStreamIngestor(
            codec="mp3",
            ring_max_bytes=4096,
        )
        spooling = SpoolingEncodedStreamIngestor(
            ingestor=ingestor,
            spool=spool,
        )

        first = _mp3_frame()
        second = _mp3_frame()

        spooling.feed(first)
        spooling.feed(second)

        assert spool.end_offset == len(first) + len(second)
        assert ingestor.ring_buffer.end_offset == spool.end_offset

        frames = ingestor.timeline.frames

        assert frames[0].offset == 0
        assert frames[1].offset == len(first)
    finally:
        spool.close()


def test_spooling_ingestor_keeps_full_spool_when_ring_evicts(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        frame = _mp3_frame()
        payload = frame * 4

        ingestor = EncodedStreamIngestor(
            codec="mp3",
            ring_max_bytes=len(frame) * 2,
        )
        spooling = SpoolingEncodedStreamIngestor(
            ingestor=ingestor,
            spool=spool,
        )

        spooling.feed(payload)

        assert ingestor.ring_buffer.start_offset > 0

        assert spool.start_offset == 0
        assert spool.end_offset == len(payload)
        assert spool.read(0, len(payload)) == payload
    finally:
        spool.close()


def test_spooling_ingestor_preserves_first_frame_offset_after_ring_eviction(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        frame = _mp3_frame()

        ingestor = EncodedStreamIngestor(
            codec="mp3",
            ring_max_bytes=len(frame) * 2,
        )
        spooling = SpoolingEncodedStreamIngestor(
            ingestor=ingestor,
            spool=spool,
        )

        spooling.feed(frame)

        first_offset = spooling.first_frame_offset

        assert first_offset == 0

        spooling.feed(frame * 5)

        assert ingestor.ring_buffer.start_offset > first_offset
        assert ingestor.timeline.frames[0].offset > first_offset

        # The timeline and ring may rotate, but the stream anchor must not.
        assert spooling.first_frame_offset == first_offset
    finally:
        spool.close()
