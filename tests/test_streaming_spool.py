from pathlib import Path

import pytest

from fluxtuner_ripper.streaming_spool import StreamingSpool


def test_streaming_spool_uses_absolute_offsets(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        assert spool.append(b"abcde") == (0, 5)
        assert spool.append(b"fghij") == (5, 10)

        assert spool.start_offset == 0
        assert spool.end_offset == 10
        assert spool.retained_bytes == 10

        assert spool.read(2, 8) == b"cdefgh"
    finally:
        spool.close()


def test_streaming_spool_discards_prefix_without_changing_absolute_offsets(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(
        directory=tmp_path,
        copy_chunk_size=3,
    )

    try:
        spool.append(b"abcdefghijkl")

        spool.discard_before(5)

        assert spool.start_offset == 5
        assert spool.end_offset == 12
        assert spool.retained_bytes == 7

        assert spool.read(5, 12) == b"fghijkl"
        assert not spool.contains(0, 5)
        assert spool.contains(5, 12)
    finally:
        spool.close()


def test_streaming_spool_can_append_after_compaction(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        spool.append(b"abcdefghij")
        spool.discard_before(6)

        assert spool.append(b"klmnop") == (10, 16)

        assert spool.start_offset == 6
        assert spool.end_offset == 16
        assert spool.read(6, 16) == b"ghijklmnop"
    finally:
        spool.close()


def test_streaming_spool_can_discard_everything_and_continue(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        spool.append(b"abcdefgh")
        spool.discard_before(8)

        assert spool.start_offset == 8
        assert spool.end_offset == 8
        assert spool.retained_bytes == 0

        assert spool.append(b"ijkl") == (8, 12)
        assert spool.read(8, 12) == b"ijkl"
    finally:
        spool.close()


def test_streaming_spool_rejects_evicted_reads(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)

    try:
        spool.append(b"abcdefghij")
        spool.discard_before(5)

        with pytest.raises(
            ValueError,
            match="outside retained range",
        ):
            spool.read(0, 5)
    finally:
        spool.close()


def test_streaming_spool_close_removes_transient_file(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(directory=tmp_path)
    path = spool.path

    assert path.exists()

    spool.close()

    assert not path.exists()

    spool.close()


def test_streaming_spool_can_back_track_output_service(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.models import TrackByteRange
    from fluxtuner_ripper.output import TrackOutputService

    class _Finalizer:
        def finalize(self, data: bytes) -> bytes:
            return b"FINAL:" + data

    spool = StreamingSpool(directory=tmp_path / "spool")

    try:
        spool.append(b"abcdefghijklmnop")

        service = TrackOutputService(
            mp3_finalizer=_Finalizer(),  # type: ignore[arg-type]
        )

        path = service.write_track(
            source=spool,
            byte_range=TrackByteRange(
                start_offset=4,
                end_offset=10,
            ),
            directory=tmp_path / "output",
            stem="segment-0001",
            codec="mp3",
        )

        assert path.read_bytes() == b"FINAL:efghij"
    finally:
        spool.close()


def test_streaming_spool_reclaims_complete_storage_chunks_without_rewriting(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(
        directory=tmp_path,
        storage_chunk_size=4,
    )

    try:
        spool.append(b"abcdefghijkl")

        chunk_paths = list(tmp_path.glob(".fluxtuner-spool-*.chunk"))

        assert len(chunk_paths) == 3
        assert {path.read_bytes() for path in chunk_paths} == {
            b"abcd",
            b"efgh",
            b"ijkl",
        }

        retained_chunk = next(path for path in chunk_paths if path.read_bytes() == b"ijkl")
        retained_inode = retained_chunk.stat().st_ino

        spool.discard_before(8)

        remaining_paths = sorted(tmp_path.glob(".fluxtuner-spool-*.chunk"))

        assert remaining_paths == [retained_chunk]
        assert retained_chunk.stat().st_ino == retained_inode

        assert spool.start_offset == 8
        assert spool.end_offset == 12
        assert spool.retained_bytes == 4
        assert spool.read(8, 12) == b"ijkl"
    finally:
        spool.close()
