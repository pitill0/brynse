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
    from fluxtuner_ripper.models import SegmentByteRange
    from fluxtuner_ripper.output import SegmentOutputService

    class _Finalizer:
        def finalize(self, data: bytes) -> bytes:
            return b"FINAL:" + data

    spool = StreamingSpool(directory=tmp_path / "spool")

    try:
        spool.append(b"abcdefghijklmnop")

        service = SegmentOutputService(
            mp3_finalizer=_Finalizer(),  # type: ignore[arg-type]
        )

        path = service.write_segment(
            source=spool,
            byte_range=SegmentByteRange(
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


def test_streaming_spool_discards_inside_chunk_without_rewriting_it(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(
        directory=tmp_path,
        storage_chunk_size=8,
    )

    try:
        spool.append(b"abcdefghijkl")

        chunk_paths = list(tmp_path.glob(".fluxtuner-spool-*.chunk"))
        first_chunk = next(path for path in chunk_paths if path.read_bytes() == b"abcdefgh")
        first_inode = first_chunk.stat().st_ino

        spool.discard_before(5)

        assert spool.start_offset == 5
        assert spool.end_offset == 12
        assert spool.retained_bytes == 7
        assert spool.read(5, 12) == b"fghijkl"

        # Partial discard must advance the logical head only.
        assert first_chunk.exists()
        assert first_chunk.stat().st_ino == first_inode
        assert first_chunk.read_bytes() == b"abcdefgh"
    finally:
        spool.close()


def test_streaming_spool_survives_repeated_append_discard_cycles(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(
        directory=tmp_path,
        storage_chunk_size=4,
    )

    try:
        assert spool.append(b"abcdefgh") == (0, 8)
        spool.discard_before(3)

        assert spool.read(3, 8) == b"defgh"

        assert spool.append(b"ijkl") == (8, 12)
        spool.discard_before(7)

        assert spool.start_offset == 7
        assert spool.end_offset == 12
        assert spool.read(7, 12) == b"hijkl"

        assert spool.append(b"mnopqr") == (12, 18)
        spool.discard_before(13)

        assert spool.start_offset == 13
        assert spool.end_offset == 18
        assert spool.retained_bytes == 5
        assert spool.read(13, 18) == b"nopqr"

        assert not spool.contains(12, 13)
        assert spool.contains(13, 18)
    finally:
        spool.close()


def test_streaming_spool_keeps_open_chunk_file_descriptors_bounded(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(
        directory=tmp_path,
        storage_chunk_size=4,
    )

    try:
        # Eight physical chunks, all still logically retained.
        spool.append(b"abcdefghijklmnopqrstuvwxyz012345")

        chunk_paths = set(tmp_path.glob(".fluxtuner-spool-*.chunk"))

        assert len(chunk_paths) == 8

        open_chunk_paths: set[Path] = set()

        for fd_path in Path("/proc/self/fd").iterdir():
            try:
                target = fd_path.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue

            if target in chunk_paths:
                open_chunk_paths.add(target)

        assert len(open_chunk_paths) <= 1
    finally:
        spool.close()


def test_streaming_spool_rejects_append_beyond_retained_byte_limit(
    tmp_path: Path,
) -> None:
    spool = StreamingSpool(
        directory=tmp_path,
        max_retained_bytes=8,
    )

    try:
        assert spool.append(b"abcdef") == (0, 6)

        with pytest.raises(
            RuntimeError,
            match="streaming spool retained byte limit exceeded",
        ):
            spool.append(b"ghij")

        assert spool.start_offset == 0
        assert spool.end_offset == 6
        assert spool.retained_bytes == 6
        assert spool.read(0, 6) == b"abcdef"
    finally:
        spool.close()


def test_safe_streaming_spool_applies_runtime_retention_limit(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.streaming_spool import create_safe_streaming_spool

    spool = create_safe_streaming_spool(
        directory=tmp_path,
    )

    try:
        assert spool._max_retained_bytes == 8 * 1024 * 1024 * 1024
    finally:
        spool.close()


def test_streaming_spool_rejects_append_when_free_disk_space_is_too_low(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import shutil

    class _DiskUsage:
        total = 1_000_000
        used = 950_000
        free = 50_000

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _: _DiskUsage(),
    )

    spool = StreamingSpool(
        directory=tmp_path,
        min_free_bytes=100_000,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="insufficient free disk space for streaming spool",
        ):
            spool.append(b"abcdef")

        assert spool.start_offset == 0
        assert spool.end_offset == 0
        assert spool.retained_bytes == 0
    finally:
        spool.close()


def test_safe_streaming_spool_applies_runtime_disk_reserve(
    tmp_path: Path,
) -> None:
    from fluxtuner_ripper.streaming_spool import create_safe_streaming_spool

    spool = create_safe_streaming_spool(
        directory=tmp_path,
    )

    try:
        assert spool._min_free_bytes == 512 * 1024 * 1024
    finally:
        spool.close()
