from io import BytesIO
from pathlib import Path

import pytest

from fluxtuner_ripper.generic_cli import GenericCliError, _read_input


def test_read_input_reads_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.aac"
    path.write_bytes(b"abc123")

    assert _read_input(str(path)) == b"abc123"


def test_read_input_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stdin:
        buffer = BytesIO(b"stdin-bytes")

    monkeypatch.setattr("sys.stdin", _Stdin())

    assert _read_input("-") == b"stdin-bytes"


def test_read_input_returns_empty_bytes_for_empty_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stdin:
        buffer = BytesIO(b"")

    monkeypatch.setattr("sys.stdin", _Stdin())

    assert _read_input("-") == b""


def test_generic_cli_error_is_runtime_error() -> None:
    error = GenericCliError("boom")

    assert isinstance(error, RuntimeError)


def test_run_generic_pipeline_returns_machine_readable_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    class _Runner:
        def __init__(
            self,
            *,
            ingestor: object,
            provider: object,
            resolver: object,
        ) -> None:
            self.ingestor = ingestor
            self.provider = provider
            self.resolver = resolver

        def run(self, chunks: object) -> object:
            assert tuple(chunks) == (b"encoded-data",)

            return SimpleNamespace(
                bytes_ingested=12,
                resolutions=(
                    SimpleNamespace(
                        candidate=SimpleNamespace(
                            time_seconds=30.0,
                            source="fixed_interval",
                        ),
                        temporal=SimpleNamespace(
                            incoming_start_seconds=30.125,
                        ),
                        split=SimpleNamespace(
                            incoming_start=1234,
                            outgoing_end=1234,
                            kind=SimpleNamespace(value="hard_cut"),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(generic_cli, "GenericRunner", _Runner)

    payload = generic_cli._run_generic_pipeline(
        data=b"encoded-data",
        codec="aac",
        interval_seconds=30.0,
    )

    assert payload == {
        "bytes_ingested": 12,
        "codec": "aac",
        "provider": "fixed_interval",
        "interval_seconds": 30.0,
        "boundaries": [
            {
                "requested_time_seconds": 30.0,
                "source": "fixed_interval",
                "resolved_time_seconds": 30.125,
                "incoming_start_offset": 1234,
                "outgoing_end_offset": 1234,
                "split_kind": "hard_cut",
            }
        ],
    }


def test_run_generic_pipeline_rejects_empty_input() -> None:
    import fluxtuner_ripper.generic_cli as generic_cli

    with pytest.raises(
        GenericCliError,
        match="input contains no encoded data",
    ):
        generic_cli._run_generic_pipeline(
            data=b"",
            codec="aac",
            interval_seconds=30.0,
        )


def test_run_generic_pipeline_materializes_segments_when_output_directory_is_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    resolution = SimpleNamespace(
        candidate=SimpleNamespace(
            time_seconds=30.0,
            source="fixed_interval",
        ),
        temporal=SimpleNamespace(
            incoming_start_seconds=30.0,
        ),
        split=SimpleNamespace(
            incoming_start=1000,
            outgoing_end=1000,
            kind=SimpleNamespace(value="hard_cut"),
        ),
    )

    class _Runner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, chunks: object) -> object:
            assert tuple(chunks) == (b"encoded-data",)
            return SimpleNamespace(
                bytes_ingested=12,
                resolutions=(resolution,),
            )

    class _SegmentWriter:
        def __init__(
            self,
            *,
            ingestor: object,
            directory: Path,
            codec: str,
            minimum_tail_seconds: float,
        ) -> None:
            assert directory == tmp_path
            assert codec == "mp3"
            assert minimum_tail_seconds == 1.0

        def write(self, resolutions: object) -> object:
            assert tuple(resolutions) == (resolution,)
            return (
                SimpleNamespace(
                    index=1,
                    start_offset=0,
                    end_offset=1000,
                    path=tmp_path / "segment-0001.mp3",
                ),
                SimpleNamespace(
                    index=2,
                    start_offset=1000,
                    end_offset=2000,
                    path=tmp_path / "segment-0002.mp3",
                ),
            )

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(generic_cli, "GenericRunner", _Runner)
    monkeypatch.setattr(generic_cli, "GenericSegmentWriter", _SegmentWriter)

    payload = generic_cli._run_generic_pipeline(
        data=b"encoded-data",
        codec="mp3",
        interval_seconds=30.0,
        output_directory=tmp_path,
    )

    assert payload["segments"] == [
        {
            "index": 1,
            "start_offset": 0,
            "end_offset": 1000,
            "path": str(tmp_path / "segment-0001.mp3"),
        },
        {
            "index": 2,
            "start_offset": 1000,
            "end_offset": 2000,
            "path": str(tmp_path / "segment-0002.mp3"),
        },
    ]


def test_run_generic_pipeline_uses_external_boundary_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    boundary_file = tmp_path / "boundaries.json"

    external_boundary = SimpleNamespace(
        time_seconds=12.5,
        source="agent",
        reference_offset=None,
    )

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes

    class _ExternalProvider:
        def __init__(self, *, boundaries: object) -> None:
            assert tuple(boundaries) == (external_boundary,)

    class _Resolver:
        pass

    class _Runner:
        def __init__(
            self,
            *,
            ingestor: object,
            provider: object,
            resolver: object,
        ) -> None:
            assert isinstance(provider, _ExternalProvider)

        def run(self, chunks: object) -> object:
            assert tuple(chunks) == (b"encoded-data",)

            return SimpleNamespace(
                bytes_ingested=12,
                resolutions=(),
            )

    def _load_external_boundaries(path: Path) -> object:
        assert path == boundary_file
        return (external_boundary,)

    monkeypatch.setattr(
        generic_cli,
        "load_external_boundaries",
        _load_external_boundaries,
    )
    monkeypatch.setattr(
        generic_cli,
        "ExternalBoundaryProvider",
        _ExternalProvider,
    )
    monkeypatch.setattr(
        generic_cli,
        "EncodedStreamIngestor",
        _Ingestor,
    )
    monkeypatch.setattr(
        generic_cli,
        "RippingOrchestrator",
        _Resolver,
    )
    monkeypatch.setattr(
        generic_cli,
        "GenericRunner",
        _Runner,
    )

    payload = generic_cli._run_generic_pipeline(
        data=b"encoded-data",
        codec="mp3",
        provider_name="external",
        boundaries_file=boundary_file,
    )

    assert payload["provider"] == "external"
    assert payload["boundaries"] == []


def test_iter_input_reads_file_incrementally(tmp_path: Path) -> None:
    import fluxtuner_ripper.generic_cli as generic_cli

    path = tmp_path / "input.aac"
    path.write_bytes(b"abcdefghij")

    chunks = tuple(
        generic_cli._iter_input(
            str(path),
            chunk_size=4,
        )
    )

    assert chunks == (
        b"abcd",
        b"efgh",
        b"ij",
    )


def test_run_generic_pipeline_accepts_incremental_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    received_chunks: list[bytes] = []

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    class _Runner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, chunks: object) -> object:
            received_chunks.extend(chunks)
            return SimpleNamespace(
                bytes_ingested=sum(len(chunk) for chunk in received_chunks),
                resolutions=(),
            )

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(generic_cli, "GenericRunner", _Runner)

    payload = generic_cli._run_generic_pipeline(
        chunks=iter((b"encoded-", b"data")),
        codec="aac",
        interval_seconds=30.0,
    )

    assert received_chunks == [
        b"encoded-",
        b"data",
    ]
    assert payload["bytes_ingested"] == 12


def test_run_generic_pipeline_materializes_completed_boundary_before_input_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    events: list[str] = []

    resolution = SimpleNamespace(
        candidate=SimpleNamespace(
            time_seconds=30.0,
            source="fixed_interval",
        ),
        temporal=SimpleNamespace(
            incoming_start_seconds=30.0,
        ),
        split=SimpleNamespace(
            incoming_start=1000,
            outgoing_end=1000,
            kind=SimpleNamespace(value="hard_cut"),
        ),
    )

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    class _BatchRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, chunks: object) -> object:
            bytes_ingested = 0

            for chunk in chunks:
                events.append(f"feed:{chunk.decode()}")
                bytes_ingested += len(chunk)

            return SimpleNamespace(
                bytes_ingested=bytes_ingested,
                resolutions=(resolution,),
            )

    class _BatchWriter:
        def __init__(self, **kwargs: object) -> None:
            pass

        def write(self, resolutions: object) -> object:
            assert tuple(resolutions) == (resolution,)
            events.append("accept")
            return ()

    class _Spool:
        def close(self) -> None:
            events.append("close")

    class _SpoolingIngestor:
        def __init__(self) -> None:
            self.spool = _Spool()

    class _StreamingRunner:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["provider"] is not None
            self.feed_count = 0

        def feed(self, chunk: bytes) -> tuple[object, ...]:
            self.feed_count += 1
            events.append(f"feed:{chunk.decode()}")

            if self.feed_count == 2:
                return (SimpleNamespace(resolution=resolution),)

            return ()

        def finalize(self) -> tuple[object, ...]:
            return ()

    class _StreamingSink:
        def __init__(self, **kwargs: object) -> None:
            pass

        def accept(self, received_resolution: object) -> object:
            assert received_resolution is resolution
            events.append("accept")

            return SimpleNamespace(
                index=1,
                start_offset=0,
                end_offset=1000,
                path=tmp_path / "segment-0001.mp3",
            )

        def finalize(self) -> None:
            return None

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(generic_cli, "GenericRunner", _BatchRunner)
    monkeypatch.setattr(generic_cli, "GenericSegmentWriter", _BatchWriter)

    monkeypatch.setattr(
        generic_cli,
        "create_safe_spooling_ingestor",
        lambda **kwargs: _SpoolingIngestor(),
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingGenericRunner",
        _StreamingRunner,
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingSegmentSink",
        _StreamingSink,
    )

    def chunks():
        yield b"one"
        yield b"two"

        assert "accept" in events

        yield b"three"

    payload = generic_cli._run_generic_pipeline(
        chunks=chunks(),
        codec="mp3",
        interval_seconds=30.0,
        output_directory=tmp_path,
    )

    assert payload["bytes_ingested"] == 11
    assert payload["segments"] == [
        {
            "index": 1,
            "start_offset": 0,
            "end_offset": 1000,
            "path": str(tmp_path / "segment-0001.mp3"),
        }
    ]

    assert events.index("accept") < events.index("feed:three")
    assert events[-1] == "close"


def test_run_generic_pipeline_finalizes_open_tail_at_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    events: list[str] = []

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    class _Spool:
        def close(self) -> None:
            events.append("close")

    class _SpoolingIngestor:
        def __init__(self) -> None:
            self.spool = _Spool()

    class _StreamingRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def feed(self, chunk: bytes) -> tuple[object, ...]:
            events.append(f"feed:{chunk.decode()}")
            return ()

        def finalize(self) -> tuple[object, ...]:
            return ()

    class _StreamingSink:
        def __init__(self, **kwargs: object) -> None:
            self.finalize_calls = 0

        def accept(self, resolution: object) -> object:
            raise AssertionError("no boundary should complete in this test")

        def finalize(self) -> object:
            self.finalize_calls += 1
            events.append("finalize")

            assert self.finalize_calls == 1

            return SimpleNamespace(
                index=1,
                start_offset=0,
                end_offset=300,
                path=tmp_path / "segment-0001.mp3",
            )

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(
        generic_cli,
        "create_safe_spooling_ingestor",
        lambda **kwargs: _SpoolingIngestor(),
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingGenericRunner",
        _StreamingRunner,
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingSegmentSink",
        _StreamingSink,
    )

    payload = generic_cli._run_generic_pipeline(
        chunks=(b"one", b"two", b"three"),
        codec="mp3",
        interval_seconds=30.0,
        output_directory=tmp_path,
    )

    assert events == [
        "feed:one",
        "feed:two",
        "feed:three",
        "finalize",
        "close",
    ]
    assert payload["segments"] == [
        {
            "index": 1,
            "start_offset": 0,
            "end_offset": 300,
            "path": str(tmp_path / "segment-0001.mp3"),
        }
    ]


def test_run_generic_pipeline_processes_pending_boundary_before_tail_at_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    events: list[str] = []

    resolution = SimpleNamespace(
        candidate=SimpleNamespace(
            time_seconds=10.0,
            source="manual",
        ),
        temporal=SimpleNamespace(
            incoming_start_seconds=10.0,
        ),
        split=SimpleNamespace(
            incoming_start=100,
            outgoing_end=100,
            kind=SimpleNamespace(value="hard_cut"),
        ),
    )

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes
            self.timeline = SimpleNamespace(
                frames=(
                    SimpleNamespace(
                        offset=0,
                        length=300,
                        time_seconds=0.0,
                        samples=1200,
                        sample_rate=100,
                    ),
                )
            )

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    class _Spool:
        def close(self) -> None:
            events.append("close")

    class _SpoolingIngestor:
        def __init__(self) -> None:
            self.spool = _Spool()

    class _StreamingRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def feed(self, chunk: bytes) -> tuple[object, ...]:
            events.append(f"feed:{chunk.decode()}")
            return ()

        def finalize(self) -> tuple[object, ...]:
            events.append("runner-finalize")
            return (
                SimpleNamespace(
                    candidate=resolution.candidate,
                    resolution=resolution,
                ),
            )

    class _StreamingSink:
        def __init__(self, **kwargs: object) -> None:
            pass

        def accept(self, received_resolution: object) -> object:
            assert received_resolution is resolution
            events.append("accept")

            return SimpleNamespace(
                index=1,
                start_offset=0,
                end_offset=100,
                path=tmp_path / "segment-0001.mp3",
            )

        def finalize(self) -> object:
            events.append("sink-finalize")

            return SimpleNamespace(
                index=2,
                start_offset=100,
                end_offset=300,
                path=tmp_path / "segment-0002.mp3",
            )

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(
        generic_cli,
        "create_safe_spooling_ingestor",
        lambda **kwargs: _SpoolingIngestor(),
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingGenericRunner",
        _StreamingRunner,
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingSegmentSink",
        _StreamingSink,
    )

    payload = generic_cli._run_generic_pipeline(
        chunks=(b"one", b"two"),
        codec="mp3",
        interval_seconds=30.0,
        output_directory=tmp_path,
    )

    assert events == [
        "feed:one",
        "feed:two",
        "runner-finalize",
        "accept",
        "sink-finalize",
        "close",
    ]

    assert payload["boundaries"] == [
        {
            "requested_time_seconds": 10.0,
            "source": "manual",
            "resolved_time_seconds": 10.0,
            "incoming_start_offset": 100,
            "outgoing_end_offset": 100,
            "split_kind": "hard_cut",
        }
    ]

    assert payload["segments"] == [
        {
            "index": 1,
            "start_offset": 0,
            "end_offset": 100,
            "path": str(tmp_path / "segment-0001.mp3"),
        },
        {
            "index": 2,
            "start_offset": 100,
            "end_offset": 300,
            "path": str(tmp_path / "segment-0002.mp3"),
        },
    ]


def test_run_generic_pipeline_merges_short_tail_after_eof_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import fluxtuner_ripper.generic_cli as generic_cli

    events: list[str] = []

    resolution = SimpleNamespace(
        candidate=SimpleNamespace(
            time_seconds=10.0,
            source="fixed_interval",
        ),
        temporal=SimpleNamespace(
            incoming_start_seconds=10.0,
        ),
        split=SimpleNamespace(
            incoming_start=100,
            outgoing_end=100,
            kind=SimpleNamespace(value="hard_cut"),
        ),
    )

    class _Ingestor:
        def __init__(self, *, codec: str, ring_max_bytes: int) -> None:
            self.codec = codec
            self.ring_max_bytes = ring_max_bytes
            self.timeline = SimpleNamespace(
                frames=(
                    SimpleNamespace(
                        offset=0,
                        length=100,
                        time_seconds=0.0,
                        samples=100,
                        sample_rate=100,
                    ),
                    SimpleNamespace(
                        offset=100,
                        length=50,
                        time_seconds=10.0,
                        samples=50,
                        sample_rate=100,
                    ),
                )
            )

    class _Provider:
        def __init__(self, *, interval_seconds: float) -> None:
            self.interval_seconds = interval_seconds

    class _Resolver:
        pass

    class _Spool:
        def close(self) -> None:
            events.append("close")

    class _SpoolingIngestor:
        def __init__(self) -> None:
            self.spool = _Spool()

    class _StreamingRunner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def feed(self, chunk: bytes) -> tuple[object, ...]:
            events.append(f"feed:{chunk.decode()}")
            return ()

        def finalize(self) -> tuple[object, ...]:
            events.append("runner-finalize")
            return (
                SimpleNamespace(
                    candidate=resolution.candidate,
                    resolution=resolution,
                ),
            )

    class _StreamingSink:
        def __init__(self, **kwargs: object) -> None:
            pass

        def accept(self, received_resolution: object) -> object:
            events.append("accept")
            raise AssertionError("short EOF tail must be merged instead of closing at boundary")

        def finalize(self) -> object:
            events.append("sink-finalize")
            return SimpleNamespace(
                index=1,
                start_offset=0,
                end_offset=150,
                path=tmp_path / "segment-0001.mp3",
            )

    monkeypatch.setattr(generic_cli, "EncodedStreamIngestor", _Ingestor)
    monkeypatch.setattr(generic_cli, "FixedIntervalBoundaryProvider", _Provider)
    monkeypatch.setattr(generic_cli, "RippingOrchestrator", _Resolver)
    monkeypatch.setattr(
        generic_cli,
        "create_safe_spooling_ingestor",
        lambda **kwargs: _SpoolingIngestor(),
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingGenericRunner",
        _StreamingRunner,
    )
    monkeypatch.setattr(
        generic_cli,
        "StreamingSegmentSink",
        _StreamingSink,
    )

    payload = generic_cli._run_generic_pipeline(
        chunks=(b"one", b"two"),
        codec="mp3",
        interval_seconds=30.0,
        output_directory=tmp_path,
        minimum_tail_seconds=1.0,
    )

    assert events == [
        "feed:one",
        "feed:two",
        "runner-finalize",
        "sink-finalize",
        "close",
    ]

    assert payload["boundaries"] == [
        {
            "requested_time_seconds": 10.0,
            "source": "fixed_interval",
            "resolved_time_seconds": 10.0,
            "incoming_start_offset": 100,
            "outgoing_end_offset": 100,
            "split_kind": "hard_cut",
        }
    ]

    assert payload["segments"] == [
        {
            "index": 1,
            "start_offset": 0,
            "end_offset": 150,
            "path": str(tmp_path / "segment-0001.mp3"),
        }
    ]


def test_main_streams_input_chunks_into_generic_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fluxtuner_ripper.generic_cli as generic_cli

    input_path = tmp_path / "input.mp3"
    input_path.write_bytes(b"abcdefghij")

    received_chunks: list[bytes] = []

    def fake_run_generic_pipeline(**kwargs: object) -> dict[str, object]:
        chunks = kwargs.get("chunks")
        assert chunks is not None
        received_chunks.extend(chunks)
        return {
            "bytes_ingested": sum(len(chunk) for chunk in received_chunks),
            "codec": "mp3",
            "provider": "fixed_interval",
            "interval_seconds": 30.0,
            "boundaries": [],
        }

    monkeypatch.setattr(
        generic_cli,
        "_run_generic_pipeline",
        fake_run_generic_pipeline,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "fluxtuner-ripper-segment",
            str(input_path),
            "--codec",
            "mp3",
            "--interval",
            "30",
        ],
    )

    assert generic_cli.main() == 0
    assert b"".join(received_chunks) == b"abcdefghij"
