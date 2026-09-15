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
