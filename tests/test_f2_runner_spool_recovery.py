"""F-2: unexpected runner failures must not destroy retained audio."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import brynse.integrations.radio.runner as runner_module
from brynse.output import SegmentFileWriteError
from brynse.source import BinaryIOStreamSource

PAYLOAD = b"F2-RECOVERY-PROBE-" * 128


@pytest.mark.parametrize(
    ("error", "expected_exception"),
    [
        pytest.param(
            RuntimeError("unexpected publication failure"),
            RuntimeError,
            id="unexpected",
        ),
        pytest.param(
            SegmentFileWriteError("recognized publication failure"),
            runner_module.RippingRunError,
            id="recognized",
        ),
    ],
)
def test_runner_preserves_real_spool_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_exception: type[Exception],
) -> None:
    class SyntheticSession:
        current_track = None

        def __init__(self, **kwargs: object) -> None:
            pass

        def feed(self, chunk: bytes) -> SimpleNamespace:
            return SimpleNamespace(
                ingest=SimpleNamespace(
                    audio=PAYLOAD,
                    timed_metadata_events=(),
                ),
                transitions=(
                    SimpleNamespace(
                        boundary=SimpleNamespace(
                            split=SimpleNamespace(
                                outgoing_end=100,
                                incoming_start=200,
                            )
                        )
                    ),
                ),
            )

    class FailingWriter:
        retained_start_offset = 0

        def __init__(self, **kwargs: object) -> None:
            pass

        def can_write_transition(self, *, outgoing_end: int, incoming_start: int) -> bool:
            return True

        def write_transition(self, transition: object) -> None:
            raise error

    monkeypatch.setattr(runner_module, "RippingSession", SyntheticSession)
    monkeypatch.setattr(runner_module, "SessionOutputWriter", FailingWriter)

    output = tmp_path / "output"
    source = BinaryIOStreamSource(
        io.BytesIO(b"stream"),
        metadata={
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        },
    )
    runner = runner_module.RippingRunner(
        runner_module.RippingRunConfig(
            url="https://example.invalid/f2",
            output_directory=output,
            codec="mp3",
        ),
        stream_opener=lambda url: source,
    )

    with pytest.raises(expected_exception):
        runner.run()

    spool_dir = output / ".brynse-spool"
    assert spool_dir.is_dir()

    chunks = tuple(path for path in spool_dir.iterdir() if path.is_file())
    assert chunks, "Spool was discarded after publication failure"

    persisted = b"".join(path.read_bytes() for path in chunks)
    assert PAYLOAD in persisted


def test_provenance_failure_preserves_real_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed mandatory receipt must not authorize encoded-byte discard."""
    import re

    observed: list[dict[str, object]] = []

    class SyntheticSession:
        current_track = None

        def __init__(self, **kwargs: object) -> None:
            pass

        def feed(self, chunk: bytes) -> SimpleNamespace:
            return SimpleNamespace(
                ingest=SimpleNamespace(
                    audio=PAYLOAD,
                    timed_metadata_events=(),
                ),
                transitions=(
                    SimpleNamespace(
                        boundary=SimpleNamespace(
                            split=SimpleNamespace(
                                outgoing_end=100,
                                incoming_start=200,
                            )
                        )
                    ),
                ),
            )

    class ReceiptWriter:
        retained_start_offset = 0

        def __init__(self, **kwargs: object) -> None:
            callback = kwargs.get("on_publication_confirmed")
            assert callable(callback), "Runner must enforce provenance"
            self.callback = callback

        def can_write_transition(self, *, outgoing_end: int, incoming_start: int) -> bool:
            return True

        def write_transition(self, transition: object) -> None:
            self.callback(object())
            # Reaching this point would simulate advancing after receipt.
            self.retained_start_offset = 200

    def fail_publish(**kwargs: object) -> Path:
        observed.append(kwargs)
        raise OSError("injected durable provenance failure")

    import brynse.publication_provenance as provenance_module

    monkeypatch.setattr(runner_module, "RippingSession", SyntheticSession)
    monkeypatch.setattr(runner_module, "SessionOutputWriter", ReceiptWriter)
    monkeypatch.setattr(provenance_module, "publish_provenance", fail_publish)

    output = tmp_path / "output"
    source = BinaryIOStreamSource(
        io.BytesIO(b"stream"),
        metadata={
            "Content-Type": "audio/mpeg",
            "icy-metaint": "1",
        },
    )
    runner = runner_module.RippingRunner(
        runner_module.RippingRunConfig(
            url="https://example.invalid/secret?token=private",
            output_directory=output,
            codec="mp3",
        ),
        stream_opener=lambda url: source,
    )

    with pytest.raises(
        runner_module.RippingRunError,
        match="injected durable provenance failure",
    ):
        runner.run()

    assert len(observed) == 1
    assert observed[0]["directory"] == output.resolve()
    assert re.fullmatch(r"[0-9a-f]{32}", str(observed[0]["session_id"]))
    assert str(observed[0]["source_id"]).startswith("sha256:")
    assert "private" not in str(observed[0]["source_id"])

    spool_dir = output / ".brynse-spool"
    assert spool_dir.is_dir()
    chunks = tuple(p for p in spool_dir.iterdir() if p.is_file())
    assert chunks, "Provenance failure must retain encoded spool"
    assert PAYLOAD in b"".join(p.read_bytes() for p in chunks)
