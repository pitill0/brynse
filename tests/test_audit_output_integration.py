"""Acceptance contracts for the three real output publication routes."""

from __future__ import annotations

from pathlib import Path

import pytest

import brynse._recoverable_publication as publication
from brynse.models import SegmentByteRange
from brynse.output import SegmentFileWriter, SegmentOutputService


@pytest.fixture
def observed_publications(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Observe actual calls to the recoverable core without mocking its work."""
    recorded: list[Path] = []
    original = publication.publish_fd_recoverably

    def observe(**kwargs):
        receipt = original(**kwargs)
        recorded.append(receipt.published_path)
        return receipt

    monkeypatch.setattr(publication, "publish_fd_recoverably", observe)
    return recorded


@pytest.mark.parametrize(
    ("codec", "extension"),
    [("mp3", ".mp3"), ("aac", ".aac")],
)
def test_segment_file_writer_uses_recoverable_core(
    tmp_path: Path,
    observed_publications: list[Path],
    codec: str,
    extension: str,
) -> None:
    target = SegmentFileWriter().write(
        directory=tmp_path,
        stem="Radio",
        codec=codec,
        data=b"COMPLETE_RECORDING",
    )
    assert target == tmp_path / f"Radio{extension}"
    assert target.read_bytes() == b"COMPLETE_RECORDING"
    assert observed_publications == [target]


def test_m4a_output_uses_recoverable_core(
    tmp_path: Path,
    observed_publications: list[Path],
) -> None:
    target = SegmentOutputService()._write_m4a(
        directory=tmp_path,
        stem="Radio",
        data=b"COMPLETE_M4A",
    )
    assert target == tmp_path / "Radio.m4a"
    assert target.read_bytes() == b"COMPLETE_M4A"
    assert observed_publications == [target]


class _Source:
    def __init__(self, data: bytes) -> None:
        self.data = data

    @property
    def end_offset(self) -> int:
        return len(self.data)

    def contains(self, start: int, end: int) -> bool:
        return 0 <= start <= end <= len(self.data)

    def read(self, start: int, end: int) -> bytes:
        return self.data[start:end]


class _StreamingFinalizer:
    def finalize_stream(
        self,
        *,
        chunks,
        output_path: Path,
        timeout_seconds: float | None = None,
    ) -> None:
        with output_path.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)


@pytest.mark.parametrize(
    ("codec", "extension"),
    [("mp3", ".mp3"), ("aac", ".m4a")],
)
def test_streaming_finalizer_uses_recoverable_core(
    tmp_path: Path,
    observed_publications: list[Path],
    codec: str,
    extension: str,
) -> None:
    finalizer = _StreamingFinalizer()
    service = SegmentOutputService(
        mp3_finalizer=finalizer,
        aac_finalizer=finalizer,
    )
    payload = b"COMPLETE_STREAM"

    target = service.write_segment(
        source=_Source(payload),
        byte_range=SegmentByteRange(
            start_offset=0,
            end_offset=len(payload),
        ),
        directory=tmp_path,
        stem="Radio",
        codec=codec,
        chunk_size=4,
    )

    assert target == tmp_path / f"Radio{extension}"
    assert target.read_bytes() == payload
    assert observed_publications == [target]


def test_output_service_has_no_shadowed_methods_or_legacy_publication() -> None:
    """Prevent hidden methods from bypassing recoverable publication."""
    import ast
    import inspect

    import brynse.output as output_module

    source = inspect.getsource(output_module)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(methods) == len(set(methods)), f"duplicate method definition in {node.name}"

    assert not any(
        isinstance(node, ast.Call)
        and (
            getattr(node.func, "id", None) == "publish_unique"
            or getattr(node.func, "attr", None) == "publish_unique"
        )
        for node in ast.walk(tree)
    ), "output.py must not use the vulnerable legacy publication API"
