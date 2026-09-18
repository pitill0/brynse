from __future__ import annotations

from collections.abc import Mapping

import pytest


def test_machine_segment_source_consumes_stream_source() -> None:
    from fluxtuner_ripper.machine import SegmentRequest, SegmentResult, segment_source

    class FakeSource:
        def __init__(self) -> None:
            self._chunks = [b"encoded-data", b""]
            self.closed = False

        @property
        def metadata(self) -> Mapping[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            assert max_bytes > 0
            return self._chunks.pop(0)

        def close(self) -> None:
            self.closed = True

    source = FakeSource()

    result = segment_source(
        source=source,
        request=SegmentRequest(
            codec="mp3",
            provider="fixed",
            interval_seconds=10.0,
        ),
    )

    assert isinstance(result, SegmentResult)
    assert source.closed is True


def test_segment_request_is_json_serializable() -> None:
    import json

    from fluxtuner_ripper.machine import SegmentRequest

    request = SegmentRequest(
        codec="mp3",
        provider="fixed",
        interval_seconds=10.0,
    )

    payload = request.to_dict()

    assert payload == {
        "codec": "mp3",
        "provider": "fixed",
        "interval_seconds": 10.0,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_segment_source_accepts_segment_request() -> None:

    from fluxtuner_ripper.machine import SegmentRequest, SegmentResult, segment_source

    class FakeSource:
        def __init__(self) -> None:
            self._chunks = [b"encoded-data", b""]
            self.closed = False

        @property
        def metadata(self) -> Mapping[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            self.closed = True

    source = FakeSource()
    request = SegmentRequest(
        codec="mp3",
        provider="fixed",
        interval_seconds=10.0,
    )

    result = segment_source(
        source=source,
        request=request,
    )

    assert isinstance(result, SegmentResult)
    assert source.closed is True


def test_segment_source_rejects_legacy_parameters() -> None:

    from fluxtuner_ripper.machine import segment_source

    class FakeSource:
        @property
        def metadata(self) -> Mapping[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            return b""

        def close(self) -> None:
            pass

    with pytest.raises(TypeError):
        segment_source(
            source=FakeSource(),
            codec="mp3",
            provider_name="fixed",
            interval_seconds=10.0,
        )


def test_segment_result_is_json_serializable() -> None:
    import json

    from fluxtuner_ripper.machine import SegmentResult

    result = SegmentResult(
        bytes_ingested=1234,
        codec="mp3",
        provider="fixed_interval",
        interval_seconds=10.0,
        boundaries=(),
        segments=(),
    )

    payload = result.to_dict()

    assert payload == {
        "bytes_ingested": 1234,
        "codec": "mp3",
        "provider": "fixed_interval",
        "interval_seconds": 10.0,
        "boundaries": [],
        "segments": [],
    }
    assert json.loads(json.dumps(payload)) == payload


def test_segment_source_returns_segment_result() -> None:

    from fluxtuner_ripper.machine import (
        SegmentRequest,
        SegmentResult,
        segment_source,
    )

    class FakeSource:
        def __init__(self) -> None:
            self._chunks = [b"encoded-data", b""]
            self.closed = False

        @property
        def metadata(self) -> Mapping[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            return self._chunks.pop(0)

        def close(self) -> None:
            self.closed = True

    source = FakeSource()

    result = segment_source(
        source=source,
        request=SegmentRequest(
            codec="mp3",
            provider="fixed",
            interval_seconds=10.0,
        ),
    )

    assert isinstance(result, SegmentResult)
    assert source.closed is True


def test_machine_error_is_json_serializable() -> None:
    import json

    from fluxtuner_ripper.machine import MachineError

    error = MachineError(
        code="invalid_request",
        message="codec is required",
        kind="validation",
    )

    payload = error.to_dict()

    assert payload == {
        "code": "invalid_request",
        "message": "codec is required",
        "kind": "validation",
    }
    assert json.loads(json.dumps(payload)) == payload
