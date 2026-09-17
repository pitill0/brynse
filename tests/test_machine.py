from __future__ import annotations

from collections.abc import Mapping


def test_machine_segment_source_consumes_stream_source() -> None:
    from fluxtuner_ripper.machine import segment_source

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
        codec="mp3",
        provider_name="fixed",
        interval_seconds=10.0,
    )

    assert isinstance(result, dict)
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

    from fluxtuner_ripper.machine import SegmentRequest, segment_source

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

    assert isinstance(result, dict)
    assert source.closed is True
