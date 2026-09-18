from __future__ import annotations

from collections.abc import Mapping

import pytest

from fluxtuner_ripper.machine import MachineError, SegmentResult


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


def test_machine_classifies_validation_error() -> None:
    from fluxtuner_ripper.generic_cli import GenericCliError
    from fluxtuner_ripper.machine import MachineError, classify_machine_error

    error = classify_machine_error(GenericCliError("--interval must be greater than zero"))

    assert error == MachineError(
        code="invalid_request",
        message="--interval must be greater than zero",
        kind="validation",
    )


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (OSError("disk full"), "io_error"),
    ],
)
def test_machine_classifies_operational_errors(
    exc: Exception,
    code: str,
) -> None:
    from fluxtuner_ripper.machine import MachineError, classify_machine_error

    error = classify_machine_error(exc)

    assert error == MachineError(
        code=code,
        message=str(exc),
        kind="operational",
    )


def test_run_segment_source_returns_structured_error() -> None:

    from fluxtuner_ripper.machine import (
        MachineError,
        SegmentRequest,
        run_segment_source,
    )

    class EmptySource:
        @property
        def metadata(self) -> Mapping[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            return b""

        def close(self) -> None:
            pass

    result = run_segment_source(
        source=EmptySource(),
        request=SegmentRequest(
            codec="mp3",
            provider="fixed",
            interval_seconds=10.0,
        ),
    )

    assert result == MachineError(
        code="invalid_request",
        message="input contains no encoded data",
        kind="validation",
    )


def test_run_segment_source_returns_segment_result() -> None:

    from fluxtuner_ripper.machine import (
        SegmentRequest,
        SegmentResult,
        run_segment_source,
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

    result = run_segment_source(
        source=source,
        request=SegmentRequest(
            codec="mp3",
            provider="fixed",
            interval_seconds=10.0,
        ),
    )

    assert isinstance(result, SegmentResult)
    assert source.closed is True


def test_segment_request_from_json() -> None:
    from fluxtuner_ripper.machine import SegmentRequest, segment_request_from_json

    request = segment_request_from_json(
        '{"codec":"mp3","provider":"fixed","interval_seconds":10.0}'
    )

    assert request == SegmentRequest(
        codec="mp3",
        provider="fixed",
        interval_seconds=10.0,
    )


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        '{"provider":"fixed","interval_seconds":10.0}',
    ],
)
def test_segment_request_from_json_returns_machine_error_for_invalid_input(
    raw: str,
) -> None:
    from fluxtuner_ripper.machine import MachineError, segment_request_from_json

    result = segment_request_from_json(raw)

    assert isinstance(result, MachineError)
    assert result.code == "invalid_request"
    assert result.kind == "validation"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            MachineError(
                code="invalid_request",
                message="bad request",
                kind="validation",
            ),
            {
                "code": "invalid_request",
                "kind": "validation",
                "message": "bad request",
            },
        ),
        (
            SegmentResult(
                bytes_ingested=12,
                codec="mp3",
                provider="fixed_interval",
                interval_seconds=10.0,
                boundaries=(),
                segments=(),
            ),
            {
                "boundaries": [],
                "bytes_ingested": 12,
                "codec": "mp3",
                "interval_seconds": 10.0,
                "provider": "fixed_interval",
                "segments": [],
            },
        ),
    ],
)
def test_machine_response_to_json(
    value: MachineError | SegmentResult,
    expected: dict[str, object],
) -> None:
    import json

    from fluxtuner_ripper.machine import machine_response_to_json

    raw = machine_response_to_json(value)

    assert json.loads(raw) == expected


def test_run_segment_json_returns_structured_json_error() -> None:
    import json

    from fluxtuner_ripper.machine import run_segment_json

    class EmptySource:
        @property
        def metadata(self) -> Mapping[str, str]:
            return {}

        def read(self, max_bytes: int) -> bytes:
            return b""

        def close(self) -> None:
            pass

    raw = run_segment_json(
        source=EmptySource(),
        request_json='{"codec":"mp3","provider":"fixed","interval_seconds":10.0}',
    )

    assert json.loads(raw) == {
        "code": "invalid_request",
        "kind": "validation",
        "message": "input contains no encoded data",
    }


def test_run_segment_json_returns_structured_json_result() -> None:
    import json

    from fluxtuner_ripper.machine import run_segment_json

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

    raw = run_segment_json(
        source=source,
        request_json='{"codec":"mp3","provider":"fixed","interval_seconds":10.0}',
    )

    payload = json.loads(raw)

    assert payload["codec"] == "mp3"
    assert payload["provider"] == "fixed_interval"
    assert payload["bytes_ingested"] == len(b"encoded-data")
    assert payload["boundaries"] == []
    assert payload["segments"] == []
    assert source.closed is True
