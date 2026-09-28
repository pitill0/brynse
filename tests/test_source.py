from collections.abc import Mapping


def test_stream_source_contract_is_structural() -> None:
    from brynse.source import StreamSource

    class FakeSource:
        metadata: Mapping[str, str] = {"content-type": "audio/mpeg"}

        def read(self, max_bytes: int) -> bytes:
            assert max_bytes > 0
            return b"audio"

        def close(self) -> None:
            pass

    source: StreamSource = FakeSource()

    assert source.read(64 * 1024) == b"audio"
    assert source.metadata["content-type"] == "audio/mpeg"
    source.close()


def test_binary_io_stream_source_adapts_existing_binary_stream() -> None:
    from io import BytesIO

    from brynse.source import BinaryIOStreamSource

    stream = BytesIO(b"abcdef")
    source = BinaryIOStreamSource(
        stream,
        metadata={"content-type": "audio/mpeg"},
    )

    assert source.read(2) == b"ab"
    assert source.read(3) == b"cde"
    assert source.metadata == {"content-type": "audio/mpeg"}

    source.close()

    assert stream.closed
