from collections.abc import Mapping


def test_stream_source_contract_is_structural() -> None:
    from fluxtuner_ripper.source import StreamSource

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
