from fluxtuner_ripper.generic_runner import GenericRunner, GenericRunResult
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    BoundaryCandidate,
    BoundaryRelation,
    BoundaryRelationResult,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from fluxtuner_ripper.orchestrator import CandidateResolution


def _adts_frame(
    *,
    frame_length: int = 100,
    sample_rate_index: int = 4,
    payload_byte: int = 0,
) -> bytes:
    profile = 1
    channel_config = 2

    b0 = 0xFF
    b1 = 0xF1
    b2 = (profile << 6) | (sample_rate_index << 2) | (channel_config >> 2)
    b3 = ((channel_config & 0x03) << 6) | ((frame_length >> 11) & 0x03)
    b4 = (frame_length >> 3) & 0xFF
    b5 = ((frame_length & 0x07) << 5) | 0x1F
    b6 = 0xFC

    return bytes((b0, b1, b2, b3, b4, b5, b6)) + bytes([payload_byte]) * (frame_length - 7)


class _Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def propose(
        self,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
    ) -> tuple[BoundaryCandidate, ...]:
        self.calls.append((start_time_seconds, end_time_seconds))

        return (
            BoundaryCandidate(
                time_seconds=end_time_seconds / 3,
                source="test",
            ),
            BoundaryCandidate(
                time_seconds=2 * end_time_seconds / 3,
                source="test",
            ),
        )


class _Resolver:
    def __init__(self) -> None:
        self.candidates: list[BoundaryCandidate] = []

    def resolve_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        timeline: object,
        ring_buffer: object,
    ) -> CandidateResolution | None:
        self.candidates.append(candidate)

        if len(self.candidates) == 2:
            return None

        temporal = TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=candidate.time_seconds,
            outgoing_end_seconds=candidate.time_seconds,
        )

        split = SplitDecision(
            kind=SplitKind.HARD_CUT,
            incoming_start=100,
            outgoing_end=100,
        )

        return CandidateResolution(
            candidate=candidate,
            acoustic=AcousticBoundaryCandidate(
                time_seconds=candidate.time_seconds,
                rms=0.1,
                relative_time_seconds=0.0,
            ),
            relation=BoundaryRelationResult(
                relation=BoundaryRelation.AGREEMENT,
                semantic_time_seconds=candidate.time_seconds,
                acoustic_time_seconds=candidate.time_seconds,
                signed_delta_seconds=0.0,
            ),
            temporal=temporal,
            split=split,
        )


def test_generic_runner_ingests_finite_stream_and_resolves_provider_candidates() -> None:
    ingestor = EncodedStreamIngestor(
        codec="aac",
        ring_max_bytes=10_000,
    )
    provider = _Provider()
    resolver = _Resolver()

    runner = GenericRunner(
        ingestor=ingestor,
        provider=provider,
        resolver=resolver,
    )

    first = _adts_frame(payload_byte=1)
    second = _adts_frame(payload_byte=2)
    third = _adts_frame(payload_byte=3)

    result = runner.run((first, second, third))

    assert result.bytes_ingested == len(first) + len(second) + len(third)
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == 0.0
    assert provider.calls[0][1] > 0.0

    assert len(resolver.candidates) == 2
    assert len(result.resolutions) == 1
    assert result.resolutions[0].candidate == resolver.candidates[0]


def test_generic_runner_returns_empty_result_without_frames() -> None:
    ingestor = EncodedStreamIngestor(
        codec="aac",
        ring_max_bytes=10_000,
    )

    provider = _Provider()
    resolver = _Resolver()

    runner = GenericRunner(
        ingestor=ingestor,
        provider=provider,
        resolver=resolver,
    )

    result = runner.run((b"",))

    assert result == GenericRunResult(
        bytes_ingested=0,
        resolutions=(),
    )
    assert provider.calls == []
    assert resolver.candidates == []
