from fluxtuner_ripper.acoustic import AcousticWindowExtractor
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    SplitKind,
)
from fluxtuner_ripper.orchestrator import RippingOrchestrator
from fluxtuner_ripper.providers import FixedIntervalBoundaryProvider


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


class _Decoder:
    def decode(self, window: object) -> object:
        return object()


class _Analyzer:
    def analyze(self, pcm: object) -> object:
        return object()


class _Finder:
    def find(
        self,
        *,
        profile: object,
        window: object,
        center_time_seconds: float,
    ) -> tuple[AcousticBoundaryCandidate, ...]:
        return (
            AcousticBoundaryCandidate(
                time_seconds=center_time_seconds,
                rms=0.1,
                relative_time_seconds=center_time_seconds,
            ),
        )


def test_non_radio_stream_reaches_frame_aligned_split() -> None:
    ingestor = EncodedStreamIngestor(
        codec="aac",
        ring_max_bytes=10_000,
    )

    audio = b"".join(
        _adts_frame(
            frame_length=100,
            payload_byte=index,
        )
        for index in range(8)
    )
    ingestor.feed(audio)

    provider = FixedIntervalBoundaryProvider(
        interval_seconds=0.05,
    )

    candidates = provider.propose(
        start_time_seconds=0.0,
        end_time_seconds=0.06,
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.source == "fixed_interval"
    assert candidate.time_seconds == 0.05

    orchestrator = RippingOrchestrator(
        window_extractor=AcousticWindowExtractor(
            search_radius_seconds=0.04,
        ),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        candidate_finder=_Finder(),
    )

    resolution = orchestrator.resolve_candidate(
        candidate=candidate,
        timeline=ingestor.timeline,
        ring_buffer=ingestor.ring_buffer,
    )

    assert resolution is not None
    assert resolution.candidate == candidate
    assert resolution.split.kind is SplitKind.HARD_CUT

    frame_offsets = {frame.offset for frame in ingestor.timeline.frames}

    assert resolution.split.incoming_start in frame_offsets
    assert resolution.split.outgoing_end in frame_offsets
    assert resolution.split.incoming_start == resolution.split.outgoing_end
