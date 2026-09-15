"""Source-agnostic finite-stream runner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.orchestrator import CandidateResolution, CandidateResolver
from fluxtuner_ripper.providers import BoundaryProvider


@dataclass(frozen=True)
class GenericRunResult:
    """Result of ingesting and resolving one finite encoded stream."""

    bytes_ingested: int
    resolutions: tuple[CandidateResolution, ...]


class GenericRunner:
    """Ingest a finite encoded stream and resolve provider boundaries."""

    def __init__(
        self,
        *,
        ingestor: EncodedStreamIngestor,
        provider: BoundaryProvider,
        resolver: CandidateResolver,
    ) -> None:
        self._ingestor = ingestor
        self._provider = provider
        self._resolver = resolver

    def run(self, chunks: Iterable[bytes]) -> GenericRunResult:
        """Consume all chunks, then resolve boundaries across the retained timeline."""

        bytes_ingested = 0

        for chunk in chunks:
            self._ingestor.feed(chunk)
            bytes_ingested += len(chunk)

        frames = self._ingestor.timeline.frames
        if not frames:
            return GenericRunResult(
                bytes_ingested=bytes_ingested,
                resolutions=(),
            )

        last_frame = frames[-1]
        end_time_seconds = last_frame.time_seconds + last_frame.samples / last_frame.sample_rate

        candidates = self._provider.propose(
            start_time_seconds=0.0,
            end_time_seconds=end_time_seconds,
        )

        resolutions: list[CandidateResolution] = []

        for candidate in candidates:
            resolution = self._resolver.resolve_candidate(
                candidate=candidate,
                timeline=self._ingestor.timeline,
                ring_buffer=self._ingestor.ring_buffer,
            )
            if resolution is not None:
                resolutions.append(resolution)

        return GenericRunResult(
            bytes_ingested=bytes_ingested,
            resolutions=tuple(resolutions),
        )
