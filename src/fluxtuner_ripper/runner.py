"""Reusable high-level stream ripping runner."""

from __future__ import annotations

import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fluxtuner_ripper.acoustic import AcousticWindowExtractor, FfmpegAcousticDecoder
from fluxtuner_ripper.live_shadow import LiveShadowBoundaryObserver
from fluxtuner_ripper.matching import NearestBoundaryMatcher
from fluxtuner_ripper.metadata import MetadataSemanticTracker
from fluxtuner_ripper.models import TimedMetadataEvent
from fluxtuner_ripper.mp3_refinement import Mp3BoundaryRefiner
from fluxtuner_ripper.orchestrator import (
    BoundaryResolver,
    HybridRippingOrchestrator,
    RippingOrchestrator,
)
from fluxtuner_ripper.ripping import RippingStreamIngestor
from fluxtuner_ripper.session import RippingSession
from fluxtuner_ripper.session_output import SessionOutputWriter, WrittenTrack
from fluxtuner_ripper.shadow import ShadowBoundaryAnalysis
from fluxtuner_ripper.source import BinaryIOStreamSource, StreamSource
from fluxtuner_ripper.streaming_spool import create_safe_streaming_spool
from fluxtuner_ripper.transient import ConservativeTransientExclusionPolicy

_CHUNK_SIZE = 64 * 1024
_DEFAULT_RING_MAX_BYTES = 16 * 1024 * 1024
_CONTENT_TYPE_CODECS = {
    "audio/aac": "aac",
    "audio/aacp": "aac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


class RippingRunError(RuntimeError):
    """Raised when a ripping run cannot be started or configured."""


@dataclass(frozen=True)
class RippingRunConfig:
    """Configuration for one ripping run."""

    url: str
    output_directory: Path
    codec: str = "auto"
    ffmpeg_binary: str = "ffmpeg"
    metadata_threshold_seconds: float = 8.0
    search_radius_seconds: float = 8.0
    transient_exclusion: bool = True
    ring_max_bytes: int = _DEFAULT_RING_MAX_BYTES


@dataclass(frozen=True)
class RippingRunResult:
    """Final observable state of one completed or stopped ripping run."""

    codec: str
    metaint: int
    stopped: bool
    incomplete_track_title: str | None


LegacyStream = tuple[BinaryIO, Mapping[str, str]]
StreamOpener = Callable[[str], StreamSource | LegacyStream]
MetadataCallback = Callable[[TimedMetadataEvent], None]
TrackCallback = Callable[[WrittenTrack], None]
StartedCallback = Callable[[str, int], None]
ShadowCallback = Callable[[ShadowBoundaryAnalysis], None]
ShadowObserverFactory = Callable[[str], LiveShadowBoundaryObserver]


def normalized_content_type(headers: Mapping[str, str]) -> str:
    raw = headers.get("Content-Type", "")
    return raw.split(";", 1)[0].strip().lower()


def resolve_codec(requested: str, headers: Mapping[str, str]) -> str:
    if requested != "auto":
        if requested not in {"mp3", "aac"}:
            raise RippingRunError("codec must be 'auto', 'mp3' or 'aac'")
        return requested

    content_type = normalized_content_type(headers)
    codec = _CONTENT_TYPE_CODECS.get(content_type)
    if codec is None:
        raise RippingRunError(
            "could not detect stream codec from Content-Type "
            f"{content_type or '<missing>'!r}; use codec='mp3' or codec='aac'"
        )
    return codec


def resolve_metaint(headers: Mapping[str, str]) -> int:
    raw = headers.get("icy-metaint")
    if raw is None:
        raise RippingRunError(
            "stream did not return icy-metaint; ICY metadata is required "
            "for the current ripping pipeline"
        )

    try:
        metaint = int(raw)
    except ValueError as exc:
        raise RippingRunError(f"invalid icy-metaint header: {raw!r}") from exc

    if metaint <= 0:
        raise RippingRunError(f"invalid icy-metaint header: {raw!r}")
    return metaint


def open_stream(url: str) -> tuple[BinaryIO, Mapping[str, str]]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RippingRunError("stream URL must use http or https")
    if not parsed.netloc:
        raise RippingRunError("stream URL must include a host")

    request = urllib.request.Request(
        url,
        headers={
            "Icy-MetaData": "1",
            "User-Agent": "FluxTuner-Ripper/0.1",
        },
    )

    try:
        response = urllib.request.urlopen(request, timeout=20)  # nosec B310
    except (urllib.error.URLError, ValueError) as exc:
        raise RippingRunError(f"could not open stream: {exc}") from exc

    return response, response.headers


def open_stream_source(url: str) -> StreamSource:
    """Open a radio stream as a StreamSource."""

    stream, headers = open_stream(url)
    return BinaryIOStreamSource(
        stream,
        metadata=headers,
    )


class RippingRunner:
    """Run the complete ripping pipeline with external stop support."""

    def __init__(
        self,
        config: RippingRunConfig,
        *,
        stream_opener: StreamOpener = open_stream_source,
        shadow_observer_factory: ShadowObserverFactory | None = None,
    ) -> None:
        self._config = config
        self._stream_opener = stream_opener
        self._shadow_observer_factory = (
            shadow_observer_factory
            if shadow_observer_factory is not None
            else lambda ffmpeg_binary: LiveShadowBoundaryObserver(ffmpeg_binary=ffmpeg_binary)
        )
        self._stop_event = threading.Event()
        self._stream: StreamSource | None = None
        self._session: RippingSession | None = None

    @property
    def current_track_title(self) -> str | None:
        session = self._session
        if session is None or session.current_track is None:
            return None
        return session.current_track.title

    def stop(self) -> None:
        """Request stop and close the active stream to unblock reads."""
        self._stop_event.set()
        stream = self._stream
        if stream is not None:
            with suppress(OSError):
                stream.close()

    def _validate_config(self) -> None:
        config = self._config
        if config.metadata_threshold_seconds <= 0:
            raise RippingRunError("metadata threshold must be greater than zero")
        if config.search_radius_seconds <= 0:
            raise RippingRunError("search radius must be greater than zero")
        if not config.ffmpeg_binary.strip():
            raise RippingRunError("ffmpeg binary must not be empty")
        if config.ring_max_bytes <= 0:
            raise RippingRunError("ring_max_bytes must be greater than zero")

    def _build_orchestrator(self, codec: str) -> BoundaryResolver:
        config = self._config
        extractor = AcousticWindowExtractor(
            search_radius_seconds=config.search_radius_seconds,
        )
        decoder = FfmpegAcousticDecoder(
            ffmpeg_binary=config.ffmpeg_binary,
        )

        if codec == "aac":
            return HybridRippingOrchestrator(
                window_extractor=extractor,
                decoder=decoder,
            )

        return RippingOrchestrator(
            window_extractor=extractor,
            decoder=decoder,
            matcher=NearestBoundaryMatcher(
                search_radius_seconds=config.search_radius_seconds,
            ),
            mp3_refiner=Mp3BoundaryRefiner(),
        )

    def run(
        self,
        *,
        on_started: StartedCallback | None = None,
        on_metadata: MetadataCallback | None = None,
        on_track_written: TrackCallback | None = None,
        on_shadow_analysis: ShadowCallback | None = None,
    ) -> RippingRunResult:
        """Run until EOF or stop() is requested."""
        self._validate_config()
        config = self._config
        config.output_directory.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()

        opened = self._stream_opener(config.url)

        if isinstance(opened, tuple):
            stream, headers = opened
            source: StreamSource = BinaryIOStreamSource(
                stream,
                metadata=headers,
            )
        else:
            source = opened

        self._stream = source
        spool = None

        try:
            codec = resolve_codec(config.codec, source.metadata)
            metaint = resolve_metaint(source.metadata)

            ingestor = RippingStreamIngestor(
                metaint=metaint,
                codec=codec,
                ring_max_bytes=config.ring_max_bytes,
            )
            transient_policy = (
                ConservativeTransientExclusionPolicy() if config.transient_exclusion else None
            )
            session = RippingSession(
                ingestor=ingestor,
                metadata_tracker=MetadataSemanticTracker(
                    transient_threshold_seconds=config.metadata_threshold_seconds,
                ),
                orchestrator=self._build_orchestrator(codec),
                transient_exclusion_policy=transient_policy,
            )
            self._session = session
            spool = create_safe_streaming_spool(
                directory=config.output_directory / ".fluxtuner-spool",
            )
            output_writer = SessionOutputWriter(
                ingestor=ingestor,
                directory=config.output_directory,
                codec=codec,
                source=spool,
            )
            shadow_observer = (
                self._shadow_observer_factory(config.ffmpeg_binary)
                if on_shadow_analysis is not None
                else None
            )

            if on_started is not None:
                on_started(codec, metaint)

            while not self._stop_event.is_set():
                try:
                    chunk = source.read(_CHUNK_SIZE)
                except Exception:
                    if self._stop_event.is_set():
                        break
                    raise

                if not chunk:
                    break

                result = session.feed(chunk)

                if result.ingest.audio:
                    spool.append(result.ingest.audio)

                if shadow_observer is not None and on_shadow_analysis is not None:
                    for analysis in shadow_observer.observe(
                        timeline=ingestor.timeline,
                        ring_buffer=ingestor.ring_buffer,
                    ):
                        on_shadow_analysis(analysis)

                if on_metadata is not None:
                    for event in result.ingest.timed_metadata_events:
                        on_metadata(event)

                for transition in result.transitions:
                    written = output_writer.write_transition(transition)

                    retained_start_offset = output_writer.retained_start_offset
                    if retained_start_offset is not None:
                        spool.discard_before(retained_start_offset)

                    if on_track_written is not None:
                        on_track_written(written)

            return RippingRunResult(
                codec=codec,
                metaint=metaint,
                stopped=self._stop_event.is_set(),
                incomplete_track_title=self.current_track_title,
            )
        finally:
            self._stream = None
            if spool is not None:
                spool.close()
            with suppress(OSError):
                source.close()
