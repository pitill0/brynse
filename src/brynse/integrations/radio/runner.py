"""Reusable high-level stream ripping runner."""

from __future__ import annotations

import http.client
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from brynse.acoustic import AcousticWindowExtractor, FfmpegAcousticDecoder
from brynse.autonomous_promotion import AutonomousBoundaryPromoter
from brynse.incoming_refinement import build_incoming_boundary_pipeline
from brynse.integrations.radio.autonomous import AutonomousSegmentState
from brynse.integrations.radio.metadata import MetadataSemanticTracker
from brynse.integrations.radio.models import TimedMetadataEvent
from brynse.integrations.radio.orchestrator import (
    BoundaryResolver,
    CandidateBoundaryResolver,
)
from brynse.integrations.radio.ripping import RippingStreamIngestor
from brynse.integrations.radio.session import RippingSession
from brynse.integrations.radio.session_output import (
    SessionOutputWriter,
    WrittenSegment,
    WrittenTrack,
)
from brynse.integrations.radio.transient import ConservativeTransientExclusionPolicy
from brynse.live_shadow import LiveShadowBoundaryObserver
from brynse.models import Segment
from brynse.orchestrator import MultiSignalAcousticCandidateResolver
from brynse.shadow import ShadowBoundaryAnalysis
from brynse.source import BinaryIOStreamSource, StreamSource
from brynse.streaming_spool import create_safe_streaming_spool

_CHUNK_SIZE = 64 * 1024
_DEFAULT_RING_MAX_BYTES = 16 * 1024 * 1024
_CONTENT_TYPE_CODECS = {
    "audio/aac": "aac",
    "audio/aacp": "aac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}

_MULTISIGNAL_ACOUSTIC_RADIUS_SECONDS = 24.0
_AUTONOMOUS_MIN_OPEN_SECONDS = 8.0


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
    transient_exclusion: bool = True
    ring_max_bytes: int = _DEFAULT_RING_MAX_BYTES
    autonomous_boundaries: bool = False


@dataclass(frozen=True)
class RippingRunResult:
    """Final observable state of one completed or stopped ripping run."""

    codec: str
    metaint: int
    stopped: bool
    incomplete_track_title: str | None


StreamOpener = Callable[[str], StreamSource]
MetadataCallback = Callable[[TimedMetadataEvent], None]
TrackCallback = Callable[[WrittenTrack], None]
SegmentCallback = Callable[[WrittenSegment], None]
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
            "User-Agent": "Mozilla/5.0",
        },
    )

    try:
        response = urllib.request.urlopen(request, timeout=20)  # nosec B310
    except (http.client.RemoteDisconnected, urllib.error.URLError, ValueError) as exc:
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
        if not config.ffmpeg_binary.strip():
            raise RippingRunError("ffmpeg binary must not be empty")
        if config.ring_max_bytes <= 0:
            raise RippingRunError("ring_max_bytes must be greater than zero")

    def _build_orchestrator(
        self,
        codec: str,
        *,
        on_incoming_observation=None,
    ) -> BoundaryResolver:
        if codec not in {"aac", "mp3"}:
            raise RippingRunError(f"unsupported codec: {codec!r}")

        config = self._config

        return CandidateBoundaryResolver(
            MultiSignalAcousticCandidateResolver(
                window_extractor=AcousticWindowExtractor(
                    search_radius_seconds=_MULTISIGNAL_ACOUSTIC_RADIUS_SECONDS,
                ),
                decoder8=FfmpegAcousticDecoder(
                    ffmpeg_binary=config.ffmpeg_binary,
                    output_sample_rate=8000,
                ),
                decoder16=FfmpegAcousticDecoder(
                    ffmpeg_binary=config.ffmpeg_binary,
                    output_sample_rate=16000,
                ),
                incoming_refiner=build_incoming_boundary_pipeline(
                    ffmpeg_binary=config.ffmpeg_binary,
                    on_observation=on_incoming_observation,
                ),
            )
        )

    def run(
        self,
        *,
        on_started: StartedCallback | None = None,
        on_metadata: MetadataCallback | None = None,
        on_track_written: TrackCallback | None = None,
        on_segment_written: SegmentCallback | None = None,
        on_shadow_analysis: ShadowCallback | None = None,
        on_incoming_observation=None,
    ) -> RippingRunResult:
        """Run until EOF or stop() is requested."""
        self._validate_config()
        config = self._config
        config.output_directory.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()

        source = self._stream_opener(config.url)
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
                orchestrator=self._build_orchestrator(
                    codec,
                    on_incoming_observation=on_incoming_observation,
                ),
                transient_exclusion_policy=transient_policy,
                acoustic_settle_seconds=_MULTISIGNAL_ACOUSTIC_RADIUS_SECONDS,
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
            autonomous_promoter = (
                AutonomousBoundaryPromoter()
                if config.autonomous_boundaries
                else None
            )
            autonomous_state = (
                AutonomousSegmentState()
                if config.autonomous_boundaries
                else None
            )
            shadow_enabled = (
                on_shadow_analysis is not None
                or config.autonomous_boundaries
            )
            shadow_observer = (
                self._shadow_observer_factory(config.ffmpeg_binary)
                if shadow_enabled
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

                if on_metadata is not None:
                    for event in result.ingest.timed_metadata_events:
                        on_metadata(event)

                for transition in result.transitions:
                    written = output_writer.write_transition(transition)

                    materialized_start = output_writer.retained_start_offset
                    if (
                        autonomous_state is not None
                        and materialized_start is not None
                    ):
                        autonomous_state.observe_materialized_segment(
                            Segment(
                                start_offset=materialized_start,
                                start_time_seconds=(
                                    transition.boundary.temporal.incoming_start_seconds
                                ),
                                label=transition.incoming.title,
                            )
                        )


                    retained_start_offset = output_writer.retained_start_offset
                    if retained_start_offset is not None:
                        spool.discard_before(retained_start_offset)

                    if on_track_written is not None:
                        on_track_written(written)

                if shadow_observer is not None:
                    for analysis in shadow_observer.observe(
                        timeline=ingestor.timeline,
                        ring_buffer=ingestor.ring_buffer,
                    ):
                        if on_shadow_analysis is not None:
                            on_shadow_analysis(analysis)

                        if autonomous_promoter is None or autonomous_state is None:
                            continue

                        current_track = session.current_track
                        if current_track is not None:
                            autonomous_state.observe_known_segment(
                                current_track.as_segment()
                            )

                        resolution = autonomous_promoter.promote(
                            analysis=analysis,
                            timeline=ingestor.timeline,
                        )
                        if resolution is None:
                            continue

                        incoming_start = resolution.split.incoming_start
                        if (
                            incoming_start is None
                            or not output_writer.can_write_boundary_at(incoming_start)
                        ):
                            continue

                        transition = autonomous_state.transition(
                            resolution,
                            minimum_open_seconds=_AUTONOMOUS_MIN_OPEN_SECONDS,
                        )
                        if transition is None:
                            continue

                        written_segment = output_writer.write_segment_transition(
                            transition
                        )

                        retained_start_offset = output_writer.retained_start_offset
                        if retained_start_offset is not None:
                            spool.discard_before(retained_start_offset)

                        if on_segment_written is not None:
                            on_segment_written(written_segment)

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
