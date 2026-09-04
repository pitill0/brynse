from __future__ import annotations

from pathlib import Path

from fluxtuner_ripper.acoustic import (
    AcousticCandidateFinder as AcousticCandidateFinder,
)
from fluxtuner_ripper.acoustic import (
    AcousticDecodeError as AcousticDecodeError,
)
from fluxtuner_ripper.acoustic import (
    AcousticWindowExtractor as AcousticWindowExtractor,
)
from fluxtuner_ripper.acoustic import (
    FfmpegAcousticDecoder as FfmpegAcousticDecoder,
)
from fluxtuner_ripper.acoustic import (
    RmsAcousticAnalyzer as RmsAcousticAnalyzer,
)

# Compatibility re-exports preserved during staged extraction.
from fluxtuner_ripper.buffer import EncodedAudioRingBuffer as EncodedAudioRingBuffer
from fluxtuner_ripper.frames import (
    IncrementalFrameTimeline as IncrementalFrameTimeline,
)
from fluxtuner_ripper.frames import (
    frame_at_or_before_offset as frame_at_or_before_offset,
)
from fluxtuner_ripper.frames import (
    frame_nearest_time as frame_nearest_time,
)
from fluxtuner_ripper.frames import (
    parse_adts_frames as parse_adts_frames,
)
from fluxtuner_ripper.frames import (
    parse_mp3_frames as parse_mp3_frames,
)
from fluxtuner_ripper.icy import IcyStreamParser as IcyStreamParser
from fluxtuner_ripper.matching import (
    BoundaryRelationClassifier as BoundaryRelationClassifier,
)
from fluxtuner_ripper.matching import (
    NearestBoundaryMatcher as NearestBoundaryMatcher,
)
from fluxtuner_ripper.matching import (
    SplitAlignmentError as SplitAlignmentError,
)
from fluxtuner_ripper.matching import (
    TemporalSplitAligner as TemporalSplitAligner,
)
from fluxtuner_ripper.matching import (
    TemporalSplitPolicy as TemporalSplitPolicy,
)
from fluxtuner_ripper.metadata import (
    MetadataSemanticTracker as MetadataSemanticTracker,
)
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate as AcousticBoundaryCandidate,
)
from fluxtuner_ripper.models import (
    AcousticLevel as AcousticLevel,
)
from fluxtuner_ripper.models import (
    AcousticProfile as AcousticProfile,
)
from fluxtuner_ripper.models import (
    AcousticWindow as AcousticWindow,
)
from fluxtuner_ripper.models import (
    BoundaryMatch as BoundaryMatch,
)
from fluxtuner_ripper.models import (
    BoundaryRelation as BoundaryRelation,
)
from fluxtuner_ripper.models import (
    BoundaryRelationResult as BoundaryRelationResult,
)
from fluxtuner_ripper.models import (
    ContentKind as ContentKind,
)
from fluxtuner_ripper.models import (
    DecodedPcm as DecodedPcm,
)
from fluxtuner_ripper.models import (
    EncodedAudioFrame as EncodedAudioFrame,
)
from fluxtuner_ripper.models import (
    IcyParseResult as IcyParseResult,
)
from fluxtuner_ripper.models import (
    MetadataEvent as MetadataEvent,
)
from fluxtuner_ripper.models import (
    MetadataSemanticDecision as MetadataSemanticDecision,
)
from fluxtuner_ripper.models import (
    RippingIngestResult as RippingIngestResult,
)
from fluxtuner_ripper.models import (
    SplitDecision as SplitDecision,
)
from fluxtuner_ripper.models import (
    SplitKind as SplitKind,
)
from fluxtuner_ripper.models import (
    TemporalSplitDecision as TemporalSplitDecision,
)
from fluxtuner_ripper.models import (
    TemporalSplitKind as TemporalSplitKind,
)
from fluxtuner_ripper.models import (
    TimedMetadataEvent as TimedMetadataEvent,
)
from fluxtuner_ripper.models import (
    TrackByteRange as TrackByteRange,
)
from fluxtuner_ripper.models import (
    TrackCandidate as TrackCandidate,
)
from fluxtuner_ripper.models import (
    TrackWritePlan as TrackWritePlan,
)


class RippingStreamIngestor:
    """Join ICY parsing, encoded buffering, and exact frame-timeline resolution."""

    def __init__(self, *, metaint: int, codec: str, ring_max_bytes: int) -> None:
        self._icy = IcyStreamParser(metaint)
        self._ring = EncodedAudioRingBuffer(ring_max_bytes)
        self._timeline = IncrementalFrameTimeline(codec)
        self._pending_metadata: list[MetadataEvent] = []

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer:
        return self._ring

    @property
    def timeline(self) -> IncrementalFrameTimeline:
        return self._timeline

    def feed(self, chunk: bytes) -> RippingIngestResult:
        """Consume one raw ICY chunk and resolve new metadata onto audio time."""

        parsed = self._icy.feed(chunk)

        if parsed.audio:
            self._ring.append(parsed.audio)
            self._timeline.feed(parsed.audio)

        self._pending_metadata.extend(parsed.events)

        timed: list[TimedMetadataEvent] = []
        still_pending: list[MetadataEvent] = []
        for event in self._pending_metadata:
            frame = self._timeline.frame_for_audio_offset(event.audio_offset)
            if frame is None:
                still_pending.append(event)
                continue

            timed.append(
                TimedMetadataEvent(
                    title=event.title,
                    audio_offset=event.audio_offset,
                    audio_time_seconds=frame.time_seconds,
                )
            )

        self._pending_metadata = still_pending
        self._timeline.discard_before(self._ring.start_offset)

        return RippingIngestResult(
            audio=parsed.audio,
            metadata_events=parsed.events,
            timed_metadata_events=tuple(timed),
        )


class TrackRangePlanner:
    """Build overlapping encoded-byte ranges from one split decision."""

    def plan(
        self,
        *,
        previous_start_offset: int,
        next_end_offset: int,
        decision: SplitDecision,
    ) -> TrackWritePlan:
        if previous_start_offset < 0:
            raise ValueError("previous_start_offset must be non-negative")
        if next_end_offset <= previous_start_offset:
            raise ValueError("next_end_offset must be greater than previous_start_offset")

        if decision.kind is SplitKind.NO_BOUNDARY:
            raise ValueError("NO_BOUNDARY cannot produce a track write plan")

        if decision.incoming_start is None or decision.outgoing_end is None:
            raise ValueError("split decision requires concrete offsets")

        if not (previous_start_offset < decision.outgoing_end <= next_end_offset):
            raise ValueError("outgoing_end lies outside the writable range")

        if not (previous_start_offset <= decision.incoming_start < next_end_offset):
            raise ValueError("incoming_start lies outside the writable range")

        return TrackWritePlan(
            outgoing=TrackByteRange(
                start_offset=previous_start_offset,
                end_offset=decision.outgoing_end,
            ),
            incoming=TrackByteRange(
                start_offset=decision.incoming_start,
                end_offset=next_end_offset,
            ),
        )


class EncodedTrackWriter:
    """Materialize frame-aligned encoded ranges without transcoding."""

    def write_range(
        self,
        *,
        source: EncodedAudioRingBuffer,
        byte_range: TrackByteRange,
    ) -> bytes:
        if not source.contains(byte_range.start_offset, byte_range.end_offset):
            raise ValueError("requested track range is not fully retained")
        return source.read(byte_range.start_offset, byte_range.end_offset)


class TrackFileWriteError(RuntimeError):
    """Raised when an encoded track cannot be persisted safely."""


class TrackFileWriter:
    """Persist encoded track bytes atomically without transcoding."""

    _EXTENSIONS = {
        "mp3": ".mp3",
        "aac": ".aac",
    }

    def extension_for_codec(self, codec: str) -> str:
        normalized = codec.strip().lower()
        try:
            return self._EXTENSIONS[normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported codec for track file: {codec}") from exc

    def write(
        self,
        *,
        directory: Path,
        stem: str,
        codec: str,
        data: bytes,
    ) -> Path:
        import os
        import tempfile

        if not stem or not stem.strip():
            raise ValueError("stem must not be empty")
        if not data:
            raise ValueError("data must not be empty")

        extension = self.extension_for_codec(codec)
        directory.mkdir(parents=True, exist_ok=True)

        target = directory / f"{stem}{extension}"
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, target)
            return target
        except OSError as exc:
            if temp_path is not None:
                from contextlib import suppress

                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise TrackFileWriteError(f"failed to write track file: {target}") from exc


class TrackFinalizeError(RuntimeError):
    """Raised when an encoded track cannot be finalized safely."""


class Mp3TrackFinalizer:
    """Remux MP3 bytes without transcoding to rebuild seek metadata."""

    def __init__(self, *, ffmpeg_binary: str = "ffmpeg") -> None:
        self._ffmpeg_binary = ffmpeg_binary

    def finalize(self, data: bytes) -> bytes:
        """Remux MP3 through FFmpeg stream-copy and emit a fresh Xing header."""

        # FFmpeg is invoked directly without a shell.
        import subprocess  # nosec B404

        if not data:
            raise ValueError("data must not be empty")

        command = [
            self._ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-write_xing",
            "1",
            "-f",
            "mp3",
            "pipe:1",
        ]

        try:
            # Fixed argv list; subprocess uses shell=False.
            completed = subprocess.run(  # nosec B603
                command,
                input=data,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise TrackFinalizeError(f"FFmpeg binary not found: {self._ffmpeg_binary}") from exc

        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise TrackFinalizeError(
                f"FFmpeg MP3 finalization failed with exit code {completed.returncode}: {error}"
            )

        if not completed.stdout:
            raise TrackFinalizeError("FFmpeg MP3 finalization produced no output")

        return completed.stdout


class AacTrackFinalizer:
    """Remux AAC/ADTS bytes to M4A without transcoding."""

    def __init__(self, *, ffmpeg_binary: str = "ffmpeg") -> None:
        self._ffmpeg_binary = ffmpeg_binary

    def finalize(self, data: bytes) -> bytes:
        """Remux raw ADTS AAC into an M4A/MP4 container using stream-copy."""

        # FFmpeg is invoked directly without a shell.
        import subprocess  # nosec B404

        if not data:
            raise ValueError("data must not be empty")

        command = [
            self._ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "aac",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov",
            "pipe:1",
        ]

        try:
            # Fixed argv list; subprocess uses shell=False.
            completed = subprocess.run(  # nosec B603
                command,
                input=data,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise TrackFinalizeError(f"FFmpeg binary not found: {self._ffmpeg_binary}") from exc

        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise TrackFinalizeError(
                f"FFmpeg AAC finalization failed with exit code {completed.returncode}: {error}"
            )

        if not completed.stdout:
            raise TrackFinalizeError("FFmpeg AAC finalization produced no output")

        return completed.stdout


class TrackOutputService:
    """Compose range extraction, codec finalization, and atomic persistence."""

    def __init__(
        self,
        *,
        mp3_finalizer: Mp3TrackFinalizer | None = None,
        aac_finalizer: AacTrackFinalizer | None = None,
        file_writer: TrackFileWriter | None = None,
    ) -> None:
        self._mp3_finalizer = mp3_finalizer or Mp3TrackFinalizer()
        self._aac_finalizer = aac_finalizer or AacTrackFinalizer()
        self._file_writer = file_writer or TrackFileWriter()
        self._encoded_writer = EncodedTrackWriter()

    def write_track(
        self,
        *,
        source: EncodedAudioRingBuffer,
        byte_range: TrackByteRange,
        directory: Path,
        stem: str,
        codec: str,
    ) -> Path:
        raw = self._encoded_writer.write_range(
            source=source,
            byte_range=byte_range,
        )

        normalized = codec.strip().lower()
        if normalized == "mp3":
            finalized = self._mp3_finalizer.finalize(raw)
            output_codec = "mp3"
        elif normalized == "aac":
            finalized = self._aac_finalizer.finalize(raw)
            output_codec = "m4a"
        else:
            raise ValueError(f"unsupported codec for track output: {codec}")

        return self._write_finalized(
            directory=directory,
            stem=stem,
            output_codec=output_codec,
            data=finalized,
        )

    def _write_finalized(
        self,
        *,
        directory: Path,
        stem: str,
        output_codec: str,
        data: bytes,
    ) -> Path:
        if output_codec == "m4a":
            return self._write_m4a(
                directory=directory,
                stem=stem,
                data=data,
            )

        return self._file_writer.write(
            directory=directory,
            stem=stem,
            codec=output_codec,
            data=data,
        )

    def _write_m4a(
        self,
        *,
        directory: Path,
        stem: str,
        data: bytes,
    ) -> Path:
        import os
        import tempfile
        from contextlib import suppress

        if not stem or not stem.strip():
            raise ValueError("stem must not be empty")
        if not data:
            raise ValueError("data must not be empty")

        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{stem}.m4a"
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, target)
            return target
        except OSError as exc:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise TrackFileWriteError(f"failed to write track file: {target}") from exc
