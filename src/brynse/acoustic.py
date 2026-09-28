from __future__ import annotations

from array import array

from brynse.buffer import EncodedAudioRingBuffer
from brynse.frames import IncrementalFrameTimeline
from brynse.models import (
    AcousticBoundaryCandidate,
    AcousticLevel,
    AcousticProfile,
    AcousticWindow,
    DecodedPcm,
)


class AcousticWindowExtractor:
    """Extract bounded, frame-aligned encoded windows from the retained ring."""

    def __init__(self, search_radius_seconds: float = 8.0) -> None:
        if search_radius_seconds <= 0:
            raise ValueError("search_radius_seconds must be greater than zero")
        self._search_radius = search_radius_seconds

    @property
    def search_radius_seconds(self) -> float:
        return self._search_radius

    def extract(
        self,
        *,
        candidate_time_seconds: float,
        timeline: IncrementalFrameTimeline,
        ring_buffer: EncodedAudioRingBuffer,
    ) -> AcousticWindow | None:
        """Return the retained frame-aligned window around a candidate time."""

        frames = timeline.frames
        if not frames:
            return None

        target_start = max(0.0, candidate_time_seconds - self._search_radius)
        target_end = candidate_time_seconds + self._search_radius

        retained = [
            frame
            for frame in frames
            if frame.offset + frame.length > ring_buffer.start_offset
            and frame.offset < ring_buffer.end_offset
        ]
        if not retained:
            return None

        start_frame = min(
            retained,
            key=lambda frame: abs(frame.time_seconds - target_start),
        )

        eligible_end_frames = [frame for frame in retained if frame.time_seconds <= target_end]
        end_frame = eligible_end_frames[-1] if eligible_end_frames else retained[0]

        start_offset = max(start_frame.offset, ring_buffer.start_offset)
        end_offset = min(
            end_frame.offset + end_frame.length,
            ring_buffer.end_offset,
        )

        if start_offset >= end_offset:
            return None

        start_time = start_frame.time_seconds
        end_time = end_frame.time_seconds + end_frame.samples / end_frame.sample_rate

        data = ring_buffer.read(start_offset, end_offset)

        return AcousticWindow(
            start_offset=start_offset,
            end_offset=end_offset,
            start_time_seconds=start_time,
            end_time_seconds=end_time,
            data=data,
        )


class AcousticDecodeError(RuntimeError):
    """Raised when bounded encoded audio cannot be decoded for analysis."""


class FfmpegAcousticDecoder:
    """Decode one bounded encoded window to mono 16-bit PCM."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        output_sample_rate: int = 8000,
    ) -> None:
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be greater than zero")
        self._ffmpeg_binary = ffmpeg_binary
        self._output_sample_rate = output_sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    def decode(self, window: AcousticWindow) -> DecodedPcm:
        """Decode only ``window.data`` through FFmpeg stdin/stdout."""

        # FFmpeg is invoked directly without a shell.
        import subprocess  # nosec B404

        command = [
            self._ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self._output_sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]

        try:
            # Fixed argv list; subprocess uses shell=False.
            completed = subprocess.run(  # nosec B603
                command,
                input=window.data,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AcousticDecodeError(f"FFmpeg binary not found: {self._ffmpeg_binary}") from exc

        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise AcousticDecodeError(
                f"FFmpeg decode failed with exit code {completed.returncode}: {error}"
            )

        if not completed.stdout:
            raise AcousticDecodeError("FFmpeg decode produced no PCM output")

        return DecodedPcm(
            sample_rate=self._output_sample_rate,
            channels=1,
            sample_width_bytes=2,
            data=completed.stdout,
        )


class RmsAcousticAnalyzer:
    """Compute RMS energy over fixed-size windows of mono 16-bit PCM."""

    def __init__(self, window_seconds: float = 0.5) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        self._window_seconds = window_seconds

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    def analyze(self, pcm: DecodedPcm) -> AcousticProfile:
        import math

        samples = array("h")
        samples.frombytes(pcm.data)

        samples_per_window = max(1, round(pcm.sample_rate * self._window_seconds))
        levels: list[AcousticLevel] = []

        for start in range(0, len(samples), samples_per_window):
            chunk = samples[start : start + samples_per_window]
            if not chunk:
                continue

            square_sum = sum(sample * sample for sample in chunk)
            rms = math.sqrt(square_sum / len(chunk))

            start_time = start / pcm.sample_rate
            end_time = (start + len(chunk)) / pcm.sample_rate
            levels.append(
                AcousticLevel(
                    start_time_seconds=start_time,
                    end_time_seconds=end_time,
                    rms=rms,
                )
            )

        return AcousticProfile(levels=tuple(levels))


class AcousticCandidateFinder:
    """Convert an RMS profile into local-minimum boundary candidates."""

    def find(
        self,
        *,
        profile: AcousticProfile,
        window: AcousticWindow,
        center_time_seconds: float | None = None,
        radius_seconds: float | None = None,
    ) -> tuple[AcousticBoundaryCandidate, ...]:
        if radius_seconds is not None and radius_seconds <= 0:
            raise ValueError("radius_seconds must be greater than zero")
        if center_time_seconds is None and radius_seconds is not None:
            raise ValueError("radius_seconds requires center_time_seconds")
        if center_time_seconds is not None and center_time_seconds < 0:
            raise ValueError("center_time_seconds must be non-negative")

        levels = profile.levels
        if not levels:
            return ()

        candidates: list[AcousticBoundaryCandidate] = []

        for index, level in enumerate(levels):
            left = levels[index - 1].rms if index > 0 else None
            right = levels[index + 1].rms if index + 1 < len(levels) else None

            is_local_minimum = True
            if left is not None and level.rms > left:
                is_local_minimum = False
            if right is not None and level.rms > right:
                is_local_minimum = False

            if not is_local_minimum:
                continue

            relative_time = (level.start_time_seconds + level.end_time_seconds) / 2.0
            absolute_time = window.start_time_seconds + relative_time

            if (
                center_time_seconds is not None
                and radius_seconds is not None
                and abs(absolute_time - center_time_seconds) > radius_seconds
            ):
                continue

            candidates.append(
                AcousticBoundaryCandidate(
                    time_seconds=absolute_time,
                    rms=level.rms,
                    relative_time_seconds=relative_time,
                )
            )

        candidates.sort(key=lambda candidate: candidate.time_seconds)
        return tuple(candidates)
