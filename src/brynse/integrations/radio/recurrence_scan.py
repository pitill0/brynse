"""List audio that recurs verbatim in a captured radio session.

Jingles, idents, sweepers and many adverts are played from the same file
again and again. ``brynse.recurrence`` finds every region that is heard more
than once; this tool runs it over a ``--session-capture`` bundle and relates
each occurrence to the nearest StreamTitle change, so clips that keep
appearing at transitions stand out as jingle/insert candidates.

This is an offline research tool: it does not change routing or output.
Use its findings, together with ``brynse.integrations.radio.evaluation``, to
decide whether recurrence should become a boundary provider.

    python -m brynse.integrations.radio.recurrence_scan run/session \
        --json run/recurrence.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from brynse.integrations.radio.icy import IcyStreamParser
from brynse.integrations.radio.replay import (
    ReplaySessionError,
    load_captured_events,
    load_replay_session,
)
from brynse.models import DecodedPcm
from brynse.recurrence import RecurrenceDetector, RecurringClip

_ANALYSIS_SAMPLE_RATE = 8000
_MAX_CAPTURE_BYTES = 256 * 1024 * 1024
_MAX_PCM_BYTES = 128 * 1024 * 1024
_PCM_LIMIT_GUARD_BYTES = 64 * 1024


@dataclass(frozen=True)
class AnnotatedOccurrence:
    start_seconds: float
    end_seconds: float
    similarity: float
    nearest_title: str | None
    distance_to_title_change_seconds: float | None


@dataclass(frozen=True)
class AnnotatedClip:
    clip_id: int
    duration_seconds: float
    occurrences: tuple[AnnotatedOccurrence, ...]
    at_transitions: int

    @property
    def transition_bound(self) -> bool:
        """Heard at least twice right next to a title change."""
        return self.at_transitions >= 2


def decode_session_audio(
    session_directory: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 600.0,
) -> DecodedPcm:
    """Strip ICY metadata from a capture and decode it to 8 kHz mono PCM."""
    session = load_replay_session(session_directory)
    if session.stream_path.stat().st_size > _MAX_CAPTURE_BYTES:
        raise ReplaySessionError("captured stream exceeds size limit")
    parser = IcyStreamParser(metaint=session.metaint)
    audio = parser.feed(session.stream_path.read_bytes()).audio
    if not audio:
        raise ReplaySessionError("captured session contains no audio")

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3" if session.codec == "mp3" else "aac",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_ANALYSIS_SAMPLE_RATE),
        "-f",
        "s16le",
        "-fs",
        str(_MAX_PCM_BYTES),
        "pipe:1",
    ]
    try:
        with tempfile.TemporaryFile(mode="w+b") as decoded:
            completed = subprocess.run(  # nosec B603
                command,
                input=audio,
                stdout=decoded,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
            output_bytes = decoded.seek(0, 2)
            if output_bytes >= _MAX_PCM_BYTES - _PCM_LIMIT_GUARD_BYTES:
                raise ReplaySessionError("decoded PCM reached size limit")
            if completed.returncode != 0 or output_bytes == 0:
                error = completed.stderr.decode("utf-8", errors="replace").strip()
                raise ReplaySessionError(f"FFmpeg session decode failed: {error}")
            decoded.seek(0)
            pcm_data = decoded.read(_MAX_PCM_BYTES + 1)
    except FileNotFoundError as exc:
        raise ReplaySessionError(f"FFmpeg binary not found: {ffmpeg_binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ReplaySessionError("FFmpeg session decode timed out") from exc

    return DecodedPcm(
        sample_rate=_ANALYSIS_SAMPLE_RATE,
        channels=1,
        sample_width_bytes=2,
        data=pcm_data,
    )


def annotate_clips(
    clips: Sequence[RecurringClip],
    title_changes: Sequence[tuple[float, str]],
    *,
    near_transition_seconds: float,
) -> tuple[AnnotatedClip, ...]:
    """Relate each occurrence to the nearest StreamTitle change."""
    if not math.isfinite(near_transition_seconds) or near_transition_seconds < 0:
        raise ValueError("near_transition_seconds must be finite and nonnegative")
    annotated = []
    for clip in clips:
        occurrences = []
        at_transitions = 0
        for occurrence in clip.occurrences:
            nearest_title = None
            distance = None
            if title_changes:
                time, title = min(
                    title_changes,
                    key=lambda change: _distance_to_interval(
                        change[0], occurrence.start_seconds, occurrence.end_seconds
                    ),
                )
                nearest_title = title
                distance = _distance_to_interval(
                    time, occurrence.start_seconds, occurrence.end_seconds
                )
                if distance <= near_transition_seconds:
                    at_transitions += 1
            occurrences.append(
                AnnotatedOccurrence(
                    start_seconds=occurrence.start_seconds,
                    end_seconds=occurrence.end_seconds,
                    similarity=occurrence.similarity,
                    nearest_title=nearest_title,
                    distance_to_title_change_seconds=distance,
                )
            )
        annotated.append(
            AnnotatedClip(
                clip_id=clip.clip_id,
                duration_seconds=clip.duration_seconds,
                occurrences=tuple(occurrences),
                at_transitions=at_transitions,
            )
        )
    annotated.sort(key=lambda clip: (not clip.transition_bound, -len(clip.occurrences)))
    return tuple(annotated)


def _distance_to_interval(time: float, start: float, end: float) -> float:
    if start <= time <= end:
        return 0.0
    return min(abs(time - start), abs(time - end))


def _title_changes(session_directory: Path) -> list[tuple[float, str]]:
    session = load_replay_session(session_directory)
    changes = []
    previous = None
    for record in load_captured_events(session):
        title = record.get("title")
        time = record.get("audio_time_seconds")
        if not isinstance(title, str) or not isinstance(time, int | float):
            continue
        if title != previous:
            changes.append((float(time), title))
            previous = title
    return changes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brynse-radio-recurrence-scan",
        description="List audio heard verbatim more than once in a captured session.",
    )
    parser.add_argument("session", type=Path, help="Captured session directory")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--json", type=Path, help="Write the full result as JSON")
    parser.add_argument(
        "--near-transition",
        type=float,
        default=20.0,
        help="Seconds from a title change that count as 'at a transition' (default: 20)",
    )
    return parser


def _format_time(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes):3d}:{rest:06.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        pcm = decode_session_audio(args.session, ffmpeg_binary=args.ffmpeg)
        clips = RecurrenceDetector().detect(pcm)
        annotated = annotate_clips(
            clips,
            _title_changes(args.session),
            near_transition_seconds=args.near_transition,
        )
    except (ReplaySessionError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not annotated:
        print("no recurring audio found")
    for clip in annotated:
        label = "AT TRANSITIONS" if clip.transition_bound else "elsewhere"
        print(
            f"clip {clip.clip_id}: {clip.duration_seconds:.2f} s, heard "
            f"{len(clip.occurrences)}x, {clip.at_transitions} at transitions [{label}]"
        )
        for occurrence in clip.occurrences:
            distance = occurrence.distance_to_title_change_seconds
            near = (
                "" if distance is None else f"  {distance:5.1f} s from '{occurrence.nearest_title}'"
            )
            print(
                f"    {_format_time(occurrence.start_seconds)} - "
                f"{_format_time(occurrence.end_seconds)}  sim={occurrence.similarity:.3f}{near}"
            )

    if args.json is not None:
        document = {
            "schema_version": 1,
            "session": str(args.session),
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "duration_seconds": clip.duration_seconds,
                    "transition_bound": clip.transition_bound,
                    "at_transitions": clip.at_transitions,
                    "occurrences": [vars(item) for item in clip.occurrences],
                }
                for clip in annotated
            ],
        }
        args.json.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
