"""Conservative MP3/AAC detection from clean encoded audio bytes."""

from __future__ import annotations

from collections.abc import Sequence

from brynse.frames import parse_adts_frames, parse_mp3_frames
from brynse.models import EncodedAudioFrame

_MIN_CONSECUTIVE_FRAMES = 3


def _has_consistent_frame_run(
    frames: Sequence[EncodedAudioFrame],
    *,
    minimum: int = _MIN_CONSECUTIVE_FRAMES,
) -> bool:
    """Require consecutive, complete frames with a stable sample rate."""
    run = 0
    previous: EncodedAudioFrame | None = None

    for frame in frames:
        if (
            previous is not None
            and frame.offset == previous.offset + previous.length
            and frame.sample_rate == previous.sample_rate
        ):
            run += 1
        else:
            run = 1

        if run >= minimum:
            return True

        previous = frame

    return False


def detect_audio_codec(audio: bytes) -> str | None:
    """Return a codec only when complete frame evidence is unambiguous.

    Input must contain clean encoded audio, with ICY metadata removed.
    Return None for insufficient, conflicting or unknown evidence.
    """
    mp3 = _has_consistent_frame_run(parse_mp3_frames(audio))
    aac = _has_consistent_frame_run(parse_adts_frames(audio))

    if mp3 == aac:
        return None

    return "mp3" if mp3 else "aac"
