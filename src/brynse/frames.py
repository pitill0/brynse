from __future__ import annotations

from brynse.models import EncodedAudioFrame

_MPEG1_LAYER3_BITRATES = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)

_MPEG2_LAYER3_BITRATES = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)

_MPEG_SAMPLE_RATES = {
    3: (44100, 48000, 32000),
    2: (22050, 24000, 16000),
    0: (11025, 12000, 8000),
}

_AAC_SAMPLE_RATES = (
    96000,
    88200,
    64000,
    48000,
    44100,
    32000,
    24000,
    22050,
    16000,
    12000,
    11025,
    8000,
    7350,
)


def _parse_mp3_header(data: bytes | bytearray, offset: int) -> tuple[int, int, int] | None:
    if offset + 4 > len(data):
        return None

    header = int.from_bytes(data[offset : offset + 4], "big")
    if header & 0xFFE00000 != 0xFFE00000:
        return None

    version_id = (header >> 19) & 0b11
    layer_id = (header >> 17) & 0b11
    bitrate_index = (header >> 12) & 0b1111
    sample_rate_index = (header >> 10) & 0b11
    padding = (header >> 9) & 0b1

    if version_id == 1 or layer_id != 1:
        return None
    if sample_rate_index == 3 or bitrate_index in (0, 15):
        return None

    sample_rate = _MPEG_SAMPLE_RATES[version_id][sample_rate_index]
    bitrate_table = _MPEG1_LAYER3_BITRATES if version_id == 3 else _MPEG2_LAYER3_BITRATES
    bitrate_kbps = bitrate_table[bitrate_index]
    if bitrate_kbps == 0:
        return None

    if version_id == 3:
        samples = 1152
        frame_length = (144000 * bitrate_kbps) // sample_rate + padding
    else:
        samples = 576
        frame_length = (72000 * bitrate_kbps) // sample_rate + padding

    if frame_length <= 4:
        return None
    return frame_length, sample_rate, samples


def parse_mp3_frames(data: bytes) -> list[EncodedAudioFrame]:
    """Parse MPEG Layer III frames and build an exact sample timeline."""

    frames: list[EncodedAudioFrame] = []
    offset = 0
    total_samples = 0

    while offset + 4 <= len(data):
        parsed = _parse_mp3_header(data, offset)
        if parsed is None:
            offset += 1
            continue

        frame_length, sample_rate, samples = parsed
        if offset + frame_length > len(data):
            break

        frames.append(
            EncodedAudioFrame(
                codec="mp3",
                offset=offset,
                length=frame_length,
                sample_rate=sample_rate,
                samples=samples,
                time_seconds=total_samples / sample_rate,
            )
        )
        total_samples += samples
        offset += frame_length

    return frames


def _parse_adts_header(data: bytes | bytearray, offset: int) -> tuple[int, int, int] | None:
    if offset + 7 > len(data):
        return None

    b0, b1, b2, b3, b4, b5, b6 = data[offset : offset + 7]
    if b0 != 0xFF or b1 & 0xF6 != 0xF0:
        return None

    sample_rate_index = (b2 >> 2) & 0x0F
    if sample_rate_index >= len(_AAC_SAMPLE_RATES):
        return None

    sample_rate = _AAC_SAMPLE_RATES[sample_rate_index]
    frame_length = ((b3 & 0x03) << 11) | (b4 << 3) | ((b5 >> 5) & 0x07)
    raw_blocks = b6 & 0x03
    samples = 1024 * (raw_blocks + 1)

    header_length = 7 if b1 & 0x01 else 9
    if frame_length < header_length:
        return None

    return frame_length, sample_rate, samples


def parse_adts_frames(data: bytes) -> list[EncodedAudioFrame]:
    """Parse AAC/ADTS frames and build an exact sample timeline."""

    frames: list[EncodedAudioFrame] = []
    offset = 0
    total_samples = 0

    while offset + 7 <= len(data):
        parsed = _parse_adts_header(data, offset)
        if parsed is None:
            offset += 1
            continue

        frame_length, sample_rate, samples = parsed
        if offset + frame_length > len(data):
            break

        frames.append(
            EncodedAudioFrame(
                codec="aac",
                offset=offset,
                length=frame_length,
                sample_rate=sample_rate,
                samples=samples,
                time_seconds=total_samples / sample_rate,
            )
        )
        total_samples += samples
        offset += frame_length

    return frames


def frame_at_or_before_offset(frames: list[EncodedAudioFrame], offset: int) -> EncodedAudioFrame:
    """Return the last frame whose start offset is not greater than offset."""

    if not frames:
        raise ValueError("frames must not be empty")
    if offset < frames[0].offset:
        return frames[0]

    candidate = frames[0]
    for frame in frames[1:]:
        if frame.offset > offset:
            break
        candidate = frame
    return candidate


def frame_nearest_time(frames: list[EncodedAudioFrame], time_seconds: float) -> EncodedAudioFrame:
    """Return the frame start closest to an exact audio timestamp."""

    if not frames:
        raise ValueError("frames must not be empty")
    if time_seconds < 0:
        raise ValueError("time_seconds must be non-negative")

    return min(frames, key=lambda frame: abs(frame.time_seconds - time_seconds))


class IncrementalFrameTimeline:
    """Incrementally frame encoded audio while preserving absolute offsets/time."""

    def __init__(self, codec: str) -> None:
        normalized = codec.strip().lower()
        if normalized not in {"mp3", "aac"}:
            raise ValueError("codec must be 'mp3' or 'aac'")

        self._codec = normalized
        self._pending = bytearray()
        self._pending_offset = 0
        self._next_input_offset = 0
        self._elapsed_seconds = 0.0
        self._frames: list[EncodedAudioFrame] = []

    @property
    def codec(self) -> str:
        return self._codec

    @property
    def frames(self) -> tuple[EncodedAudioFrame, ...]:
        return tuple(self._frames)

    def feed(self, data: bytes) -> tuple[EncodedAudioFrame, ...]:
        """Consume clean encoded bytes and return newly completed frames."""

        if not data:
            return ()

        if not self._pending:
            self._pending_offset = self._next_input_offset

        self._pending.extend(data)
        self._next_input_offset += len(data)

        new_frames: list[EncodedAudioFrame] = []
        local = 0

        while True:
            if self._codec == "mp3":
                if local + 4 > len(self._pending):
                    break
                parsed = _parse_mp3_header(self._pending, local)
            else:
                if local + 7 > len(self._pending):
                    break
                parsed = _parse_adts_header(self._pending, local)

            if parsed is None:
                local += 1
                continue

            frame_length, sample_rate, samples = parsed
            if local + frame_length > len(self._pending):
                break

            frame = EncodedAudioFrame(
                codec=self._codec,
                offset=self._pending_offset + local,
                length=frame_length,
                sample_rate=sample_rate,
                samples=samples,
                time_seconds=self._elapsed_seconds,
            )
            self._frames.append(frame)
            new_frames.append(frame)
            self._elapsed_seconds += samples / sample_rate
            local += frame_length

        if local:
            del self._pending[:local]
            self._pending_offset += local

        return tuple(new_frames)

    def frame_for_audio_offset(self, offset: int) -> EncodedAudioFrame | None:
        """Resolve an audio byte offset only when its frame is known."""

        for frame in self._frames:
            frame_end = frame.offset + frame.length
            if frame.offset <= offset < frame_end:
                return frame
            if frame.offset > offset:
                break
        return None

    def discard_before(self, offset: int) -> None:
        """Discard completed frame records that cannot overlap ``offset``."""

        keep_from = 0
        for index, frame in enumerate(self._frames):
            if frame.offset + frame.length > offset:
                keep_from = index
                break
        else:
            self._frames.clear()
            return

        if keep_from:
            del self._frames[:keep_from]
