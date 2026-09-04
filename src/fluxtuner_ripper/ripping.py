from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ContentKind(StrEnum):
    """Semantic content classification for one stream interval."""

    MUSIC = "music"
    ADVERTISEMENT = "advertisement"
    JINGLE = "jingle"
    STATION_ID = "station_id"
    TALK = "talk"
    UNKNOWN = "unknown"


class SplitKind(StrEnum):
    """Relationship between consecutive track candidates."""

    NO_BOUNDARY = "no_boundary"
    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"


@dataclass(frozen=True)
class MetadataEvent:
    """One metadata observation anchored to an absolute encoded-byte offset."""

    title: str
    audio_offset: int

    def __post_init__(self) -> None:
        if self.audio_offset < 0:
            raise ValueError("audio_offset must be non-negative")


@dataclass(frozen=True)
class SplitDecision:
    """Resolved relationship between consecutive content items.

    ``incoming_start`` and ``outgoing_end`` are absolute encoded-byte offsets.

    A hard cut uses the same offset for both boundaries. A crossfade preserves
    the shared audio interval by allowing the incoming item to start before the
    outgoing item ends. NO_BOUNDARY intentionally carries no offsets.
    """

    kind: SplitKind
    incoming_start: int | None = None
    outgoing_end: int | None = None

    def __post_init__(self) -> None:
        if self.kind is SplitKind.NO_BOUNDARY:
            if self.incoming_start is not None or self.outgoing_end is not None:
                raise ValueError("NO_BOUNDARY cannot carry boundary offsets")
            return

        if self.incoming_start is None or self.outgoing_end is None:
            raise ValueError("split decisions require both boundary offsets")
        if self.incoming_start < 0 or self.outgoing_end < 0:
            raise ValueError("boundary offsets must be non-negative")

        if self.kind is SplitKind.HARD_CUT:
            if self.incoming_start != self.outgoing_end:
                raise ValueError("HARD_CUT boundaries must be identical")
            return

        if self.kind is SplitKind.CROSSFADE:
            if self.incoming_start >= self.outgoing_end:
                raise ValueError("CROSSFADE requires incoming_start < outgoing_end")
            return

        raise ValueError(f"unsupported split kind: {self.kind}")


class EncodedAudioRingBuffer:
    """Bounded encoded-audio buffer addressed by absolute byte offsets."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

        self._max_bytes = max_bytes
        self._data = bytearray()
        self._start_offset = 0
        self._end_offset = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def start_offset(self) -> int:
        """Oldest absolute byte offset still retained."""

        return self._start_offset

    @property
    def end_offset(self) -> int:
        """Absolute byte offset immediately after the newest byte."""

        return self._end_offset

    @property
    def retained_bytes(self) -> int:
        return len(self._data)

    def append(self, data: bytes) -> tuple[int, int]:
        """Append bytes and return their absolute half-open span."""

        if not data:
            return self._end_offset, self._end_offset

        chunk_start = self._end_offset
        self._data.extend(data)
        self._end_offset += len(data)

        overflow = len(self._data) - self._max_bytes
        if overflow > 0:
            del self._data[:overflow]
            self._start_offset += overflow

        return chunk_start, self._end_offset

    def contains(self, start: int, end: int) -> bool:
        """Return whether the complete half-open span is retained."""

        if start < 0 or end < start:
            return False
        return self._start_offset <= start and end <= self._end_offset

    def read(self, start: int, end: int) -> bytes:
        """Return a retained absolute half-open byte range."""

        if start < 0:
            raise ValueError("start must be non-negative")
        if end < start:
            raise ValueError("end must not be smaller than start")
        if not self.contains(start, end):
            raise ValueError(
                f"requested span [{start}, {end}) is outside retained range "
                f"[{self._start_offset}, {self._end_offset})"
            )

        local_start = start - self._start_offset
        local_end = end - self._start_offset
        return bytes(self._data[local_start:local_end])


@dataclass(frozen=True)
class EncodedAudioFrame:
    """One encoded audio frame placed on an exact accumulated audio timeline."""

    codec: str
    offset: int
    length: int
    sample_rate: int
    samples: int
    time_seconds: float

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("frame offset must be non-negative")
        if self.length <= 0:
            raise ValueError("frame length must be greater than zero")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.samples <= 0:
            raise ValueError("samples must be greater than zero")
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")


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


@dataclass(frozen=True)
class IcyParseResult:
    """Audio bytes and metadata events produced from one incremental feed."""

    audio: bytes
    events: tuple[MetadataEvent, ...]


class IcyStreamParser:
    """Incremental ICY metadata parser for streams using ``icy-metaint``.

    The parser accepts arbitrary chunk boundaries, strips ICY metadata blocks,
    and reports metadata events anchored to absolute offsets in the clean audio
    byte stream.
    """

    def __init__(self, metaint: int) -> None:
        if metaint <= 0:
            raise ValueError("metaint must be greater than zero")

        self._metaint = metaint
        self._audio_until_metadata = metaint
        self._audio_offset = 0
        self._state = "audio"
        self._metadata_remaining = 0
        self._metadata_buffer = bytearray()

    @property
    def metaint(self) -> int:
        return self._metaint

    @property
    def audio_offset(self) -> int:
        """Absolute count of clean audio bytes emitted so far."""

        return self._audio_offset

    def feed(self, chunk: bytes) -> IcyParseResult:
        """Consume one raw ICY chunk and return clean audio plus new events."""

        if not chunk:
            return IcyParseResult(audio=b"", events=())

        audio_out = bytearray()
        events: list[MetadataEvent] = []
        pos = 0

        while pos < len(chunk):
            if self._state == "audio":
                take = min(self._audio_until_metadata, len(chunk) - pos)
                if take:
                    audio_out.extend(chunk[pos : pos + take])
                    pos += take
                    self._audio_offset += take
                    self._audio_until_metadata -= take

                if self._audio_until_metadata == 0:
                    self._state = "metadata_length"
                continue

            if self._state == "metadata_length":
                length_byte = chunk[pos]
                pos += 1
                self._metadata_remaining = length_byte * 16
                self._metadata_buffer.clear()

                if self._metadata_remaining == 0:
                    self._audio_until_metadata = self._metaint
                    self._state = "audio"
                else:
                    self._state = "metadata_body"
                continue

            if self._state == "metadata_body":
                take = min(self._metadata_remaining, len(chunk) - pos)
                if take:
                    self._metadata_buffer.extend(chunk[pos : pos + take])
                    pos += take
                    self._metadata_remaining -= take

                if self._metadata_remaining == 0:
                    event = self._decode_metadata_event()
                    if event is not None:
                        events.append(event)

                    self._metadata_buffer.clear()
                    self._audio_until_metadata = self._metaint
                    self._state = "audio"
                continue

            raise RuntimeError(f"invalid ICY parser state: {self._state}")

        return IcyParseResult(audio=bytes(audio_out), events=tuple(events))

    def _decode_metadata_event(self) -> MetadataEvent | None:
        raw = bytes(self._metadata_buffer).rstrip(b"\x00")
        if not raw:
            return None

        text = raw.decode("utf-8", errors="replace")
        title = _extract_stream_title(text)
        if title is None:
            return None

        return MetadataEvent(title=title, audio_offset=self._audio_offset)


def _extract_stream_title(metadata: str) -> str | None:
    """Extract the ICY ``StreamTitle`` value from one metadata block."""

    marker = "StreamTitle='"
    start = metadata.find(marker)
    if start < 0:
        return None

    start += len(marker)
    end = metadata.find("';", start)
    if end < 0:
        return None

    title = metadata[start:end].strip()
    return title or None


@dataclass(frozen=True)
class TimedMetadataEvent:
    """ICY metadata event resolved onto the encoded-audio frame timeline."""

    title: str
    audio_offset: int
    audio_time_seconds: float


@dataclass(frozen=True)
class RippingIngestResult:
    """Result of feeding one raw ICY chunk into the ripping ingest pipeline."""

    audio: bytes
    metadata_events: tuple[MetadataEvent, ...]
    timed_metadata_events: tuple[TimedMetadataEvent, ...]


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


@dataclass(frozen=True)
class TrackCandidate:
    """Durable metadata title eligible for later acoustic boundary matching."""

    title: str
    start_offset: int
    start_time_seconds: float
    confirmed_at_offset: int
    confirmed_at_time_seconds: float


@dataclass(frozen=True)
class MetadataSemanticDecision:
    """Semantic interpretation of one metadata title transition."""

    title: str
    kind: SplitKind
    start_offset: int
    start_time_seconds: float
    lifetime_seconds: float


class MetadataSemanticTracker:
    """Classify metadata lifetimes before any acoustic split analysis.

    A metadata title becomes durable only after it has survived at least
    ``transient_threshold_seconds``. Titles that are replaced earlier are
    classified as ``NO_BOUNDARY``.
    """

    def __init__(self, transient_threshold_seconds: float = 8.0) -> None:
        if transient_threshold_seconds <= 0:
            raise ValueError("transient_threshold_seconds must be greater than zero")

        self._threshold = transient_threshold_seconds
        self._current: TimedMetadataEvent | None = None
        self._current_confirmed = False

    @property
    def transient_threshold_seconds(self) -> float:
        return self._threshold

    @property
    def current_title(self) -> str | None:
        return self._current.title if self._current is not None else None

    def feed(
        self, event: TimedMetadataEvent
    ) -> tuple[tuple[MetadataSemanticDecision, ...], tuple[TrackCandidate, ...]]:
        """Consume one timed metadata event.

        Returns semantic decisions about completed metadata lifetimes plus any
        newly confirmed durable track candidates.
        """

        decisions: list[MetadataSemanticDecision] = []
        candidates: list[TrackCandidate] = []

        if self._current is None:
            self._current = event
            return (), ()

        if event.title == self._current.title:
            if not self._current_confirmed:
                lifetime = event.audio_time_seconds - self._current.audio_time_seconds
                if lifetime >= self._threshold:
                    self._current_confirmed = True
                    candidates.append(
                        TrackCandidate(
                            title=self._current.title,
                            start_offset=self._current.audio_offset,
                            start_time_seconds=self._current.audio_time_seconds,
                            confirmed_at_offset=event.audio_offset,
                            confirmed_at_time_seconds=event.audio_time_seconds,
                        )
                    )
            return tuple(decisions), tuple(candidates)

        lifetime = event.audio_time_seconds - self._current.audio_time_seconds
        if lifetime < 0:
            raise ValueError("metadata events must be monotonic in audio time")

        decisions.append(
            MetadataSemanticDecision(
                title=self._current.title,
                kind=(
                    SplitKind.HARD_CUT
                    if self._current_confirmed or lifetime >= self._threshold
                    else SplitKind.NO_BOUNDARY
                ),
                start_offset=self._current.audio_offset,
                start_time_seconds=self._current.audio_time_seconds,
                lifetime_seconds=lifetime,
            )
        )

        if not self._current_confirmed and lifetime >= self._threshold:
            candidates.append(
                TrackCandidate(
                    title=self._current.title,
                    start_offset=self._current.audio_offset,
                    start_time_seconds=self._current.audio_time_seconds,
                    confirmed_at_offset=event.audio_offset,
                    confirmed_at_time_seconds=event.audio_time_seconds,
                )
            )

        self._current = event
        self._current_confirmed = False
        return tuple(decisions), tuple(candidates)

    def confirm_current(
        self, *, audio_offset: int, audio_time_seconds: float
    ) -> tuple[TrackCandidate, ...]:
        """Confirm the current title after enough audio time has elapsed.

        This is used when no repeated metadata packet arrives for the same
        title but the stream has nevertheless advanced beyond the transient
        threshold.
        """

        if self._current is None or self._current_confirmed:
            return ()

        if audio_time_seconds < self._current.audio_time_seconds:
            raise ValueError("audio_time_seconds must be monotonic")

        lifetime = audio_time_seconds - self._current.audio_time_seconds
        if lifetime < self._threshold:
            return ()

        self._current_confirmed = True
        return (
            TrackCandidate(
                title=self._current.title,
                start_offset=self._current.audio_offset,
                start_time_seconds=self._current.audio_time_seconds,
                confirmed_at_offset=audio_offset,
                confirmed_at_time_seconds=audio_time_seconds,
            ),
        )


@dataclass(frozen=True)
class AcousticWindow:
    """Frame-aligned encoded audio window prepared for acoustic analysis."""

    start_offset: int
    end_offset: int
    start_time_seconds: float
    end_time_seconds: float
    data: bytes

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if len(self.data) != self.end_offset - self.start_offset:
            raise ValueError("data length must match offset span")


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


@dataclass(frozen=True)
class DecodedPcm:
    """Mono signed 16-bit little-endian PCM prepared for acoustic analysis."""

    sample_rate: int
    channels: int
    sample_width_bytes: int
    data: bytes

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.channels != 1:
            raise ValueError("DecodedPcm must be mono")
        if self.sample_width_bytes != 2:
            raise ValueError("DecodedPcm must use 16-bit samples")
        if len(self.data) % self.sample_width_bytes != 0:
            raise ValueError("PCM byte length must align to sample width")

    @property
    def sample_count(self) -> int:
        return len(self.data) // self.sample_width_bytes


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


@dataclass(frozen=True)
class AcousticLevel:
    """RMS energy measured over one PCM analysis window."""

    start_time_seconds: float
    end_time_seconds: float
    rms: float

    def __post_init__(self) -> None:
        if self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if self.rms < 0:
            raise ValueError("rms must be non-negative")


@dataclass(frozen=True)
class AcousticProfile:
    """Ordered RMS measurements derived from decoded PCM."""

    levels: tuple[AcousticLevel, ...]

    def __post_init__(self) -> None:
        previous_end = -1.0
        for level in self.levels:
            if level.start_time_seconds < previous_end:
                raise ValueError("levels must be ordered and non-overlapping")
            previous_end = level.end_time_seconds

    def minimum_level(self) -> AcousticLevel | None:
        if not self.levels:
            return None
        return min(self.levels, key=lambda level: level.rms)


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
        from array import array

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


@dataclass(frozen=True)
class AcousticBoundaryCandidate:
    """One local RMS minimum expressed on the absolute stream timeline."""

    time_seconds: float
    rms: float
    relative_time_seconds: float

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
        if self.relative_time_seconds < 0:
            raise ValueError("relative_time_seconds must be non-negative")
        if self.rms < 0:
            raise ValueError("rms must be non-negative")


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


@dataclass(frozen=True)
class BoundaryMatch:
    """One acoustic candidate selected for a semantic track transition."""

    track: TrackCandidate
    acoustic: AcousticBoundaryCandidate
    delta_seconds: float

    def __post_init__(self) -> None:
        if self.delta_seconds < 0:
            raise ValueError("delta_seconds must be non-negative")


class NearestBoundaryMatcher:
    """Match a semantic track candidate to the nearest acoustic minimum."""

    def __init__(self, search_radius_seconds: float = 8.0) -> None:
        if search_radius_seconds <= 0:
            raise ValueError("search_radius_seconds must be greater than zero")
        self._search_radius = search_radius_seconds

    @property
    def search_radius_seconds(self) -> float:
        return self._search_radius

    def match(
        self,
        *,
        track: TrackCandidate,
        acoustic_candidates: tuple[AcousticBoundaryCandidate, ...],
    ) -> BoundaryMatch | None:
        """Return the nearest acoustic candidate inside the configured radius."""

        eligible = [
            candidate
            for candidate in acoustic_candidates
            if abs(candidate.time_seconds - track.start_time_seconds) <= self._search_radius
        ]
        if not eligible:
            return None

        selected = min(
            eligible,
            key=lambda candidate: (
                abs(candidate.time_seconds - track.start_time_seconds),
                candidate.rms,
                candidate.time_seconds,
            ),
        )

        return BoundaryMatch(
            track=track,
            acoustic=selected,
            delta_seconds=abs(selected.time_seconds - track.start_time_seconds),
        )


class BoundaryRelation(StrEnum):
    """Relative placement of semantic and broad acoustic boundary candidates."""

    AGREEMENT = "agreement"
    ACOUSTIC_EARLIER = "acoustic_earlier"
    SEMANTIC_EARLIER = "semantic_earlier"


@dataclass(frozen=True)
class BoundaryRelationResult:
    """Intermediate comparison between semantic and acoustic candidates."""

    relation: BoundaryRelation
    semantic_time_seconds: float
    acoustic_time_seconds: float
    signed_delta_seconds: float

    def __post_init__(self) -> None:
        if self.semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if self.acoustic_time_seconds < 0:
            raise ValueError("acoustic_time_seconds must be non-negative")


class BoundaryRelationClassifier:
    """Classify semantic-vs-acoustic timing before final split decisions."""

    def __init__(self, divergence_threshold_seconds: float = 3.0) -> None:
        if divergence_threshold_seconds <= 0:
            raise ValueError("divergence_threshold_seconds must be greater than zero")
        self._threshold = divergence_threshold_seconds

    @property
    def divergence_threshold_seconds(self) -> float:
        return self._threshold

    def classify(
        self,
        *,
        semantic_time_seconds: float,
        acoustic_time_seconds: float,
    ) -> BoundaryRelationResult:
        if semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if acoustic_time_seconds < 0:
            raise ValueError("acoustic_time_seconds must be non-negative")

        signed_delta = semantic_time_seconds - acoustic_time_seconds

        if signed_delta > self._threshold:
            relation = BoundaryRelation.ACOUSTIC_EARLIER
        elif signed_delta < -self._threshold:
            relation = BoundaryRelation.SEMANTIC_EARLIER
        else:
            relation = BoundaryRelation.AGREEMENT

        return BoundaryRelationResult(
            relation=relation,
            semantic_time_seconds=semantic_time_seconds,
            acoustic_time_seconds=acoustic_time_seconds,
            signed_delta_seconds=signed_delta,
        )


class TemporalSplitKind(StrEnum):
    """Split policy expressed in absolute stream time before frame alignment."""

    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"


@dataclass(frozen=True)
class TemporalSplitDecision:
    """Intermediate split decision expressed only in absolute stream time."""

    kind: TemporalSplitKind
    incoming_start_seconds: float
    outgoing_end_seconds: float

    def __post_init__(self) -> None:
        if self.incoming_start_seconds < 0:
            raise ValueError("incoming_start_seconds must be non-negative")
        if self.outgoing_end_seconds < 0:
            raise ValueError("outgoing_end_seconds must be non-negative")

        if self.kind is TemporalSplitKind.HARD_CUT:
            if self.incoming_start_seconds != self.outgoing_end_seconds:
                raise ValueError("HARD_CUT requires identical temporal boundaries")
        elif (
            self.kind is TemporalSplitKind.CROSSFADE
            and self.incoming_start_seconds >= self.outgoing_end_seconds
        ):
            raise ValueError("CROSSFADE requires incoming_start_seconds < outgoing_end_seconds")


class TemporalSplitPolicy:
    """Convert semantic/acoustic relation into an intermediate temporal split."""

    def decide(self, relation: BoundaryRelationResult) -> TemporalSplitDecision:
        if relation.relation is BoundaryRelation.AGREEMENT:
            boundary = relation.semantic_time_seconds
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )

        if relation.relation is BoundaryRelation.ACOUSTIC_EARLIER:
            return TemporalSplitDecision(
                kind=TemporalSplitKind.CROSSFADE,
                incoming_start_seconds=relation.acoustic_time_seconds,
                outgoing_end_seconds=relation.semantic_time_seconds,
            )

        if relation.relation is BoundaryRelation.SEMANTIC_EARLIER:
            boundary = relation.acoustic_time_seconds
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )

        raise RuntimeError(f"unsupported boundary relation: {relation.relation}")


class SplitAlignmentError(RuntimeError):
    """Raised when a temporal split cannot be aligned to retained codec frames."""


class TemporalSplitAligner:
    """Convert temporal split decisions into frame-aligned byte offsets."""

    def align(
        self,
        *,
        decision: TemporalSplitDecision,
        timeline: IncrementalFrameTimeline,
    ) -> SplitDecision:
        frames = timeline.frames
        if not frames:
            raise SplitAlignmentError("cannot align split without retained frames")

        incoming_frame = self._nearest_frame(
            frames,
            decision.incoming_start_seconds,
        )
        outgoing_frame = self._nearest_frame(
            frames,
            decision.outgoing_end_seconds,
        )

        if incoming_frame is None or outgoing_frame is None:
            raise SplitAlignmentError("temporal split lies outside the retained frame timeline")

        if decision.kind is TemporalSplitKind.HARD_CUT:
            return SplitDecision(
                kind=SplitKind.HARD_CUT,
                incoming_start=incoming_frame.offset,
                outgoing_end=incoming_frame.offset,
            )

        if decision.kind is TemporalSplitKind.CROSSFADE:
            if incoming_frame.offset >= outgoing_frame.offset:
                raise SplitAlignmentError(
                    "aligned crossfade boundaries must preserve incoming < outgoing"
                )

            return SplitDecision(
                kind=SplitKind.CROSSFADE,
                incoming_start=incoming_frame.offset,
                outgoing_end=outgoing_frame.offset,
            )

        raise RuntimeError(f"unsupported temporal split kind: {decision.kind}")

    @staticmethod
    def _nearest_frame(
        frames: tuple[EncodedAudioFrame, ...],
        time_seconds: float,
    ) -> EncodedAudioFrame | None:
        first = frames[0]
        last = frames[-1]
        last_end_time = last.time_seconds + last.samples / last.sample_rate

        if time_seconds < first.time_seconds or time_seconds > last_end_time:
            return None

        return min(
            frames,
            key=lambda frame: (
                abs(frame.time_seconds - time_seconds),
                frame.time_seconds,
            ),
        )


@dataclass(frozen=True)
class TrackByteRange:
    """Absolute encoded-byte range to materialize for one track."""

    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")


@dataclass(frozen=True)
class TrackWritePlan:
    """Encoded-byte ranges for the outgoing and incoming tracks."""

    outgoing: TrackByteRange
    incoming: TrackByteRange


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
