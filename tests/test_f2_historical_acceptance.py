"""F-2: reproduce historical excluded-byte publication without acoustic replay.

The planner checks always run. The real MP3 integration cases require
the original local October 2026 capture bundles and skip otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from brynse.models import SegmentByteRange, SplitDecision, SplitKind
from brynse.output import (
    SegmentRangePlanner,
    create_safe_segment_output_service,
)

REPO = Path(__file__).resolve().parents[1]

# Exact half-open byte ranges reconstructed from 20 historic MP3 payloads.
CASES = (
    (
        "live-phase6e-complete-pcm-authority-replayable-final-20261013-154148",
        "23a9bde85de129a9660acf92c5cee1ab6d730cb54380b02ab88b910c71f7762f",
        (
            (
                1248235,
                1768594,
                "8a81c2a82173e338288bca5bd6ec0c0fd933f975c8df08a871c88804dcd26d67",
            ),
            (
                6760908,
                6868114,
                "32ebcf8d436a594a2aaabc475425237cd28b6f3b5a395d53a7a3391142e336a0",
            ),
            (
                15025214,
                15170037,
                "961e4d5af30933423f3f386fdda6cc006e97cb81c1604d929ad7bdfb6129a8d0",
            ),
            (
                21694589,
                21823112,
                "6b75c4474e72c722ce5e682e1f5fb0b7ebae962156872ad99c20a3bc855a5ed7",
            ),
            (
                28473678,
                28562703,
                "e4a832d2306574b2cc50af75b808fdab748f1ecd1fac9a5057ca786f6f18bf2c",
            ),
            (
                37344862,
                37569306,
                "66cb9e33584835586ee0fae4899db07f5ea9a095794dc1aeb954a3f06b3845a9",
            ),
            (
                43836186,
                43974112,
                "68b565ada018d4fd005c35f84bc80242d2a5305be7f80587cf3dc585b5da0e7c",
            ),
        ),
        8,
    ),
    (
        "live-phase6-main-20261014-184257",
        "d4b56331c885f12bcf123f3bc9393a4a845e0277a269c403731de234a19d3112",
        (
            (
                1343530,
                1863262,
                "f6006c2ba88dc99b896de6e9d7121c5611812d8a2b74606d88a1fa4f67682693",
            ),
            (
                14705476,
                14837133,
                "d7a80a5a56947ad2811727687da6484d6fa76520da04dd5d12f8dc8d0854adaa",
            ),
            (
                20606224,
                21058247,
                "714634b4d74fb933032b10630d3d13fbf7ec4775645da45da0a8bae1ddb80cb2",
            ),
            (
                24354691,
                24433058,
                "1951c73729af79c5455a4d83e849bcb58d3c47e90334e4c235093952eb2c9878",
            ),
            (
                29093721,
                29510635,
                "094fb1b2eabb621ec00168316dca23b44d8fb06042a18ff2dfea211b308d8955",
            ),
            (
                42737163,
                42883239,
                "2d18cd08eb2c94bec8c669203d5819df39aedb579241cf91055d5e8fb113432b",
            ),
            (
                47914423,
                48116924,
                "ad16dc006d74e0061d86fe15c437575e8427f564394219240b97da6418138e02",
            ),
            (
                52010841,
                52106763,
                "e083a39a574a2000b85b47101fe218c2cf079b9bb3fc5f47f453795f963c6716",
            ),
        ),
        12,
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def id3_length(data: bytes) -> int:
    assert len(data) >= 10 and data.startswith(b"ID3")
    size = data[6:10]
    assert all(value < 128 for value in size)
    body = (size[0] << 21) | (size[1] << 14) | (size[2] << 7) | size[3]
    length = 10 + body + (10 if data[5] & 16 else 0)
    assert length <= len(data)
    return length


def extract_icy_audio(raw: bytes, metaint: int) -> bytes:
    assert type(metaint) is int and metaint > 0
    audio = bytearray()
    position = 0

    while position < len(raw):
        length = min(metaint, len(raw) - position)
        audio.extend(raw[position : position + length])
        position += length

        if length < metaint or position == len(raw):
            break

        metadata_length = raw[position] * 16
        position += 1
        assert position + metadata_length <= len(raw)
        position += metadata_length

    assert position == len(raw)
    return bytes(audio)


class ImmutableAudioSource:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio

    @property
    def end_offset(self) -> int:
        return len(self.audio)

    def contains(self, start: int, end: int) -> bool:
        return 0 <= start <= end <= len(self.audio)

    def read(self, start: int, end: int) -> bytes:
        if not self.contains(start, end):
            raise ValueError("range outside source")
        return self.audio[start:end]


ALL_GAPS = tuple(
    (session, start, end, digest)
    for session, _, ranges, _ in CASES
    for start, end, digest in ranges
)


@pytest.mark.parametrize(
    ("session", "start", "end", "expected_sha"),
    ALL_GAPS,
    ids=[f"{session[-15:]}-{start}-{end}" for session, start, end, _ in ALL_GAPS],
)
def test_historical_exclusion_plan(
    session: str,
    start: int,
    end: int,
    expected_sha: str,
) -> None:
    """All historical gaps must be first-class unowned ranges."""
    assert session and len(expected_sha) == 64
    assert 0 < start < end

    plan = SegmentRangePlanner().plan(
        previous_start_offset=start - 1,
        next_end_offset=end + 1,
        decision=SplitDecision(
            kind=SplitKind.EXCLUSION,
            outgoing_end=start,
            incoming_start=end,
        ),
    )

    assert plan.outgoing == SegmentByteRange(start - 1, start)
    assert plan.unowned == SegmentByteRange(start, end)
    assert plan.incoming == SegmentByteRange(end, end + 1)


@pytest.mark.parametrize(
    ("session_name", "source_hash", "gaps", "track_count"),
    CASES,
    ids=("phase6e", "phase6-main"),
)
def test_historical_real_mp3_publication(
    session_name: str,
    source_hash: str,
    gaps: tuple[tuple[int, int, str], ...],
    track_count: int,
    tmp_path: Path,
) -> None:
    """Real publisher must preserve exact captured bytes and historical files."""
    base = REPO / session_name
    capture = base / "session"
    stream = capture / "stream.icy"
    manifest_path = capture / "manifest.json"
    tracks_dir = base / "tracks"

    if not (stream.is_file() and manifest_path.is_file() and tracks_dir.is_dir()):
        pytest.skip("Local historical capture bundle is not available")

    tracks = sorted(tracks_dir.glob("*.mp3"))
    assert len(tracks) == track_count

    protected = [stream, manifest_path, *tracks]
    before = {path: file_sha256(path) for path in protected}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["codec"] == "mp3"
    assert manifest["metaint"] == 32768

    audio = extract_icy_audio(stream.read_bytes(), manifest["metaint"])
    assert sha256(audio) == source_hash

    source = ImmutableAudioSource(audio)
    planner = SegmentRangePlanner()
    service = create_safe_segment_output_service()

    published = []
    for start, end, expected_hash in gaps:
        plan = planner.plan(
            previous_start_offset=start - 1,
            next_end_offset=end + 1,
            decision=SplitDecision(
                kind=SplitKind.EXCLUSION,
                outgoing_end=start,
                incoming_start=end,
            ),
        )
        assert plan.unowned == SegmentByteRange(start, end)

        target = service.write_segment(
            source=source,
            byte_range=plan.unowned,
            directory=tmp_path,
            stem=f"unowned-{start:012d}-{end:012d}",
            codec="mp3",
        )
        assert target.is_file() and not target.is_symlink()
        assert target.parent == tmp_path

        data = target.read_bytes()
        payload = data[id3_length(data) :]
        assert len(payload) == end - start
        assert payload == audio[start:end]
        assert sha256(payload) == expected_hash
        published.append(target)

    assert len(published) == len(gaps)
    assert len(list(tmp_path.glob("unowned-*.mp3"))) == len(gaps)
    assert {path: file_sha256(path) for path in protected} == before


def test_historical_fixture_totals() -> None:
    assert sum(len(gaps) for _, _, gaps, _ in CASES) == 15
    assert sum(end - start for _, _, gaps, _ in CASES for start, end, _ in gaps) == 3_395_498
    assert sum(track_count for _, _, _, track_count in CASES) == 20
