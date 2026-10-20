from __future__ import annotations

import errno
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from brynse._filename import publish_unique


def test_concurrent_publication_preserves_every_recording(
    tmp_path: Path,
) -> None:
    count = 20

    def publish(index: int) -> Path:
        temporary = tmp_path / f".recording-{index}.tmp"
        temporary.write_bytes(f"recording-{index}".encode())

        return publish_unique(
            temp_path=temporary,
            directory=tmp_path,
            stem="Cadena SER",
            extension=".mp3",
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        paths = list(executor.map(publish, range(count)))

    assert len(set(paths)) == count
    assert {path.read_bytes() for path in paths} == {
        f"recording-{index}".encode() for index in range(count)
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_recording_is_never_replaced(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "Cadena SER.mp3"
    existing.write_bytes(b"ORIGINAL")

    temporary = tmp_path / ".incoming.tmp"
    temporary.write_bytes(b"NEW")

    result = publish_unique(
        temp_path=temporary,
        directory=tmp_path,
        stem="Cadena SER",
        extension=".mp3",
    )

    assert existing.read_bytes() == b"ORIGINAL"
    assert result.name == "Cadena SER-2.mp3"
    assert result.read_bytes() == b"NEW"


def test_publication_failure_preserves_original_and_temporary(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "Cadena SER.mp3"
    existing.write_bytes(b"ORIGINAL")

    temporary = tmp_path / ".incoming.tmp"
    temporary.write_bytes(b"NEW")

    with (
        patch(
            "brynse._filename.os.link",
            side_effect=OSError(errno.EPERM, "link forbidden"),
        ),
        pytest.raises(OSError) as captured,
    ):
        publish_unique(
            temp_path=temporary,
            directory=tmp_path,
            stem="Cadena SER",
            extension=".mp3",
        )

    assert captured.value.errno == errno.EPERM
    assert existing.read_bytes() == b"ORIGINAL"
    assert temporary.read_bytes() == b"NEW"


@pytest.mark.parametrize(
    "stem",
    ["../escape", "foo/bar", r"foo\bar", ".", ".."],
)
def test_publication_rejects_unsafe_stems(
    tmp_path: Path,
    stem: str,
) -> None:
    temporary = tmp_path / ".incoming.tmp"
    temporary.write_bytes(b"NEW")

    with pytest.raises(ValueError):
        publish_unique(
            temp_path=temporary,
            directory=tmp_path,
            stem=stem,
            extension=".mp3",
        )

    assert temporary.read_bytes() == b"NEW"


def test_long_unicode_collisions_remain_distinct(
    tmp_path: Path,
) -> None:
    paths = []

    for index in range(3):
        temporary = tmp_path / f".unicode-{index}.tmp"
        temporary.write_bytes(f"payload-{index}".encode())

        paths.append(
            publish_unique(
                temp_path=temporary,
                directory=tmp_path,
                stem="界" * 150,
                extension=".m4a",
            )
        )

    assert len(set(paths)) == 3
    assert all(len(path.name.encode("utf-8")) <= 240 for path in paths)
    assert [path.read_bytes() for path in paths] == [
        b"payload-0",
        b"payload-1",
        b"payload-2",
    ]


def _publish_from_process(args: tuple[str, int]) -> str:
    directory_name, index = args
    directory = Path(directory_name)
    temporary = directory / f".process-{index}.tmp"
    temporary.write_bytes(f"process-{index}".encode())

    result = publish_unique(
        temp_path=temporary,
        directory=directory,
        stem="Radio Pública",
        extension=".mp3",
    )
    return str(result)


def test_multiprocess_publication(tmp_path: Path) -> None:
    from concurrent.futures import ProcessPoolExecutor

    count = 12

    with ProcessPoolExecutor(max_workers=4) as executor:
        paths = [
            Path(value)
            for value in executor.map(
                _publish_from_process,
                [(str(tmp_path), index) for index in range(count)],
            )
        ]

    assert len(set(paths)) == count
    assert {path.read_bytes() for path in paths} == {
        f"process-{index}".encode() for index in range(count)
    }
    assert not list(tmp_path.glob("*.tmp"))
