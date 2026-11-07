"""POSIX permissions for radio session capture artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = """
import json
import os
import stat
import sys
from pathlib import Path

from brynse.integrations.radio.capture import RadioSessionCapture

directory = Path(sys.argv[1])
preexisting = sys.argv[2] == "existing"

os.umask(0o022)

if preexisting:
    directory.mkdir(parents=True)

before = None
if preexisting:
    before = stat.S_IMODE(directory.stat().st_mode)

capture = RadioSessionCapture(
    directory=directory,
    codec="mp3",
    metaint=16000,
    metadata_threshold_seconds=8.0,
    transient_exclusion=True,
    autonomous_boundaries=False,
    source_url="https://example.test/stream",
    source_metadata={"Content-Type": "audio/mpeg"},
)

capture.write_stream_chunk(b"sample")
capture.close()

def mode(path):
    return stat.S_IMODE(path.stat().st_mode)

sibling = directory.parent / "umask-control.txt"
sibling.write_text("control", encoding="utf-8")

print(json.dumps({
    "directory": mode(directory),
    "directory_before": before,
    "manifest": mode(capture.manifest_path),
    "stream": mode(capture.stream_path),
    "events": mode(capture.events_path),
    "sibling": mode(sibling),
}))
"""


def _capture_modes(directory: Path, *, existing: bool) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _SCRIPT,
            str(directory),
            "existing" if existing else "new",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_new_capture_directory_and_files_are_private(tmp_path):
    modes = _capture_modes(tmp_path / "session", existing=False)

    assert modes["directory"] == 0o700
    assert modes["manifest"] == 0o600
    assert modes["stream"] == 0o600
    assert modes["events"] == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_capture_in_existing_directory_preserves_directory_mode(tmp_path):
    modes = _capture_modes(tmp_path / "shared", existing=True)

    assert modes["directory_before"] == 0o755
    assert modes["directory"] == 0o755
    assert modes["manifest"] == 0o600
    assert modes["stream"] == 0o600
    assert modes["events"] == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_capture_does_not_change_process_umask(tmp_path):
    modes = _capture_modes(tmp_path / "session", existing=False)

    assert modes["sibling"] == 0o644
