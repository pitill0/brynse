from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from fluxtuner_ripper.models import (
    SplitDecision,
    SplitKind,
    TrackByteRange,
    TrackWritePlan,
)


class EncodedByteSource(Protocol):
    """Readable encoded-byte source addressed by absolute offsets."""

    @property
    def end_offset(self) -> int:
        """Absolute offset immediately after the newest available byte."""

    def contains(self, start: int, end: int) -> bool:
        """Return whether the complete half-open span is available."""

    def read(self, start: int, end: int) -> bytes:
        """Return one absolute half-open byte range."""


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

    def iter_range(
        self,
        *,
        source: EncodedByteSource,
        byte_range: TrackByteRange,
        chunk_size: int = 64 * 1024,
    ) -> Iterator[bytes]:
        """Yield one retained encoded range using bounded reads."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if not source.contains(
            byte_range.start_offset,
            byte_range.end_offset,
        ):
            raise ValueError("requested track range is not fully retained")

        current = byte_range.start_offset

        while current < byte_range.end_offset:
            chunk_end = min(
                current + chunk_size,
                byte_range.end_offset,
            )
            yield source.read(current, chunk_end)
            current = chunk_end

    def write_range(
        self,
        *,
        source: EncodedByteSource,
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

    def finalize_stream(
        self,
        *,
        chunks: Iterator[bytes],
        output_path: Path,
    ) -> None:
        """Remux MP3 chunks to disk without materializing the full segment in RAM."""

        import os
        import subprocess  # nosec B404
        import tempfile
        from contextlib import suppress

        iterator = iter(chunks)

        first_chunk: bytes | None = None
        for chunk in iterator:
            if chunk:
                first_chunk = chunk
                break

        if first_chunk is None:
            raise ValueError("chunks must contain encoded data")

        output_path.parent.mkdir(parents=True, exist_ok=True)

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

        temp_path: Path | None = None
        stderr_path: Path | None = None
        process: subprocess.Popen[bytes] | None = None
        replaced = False

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_handle:
                temp_path = Path(output_handle.name)

                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    dir=output_path.parent,
                    prefix=".fluxtuner-ffmpeg-stderr-",
                    suffix=".tmp",
                    delete=False,
                ) as stderr_handle:
                    stderr_path = Path(stderr_handle.name)

                    try:
                        process = subprocess.Popen(  # nosec B603
                            command,
                            stdin=subprocess.PIPE,
                            stdout=output_handle,
                            stderr=stderr_handle,
                            bufsize=0,
                        )
                    except FileNotFoundError as exc:
                        raise TrackFinalizeError(
                            f"FFmpeg binary not found: {self._ffmpeg_binary}"
                        ) from exc

                    if process.stdin is None:
                        raise TrackFinalizeError("FFmpeg MP3 finalization did not provide stdin")

                    try:
                        process.stdin.write(first_chunk)

                        for chunk in iterator:
                            if chunk:
                                process.stdin.write(chunk)
                    except BrokenPipeError:
                        # FFmpeg may close stdin early when it detects invalid input.
                        pass
                    finally:
                        with suppress(BrokenPipeError, OSError):
                            process.stdin.close()

                    returncode = process.wait()

                    output_handle.flush()
                    os.fsync(output_handle.fileno())

                    stderr_handle.flush()
                    stderr_handle.seek(0, os.SEEK_END)
                    stderr_size = stderr_handle.tell()
                    stderr_handle.seek(max(0, stderr_size - 64 * 1024))
                    stderr_tail = stderr_handle.read(64 * 1024)

            if returncode != 0:
                error = stderr_tail.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
                raise TrackFinalizeError(
                    f"FFmpeg MP3 finalization failed with exit code {returncode}: {error}"
                )

            if temp_path is None or temp_path.stat().st_size == 0:
                raise TrackFinalizeError("FFmpeg MP3 finalization produced no output")

            os.replace(temp_path, output_path)
            replaced = True

        finally:
            if process is not None and process.poll() is None:
                with suppress(OSError):
                    if process.stdin is not None:
                        process.stdin.close()

                with suppress(OSError):
                    process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        process.kill()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=5)

            if temp_path is not None and not replaced:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)

            if stderr_path is not None:
                with suppress(OSError):
                    stderr_path.unlink(missing_ok=True)


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

    def finalize_stream(
        self,
        *,
        chunks: Iterator[bytes],
        output_path: Path,
    ) -> None:
        """Remux AAC chunks to M4A without materializing the segment in RAM."""

        import os
        import subprocess  # nosec B404
        import tempfile
        from contextlib import suppress

        iterator = iter(chunks)

        first_chunk: bytes | None = None
        for chunk in iterator:
            if chunk:
                first_chunk = chunk
                break

        if first_chunk is None:
            raise ValueError("chunks must contain encoded data")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_fd, temp_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        os.close(temp_fd)
        temp_path = Path(temp_name)

        stderr_path: Path | None = None
        process: subprocess.Popen[bytes] | None = None
        replaced = False

        command = [
            self._ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "aac",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temp_path),
        ]

        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=output_path.parent,
                prefix=".fluxtuner-ffmpeg-stderr-",
                suffix=".tmp",
                delete=False,
            ) as stderr_handle:
                stderr_path = Path(stderr_handle.name)

                try:
                    process = subprocess.Popen(  # nosec B603
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_handle,
                        bufsize=0,
                    )
                except FileNotFoundError as exc:
                    raise TrackFinalizeError(
                        f"FFmpeg binary not found: {self._ffmpeg_binary}"
                    ) from exc

                if process.stdin is None:
                    raise TrackFinalizeError("FFmpeg AAC finalization did not provide stdin")

                try:
                    process.stdin.write(first_chunk)

                    for chunk in iterator:
                        if chunk:
                            process.stdin.write(chunk)
                except BrokenPipeError:
                    # FFmpeg may stop consuming invalid encoded input early.
                    pass
                finally:
                    with suppress(BrokenPipeError, OSError):
                        process.stdin.close()

                returncode = process.wait()

                stderr_handle.flush()
                stderr_handle.seek(0, os.SEEK_END)
                stderr_size = stderr_handle.tell()
                stderr_handle.seek(max(0, stderr_size - 64 * 1024))
                stderr_tail = stderr_handle.read(64 * 1024)

            if returncode != 0:
                error = stderr_tail.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
                raise TrackFinalizeError(
                    f"FFmpeg AAC finalization failed with exit code {returncode}: {error}"
                )

            if temp_path.stat().st_size == 0:
                raise TrackFinalizeError("FFmpeg AAC finalization produced no output")

            with temp_path.open("r+b", buffering=0) as finalized_handle:
                os.fsync(finalized_handle.fileno())

            os.replace(temp_path, output_path)
            replaced = True

        finally:
            if process is not None and process.poll() is None:
                with suppress(OSError):
                    if process.stdin is not None:
                        process.stdin.close()

                with suppress(OSError):
                    process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        process.kill()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=5)

            if not replaced:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)

            if stderr_path is not None:
                with suppress(OSError):
                    stderr_path.unlink(missing_ok=True)


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
        source: EncodedByteSource,
        byte_range: TrackByteRange,
        directory: Path,
        stem: str,
        codec: str,
        chunk_size: int = 64 * 1024,
    ) -> Path:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not stem or not stem.strip():
            raise ValueError("stem must not be empty")

        normalized = codec.strip().lower()
        finalizer: object

        if normalized == "mp3":
            finalizer = self._mp3_finalizer
            output_codec = "mp3"
        elif normalized == "aac":
            finalizer = self._aac_finalizer
            output_codec = "m4a"
        else:
            raise ValueError(f"unsupported codec for track output: {codec}")

        finalize_stream = getattr(finalizer, "finalize_stream", None)

        if callable(finalize_stream):
            directory.mkdir(parents=True, exist_ok=True)
            output_path = directory / f"{stem}.{output_codec}"

            chunks = self._encoded_writer.iter_range(
                source=source,
                byte_range=byte_range,
                chunk_size=chunk_size,
            )

            finalize_stream(
                chunks=chunks,
                output_path=output_path,
            )
            return output_path

        raw = self._encoded_writer.write_range(
            source=source,
            byte_range=byte_range,
        )
        finalized = finalizer.finalize(raw)

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
