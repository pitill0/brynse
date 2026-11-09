"""Linux descriptor-based publication with a private durable recovery name."""

from __future__ import annotations

import errno
import logging
import os
import re
import secrets
import stat
import sys
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from brynse._filename import (
    _MAX_FILENAME_BYTES,
    _completed_file_mode,
    _utf8_prefix,
)

_RECOVERY_NAME = ".brynse-recovery"
_LOG = logging.getLogger(__name__)

_ACTIVE_OUTPUT_IDENTITY: ContextVar[OutputDirectoryIdentity | None] = ContextVar(
    "brynse_active_output_identity", default=None
)


def _open_checked_output_directory(directory: Path) -> int:
    """Open a destination inode and fail closed if it differs from session preflight.

    Keeping the descriptor for the actual link operation prevents a pathname
    swap between verification and publication from redirecting output.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    fd = os.open(directory.resolve(strict=True), flags)
    try:
        identity = _ACTIVE_OUTPUT_IDENTITY.get()
        if identity is not None:
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) != (identity.device, identity.inode):
                raise OSError(errno.ESTALE, "output directory changed since preflight")
            # Reject symlink changes even when the destination inode is unchanged.
            identity.verify()
        return fd
    except BaseException:
        os.close(fd)
        raise


@dataclass(frozen=True)
class PublicationReceipt:
    """Both paths refer to the same completed recording inode."""

    published_path: Path
    recovery_path: Path


class RecoverablePublicationError(OSError):
    """Publication failed after durable recovery was established."""

    def __init__(self, recovery_path: Path, cause: OSError) -> None:
        self.recovery_path = recovery_path
        super().__init__(
            cause.errno,
            f"publication failed; recording preserved at {recovery_path}",
        )


class PublicationDurabilityError(RecoverablePublicationError):
    """The final link exists, but directory durability is unconfirmed."""

    def __init__(
        self,
        recovery_path: Path,
        published_path: Path,
        cause: OSError,
    ) -> None:
        self.published_path = published_path
        self.recovery_path = recovery_path
        OSError.__init__(
            self,
            cause.errno,
            f"published at {published_path}; directory durability unconfirmed; "
            f"recovery copy at {recovery_path}",
        )


class RecoveryDurabilityError(RecoverablePublicationError):
    """A recovery link exists, but its directory sync failed."""


def _link_fd(source_fd: int, directory_fd: int, name: str) -> None:
    """Exclusively link the open inode via procfs, without AT_EMPTY_PATH privilege."""
    if not sys.platform.startswith("linux"):
        raise OSError(errno.ENOTSUP, "Linux with /proc/self/fd is required")
    try:
        os.link(
            f"/proc/self/fd/{source_fd}",
            name,
            dst_dir_fd=directory_fd,
            follow_symlinks=True,
        )
    except OSError as exc:
        if (
            exc.errno in (errno.ENOENT, errno.ENOTDIR)
            and not Path(f"/proc/self/fd/{source_fd}").exists()
        ):
            raise OSError(errno.ENOTSUP, "procfs file-descriptor links unavailable") from exc
        raise


def _open_private_recovery_dir(output_fd: int) -> int:
    """Create or validate the exclusively owned recovery directory."""
    with suppress(FileExistsError):
        os.mkdir(_RECOVERY_NAME, mode=0o700, dir_fd=output_fd)

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(_RECOVERY_NAME, flags, dir_fd=output_fd)
    try:
        current = os.fstat(fd)
        if not stat.S_ISDIR(current.st_mode):
            raise PermissionError("recovery path is not a directory")
        if current.st_uid != os.geteuid():
            raise PermissionError("recovery directory has a foreign owner")
        if stat.S_IMODE(current.st_mode) != 0o700:
            raise PermissionError("recovery directory must have mode 0700")
        if current.st_dev != os.fstat(output_fd).st_dev:
            raise OSError(errno.EXDEV, "recovery must share the output filesystem")
        return fd
    except BaseException:
        os.close(fd)
        raise


def publish_fd_recoverably(
    *,
    source_fd: int,
    directory: Path,
    stem: str,
    extension: str,
) -> PublicationReceipt:
    """Back up a completed inode, then publish it without overwriting.

    The caller owns source_fd. No path-based source cleanup is performed.
    The private recovery link is removed after confirmed publication by
    publish_private_file; failures preserve the recovery link for inspection.
    """
    if not stem or not stem.strip():
        raise ValueError("stem must not be empty")
    if stem in {".", ".."} or any(c in stem for c in ("/", "\\", "\x00")):
        raise ValueError("unsafe output stem")
    if extension not in {".mp3", ".aac", ".m4a"}:
        raise ValueError("unsupported output extension")

    opened = os.fstat(source_fd)
    if not stat.S_ISREG(opened.st_mode):
        raise ValueError("source descriptor must be a regular file")

    output_fd = _open_checked_output_directory(directory)
    try:
        recovery_fd = _open_private_recovery_dir(output_fd)
        try:
            # Persist the recovery directory entry in its parent before
            # creating the completed recording's recovery hardlink.
            os.fsync(output_fd)

            # Finalize inode metadata before establishing its durable name.
            os.fchmod(source_fd, _completed_file_mode())
            os.fsync(source_fd)

            recovery_name = ""
            for _ in range(10):
                candidate = f"recovery-{secrets.token_hex(16)}{extension}"
                try:
                    _link_fd(source_fd, recovery_fd, candidate)
                    recovery_name = candidate
                    break
                except FileExistsError:
                    continue

            if not recovery_name:
                raise FileExistsError("unable to allocate recovery filename")

            recovery_path = directory / _RECOVERY_NAME / recovery_name
            try:
                os.fsync(recovery_fd)
            except OSError as exc:
                raise RecoveryDurabilityError(recovery_path, exc) from exc

            try:
                index = 1
                while True:
                    suffix = "" if index == 1 else f"-{index}"
                    available = _MAX_FILENAME_BYTES - len((suffix + extension).encode("utf-8"))
                    if available <= 0:
                        raise ValueError("collision suffix exceeds filename budget")

                    bounded = _utf8_prefix(stem, available).rstrip(" .")
                    name = f"{bounded or 'track'}{suffix}{extension}"
                    try:
                        _link_fd(source_fd, output_fd, name)
                        break
                    except FileExistsError:
                        index += 1

                try:
                    os.fsync(output_fd)
                except OSError as exc:
                    raise PublicationDurabilityError(
                        recovery_path,
                        directory / name,
                        exc,
                    ) from exc
            except RecoverablePublicationError:
                raise
            except OSError as exc:
                raise RecoverablePublicationError(recovery_path, exc) from exc

            return PublicationReceipt(
                published_path=directory / name,
                recovery_path=recovery_path,
            )
        finally:
            os.close(recovery_fd)
    finally:
        os.close(output_fd)


def inspect_recovery_directory(directory: Path) -> tuple[str, ...]:
    """List retained private artifacts without following links or deleting audio.

    The inventory is informational: an unrecognized/ambiguous inode is never
    reclaimed automatically. A subsequent explicit reconciliation policy may
    only delete entries whose identity and published hardlink are verified.
    """
    output_fd = _open_checked_output_directory(directory)
    try:
        try:
            private_fd = os.open(
                _RECOVERY_NAME,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=output_fd,
            )
        except FileNotFoundError:
            return ()
        try:
            info = os.fstat(private_fd)
            if (
                info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_dev != os.fstat(output_fd).st_dev
            ):
                raise PermissionError("recovery directory ownership/mode/filesystem changed")
            names = []
            for name in os.listdir(private_fd):
                artifact = os.stat(name, dir_fd=private_fd, follow_symlinks=False)
                if not stat.S_ISREG(artifact.st_mode):
                    _LOG.warning("non-regular recovery entry retained: %s", name)
                names.append(name)
            return tuple(sorted(names))
        finally:
            os.close(private_fd)
    finally:
        os.close(output_fd)


@dataclass(frozen=True)
class ReconciliationReport:
    """Conservative reconciliation: only proved duplicate recovery links are removed."""

    removed: tuple[str, ...]
    retained: tuple[str, ...]


def reconcile_recovery_directory(directory: Path) -> ReconciliationReport:
    """Reclaim confirmed published recovery links, retaining everything uncertain.

    Only a name produced by our publisher and a regular inode with exactly one
    matching published recording are eligible. Unknown staging, quarantine,
    symlinks, and recovery-only inodes are never deleted. This is intended for
    single-owner startup, not simultaneous writers under the same UID.
    """
    output_fd = _open_checked_output_directory(directory)
    try:
        try:
            private_fd = os.open(
                _RECOVERY_NAME,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=output_fd,
            )
        except FileNotFoundError:
            return ReconciliationReport((), ())
        try:
            info = os.fstat(private_fd)
            if (
                info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_dev != os.fstat(output_fd).st_dev
            ):
                raise PermissionError("recovery directory ownership/mode/filesystem changed")

            recovery_entries = sorted(os.listdir(private_fd))
            reclaimable = (
                r"(?:recovery-[0-9a-f]{32}|\.brynse-stream-[A-Za-z0-9_-]+)"
                r"\.(?:mp3|aac|m4a)"
            )
            if not any(re.fullmatch(reclaimable, name) for name in recovery_entries):
                return ReconciliationReport((), tuple(recovery_entries))

            # Index only when a candidate can actually be reclaimed.
            published: dict[tuple[int, int], list[str]] = {}
            for name in os.listdir(output_fd):
                if name.startswith(".") or not name.endswith((".mp3", ".aac", ".m4a")):
                    continue
                try:
                    candidate = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(candidate.st_mode):
                    published.setdefault((candidate.st_dev, candidate.st_ino), []).append(name)

            removed: list[str] = []
            retained: list[str] = []
            # Durability of the final directory is established before touching
            # recovery, including an earlier uncertain final-directory fsync.
            try:
                os.fsync(output_fd)
            except OSError as exc:
                _LOG.warning("output durability unavailable; retaining recovery artifacts: %s", exc)
                return ReconciliationReport((), tuple(sorted(os.listdir(private_fd))))

            for name in recovery_entries:
                if not re.fullmatch(reclaimable, name):
                    retained.append(name)
                    continue
                try:
                    entry = os.stat(name, dir_fd=private_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                identity = (entry.st_dev, entry.st_ino)
                matches = published.get(identity, ())
                if not stat.S_ISREG(entry.st_mode) or entry.st_nlink < 2 or len(matches) != 1:
                    retained.append(name)
                    continue
                # Recheck the published inode by pinned fd before the private
                # rename+identity checked unlink. No pathname-based source unlink.
                try:
                    current = os.stat(matches[0], dir_fd=output_fd, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or (current.st_dev, current.st_ino) != identity
                    ):
                        retained.append(name)
                        continue
                    cleaned = _cleanup_verified_link(private_fd, name, identity)
                    os.fsync(private_fd)
                    if not cleaned:
                        _LOG.warning(
                            "recovery inode changed; quarantined artifact retained: %s", name
                        )
                        retained.append(name)
                        continue
                except OSError as exc:
                    _LOG.warning(
                        "retained recovery artifact %s/%s: %s",
                        directory / _RECOVERY_NAME,
                        name,
                        exc,
                    )
                    retained.append(name)
                    continue
                removed.append(name)
            return ReconciliationReport(tuple(removed), tuple(retained))
        finally:
            os.close(private_fd)
    finally:
        os.close(output_fd)


def ensure_private_recovery_directory(directory: Path) -> Path:
    """Return a validated owner-only staging directory."""
    output_fd = _open_checked_output_directory(directory)
    try:
        private_fd = _open_private_recovery_dir(output_fd)
        try:
            os.fsync(output_fd)
        finally:
            os.close(private_fd)
    finally:
        os.close(output_fd)
    return directory / _RECOVERY_NAME


def _cleanup_verified_link(
    private_fd: int,
    name: str,
    identity: tuple[int, int],
) -> bool:
    """Quarantine a private link and only delete the expected inode.

    Same-UID hostile processes remain outside the isolation guarantee.
    A mismatch is deliberately retained for manual reconciliation.
    """
    quarantine = f"cleanup-{secrets.token_hex(24)}"
    os.rename(
        name,
        quarantine,
        src_dir_fd=private_fd,
        dst_dir_fd=private_fd,
    )
    current = os.stat(
        quarantine,
        dir_fd=private_fd,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
        return False

    os.unlink(quarantine, dir_fd=private_fd)
    return True


def publish_private_file(
    *,
    temp_path: Path,
    directory: Path,
    stem: str,
    extension: str,
) -> Path:
    """Publish a completed private staging file by pinned descriptor.

    After confirmed publication, best-effort cleanup is confined to the
    owner-only staging directory. Failed publications retain their source.
    """
    directory = directory.resolve(strict=True)
    private_dir = ensure_private_recovery_directory(directory)

    if temp_path.parent.resolve() != private_dir:
        raise ValueError("source is not in the private staging directory")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    source_fd = os.open(temp_path, flags)
    try:
        source = os.fstat(source_fd)
        if not stat.S_ISREG(source.st_mode) or source.st_nlink != 1:
            raise ValueError("private source must be a single-link regular file")

        source_identity = (source.st_dev, source.st_ino)
        receipt = publish_fd_recoverably(
            source_fd=source_fd,
            directory=directory,
            stem=stem,
            extension=extension,
        )
    finally:
        os.close(source_fd)

    # Publication has completed. Errors here cannot invite retry.
    # Do not remove a pathname whose inode has been substituted.
    try:
        private_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        private_fd = os.open(private_dir, private_flags)
        try:
            for name in (temp_path.name, receipt.recovery_path.name):
                try:
                    _cleanup_verified_link(private_fd, name, source_identity)
                except OSError as exc:
                    _LOG.warning("retained recovery artifact %s: %s", private_dir / name, exc)
            try:
                os.fsync(private_fd)
            except OSError as exc:
                _LOG.warning("recovery cleanup durability unconfirmed at %s: %s", private_dir, exc)
        finally:
            os.close(private_fd)
    except OSError as exc:
        _LOG.warning("recovery cleanup unavailable at %s: %s", private_dir, exc)

    return receipt.published_path


@dataclass(frozen=True)
class OutputDirectoryIdentity:
    """Capture the resolved destination, refusing later mount/symlink swaps."""

    requested: Path
    resolved: Path
    device: int
    inode: int

    @classmethod
    def capture(cls, directory: Path) -> OutputDirectoryIdentity:
        resolved = directory.resolve(strict=True)
        current = resolved.stat()
        if not stat.S_ISDIR(current.st_mode):
            raise NotADirectoryError(str(resolved))
        return cls(directory, resolved, current.st_dev, current.st_ino)

    def verify(self) -> None:
        try:
            resolved = self.requested.resolve(strict=True)
            current = resolved.stat()
        except OSError as exc:
            raise OSError(errno.ESTALE, f"output directory unavailable: {self.requested}") from exc
        if resolved != self.resolved or (current.st_dev, current.st_ino) != (
            self.device,
            self.inode,
        ):
            raise OSError(
                errno.ESTALE,
                f"output directory changed since preflight: {self.requested}",
            )


def mkdir_durably(directory: Path) -> None:
    """Create missing output parents and sync each new entry's parent."""
    pending: list[Path] = []
    current = directory.absolute()
    while not current.exists():
        pending.append(current)
        if current.parent == current:
            raise FileNotFoundError(str(current))
        current = current.parent
    for child in reversed(pending):
        # Two recorders may create the same parent concurrently.
        child.mkdir(exist_ok=True)
        if not stat.S_ISDIR(child.lstat().st_mode):
            raise NotADirectoryError(f"output path is not a directory: {child}")
        parent_fd = os.open(
            child.parent.resolve(strict=True),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def preflight_publication_directory(directory: Path) -> Path:
    """Test the real publication mechanism before recording any stream bytes."""
    import tempfile

    directory = directory.resolve(strict=True)
    private = ensure_private_recovery_directory(directory)
    fd = -1
    path: Path | None = None
    target: Path | None = None
    output_probe: str | None = None
    output_fd = -1
    try:
        fd, temporary = tempfile.mkstemp(prefix=".brynse-probe-", dir=private)
        path = Path(temporary)
        os.write(fd, b"probe")
        os.fsync(fd)
        with os.scandir(private) as entries:
            # The probe is created within a validated 0700 directory.
            if not any(entry.name == path.name for entry in entries):
                raise OSError(errno.ENOENT, "publication probe disappeared")
        with os.fdopen(os.dup(fd), "rb") as held:
            private_fd = os.open(private, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                candidate = f".brynse-probe-link-{secrets.token_hex(12)}"
                _link_fd(held.fileno(), private_fd, candidate)
                target = private / candidate
                os.fsync(private_fd)
            finally:
                os.close(private_fd)

        # A successful recovery-only link is insufficient: the final output
        # directory can independently deny link creation (ACLs, permissions).
        output_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        output_probe = f".brynse-probe-out-{secrets.token_hex(16)}"
        _link_fd(fd, output_fd, output_probe)
        os.fsync(output_fd)
    finally:
        if output_fd >= 0:
            if output_probe is not None:
                # Do not unlink a name replaced by another inode.
                try:
                    candidate_stat = os.stat(output_probe, dir_fd=output_fd, follow_symlinks=False)
                    source_stat = os.fstat(fd)
                    if stat.S_ISREG(candidate_stat.st_mode) and (
                        candidate_stat.st_dev,
                        candidate_stat.st_ino,
                    ) == (source_stat.st_dev, source_stat.st_ino):
                        os.unlink(output_probe, dir_fd=output_fd)
                        os.fsync(output_fd)
                except OSError:
                    pass  # Retain ambiguous artifacts for safe reconciliation.
            os.close(output_fd)
        if fd >= 0:
            os.close(fd)
        if target is not None:
            target.unlink(missing_ok=True)
        if path is not None:
            path.unlink(missing_ok=True)
    return directory
