"""Exclusive, collision-safe publication of completed recording files."""

from __future__ import annotations

_MAX_FILENAME_BYTES = 240


def _utf8_prefix(value: str, limit: int) -> str:
    """Truncate text without splitting any UTF-8 character."""
    result: list[str] = []
    remaining = limit
    for char in value:
        size = len(char.encode("utf-8"))
        if size > remaining:
            break
        result.append(char)
        remaining -= size
    return "".join(result)


def _completed_file_mode() -> int:
    """Honor the Linux process umask without changing it globally."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    mask = int(line.split(":", 1)[1].strip(), 8)
                    if 0 <= mask <= 0o777:
                        return 0o666 & ~mask
    except (OSError, ValueError):
        pass
    # Fail closed when the umask is not observable.
    return 0o600
