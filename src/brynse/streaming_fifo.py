"""Async readers for Unix FIFO inputs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class AsyncFifoReader:
    """One FIFO connected to an asyncio StreamReader."""

    reader: asyncio.StreamReader
    transport: asyncio.ReadTransport
    pipe: BinaryIO

    def close(self) -> None:
        self.transport.close()
        self.pipe.close()


async def open_fifo_reader(path: Path) -> AsyncFifoReader:
    """Open a Unix FIFO and expose it as an asyncio StreamReader."""

    if not path.exists():
        raise FileNotFoundError(path)

    loop = asyncio.get_running_loop()

    # FIFO open is intentionally moved out of the event loop because opening
    # the read side blocks until a writer connects.
    pipe = await asyncio.to_thread(
        open,
        path,
        "rb",
        buffering=0,
    )

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)

    transport, _ = await loop.connect_read_pipe(
        lambda: protocol,
        pipe,
    )

    return AsyncFifoReader(
        reader=reader,
        transport=transport,
        pipe=pipe,
    )
