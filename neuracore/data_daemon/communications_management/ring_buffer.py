"""Byte-oriented ring buffer with overwrite-on-full semantics."""

from __future__ import annotations

import time


class RingBuffer:
    """Byte-oriented ring buffer with overwrite-on-full semantics.

    - Single writer (daemon _handle_data_chunk).
    - Single reader (message consumer).
    """

    def __init__(self, size: int = 1024):
        """Initialize the ring buffer."""
        self.size = size
        self.buffer = bytearray(size)
        self.write_pos = 0
        self.read_pos = 0
        self.used = 0

    def available(self) -> int:
        """Return number of bytes available to read."""
        return self.used

    def write(self, data: bytes) -> None:
        """Write data to the ring buffer.

        Blocks until enough space is available in the ring buffer if the data
        exceeds the available space.

        :param data: the data to write to the ring buffer
        :type data: bytes
        :raises ValueError: if the data exceeds the ring buffer size
        """
        if not data:
            return

        data_len = len(data)
        if data_len > self.size:
            raise ValueError("chunk exceeds ring buffer")

        # Block until enough space
        while data_len > self.size - self.used:
            time.sleep(0.001)

        end_space = self.size - self.write_pos
        if data_len <= end_space:
            self.buffer[self.write_pos : self.write_pos + data_len] = data
        else:
            first_part = end_space
            self.buffer[self.write_pos :] = data[:first_part]
            self.buffer[: data_len - first_part] = data[first_part:]

        self.write_pos = (self.write_pos + data_len) % self.size
        self.used += data_len

    def peek(self, length: int) -> bytes | None:
        """Return up to `length` bytes without advancing read_pos.

        Returns None if there are fewer than `length` bytes available.
        """
        if length > self.used:
            return None

        end_space = self.size - self.read_pos
        if length <= end_space:
            return bytes(self.buffer[self.read_pos : self.read_pos + length])

        # Wrap-around read
        first_part = end_space
        return bytes(self.buffer[self.read_pos :] + self.buffer[: length - first_part])

    def read(self, length: int) -> bytes | None:
        """Read and consume exactly `length` bytes.

        Returns None if there are fewer than `length` bytes available.
        """
        if length > self.used:
            return None

        end_space = self.size - self.read_pos
        if length <= end_space:
            data = bytes(self.buffer[self.read_pos : self.read_pos + length])
        else:
            first_part = end_space
            data = bytes(
                self.buffer[self.read_pos :] + self.buffer[: length - first_part]
            )

        self.read_pos = (self.read_pos + length) % self.size
        self.used -= length
        return data
