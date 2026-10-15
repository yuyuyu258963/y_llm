"""Logical-to-physical token block mapping, the heart of paged KV memory."""

from __future__ import annotations

from collections import deque
from math import ceil


class BlockExhaustedError(RuntimeError):
    """Raised when no physical KV block remains for a live sequence."""


class BlockManager:
    """Own fixed-size physical blocks and per-sequence logical block tables.

    This chapter deliberately keeps only metadata on the CPU.  The next cache
    representation can use the same table to gather GPU K/V blocks.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._tables: dict[str, list[int]] = {}
        self._owners: dict[int, str] = {}

    @property
    def free_block_count(self) -> int:
        return len(self._free)

    def allocate_for_tokens(self, sequence_id: str, token_count: int) -> list[int]:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        required_blocks = ceil(token_count / self.block_size)
        return self._ensure_block_count(sequence_id, required_blocks)

    def append_token(self, sequence_id: str, position: int) -> tuple[int, int]:
        """Ensure storage for a logical position and return (physical, offset)."""
        if position < 0:
            raise ValueError("position cannot be negative")
        table = self._ensure_block_count(sequence_id, position // self.block_size + 1)
        return table[position // self.block_size], position % self.block_size

    def table(self, sequence_id: str) -> tuple[int, ...]:
        return tuple(self._tables.get(sequence_id, ()))

    def release(self, sequence_id: str) -> None:
        for block in self._tables.pop(sequence_id, []):
            if self._owners.pop(block) != sequence_id:
                raise RuntimeError("block ownership is corrupt")
            self._free.append(block)

    def _ensure_block_count(self, sequence_id: str, required_blocks: int) -> list[int]:
        table = self._tables.setdefault(sequence_id, [])
        missing = required_blocks - len(table)
        if missing <= 0:
            return table
        if missing > len(self._free):
            raise BlockExhaustedError("not enough physical blocks for this sequence")
        for _ in range(missing):
            block = self._free.popleft()
            table.append(block)
            self._owners[block] = sequence_id
        return table
