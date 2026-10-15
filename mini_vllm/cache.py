"""GPU-resident, fixed-slot KV cache storage for the teaching engine."""

from __future__ import annotations

from collections import deque

import torch


class CacheExhaustedError(RuntimeError):
    """Raised when an engine accepts more live sequences than cache slots."""


class KVCachePool:
    """Preallocated contiguous KV storage with explicit slot ownership.

    This is intentionally simpler than vLLM's paged cache: every live
    sequence owns one fixed-size slot.  It removes repeated ``torch.cat``
    allocations and provides the resource lifecycle that the scheduler needs.
    """

    def __init__(
        self,
        num_slots: int,
        max_tokens: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(num_slots, max_tokens, hidden_size) <= 0:
            raise ValueError("num_slots, max_tokens and hidden_size must be positive")
        self.num_slots = num_slots
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.device = device
        self.key = torch.zeros((num_slots, max_tokens, hidden_size), device=device, dtype=dtype)
        self.value = torch.zeros_like(self.key)
        self._free_slots: deque[int] = deque(range(num_slots))
        self._owners: dict[int, str] = {}

    @property
    def available_slots(self) -> int:
        return len(self._free_slots)

    @property
    def used_slots(self) -> int:
        return len(self._owners)

    @property
    def allocated_bytes(self) -> int:
        return self.key.numel() * self.key.element_size() * 2

    def acquire(self, owner_id: str) -> int:
        if owner_id in self._owners.values():
            raise ValueError(f"owner {owner_id!r} already owns a cache slot")
        if not self._free_slots:
            raise CacheExhaustedError("no KV cache slot is available")
        slot = self._free_slots.popleft()
        self._owners[slot] = owner_id
        return slot

    def release(self, owner_id: str, slot: int) -> None:
        if self._owners.get(slot) != owner_id:
            raise ValueError("only the current owner may release a cache slot")
        del self._owners[slot]
        self._free_slots.append(slot)

    def read(
        self, slot_indices: torch.Tensor, active_tokens: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slot_indices.device != self.device:
            raise ValueError("slot indices must be on the cache device")
        active_tokens = self.max_tokens if active_tokens is None else active_tokens
        if not 0 <= active_tokens <= self.max_tokens:
            raise ValueError("active token count is outside this cache slot")
        return self.key[slot_indices, :active_tokens], self.value[slot_indices, :active_tokens]

    def write(
        self,
        slot_indices: torch.Tensor,
        start_positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write a batch of new K/V values at each row's next token position."""
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must be matching [batch, tokens, hidden] tensors")
        batch_size, new_tokens, hidden_size = key.shape
        if hidden_size != self.hidden_size:
            raise ValueError("K/V hidden size must match the pool")
        if slot_indices.shape != (batch_size,) or start_positions.shape != (batch_size,):
            raise ValueError("slot indices and positions must match the K/V batch")
        if key.device != self.device or value.device != self.device:
            raise ValueError("K/V tensors must remain on the cache device")
        offsets = torch.arange(new_tokens, device=self.device)
        positions = start_positions[:, None] + offsets[None, :]
        self.key[slot_indices[:, None], positions] = key
        self.value[slot_indices[:, None], positions] = value
