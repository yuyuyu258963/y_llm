"""Chapter 03 extension: execute a Chapter 01 model against fixed KV slots.

Keeping this code outside ``toy_model.py`` is intentional. The Chapter 01
reference model remains a complete, readable implementation on its own; this
module adds only the later storage optimization.
"""

from __future__ import annotations

from math import sqrt

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.toy_model import ToyAttentionLM


class PooledAttentionExecutor:
    """Adapt ``ToyAttentionLM`` to heterogeneous, preallocated KV slots."""

    def __init__(self, model: ToyAttentionLM) -> None:
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_pool: KVCachePool,
        slot_indices: torch.Tensor,
        cache_lengths: torch.Tensor,
        active_cache_tokens: int,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        batch_size, new_tokens = input_ids.shape
        if batch_size == 0:
            raise ValueError("an execution batch cannot be empty")
        if slot_indices.shape != (batch_size,) or cache_lengths.shape != (batch_size,):
            raise ValueError("one slot index and cache length are required per batch row")
        if input_ids.device != cache_pool.device:
            raise ValueError("input_ids and cache pool must share a device")
        if cache_pool.hidden_size != self.model.hidden_size:
            raise ValueError("cache pool hidden size must match model")
        if active_cache_tokens < 0 or active_cache_tokens > cache_pool.max_tokens:
            raise ValueError("active cache length is outside the pool capacity")
        if torch.any(cache_lengths < 0) or torch.any(
            cache_lengths + new_tokens > cache_pool.max_tokens
        ):
            raise ValueError("cache write would exceed the allocated slot")

        token_offsets = torch.arange(new_tokens, device=input_ids.device)
        positions = cache_lengths[:, None] + token_offsets[None, :]
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(positions)
        query = self.model.q_proj(hidden)
        new_key = self.model.k_proj(hidden)
        new_value = self.model.v_proj(hidden)

        # Read only the longest live history in this batch; a validity mask
        # hides unused tails for shorter sequences.
        past_key, past_value = cache_pool.read(slot_indices, active_cache_tokens)
        all_key = torch.cat((past_key, new_key), dim=1)
        all_value = torch.cat((past_value, new_value), dim=1)
        attention_scores = query @ all_key.transpose(-2, -1) / sqrt(self.model.hidden_size)

        past_positions = torch.arange(active_cache_tokens, device=input_ids.device)
        valid_past = past_positions[None, None, :] < cache_lengths[:, None, None]
        valid_new = token_offsets[None, None, :] <= token_offsets[None, :, None]
        valid_keys = torch.cat(
            (valid_past.expand(-1, new_tokens, -1), valid_new.expand(batch_size, -1, -1)),
            dim=-1,
        )
        attention = torch.softmax(attention_scores.masked_fill(~valid_keys, float("-inf")), dim=-1)
        context = attention @ all_value
        cache_pool.write(slot_indices, cache_lengths, new_key, new_value)
        return self.model.lm_head(self.model.output_proj(context))
