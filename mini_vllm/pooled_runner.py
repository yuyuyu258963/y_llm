"""Chapter 03 runner: a storage-optimized extension of the reference runner."""

from __future__ import annotations

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.pooled_attention import PooledAttentionExecutor
from mini_vllm.toy_model import ToyAttentionLM


class PooledModelRunner:
    """Execute prefill/decode requests using the Chapter 03 fixed-slot pool."""

    def __init__(self, model: ToyAttentionLM, cache_pool: KVCachePool) -> None:
        if next(model.parameters()).device != cache_pool.device:
            raise ValueError("model and cache pool must be on the same GPU")
        self.model = model.eval()
        self.cache_pool = cache_pool
        self.device = cache_pool.device
        self.executor = PooledAttentionExecutor(model)

    @torch.inference_mode()
    def prefill(self, slot: int, prompt_token_ids: list[int]) -> int:
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long).to(self.device)
        slot_indices = torch.tensor([slot], dtype=torch.long, device=self.device)
        lengths = torch.zeros(1, dtype=torch.long, device=self.device)
        logits = self.executor.forward(
            input_ids, self.cache_pool, slot_indices, lengths, active_cache_tokens=0
        )
        return int(logits[:, -1].argmax(dim=-1).cpu().item())

    @torch.inference_mode()
    def decode(
        self, slots: list[int], cache_lengths: list[int], input_token_ids: list[int]
    ) -> list[int]:
        if not slots:
            return []
        if not (len(slots) == len(cache_lengths) == len(input_token_ids)):
            raise ValueError("each decode row needs a slot, length and input token")
        input_ids = torch.tensor(input_token_ids, dtype=torch.long).reshape(-1, 1).to(self.device)
        slot_indices = torch.tensor(slots, dtype=torch.long, device=self.device)
        lengths = torch.tensor(cache_lengths, dtype=torch.long, device=self.device)
        logits = self.executor.forward(
            input_ids,
            self.cache_pool,
            slot_indices,
            lengths,
            active_cache_tokens=max(cache_lengths),
        )
        return [int(token) for token in logits[:, -1].argmax(dim=-1).cpu().tolist()]
