"""Chapter 02 reference runner: a GPU boundary without cache-pool policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_vllm.toy_model import KVCache, ToyAttentionLM


@dataclass(frozen=True)
class ReferenceStep:
    """One sampled CPU token plus the GPU-resident cache for the next step."""

    token_id: int
    cache: KVCache


class ReferenceModelRunner:
    """Own model invocation and sampling, but deliberately not cache allocation.

    This is the minimal boundary introduced in Chapter 02. A later chapter
    replaces its growing ``KVCache`` with a ``KVCachePool`` without changing
    Chapter 01's model or generation reference path.
    """

    def __init__(self, model: ToyAttentionLM) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def prefill(self, prompt_token_ids: list[int]) -> ReferenceStep:
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long).to(self.device)
        logits, cache = self.model(input_ids)
        return ReferenceStep(token_id=int(logits[:, -1].argmax(dim=-1).cpu().item()), cache=cache)

    @torch.inference_mode()
    def decode(self, input_token_id: int, cache: KVCache) -> ReferenceStep:
        input_ids = torch.tensor([[input_token_id]], dtype=torch.long).to(self.device)
        logits, next_cache = self.model(input_ids, cache)
        return ReferenceStep(
            token_id=int(logits[:, -1].argmax(dim=-1).cpu().item()), cache=next_cache
        )
