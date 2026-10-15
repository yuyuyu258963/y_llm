"""Reference decoding loops before batching and scheduling are introduced."""

from __future__ import annotations

import torch

from mini_vllm.toy_model import ToyAttentionLM


@torch.inference_mode()
def greedy_generate_full(
    model: ToyAttentionLM, prompt_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    """Correct but wasteful decoding: recompute every token on every step."""
    generated = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return generated


@torch.inference_mode()
def greedy_generate_cached(
    model: ToyAttentionLM, prompt_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    """The same greedy policy while calculating each key/value exactly once."""
    generated = prompt_ids.clone()
    logits, cache = model(prompt_ids)  # prefill
    for step in range(max_new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if step + 1 < max_new_tokens:
            logits, cache = model(next_token, cache)  # decode one new token
    return generated
