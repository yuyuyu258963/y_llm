"""Chapter 01: prefill and decode must produce the same token sequence.

Run from the repository root:
    python lessons/01_prefill_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.device import require_cuda
from mini_vllm.generation import greedy_generate_cached, greedy_generate_full
from mini_vllm.toy_model import ToyAttentionLM


def main() -> None:
    accelerator = require_cuda()
    torch.manual_seed(7)
    model = ToyAttentionLM(vocab_size=32, hidden_size=16, max_positions=32).to(
        accelerator.device
    ).eval()
    prompt = torch.tensor([[3, 1, 4, 1]], device=accelerator.device)

    prefill_logits, cache = model(prompt)
    next_token = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)
    decode_logits, next_cache = model(next_token, cache)
    print(f"prefill: input shape={tuple(prompt.shape)}, cache length={cache.length}")
    print(f"decode:  input shape={tuple(next_token.shape)}, cache length={next_cache.length}")

    uncached = greedy_generate_full(model, prompt, max_new_tokens=6)
    cached = greedy_generate_cached(model, prompt, max_new_tokens=6)
    print(f"full recomputation: {uncached.cpu().tolist()}")
    print(f"with KV cache:      {cached.cpu().tolist()}")
    torch.testing.assert_close(uncached, cached)
    print("PASS: cached decoding is numerically equivalent to full recomputation.")


if __name__ == "__main__":
    main()
