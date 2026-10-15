"""Chapter 02: ReferenceModelRunner is the CPU-plan/GPU-execution boundary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.device import require_cuda
from mini_vllm.generation import greedy_generate_full
from mini_vllm.runner import ReferenceModelRunner
from mini_vllm.toy_model import ToyAttentionLM


def main() -> None:
    accelerator = require_cuda()
    torch.manual_seed(23)
    model = ToyAttentionLM(32, 16, 32).to(accelerator.device).eval()
    runner = ReferenceModelRunner(model)

    # The scheduler would own this ordinary Python list; it contains no CUDA tensors.
    prompt = [3, 1, 4]
    first_step = runner.prefill(prompt)
    second_step = runner.decode(first_step.token_id, first_step.cache)

    reference = greedy_generate_full(
        model, torch.tensor([prompt], device=accelerator.device), max_new_tokens=2
    )[0, -2:].cpu().tolist()
    print(f"CPU execution plan: prompt={prompt}")
    print(f"reference KV cache stays on: {first_step.cache.key.device}")
    print(f"Runner emitted: {[first_step.token_id, second_step.token_id]}")
    print(f"Reference emitted: {reference}")
    assert [first_step.token_id, second_step.token_id] == reference
    print("PASS: ReferenceModelRunner preserved the Chapter 01 generation semantics.")


if __name__ == "__main__":
    main()
