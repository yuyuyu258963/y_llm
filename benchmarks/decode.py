"""Reproducible toy-model comparison: full recomputation versus cache-backed decode.

Run from the repository root:
    python benchmarks/decode.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.device import require_cuda, synchronize
from mini_vllm.generation import greedy_generate_cached, greedy_generate_full
from mini_vllm.pooled_runner import PooledModelRunner
from mini_vllm.toy_model import ToyAttentionLM


def cached_generate(runner: PooledModelRunner, prompt: list[int], max_new_tokens: int) -> list[int]:
    slot = runner.cache_pool.acquire("benchmark")
    try:
        output = [runner.prefill(slot, prompt)]
        for _ in range(max_new_tokens - 1):
            output.append(runner.decode([slot], [len(prompt) + len(output) - 1], [output[-1]])[0])
        return output
    finally:
        runner.cache_pool.release("benchmark", slot)


def timed_ms(device: torch.device, function: object, repeats: int = 10) -> float:
    assert callable(function)
    function()  # warm-up CUDA context and kernels
    synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Events record on the current CUDA stream.  ``device`` is not a
        # stream argument; passing it here would silently select an invalid
        # ordinal on some PyTorch versions.
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    accelerator = require_cuda()
    torch.manual_seed(41)
    model = ToyAttentionLM(64, 64, 64).to(accelerator.device).eval()
    prompt = list(range(16))
    new_tokens = 16
    prompt_tensor = torch.tensor([prompt], device=accelerator.device)
    pool = KVCachePool(1, 64, 64, accelerator.device)
    runner = PooledModelRunner(model, pool)

    full = greedy_generate_full(model, prompt_tensor, new_tokens)[0, -new_tokens:].cpu().tolist()
    contiguous_cached = greedy_generate_cached(model, prompt_tensor, new_tokens)[0, -new_tokens:]
    pooled_cached = cached_generate(runner, prompt, new_tokens)
    assert full == contiguous_cached.cpu().tolist() == pooled_cached, (
        "benchmark inputs must preserve generation semantics"
    )

    full_ms = timed_ms(
        accelerator.device, lambda: greedy_generate_full(model, prompt_tensor, new_tokens)
    )
    contiguous_ms = timed_ms(
        accelerator.device, lambda: greedy_generate_cached(model, prompt_tensor, new_tokens)
    )
    pooled_ms = timed_ms(accelerator.device, lambda: cached_generate(runner, prompt, new_tokens))
    print(f"GPU: {accelerator.name}")
    print(f"full recomputation median: {full_ms:.3f} ms")
    print(f"contiguous cache median:   {contiguous_ms:.3f} ms ({full_ms / contiguous_ms:.2f}x)")
    print(f"engine cache-pool median:  {pooled_ms:.3f} ms ({full_ms / pooled_ms:.2f}x)")
    print("Interpret carefully: the engine path adds Python scheduling, slot gathers and D2H token handoff.")


if __name__ == "__main__":
    main()
