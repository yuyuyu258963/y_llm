"""Chapter 00: make the CPU/GPU boundary visible.

Run from the repository root:
    python lessons/00_cpu_gpu.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow this lesson to run directly without an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.device import require_cuda, synchronize


def main() -> None:
    accelerator = require_cuda()
    print(f"GPU: {accelerator.name}")

    # CPU owns request parsing, tokenization, scheduling and initial input creation.
    host_input = torch.randn((1024, 1024), device="cpu")
    print(f"1. request tensor lives on: {host_input.device}")

    weight = torch.randn((1024, 1024), device=accelerator.device)
    # CUDA context creation and cuBLAS setup make the very first operation unusual.
    # Warm up once, then time a steady-state request.
    warmup = host_input.to(accelerator.device) @ weight
    synchronize(accelerator.device)                   # Wait before trusting the elapsed time.

    start = time.perf_counter()
    device_input = host_input.to(accelerator.device)  # H2D: host -> device
    logits = device_input @ weight                    # GPU work is queued asynchronously.
    synchronize(accelerator.device)                   # Wait before trusting the elapsed time.
    gpu_elapsed_ms = (time.perf_counter() - start) * 1_000
    print(f"2. model tensors live on: {logits.device}; H2D + matmul: {gpu_elapsed_ms:.2f} ms")

    # A real engine transfers only the selected token, not the full logits tensor.
    next_token_gpu = logits[0].argmax()
    next_token_cpu = next_token_gpu.cpu()             # D2H: device -> host
    print(f"3. sampled token lives on: {next_token_cpu.device}; token id: {next_token_cpu.item()}")


if __name__ == "__main__":
    main()
