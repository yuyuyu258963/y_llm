"""Chapter 03: inspect preallocated GPU KV slots and their ownership."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.device import require_cuda


def main() -> None:
    accelerator = require_cuda()
    pool = KVCachePool(num_slots=3, max_tokens=32, hidden_size=16, device=accelerator.device)
    first = pool.acquire("first-request")
    second = pool.acquire("second-request")
    print(f"K/V layout: {tuple(pool.key.shape)} on {pool.key.device}")
    print(f"allocated K+V memory: {pool.allocated_bytes} bytes")
    print(f"owners acquired slots: first={first}, second={second}; available={pool.available_slots}")
    pool.release("first-request", first)
    print(f"after first request finishes: available={pool.available_slots}, used={pool.used_slots}")
    pool.release("second-request", second)


if __name__ == "__main__":
    main()
