"""Chapter 05: different requests enter and leave a decode batch over time."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.device import require_cuda
from mini_vllm.engine import Engine
from mini_vllm.requests import Request
from mini_vllm.pooled_runner import PooledModelRunner
from mini_vllm.scheduler import Scheduler
from mini_vllm.toy_model import ToyAttentionLM


def main() -> None:
    accelerator = require_cuda()
    torch.manual_seed(29)
    model = ToyAttentionLM(32, 16, 32).to(accelerator.device).eval()
    pool = KVCachePool(4, 32, 16, accelerator.device)
    engine = Engine(PooledModelRunner(model, pool), pool, Scheduler(max_batched_tokens=8))
    first = Request("A", [1, 2], max_new_tokens=3)
    second = Request("B", [3, 4, 5], max_new_tokens=2)
    engine.add_request(first)
    engine.add_request(second)

    step = 0
    while not (first.is_terminal and second.is_terminal):
        events = engine.step()
        print(
            f"step {step}: emitted={events}; "
            f"A={first.status}/{first.generated_token_ids}; "
            f"B={second.status}/{second.generated_token_ids}; "
            f"live_slots={pool.used_slots}"
        )
        step += 1


if __name__ == "__main__":
    main()
