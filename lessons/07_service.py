"""Chapter 07: start a minimal HTTP inference service using the toy engine."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.device import require_cuda
from mini_vllm.engine import Engine
from mini_vllm.pooled_runner import PooledModelRunner
from mini_vllm.scheduler import Scheduler
from mini_vllm.service import InferenceService, create_http_server
from mini_vllm.toy_model import ToyAttentionLM


def main() -> None:
    accelerator = require_cuda()
    torch.manual_seed(37)
    model = ToyAttentionLM(32, 16, 32).to(accelerator.device).eval()
    pool = KVCachePool(4, 32, 16, accelerator.device)
    service = InferenceService(Engine(PooledModelRunner(model, pool), pool, Scheduler(16)))
    server = create_http_server(("127.0.0.1", 8000), service)
    print("Serving on http://127.0.0.1:8000")
    print('Try: Invoke-RestMethod http://127.0.0.1:8000/health')
    print('Then: Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/completions -ContentType application/json -Body \'{"prompt_token_ids":[1,2],"max_new_tokens":3}\'')
    server.serve_forever()


if __name__ == "__main__":
    main()
