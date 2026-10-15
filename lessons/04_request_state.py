"""Chapter 04: request lifecycle is a CPU-only state machine."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.requests import Request


def main() -> None:
    request = Request("demo", prompt_token_ids=[1, 2, 3], max_new_tokens=2)
    print(f"new request: {request.status}")
    request.start(cache_slot=0)
    request.accept_token(8)  # returned by prefill; it is not in cache yet
    print(f"after prefill: {request.status}, cache_length={request.cache_length}, output={request.generated_token_ids}")
    request.accept_token(5)  # returned by one decode step
    print(f"after decode: {request.status}, cache_length={request.cache_length}, output={request.generated_token_ids}")


if __name__ == "__main__":
    main()
