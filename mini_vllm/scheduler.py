"""A deterministic continuous-batching policy, independent of model details."""

from __future__ import annotations

from dataclasses import dataclass

from mini_vllm.requests import Request


@dataclass(frozen=True)
class Schedule:
    decode: tuple[Request, ...]
    prefill: tuple[Request, ...]


class Scheduler:
    """Spend a per-step token budget on old decode work, then new prefills."""

    def __init__(self, max_batched_tokens: int) -> None:
        if max_batched_tokens <= 0:
            raise ValueError("max_batched_tokens must be positive")
        self.max_batched_tokens = max_batched_tokens
        self._decode_cursor = 0

    def schedule(
        self, waiting: list[Request], running: list[Request], free_slots: int
    ) -> Schedule:
        budget = self.max_batched_tokens
        decode: list[Request] = []
        if running:
            start = self._decode_cursor % len(running)
            ordered = running[start:] + running[:start]
            decode = ordered[:budget]
            budget -= len(decode)
            self._decode_cursor = (start + len(decode)) % len(running)

        prefill: list[Request] = []
        for request in waiting:
            prompt_tokens = len(request.prompt_token_ids)
            if len(prefill) == free_slots or prompt_tokens > budget:
                break
            prefill.append(request)
            budget -= prompt_tokens
        return Schedule(decode=tuple(decode), prefill=tuple(prefill))
