"""Single-threaded engine coordinating state, scheduling, cache and runner."""

from __future__ import annotations

from collections import deque

from mini_vllm.cache import KVCachePool
from mini_vllm.requests import Request, RequestStatus
from mini_vllm.pooled_runner import PooledModelRunner
from mini_vllm.scheduler import Scheduler


class Engine:
    """A small, deterministic version of the inference-engine control loop."""

    def __init__(
        self, runner: PooledModelRunner, cache_pool: KVCachePool, scheduler: Scheduler
    ) -> None:
        self.runner = runner
        self.cache_pool = cache_pool
        self.scheduler = scheduler
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
        self.requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request id {request.request_id!r}")
        if len(request.prompt_token_ids) > self.scheduler.max_batched_tokens:
            raise ValueError("prompt is larger than the scheduler's per-step token budget")
        # The last sampled token need not be cached, because no later decode
        # consumes it.  Every token before that does require one cache entry.
        required_cache_tokens = len(request.prompt_token_ids) + request.max_new_tokens - 1
        if required_cache_tokens > self.cache_pool.max_tokens:
            raise ValueError("prompt plus generation budget exceeds KV cache capacity")
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def cancel(self, request_id: str) -> None:
        request = self.requests[request_id]
        request.cancel()
        self._release_if_terminal(request)

    def step(self) -> list[tuple[str, int]]:
        """Run at most one model action per request and return emitted tokens."""
        self._discard_terminal_waiting()
        plan = self.scheduler.schedule(
            list(self.waiting), list(self.running.values()), self.cache_pool.available_slots
        )
        emitted: list[tuple[str, int]] = []

        # Old requests decode as a real heterogeneous GPU batch.
        decode_requests = list(plan.decode)
        if decode_requests:
            tokens = self.runner.decode(
                [self._require_slot(request) for request in decode_requests],
                [request.cache_length for request in decode_requests],
                [request.next_input_token_id for request in decode_requests],
            )
            for request, token in zip(decode_requests, tokens, strict=True):
                request.accept_token(token)
                emitted.append((request.request_id, token))
                self._release_if_terminal(request)

        # New prompts run as individual prefills.  They join the decode batch
        # from the next scheduler step, so no request advances twice per step.
        for request in plan.prefill:
            self.waiting.remove(request)
            slot = self.cache_pool.acquire(request.request_id)
            request.start(slot)
            try:
                token = self.runner.prefill(slot, request.prompt_token_ids)
                request.accept_token(token)
            except Exception as exc:
                request.fail(str(exc))
            emitted.extend((request.request_id, token) for token in request.generated_token_ids[-1:])
            if request.status == RequestStatus.RUNNING:
                self.running[request.request_id] = request
            self._release_if_terminal(request)
        return emitted

    def _discard_terminal_waiting(self) -> None:
        self.waiting = deque(request for request in self.waiting if not request.is_terminal)

    def _require_slot(self, request: Request) -> int:
        if request.cache_slot is None:
            raise RuntimeError("running request has no KV cache slot")
        return request.cache_slot

    def _release_if_terminal(self, request: Request) -> None:
        if not request.is_terminal or request.cache_slot is None:
            return
        self.cache_pool.release(request.request_id, request.cache_slot)
        request.cache_slot = None
        self.running.pop(request.request_id, None)
