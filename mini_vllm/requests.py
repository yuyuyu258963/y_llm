"""Pure-Python request lifecycle: deliberately independent of CUDA tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATUSES = {RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED}


@dataclass
class Request:
    """CPU-side state for one greedy-generation request."""

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None
    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    cache_slot: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("a request prompt must contain at least one token")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    @property
    def cache_length(self) -> int:
        """Tokens already represented in KV cache (the newest output is not yet)."""
        return len(self.prompt_token_ids) + max(0, len(self.generated_token_ids) - 1)

    @property
    def next_input_token_id(self) -> int:
        if not self.generated_token_ids:
            raise RuntimeError("prefill, not decode, is required before the first generated token")
        return self.generated_token_ids[-1]

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def start(self, cache_slot: int) -> None:
        if self.status != RequestStatus.WAITING:
            raise RuntimeError("only a waiting request may start")
        self.status = RequestStatus.RUNNING
        self.cache_slot = cache_slot

    def accept_token(self, token_id: int) -> None:
        if self.status != RequestStatus.RUNNING:
            raise RuntimeError("only a running request may accept a token")
        self.generated_token_ids.append(token_id)
        if token_id == self.eos_token_id or len(self.generated_token_ids) >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED

    def cancel(self) -> None:
        if self.is_terminal:
            return
        self.status = RequestStatus.CANCELLED

    def fail(self, message: str) -> None:
        if self.is_terminal:
            return
        self.status = RequestStatus.FAILED
        self.error = message
