"""A deliberately small causal attention model used to teach inference mechanics.

It is not a useful language model.  Its job is to make the prefill/decode and
KV-cache contracts small enough to inspect and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

@dataclass(frozen=True)
class KVCache:
    """Keys and values already computed for one batch of sequences.

    Layout is ``[batch, token_position, hidden_size]``.  Later chapters will
    replace this contiguous representation with paged, per-layer blocks.
    """

    key: torch.Tensor
    value: torch.Tensor

    @property
    def length(self) -> int:
        return self.key.size(1)


class ToyAttentionLM(nn.Module):
    """One-head causal decoder with an explicit KV-cache interface."""

    def __init__(self, vocab_size: int, hidden_size: int, max_positions: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_positions = max_positions
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        """Run prefill (many tokens) or decode (normally one token).

        Args:
            input_ids: Integer token IDs with shape ``[batch, new_tokens]``.
            cache: Keys and values from tokens before ``input_ids``.

        Returns:
            Logits for each new token and a cache containing both old and new
            tokens.  The caller owns the returned cache; this method never
            mutates a cache in place.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")

        batch_size, new_tokens = input_ids.shape
        past_tokens = 0 if cache is None else cache.length
        if past_tokens + new_tokens > self.max_positions:
            raise ValueError("sequence is longer than max_positions")
        if cache is not None:
            if cache.key.shape != cache.value.shape:
                raise ValueError("cache key and value shapes must match")
            if cache.key.size(0) != batch_size:
                raise ValueError("cache batch size must match input_ids")
            if cache.key.size(2) != self.hidden_size:
                raise ValueError("cache hidden size must match model")
            if cache.key.device != input_ids.device:
                raise ValueError("cache and input_ids must be on the same device")

        positions = torch.arange(
            past_tokens, past_tokens + new_tokens, device=input_ids.device
        )
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        query = self.q_proj(hidden)
        new_key = self.k_proj(hidden)
        new_value = self.v_proj(hidden)

        key = new_key if cache is None else torch.cat((cache.key, new_key), dim=1)
        value = new_value if cache is None else torch.cat((cache.value, new_value), dim=1)

        # A query at absolute position p may attend to keys 0 through p.
        attention_scores = query @ key.transpose(-2, -1) / sqrt(self.hidden_size)
        key_positions = torch.arange(key.size(1), device=input_ids.device)
        query_positions = past_tokens + torch.arange(new_tokens, device=input_ids.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        attention = torch.softmax(attention_scores, dim=-1)
        context = attention @ value
        logits = self.lm_head(self.output_proj(context))
        return logits, KVCache(key=key, value=value)

