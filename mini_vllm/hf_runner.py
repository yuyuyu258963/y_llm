"""Chapters 10-11: explicit prefill/decode around native Transformers cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mini_vllm.hf_backend import HuggingFaceReference


@dataclass(frozen=True)
class HuggingFaceStep:
    """The CPU-visible next token and GPU-resident native cache state."""

    token_id: int
    logits: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor


class HuggingFaceReferenceRunner:
    """Greedy native-cache runner used as a bridge, not a custom kernel."""

    def __init__(self, reference: HuggingFaceReference) -> None:
        self.reference = reference
        self.model = reference.model
        self.device = reference.device

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> HuggingFaceStep:
        """Run the full prompt once and retain Transformers' native cache."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        return self._step_from_output(output, attention_mask)

    @torch.inference_mode()
    def decode(self, input_token_id: int, previous: HuggingFaceStep) -> HuggingFaceStep:
        """Feed exactly one token and extend the cache by one position."""
        input_ids = torch.tensor([[input_token_id]], dtype=torch.long, device=self.device)
        attention_mask = torch.cat(
            (previous.attention_mask, torch.ones_like(input_ids, device=self.device)), dim=1
        )
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=previous.past_key_values,
            use_cache=True,
        )
        return self._step_from_output(output, attention_mask)

    def _step_from_output(self, output: Any, attention_mask: torch.Tensor) -> HuggingFaceStep:
        token = int(output.logits[:, -1].argmax(dim=-1).cpu().item())
        return HuggingFaceStep(
            token_id=token,
            logits=output.logits,
            past_key_values=output.past_key_values,
            attention_mask=attention_mask,
        )
