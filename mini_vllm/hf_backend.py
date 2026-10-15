"""Optional Chapter 09 bridge to a real Hugging Face causal language model.

This module is intentionally isolated from the toy-engine path. Importing it
does not download weights; ``HuggingFaceReference.load`` does, only when the
learner explicitly runs the real-model lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class HuggingFaceReference:
    """Native-Transformers reference path used before custom cache integration."""

    model_id: str
    tokenizer: Any
    model: Any
    device: torch.device

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "HuggingFaceReference":
        """Download/load an instruction model only when explicitly called."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real-model chapters need the optional dependency group: "
                'pip install -e ".[hf]"'
            ) from exc

        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # ``dtype`` is the current Transformers spelling.  ``torch_dtype``
        # keeps the optional learning track usable with its documented 4.37+
        # baseline, where the newer spelling did not exist yet.
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device).eval()
        return cls(model_id=model_id, tokenizer=tokenizer, model=model, device=device)

    @torch.inference_mode()
    def generate_chat(self, messages: list[dict[str, str]], max_new_tokens: int = 64) -> str:
        """Use the model's own chat template and ``generate`` as the oracle."""
        input_ids = self.chat_input_ids(messages)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_ids = output_ids[:, input_ids.size(1) :]
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]

    def chat_input_ids(self, messages: list[dict[str, str]]) -> torch.Tensor:
        """Apply a chat template across Transformers tensor-return variants.

        Older releases return the tensor itself; current releases may return a
        ``BatchEncoding`` even when no attention-mask field is requested.
        The runner intentionally consumes only token IDs for batch size one.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = encoded["input_ids"]
        return input_ids.to(self.device)
