"""Chapter 10: cached decode logits must match full-context logits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_vllm.hf_backend import HuggingFaceReference
from mini_vllm.hf_cache import summarize_past_key_values
from mini_vllm.hf_runner import HuggingFaceReferenceRunner


def main() -> None:
    # The default teaching service uses BF16 to conserve VRAM.  This
    # correctness checkpoint uses FP32 instead: cached decode and a full
    # forward have different matrix shapes, so BF16 accumulation can differ
    # by roughly 1e-1 even when the token sequence is identical.
    reference = HuggingFaceReference.load(dtype=torch.float32)
    input_ids = reference.tokenizer("KV cache makes decode incremental.", return_tensors="pt").input_ids
    runner = HuggingFaceReferenceRunner(reference)
    prefill = runner.prefill(input_ids)
    cached = runner.decode(prefill.token_id, prefill)

    full_ids = torch.cat(
        (input_ids.to(reference.device), torch.tensor([[prefill.token_id]], device=reference.device)),
        dim=1,
    )
    full_logits = reference.model(full_ids, use_cache=False).logits[:, -1]
    torch.testing.assert_close(cached.logits[:, -1], full_logits, rtol=1e-3, atol=1e-3)
    first_layer = summarize_past_key_values(prefill.past_key_values)[0]
    print(
        "PASS: cached and full logits match; "
        f"layer 0 K={first_layer.key_shape}, {first_layer.dtype}, {first_layer.device}"
    )


if __name__ == "__main__":
    main()
