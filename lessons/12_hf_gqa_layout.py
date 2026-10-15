"""Chapter 12: inspect Qwen's real GQA K/V cache before introducing blocks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.hf_backend import HuggingFaceReference
from mini_vllm.hf_cache import describe_gqa_cache, summarize_past_key_values
from mini_vllm.hf_runner import HuggingFaceReferenceRunner


def main() -> None:
    reference = HuggingFaceReference.load()
    input_ids = reference.tokenizer("inspect this cache", return_tensors="pt").input_ids
    step = HuggingFaceReferenceRunner(reference).prefill(input_ids)
    layout = describe_gqa_cache(reference.model.config, next(reference.model.parameters()).dtype)
    layer_zero = summarize_past_key_values(step.past_key_values)[0]
    print(f"layers={layout.layers}, Q heads={layout.query_heads}, KV heads={layout.kv_heads}")
    print(f"one sequence adds {layout.bytes_per_token_per_sequence} bytes of K/V per token")
    print(f"layer 0: K={layer_zero.key_shape}, V={layer_zero.value_shape}, {layer_zero.dtype}")


if __name__ == "__main__":
    main()
