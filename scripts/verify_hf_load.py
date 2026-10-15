"""Staged diagnostic for the optional real-model track.

It is intentionally separate from lessons: the printed checkpoints make slow
imports, downloads, weight loading and GPU placement distinguishable.
"""

from __future__ import annotations

import torch


def main() -> None:
    print("torch-ready", flush=True)
    import transformers

    print(f"transformers-ready {transformers.__version__}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print("tokenizer-start", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("tokenizer-ready", flush=True)
    print("model-start", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    print("model-ready", flush=True)
    model.to("cuda:0").eval()
    print(f"cuda-ready {tokenizer.__class__.__name__}", flush=True)


if __name__ == "__main__":
    main()
