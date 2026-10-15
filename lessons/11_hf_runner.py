"""Chapter 11: generate with the explicit HF reference runner."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.hf_backend import HuggingFaceReference
from mini_vllm.hf_runner import HuggingFaceReferenceRunner


def main() -> None:
    reference = HuggingFaceReference.load()
    input_ids = reference.tokenizer("用中文简短解释什么是 prefill。", return_tensors="pt").input_ids
    runner = HuggingFaceReferenceRunner(reference)
    step = runner.prefill(input_ids)
    generated = [step.token_id]
    for _ in range(31):
        step = runner.decode(generated[-1], step)
        generated.append(step.token_id)
    print(reference.tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
