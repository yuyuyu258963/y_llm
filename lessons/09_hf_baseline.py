"""Chapter 09: verify a real model with native Hugging Face generation.

This is the first command that downloads model weights. Install optional
dependencies first if necessary:
    pip install -e ".[hf]"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.hf_backend import HuggingFaceReference


def main() -> None:
    reference = HuggingFaceReference.load()
    answer = reference.generate_chat(
        [{"role": "user", "content": "用两句话解释 KV cache 的作用。"}], max_new_tokens=64
    )
    print(f"model: {reference.model_id}")
    print(f"device: {reference.device}")
    print(answer)


if __name__ == "__main__":
    main()
