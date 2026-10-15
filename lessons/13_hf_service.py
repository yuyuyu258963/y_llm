"""Chapter 13a: expose the real-model reference path over local HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_vllm.hf_backend import HuggingFaceReference
from mini_vllm.hf_service import HuggingFaceTextService, create_hf_http_server


def main() -> None:
    service = HuggingFaceTextService(HuggingFaceReference.load())
    server = create_hf_http_server(("127.0.0.1", 8001), service)
    print("Serving Qwen on http://127.0.0.1:8001")
    print('POST /v1/chat/completions with {"messages":[{"role":"user","content":"你好"}]}')
    server.serve_forever()


if __name__ == "__main__":
    main()
