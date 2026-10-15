"""Chapter 13a: minimal real-model service without changing the toy Engine."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from mini_vllm.hf_backend import HuggingFaceReference
from mini_vllm.hf_runner import HuggingFaceReferenceRunner


class HuggingFaceTextService:
    """Single-request greedy generation over the Chapter 11 reference runner."""

    def __init__(self, reference: HuggingFaceReference) -> None:
        self.reference = reference
        self.runner = HuggingFaceReferenceRunner(reference)

    def complete(self, messages: list[dict[str, str]], max_new_tokens: int = 64) -> dict[str, Any]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        input_ids = self.reference.chat_input_ids(messages)
        step = self.runner.prefill(input_ids)
        generated = [step.token_id]
        eos_token_ids = self._eos_token_ids()
        while len(generated) < max_new_tokens and generated[-1] not in eos_token_ids:
            step = self.runner.decode(generated[-1], step)
            generated.append(step.token_id)
        return {
            "model": self.reference.model_id,
            "token_ids": generated,
            "text": self.reference.tokenizer.decode(generated, skip_special_tokens=True),
            "finish_reason": "eos" if generated[-1] in eos_token_ids else "length",
        }

    def _eos_token_ids(self) -> set[int]:
        eos = self.reference.tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return {int(token) for token in eos}


def create_hf_http_server(
    address: tuple[str, int], service: HuggingFaceTextService
) -> ThreadingHTTPServer:
    """Expose ``POST /v1/chat/completions`` for the real-model teaching track."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib interface name
            if self.path != "/v1/chat/completions":
                self._send(HTTPStatus.NOT_FOUND, {"error": "unknown path"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                self._send(
                    HTTPStatus.OK,
                    service.complete(body["messages"], int(body.get("max_new_tokens", 64))),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_GET(self) -> None:  # noqa: N802 - stdlib interface name
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok", "model": service.reference.model_id})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "unknown path"})

        def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer(address, Handler)
