"""Small standard-library HTTP surface around the single-threaded Engine."""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mini_vllm.engine import Engine
from mini_vllm.requests import Request


class InferenceService:
    """Thread-safe request facade; all actual GPU work still runs in Engine.step."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._lock = threading.RLock()
        self._events: dict[str, deque[int]] = defaultdict(deque)

    def submit(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or uuid.uuid4().hex
        request = Request(request_id, prompt_token_ids, max_new_tokens, eos_token_id)
        with self._lock:
            self.engine.add_request(request)
        return request_id

    def step(self) -> None:
        with self._lock:
            for request_id, token in self.engine.step():
                self._events[request_id].append(token)

    def poll(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            if request_id not in self.engine.requests:
                raise KeyError(request_id)
            request = self.engine.requests[request_id]
            tokens = list(self._events[request_id])
            self._events[request_id].clear()
            return {
                "id": request_id,
                "tokens": tokens,
                "status": request.status.value,
                "finished": request.is_terminal,
                "error": request.error,
            }

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self.engine.cancel(request_id)

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "waiting_requests": len(self.engine.waiting),
                "running_requests": len(self.engine.running),
                "cache_slots_used": self.engine.cache_pool.used_slots,
                "cache_slots_available": self.engine.cache_pool.available_slots,
            }

    def complete(self, prompt_token_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
        """Convenience API for a non-streaming completion response."""
        request_id = self.submit(prompt_token_ids, max_new_tokens)
        output: list[int] = []
        while True:
            self.step()
            update = self.poll(request_id)
            output.extend(update["tokens"])
            if update["finished"]:
                return {"id": request_id, "tokens": output, "status": update["status"]}


def create_http_server(address: tuple[str, int], service: InferenceService) -> ThreadingHTTPServer:
    """Expose JSON completion, polling, cancellation, health and metrics endpoints."""

    class Handler(BaseHTTPRequestHandler):
        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length))

        def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - required stdlib handler name
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/metrics":
                self._send_json(HTTPStatus.OK, service.metrics())
            elif self.path.startswith("/v1/requests/"):
                request_id = self.path.rsplit("/", 1)[-1]
                try:
                    self._send_json(HTTPStatus.OK, service.poll(request_id))
                except KeyError:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown request"})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown path"})

        def do_POST(self) -> None:  # noqa: N802 - required stdlib handler name
            try:
                body = self._read_json()
                if self.path == "/v1/completions":
                    prompt = body["prompt_token_ids"]
                    max_tokens = body["max_new_tokens"]
                    if body.get("stream"):
                        self._stream(prompt, max_tokens)
                    else:
                        self._send_json(HTTPStatus.OK, service.complete(prompt, max_tokens))
                elif self.path.startswith("/v1/requests/") and self.path.endswith("/cancel"):
                    request_id = self.path.split("/")[-2]
                    service.cancel(request_id)
                    self._send_json(HTTPStatus.OK, {"id": request_id, "status": "cancelled"})
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown path"})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def _stream(self, prompt: list[int], max_tokens: int) -> None:
            request_id = service.submit(prompt, max_tokens)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            while True:
                service.step()
                update = service.poll(request_id)
                for token in update["tokens"]:
                    self.wfile.write(json.dumps({"id": request_id, "token": token}).encode() + b"\n")
                    self.wfile.flush()
                if update["finished"]:
                    self.wfile.write(json.dumps({"id": request_id, "done": True}).encode() + b"\n")
                    self.wfile.flush()
                    return

        def log_message(self, format: str, *args: object) -> None:
            """Keep the teaching server quiet; its metrics are the observability path."""

    return ThreadingHTTPServer(address, Handler)
