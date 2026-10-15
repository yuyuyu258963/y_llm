import unittest
import json
import threading
from urllib.request import Request as URLRequest
from urllib.request import urlopen
from urllib.error import URLError

import torch

from mini_vllm.cache import KVCachePool
from mini_vllm.engine import Engine
from mini_vllm.generation import greedy_generate_full
from mini_vllm.requests import Request, RequestStatus
from mini_vllm.pooled_runner import PooledModelRunner
from mini_vllm.scheduler import Scheduler
from mini_vllm.service import InferenceService, create_http_server
from mini_vllm.toy_model import ToyAttentionLM


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class CacheAndRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.device = torch.device("cuda:0")
        self.model = ToyAttentionLM(32, 16, 32).to(self.device).eval()
        self.pool = KVCachePool(4, 32, 16, self.device)
        self.runner = PooledModelRunner(self.model, self.pool)

    def test_runner_cache_matches_full_context(self) -> None:
        prompt = [3, 1, 4]
        slot = self.pool.acquire("one")
        first = self.runner.prefill(slot, prompt)
        second = self.runner.decode([slot], [len(prompt)], [first])[0]

        expected = greedy_generate_full(
            self.model, torch.tensor([prompt], device=self.device), max_new_tokens=2
        )
        self.assertEqual([first, second], expected[0, -2:].cpu().tolist())

    def test_pool_refuses_release_by_wrong_owner(self) -> None:
        slot = self.pool.acquire("owner-a")
        with self.assertRaises(ValueError):
            self.pool.release("owner-b", slot)
        self.pool.release("owner-a", slot)
        self.assertEqual(self.pool.available_slots, 4)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class EngineTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        device = torch.device("cuda:0")
        self.model = ToyAttentionLM(32, 16, 32).to(device).eval()
        pool = KVCachePool(4, 32, 16, device)
        self.engine = Engine(PooledModelRunner(self.model, pool), pool, Scheduler(max_batched_tokens=8))

    def test_continuous_batching_matches_single_request_reference(self) -> None:
        first = Request("first", [1, 2], max_new_tokens=3)
        second = Request("second", [3, 4, 5], max_new_tokens=2)
        self.engine.add_request(first)
        self.engine.add_request(second)

        emissions: list[tuple[str, int]] = []
        while not (first.is_terminal and second.is_terminal):
            emissions.extend(self.engine.step())

        device = self.model.token_embedding.weight.device
        first_expected = greedy_generate_full(self.model, torch.tensor([[1, 2]], device=device), 3)[
            0, -3:
        ].cpu().tolist()
        second_expected = greedy_generate_full(self.model, torch.tensor([[3, 4, 5]], device=device), 2)[
            0, -2:
        ].cpu().tolist()
        self.assertEqual(first.generated_token_ids, first_expected)
        self.assertEqual(second.generated_token_ids, second_expected)
        self.assertEqual(len(emissions), 5)
        self.assertEqual(self.engine.cache_pool.used_slots, 0)

    def test_cancelled_waiting_request_never_executes(self) -> None:
        request = Request("cancel-me", [1, 2], max_new_tokens=3)
        self.engine.add_request(request)
        self.engine.cancel("cancel-me")
        self.assertEqual(self.engine.step(), [])
        self.assertEqual(request.status, RequestStatus.CANCELLED)

    def test_engine_rejects_request_that_can_never_fit(self) -> None:
        too_large_for_step = Request("too-wide", list(range(9)), max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "token budget"):
            self.engine.add_request(too_large_for_step)

        too_long_for_cache = Request("too-long", list(range(8)), max_new_tokens=26)
        with self.assertRaisesRegex(ValueError, "KV cache"):
            self.engine.add_request(too_long_for_cache)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class ServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        device = torch.device("cuda:0")
        model = ToyAttentionLM(32, 16, 32).to(device).eval()
        pool = KVCachePool(2, 32, 16, device)
        engine = Engine(PooledModelRunner(model, pool), pool, Scheduler(max_batched_tokens=8))
        self.service = InferenceService(engine)

    def test_service_collects_per_request_token_events(self) -> None:
        first = self.service.submit([1, 2], 2, request_id="first")
        second = self.service.submit([3, 4, 5], 2, request_id="second")
        outputs = {first: [], second: []}
        finished: set[str] = set()

        while len(finished) < 2:
            self.service.step()
            for request_id in outputs:
                update = self.service.poll(request_id)
                outputs[request_id].extend(update["tokens"])
                if update["finished"]:
                    finished.add(request_id)

        self.assertEqual(len(outputs[first]), 2)
        self.assertEqual(len(outputs[second]), 2)
        self.assertEqual(self.service.metrics()["cache_slots_used"], 0)

    def test_http_service_health_and_completion(self) -> None:
        server = create_http_server(("127.0.0.1", 0), self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            try:
                with urlopen(f"{address}/health", timeout=5) as response:
                    self.assertEqual(json.load(response), {"status": "ok"})
                body = json.dumps({"prompt_token_ids": [1, 2], "max_new_tokens": 2}).encode()
                request = URLRequest(
                    f"{address}/v1/completions", data=body, headers={"Content-Type": "application/json"}
                )
                with urlopen(request, timeout=5) as response:
                    completion = json.load(response)
                self.assertEqual(completion["status"], "finished")
                self.assertEqual(len(completion["tokens"]), 2)
            except URLError as exc:
                if isinstance(exc.reason, PermissionError):
                    self.skipTest("the execution sandbox forbids localhost sockets")
                raise
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
