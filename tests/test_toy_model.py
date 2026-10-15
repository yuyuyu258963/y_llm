import unittest

import torch

from mini_vllm.generation import greedy_generate_cached, greedy_generate_full
from mini_vllm.runner import ReferenceModelRunner
from mini_vllm.toy_model import ToyAttentionLM


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class PrefillDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.device = torch.device("cuda:0")
        self.model = ToyAttentionLM(32, 16, 32).to(self.device).eval()

    def test_decode_logits_equal_full_context_logits(self) -> None:
        prompt = torch.tensor([[1, 2, 3]], device=self.device)
        next_input = torch.tensor([[4]], device=self.device)

        _, cache = self.model(prompt)
        decode_logits, extended_cache = self.model(next_input, cache)
        full_logits, _ = self.model(torch.cat((prompt, next_input), dim=1))

        self.assertEqual(extended_cache.length, 4)
        torch.testing.assert_close(decode_logits[:, -1], full_logits[:, -1])

    def test_cached_greedy_generation_equals_full_recomputation(self) -> None:
        prompt = torch.tensor([[2, 7, 1, 8]], device=self.device)
        full = greedy_generate_full(self.model, prompt, max_new_tokens=6)
        cached = greedy_generate_cached(self.model, prompt, max_new_tokens=6)

        torch.testing.assert_close(cached, full)

    def test_reference_runner_preserves_the_chapter_one_result(self) -> None:
        prompt = [2, 7, 1]
        runner = ReferenceModelRunner(self.model)
        first = runner.prefill(prompt)
        second = runner.decode(first.token_id, first.cache)

        expected = greedy_generate_full(
            self.model, torch.tensor([prompt], device=self.device), max_new_tokens=2
        )[0, -2:].cpu().tolist()
        self.assertEqual([first.token_id, second.token_id], expected)
        self.assertEqual(first.cache.key.device, self.device)
