"""Fast, checkpoint-free contracts for the Chapter 11 adapter."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from mini_vllm.hf_runner import HuggingFaceReferenceRunner


class _FakeModel:
    """Records the public Transformers-style calls made by the runner."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        old_length = int(kwargs.get("past_key_values") or 0)
        cache_length = old_length + input_ids.size(1)
        logits = torch.zeros((1, input_ids.size(1), 7))
        logits[:, -1, cache_length % 7] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=cache_length)


class HuggingFaceRunnerContractTest(unittest.TestCase):
    def test_prefill_then_decode_extends_mask_and_cache_once(self) -> None:
        model = _FakeModel()
        reference = SimpleNamespace(model=model, device=torch.device("cpu"))
        runner = HuggingFaceReferenceRunner(reference)

        prefill = runner.prefill(torch.tensor([[3, 4, 5]]))
        decoded = runner.decode(prefill.token_id, prefill)

        self.assertEqual(prefill.past_key_values, 3)
        self.assertEqual(decoded.past_key_values, 4)
        self.assertEqual(tuple(decoded.attention_mask.shape), (1, 4))
        self.assertTrue(torch.equal(decoded.attention_mask, torch.ones((1, 4), dtype=torch.long)))
        self.assertEqual(tuple(model.calls[1]["input_ids"].shape), (1, 1))
        self.assertEqual(model.calls[1]["past_key_values"], 3)


if __name__ == "__main__":
    unittest.main()
