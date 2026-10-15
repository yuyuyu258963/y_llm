"""Optional real-model checks; never download a checkpoint during ordinary tests."""

from __future__ import annotations

import os
import unittest

import torch

from mini_vllm.hf_backend import HuggingFaceReference
from mini_vllm.hf_runner import HuggingFaceReferenceRunner
from mini_vllm.hf_service import HuggingFaceTextService


RUN_REAL_MODEL = os.environ.get("MINI_VLLM_RUN_HF") == "1"


@unittest.skipUnless(RUN_REAL_MODEL, "set MINI_VLLM_RUN_HF=1 to run the downloaded-model checks")
class HuggingFaceIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # See Chapter 10: use FP32 for the strict logits oracle.  The normal
        # chapter/service default remains BF16 for an 8 GiB consumer GPU.
        cls.reference = HuggingFaceReference.load(dtype=torch.float32)

    def test_cached_decode_matches_full_context_logits(self) -> None:
        input_ids = self.reference.tokenizer("cache parity", return_tensors="pt").input_ids
        runner = HuggingFaceReferenceRunner(self.reference)
        prefill = runner.prefill(input_ids)
        cached = runner.decode(prefill.token_id, prefill)
        full_ids = torch.cat(
            (
                input_ids.to(self.reference.device),
                torch.tensor([[prefill.token_id]], dtype=torch.long, device=self.reference.device),
            ),
            dim=1,
        )
        full = self.reference.model(full_ids, use_cache=False).logits[:, -1]
        torch.testing.assert_close(cached.logits[:, -1], full, rtol=1e-3, atol=1e-3)

    def test_explicit_runner_matches_native_generate(self) -> None:
        messages = [{"role": "user", "content": "Say hi."}]
        service = HuggingFaceTextService(self.reference)
        explicit = service.complete(messages, max_new_tokens=4)["token_ids"]
        input_ids = self.reference.chat_input_ids(messages)
        native = self.reference.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=4,
            do_sample=False,
        )
        self.assertEqual(explicit, native[0, input_ids.size(1) :].cpu().tolist())
