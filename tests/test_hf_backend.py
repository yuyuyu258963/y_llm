from types import SimpleNamespace
import unittest

import torch

from mini_vllm.hf_backend import DEFAULT_MODEL_ID, HuggingFaceReference


class HuggingFaceBoundaryTest(unittest.TestCase):
    def test_importing_the_optional_backend_does_not_load_a_model(self) -> None:
        # The test deliberately does not call .load(): model download belongs
        # to the explicit Chapter 09 command, never to normal test execution.
        self.assertEqual(DEFAULT_MODEL_ID, "Qwen/Qwen2.5-0.5B-Instruct")

    def test_chat_input_ids_accepts_tensor_and_batch_encoding_variants(self) -> None:
        class TensorTokenizer:
            def apply_chat_template(self, _messages: object, **_: object) -> torch.Tensor:
                return torch.tensor([[1, 2]])

        class EncodingTokenizer:
            def apply_chat_template(self, _messages: object, **_: object) -> dict[str, torch.Tensor]:
                return {"input_ids": torch.tensor([[3, 4]])}

        for tokenizer, expected in ((TensorTokenizer(), [[1, 2]]), (EncodingTokenizer(), [[3, 4]])):
            reference = HuggingFaceReference(
                model_id="test", tokenizer=tokenizer, model=SimpleNamespace(), device=torch.device("cpu")
            )
            self.assertEqual(reference.chat_input_ids([{"role": "user", "content": "x"}]).tolist(), expected)
