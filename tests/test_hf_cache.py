import unittest

import torch

from mini_vllm.hf_cache import describe_gqa_cache, summarize_past_key_values


class _Config:
    num_hidden_layers = 24
    num_attention_heads = 14
    num_key_value_heads = 2
    hidden_size = 896


class HuggingFaceCacheHelperTest(unittest.TestCase):
    def test_gqa_memory_uses_kv_heads_not_query_heads(self) -> None:
        layout = describe_gqa_cache(_Config(), torch.bfloat16, batch_size=1)
        self.assertEqual(layout.head_dim, 64)
        self.assertEqual(layout.bytes_per_token_per_sequence, 2 * 24 * 2 * 64 * 2)

    def test_legacy_cache_summary_retains_tensor_metadata(self) -> None:
        key = torch.zeros((1, 2, 3, 64), dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        summary = summarize_past_key_values(((key, value),))
        self.assertEqual(summary[0].key_shape, (1, 2, 3, 64))
        self.assertEqual(summary[0].dtype, torch.bfloat16)
