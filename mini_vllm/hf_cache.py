"""Chapter 12 helpers for observing real GQA cache layout before paging it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CacheTensorInfo:
    layer: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class GQACacheLayout:
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    bytes_per_token_per_sequence: int


def summarize_past_key_values(past_key_values: Any) -> list[CacheTensorInfo]:
    """Return layer K/V metadata across legacy and modern Transformers caches."""
    if hasattr(past_key_values, "to_legacy_cache"):
        layers = past_key_values.to_legacy_cache()
    elif isinstance(past_key_values, (tuple, list)):
        layers = past_key_values
    elif hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
    else:
        raise TypeError(f"unsupported cache type: {type(past_key_values)!r}")

    summary: list[CacheTensorInfo] = []
    for layer_index, layer in enumerate(layers):
        if isinstance(layer, (tuple, list)):
            key, value = layer[:2]
        else:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise TypeError(f"cannot read K/V tensors from layer {layer_index}")
        summary.append(
            CacheTensorInfo(
                layer=layer_index,
                key_shape=tuple(key.shape),
                value_shape=tuple(value.shape),
                dtype=key.dtype,
                device=key.device,
            )
        )
    return summary


def describe_gqa_cache(config: Any, dtype: torch.dtype, batch_size: int = 1) -> GQACacheLayout:
    """Compute the K+V bytes added by one generated token for a batch."""
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // query_heads))
    layers = int(config.num_hidden_layers)
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    bytes_per_token = 2 * layers * batch_size * kv_heads * head_dim * bytes_per_element
    return GQACacheLayout(
        layers=layers,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        bytes_per_token_per_sequence=bytes_per_token,
    )
