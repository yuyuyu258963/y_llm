"""Explicit device-boundary helpers used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """The one accelerator selected for this single-GPU tutorial."""

    device: torch.device
    name: str


def require_cuda() -> DeviceInfo:
    """Return the tutorial GPU or explain why this project cannot proceed."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not found. This tutorial is intentionally GPU-first; "
            "check your NVIDIA driver and the PyTorch CUDA build."
        )
    device = torch.device("cuda:0")
    return DeviceInfo(device=device, name=torch.cuda.get_device_name(device))


def synchronize(device: torch.device) -> None:
    """Make the GPU/CPU boundary explicit before measuring or reading results."""
    torch.cuda.synchronize(device)
