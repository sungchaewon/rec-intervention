from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_device(device: str) -> torch.device:
    """Map a config value (`auto` | `cpu` | `cuda` | `cuda:N`) to a torch device.

    `auto` uses CUDA when available, otherwise CPU. Explicit `cuda` fails loudly
    instead of silently falling back, so server runs never train on CPU by mistake.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.split(":")[0] not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES} (or cuda:N), got {device!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but CUDA is not available")
    return torch.device(device)
