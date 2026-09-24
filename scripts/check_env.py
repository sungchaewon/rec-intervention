"""Verify the research environment on a new machine (versions, recint import, devices).

Example:
    python scripts/check_env.py
    python scripts/check_env.py --require-cuda   # exit 1 if no usable GPU
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "tqdm", "yaml", "pytest", "torch", "recint"]
SMOKE_MATRIX_SIZE = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")
    args = parser.parse_args()

    print(f"python   {sys.version.split()[0]} ({platform.platform()})")
    for name in PACKAGES:
        module = importlib.import_module(name)
        print(f"{name:<8} {getattr(module, '__version__', '?')}")

    import torch

    print(f"torch CUDA build: {torch.version.cuda or 'none (CPU build)'}")
    devices = ["cpu"]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda:{index}   {torch.cuda.get_device_name(index)}")
            devices.append(f"cuda:{index}")
    else:
        print("cuda     not available")

    for device in devices:
        x = torch.randn(SMOKE_MATRIX_SIZE, SMOKE_MATRIX_SIZE, device=device)
        torch.testing.assert_close((x @ x.T).T, x @ x.T)
        print(f"matmul on {device}: ok")

    if args.require_cuda and len(devices) == 1:
        print("ERROR: --require-cuda given but no CUDA device is usable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
