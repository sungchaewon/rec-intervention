import pytest
import torch

from recint.utils.device import resolve_device


def test_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_invalid_device_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU failure path")
def test_explicit_cuda_without_gpu_raises():
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
