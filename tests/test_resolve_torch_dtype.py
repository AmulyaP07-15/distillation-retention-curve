import torch

import pytest

from src.trainer import resolve_torch_dtype


def test_float32_resolves_correctly():
    assert resolve_torch_dtype("float32") is torch.float32


def test_bfloat16_resolves_correctly():
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16


def test_unknown_name_raises_clear_error():
    with pytest.raises(ValueError, match="student_torch_dtype"):
        resolve_torch_dtype("not_a_real_dtype")


def test_non_dtype_torch_attribute_is_rejected():
    """torch.arange etc exist as attributes but aren't dtypes, must not be mistaken for one."""
    with pytest.raises(ValueError):
        resolve_torch_dtype("arange")
