import pytest
import torch
from transformers import BitsAndBytesConfig

from src.quantization import build_quantization_kwargs


def test_fp16_uses_plain_torch_dtype():
    kwargs = build_quantization_kwargs("fp16")
    assert kwargs == {"torch_dtype": torch.float16}


def test_int8_uses_bitsandbytes_load_in_8bit():
    kwargs = build_quantization_kwargs("int8")
    quant_config = kwargs["quantization_config"]
    assert isinstance(quant_config, BitsAndBytesConfig)
    assert quant_config.load_in_8bit is True


def test_nf4_uses_bitsandbytes_nf4_with_fp16_compute():
    kwargs = build_quantization_kwargs("nf4")
    quant_config = kwargs["quantization_config"]
    assert isinstance(quant_config, BitsAndBytesConfig)
    assert quant_config.load_in_4bit is True
    assert quant_config.bnb_4bit_quant_type == "nf4"
    assert quant_config.bnb_4bit_compute_dtype == torch.float16


def test_unknown_quant_variant_raises():
    with pytest.raises(ValueError):
        build_quantization_kwargs("fp32")
