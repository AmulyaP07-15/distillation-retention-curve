import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

QUANT_VARIANTS = ["fp16", "int8", "nf4"]


def build_quantization_kwargs(quant: str) -> dict:
    """
    from_pretrained kwargs for one quant variant.

    fp16 is the deployment baseline (half the memory of the fp32/bf16
    training dtype, no quantization library involved). int8 and nf4 both
    load the fp16 weights through bitsandbytes, which quantizes them into
    device memory at load time rather than needing a separately saved
    quantized checkpoint.
    """
    if quant == "fp16":
        return {"torch_dtype": torch.float16}

    if quant == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}

    if quant == "nf4":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        }

    raise ValueError(f"Unknown quant variant '{quant}', expected one of {QUANT_VARIANTS}")


def load_quantized_model(checkpoint_dir: str, quant: str, device: torch.device):
    """
    bitsandbytes places the quantized weights on the GPU itself as part of
    from_pretrained, so passing device_map matters for int8/nf4 and calling
    .to(device) afterwards would just be a no-op (and errors for 4-bit).
    fp16 has no such placement step, so it needs the explicit .to(device).
    """
    kwargs = build_quantization_kwargs(quant)

    if quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **kwargs).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map={"": device}, **kwargs)

    model.eval()
    return model
