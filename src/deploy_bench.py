import threading
import time

import numpy as np
import torch

from src.capability import build_chat_prompt, resolve_generation_eos_token_id
from src.measure_config import MeasureConfig


def select_fixed_prompts(split_b_df, measure_config: MeasureConfig, tokenizer) -> tuple:
    """
    One fixed prompt set per measure config (same measure_seed every call),
    independent of any single student's own capability_eval_samples/seed,
    so every quant variant of every student in the grid is timed on the
    exact same prompts. First num_warmup_prompts are discarded, the rest
    are what gets measured.
    """
    sample_size = min(measure_config.num_warmup_prompts + measure_config.num_measured_prompts, len(split_b_df))
    rows = split_b_df.sample(n=sample_size, random_state=measure_config.measure_seed).to_dict("records")
    prompts = [build_chat_prompt(row, tokenizer) for row in rows]
    warmup_prompts = prompts[: measure_config.num_warmup_prompts]
    measured_prompts = prompts[measure_config.num_warmup_prompts :]
    return warmup_prompts, measured_prompts


class PowerSampler:
    """
    Samples GPU power draw on a background thread at a fixed interval while
    the measured inference loop runs on the main thread. A single power
    reading per generate() call would miss whatever the draw does between
    calls, so this samples continuously for the whole measured pass instead.
    `read_watts` is injected so this is testable without a real GPU or
    pynvml handle.
    """

    def __init__(self, read_watts, interval_s: float):
        self.read_watts = read_watts
        self.interval_s = interval_s
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.is_set():
            self.samples.append(self.read_watts())
            self._stop_event.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop_event.set()
        self._thread.join()
        return False

    def average_watts(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


def make_nvml_power_reader(device_index: int = 0):
    """Real power reader for GPU runs. Kept out of PowerSampler itself so tests never need pynvml or a GPU."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def read_watts() -> float:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

    return read_watts


def summarize_latencies_ms(per_token_latencies_ms: list) -> dict:
    return {
        "p50_ms_per_token": float(np.percentile(per_token_latencies_ms, 50)),
        "p99_ms_per_token": float(np.percentile(per_token_latencies_ms, 99)),
    }


def compute_power_efficiency(throughput_toks_per_sec: float, avg_power_w: float) -> dict:
    return {
        "toks_per_sec_per_watt": throughput_toks_per_sec / avg_power_w,
        "joules_per_token": avg_power_w / throughput_toks_per_sec,
    }


def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / 1e9


def run_generation_loop(model, tokenizer, prompts: list, max_new_tokens: int, device: torch.device, eos_token_id: int):
    """
    One pass generating up to max_new_tokens for each prompt in `prompts`,
    one at a time (single-request latency, the number that matters for an
    edge-inference caller, not batched server throughput). Returns
    (per_token_latencies_ms, total_tokens_generated). Used for both the
    discarded warmup pass and the measured pass.
    """
    per_token_latencies_ms = []
    total_tokens = 0

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - start

        num_new_tokens = generated.shape[1] - input_ids.shape[1]
        if num_new_tokens > 0:
            per_token_latencies_ms.append((elapsed_s / num_new_tokens) * 1000.0)
            total_tokens += num_new_tokens

    return per_token_latencies_ms, total_tokens


def measure_deployment_cost(
    model,
    tokenizer,
    split_b_df,
    measure_config: MeasureConfig,
    device: torch.device,
    read_watts,
) -> dict:
    """
    Latency (p50/p99 per-token), peak memory, throughput, and perf-per-watt
    over measure_config's fixed prompt set, on the fixed device this
    process is running on. Warmup runs first and is discarded so cudnn
    autotune / first-call allocator overhead doesn't leak into the numbers;
    peak memory is reset right after warmup so it reflects the measured
    pass, not warmup's own allocations.
    """
    if measure_config.batch_size != 1:
        raise ValueError(
            "measure_deployment_cost only supports batch_size 1 (single-request latency). "
            f"Got batch_size={measure_config.batch_size}."
        )

    eos_token_id = resolve_generation_eos_token_id(tokenizer)
    warmup_prompts, measured_prompts = select_fixed_prompts(split_b_df, measure_config, tokenizer)

    run_generation_loop(model, tokenizer, warmup_prompts, measure_config.max_new_tokens, device, eos_token_id)

    reset_peak_memory(device)

    with PowerSampler(read_watts, measure_config.power_sample_interval_s) as sampler:
        start = time.perf_counter()
        per_token_latencies_ms, total_tokens = run_generation_loop(
            model, tokenizer, measured_prompts, measure_config.max_new_tokens, device, eos_token_id
        )
        elapsed_s = time.perf_counter() - start

    throughput_toks_per_sec = total_tokens / elapsed_s
    avg_power_w = sampler.average_watts()

    return {
        **summarize_latencies_ms(per_token_latencies_ms),
        "peak_memory_gb": peak_memory_gb(device),
        "throughput_toks_per_sec": throughput_toks_per_sec,
        "avg_power_w": avg_power_w,
        **compute_power_efficiency(throughput_toks_per_sec, avg_power_w),
        "num_measured_prompts": len(measured_prompts),
    }
