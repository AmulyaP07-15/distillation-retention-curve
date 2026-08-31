import pandas as pd
import torch

from src.deploy_bench import (
    PowerSampler,
    compute_power_efficiency,
    peak_memory_gb,
    run_generation_loop,
    select_fixed_prompts,
    summarize_latencies_ms,
)
from src.measure_config import MeasureConfig


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def convert_tokens_to_ids(self, token):
        return None

    eos_token_id = 999

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        # Encodes to a fixed 4-token prompt regardless of text, just enough
        # for run_generation_loop to compute a real new-token count against.
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}


class FakeModel:
    def __init__(self, num_new_tokens: int):
        self.num_new_tokens = num_new_tokens

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id, eos_token_id):
        extra = torch.zeros((1, self.num_new_tokens), dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


def make_split_b_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"instruction": f"Do task {i}", "input": "", "output": f"Result {i}"} for i in range(n)]
    )


def test_select_fixed_prompts_splits_warmup_from_measured():
    df = make_split_b_df(10)
    measure_config = MeasureConfig(num_warmup_prompts=3, num_measured_prompts=4, measure_seed=1)

    warmup, measured = select_fixed_prompts(df, measure_config, FakeTokenizer())

    assert len(warmup) == 3
    assert len(measured) == 4


def test_select_fixed_prompts_is_deterministic_for_a_fixed_seed():
    df = make_split_b_df(10)
    measure_config = MeasureConfig(num_warmup_prompts=2, num_measured_prompts=2, measure_seed=42)

    first = select_fixed_prompts(df, measure_config, FakeTokenizer())
    second = select_fixed_prompts(df, measure_config, FakeTokenizer())

    assert first == second


def test_run_generation_loop_counts_new_tokens_and_times_each_call():
    model = FakeModel(num_new_tokens=5)
    tokenizer = FakeTokenizer()
    device = torch.device("cpu")

    per_token_latencies_ms, total_tokens = run_generation_loop(
        model, tokenizer, ["prompt a", "prompt b"], max_new_tokens=5, device=device, eos_token_id=999
    )

    assert total_tokens == 10
    assert len(per_token_latencies_ms) == 2
    assert all(latency >= 0 for latency in per_token_latencies_ms)


def test_summarize_latencies_ms_computes_p50_and_p99():
    latencies = list(range(1, 101))  # 1..100
    summary = summarize_latencies_ms(latencies)

    assert summary["p50_ms_per_token"] == 50.5
    assert summary["p99_ms_per_token"] == 99.01


def test_compute_power_efficiency_matches_hand_computed_values():
    result = compute_power_efficiency(throughput_toks_per_sec=100.0, avg_power_w=50.0)

    assert result["toks_per_sec_per_watt"] == 2.0
    assert result["joules_per_token"] == 0.5


def test_peak_memory_gb_returns_nan_off_cuda():
    assert peak_memory_gb(torch.device("cpu")) != peak_memory_gb(torch.device("cpu"))  # nan != nan


def test_power_sampler_collects_samples_from_injected_reader():
    watts_sequence = iter([10.0, 20.0, 30.0])

    def read_watts():
        return next(watts_sequence, 30.0)

    with PowerSampler(read_watts, interval_s=0.01) as sampler:
        import time

        time.sleep(0.05)

    assert len(sampler.samples) >= 1
    assert sampler.average_watts() > 0
