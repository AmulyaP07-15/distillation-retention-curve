import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.capability import compute_capability
from src.config import load_config
from src.dataset import load_manifest, load_split
from src.deploy_bench import make_nvml_power_reader, measure_deployment_cost
from src.measure_config import load_measure_config
from src.quantization import load_quantized_model


def default_run_log_path(student: str, signal: str, quant: str) -> str:
    return f"runs/phase3_{student}_{signal}_{quant}_log.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Quantize one trained student checkpoint and measure its deployment cost + capability"
    )
    parser.add_argument("--config", required=True, help="Path to the student's YAML config (same one used to train it)")
    parser.add_argument("--student", required=True, help="Student name, e.g. qwen0_5b (for logging only)")
    parser.add_argument("--signal", required=True, help="Signal name, e.g. logit or sequence (for logging only)")
    parser.add_argument("--quant", required=True, choices=["fp16", "int8", "nf4"])
    parser.add_argument("--measure-config", required=True, help="Path to the YAML measurement config (pins the GPU)")
    parser.add_argument("--run-log", default=None, help="Where to append this cell's result (default derived from student/signal/quant)")
    args = parser.parse_args()

    config = load_config(args.config)
    measure_config = load_measure_config(args.measure_config)

    checkpoints = sorted(Path(config.checkpoint_dir).glob("epoch_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {config.checkpoint_dir}. Train this student first.")
    latest_checkpoint = checkpoints[-1]
    print(f"[{args.student}/{args.signal}/{args.quant}] Loading checkpoint: {latest_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(str(latest_checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(measure_config.device)
    model = load_quantized_model(str(latest_checkpoint), args.quant, device)

    read_watts = make_nvml_power_reader() if device.type == "cuda" else (lambda: 0.0)

    manifest = load_manifest(measure_config.data_dir)
    split_b_df = load_split(manifest, "split_b")

    print(f"[{args.student}/{args.signal}/{args.quant}] Measuring deployment cost on {measure_config.gpu_name}...")
    deploy_metrics = measure_deployment_cost(model, tokenizer, split_b_df, measure_config, device, read_watts)

    print(f"[{args.student}/{args.signal}/{args.quant}] Running capability eval...")
    capability_metrics = compute_capability(model, tokenizer, split_b_df, config, device)

    result = {
        "student": args.student,
        "signal": args.signal,
        "quant": args.quant,
        "gpu_name": measure_config.gpu_name,
        "checkpoint": str(latest_checkpoint),
        "deploy": deploy_metrics,
        "capability": capability_metrics,
    }

    print()
    print(f"=== {args.student} / {args.signal} / {args.quant} on {measure_config.gpu_name} ===")
    print(f"p50 latency (ms/token)   : {deploy_metrics['p50_ms_per_token']:.3f}")
    print(f"p99 latency (ms/token)   : {deploy_metrics['p99_ms_per_token']:.3f}")
    print(f"Peak memory (GB)         : {deploy_metrics['peak_memory_gb']:.3f}")
    print(f"Throughput (tok/s)       : {deploy_metrics['throughput_toks_per_sec']:.3f}")
    print(f"Avg power (W)            : {deploy_metrics['avg_power_w']:.3f}")
    print(f"Perf/watt (tok/s/W)      : {deploy_metrics['toks_per_sec_per_watt']:.5f}")
    print(f"Energy (J/token)         : {deploy_metrics['joules_per_token']:.5f}")
    print(f"Token F1 vs reference    : {capability_metrics['token_f1']:.4f}")
    print(f"BERTScore F1             : {capability_metrics['bertscore_f1']:.4f}")
    print(f"Ground-truth perplexity  : {capability_metrics['ground_truth_perplexity']:.4f}")

    run_log_path = Path(args.run_log or default_run_log_path(args.student, args.signal, args.quant))
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_log_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


if __name__ == "__main__":
    main()
