import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_phase2_grid import GRID, merge_rows
from run_phase3_cell import default_run_log_path

# Same reasoning as run_phase2_grid.py: each cell loads its own model (a
# different quant variant of a possibly different student each time), so
# running each as its own subprocess guarantees the OS reclaims that memory
# before the next cell starts, instead of the parent process accumulating
# allocator fragmentation across 18 back-to-back loads.
SUBPROCESS_ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

QUANT_VARIANTS = ["fp16", "int8", "nf4"]
KEY_COLS = ["student", "signal", "quant"]


def read_latest_phase3_eval(run_log_path: str) -> dict:
    path = Path(run_log_path)
    if not path.exists():
        return {}

    latest = None
    with open(path, "r") as f:
        for line in f:
            latest = json.loads(line)

    if latest is None:
        return {}

    return {
        "gpu_name": latest["gpu_name"],
        **latest["deploy"],
        "token_f1": latest["capability"]["token_f1"],
        "rouge_l": latest["capability"].get("rouge_l"),
        "bertscore_f1": latest["capability"].get("bertscore_f1"),
        "ground_truth_perplexity": latest["capability"]["ground_truth_perplexity"],
        "num_capability_samples": latest["capability"]["num_samples"],
    }


def run_cell(student: str, signal: str, config_path: str, quant: str, measure_config_path: str):
    print(f"\n=== Quantizing + measuring {student} / {signal} / {quant} ===")
    subprocess.run(
        [
            sys.executable,
            "run_phase3_cell.py",
            "--config",
            config_path,
            "--student",
            student,
            "--signal",
            signal,
            "--quant",
            quant,
            "--measure-config",
            measure_config_path,
        ],
        check=True,
        env=SUBPROCESS_ENV,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantize each trained student x signal checkpoint into fp16/int8/nf4 and measure "
            "deployment cost (latency, memory, throughput, perf-per-watt) plus capability under "
            "quantization. Offline: reuses existing checkpoints and eval data, never retrains."
        )
    )
    parser.add_argument("--students", nargs="*", default=None, help="Subset of student names, e.g. qwen0_5b")
    parser.add_argument("--signals", nargs="*", default=None, help="Subset of signals, e.g. logit sequence")
    parser.add_argument("--quant", nargs="*", default=None, choices=QUANT_VARIANTS, help="Subset of quant variants")
    parser.add_argument(
        "--measure-config",
        default="config/phase3_measurement_t4.yaml",
        help="Path to the YAML measurement config that pins the GPU and prompt set",
    )
    parser.add_argument(
        "--skip-measure",
        action="store_true",
        help="Only re-read existing per-cell run logs and rebuild the table, without re-measuring anything",
    )
    parser.add_argument("--out", default="runs/phase3.csv", help="Where to write the aggregated table")
    args = parser.parse_args()

    quant_variants = args.quant or QUANT_VARIANTS

    rows = []

    for student, signal, config_path in GRID:
        if args.students and student not in args.students:
            continue
        if args.signals and signal not in args.signals:
            continue

        for quant in quant_variants:
            if not args.skip_measure:
                run_cell(student, signal, config_path, quant, args.measure_config)

            run_log_path = default_run_log_path(student, signal, quant)
            metrics = read_latest_phase3_eval(run_log_path)
            if not metrics:
                print(f"WARNING: no result found for {student}/{signal}/{quant} at {run_log_path}, skipping")
                continue
            rows.append({"student": student, "signal": signal, "quant": quant, **metrics})

    if not rows:
        print("Nothing matched the requested --students/--signals/--quant filter.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = merge_rows(out_path, rows, key_cols=KEY_COLS)
    table.to_csv(out_path, index=False)

    print("\n=== Phase 3 grid (student x signal x quant) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
