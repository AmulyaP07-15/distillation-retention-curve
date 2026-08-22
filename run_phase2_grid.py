import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Reduces CUDA allocator fragmentation, which matters here since each cell
# is its own subprocess loading a fresh model of a different size. Without
# this, a smaller student's run can leave the allocator fragmented in a way
# that makes the next, larger student harder to fit even with the same
# amount of free memory.
SUBPROCESS_ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

GRID = [
    ("qwen0_5b", "logit", "config/students/qwen0_5b_logit.yaml"),
    ("qwen0_5b", "sequence", "config/students/qwen0_5b_sequence.yaml"),
    ("qwen1_5b", "logit", "config/students/qwen1_5b_logit.yaml"),
    ("qwen1_5b", "sequence", "config/students/qwen1_5b_sequence.yaml"),
    ("qwen3b", "logit", "config/students/qwen3b_logit.yaml"),
    ("qwen3b", "sequence", "config/students/qwen3b_sequence.yaml"),
]

TRAIN_SCRIPT_BY_SIGNAL = {
    "logit": "train_student.py",
    "sequence": "train_student_sequence.py",
}


def read_latest_eval(run_log_path: str) -> dict:
    path = Path(run_log_path)
    if not path.exists():
        return {}

    latest = None
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("eval"):
                latest = entry

    if latest is None:
        return {}

    return {
        "top1_agreement": latest["fidelity"]["top1_agreement"],
        "kl_divergence": latest["fidelity"]["kl_divergence"],
        "token_f1": latest["capability"]["token_f1"],
        "ground_truth_perplexity": latest["capability"]["ground_truth_perplexity"],
    }


def run_teacher_step(teacher_config_path: str):
    """
    Build the shared dataset + teacher logits as a subprocess, not an
    in-process function call. The teacher pass loads the full 7B model onto
    the GPU, and running it as its own process guarantees the OS reclaims
    that memory when the process exits, before any training cell's
    subprocess starts. Calling ensure_teacher_pass() directly here instead
    would leave the 7B model resident in this long-lived parent process for
    the entire grid run, starving every cell of GPU memory regardless of
    student size.
    """
    print("=== Building shared teacher data (once for the whole grid) ===")
    subprocess.run(
        [sys.executable, "run_teacher.py", "--config", teacher_config_path],
        check=True,
        env=SUBPROCESS_ENV,
    )


def run_cell(student: str, signal: str, config_path: str, skip_training: bool):
    train_script = TRAIN_SCRIPT_BY_SIGNAL[signal]

    if not skip_training:
        print(f"\n=== Training {student} / {signal} ===")
        subprocess.run([sys.executable, train_script, "--config", config_path], check=True, env=SUBPROCESS_ENV)

    print(f"=== Evaluating {student} / {signal} ===")
    subprocess.run([sys.executable, "run_eval.py", "--config", config_path], check=True, env=SUBPROCESS_ENV)


def main():
    parser = argparse.ArgumentParser(
        description="Run the student x signal grid (train + eval) and aggregate fidelity/capability into one table"
    )
    parser.add_argument("--students", nargs="*", default=None, help="Subset of student names to run, e.g. qwen0_5b")
    parser.add_argument("--signals", nargs="*", default=None, help="Subset of signals to run, e.g. logit sequence")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only re-read existing run logs and rebuild the table, without training or re-evaluating",
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Skip the upfront teacher pass entirely (use existing data/ as-is, no hash check)",
    )
    parser.add_argument("--out", default="runs/phase2_grid.csv", help="Where to write the aggregated table")
    args = parser.parse_args()

    from src.config import load_config

    if not args.skip_teacher and not args.skip_training:
        # All 6 configs share teacher_model/dataset_name/num_samples/split_a_ratio/
        # seed/top_k_logits/max_length, so any one of them can build the shared
        # dataset + teacher_logits.parquet that every cell below reads from. This
        # runs Qwen2.5-7B-Instruct inference exactly once for the whole grid,
        # instead of once per student/signal cell.
        run_teacher_step(GRID[0][2])

    rows = []

    for student, signal, config_path in GRID:
        if args.students and student not in args.students:
            continue
        if args.signals and signal not in args.signals:
            continue

        config = load_config(config_path)

        if not args.skip_training:
            run_cell(student, signal, config_path, skip_training=False)

        metrics = read_latest_eval(config.run_log)
        rows.append({"student": student, "signal": signal, **metrics})

    if not rows:
        print("Nothing matched the requested --students/--signals filter.")
        return

    table = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print("\n=== Phase 2 grid (student x signal) ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
