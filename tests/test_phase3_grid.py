import json

import pandas as pd

from run_phase2_grid import merge_rows
from run_phase3_grid import KEY_COLS, read_latest_phase3_eval


def write_log(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_missing_log_returns_empty_dict(tmp_path):
    metrics = read_latest_phase3_eval(str(tmp_path / "does_not_exist.jsonl"))
    assert metrics == {}


def test_reads_the_most_recent_entry_and_flattens_deploy_and_capability(tmp_path):
    log_path = tmp_path / "cell_log.jsonl"
    write_log(
        log_path,
        [
            {
                "gpu_name": "NVIDIA T4",
                "deploy": {"p50_ms_per_token": 20.0, "peak_memory_gb": 1.0},
                "capability": {"token_f1": 0.3, "ground_truth_perplexity": 30.0, "num_samples": 200},
            },
            {
                "gpu_name": "NVIDIA T4",
                "deploy": {"p50_ms_per_token": 15.0, "peak_memory_gb": 0.9},
                "capability": {"token_f1": 0.5, "ground_truth_perplexity": 20.0, "num_samples": 200},
            },
        ],
    )

    metrics = read_latest_phase3_eval(str(log_path))

    assert metrics["gpu_name"] == "NVIDIA T4"
    assert metrics["p50_ms_per_token"] == 15.0
    assert metrics["peak_memory_gb"] == 0.9
    assert metrics["token_f1"] == 0.5
    assert metrics["ground_truth_perplexity"] == 20.0
    assert metrics["num_capability_samples"] == 200


def test_merge_rows_keyed_on_student_signal_quant(tmp_path):
    out_path = tmp_path / "phase3.csv"
    pd.DataFrame(
        [
            {"student": "qwen0_5b", "signal": "logit", "quant": "fp16", "token_f1": 0.4},
            {"student": "qwen0_5b", "signal": "logit", "quant": "int8", "token_f1": 0.35},
        ]
    ).to_csv(out_path, index=False)

    new_rows = [{"student": "qwen0_5b", "signal": "logit", "quant": "int8", "token_f1": 0.38}]
    table = merge_rows(out_path, new_rows, key_cols=KEY_COLS)

    by_cell = {(r["student"], r["signal"], r["quant"]): r["token_f1"] for r in table.to_dict("records")}
    assert by_cell == {
        ("qwen0_5b", "logit", "fp16"): 0.4,
        ("qwen0_5b", "logit", "int8"): 0.38,
    }
