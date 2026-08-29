import json

import pandas as pd

from run_phase2_grid import merge_rows, read_latest_eval


def write_log(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_missing_log_returns_empty_dict(tmp_path):
    metrics = read_latest_eval(str(tmp_path / "does_not_exist.jsonl"))
    assert metrics == {}


def test_reads_the_most_recent_eval_entry(tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    write_log(log_path, [
        {"step": 1, "total_loss": 1.0},
        {
            "eval": True,
            "fidelity": {"top1_agreement": 0.5, "kl_divergence": 1.0},
            "capability": {"token_f1": 0.2, "ground_truth_perplexity": 50.0},
        },
        {"step": 2, "total_loss": 0.9},
        {
            "eval": True,
            "fidelity": {"top1_agreement": 0.8, "kl_divergence": 0.4},
            "capability": {"token_f1": 0.4, "ground_truth_perplexity": 20.0},
        },
    ])

    metrics = read_latest_eval(str(log_path))

    assert metrics["top1_agreement"] == 0.8
    assert metrics["kl_divergence"] == 0.4
    assert metrics["token_f1"] == 0.4
    assert metrics["ground_truth_perplexity"] == 20.0


def test_ignores_non_eval_entries(tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    write_log(log_path, [{"step": 1, "total_loss": 1.0}])

    metrics = read_latest_eval(str(log_path))
    assert metrics == {}


def test_merge_rows_with_no_existing_csv_returns_new_rows(tmp_path):
    out_path = tmp_path / "phase2_grid.csv"
    new_rows = [{"student": "qwen3b", "signal": "logit", "top1_agreement": 0.9}]

    table = merge_rows(out_path, new_rows)

    assert table.to_dict("records") == new_rows


def test_merge_rows_keeps_untouched_cells_and_replaces_matching_ones(tmp_path):
    out_path = tmp_path / "phase2_grid.csv"
    pd.DataFrame(
        [
            {"student": "qwen0_5b", "signal": "logit", "top1_agreement": 0.1},
            {"student": "qwen3b", "signal": "logit", "top1_agreement": 0.5},
        ]
    ).to_csv(out_path, index=False)

    new_rows = [{"student": "qwen3b", "signal": "logit", "top1_agreement": 0.9}]
    table = merge_rows(out_path, new_rows)

    by_cell = {(r["student"], r["signal"]): r["top1_agreement"] for r in table.to_dict("records")}
    assert by_cell == {("qwen0_5b", "logit"): 0.1, ("qwen3b", "logit"): 0.9}
