import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.capability import compute_capability
from src.config import Config
from src.dataset import load_manifest, load_split, load_teacher_logits


def pad_input_ids(batch_rows: list, pad_token_id: int) -> list:
    """
    row["input_ids"] comes back from parquet as a numpy array, not a Python
    list, so "+" would do elementwise addition (and broadcast-error on
    mismatched lengths) instead of concatenation. list(...) first fixes that.
    """
    max_len = max(len(row["input_ids"]) for row in batch_rows)
    return [
        list(row["input_ids"]) + [pad_token_id] * (max_len - len(row["input_ids"]))
        for row in batch_rows
    ]


def row_top_k_tensors(row: dict) -> tuple:
    """
    A row's top_logit_indices/values is a parquet-round-tripped object array
    of shape (seq_len,), where each element is itself a (top_k,) array.
    np.stack turns that into the real 2D numeric array torch.tensor needs.
    """
    top_k_indices = torch.tensor(np.stack(row["top_logit_indices"]).astype(np.int64), dtype=torch.long)
    top_k_values = torch.tensor(np.stack(row["top_logit_values"]).astype(np.float32), dtype=torch.float)
    return top_k_indices, top_k_values


def compute_fidelity(
    model,
    tokenizer,
    teacher_df: pd.DataFrame,
    config: Config,
    device: torch.device,
) -> dict:
    """
    Compare the student's predictions to the teacher's at every token position.

    Metrics:
      top1_agreement  - fraction of positions where student and teacher agree on the most likely token
      kl_divergence   - average KL(teacher || student) using the stored top-k teacher logits
    """
    model.eval()

    top1_agreements = []
    kl_divergences = []

    rows = teacher_df.to_dict("records")

    for batch_start in tqdm(range(0, len(rows), config.batch_size), desc="Fidelity eval"):
        batch_rows = rows[batch_start : batch_start + config.batch_size]

        padded_ids = pad_input_ids(batch_rows, tokenizer.pad_token_id)
        input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            student_logits = outputs.logits.cpu()

        for i, row in enumerate(batch_rows):
            seq_len = len(row["input_ids"])

            student_seq = student_logits[i, :seq_len]

            top_k_indices, top_k_values = row_top_k_tensors(row)

            teacher_top1 = top_k_values.argmax(dim=-1)
            teacher_top1_tokens = top_k_indices.gather(-1, teacher_top1.unsqueeze(-1)).squeeze(-1)
            student_top1_tokens = student_seq.argmax(dim=-1)

            agreement = (teacher_top1_tokens == student_top1_tokens).float().mean().item()
            top1_agreements.append(agreement)

            teacher_probs = F.softmax(top_k_values, dim=-1)
            teacher_log_probs = F.log_softmax(top_k_values, dim=-1)

            student_log_probs_full = F.log_softmax(student_seq, dim=-1)
            student_log_probs_at_top_k = student_log_probs_full.gather(-1, top_k_indices)

            kl = (teacher_probs * (teacher_log_probs - student_log_probs_at_top_k)).sum(dim=-1).mean().item()
            kl_divergences.append(kl)

    return {
        "top1_agreement": float(np.mean(top1_agreements)),
        "kl_divergence": float(np.mean(kl_divergences)),
    }


def run_eval(config: Config):
    manifest = load_manifest(config.data_dir)
    teacher_df = load_teacher_logits(manifest)

    checkpoints = sorted(Path(config.checkpoint_dir).glob("epoch_*"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {config.checkpoint_dir}. Run train_student.py or "
            "train_student_sequence.py first."
        )

    latest_checkpoint = checkpoints[-1]
    print(f"Loading student from: {latest_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(str(latest_checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(config.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(latest_checkpoint),
        torch_dtype=torch.float32,
    ).to(device)

    print("Running fidelity eval on split A (vs teacher)...")
    fidelity_metrics = compute_fidelity(model, tokenizer, teacher_df, config, device)

    print()
    print("=== Fidelity Results (Split A, vs teacher) ===")
    print(f"Top-1 Agreement with Teacher : {fidelity_metrics['top1_agreement']:.4f}")
    print(f"KL Divergence from Teacher   : {fidelity_metrics['kl_divergence']:.4f}")

    print()
    print("Running capability eval on split B (vs ground truth)...")
    split_b_df = load_split(manifest, "split_b")
    capability_metrics = compute_capability(model, tokenizer, split_b_df, config, device)

    print()
    print("=== Capability Results (Split B, vs ground truth) ===")
    print(f"Token F1 vs Reference        : {capability_metrics['token_f1']:.4f}")
    print(f"Ground-Truth Perplexity      : {capability_metrics['ground_truth_perplexity']:.4f}")

    run_log_path = Path(config.run_log)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    eval_entry = {
        "eval": True,
        "checkpoint": str(latest_checkpoint),
        "student_model": config.student_model,
        "fidelity": fidelity_metrics,
        "capability": capability_metrics,
    }
    with open(run_log_path, "a") as f:
        f.write(json.dumps(eval_entry) + "\n")

    return {"fidelity": fidelity_metrics, "capability": capability_metrics}
