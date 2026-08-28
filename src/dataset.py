import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.config import Config


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_and_save_splits(config: Config) -> dict:
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {config.dataset_name}")
    raw_dataset = load_dataset(config.dataset_name, split="train")

    random.seed(config.seed)
    all_indices = list(range(len(raw_dataset)))
    random.shuffle(all_indices)
    selected = all_indices[: config.num_samples]

    cut = int(len(selected) * config.split_a_ratio)
    split_a_indices = sorted(selected[:cut])
    split_b_indices = sorted(selected[cut:])

    assert len(set(split_a_indices) & set(split_b_indices)) == 0, "Splits must not overlap"

    split_a_data = raw_dataset.select(split_a_indices)
    split_b_data = raw_dataset.select(split_b_indices)

    split_a_path = str(data_dir / "split_a.parquet")
    split_b_path = str(data_dir / "split_b.parquet")

    split_a_data.to_parquet(split_a_path)
    split_b_data.to_parquet(split_b_path)

    manifest = {
        "split_a": {
            "path": split_a_path,
            "sha256": compute_file_hash(split_a_path),
            "num_rows": len(split_a_data),
            "indices": split_a_indices,
        },
        "split_b": {
            "path": split_b_path,
            "sha256": compute_file_hash(split_b_path),
            "num_rows": len(split_b_data),
            "indices": split_b_indices,
        },
        "config": {
            "dataset_name": config.dataset_name,
            "num_samples": config.num_samples,
            "split_a_ratio": config.split_a_ratio,
            "seed": config.seed,
        },
    }

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split A: {len(split_a_data)} rows -> {split_a_path}")
    print(f"Split B: {len(split_b_data)} rows -> {split_b_path}")
    print(f"Manifest: {manifest_path}")

    return manifest


def load_manifest(data_dir: str) -> dict:
    manifest_path = Path(data_dir) / "manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def verify_manifest(manifest: dict) -> bool:
    for split_name in ["split_a", "split_b"]:
        info = manifest[split_name]
        actual_hash = compute_file_hash(info["path"])
        if actual_hash != info["sha256"]:
            print(f"Hash mismatch for {split_name}")
            print(f"  Expected : {info['sha256']}")
            print(f"  Got      : {actual_hash}")
            return False
    return True


def load_split(manifest: dict, split_name: str) -> pd.DataFrame:
    return pd.read_parquet(manifest[split_name]["path"])


# Columns in teacher_logits.parquet that hold one value per token position
# (a flat sequence, e.g. token ids). Parquet round-trips these as a genuine
# 1-D numpy array per row, which breaks "+" list concatenation (numpy
# broadcasts elementwise instead of appending) even though torch.tensor(...)
# handles them fine. Normalizing to a plain Python list makes both usages safe.
TEACHER_SEQUENCE_COLUMNS = ["input_ids", "generated_ids"]

# Columns that hold one small vector per token position (e.g. the top-k
# logits at each position). Parquet round-trips these as an *object*-dtype
# array of shape (seq_len,), where every element is itself a (top_k,) array,
# not a true 2-D numeric array. torch.tensor(...) can't infer a dtype from
# that directly. np.stack rebuilds the real (seq_len, top_k) array.
TEACHER_PER_TOKEN_COLUMNS = {
    "top_logit_indices": np.int64,
    "top_logit_values": np.float32,
}


def load_teacher_logits(manifest: dict, include_per_token_columns: bool = True) -> pd.DataFrame:
    """
    Load teacher_logits.parquet and normalize every list/array column once,
    at the read boundary, so every downstream reader (training datasets,
    fidelity eval, sequence distillation) gets clean, consistently-typed
    values and never has to know about parquet's round-trip quirks itself.

    include_per_token_columns=False drops top_logit_indices/top_logit_values
    entirely instead of normalizing them. Sequence-level distillation trains
    on generated_ids alone and never reads the teacher's top-k logits, so
    stacking and dtype-casting a (seq_len, top_k) array per row for every
    row in the dataset would just be host-memory and CPU work spent on data
    nothing downstream touches.
    """
    df = pd.read_parquet(manifest["teacher_logits"]["path"])

    for col in TEACHER_SEQUENCE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(list)

    if include_per_token_columns:
        for col, dtype in TEACHER_PER_TOKEN_COLUMNS.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda cell, dtype=dtype: np.stack(cell).astype(dtype))
    else:
        drop_cols = [col for col in TEACHER_PER_TOKEN_COLUMNS if col in df.columns]
        df = df.drop(columns=drop_cols)

    return df
