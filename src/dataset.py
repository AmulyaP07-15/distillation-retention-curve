import hashlib
import json
import random
from pathlib import Path

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
