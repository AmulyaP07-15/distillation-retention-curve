from src.config import Config
from src.dataset import build_and_save_splits, load_manifest, verify_manifest


def test_splits_are_disjoint(tmp_path):
    """Split A (fidelity) and split B (capability) must never share the same source example."""

    config = Config(
        dataset_name="tatsu-lab/alpaca",
        num_samples=200,
        data_dir=str(tmp_path),
        split_a_ratio=0.5,
        seed=42,
    )

    manifest = build_and_save_splits(config)

    split_a_indices = set(manifest["split_a"]["indices"])
    split_b_indices = set(manifest["split_b"]["indices"])

    overlap = split_a_indices & split_b_indices
    assert len(overlap) == 0, (
        f"Splits share {len(overlap)} indices. "
        f"First few: {sorted(overlap)[:5]}"
    )


def test_manifest_hashes_match_files(tmp_path):
    """Parquet files saved to disk must match the SHA-256 hashes in manifest.json."""

    config = Config(
        dataset_name="tatsu-lab/alpaca",
        num_samples=200,
        data_dir=str(tmp_path),
        split_a_ratio=0.5,
        seed=42,
    )

    build_and_save_splits(config)
    manifest = load_manifest(str(tmp_path))

    assert verify_manifest(manifest), "File hashes do not match what is recorded in manifest.json"


def test_split_sizes_match_ratio(tmp_path):
    """The two splits should have sizes proportional to split_a_ratio."""

    config = Config(
        dataset_name="tatsu-lab/alpaca",
        num_samples=200,
        data_dir=str(tmp_path),
        split_a_ratio=0.6,
        seed=42,
    )

    manifest = build_and_save_splits(config)

    total = manifest["split_a"]["num_rows"] + manifest["split_b"]["num_rows"]
    actual_ratio = manifest["split_a"]["num_rows"] / total

    assert abs(actual_ratio - 0.6) < 0.05, (
        f"Expected split A ratio near 0.6, got {actual_ratio:.3f}"
    )
