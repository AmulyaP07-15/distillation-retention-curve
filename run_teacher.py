import argparse
from pathlib import Path

from src.config import load_config
from src.dataset import build_and_save_splits, load_manifest, verify_manifest
from src.teacher import run_teacher_pass


def main():
    parser = argparse.ArgumentParser(description="Run the teacher model and save logits to disk")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--rebuild", action="store_true", help="Re-download and re-split the dataset")
    args = parser.parse_args()

    config = load_config(args.config)

    manifest_path = Path(config.data_dir) / "manifest.json"

    if args.rebuild or not manifest_path.exists():
        print("Building dataset splits...")
        build_and_save_splits(config)
    else:
        manifest = load_manifest(config.data_dir)
        if not verify_manifest(manifest):
            raise RuntimeError(
                "Dataset files do not match their stored hashes. "
                "Re-run with --rebuild to regenerate them."
            )
        print("Dataset already exists and hashes check out.")

    print("Running teacher inference pass...")
    run_teacher_pass(config)
    print("Done.")


if __name__ == "__main__":
    main()
