import argparse

from src.config import load_config
from src.teacher import ensure_teacher_pass


def main():
    parser = argparse.ArgumentParser(description="Run the teacher model and save logits to disk")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-download the dataset splits and re-run the teacher pass even if they already exist",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_teacher_pass(config, force=args.rebuild)
    print("Done.")


if __name__ == "__main__":
    main()
