import argparse

from src.config import load_config
from src.evaluator import run_eval


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model fidelity against the teacher")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_eval(config)


if __name__ == "__main__":
    main()
