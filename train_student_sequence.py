import argparse
import torch.multiprocessing as mp

from src.config import load_config
from src.sequence_trainer import train_sequence


def main():
    parser = argparse.ArgumentParser(description="Train the student via sequence-level distillation")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    if config.num_gpus > 1:
        mp.spawn(
            train_sequence,
            args=(config, config.num_gpus),
            nprocs=config.num_gpus,
            join=True,
        )
    else:
        train_sequence(rank=0, config=config, world_size=1)


if __name__ == "__main__":
    main()
