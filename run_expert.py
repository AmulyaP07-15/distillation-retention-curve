import argparse
import json
from pathlib import Path

from src.rl_config import load_trajectory_config
from src.rl_expert import (
    build_and_save_expert_dataset,
    evaluate_policy,
    expert_act_fn,
    load_expert,
    train_expert,
)


def main():
    parser = argparse.ArgumentParser(description="Train the PPO expert on LunarLander and save its trajectories")
    parser.add_argument("--config", default="config/lunarlander.yaml", help="Path to YAML config")
    parser.add_argument("--retrain", action="store_true", help="Retrain the expert even if a checkpoint exists")
    args = parser.parse_args()

    config = load_trajectory_config(args.config)

    checkpoint_path = Path(config.checkpoint_dir) / "expert.zip"

    if args.retrain or not checkpoint_path.exists():
        model = train_expert(config)
    else:
        print(f"Loading existing expert from {checkpoint_path}")
        model = load_expert(config)

    print("Evaluating expert...")
    metrics = evaluate_policy(expert_act_fn(model), config.env_id, config.expert_eval_episodes)
    print(f"Expert mean reward over {config.expert_eval_episodes} episodes: {metrics['mean_reward']:.1f}")

    manifest = build_and_save_expert_dataset(model, config)

    run_log_path = Path(config.run_log)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "track": "lunarlander",
        "phase": "expert",
        "env_id": config.env_id,
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "dataset_rows": manifest["expert_trajectories"]["num_rows"],
    }
    with open(run_log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
