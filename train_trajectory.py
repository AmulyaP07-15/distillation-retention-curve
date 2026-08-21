import argparse
import json
from pathlib import Path

from src.rl_bc import train_bc
from src.rl_config import load_trajectory_config
from src.rl_dagger import run_dagger
from src.rl_expert import evaluate_policy, load_expert, load_expert_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train a student policy via behavioral cloning, then fix its off-distribution collapse with DAgger"
    )
    parser.add_argument("--config", default="config/lunarlander.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_trajectory_config(args.config)

    print(f"Loading expert from {config.checkpoint_dir}")
    expert_model = load_expert(config)

    print(f"Loading expert trajectory dataset from {config.data_dir}")
    expert_df = load_expert_dataset(config)

    run_log_path = Path(config.run_log)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 1: Behavioral cloning ===")
    bc_policy = train_bc(expert_df, config)

    bc_metrics = evaluate_policy(
        lambda obs: bc_policy.act(obs),
        config.env_id,
        config.bc_eval_episodes,
        seed=2000,
    )
    print(f"BC policy mean reward over {config.bc_eval_episodes} episodes: {bc_metrics['mean_reward']:.1f}")
    print("This is expected to collapse relative to the expert: BC never sees states")
    print("caused by its own mistakes, so it has no idea how to recover from them.")

    bc_log_entry = {
        "track": "lunarlander",
        "phase": "behavioral_cloning",
        "mean_reward": bc_metrics["mean_reward"],
        "std_reward": bc_metrics["std_reward"],
        "training_pairs": len(expert_df),
    }
    with open(run_log_path, "a") as f:
        f.write(json.dumps(bc_log_entry) + "\n")

    print("\n=== Phase 2: DAgger ===")
    dagger_policy, dagger_history = run_dagger(expert_model, expert_df, bc_policy, config)

    for round_summary in dagger_history:
        entry = {"track": "lunarlander", "phase": "dagger", **round_summary}
        with open(run_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    final_reward = dagger_history[-1]["eval_mean_reward"]

    print("\n=== Reward before vs after DAgger fix ===")
    print(f"Behavioral cloning (collapse) : {bc_metrics['mean_reward']:.1f}")
    print(f"DAgger, final iteration       : {final_reward:.1f}")
    print(f"Delta                         : {final_reward - bc_metrics['mean_reward']:+.1f}")

    summary_entry = {
        "track": "lunarlander",
        "phase": "summary",
        "bc_mean_reward": bc_metrics["mean_reward"],
        "dagger_final_mean_reward": final_reward,
        "delta": final_reward - bc_metrics["mean_reward"],
    }
    with open(run_log_path, "a") as f:
        f.write(json.dumps(summary_entry) + "\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
