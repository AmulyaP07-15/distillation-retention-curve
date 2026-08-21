import pandas as pd
from stable_baselines3 import PPO

from src.rl_bc import train_bc
from src.rl_config import TrajectoryConfig
from src.rl_expert import evaluate_policy, expert_act_fn, make_env, rollout_episode
from src.rl_policy import MLPPolicy


def run_dagger(
    expert_model: PPO,
    initial_df: pd.DataFrame,
    initial_policy: MLPPolicy,
    config: TrajectoryConfig,
) -> tuple:
    """
    DAgger: fix the distribution-shift problem that plain behavioral cloning
    has. Each round, let the *student's own policy* pick where it goes, then
    ask the expert what it should have done there, and add that to the
    training set. This directly teaches the student how to recover from its
    own mistakes, which the expert's on-distribution demonstrations never
    show it.
    """
    expert_fn = expert_act_fn(expert_model)
    aggregated_records = initial_df.to_dict("records")
    policy = initial_policy
    history = []

    for iteration in range(config.dagger_iterations):
        env = make_env(config.env_id)

        def student_act(obs):
            return policy.act(obs)

        new_records = []
        rollout_rewards = []

        for episode in range(config.dagger_episodes_per_iteration):
            seed = config.seed + 1000 * (iteration + 1) + episode
            result = rollout_episode(env, student_act, seed=seed)
            rollout_rewards.append(result["total_reward"])

            for obs in result["observations"]:
                new_records.append({
                    "episode": f"dagger_{iteration}_{episode}",
                    "observation": obs.tolist(),
                    "action": expert_fn(obs),
                })

        env.close()

        aggregated_records.extend(new_records)
        aggregated_df = pd.DataFrame(aggregated_records)

        policy = train_bc(aggregated_df, config, policy=policy)

        eval_metrics = evaluate_policy(
            lambda obs: policy.act(obs),
            config.env_id,
            config.dagger_eval_episodes,
            seed=5000 + iteration,
        )

        round_summary = {
            "iteration": iteration,
            "dataset_size": len(aggregated_records),
            "student_rollout_reward": sum(rollout_rewards) / len(rollout_rewards),
            "eval_mean_reward": eval_metrics["mean_reward"],
            "eval_std_reward": eval_metrics["std_reward"],
        }
        history.append(round_summary)
        print(
            f"DAgger iter {iteration} | dataset={round_summary['dataset_size']:5d} | "
            f"eval mean reward={round_summary['eval_mean_reward']:.1f}"
        )

    return policy, history
