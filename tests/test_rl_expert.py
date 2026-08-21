import random

from src.rl_config import TrajectoryConfig
from src.rl_expert import (
    build_and_save_expert_dataset,
    compute_file_hash,
    evaluate_policy,
    make_env,
    rollout_episode,
)


class StubExpert:
    """Mimics the tiny slice of the SB3 PPO interface our code depends on: predict(obs)."""

    def __init__(self, num_actions=4, seed=0):
        self.num_actions = num_actions
        self.rng = random.Random(seed)

    def predict(self, obs, deterministic=True):
        return self.rng.randrange(self.num_actions), None


def random_act_fn(num_actions=4, seed=0):
    rng = random.Random(seed)

    def act(obs):
        return rng.randrange(num_actions)

    return act


def test_rollout_episode_has_matching_observation_and_action_counts():
    env = make_env("LunarLander-v3")
    result = rollout_episode(env, random_act_fn(), seed=0)
    env.close()

    assert len(result["observations"]) == len(result["actions"])
    assert len(result["observations"]) > 0
    assert isinstance(result["total_reward"], float)


def test_evaluate_policy_reports_requested_episode_count():
    metrics = evaluate_policy(random_act_fn(), "LunarLander-v3", num_episodes=3, seed=0)

    assert metrics["num_episodes"] == 3
    assert len(metrics["rewards"]) == 3
    assert isinstance(metrics["mean_reward"], float)


def test_expert_dataset_hash_matches_saved_file(tmp_path):
    config = TrajectoryConfig(
        env_id="LunarLander-v3",
        data_dir=str(tmp_path),
        num_expert_episodes_for_dataset=2,
        seed=0,
    )

    manifest = build_and_save_expert_dataset(StubExpert(), config)

    info = manifest["expert_trajectories"]
    actual_hash = compute_file_hash(info["path"])

    assert actual_hash == info["sha256"], "Saved parquet must match the hash recorded in the manifest"
    assert info["num_rows"] > 0
