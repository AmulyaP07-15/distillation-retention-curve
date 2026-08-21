import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.rl_config import TrajectoryConfig


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def make_env(env_id: str, seed: int = None):
    env = gym.make(env_id)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def train_expert(config: TrajectoryConfig) -> PPO:
    """Train a PPO expert on the control task. This is the teacher policy."""
    env = make_env(config.env_id, seed=config.seed)

    model = PPO(
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=0,
    )
    print(f"Training PPO expert on {config.env_id} for {config.expert_total_timesteps} timesteps...")
    model.learn(total_timesteps=config.expert_total_timesteps, progress_bar=True)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "expert.zip"
    model.save(str(model_path))
    print(f"Expert saved to {model_path}")

    env.close()
    return model


def load_expert(config: TrajectoryConfig) -> PPO:
    model_path = Path(config.checkpoint_dir) / "expert.zip"
    return PPO.load(str(model_path))


def expert_act_fn(model: PPO):
    """Wrap a PPO model as a plain obs -> action callable."""

    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


def rollout_episode(env, act_fn, seed: int = None) -> dict:
    """
    Run one episode with the given action function.

    Returns the observations, actions taken, and total reward. This is
    reused for expert dataset collection, policy evaluation, and DAgger
    state collection.
    """
    obs, info = env.reset(seed=seed)
    observations = []
    actions = []
    total_reward = 0.0
    done = False

    while not done:
        action = act_fn(obs)
        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "observations": observations,
        "actions": actions,
        "total_reward": total_reward,
    }


def evaluate_policy(act_fn, env_id: str, num_episodes: int, seed: int = 0) -> dict:
    """Roll out an act_fn for num_episodes and report mean/std reward."""
    env = make_env(env_id)
    rewards = []

    for episode in range(num_episodes):
        result = rollout_episode(env, act_fn, seed=seed + episode)
        rewards.append(result["total_reward"])

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "num_episodes": num_episodes,
        "rewards": rewards,
    }


def build_and_save_expert_dataset(model: PPO, config: TrajectoryConfig) -> dict:
    """
    Roll out the trained expert and save its (state, action) pairs as a
    versioned parquet dataset. This is the training set behavioral cloning
    learns from, equivalent in spirit to the teacher_logits dataset on the
    LLM track.
    """
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config.env_id)
    act_fn = expert_act_fn(model)

    records = []
    episode_rewards = []

    for episode in range(config.num_expert_episodes_for_dataset):
        result = rollout_episode(env, act_fn, seed=config.seed + episode)
        episode_rewards.append(result["total_reward"])

        for obs, action in zip(result["observations"], result["actions"]):
            records.append({
                "episode": episode,
                "observation": obs.tolist(),
                "action": action,
            })

    env.close()

    df = pd.DataFrame(records)
    dataset_path = data_dir / "expert_trajectories.parquet"
    df.to_parquet(str(dataset_path))

    manifest = {
        "expert_trajectories": {
            "path": str(dataset_path),
            "sha256": compute_file_hash(str(dataset_path)),
            "num_rows": len(df),
            "num_episodes": config.num_expert_episodes_for_dataset,
            "mean_expert_reward": float(np.mean(episode_rewards)),
        },
        "config": {
            "env_id": config.env_id,
            "seed": config.seed,
        },
    }

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Expert dataset: {len(df)} (state, action) pairs from {config.num_expert_episodes_for_dataset} episodes")
    print(f"Mean expert reward: {np.mean(episode_rewards):.1f}")
    print(f"Saved to {dataset_path}")

    return manifest


def load_expert_dataset(config: TrajectoryConfig) -> pd.DataFrame:
    manifest_path = Path(config.data_dir) / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return pd.read_parquet(manifest["expert_trajectories"]["path"])
