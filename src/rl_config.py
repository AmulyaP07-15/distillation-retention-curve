import yaml
from dataclasses import dataclass


@dataclass
class TrajectoryConfig:
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # PPO expert
    expert_total_timesteps: int = 500000
    expert_eval_episodes: int = 20
    checkpoint_dir: str = "checkpoints/lunarlander"

    # Expert trajectory dataset (state, action pairs used to train the student).
    # Kept deliberately small: a wide demo budget already covers enough of the
    # state space for BC to generalize, which hides the collapse DAgger fixes.
    data_dir: str = "data/lunarlander"
    num_expert_episodes_for_dataset: int = 5

    # Behavioral cloning student
    hidden_dim: int = 64
    bc_epochs: int = 20
    bc_batch_size: int = 256
    bc_learning_rate: float = 1.0e-3
    bc_eval_episodes: int = 20

    # DAgger
    dagger_iterations: int = 5
    dagger_episodes_per_iteration: int = 10
    dagger_eval_episodes: int = 20

    run_log: str = "runs/lunarlander_run_log.jsonl"


def load_trajectory_config(config_path: str) -> TrajectoryConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    valid_keys = {field for field in TrajectoryConfig.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return TrajectoryConfig(**filtered)
