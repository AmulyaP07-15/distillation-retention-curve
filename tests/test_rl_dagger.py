import torch

from src.rl_bc import train_bc
from src.rl_config import TrajectoryConfig
from src.rl_dagger import run_dagger
from tests.test_rl_expert import StubExpert


def make_initial_dataset(config, num_rows=40):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    observations = rng.normal(size=(num_rows, 8)).astype(np.float32)
    actions = rng.integers(0, 4, size=num_rows)

    return pd.DataFrame({
        "episode": [0] * num_rows,
        "observation": [row.tolist() for row in observations],
        "action": actions.tolist(),
    })


def test_dagger_history_has_one_entry_per_iteration():
    config = TrajectoryConfig(
        env_id="LunarLander-v3",
        dagger_iterations=2,
        dagger_episodes_per_iteration=1,
        dagger_eval_episodes=1,
        bc_epochs=1,
        hidden_dim=16,
    )

    initial_df = make_initial_dataset(config)
    torch.manual_seed(0)
    initial_policy = train_bc(initial_df, config)

    expert = StubExpert()
    _, history = run_dagger(expert, initial_df, initial_policy, config)

    assert len(history) == config.dagger_iterations


def test_dagger_dataset_grows_each_iteration():
    config = TrajectoryConfig(
        env_id="LunarLander-v3",
        dagger_iterations=2,
        dagger_episodes_per_iteration=1,
        dagger_eval_episodes=1,
        bc_epochs=1,
        hidden_dim=16,
    )

    initial_df = make_initial_dataset(config)
    torch.manual_seed(0)
    initial_policy = train_bc(initial_df, config)

    expert = StubExpert()
    _, history = run_dagger(expert, initial_df, initial_policy, config)

    sizes = [round_summary["dataset_size"] for round_summary in history]
    assert sizes == sorted(sizes), "Aggregated dataset must never shrink between DAgger rounds"
    assert sizes[-1] > len(initial_df), "DAgger should add newly-labeled states on top of the initial dataset"
