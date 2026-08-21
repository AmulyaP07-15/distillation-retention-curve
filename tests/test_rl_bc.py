import numpy as np
import pandas as pd
import torch

from src.rl_bc import train_bc
from src.rl_config import TrajectoryConfig


def make_synthetic_dataset(n=400, obs_dim=8, num_actions=4, seed=0):
    """
    A dataset where the correct action is just argmax of the first
    num_actions observation dims. Trivial for the policy to learn, so we
    can check that training actually drives the loss down and not just
    that it runs without crashing.
    """
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(n, obs_dim)).astype(np.float32)
    actions = observations[:, :num_actions].argmax(axis=1)

    return pd.DataFrame({
        "observation": [row.tolist() for row in observations],
        "action": actions.tolist(),
    })


def test_bc_reduces_training_loss():
    df = make_synthetic_dataset()
    config = TrajectoryConfig(bc_epochs=1, bc_batch_size=64, bc_learning_rate=1e-2, hidden_dim=32)

    torch.manual_seed(0)
    policy_after_one_epoch = train_bc(df, config)

    config_more = TrajectoryConfig(bc_epochs=30, bc_batch_size=64, bc_learning_rate=1e-2, hidden_dim=32)
    torch.manual_seed(0)
    policy_after_many_epochs = train_bc(df, config_more)

    from src.rl_bc import dataframe_to_tensors
    obs_tensor, action_tensor = dataframe_to_tensors(df)

    with torch.no_grad():
        preds_early = policy_after_one_epoch(obs_tensor).argmax(dim=-1)
        preds_late = policy_after_many_epochs(obs_tensor).argmax(dim=-1)

    acc_early = (preds_early == action_tensor).float().mean().item()
    acc_late = (preds_late == action_tensor).float().mean().item()

    assert acc_late > acc_early, "More BC training should improve fit on an easily learnable dataset"
    assert acc_late > 0.9, f"Expected near-perfect training accuracy on trivial data, got {acc_late:.2f}"


def test_bc_warm_start_continues_from_existing_policy():
    df = make_synthetic_dataset()
    config = TrajectoryConfig(bc_epochs=5, bc_batch_size=64, bc_learning_rate=1e-2, hidden_dim=32)

    torch.manual_seed(0)
    policy = train_bc(df, config)
    warm_started = train_bc(df, config, policy=policy)

    assert warm_started is policy, "Passing an existing policy should continue training it in place"
