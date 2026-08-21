import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.rl_config import TrajectoryConfig
from src.rl_policy import MLPPolicy


def dataframe_to_tensors(df: pd.DataFrame):
    observations = np.stack(df["observation"].to_numpy())
    actions = df["action"].to_numpy()
    return (
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )


def train_bc(
    df: pd.DataFrame,
    config: TrajectoryConfig,
    policy: MLPPolicy = None,
) -> MLPPolicy:
    """
    Plain behavioral cloning: supervised classification from observation to
    the expert's action, ignoring that the student's own mistakes will move
    it into states the expert never showed it.

    If a policy is passed in, training continues from its current weights
    (used by DAgger so each round warm-starts from the last one instead of
    relearning from scratch).
    """
    obs_tensor, action_tensor = dataframe_to_tensors(df)

    if policy is None:
        obs_dim = obs_tensor.shape[1]
        num_actions = int(action_tensor.max().item()) + 1
        policy = MLPPolicy(obs_dim=obs_dim, hidden_dim=config.hidden_dim, num_actions=num_actions)

    dataset = TensorDataset(obs_tensor, action_tensor)
    loader = DataLoader(dataset, batch_size=config.bc_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.bc_learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    policy.train()
    final_loss = None
    for epoch in range(config.bc_epochs):
        epoch_loss = 0.0
        for obs_batch, action_batch in loader:
            optimizer.zero_grad()
            logits = policy(obs_batch)
            loss = loss_fn(logits, action_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * obs_batch.shape[0]
        final_loss = epoch_loss / len(dataset)

    print(f"BC training on {len(dataset)} pairs done, final epoch loss {final_loss:.4f}")

    policy.eval()
    return policy
