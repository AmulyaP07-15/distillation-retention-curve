import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    The distillation student for the control task.

    A tiny two-layer network that maps a LunarLander observation (8 floats)
    to logits over the four discrete actions. This is the thing we are
    trying to train to imitate the PPO expert, first with plain behavioral
    cloning and then with DAgger.
    """

    def __init__(self, obs_dim: int = 8, num_actions: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def act(self, obs) -> int:
        """Pick a greedy action for a single observation, used at rollout time."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(obs_tensor)
            return int(logits.argmax(dim=-1).item())
