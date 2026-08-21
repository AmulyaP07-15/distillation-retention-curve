import torch

from src.rl_policy import MLPPolicy


def test_forward_output_shape():
    policy = MLPPolicy(obs_dim=8, num_actions=4, hidden_dim=32)
    obs_batch = torch.randn(5, 8)
    logits = policy(obs_batch)
    assert logits.shape == (5, 4)


def test_act_returns_valid_action_index():
    policy = MLPPolicy(obs_dim=8, num_actions=4, hidden_dim=32)
    obs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.0, 1.0]
    action = policy.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_act_is_deterministic_for_fixed_weights():
    policy = MLPPolicy(obs_dim=8, num_actions=4, hidden_dim=32)
    policy.eval()
    obs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.0, 1.0]
    first = policy.act(obs)
    second = policy.act(obs)
    assert first == second, "Greedy action for a fixed policy and input should not change between calls"
