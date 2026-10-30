import math

import pytest
import torch
from torch import nn
from torchrl.modules import TanhNormal

from autotsaugment.components.agents.distributions import (
    gaussian_entropy,
    log_prob,
    sample_pre_squash,
)
from autotsaugment.components.agents.torchrl.networks import flatten_observation, squash
from autotsaugment.components.agents.transformer import HistoryNetwork, HistoryPolicy
from autotsaugment.core.agent import StatefulPolicy
from autotsaugment.core.seeding import make_generator
from autotsaugment.core.spaces import BoxSpace

OBSERVATION_DIM = 5
ACTION_DIM = 2
LOW = torch.tensor([-1.0, 0.0])
HIGH = torch.tensor([1.0, 4.0])


def make_network(
    device: torch.device, *, max_steps: int = 6, init_scale: float = 0.3
) -> HistoryNetwork:
    return HistoryNetwork(
        token_dim=OBSERVATION_DIM + ACTION_DIM,
        action_dim=ACTION_DIM,
        max_steps=max_steps,
        d_model=16,
        n_layers=2,
        n_heads=2,
        init_scale=init_scale,
        min_scale=0.01,
        device=device,
    )


def randomize_heads(network: HistoryNetwork) -> None:
    """Give the zero-initialized heads random weights, so the outputs depend on the tokens."""
    with torch.no_grad():
        for head in (network.loc, network.scale):
            nn.init.normal_(head.weight, std=0.5)
            nn.init.normal_(head.bias, std=0.5)


def make_policy(device: torch.device, network: HistoryNetwork) -> HistoryPolicy:
    space = BoxSpace(low=LOW.to(device), high=HIGH.to(device), names=("x", "y"))
    return HistoryPolicy(network, space, make_generator(device, 0))


def observation(device: torch.device, k: int = 3, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = make_generator(device, seed)
    return {
        "b": torch.randn((k, 3), generator=generator, device=device),
        "a": torch.randn((k, 2), generator=generator, device=device),
    }


# --- network -------------------------------------------------------------------------------


def test_network_output_shapes(device: torch.device):
    torch.manual_seed(0)
    network = make_network(device)
    tokens = torch.randn(4, 5, OBSERVATION_DIM + ACTION_DIM, device=device)
    loc, scale = network(tokens)
    assert loc.shape == (4, 5, ACTION_DIM)
    assert scale.shape == (4, 5, ACTION_DIM)
    assert loc.dtype == torch.float32
    dropouts = [module for module in network.modules() if isinstance(module, nn.Dropout)]
    assert all(module.p == 0.0 for module in dropouts)
    with pytest.raises(ValueError, match="at most 6"):
        network(torch.randn(1, 7, OBSERVATION_DIM + ACTION_DIM, device=device))


def test_network_is_causal(device: torch.device):
    torch.manual_seed(0)
    network = make_network(device)
    randomize_heads(network)
    tokens = torch.randn(2, 6, OBSERVATION_DIM + ACTION_DIM, device=device)
    changed = tokens.clone()
    changed[:, 3:] = torch.randn(2, 3, OBSERVATION_DIM + ACTION_DIM, device=device)
    with torch.no_grad():
        loc, scale = network(tokens)
        new_loc, new_scale = network(changed)
    torch.testing.assert_close(new_loc[:, :3], loc[:, :3])
    torch.testing.assert_close(new_scale[:, :3], scale[:, :3])
    assert not torch.allclose(new_loc[:, 3:], loc[:, 3:])
    # A prefix gives the same outputs as the full sequence.
    with torch.no_grad():
        prefix_loc, _ = network(tokens[:, :3])
    torch.testing.assert_close(prefix_loc, loc[:, :3])


def test_layers_start_with_different_weights(device: torch.device):
    torch.manual_seed(0)
    network = make_network(device)
    first, second = network.encoder.layers[0], network.encoder.layers[1]
    assert isinstance(first, nn.TransformerEncoderLayer)
    assert isinstance(second, nn.TransformerEncoderLayer)
    assert not torch.equal(first.linear1.weight, second.linear1.weight)


def test_initial_network_gives_zero_location_and_the_initial_scale(device: torch.device):
    torch.manual_seed(0)
    network = make_network(device, init_scale=0.4)
    tokens = torch.randn(3, 4, OBSERVATION_DIM + ACTION_DIM, device=device)
    loc, scale = network(tokens)
    torch.testing.assert_close(loc, torch.zeros_like(loc))
    torch.testing.assert_close(scale, torch.full_like(scale, 0.4))
    with pytest.raises(ValueError, match="above min_scale"):
        make_network(device, init_scale=0.01)


def test_large_network_shape_is_accepted(device: torch.device):
    network = HistoryNetwork(
        token_dim=109,
        action_dim=39,
        max_steps=50,
        d_model=512,
        n_layers=8,
        n_heads=8,
        init_scale=0.3,
        min_scale=0.01,
        device="meta",
    )
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert 24e6 < parameters < 27e6


# --- squashed Gaussian ---------------------------------------------------------------------


def test_log_prob_matches_torchrl_tanh_normal(device: torch.device):
    generator = make_generator(device, 3)
    shape = (64, ACTION_DIM)
    loc = torch.rand(shape, generator=generator, device=device) * 2.0 - 1.0
    scale = 0.1 + 0.4 * torch.rand(shape, generator=generator, device=device)
    pre_squash = sample_pre_squash(loc, scale, generator).clamp(-2.5, 2.5)
    low, high = LOW.to(device), HIGH.to(device)
    action = squash(pre_squash, low, high)
    expected = TanhNormal(loc, scale, low=low, high=high).log_prob(action)
    torch.testing.assert_close(
        log_prob(pre_squash, loc, scale, low, high), expected, atol=1e-4, rtol=1e-4
    )


def test_gaussian_entropy_matches_torch(device: torch.device):
    scale = torch.tensor([[0.1, 0.5], [1.0, 2.0]], device=device)
    expected = torch.distributions.Normal(torch.zeros_like(scale), scale).entropy().sum(dim=-1)
    torch.testing.assert_close(gaussian_entropy(scale), expected)
    assert gaussian_entropy(torch.ones(1, 3)).item() == pytest.approx(
        3 * (0.5 + 0.5 * math.log(2 * math.pi))
    )


# --- policy --------------------------------------------------------------------------------


def test_initial_policy_returns_the_box_center(device: torch.device):
    torch.manual_seed(0)
    policy = make_policy(device, make_network(device))
    assert isinstance(policy, StatefulPolicy)
    for _ in range(3):
        action = policy(observation(device))
        torch.testing.assert_close(action, ((LOW + HIGH) / 2).expand(3, -1).to(device))


def test_policy_stores_the_tokens_and_the_previous_action(device: torch.device):
    torch.manual_seed(0)
    network = make_network(device)
    randomize_heads(network)
    policy = make_policy(device, network)
    observations = [observation(device, seed=step) for step in range(4)]
    actions = [policy(obs, deterministic=False) for obs in observations]
    assert policy.last_sample is not None
    space = policy.space
    torch.testing.assert_close(
        actions[-1], space.clip(squash(policy.last_sample, space.low, space.high))
    )
    tokens = policy.tokens()
    assert tokens.shape == (3, 4, OBSERVATION_DIM + ACTION_DIM)
    for step, obs in enumerate(observations):
        torch.testing.assert_close(tokens[:, step, :OBSERVATION_DIM], flatten_observation(obs))
    torch.testing.assert_close(tokens[:, 0, OBSERVATION_DIM:], torch.zeros(3, ACTION_DIM))
    for step in range(1, 4):
        scaled = (actions[step - 1] - LOW) / (HIGH - LOW) * 2.0 - 1.0
        torch.testing.assert_close(tokens[:, step, OBSERVATION_DIM:], scaled)
        assert tokens[:, step, OBSERVATION_DIM:].abs().max() <= 1.0


def test_deterministic_actions_equal_the_network_over_the_episode(device: torch.device):
    # The learner reads the stored tokens in one pass: it must see what the policy saw.
    torch.manual_seed(0)
    network = make_network(device)
    randomize_heads(network)
    policy = make_policy(device, network)
    actions = torch.stack([policy(observation(device, seed=step)) for step in range(5)], dim=1)
    with torch.no_grad():
        loc, _ = network(policy.tokens())
    space = policy.space
    torch.testing.assert_close(actions, space.clip(squash(loc, space.low, space.high)))


def test_reset_empties_the_memory(device: torch.device):
    torch.manual_seed(0)
    network = make_network(device, max_steps=3)
    randomize_heads(network)
    policy = make_policy(device, network)
    first = policy(observation(device))
    policy(observation(device, seed=1))
    assert policy.steps == 2
    policy.reset()
    assert policy.steps == 0
    assert policy.last_sample is None
    with pytest.raises(RuntimeError, match="no token"):
        policy.tokens()
    again = policy(observation(device))
    assert policy.steps == 1
    assert policy.tokens().shape == (3, 1, OBSERVATION_DIM + ACTION_DIM)
    torch.testing.assert_close(again, first)
    policy(observation(device, seed=1))
    policy(observation(device, seed=2))
    with pytest.raises(ValueError, match="call reset"):
        policy(observation(device, seed=3))


def test_policy_needs_a_box_with_width(device: torch.device):
    space = BoxSpace(low=torch.zeros(2), high=torch.tensor([1.0, 0.0]), names=("x", "y"))
    with pytest.raises(ValueError, match="high > low"):
        HistoryPolicy(make_network(device), space, make_generator(device, 0))


def test_policy_rejects_a_new_batch_size_inside_an_episode(device: torch.device):
    policy = make_policy(device, make_network(device))
    policy(observation(device, k=3))
    with pytest.raises(ValueError, match="3 rows"):
        policy(observation(device, k=2))
