"""A causal transformer policy that reads the whole episode so far.

The token of step t is ``concat(o_t, s_{t-1})``: the flat observation of step t (entries in
sorted key order, see :func:`~autotsaugment.components.agents.torchrl.networks.flatten_observation`)
and the action of the previous step, scaled to ``[-1, 1]``:

    s_{t-1} = (a_{t-1} - low) / (high - low) * 2 - 1

with ``s_{-1} = 0`` (the box center) at t = 0. The observation groups are normalized, so the scale
keeps the raw action (``[-5, 5]`` for KernelSynth) from dominating the token. The previous actions
are the memory of the synthetic data that the policy already created in the episode.
:class:`HistoryNetwork` maps the tokens of an episode to the location and scale of a squashed
Gaussian at every step, and attends only to the steps up to the current one. The network has no
dropout, so the acting pass and the training pass compute the same function.
:class:`HistoryPolicy` keeps the tokens of the current episode and acts with the output at the
last step.

At initialization the location head is zero and the scale is ``init_scale`` everywhere. The mean
action is then the center of the box: for KernelSynth, all logits 0, the uniform recipe.
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor, nn

from autotsaugment.components.agents.distributions import sample_pre_squash
from autotsaugment.components.agents.torchrl.networks import flatten_observation, squash
from autotsaugment.core.featurizer import Observation
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.typechecking import ActionBatch, checked

Tokens = Float[Tensor, "b t d"]
"""The tokens of b episodes over t steps."""

StepParams = Float[Tensor, "b t a"]
"""One Gaussian parameter per episode, step and action dimension."""


class HistoryNetwork(nn.Module):
    """A causal transformer encoder from episode tokens to a squashed Gaussian per step.

    ``embed(token) + position(t)``, then ``n_layers`` pre-norm encoder layers with a causal
    mask and no dropout, a final layer norm, and two linear heads at every step: ``loc`` and
    ``scale = softplus(linear + b0) + min_scale`` with ``b0 = log(exp(init_scale - min_scale) - 1)``.
    Both heads start at zero, so the initial location is 0 and the initial scale is
    ``init_scale``.

    Args:
        token_dim: Length of one token, the flat observation plus the scaled previous action,
            ``D_obs + a``.
        action_dim: Action dimensions, a.
        max_steps: Length of the positional table, the longest episode the network accepts.
        d_model: Width of the residual stream.
        n_layers: Encoder layers.
        n_heads: Attention heads per layer. ``d_model`` must be a multiple of it.
        init_scale: Scale of the Gaussian at initialization.
        min_scale: Lower bound of the scale, below ``init_scale``.
        device: Device of the parameters.

    Raises:
        ValueError: If ``init_scale`` is not above ``min_scale``.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        max_steps: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        init_scale: float,
        min_scale: float,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if init_scale <= min_scale:
            msg = f"init_scale ({init_scale}) must be above min_scale ({min_scale})"
            raise ValueError(msg)
        self.max_steps = max_steps
        self.min_scale = min_scale
        self.scale_offset = math.log(math.expm1(init_scale - min_scale))
        self.embed = nn.Linear(token_dim, d_model, device=device)
        self.position = nn.Embedding(max_steps, d_model, device=device)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            device=device,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            n_layers,
            norm=nn.LayerNorm(d_model, device=device),
            enable_nested_tensor=False,
        )
        # The encoder deep-copies one layer, so every layer starts with the same weights.
        # Draw the matrices again per layer, as nn.Transformer does.
        for parameter in self.encoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        self.loc = nn.Linear(d_model, action_dim, device=device)
        self.scale = nn.Linear(d_model, action_dim, device=device)
        for head in (self.loc, self.scale):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @checked
    def gaussian(self, tokens: Tokens) -> tuple[StepParams, StepParams]:
        """Return the location and the scale at every step. ``network(tokens)`` calls it.

        The output at step t depends on the tokens of steps 0 to t only.

        Args:
            tokens: Shape ``(b, t, D_tok)``, float32.

        Returns:
            ``(loc, scale)``, each of shape ``(b, t, a)``.

        Raises:
            ValueError: If ``t`` is above ``max_steps``.
        """
        steps = tokens.shape[1]
        if steps > self.max_steps:
            msg = f"the episode has {steps} steps, the network accepts at most {self.max_steps}"
            raise ValueError(msg)
        positions = torch.arange(steps, device=tokens.device)
        hidden = self.embed(tokens) + self.position(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(
            steps, device=tokens.device, dtype=hidden.dtype
        )
        # is_causal=True is a hint that the mask is causal. Without it, the encoder compares the
        # mask with a causal mask on the device and reads the result on the host.
        hidden = self.encoder(hidden, mask=mask, is_causal=True)
        loc = self.loc(hidden)
        scale = torch.nn.functional.softplus(self.scale(hidden) + self.scale_offset)
        return loc, scale + self.min_scale

    # nn.Module declares ``forward`` as a mutable attribute, and ty rejects a ``def`` that
    # overrides it. The method has its own name, and ``forward`` refers to it.
    forward = gaussian


class HistoryPolicy:
    """The acting policy of a :class:`HistoryNetwork`, with the memory of the current episode.

    Each call adds the token of the current step to the memory, runs the network over all the
    tokens of the episode and acts with the output at the last step. Deterministic calls return
    the squashed location. Stochastic calls draw the pre-squash value ``u`` from the policy's
    generator, store it in :attr:`last_sample` and return its squashed value. The action is
    clipped to the box in both cases. Scaled to ``[-1, 1]``, it goes into the token of the next
    step. The calls make no host synchronization.

    :meth:`reset` empties the memory. The episode loop calls it after each ``env.reset``.

    Args:
        network: The network.
        space: The action box, on the device of the network. Every dimension needs a positive
            width.
        generator: Generator of the exploration noise, on the same device.

    Attributes:
        last_sample: The pre-squash value of the last call, shape ``(k, a)``: the sample of a
            stochastic call, the location of a deterministic call. ``None`` after a reset.

    Raises:
        ValueError: If a dimension of the box has no width.
    """

    last_sample: Tensor | None

    def __init__(
        self, network: HistoryNetwork, space: BoxSpace, generator: torch.Generator
    ) -> None:
        width = space.high - space.low
        if not bool((width > 0).all()):
            msg = "every dimension of the action box needs high > low to scale the action"
            raise ValueError(msg)
        self.network = network
        self.space = space
        self.generator = generator
        self.last_sample = None
        self._width = width
        self._tokens: list[Tensor] = []
        self._previous: Tensor | None = None

    @property
    def steps(self) -> int:
        """Return the number of tokens in the memory, the steps of the episode so far."""
        return len(self._tokens)

    def reset(self) -> None:
        """Forget the current episode: no token, no previous action, no last sample."""
        self._tokens = []
        self._previous = None
        self.last_sample = None

    def tokens(self) -> Tensor:
        """Return the tokens of the episode so far, exactly what the network read.

        Shape ``(k, t, D_tok)``. The last ``a`` entries of a token are the scaled previous action.

        Raises:
            RuntimeError: If the memory is empty.
        """
        if not self._tokens:
            msg = "the policy holds no token: call it at least once after reset()"
            raise RuntimeError(msg)
        return torch.stack(self._tokens, dim=1)

    @checked
    def __call__(self, observation: Observation, *, deterministic: bool = True) -> ActionBatch:
        """Return one action per environment, shape ``(k, a)``.

        Raises:
            ValueError: If the episode becomes longer than the ``max_steps`` of the network, or
                if the batch size changes inside an episode.
        """
        if self.steps >= self.network.max_steps:
            msg = (
                f"the episode has more than {self.network.max_steps} steps, the max_steps of the "
                "network: call reset() at the start of each episode"
            )
            raise ValueError(msg)
        flat = flatten_observation(observation)
        if self._tokens and flat.shape[0] != self._tokens[0].shape[0]:
            msg = (
                f"the episode has {self._tokens[0].shape[0]} rows, the observation {flat.shape[0]}"
            )
            raise ValueError(msg)
        previous = self._previous
        if previous is None:
            previous = flat.new_zeros((flat.shape[0], self.space.dim))
        self._tokens.append(torch.cat([flat, previous], dim=-1))
        with torch.no_grad():
            loc, scale = self.network(self.tokens())
        loc, scale = loc[:, -1], scale[:, -1]
        sample = loc if deterministic else sample_pre_squash(loc, scale, self.generator)
        action = self.space.clip(squash(sample, self.space.low, self.space.high))
        self.last_sample = sample
        self._previous = (action - self.space.low) / self._width * 2.0 - 1.0
        return action
