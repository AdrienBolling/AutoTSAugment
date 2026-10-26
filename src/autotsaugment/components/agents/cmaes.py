"""CMA-ES control: a search over one static action held for the whole episode.

If the CMA-ES action matches the RL curriculum, the sequential decision adds nothing. Each
generation evaluates its candidates in groups of K, one episode per group: candidate j stays
constant for all E steps in lockstep environment j. The fitness of a candidate is the geometric
mean over datasets of its comparison skill from ``env.evaluate()``, the aggregate of the primary
metric (:func:`autotsaugment.core.metrics.aggregate`). Lower is better, and CMA-ES minimizes.

``cma`` works in NumPy on the CPU. The agent moves the fitness of the K candidates to the host
once per episode, and the box bounds once per call to :meth:`CMAESAgent.fit`.
"""

import math
from pathlib import Path
from typing import Any, Literal

import cma
import numpy as np
import torch
from pydantic import Field

from autotsaugment.components.agents.loop import episode_seed, geometric_mean, run_episode
from autotsaugment.components.agents.policies import (
    ConstantActionPolicy,
    SlotActionPolicy,
    load_state,
    not_fitted,
    save_state,
)
from autotsaugment.core.agent import Agent, AgentConfig, Policy
from autotsaugment.core.environment import Environment
from autotsaugment.core.seeding import MAX_SEED, derive_seed
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tracker import Tracker


class CMAESAgentConfig(AgentConfig):
    """Config of :class:`CMAESAgent`.

    Attributes:
        episodes: Evaluation episodes. Each episode evaluates K candidates.
        sigma0: Initial step size of the search, in action units.
        population: Candidates per generation. ``None`` rounds the default CMA-ES population
            ``4 + floor(3 ln a)`` up to a multiple of K, so no episode slot is wasted.
        fixed_seed: Evaluate every episode with the same seed (common random numbers), so
            that all candidates see the same noise. ``False`` gives each episode its own seed.
        final: The action of the returned policy. ``"best"`` is the evaluated candidate with
            the lowest fitness, the right choice under common random numbers. ``"mean"`` is
            the final mean of the search distribution, more robust when the seed changes.
    """

    episodes: int = Field(default=40, ge=0)
    sigma0: float = Field(default=1.0, gt=0.0)
    population: int | None = Field(default=None, ge=2)
    fixed_seed: bool = True
    final: Literal["best", "mean"] = "best"


SKILL_FLOOR = 1e-12
"""Lower clamp of the skill before the logarithm, so a zero skill keeps a finite fitness."""


def candidate_fitness(final_skill: torch.Tensor) -> torch.Tensor:
    """Return the fitness of each slot: the geometric mean of its skill over datasets.

    This is the aggregate of the primary metric, so CMA-ES ranks candidates as the experiment
    does. The skill is positive; values below :data:`SKILL_FLOOR` are clamped to it.

    Args:
        final_skill: Comparison skill per slot and dataset, shape ``(k, d)``.

    Returns:
        Shape ``(k,)``. Lower is better.
    """
    return geometric_mean(final_skill.clamp_min(SKILL_FLOOR))


def default_population(action_dim: int, batch_envs: int) -> int:
    """Return the default CMA-ES population rounded up to a multiple of ``batch_envs``."""
    base = 4 + math.floor(3 * math.log(action_dim))
    return batch_envs * math.ceil(base / batch_envs)


class _BestCandidate:
    """Keeps the evaluated candidate with the lowest fitness."""

    def __init__(self, start: np.ndarray) -> None:
        self.action = start
        self.fitness = math.inf

    def update(self, candidates: list[np.ndarray], fitness: list[float]) -> None:
        for candidate, value in zip(candidates, fitness, strict=True):
            if value < self.fitness:
                self.action, self.fitness = candidate, value


class CMAESAgent(Agent[CMAESAgentConfig]):
    """Searches one static action with CMA-ES and returns it as a constant policy.

    The search starts at the zero action, clipped to the box. When the budget ends inside a
    generation, CMA-ES does not receive the incomplete generation, but its candidates still
    compete for the best action.

    Args:
        config: The agent config.
        device: Device of the candidate actions.
    """

    _action: torch.Tensor | None

    def __init__(self, *, config: CMAESAgentConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._action = None

    def _strategy(self, space: BoxSpace, batch_envs: int, seed: int) -> cma.CMAEvolutionStrategy:
        """Return a new CMA-ES search over ``space``, seeded from ``seed``."""
        low = space.low.detach().cpu().numpy().astype(np.float64)
        high = space.high.detach().cpu().numpy().astype(np.float64)
        options = {
            "bounds": [low.tolist(), high.tolist()],
            "popsize": self.config.population or default_population(space.dim, batch_envs),
            "seed": 1 + derive_seed(seed, "cmaes") % MAX_SEED,
            "verbose": -9,
        }
        start = np.clip(np.zeros(space.dim), low, high)
        return cma.CMAEvolutionStrategy(start, self.config.sigma0, options)

    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Run CMA-ES for ``config.episodes`` episodes and return the constant policy."""
        space = env.action_space.to(self.device)
        k = env.batch_size
        strategy = self._strategy(space, k, seed)
        best = _BestCandidate(np.asarray(strategy.mean, dtype=np.float64).copy())
        episode = 0
        while episode < self.config.episodes:
            candidates: list[np.ndarray] = list(strategy.ask())
            fitness: list[float] = []
            for start in range(0, len(candidates), k):
                if episode >= self.config.episodes:
                    break
                group = candidates[start : start + k]
                values = self._evaluate(
                    env, group, strategy.mean, tracker=tracker, seed=seed, episode=episode
                )
                best.update(group, values)
                fitness.extend(values)
                episode += 1
            if len(fitness) == len(candidates):
                strategy.tell(candidates, fitness)
        final = best.action if self.config.final == "best" else strategy.result.xfavorite
        action = torch.as_tensor(np.asarray(final), dtype=torch.float32, device=self.device)
        self._action = space.clip(action.unsqueeze(0)).squeeze(0)
        return self.policy()

    def _evaluate(
        self,
        env: Environment[Any],
        group: list[np.ndarray],
        filler: np.ndarray,
        *,
        tracker: Tracker,
        seed: int,
        episode: int,
    ) -> list[float]:
        """Run one episode with one candidate per slot, return the fitness of the group.

        A group smaller than K fills the free slots with ``filler``; their fitness is dropped.
        """
        k = env.batch_size
        rows = group + [filler] * (k - len(group))
        actions = torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device)
        seed_index = 0 if self.config.fixed_seed else episode
        result = run_episode(
            env,
            SlotActionPolicy(actions),
            seed=episode_seed(seed, seed_index),
            tracker=tracker,
            step_offset=episode * env.episode_length,
            episode=episode,
            deterministic=True,
        )
        fitness = candidate_fitness(result.final_skill.detach()).cpu().tolist()
        return [float(value) for value in fitness[: len(group)]]

    @property
    def action(self) -> torch.Tensor | None:
        """Return the selected action, shape ``(a,)``, or ``None`` before :meth:`fit`."""
        return self._action

    def policy(self) -> Policy:
        """Return the constant policy of the selected action.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._action is None:
            raise not_fitted(self)
        return ConstantActionPolicy(self._action)

    def save(self, path: Path) -> None:
        """Write the selected action to ``path/agent.pt``.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._action is None:
            raise not_fitted(self)
        save_state(path, {"action": self._action})

    def load(self, path: Path) -> None:
        """Restore the selected action that :meth:`save` wrote under ``path``."""
        self._action = load_state(path)["action"].to(self.device)
