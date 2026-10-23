"""Reward functions of the curriculum environment.

Sign conventions of :class:`ImprovementReward`:

- The aggregate skill is lower when the tsmodel is better. A decrease of the aggregate gives a
  positive reward: ``scale * (aggregate_{t-1} - aggregate_t)``.
- The training-quality terms are penalties: a high variance of the training loss gives a
  negative contribution, and a rising loss (positive slope) gives a negative contribution.
- The weights of the quality terms are zero by default. Then the rewards of an episode sum
  to ``scale * (aggregate_0 - aggregate_E)``.
"""

from pydantic import Field

from autotsaugment.components.env.history import RecentValues
from autotsaugment.core.featurizer import EpochRecord
from autotsaugment.core.reward import RewardConfig, RewardFunction
from autotsaugment.core.typechecking import PerEnv, checked


class ImprovementRewardConfig(RewardConfig):
    """Config of :class:`ImprovementReward`.

    Attributes:
        scale: Multiplier of the aggregate decrease.
        loss_variance_weight: Weight of the penalty on the variance of the training loss over
            the last ``loss_window`` epochs. Zero turns the term off.
        loss_slope_weight: Weight of the penalty on the least-squares slope of the training
            loss over the last ``loss_window`` epochs. Zero turns the term off.
        loss_window: Epochs that the variance and the slope use.
    """

    scale: float = 1.0
    loss_variance_weight: float = 0.0
    loss_slope_weight: float = 0.0
    loss_window: int = Field(default=5, ge=2)


class ImprovementReward(RewardFunction[ImprovementRewardConfig]):
    """``scale * (aggregate_{t-1} - aggregate_t) - w_var * variance - w_slope * slope``.

    ``aggregate_{t-1}`` is the aggregate of the previous step, or the aggregate given to
    :meth:`reset` at the first step. The variance and the slope are those of
    ``record.train["loss"]`` over the last ``loss_window`` epochs of the episode, the current
    epoch included. See the module docstring for the sign conventions.

    Args:
        config: The reward config.
    """

    def __init__(self, *, config: ImprovementRewardConfig) -> None:
        self.config = config
        self._previous: PerEnv | None = None
        self._losses = RecentValues(config.loss_window)

    @checked
    def reset(self, initial_aggregate: PerEnv) -> None:
        """Start an episode from the aggregate of the untrained tsmodels, shape ``(k,)``."""
        self._previous = initial_aggregate.detach().clone()
        self._losses.clear()

    @checked
    def __call__(self, record: EpochRecord) -> PerEnv:
        """Return the reward of the step described by ``record``, shape ``(k,)``.

        Raises:
            RuntimeError: If :meth:`reset` was not called first.
        """
        if self._previous is None:
            msg = "call reset() before the first step"
            raise RuntimeError(msg)
        aggregate = record.aggregate.detach()
        reward = self.config.scale * (self._previous - aggregate)
        self._losses.append(record.train["loss"].detach().float())
        if self.config.loss_variance_weight != 0.0:
            reward = reward - self.config.loss_variance_weight * self._losses.variance()
        if self.config.loss_slope_weight != 0.0:
            reward = reward - self.config.loss_slope_weight * self._losses.slope()
        self._previous = aggregate.clone()
        return reward
