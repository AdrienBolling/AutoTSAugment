"""Agents and library adapters (work packages WP5 and WP7 of design_docs/00-plan.md).

Modules: ``loop`` (the episode loop of every agent and experiment), ``policies`` (policies
without parameters and checkpoint helpers), ``fixed`` and ``random`` (baselines), ``cmaes``
(the static-action control), ``torchrl`` (TorchRL adapter, networks, SAC and PPO),
``gymnasium_adapter`` (a NumPy view for K = 1), ``grpo`` (GRPO), ``transformer`` (the causal
transformer policy with an episode memory) and ``distributions`` (its squashed Gaussian).
"""
