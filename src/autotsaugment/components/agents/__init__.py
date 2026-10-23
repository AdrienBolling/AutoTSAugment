"""Agents and library adapters (work package WP5 of design_docs/00-plan.md).

Modules: ``loop`` (the episode loop of every agent and experiment), ``policies`` (policies
without parameters and checkpoint helpers), ``fixed`` and ``random`` (baselines), ``cmaes``
(the static-action control), ``torchrl`` (TorchRL adapter, networks, SAC and PPO) and
``gymnasium_adapter`` (a NumPy view for K = 1).
"""
