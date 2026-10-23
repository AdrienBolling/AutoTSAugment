# AutoTSAugment

RL agent that steers synthetic time-series generation to train a forecaster.

One RL step is one training epoch of the forecaster (the tsmodel). The action is the parameter
vector of the synthetic generator for that epoch. The reward is the improvement of the tsmodel on
real benchmark data. v1 uses KernelSynth as the generator, Timestral-5M as the tsmodel, and
GIFT-Eval as the benchmark.

## Setup

    uv sync                            # Python 3.13 environment with the dev tools
    uv run pre-commit install          # lint, format, type check, lockfile hooks
    uv run autotsaugment-run experiment=smoke   # one short episode on fake components

## Layout

    src/autotsaugment/core/        contracts: abstract base classes, configs, type aliases
    src/autotsaugment/components/  implementations: augmentation, tsmodel, data, env, agents, tracking, experiments
    configs/                       Hydra config groups: experiment/, agent/, env/
    design_docs/                   plan, decisions, contracts
    tests/                         CPU tests by default; `-m gpu`, `-m network`, `-m slow` opt in

## Checks

    uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q

## Terminology

| Term | Definition |
|---|---|
| **tsmodel** | The forecaster the RL agent trains. v1: Timestral-5M, bare or LoRA. |
| **augmentation method** | A synthetic series generator with an action space. v1: KernelSynth. |
| **action** | The parameter vector of the augmentation method for one epoch. |
| **episode** | One full training run of the tsmodel, E epochs, from fixed initial weights. |
| **reward set** | Fixed windows from the benchmark train split, scored every step. |
| **comparison set** | The benchmark val split, scored once per episode, used to compare agents. |
| **aggregate** | Geometric mean over datasets of the skill-normalized WQL. |
