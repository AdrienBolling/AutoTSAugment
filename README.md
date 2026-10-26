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

## Experiments

Run every command from the repository root. Hydra composes `configs/run.yaml`. You can change
any leaf on the command line, for example `agent.config.episodes=20`.

    uv run autotsaugment-run                                    # smoke: fake env, CPU, no tracker, seconds
    uv run autotsaugment-run env=default device=cuda            # smoke on the real env, on the GPU
    uv run autotsaugment-run experiment=train_agent agent=sac   # train SAC, save it, evaluate it on 3 seeds
    uv run autotsaugment-run experiment=train_agent agent=cmaes # the CMA-ES control
    uv run autotsaugment-run experiment=compare                 # fixed uniform vs random, same seeds

To compare trained agents, add their checkpoints to the policies. Use the commented examples in
`configs/experiment/compare.yaml`, or the command line:

    uv run autotsaugment-run experiment=compare \
        +experiment.policies.sac.agent=sac \
        +experiment.policies.sac.checkpoint=checkpoints/train_agent/sac

`train_agent` and `compare` use the real environment, the GPU and MLflow by default. Add
`env=fake device=cpu` for a fast check. Add `tracker=none` to stop the tracking. (`tracker=null`
is not valid: in Hydra, `null` removes the group.)

### Outputs

| Path | Content | In git |
|---|---|---|
| `outputs/<date>/<time>/` | The Hydra output directory: composed config, log, `metrics.json` | no |
| `mlruns/` | The MLflow store: the SQLite database `mlflow.db` and `artifacts/` | no |
| `checkpoints/train_agent/<label>/` | `agent.pt`, `agent.yaml` (the agent config), `env.pt` (the featurizer state) | no |
| `metrics/` | `train_agent-<label>.json` and `compare.json`, the DVC metrics files | yes |
| `models/` | The `.dvc` files of the keeper checkpoints | yes |

`<label>` is the agent class in snake case: `sac`, `ppo`, `cmaes`, `fixed_action`, `random`.
Set another label with `experiment.label=<name>`.

### Metrics

Skill values are the WQL divided by the WQL of the seasonal-naive forecaster: lower is better.
Each evaluation seed runs one full episode with a deterministic policy.

| Key | Value |
|---|---|
| `comparison/aggregate/mean` | The primary metric: the aggregate on the comparison set, mean over the K tsmodels and the seeds |
| `comparison/aggregate/std` | The standard deviation of the aggregate over the seeds |
| `comparison/aggregate/seed<s>` | The aggregate of evaluation seed `s` |
| `comparison/<dataset>/mean` | The comparison skill of one dataset |
| `curve/auc/mean` | The secondary metric: the mean aggregate on the reward subset over the steps of the episode |
| `return/mean` | The episode return |
| `eval/seeds`, `eval/wall_time_s` | The number of seeds, the wall time of the evaluation |

`train_agent` adds `train/wall_time_s`, `train/episodes` and `wall_time_s`. `compare` puts each
key under `<policy>/`, for example `sac/comparison/aggregate/mean`, and logs one table. The
docstring of `src/autotsaugment/components/experiments/common.py` gives the full definitions.

### MLflow

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

MLflow 3.17 refuses the plain file store, so the local store is SQLite. Each run holds the
resolved config as parameters, the training steps and episodes, the evaluation under `eval/`
(`compare`: under `<policy>/`), the checkpoint and the metrics file as artifacts, and the git
commit as a tag.

### DVC

Training writes to `checkpoints/`, which git ignores. DVC cannot track a file there, so a keeper
checkpoint goes to `models/`. At the end, `train_agent` logs the commands, for example:

    dvc add checkpoints/train_agent/sac -o models/train_agent-sac
    git add models/train_agent-sac.dvc models/.gitignore metrics/train_agent-sac.json
    git commit -m "chore(models): add train_agent-sac"
    dvc push

Compare the metrics files:

    dvc metrics show metrics/train_agent-sac.json metrics/train_agent-cmaes.json
    dvc metrics diff --targets metrics/compare.json     # the workspace against HEAD
    dvc metrics diff main --targets metrics             # the workspace against main

## Layout

    src/autotsaugment/core/        contracts: abstract base classes, configs, type aliases
    src/autotsaugment/components/  implementations: augmentation, tsmodel, data, env, agents, tracking, experiments
    configs/                       Hydra config groups: experiment/, agent/, env/, tracker/
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
