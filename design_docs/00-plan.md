# 00 — Plan: AutoTSAugment v1

Status: locked 2026-10-23 after a 12-question design review. This document is the contract
for the first implementation. Change it only through a pull request.

## 1. Goal and hypothesis

We train a forecaster (the tsmodel) only on synthetic series. An RL agent chooses, at each
training epoch, the parameters of the synthetic generator. The hypothesis of experiment 1:

> An RL curriculum over KernelSynth beats the uniform KernelSynth recipe of Chronos at an
> equal number of synthetic series and tsmodel updates, measured on held-out real data.

Success criterion:

- Primary metric: the aggregate score on the comparison set after E epochs, averaged over
  at least 3 seeds. The aggregate is the geometric mean over datasets of the skill-normalized
  WQL (Weighted Quantile Loss divided by the WQL of the seasonal-naive forecaster).
- Baseline: the fixed uniform action (the Chronos recipe), same tsmodel, same E, same N.
- Control: CMA-ES over one static action held for the whole episode. If it matches the RL
  curriculum, the sequential decision adds nothing.
- Secondary: area under the aggregate curve over the episode.

## 2. Locked decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Data splits | GIFT-Eval official test part (last 10%) is never used. The rest is split along time: 85% reward split (scored every step), 15% comparison split (scored once per episode). |
| 2 | Episode | One full tsmodel training run from fixed initial weights. E = 50 epochs, N = 4096 series of 1024 steps per epoch, K = 4 tsmodels in lockstep on one GPU. Target: under 10 minutes per episode. All configurable. |
| 3 | Action | Continuous box of 38 logits: 32 over the kernel bank, 5 over the number of kernels (1..5), 1 for P(×) versus P(+). Zeros = Chronos recipe. Bank hyperparameters fixed. |
| 4 | Observation | Dict of 4 normalized groups: `progress`, `train_stats`, `reward_vector`, `action_history`. About 100 dims. Featurizer is swappable. |
| 5 | Reward | `r_t = agg_{t-1} - agg_t`. Per-dataset vector and training-quality metrics in `info`, with weights that default to zero. Reward windows: a random subset per step. |
| 6 | Data | Standalone `components/data`, no tslake dependency. GIFT-Eval from the Hugging Face cache, at most 10 GB on disk, 8 default datasets over 7 domains. |
| 7 | tsmodel | Timestral-5M bare (random init) first, LoRA on the pretrained weights second, one `TSModel` interface. Pinball loss on the 21 knots. Windows: context 768, horizon 256. |
| 8 | Agent | SAC first, PPO second, TorchRL. Own `Environment` protocol with TorchRL and gymnasium adapters. Baselines: fixed uniform, random, CMA-ES. |
| 9 | Config | Hydra groups `experiment/`, `agent/`, `env/` (`tsmodel/`, `augmentation/`, `benchmark/`, `featurizer/`, `reward/`), `tracker/`. Pydantic validates each component config at startup. `_target_` is the registry. |
| 10 | Conventions | tslake rule set: ruff ALL with documented ignores, ty strict, Google docstrings, jaxtyping with opt-in beartype, pre-commit chain. Python 3.13. |
| 11 | Tracking | `Tracker` interface: `NullTracker`, `MLflowTracker` (local `mlruns/`). Step metrics buffered on the device, flushed once per episode. DVC for keeper checkpoints, `metrics/<experiment>.json`. |
| 12 | Git | Bootstrap on `main`. One branch per work package in its own worktree, one pull request per branch, merged by the user. |

## 3. Architecture

### 3.1 One step

```
agent.policy(obs) ──► action (k, 38)
                        │
                        ▼
augmentation.sample(action, N) ──► series (k, N, 1024)      GPU, batched Cholesky
                        │
                        ▼
tsmodel[i].train_epoch(series[i]) for i in 0..K-1 ──► TrainStats (k,)   bf16 autocast
                        │
                        ▼
scorer(tsmodel[i], reward_windows.subset(m)) ──► scores (k, d), aggregate (k,)
                        │
                        ▼
reward(record) ──► r (k,)        featurizer(record) ──► obs (dict of (k, ...))
                        │
                        ▼
tracker.log_step(...)  buffered; flushed at log_episode
```

At the end of an episode, `env.evaluate()` scores the full comparison set once.

### 3.2 Packages

```
src/autotsaugment/
  core/           contracts only; never imports components
    typechecking  shape aliases, @checked (AUTOTSAUGMENT_TYPECHECK=1)
    config        ComponentConfig, EpisodeConfig, RunConfig
    seeding       derive_seed, make_generator, seed_everything
    spaces        BoxSpace
    metrics       pinball_loss, wql_per_dataset, skill_scores, aggregate, seasonal_naive
    augmentation  AugmentationMethod
    tsmodel       TSModel, TrainStats, TSModelFactory
    benchmark     Benchmark, WindowSet, BenchmarkConfig
    featurizer    Featurizer, EpochRecord, Observation
    reward        RewardFunction
    environment   Environment, StepResult, EnvironmentConfig
    agent         Agent, Policy
    tracker       Tracker, to_host
    experiment    Experiment, write_metrics
  components/
    fakes         fast fakes of every contract (tests, smoke runs, agent development)
    augmentation/ WP2   kernelsynth
    tsmodel/      WP3   timestral (bare, lora)
    data/         WP1   gifteval, windows, cache
    env/          WP4   curriculum, featurizers, rewards, scoring
    agents/       WP5   fixed, random, cmaes, torchrl/{adapter,sac,ppo}, gymnasium_adapter
    tracking/     WP6   null (done), mlflow
    experiments/  WP6   smoke, train_agent, compare
  cli             hydra entry point
configs/          one YAML per component, `_target_` + `config:` block
tests/            CPU by default; markers gpu, network, slow
```

### 3.3 Contract summary

Read the docstrings in `core/`. The rules every implementation follows:

- Constructor signature is fixed per contract: `AugmentationMethod(*, config, device)`,
  `TSModel(*, config, device, episode_epochs)`, `Benchmark(*, config, device)`,
  `Featurizer(*, config, n_datasets, action_dim, batch_envs, device)`,
  `RewardFunction(*, config)`, `Environment(*, config, device, tsmodel, augmentation,
  benchmark, featurizer, reward)` where `tsmodel` is a factory, `Agent(*, config, device)`.
- Every tensor an environment, method, model or benchmark returns lives on `device`.
- Randomness comes from the `torch.Generator` passed in, never from the global stream.
- `TrainStats` fields are 0-d tensors. `WindowSet` targets may hold NaN: zero loss.
- `WindowSet` pads the context on the left and the target on the right with NaN, so one
  tensor holds datasets of different frequencies.

## 4. Performance rules

Training time is the budget. These rules are not optional:

1. No host synchronization inside `env.step`: no `.item()`, `.cpu()`, `.tolist()`, Python
   `float()` on a tensor, or `if tensor:`. Trackers convert once per episode through `to_host`.
   GPU tests may assert this with `torch.cuda.set_sync_debug_mode("error")`.
2. KernelSynth samples on the GPU. The input grid is fixed, so the 32 bank Gram matrices are
   computed once (32 × 1024 × 1024 fp32 = 128 MB). A composed kernel is an element-wise sum or
   product of bank matrices. Factorize in chunks with `torch.linalg.cholesky_ex`, add jitter,
   retry the failed rows with 10× jitter, never fall back to the CPU.
3. The K tsmodels train in lockstep in one process. v1 loops over K; a stacked
   `torch.func` version is a later optimization behind the same interface.
4. The tsmodel trains with bf16 autocast, AdamW, a cosine schedule over the episode, and
   `torch.compile` as an opt-in flag. Windows are cut on the device with `gather`.
5. The reward scorer forwards the full reward split every step by default
   (`reward_subset_per_dataset: null`), in batches of `score_batch_size`, under `torch.no_grad`
   and bf16. On the default benchmark this costs 0.34 s per step for K = 4, 8 % of the step,
   and makes the reward exact. A random subset of 128 windows per dataset gave a reward noise
   of 0.03 against a true improvement of 0.016 per step after step 10. An integer value scores
   that many random windows per dataset and step; use it for large benchmarks.
6. The benchmark window sets are built once per process and kept on the device.
7. Config validation happens at startup, before any data or weight loads.

## 5. Work packages

Each work package is one branch, one worktree, one pull request. The owner implements the
listed files, the configs that name them, and the tests. The owner touches no file outside
the package except its configs and tests. Shared changes to `core/` go through a separate
small pull request first.

### WP1 — data (`feat/data`)

Files: `components/data/{gifteval,windows,cache,download}.py`, `scripts/download_benchmarks.py`,
`configs/env/benchmark/gifteval_{small,full}.yaml`, `tests/test_data_*.py`.

- `GiftEvalBenchmarkConfig(BenchmarkConfig)`: `root`, `repo_id`, `datasets` (ids `name/freq`),
  `max_total_gb`, `windows_per_series_stride`, `seed`.
- Download only the selected dataset directories from `Salesforce/GiftEval` with
  `huggingface_hub.snapshot_download(allow_patterns=...)`. Refuse to exceed `max_total_gb`;
  report the size of each dataset before download with the Hub file metadata.
- Read the Arrow files with `pyarrow` (no `datasets`, no `gluonts`). One series per row:
  `item_id`, `start`, `freq`, `target` (1 or more variates). Treat each variate as its own
  univariate series, tagged with the dataset.
- Splits per series along time: drop the last 10% (official test), then 85% reward and 15%
  comparison. The windows of one split never read values of another split.
- Windows: context and horizon by frequency (`BenchmarkConfig`), cut with a fixed stride,
  then subsampled to `*_windows_per_dataset` with the config seed. Context left-padded with
  NaN to `max_context_length`, target right-padded to `max_horizon`. Precompute
  `naive_loss_sum` with `seasonal_naive` and the 21 Timestral knots, and `abs_sum`.
- Cache each dataset's two `WindowSet` tensors under `root/<name>/<freq>/<hash of config>.pt`;
  the benchmark loads from the cache when present.
- Tests: a synthetic Arrow fixture written by the test (no network) covers parsing, splits,
  padding, naive scores and the cache round trip; `@pytest.mark.network` loads `covid_deaths/D`
  end to end; skill of the naive forecaster is 1.0 by construction.

### WP2 — augmentation (`feat/augmentation`)

Files: `components/augmentation/{kernelsynth,bank}.py`, `tests/test_kernelsynth.py`.

- Port the 32-kernel bank of `chronos-forecasting/scripts/kernel-synth.py` to torch on the
  grid `linspace(0, 1, L)`: 21 `ExpSineSquared` (length_scale 1, the listed periodicities),
  `DotProduct(sigma_0 in {0, 1, 10})`, `RBF(length_scale in {0.1, 1, 10})`,
  `RationalQuadratic(alpha in {0.1, 1, 10}, length_scale 1)`, `WhiteKernel(noise in {0.1, 1})`,
  `ConstantKernel(1)`. Keep the bank order and the comments of the original file.
- `KernelSynthConfig(AugmentationConfig)`: `max_kernels`, `logit_bound`, `jitter`,
  `max_jitter_retries`, `chunk_size`.
- Action: 32 + 5 + 1 logits in `[-logit_bound, logit_bound]`; names
  `kernel/<name>`, `count/<j>`, `op/mul`. `uniform_action()` is all zeros and reproduces the
  Chronos marginals: uniform kernels with replacement, uniform count in 1..5, P(×) = 0.5.
- `sample`: per environment, draw counts, kernels and operators from the action's
  distributions, compose Gram matrices left to right as in the original, factorize in chunks,
  multiply by standard normal draws. All randomness from the passed generator.
- `describe`: kernel-usage probabilities, count probabilities, P(×).
- Tests: with the uniform action the empirical count histogram is uniform (chi-square at
  n = 20000); a one-hot action (large logit) uses only that kernel; determinism per seed;
  outputs finite; CPU test at L = 64; `@pytest.mark.gpu` throughput test: 4096 × 1024 in under
  2 s on the 4080, and no host synchronization under `set_sync_debug_mode("error")`.

### WP3 — tsmodel (`feat/tsmodel`)

Files: `components/tsmodel/{timestral,windows,loss,lora}.py`, `tests/test_timestral.py`.

- `TimestralBareConfig(TSModelConfig)` with an `architecture` dict that mirrors the
  Timestral `config.json`. `TimestralLoraConfig(TSModelConfig)` with `checkpoint`, `lora_*`,
  `fully_trainable`.
- Read `timestral.model.TimestralModel` and `timestral.forecaster` before writing code.
  `predict_block(inputs, observed, group_ids, horizon_patches)` takes packed rows `(R, T)`
  and returns raw-space quantiles `(R, horizon_patches × patch_size, Q)`; `group_ids =
  arange(R)` for univariate rows. Decide, and document in the module docstring, whether the
  pinball loss is computed in raw space or in the model's normalized space, by reading
  `TimestralModel.scale` and `_predict_window`.
- Training windows: a random split point per series with `context_length` and
  `horizon_length` from the config, the horizon capped at `block_size × patch_size`. Cut with
  `gather` on the device. Shuffle with the passed generator.
- `train_epoch`: AdamW, bf16 autocast, gradient clipping, cosine schedule over
  `episode_epochs × steps_per_epoch`, `torch.compile` opt-in. Return `TrainStats` as 0-d
  tensors with no `.item()`.
- `forecast`: left-pad the context with NaN to a multiple of the patch size, mark NaN as
  unobserved, round the horizon up to patches, truncate the output.
- `reset`: bare re-initializes every parameter from the seed; LoRA reloads the checkpoint
  through `timestral.loading.load_model` and re-initializes the adapters from the seed.
- LoRA with `peft.LoraConfig(target_modules=[...])` on `wQKV`, `wO`, `w12`, `w3`;
  `modules_to_save` for `head` and `representation`.
- Tests: a tiny architecture (`d_model 32`, 2 blocks, `patch_size 8`) on the CPU: the loss
  on a sine batch decreases over 10 epochs; forecast shapes and monotone quantiles; `reset`
  restores the initial weights exactly; `state_dict` round trip; `@pytest.mark.gpu`: one epoch
  of the real 5M architecture on 512 series, timed and logged; `@pytest.mark.network`: load
  `mistralai/Timestral-5M`, zero-shot forecast of a sine has lower pinball loss than the naive
  forecast.

### WP4 — environment (`feat/env`)

Files: `components/env/{curriculum,featurizers,rewards,scoring}.py`, `tests/test_env_*.py`.

- `scoring.score(model, windows, *, batch_size) -> (scores (d,), aggregate ())` using
  `core.metrics`, `torch.no_grad`, bf16 autocast when the model is on CUDA.
- `StatsFeaturizer`: the four groups of decision 4, running mean and variance per feature
  (Welford, on the device), `observation_spec` computed from `n_datasets` and `action_dim`.
- `ImprovementReward`: `scale × (agg_prev - agg)` plus the weighted training-quality terms.
- `CurriculumEnv`: builds K tsmodels from the factory, holds the benchmark window sets,
  draws a reward subset per step with the episode generator, runs the step of section 3.1,
  fills `info` with `scores`, `aggregate`, the train stats and the action, and sets `done`
  at step E. `evaluate()` scores the full comparison set per tsmodel.
- Tests with `components/fakes`: shapes of every output; `done` only at step E; the sum of
  rewards over an episode equals `agg_0 - agg_E` when the quality weights are zero; two runs
  with the same seed give equal rewards; the reward subset changes between steps;
  `observation_spec` matches the observation; `@pytest.mark.gpu`: one fake episode under
  `torch.cuda.set_sync_debug_mode("error")` raises nothing.

### WP5 — agents (`feat/agent`)

Files: `components/agents/{fixed,random,cmaes,gymnasium_adapter}.py`,
`components/agents/torchrl/{adapter,sac,ppo,networks}.py`, `tests/test_agents_*.py`.

- `FixedActionAgent`: the uniform action or a given vector. `RandomAgent`: uniform in the
  box. Both have `episodes = 0` and return a policy without training.
- `CMAESAgent`: `cma.CMAEvolutionStrategy` over one static action; each episode evaluates K
  candidates, one per lockstep environment, with the fitness equal to the final aggregate from
  `env.evaluate()`. The policy returns the best mean.
- `torchrl/adapter.py`: `TorchRLEnv(EnvBase)` with `batch_size = [K]`, `Composite`
  observation spec from `observation_spec`, `Bounded` action spec from `BoxSpace`, and
  `_reset`/`_step` that call the `Environment`. `check_env_specs` passes.
- `SACAgent` and `PPOAgent`: TorchRL loss modules, a replay buffer on the device for SAC,
  the policy as an MLP over the flattened observation, `fit` runs `config.episodes` episodes
  and logs through the tracker, `save`/`load` with `torch.save`.
- `gymnasium_adapter.py`: `GymEnv(gymnasium.Env)` for K = 1, NumPy in and out, for
  cross-checks with other libraries. Not used in v1 experiments.
- Tests with `FakeEnvironment`: every agent's `fit` returns a policy whose actions have shape
  `(K, a)` inside the box; SAC and PPO on the fake bandit reach a mean distance to the target
  below the random policy after 30 short episodes on the CPU; `check_env_specs` on the adapter;
  `save`/`load` round trip.

### WP6 — experiments and tracking (`feat/experiments`)

Files: `components/tracking/mlflow.py`, `components/experiments/{smoke,train_agent,compare}.py`,
`cli.py` completion, `tests/test_experiments.py`, `scripts/`.

- `MLflowTracker`: local file store, buffered steps written with `log_batch` once per episode.
- `SmokeExperiment`: build env, agent and tracker from the config, run `episodes` episodes
  with the agent's policy, return the final aggregate. `uv run autotsaugment-run` with the
  defaults must finish in seconds on the CPU.
- `TrainAgentExperiment`: `agent.fit`, then `eval_seeds` fresh episodes with the policy,
  the comparison skill per dataset and the aggregate per seed, the area under the aggregate
  curve, written to `metrics/train_agent-<agent>.json`. Checkpoints under `checkpoint_dir`
  and `dvc add` instructions in the log.
- `CompareExperiment`: the same evaluation for a list of policies, one table.
- Tests: the smoke experiment through `hydra.compose` with `env=fake`; the metrics file has
  the documented keys.

## 6. Order of work

1. Wave 1, in parallel: WP1, WP2, WP3, WP4, WP5. Each starts from the bootstrap commit on
   `main`. Each ends with a pull request that passes ruff, ty and the CPU tests.
2. The user merges wave 1. Conflicts are limited to `pyproject.toml` (new dependencies) and
   `tests/conftest.py` (new fixtures).
3. Wave 2: WP6 on top of the merged `main`, then the first real smoke run on the GPU with
   `env=default experiment=smoke`.
4. Experiment 1: `experiment=train_agent agent=sac`, `agent=cmaes`, then `experiment=compare`.

## 7. Later work, not in v1

- A probe-batch embedding of the tsmodel as a fifth observation group.
- A multi-method hierarchical action: method id plus method parameters.
- Stacked K-model training with `torch.func`, and multi-GPU collection.
- A tslake adapter for `Benchmark` once the lake is in production.
- A leakage check of the LoRA variant: Mistral's stage-two post-training uses real data.
