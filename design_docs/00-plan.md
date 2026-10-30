# 00 — Plan: AutoTSAugment v1

Status: locked 2026-10-23 after a 12-question design review. Updated 2026-10-26 to match the
merged wave 1 (pull requests #1 to #5) and its follow-up pull request. This document is the
contract for the first implementation. Change it only through a pull request.

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
| 1 | Data splits | GIFT-Eval official test part (last 10%) is never used. The rest is split along time: 85% reward split (scored every step), 15% comparison split (scored once per episode). A comparison context may read the end of the reward split (`comparison_context_may_cross`, default true). |
| 2 | Episode | One full tsmodel training run from fixed initial weights. E = 50 epochs, N = 4096 series of 1024 steps per epoch, K = 4 tsmodels in lockstep on one GPU. Target: under 10 minutes per episode. All configurable. |
| 3 | Action | Continuous box of 39 logits: 33 over the kernel bank, 5 over the number of kernels (1..5), 1 for P(×) versus P(+). The bounds are ±`logit_bound` = ±5. Zeros = Chronos recipe. Bank hyperparameters fixed. |
| 4 | Observation | Dict of 4 normalized groups: `progress`, `train_stats`, `reward_vector`, `action_history`. 2 + 5 + 3d + a dims: 70 with d = 8 and a = 39. Featurizer is swappable. |
| 5 | Reward | `r_t = agg_{t-1} - agg_t`. Per-dataset vector and training-quality metrics in `info`, with weights that default to zero. Reward windows: a random subset per step, the same for all K tsmodels, from its own random stream. |
| 6 | Data | Standalone `components/data`, no tslake dependency. GIFT-Eval at a pinned Hub revision, at most 10 GB on disk, 8 default datasets over 7 domains. |
| 7 | tsmodel | Timestral-5M bare (random init) first, LoRA on the pretrained weights second, one `TSModel` interface. Pinball loss on the 21 knots, in the normalized frame of the model. Windows: context 768, horizon 256. |
| 8 | Agent | SAC first, PPO second, TorchRL losses. Data collection through our own episode loop, `run_episode`. Own `Environment` contract with TorchRL and gymnasium adapters. Baselines: fixed uniform, random, CMA-ES. |
| 9 | Config | Hydra groups `experiment/`, `agent/`, `env/` (`tsmodel/`, `augmentation/`, `benchmark/`, `featurizer/`, `reward/`), `tracker/`. Pydantic validates each component config at startup. `_target_` is the registry. |
| 10 | Conventions | tslake rule set: ruff ALL with documented ignores, ty strict, Google docstrings, jaxtyping with opt-in beartype, pre-commit chain. Python 3.13. |
| 11 | Tracking | `Tracker` interface: `NullTracker` (`tracker=none`), `MLflowTracker` with a local SQLite store under `tracking_uri` (`mlruns/mlflow.db` and `mlruns/artifacts/`). MLflow 3.17 refuses the plain file store. Step metrics buffered on the device, flushed once per episode. DVC for keeper checkpoints in `models/`, `metrics/<experiment>-<label>.json` and `metrics/compare.json`. |
| 12 | Git | Bootstrap on `main`. One branch per work package in its own worktree, one pull request per branch, merged by the user. |

## 3. Architecture

### 3.1 One step

```
agent.policy(obs) ──► action (k, 39)
                        │
                        ▼
augmentation.sample(action, N) ──► series (k, N, 1024)      GPU, batched Cholesky
                        │
                        ▼
tsmodel[i].train_epoch(series[i]) for i in 0..K-1 ──► TrainStats (k,)   bf16 autocast
                        │
                        ▼
scorer(tsmodel[i], reward subset of the step) ──► scores (k, d), aggregate (k,)
                        │
                        ▼
reward(record) ──► r (k,)        featurizer(record) ──► obs (dict of (k, ...))
                        │
                        ▼
run_episode: tracker.log_step(...)  buffered; flushed at log_episode
```

`reset(seed)` resets the K tsmodels, scores them once on a reward subset, and gives these
scores to the featurizer through an `EpochRecord` with `epoch = 0`. At the end of an episode,
`env.evaluate()` scores the full comparison set once. The environment never logs: the episode
loop and the experiments read `StepResult.info`.

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
    agent         Agent, Policy, StatefulPolicy
    tracker       Tracker, to_host
    experiment    Experiment, write_metrics
  components/
    fakes         fast fakes of every contract (tests, smoke runs, agent development)
    augmentation/ WP2   bank, kernelsynth
    tsmodel/      WP3   timestral (bare, lora), network, loss, windows, schedule, lora
    data/         WP1   gifteval, download, arrow, windows, cache
    env/          WP4   curriculum, featurizers, rewards, scoring, history
    agents/       WP5   loop, policies, fixed, random, cmaes,
                        torchrl/{adapter,networks,sac,ppo}, gymnasium_adapter
                  WP7   grpo, transformer, distributions
    tracking/     WP6   null, mlflow, prefixed
    experiments/  WP6   common, smoke, train_agent, compare
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
  benchmark, featurizer, reward)` where `tsmodel` and `featurizer` are factories (Hydra
  `_partial_: true`), `Agent(*, config, device)`.
- Every tensor an environment, method, model or benchmark returns lives on `device`.
- Randomness comes from the `torch.Generator` passed in, never from the global stream.
- `TrainStats` fields are 0-d tensors. `WindowSet` targets may hold NaN: zero loss.
- `WindowSet` pads the context on the left and the target on the right with NaN, so one
  tensor holds datasets of different frequencies.
- `WindowSet.quantile_levels` holds the levels of `naive_loss_sum`. A tsmodel scored on the
  set must forecast the same levels. `CurriculumEnv` checks this once, at construction.
- `WindowSet.rows_by_dataset` holds the rows of each dataset. It is computed once per
  instance with `torch.nonzero`, which synchronizes. Read it first outside the step, for
  example in a constructor. `WindowSet.subset` uses it.
- `Environment.uniform_action()` returns the reference action `(a,)` of the augmentation
  method: zeros for KernelSynth, the Chronos recipe. The fixed baseline uses it.
- `Environment.state_dict()` and `load_state_dict(state)` carry the state a saved policy
  needs to reproduce its observations. The experiments save it next to the agent
  checkpoint. `CurriculumEnv` gives the featurizer state under the prefix `featurizer/`.
  The default is an empty state.
- `Featurizer.state_dict()` and `load_state_dict(state)` carry the cross-episode state only:
  the running normalization moments, not the per-episode buffers. `StatsFeaturizer` uses the
  keys `<group>/mean`, `<group>/m2` and `<group>/count`. A wrong key or shape raises
  `ValueError`.

## 4. Performance rules

Training time is the budget. These rules are not optional:

1. No host synchronization inside `env.step`: no `.item()`, `.cpu()`, `.tolist()`, Python
   `float()` on a tensor, `torch.nonzero`, or `if tensor:`. Trackers convert once per episode
   through `to_host`. GPU tests assert this with `torch.cuda.set_sync_debug_mode("error")`
   for the environment, KernelSynth and Timestral.
2. KernelSynth samples on the GPU. The input grid is fixed, so the 33 bank Gram matrices are
   computed once (33 × 1024 × 1024 fp32 = 132 MB). A composed kernel is an element-wise sum or
   product of bank matrices; `torch.compile` compiles the composition on CUDA. Factorize in
   chunks with `torch.linalg.cholesky_ex`. Add a jitter relative to the mean of the Gram
   diagonal, `jitter × mean(diag K)`. Retry the failed rows with 10× jitter, in fixed batches
   of `chunk_size` rows. Never fall back to the CPU: a row that still fails gets an
   independent Gaussian draw with the marginal variance of its kernel.
3. The K tsmodels train in lockstep in one process. v1 loops over K; a stacked
   `torch.func` version is a later optimization behind the same interface.
4. The tsmodel trains with bf16 autocast, AdamW, a linear warmup and a cosine schedule over
   the episode, and `torch.compile` of the loss as a flag. The Timestral YAML files set
   `compile: true` (2.8× faster per epoch, about 17 s of cold compile per process); the fake
   tsmodel keeps `compile: false`. Windows are cut on the device with `gather`.
5. The reward scorer forwards `reward_subset_per_dataset` windows per dataset per step, in
   batches of `score_batch_size`, under `torch.no_grad` and bf16. The full reward set is never
   scored inside an episode. The rows of each dataset come from `rows_by_dataset`, read once
   in the constructor.
6. The benchmark window sets are built once per process and kept on the device.
7. Config validation happens at startup, before any data or weight loads. The environment
   checks the devices and the quantile levels of its components at construction.

## 5. Work packages

Each work package is one branch, one worktree, one pull request. The owner implements the
listed files, the configs that name them, and the tests. The owner touches no file outside
the package except its configs and tests. Shared changes to `core/` go through a separate
small pull request first.

### Status

| WP | Package | Pull request | Merged | CPU tests, branch alone |
|---|---|---|---|---|
| WP1 | data | #4 | 2026-10-26 | 132 |
| WP2 | augmentation | #3 | 2026-10-26 | 85 |
| WP3 | tsmodel | #2 | 2026-10-26 | 74 |
| WP4 | env | #1 | 2026-10-26 | 83 |
| WP5 | agents | #5 | 2026-10-26 | 97 |
| WP6 | experiments, tracking | #7 | no | 330 |
| WP7 | GRPO, transformer policy | #9 | no | 364 |

Each branch count includes the tests of the bootstrap commit, so the counts overlap. Together
on `main`, before the follow-up pull request, the CPU suite gives 275 passed. The follow-up
pull request (`feat/core-followups`, #6) adds the contract methods of section 3.3, tunes the
real-run configs and brings the count to 294. WP6 (#7) builds on #6 and adds 36 CPU tests:
330 passed. The GPU suite (`-m gpu`) gives 9 passed. WP7 (#9) builds on #7 and adds 34 CPU tests
(364 passed) and 1 GPU test (10 passed).

### WP1 — data (`feat/data`, #4)

Files: `components/data/{gifteval,download,arrow,windows,cache}.py`,
`scripts/download_benchmarks.py`, `configs/env/benchmark/gifteval_{small,full}.yaml`,
`tests/test_data_*.py`.

- `GiftEvalBenchmarkConfig(BenchmarkConfig)`: `root`, `repo_id`, `revision`, `max_total_gb`,
  `datasets` (ids `name/freq`), `stride_in_horizons`, `comparison_context_may_cross`,
  `min_windows_per_split`, `seed`, `quantile_levels` (default: the 21 Timestral knots). The
  config validates ids, frequencies, window lengths, duplicates and levels at startup.
- Download only the shards of the selected datasets from `Salesforce/GiftEval`, at the pinned
  `revision`, with `huggingface_hub.snapshot_download(allow_patterns=...)`. The YAML files pin
  `30841734ac5c…` (main on 2026-10-23). The shards go to `root/hub/<repo>/<revision>`, where
  `<repo>` is `repo_id` with `/` replaced by `--`. The plan reports the size of each dataset
  from the Hub file metadata and refuses to exceed `max_total_gb`. The benchmark downloads
  missing shards itself.
- `arrow.py` reads the Arrow shards with `pyarrow` (no `datasets`, no `gluonts`): IPC stream,
  then IPC file. One series per row: `item_id`, `start`, `freq`, `target` (1 or more variates).
  Each variate is its own univariate series, tagged with the dataset. Arrow nulls become NaN.
- Splits per series of length T: `test_start = floor(0.9 T)`. No window reads an index at or
  after `test_start`, so the official test part stays untouched. The reward part is
  `[0, floor(0.85 test_start))`. The comparison part is the rest, up to `test_start`. The
  horizon of a window lies inside its split. A reward window reads only the reward part. A
  comparison context may read the end of the reward part, as a GIFT-Eval test window reads
  the training part; `comparison_context_may_cross: false` keeps the whole comparison window
  inside the comparison part.
- Windows: context and horizon by frequency (`BenchmarkConfig`). Candidates are anchored at
  the end of the split, `stride_in_horizons` horizons apart. A candidate is valid when its
  target has one observed value and its context `season` observed values. A dataset with
  fewer than `min_windows_per_split` (32) valid windows in a split raises
  `TooFewWindowsError`. Each split is subsampled to `*_windows_per_dataset` with
  `derive_seed(seed, dataset_id, split)`, so a new dataset does not change the windows of the
  others. Context left-padded with NaN to `max_context_length`, target right-padded to
  `max_horizon`. Precompute `naive_loss_sum` with `seasonal_naive` on a forward-filled context
  and the `quantile_levels`, and `abs_sum`. The `WindowSet` carries the levels.
- Cache: one file per dataset id, `root/cache/<name>__<freq>__<sha1>.pt`. The SHA-1 covers
  every field that changes the windows, and the cache version. Version 2 stores the quantile
  levels. With a complete cache, the benchmark reads no Arrow and makes no network call. A
  cache write that would exceed `max_total_gb` is skipped with a warning.
- Default datasets (`gifteval_small.yaml`, 7 domains): `ett1/H`, `electricity/H`,
  `jena_weather/H`, `restaurant/D`, `us_births/D`, `bizitobs_l2c/H`, `LOOP_SEATTLE/H`,
  `m4_hourly/H`. `covid_deaths/D` and `car_parts_with_missing/M` have a comparison part
  shorter than one horizon; `restaurant/D` and `us_births/D` replace them.
- Sizes: the 8 defaults are 67 MB of shards, with 3545 reward and 2976 comparison windows
  and 15 MB of window caches. The 44 usable ids of `gifteval_full.yaml` are 1174 MB of
  shards. The other 11 of the 55 GIFT-Eval ids give too few windows.
- Tests: a synthetic Hub-shaped tree of Arrow shards written by the test (no network) covers
  parsing, the splits with both values of the crossing flag, padding, naive scores, the
  window minimum and the cache round trip; `@pytest.mark.network` loads `covid_deaths/D` end
  to end; skill of the naive forecaster is 1.0 by construction.

### WP2 — augmentation (`feat/augmentation`, #3)

Files: `components/augmentation/{kernelsynth,bank}.py`, `tests/test_kernelsynth{,_bank}.py`.

- Port the 33-kernel bank of `chronos-forecasting/scripts/kernel-synth.py` to torch on the
  grid `linspace(0, 1, L)`: 21 `ExpSineSquared` (length_scale 1, the listed periodicities),
  3 `DotProduct(sigma_0 in {0, 1, 10})`, 3 `RBF(length_scale in {0.1, 1, 10})`,
  3 `RationalQuadratic(alpha in {0.1, 1, 10}, length_scale 1)`, 2 `WhiteKernel(noise in
  {0.1, 1})`, 1 `ConstantKernel(1)`. Keep the bank order and the comments of the original
  file. The Gram matrices match scikit-learn to 1e-13 in float64.
- `KernelSynthConfig(AugmentationConfig)`: `max_kernels` (1..8, default 5), `logit_bound`
  (5.0 in the YAML; the code default is 4.0), `jitter` (1e-4, relative to the mean of the
  Gram diagonal), `max_jitter_retries` (3), `chunk_size` (256).
- Action: 33 + `max_kernels` + 1 logits, 39 with the defaults, in
  `[-logit_bound, logit_bound]`; names `kernel/<index>_<name>`, `count/<j>`, `op/mul`.
  `uniform_action()` is all zeros and reproduces the Chronos marginals: uniform kernels with
  replacement, uniform count in 1..5, P(×) = 0.5. With a bound of 5, a one-hot action puts up
  to 0.9985 of the kernel mass on one kernel; a bound of 4 gives 0.9894.
- `sample`: per environment, draw counts, kernels and operators from the action's
  distributions, compose Gram matrices left to right as in the original, factorize in chunks,
  multiply by standard normal draws. All randomness from the passed generator. `last_stage`,
  shape `(k, n)` on the device, tells which factorization pass made each series.
- `describe`: kernel-usage probabilities, count probabilities, P(×).
- Tests: with the uniform action the empirical count frequencies are uniform (3 standard
  errors) and the kernel frequencies pass a chi-square test; a one-hot action (large logit)
  uses only that kernel; determinism per seed; outputs finite; CPU test at L = 64;
  `@pytest.mark.gpu` throughput test: 4096 × 1024 in under 2 s on the 4080 (measured 0.67 s,
  2.19 GiB peak), and no host synchronization under `set_sync_debug_mode("error")`. The
  relative jitter gave 0 Cholesky failures over 16384 uniform series.

### WP3 — tsmodel (`feat/tsmodel`, #2)

Files: `components/tsmodel/{timestral,network,windows,loss,schedule,lora}.py`,
`scripts/bench_tsmodel.py`, `tests/test_timestral.py`.

- `TimestralBareConfig(TSModelConfig)` with an `architecture` dict that mirrors the
  Timestral `config.json`. `TimestralLoraConfig(TSModelConfig)` with `checkpoint`,
  `patch_size`, `block_size`, `lora_*`, `fully_trainable`. Both have `warmup_fraction` (0.05)
  and `min_lr_fraction` (0.1).
- `network.py`: our own `UnivariateDecoder` runs the submodules of `TimestralModel` for
  univariate rows, with cached tensors. It replaces `predict_block`, which synchronizes the
  host: it calls `torch.unique`, and `QuantileHead` builds a tensor from a Python list. Tests
  show that the decoder equals `predict_block` and `TimestralForecaster`.
- Loss space: the pinball loss on the 21 knots is computed in the normalized frame of the
  model: causal per-patch statistics, then `asinh`, and the advancing refine frame of
  `TimestralForecaster` for the horizon patches. The loss does not change when a series is
  scaled or shifted, and LoRA starts from the objective of the checkpoint. The module
  docstring of `timestral.py` gives the full argument.
- Training windows: a random split point per series with `context_length` and
  `horizon_length` from the config, the horizon capped at `block_size × patch_size`. Cut with
  `gather` on the device. Shuffle with the passed generator.
- `train_epoch`: AdamW (weight decay on matrices only), bf16 autocast, gradient clipping, a
  linear warmup and a cosine schedule over `episode_epochs × steps_per_epoch`,
  `torch.compile` of the loss as a flag. Return `TrainStats` as 0-d tensors with no `.item()`.
- `forecast`: left-pad the context with NaN to a multiple of the patch size, mark NaN as
  unobserved, decode block by block in the refine frame, cap extreme values, truncate the
  output.
- `reset`: bare re-initializes every parameter from the seed; LoRA restores a CPU copy of
  the checkpoint, loaded once through `timestral.loading.load_model`, and re-initializes the
  adapters from the seed.
- LoRA through `peft.inject_adapter_in_model` on `wQKV`, `wO`, `w12`, `w3`; `head` and
  `representation` are fully trainable. `lora_dropout` must be 0: dropout draws from the
  global random stream.
- Speed on the RTX 4080 SUPER (4096 series of 1024 steps, batch 128, bf16):

  | Variant | Seconds per epoch, eager | Seconds per epoch, compiled |
  |---|---|---|
  | bare 5M | 0.454 | 0.160 |
  | LoRA 5M | 0.59 | 0.20 |

  A cold compile adds about 17 s per process. One episode of K = 4, E = 50 trains in about
  32 s compiled and 91 s eager.
- Tests: a tiny architecture (`d_model 32`, 2 blocks, `patch_size 8`) on the CPU: the loss on
  a sine batch decreases over 10 epochs; forecast shapes and monotone quantiles; `reset`
  restores the initial weights exactly; `state_dict` round trip; `@pytest.mark.gpu`: one epoch
  and one forecast of the 5M bare and LoRA models, with and without compile, under
  `set_sync_debug_mode("error")`; `@pytest.mark.network`: load `mistralai/Timestral-5M`,
  zero-shot forecast of a sine has lower pinball loss than the naive forecast, and
  `forecast` equals `TimestralForecaster`.

### WP4 — environment (`feat/env`, #1)

Files: `components/env/{curriculum,featurizers,rewards,scoring,history}.py`,
`tests/test_env_*.py`.

- `scoring.score(model, windows, *, batch_size) -> (scores (d,), aggregate ())` using
  `core.metrics`, `torch.no_grad`, bf16 autocast when the model is on CUDA. A naive
  forecaster scores exactly 1.0.
- `StatsFeaturizer`: the four groups of decision 4, a running mean and variance per feature
  (parallel Welford, on the device). The moments never freeze and persist across episodes;
  `state_dict` holds them. `observation_spec` is computed from `n_datasets` and `action_dim`.
- `ImprovementReward`: `scale × (agg_prev - agg)` minus the weighted variance and slope of
  the training loss. The weights are zero by default.
- `CurriculumEnv`: builds K tsmodels from the factory and the featurizer from its factory. At
  construction it checks the devices, checks that every tsmodel forecasts the quantile levels
  of the window sets, and reads `rows_by_dataset`. Each step runs section 3.1, and `done` is
  True at step E. `evaluate()` scores the full comparison set per tsmodel.
- Random streams derive from the `reset` seed: `("tsmodel", i)`, `("train", i)`,
  `"augmentation"` and `"reward_subset"`. With the same seed, two policies score the same
  windows at every step, so their rewards are paired.
- `info` of a step, each entry with a leading k: `scores` (k, d) and `aggregate` (k,) on the
  reward subset; `train/<name>` (k,) for each `TrainStats` field and `series`; `action`
  (k, a), the clipped action; `reward_rows` (k, m), int64, equal for all k.
- Initial scores: `reset` calls `featurizer.reset()`, then the featurizer with an
  `EpochRecord` with `epoch = 0` that holds the scores of the untrained tsmodels. A record
  with `epoch = 0` sets the scores only.
- Tests with `components/fakes`: shapes of every output; `done` only at step E; the sum of
  rewards over an episode equals `agg_0 - agg_E` when the quality weights are zero; two runs
  with the same seed give equal rewards; the reward subset changes between steps;
  `observation_spec` matches the observation; the state round trip through `torch.save`;
  other quantile levels raise; `@pytest.mark.gpu`: one fake episode under
  `torch.cuda.set_sync_debug_mode("error")` raises nothing.

### WP5 — agents (`feat/agent`, #5)

Files: `components/agents/{loop,policies,fixed,random,cmaes,gymnasium_adapter}.py`,
`components/agents/torchrl/{adapter,sac,ppo,networks}.py`, `tests/test_agents_*.py`.

- `run_episode(env, policy, *, seed, tracker, step_offset, episode, deterministic,
  on_step)`: the one episode loop of every agent and experiment. It resets, runs E steps and
  calls `env.evaluate()`, with no host synchronization. `episode` is the index for
  `log_episode`. `on_step` is a callback after each step: SAC and PPO store transitions and
  train in it. The result holds the returns, the final skill and the aggregate curve.
- Step metrics: `reward`, every `(k,)` entry of `info` under its own name (for example
  `aggregate` and `train/<name>`), and `score/<dataset>` from `info["scores"]`. Episode
  metrics: `return`, `comparison/aggregate`, `comparison/<dataset>`. Each is a mean over k, a
  0-d device tensor.
- `FixedActionAgent`: the configured action or `env.uniform_action()`. `RandomAgent`:
  uniform in the box. Both have `episodes = 0` and return a policy without training.
- `CMAESAgent`: `cma.CMAEvolutionStrategy` over one static action; each episode evaluates K
  candidates, one per lockstep environment. The fitness is the geometric mean over datasets
  of the comparison skill from `env.evaluate()`. `fixed_seed: true` evaluates every candidate
  on the same episode seed. The policy returns the best evaluated candidate (`final: best`);
  `final: mean` returns the final search mean.
- `torchrl/adapter.py`: `TorchRLEnv(EnvBase)` with `batch_size = [K]`, `Composite`
  observation spec from `observation_spec`, `Bounded` action spec from `BoxSpace`, and
  `_reset`/`_step` that call the `Environment`. `check_env_specs` passes.
- `SACAgent` and `PPOAgent`: TorchRL loss modules (`SACLoss`, `ClipPPOLoss` with GAE), a
  replay buffer on the device for SAC, a squashed Gaussian actor over the flattened
  observation. They collect data through `run_episode`, not through a TorchRL collector.
  `fit` forks the global torch stream and seeds it from the run seed. `sac.yaml` sets
  `updates_per_step: 4`; PPO clips the gradient norm at `max_grad_norm` (1.0).
- Checkpoints: `save(path)` and `load(path)` treat `path` as a directory that holds
  `agent.pt`.
- `gymnasium_adapter.py`: `GymEnv(gymnasium.Env)` for K = 1, NumPy in and out, for
  cross-checks with other libraries. Not used in v1 experiments.
- Tests with `FakeEnvironment`: every agent's `fit` returns a policy whose actions have shape
  `(K, a)` inside the box; SAC and PPO on the fake bandit reach a mean distance to the target
  below the random policy after 40 (SAC) and 60 (PPO) short episodes on the CPU;
  `check_env_specs` on the adapter; `save`/`load` round trip. `FakeEnvironment` reports
  `aggregate` and `scores` like `CurriculumEnv`, and `shared_target` gives one hidden target
  to all K slots for the CMA-ES tests.

### WP6 — experiments and tracking (`feat/experiments`, #7)

Files: `components/tracking/{mlflow,prefixed}.py`,
`components/experiments/{common,smoke,train_agent,compare}.py`, `cli.py` completion,
`configs/tracker/{mlflow,none}.yaml`, `configs/experiment/*.yaml`,
`tests/test_{experiments,tracking_mlflow,configs}.py`.

- `MLflowTracker`: a `tracking_uri` without a scheme is a directory, relative to the working
  directory (the repository root, because `hydra.job.chdir` is false). The directory holds
  the SQLite database `mlflow.db` and `artifacts/`. MLflow 3.17 refuses the plain file store.
  A URI with a scheme goes to MLflow unchanged. The UI:
  `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- `log_step` appends to a buffer and touches no tensor. `flush` moves the buffer to the host
  with one transfer per device, then writes `log_batch` calls of at most 1000 metrics.
  `log_params` writes the resolved config as dotted keys, 100 per call. Episode metrics go
  under `episode/`, with the episode index as the step. A run carries the git commit as a
  tag. A failed run ends as `FAILED`. `mlflow` is imported in the constructor only.
- `NullTracker`: `configs/tracker/none.yaml`, selected with `tracker=none`. `tracker=null` is
  not valid: in Hydra, `null` removes the group. A run without a tracker group also gets a
  `NullTracker`.
- `PrefixedTracker(tracker, prefix)`: a view that puts `prefix` before every key and leaves
  the tracker open. The evaluation logs under `eval/`. `compare` logs each policy under
  `<policy>/` and its fit under `<policy>/fit/`. Steps and episodes start at 0 in each phase.
- An experiment takes `run_config` and the flat leaves of its YAML file as keyword
  arguments. It validates the leaves into a frozen config (`SmokeConfig`, `TrainAgentConfig`,
  `CompareConfig`) when Hydra builds it, before any data loads. `checkpoint_dir`,
  `metrics_dir` and `checkpoint` paths are relative to the working directory.
- `evaluate_policy(env, policy, *, seeds, tracker, root_seed, step_offset, episode_offset)`:
  one deterministic episode per evaluation seed `s`, with the episode seed
  `derive_seed(root_seed, "eval", s)`. Two policies on the same seeds see the same reward
  windows. The values stay on the device until one `to_host` call. Summary keys, lower is
  better for skill values:

  | Key | Value |
  |---|---|
  | `comparison/aggregate/mean` | The primary metric: the comparison aggregate, mean over K, then over the seeds |
  | `comparison/aggregate/std` | The sample standard deviation over the seeds of the K-mean; 0 for one seed |
  | `comparison/aggregate/seed<s>` | The K-mean of evaluation seed `s` |
  | `comparison/<dataset>/mean` | The comparison skill of one dataset, mean over K and the seeds |
  | `curve/auc/mean` | The secondary metric: the aggregate on the reward subset, mean over the steps, K and the seeds |
  | `return/mean` | The episode return, mean over K and the seeds |
  | `eval/seeds`, `eval/wall_time_s` | The number of seeds, the wall time of the evaluation |

- `SmokeExperiment(name, episodes)`: build the run, fit the agent, evaluate its policy on the
  seeds `0 .. episodes - 1`. It returns the summary and `wall_time_s`. With the defaults
  (fake env, `tracker=none`, CPU, fixed uniform agent), `uv run autotsaugment-run` takes 1.6 s.
- `TrainAgentExperiment(name, eval_seeds, checkpoint_dir, metrics_dir, label)`: fit with the
  run seed, save, then evaluate on `eval_seeds`. `<label>` defaults to the agent class in
  snake case: `sac`, `ppo`, `cmaes`, `fixed_action`, `random`. The checkpoint directory
  `checkpoint_dir/<label>/` holds `agent.pt` (`agent.save`), `agent.yaml` (the agent config
  of the run) and `env.pt` (`env.state_dict()` after the fit, on the CPU, when it is not
  empty). The metrics file is `metrics/<name>-<label>.json`. It adds `train/wall_time_s`,
  `train/episodes` and `wall_time_s` to the summary.
- Git ignores `checkpoints/`, so `dvc add` cannot track a checkpoint there. The log gives the
  commands for a keeper: `dvc add checkpoints/train_agent/<label> -o models/<name>-<label>`,
  `git add models/<name>-<label>.dvc models/.gitignore metrics/<name>-<label>.json`,
  `git commit`, `dvc push`.
- `CompareExperiment(name, eval_seeds, policies, metrics_dir)`: each entry names an agent file
  in `configs/agent/` and an optional `checkpoint`. With a checkpoint, the agent loads it. It
  is built from the `agent.yaml` of the checkpoint when that file exists (same `_target_`
  as the named file, device of the run), so the networks match after command-line
  overrides. Without a checkpoint, the agent is fitted with the run seed: instant for the
  baselines, slow for CMA-ES and the RL agents. One environment serves all policies on the
  same seeds. A policy with `env.pt` gets that state before its evaluation; the others get
  the state after construction. Missing files fail when Hydra builds the experiment, and a
  saved state that does not fit fails before the first episode. Every summary key goes under
  `<policy>/`, with `wall_time_s`, in `metrics/<name>.json`. One table goes to the log.
- Tests: the smoke experiment with the defaults and with `tracker=mlflow`; `train_agent` with
  `fixed_uniform` and with CMA-ES; `compare` with the baselines and a checkpoint, with the
  same numbers as `train_agent`; the SAC numbers of `compare` and `train_agent` are equal
  with `env.pt` and differ without it; a CLI subprocess run; every `_target_` imports. GPU:
  `log_step` without synchronization, `train_agent` and `compare` on CUDA.

### WP7 — GRPO with a transformer memory policy (`feat/grpo`)

Files: `components/agents/{grpo,transformer,distributions}.py`, `core/agent.py`
(`StatefulPolicy`), `components/agents/loop.py` (the reset call), `configs/agent/grpo.yaml`,
`tests/test_agents_{grpo,transformer}.py`.

Why: in experiment 1, PPO learned slowly. Its critic learned a reward of about 0.01 per step, its
MLP policy had no memory, and its entropy bonus held the 39-dimensional policy at the uniform
distribution. WP7 replaces the critic with a group baseline, gives the policy the whole episode
so far, and starts the policy at the fixed recipe.

- D1, group and advantage. A group is every trajectory with the same episode seed: the K
  lockstep environments times `group_episodes` (2) repeats, so G = 8 with K = 4. The group seed
  is `episode_seed(seed, group_index)`. `advantage: step` standardizes the return to go
  `G_t = sum_{s >= t} gamma^(s-t) r_s` (`gamma` 1) over the group at the same step t.
  `advantage: outcome` standardizes the episode return and gives it to every step.
  `normalize_std: false` keeps the mean baseline only. The standard deviation is the sample
  standard deviation, plus `eps = 1e-6`. A group of one trajectory raises `ValueError` in `fit`.
  The loss per token is the PPO clipped surrogate (`clip_epsilon` 0.2), the mean over the tokens
  of the minibatch, with an optional k3 estimate of `KL(new || old)` (`kl_coef` 0) and an
  optional entropy bonus (`entropy_coef` 0). There is no critic. Each update runs `epochs` (4)
  passes over the trajectories of `groups_per_update` (1) groups. A minibatch holds whole
  trajectories (`trajectories_per_minibatch`, null = all). AdamW, learning rate 1e-4, weight
  decay 0.01 on the weight matrices, gradient-norm clip 1.0. The tracker gets
  `grpo/{loss_policy,kl_approx,clip_fraction,entropy,group_return_std,scale_mean}` at the last
  step of each update.
- D2, policy. The token of step t is `concat(o_t, a_{t-1})`, with `a_{-1} = 0`. A linear
  embedding plus a learned position (a table of `max_steps = E` rows, stored in the
  checkpoint; a longer episode raises `ValueError`), `n_layers` pre-norm GELU encoder layers
  with a causal mask, a final layer norm, and two heads at every step: `loc` (zero at
  initialization) and `scale = softplus(linear + b0) + min_scale` (`init_scale` 0.3 at
  initialization). So the initial mean action is the box center, the Chronos recipe. Defaults
  `d_model` 256, 4 layers, 8 heads, dropout 0: 3.22 M parameters with d = 8 and a = 39.
  `d_model` 512 with 8 layers gives 25.3 M. `distributions.py` holds the squashed Gaussian:
  the log-probability uses the stable form of the tanh Jacobian and equals TorchRL's
  `TanhNormal` to 1e-4. `HistoryPolicy` keeps the tokens of the episode, runs the network over
  all of them at each step and acts with the last output. The acting path makes no host
  synchronization (GPU test).
- D3, core. `StatefulPolicy(Policy, Protocol)`, runtime checkable, adds `reset()`.
  `run_episode` calls `policy.reset()` right after `env.reset`.
- D4, agent. `fit` builds the network in a forked global stream seeded with
  `derive_seed(seed, "grpo", "torch")`. The policy noise comes from `("grpo", "policy")` and the
  minibatch order from `("grpo", "minibatch")`. The agent runs
  `episodes // (groups_per_update * group_episodes)` updates and drops the remainder with a
  warning. After each episode, the rollout keeps the tokens of the policy, the pre-squash samples
  and the rewards. The behaviour log-probabilities come from one forward pass before the
  epochs. The checkpoint holds the observation spec, `max_steps`, the network shape fields and
  the network state. The returned policy is deterministic by default.
- Tests: shapes, causality, the initial box center and scale, the log-probability against
  `TanhNormal`, the reset of the policy memory, the group advantages, a smoke run with the
  metric keys, reproducibility, the `save`/`load` round trip, better than random, and the
  target learned on the fake bandit (the plain fake env, `gamma` 0: each reward depends on the
  action of its own step only). `@pytest.mark.gpu`: `fit` on the real environment (2 epochs,
  256 series), `save`/`load`, one `evaluate_policy`, in 23 s with the first compilation. It reads
  the shards from `data/gifteval`, or from `AUTOTSAUGMENT_GIFTEVAL_ROOT` when it is set, and
  skips when they are missing: it never downloads.

## 6. Order of work

1. Wave 1, in parallel: WP1, WP2, WP3, WP4, WP5. Each starts from the bootstrap commit on
   `main`. Each ends with a pull request that passes ruff, ty and the CPU tests.
2. The user merges wave 1 (done on 2026-10-26). Conflicts are limited to `pyproject.toml`
   (new dependencies) and `tests/conftest.py` (new fixtures). A follow-up pull request then
   adds the contract methods of section 3.3, tunes the real-run configs and updates this
   document.
3. Wave 2 (done, #7): WP6 on top of the merged `main` and #6. The first real smoke run on
   the GPU, `uv run autotsaugment-run experiment=smoke env=default device=cuda tracker=none
   env.config.episode.epochs=3 env.config.episode.series_per_epoch=512`, ran in 25 s wall
   time on the RTX 4080 SUPER. 22.9 s of it is inside the experiment, the first compilation
   included. The aggregate is 1.115 after 3 small epochs.
4. Experiment 1: `experiment=train_agent agent=sac`, `agent=cmaes`, then `experiment=compare`.

## 7. Later work, not in v1

- A probe-batch embedding of the tsmodel as a fifth observation group.
- A multi-method hierarchical action: method id plus method parameters.
- Stacked K-model training with `torch.func`, and multi-GPU collection.
- A tslake adapter for `Benchmark` once the lake is in production.
- A leakage check of the LoRA variant: Mistral's stage-two post-training uses real data.
- A `min_context_length` mask of the training windows: GIFT-Eval sends contexts down to 12
  steps for annual data, but the tsmodel trains on full 768-step contexts.
- A paired reward: score the tsmodel before and after the epoch on the same subset.
- A histogram of `last_stage` (the Cholesky retries of KernelSynth), logged once per episode.
- Symmetric quantile averaging and non-negative clipping in the comparison forecast, as
  `TimestralForecaster` does.
- An own squashed Gaussian for SAC and PPO, so their updates make no host synchronization.
  TorchRL's `TanhNormal` reads its bounds on the host. WP7's `components/agents/distributions.py`
  has the log-probability and the entropy.
- Bitwise determinism on CUDA: `scatter_add` in `core.metrics` is not bitwise deterministic.
- GRPO ablations that stay out of WP7: a GRU memory in place of the transformer, a reward token
  in the policy input, and several groups per episode seed.
