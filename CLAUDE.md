# AutoTSAugment — repo rules for Claude

Read `design_docs/00-plan.md` first. It holds the locked decisions and the module contracts.
Global rules (git, style, DVC) are in `~/.claude/CLAUDE.md` and `~/repositories/CLAUDE.md`.

## What this repo is

An RL agent that chooses, at each training epoch of a forecaster (the "tsmodel"), the
parameters of a synthetic time-series generator. One RL step is one epoch. The reward is the
improvement of the tsmodel on real benchmark data. v1: KernelSynth as the only generator,
Timestral-5M as the only tsmodel, GIFT-Eval as the benchmark.

## Workflow

- `uv` only. Run all checks before you report done:
  `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
- GPU, network and slow tests are opt-in: `uv run pytest -q -m gpu`.
- Runtime shape checks: `AUTOTSAUGMENT_TYPECHECK=1 uv run pytest -q` (tests set it in `conftest.py`).
- Branch per module, one pull request per branch, merged by the user. Conventional Commits.
- Prose in Simplified Technical English. Docstrings in Google style.

## Design decisions to preserve

- `core/` holds contracts only: abstract base classes, protocols, Pydantic config bases, type
  aliases. `components/` holds implementations. `core` never imports `components`.
- Everything that touches tensors stays on the GPU: synthetic series, reward windows, observations,
  actions, replay data. Convert to NumPy or CPU only inside trackers, once per episode.
- Every component: a frozen Pydantic `XConfig`, a class with `__init__(self, *, config, ...)`, and a
  YAML file under `configs/` with a `_target_`. Hydra `instantiate` is the registry.
- Config groups: `experiment/`, `agent/`, `env/` (with `env/tsmodel/`, `env/augmentation/`,
  `env/benchmark/`). `configs/run.yaml` composes one default per group.
- Runtime checks: `@checked` on batch-level methods only, enabled by `AUTOTSAUGMENT_TYPECHECK=1`.
  No jaxtyping import hook. No `from __future__ import annotations` (beartype needs runtime types).
- Suppress a lint or type rule only as `# noqa: RULE  # reason: ...` or
  `# ty: ignore[rule]  # reason: ...`, and only with explicit user authorization.
- Keeper checkpoints go through DVC (`~/model-store`); one `metrics/<experiment>.json` per experiment.

## Environment facts

- Host: Arch Linux, one RTX 4080 SUPER 16 GB, uv 0.12.9, Python 3.13 via uv.
- Timestral code is a git dependency (inference only, Apache 2.0). Weights come from the
  Hugging Face cache (`mistralai/Timestral-5M`, Modified MIT).
- Benchmark data: GIFT-Eval from the Hugging Face cache, at most 10 GB on disk, in `data/`.
