"""Measure the training and forecast speed of a Timestral tsmodel on the GPU.

The script reads the tsmodel config from ``configs/env/tsmodel/<variant>.yaml``, trains on
synthetic sines, and prints the seconds per epoch, the forecast time of one score batch and
the peak memory. The data does not change the speed: every window has the same length.

Usage::

    uv run python scripts/bench_tsmodel.py
    uv run python scripts/bench_tsmodel.py --compile --warmup 2
    uv run python scripts/bench_tsmodel.py --variant timestral_lora
"""

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from autotsaugment.components.tsmodel.timestral import (
    TimestralBare,
    TimestralBareConfig,
    TimestralLora,
    TimestralLoraConfig,
)
from autotsaugment.core.seeding import make_generator

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env" / "tsmodel"


def parse_args() -> argparse.Namespace:
    """Return the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variant", choices=["timestral_bare", "timestral_lora"], default="timestral_bare"
    )
    parser.add_argument("--series", type=int, default=4096, help="series per epoch, N")
    parser.add_argument("--length", type=int, default=1024, help="steps per series")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3, help="timed epochs")
    parser.add_argument("--warmup", type=int, default=1, help="untimed epochs before the timing")
    parser.add_argument("--compile", action="store_true", help="wrap the loss with torch.compile")
    parser.add_argument("--fp32", action="store_true", help="switch off bf16 autocast")
    parser.add_argument("--score-windows", type=int, default=1024, help="windows per forecast")
    parser.add_argument("--score-context", type=int, default=512)
    parser.add_argument("--score-horizon", type=int, default=48)
    return parser.parse_args()


def build(args: argparse.Namespace, device: torch.device) -> TimestralBare | TimestralLora:
    """Return the tsmodel of the YAML file, with the overrides of ``args``."""
    node = OmegaConf.load(CONFIGS / f"{args.variant}.yaml")
    fields: Any = OmegaConf.to_container(node.config)
    fields.pop("_target_")
    fields |= {
        "batch_size": args.batch_size,
        "compile": args.compile,
        "mixed_precision": not args.fp32,
    }
    if args.variant == "timestral_bare":
        return TimestralBare(config=TimestralBareConfig(**fields), device=device, episode_epochs=50)
    return TimestralLora(config=TimestralLoraConfig(**fields), device=device, episode_epochs=50)


def sines(n: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """Return ``n`` noisy sines with random period, phase and level."""
    device = generator.device
    steps = torch.arange(length, dtype=torch.float32, device=device)
    period = 8.0 + 200.0 * torch.rand(n, 1, generator=generator, device=device)
    phase = 2 * math.pi * torch.rand(n, 1, generator=generator, device=device)
    level = torch.randn(n, 1, generator=generator, device=device)
    noise = 0.1 * torch.randn(n, length, generator=generator, device=device)
    return level + torch.sin(2 * math.pi * steps / period + phase) + noise


def main() -> None:
    """Run the benchmark and print the results."""
    args = parse_args()
    device = torch.device("cuda")
    model = build(args, device)
    generator = make_generator(device, 0)
    series = sines(args.series, args.length, generator)
    context = series[: args.score_windows, : args.score_context]

    for _ in range(args.warmup):
        model.train_epoch(series, generator=generator)
        model.forecast(context, args.score_horizon)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    stats = [model.train_epoch(series, generator=generator) for _ in range(args.epochs)][-1]
    torch.cuda.synchronize()
    per_epoch = (time.perf_counter() - start) / args.epochs

    start = time.perf_counter()
    for _ in range(args.epochs):
        model.forecast(context, args.score_horizon)
    torch.cuda.synchronize()
    per_forecast = (time.perf_counter() - start) / args.epochs

    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(
        f"{args.variant}: N={args.series} x {args.length}, batch {args.batch_size}, "
        f"bf16={not args.fp32}, compile={args.compile}, "
        f"{model.n_trainable_parameters():,} trainable parameters"
    )
    print(
        f"  train: {per_epoch:.3f} s/epoch ({stats.n_steps.item():.0f} steps), loss {stats.loss.item():.4f}"
    )
    print(
        f"  forecast: {per_forecast * 1000:.1f} ms for {args.score_windows} windows "
        f"(context {args.score_context}, horizon {args.score_horizon})"
    )
    print(f"  peak memory: {peak:.2f} GiB")


if __name__ == "__main__":
    main()
