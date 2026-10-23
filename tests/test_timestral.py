"""Tests of the Timestral tsmodels: a tiny architecture on the CPU, the 5M on the GPU."""

import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydantic import ValidationError
from safetensors.torch import save_file
from timestral import TimestralForecaster
from timestral.loading import load_model

from autotsaugment.components.tsmodel.network import UnivariateDecoder
from autotsaugment.components.tsmodel.schedule import build_scheduler, warmup_cosine
from autotsaugment.components.tsmodel.timestral import (
    TimestralBare,
    TimestralBareConfig,
    TimestralLora,
    TimestralLoraConfig,
)
from autotsaugment.components.tsmodel.windows import pad_context, random_windows
from autotsaugment.core.metrics import pinball_loss, seasonal_naive
from autotsaugment.core.seeding import make_generator
from autotsaugment.core.tsmodel import TrainStats

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
KNOTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TINY: dict[str, Any] = {
    "patch_size": 8,
    "d_model": 32,
    "num_heads": 2,
    "mlp_hidden_dim": 64,
    "block_schedule": ["time", "time_variate"],
    "quantile_knots": KNOTS,
    "block_size": 8,
    "context_length": 1024,
}
GIB = 1024**3


def bare_config(**overrides: Any) -> TimestralBareConfig:
    fields: dict[str, Any] = {
        "architecture": TINY,
        "context_length": 32,
        "horizon_length": 16,
        "batch_size": 16,
        "learning_rate": 3e-3,
        "mixed_precision": False,
    }
    fields.update(overrides)
    return TimestralBareConfig(**fields)


def lora_config(checkpoint: Path, **overrides: Any) -> TimestralLoraConfig:
    fields: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "patch_size": 8,
        "block_size": 8,
        "context_length": 32,
        "horizon_length": 16,
        "batch_size": 16,
        "learning_rate": 3e-3,
        "mixed_precision": False,
        "lora_rank": 4,
        "lora_alpha": 8,
    }
    fields.update(overrides)
    return TimestralLoraConfig(**fields)


def noisy_sines(n: int, length: int, generator: torch.Generator) -> torch.Tensor:
    device = generator.device
    steps = torch.arange(length, dtype=torch.float32, device=device)
    period = 6.0 + 10.0 * torch.rand(n, 1, generator=generator, device=device)
    phase = 2 * math.pi * torch.rand(n, 1, generator=generator, device=device)
    level = torch.randn(n, 1, generator=generator, device=device)
    noise = 0.1 * torch.randn(n, length, generator=generator, device=device)
    return level + torch.sin(2 * math.pi * steps / period + phase) + noise


def yaml_architecture() -> dict[str, Any]:
    node = OmegaConf.load(CONFIGS / "env" / "tsmodel" / "timestral_bare.yaml")
    architecture = OmegaConf.to_container(node.config.architecture)
    assert isinstance(architecture, dict)
    return {str(key): value for key, value in architecture.items()}


@pytest.fixture
def tiny_checkpoint(tmp_path: Path, device: torch.device) -> Path:
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=1)
    model.reset(seed=7)
    model.network.config.save(tmp_path / "config.json")
    weights = {
        name: p.detach().cpu().contiguous() for name, p in model.network.state_dict().items()
    }
    save_file(weights, tmp_path / "model.safetensors")
    return tmp_path


# --- helpers -------------------------------------------------------------------------------


def test_random_windows_are_shuffled_slices(device: torch.device):
    n, length = 6, 20
    series = 1000.0 * torch.arange(n, dtype=torch.float32).unsqueeze(1) + torch.arange(40.0)
    windows = random_windows(series, length, generator=make_generator(device, 1))
    assert windows.shape == (n, length)
    torch.testing.assert_close(windows.diff(dim=1), torch.ones(n, length - 1))
    rows = (windows[:, 0] // 1000).long()
    assert sorted(rows.tolist()) == list(range(n))
    again = random_windows(series, length, generator=make_generator(device, 1))
    torch.testing.assert_close(windows, again)
    with pytest.raises(ValueError, match="shorter"):
        random_windows(series, 41, generator=make_generator(device, 1))


def test_pad_context_left_pads_with_missing_values():
    context = torch.arange(26.0).reshape(2, 13)
    context[1, 5] = float("nan")
    inputs, observed = pad_context(context, 8)
    assert inputs.shape == observed.shape == (2, 16)
    assert not observed[:, :3].any()
    assert not observed[1, 8]
    assert observed[0, 3:].all()
    assert (inputs[~observed] == 0).all()
    torch.testing.assert_close(inputs[0, 3:], context[0])


def test_warmup_cosine_schedule():
    factor = {
        step: warmup_cosine(step, total_steps=100, warmup_steps=10, min_ratio=0.1)
        for step in range(101)
    }
    assert factor[0] == pytest.approx(0.1)
    assert factor[9] == pytest.approx(1.0)
    assert factor[10] == pytest.approx(1.0)
    assert factor[55] == pytest.approx(0.55)
    assert factor[100] == pytest.approx(0.1)
    assert all(factor[s] >= factor[s + 1] for s in range(10, 100))
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=2.0)
    scheduler = build_scheduler(optimizer, total_steps=100, warmup_fraction=0.1, min_ratio=0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


# --- configs -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_length": 30}, "context_length"),
        ({"horizon_length": 4}, "horizon_length"),
        ({"horizon_length": 72}, "longer than one decode block"),
        ({"architecture": {**TINY, "unknown": 1}}, "unknown config keys"),
    ],
)
def test_bare_config_rejects_bad_windows(overrides: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        bare_config(**overrides)


def test_lora_config_rejects_dropout_and_bad_windows(tmp_path: Path):
    with pytest.raises(ValidationError, match="lora_dropout"):
        lora_config(tmp_path, lora_dropout=0.1)
    with pytest.raises(ValidationError, match="horizon_length"):
        lora_config(tmp_path, horizon_length=20)


@pytest.mark.parametrize("tsmodel", ["timestral_bare", "timestral_lora"])
def test_yaml_configs_validate(tsmodel: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose("run", overrides=["env=default", f"env/tsmodel={tsmodel}"])
    config = instantiate(cfg.env.tsmodel.config)
    assert isinstance(config, TimestralBareConfig | TimestralLoraConfig)
    assert (config.context_length, config.horizon_length, config.batch_size) == (768, 256, 128)
    if isinstance(config, TimestralBareConfig):
        assert config.timestral_config.patch_size == 32
        assert isinstance(config.architecture["block_schedule"], list)


def test_yaml_factory_builds_the_5m_architecture_on_the_cpu():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose("run", overrides=["env=default", "device=cpu"])
    factory = instantiate(cfg.env.tsmodel)
    model = factory()
    assert isinstance(model, TimestralBare)
    assert model.device == torch.device("cpu")
    assert 4_000_000 < model.n_trainable_parameters() < 7_000_000
    assert model.quantile_levels.shape == (21,)
    assert model._episode_epochs == cfg.env.config.episode.epochs


# --- bare model ----------------------------------------------------------------------------


def test_training_loss_decreases(device: torch.device, generator: torch.Generator):
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=10)
    model.reset(seed=0)
    series = noisy_sines(128, 64, generator)
    losses = [model.train_epoch(series, generator=generator).loss for _ in range(10)]
    assert torch.isfinite(torch.stack(losses)).all()
    assert losses[-1] < 0.8 * losses[0]


def test_train_stats_are_0d_tensors_on_the_device(device: torch.device, generator: torch.Generator):
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=3)
    stats = model.train_epoch(noisy_sines(40, 64, generator), generator=generator)
    assert isinstance(stats, TrainStats)
    for name, value in stats.as_dict().items():
        assert isinstance(value, torch.Tensor), name
        assert value.shape == (), name
        assert value.device == device, name
        assert value.dtype == torch.float32, name
        assert torch.isfinite(value), name
    assert stats.n_steps == 3  # ceil(40 / 16)
    assert 0 < stats.lr <= 3e-3
    assert stats.grad_norm > 0


def test_forecast_shapes_and_monotone_quantiles(device: torch.device, generator: torch.Generator):
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    model.train_epoch(noisy_sines(32, 64, generator), generator=generator)
    context = noisy_sines(5, 40, generator)
    for horizon in (3, 8, 12, 80):
        forecast = model.forecast(context, horizon)
        assert forecast.shape == (5, horizon, 9)
        assert forecast.dtype == torch.float32
        assert torch.isfinite(forecast).all()
        assert (forecast[..., 1:] >= forecast[..., :-1]).all()
    torch.testing.assert_close(model.quantile_levels, torch.tensor(KNOTS))


def test_forecast_of_a_nan_padded_context_is_finite(
    device: torch.device, generator: torch.Generator
):
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    context = noisy_sines(4, 30, generator)
    context[:, :10] = float("nan")
    context[1, :27] = float("nan")
    context[2, 15] = float("nan")
    forecast = model.forecast(context, 12)
    assert forecast.shape == (4, 12, 9)
    assert torch.isfinite(forecast).all()


def test_forecast_matches_the_timestral_decoders(device: torch.device, generator: torch.Generator):
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    model.train_epoch(noisy_sines(32, 64, generator), generator=generator)
    context = noisy_sines(3, 37, generator)
    context[0, :9] = float("nan")
    inputs, observed = pad_context(context, 8)
    with torch.no_grad():
        block = model.network.predict_block(inputs, observed, torch.arange(3), 1)
    torch.testing.assert_close(model.forecast(context, 5), block[:, :5])
    reference = TimestralForecaster(model.network)
    for horizon in (20, 80):  # one block with refined frames; two blocks
        expected = reference.forecast(
            context, horizon, symmetric_averaging=False, clip_nonnegative=False
        ).quantiles[:, 0]
        torch.testing.assert_close(model.forecast(context, horizon), expected, rtol=1e-4, atol=1e-4)


def test_loss_is_invariant_to_the_scale_of_a_series(
    device: torch.device, generator: torch.Generator
):
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=1)
    window = noisy_sines(8, 48, generator)
    loss = model._batch_loss(window)
    torch.testing.assert_close(model._batch_loss(250.0 * window - 40.0), loss, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(model._batch_loss(window.flip(1)), loss)


def test_reset_restores_the_initial_weights(device: torch.device, generator: torch.Generator):
    series = noisy_sines(32, 64, generator)
    model = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    model.reset(seed=3)
    initial = model.state_dict()
    first = model.train_epoch(series, generator=make_generator(device, 5))
    trained = model.state_dict()
    assert any(not torch.equal(initial[name], trained[name]) for name in initial)
    model.reset(seed=3)
    assert model._optimizer.state == {}
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name]), name
    again = model.train_epoch(series, generator=make_generator(device, 5))
    assert torch.equal(first.loss, again.loss)
    other = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    other.reset(seed=3)
    for name, value in other.state_dict().items():
        assert torch.equal(value, initial[name]), name
    other.reset(seed=4)
    assert any(not torch.equal(initial[name], value) for name, value in other.state_dict().items())


def test_state_dict_round_trip(device: torch.device, generator: torch.Generator):
    source = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    source.train_epoch(noisy_sines(32, 64, generator), generator=generator)
    target = TimestralBare(config=bare_config(), device=device, episode_epochs=2)
    target.reset(seed=1)
    state = source.state_dict()
    assert set(state) == {name for name, _ in source.network.named_parameters()}
    assert sum(value.numel() for value in state.values()) == source.n_trainable_parameters()
    target.load_state_dict(state)
    context = noisy_sines(3, 32, generator)
    torch.testing.assert_close(target.forecast(context, 10), source.forecast(context, 10))
    with pytest.raises(KeyError, match="missing"):
        target.load_state_dict({})


# --- LoRA ----------------------------------------------------------------------------------


def is_adapter(name: str) -> bool:
    return "lora_" in name or ".modules_to_save." in name


def test_lora_trains_only_adapters_and_fully_trainable_modules(
    tiny_checkpoint: Path, device: torch.device
):
    model = TimestralLora(config=lora_config(tiny_checkpoint), device=device, episode_epochs=2)
    names = dict(model.network.named_parameters())
    trainable = {name for name, p in names.items() if p.requires_grad}
    assert trainable == {name for name in names if is_adapter(name)}
    assert any(name.startswith("head.modules_to_save.") for name in trainable)
    assert any(name.startswith("representation.modules_to_save.") for name in trainable)
    assert any("wQKV.lora_A" in name for name in trainable)
    assert not any("_gate" in name for name in trainable)
    assert set(model.state_dict()) == trainable
    assert model.n_trainable_parameters() == sum(names[name].numel() for name in trainable)
    assert model.n_trainable_parameters() < sum(p.numel() for p in names.values())


def test_lora_reset_restores_the_checkpoint(
    tiny_checkpoint: Path, device: torch.device, generator: torch.Generator
):
    model = TimestralLora(config=lora_config(tiny_checkpoint), device=device, episode_epochs=3)
    model.reset(seed=11)
    initial = {name: value.clone() for name, value in model.network.state_dict().items()}
    base = UnivariateDecoder(load_model(tiny_checkpoint, device=device), device)
    context = noisy_sines(4, 32, generator)
    inputs, observed = pad_context(context, 8)
    with torch.no_grad():
        expected = base.forecast(inputs, observed, 12)
    torch.testing.assert_close(model.forecast(context, 12), expected)

    series = noisy_sines(48, 64, generator)
    for _ in range(3):
        model.train_epoch(series, generator=generator)
    trained = model.network.state_dict()
    for name, value in trained.items():
        changed = not torch.equal(value, initial[name])
        assert changed == is_adapter(name) or "lora_A" in name, name
    assert not torch.allclose(model.forecast(context, 12), expected)

    model.reset(seed=11)
    for name, value in model.network.state_dict().items():
        assert torch.equal(value, initial[name]), name
    torch.testing.assert_close(model.forecast(context, 12), expected)
    model.reset(seed=12)
    lora_a = [name for name in initial if "lora_A" in name]
    assert any(not torch.equal(model.network.state_dict()[name], initial[name]) for name in lora_a)


def test_lora_state_dict_round_trip(
    tiny_checkpoint: Path, device: torch.device, generator: torch.Generator
):
    source = TimestralLora(config=lora_config(tiny_checkpoint), device=device, episode_epochs=2)
    source.train_epoch(noisy_sines(32, 64, generator), generator=generator)
    target = TimestralLora(config=lora_config(tiny_checkpoint), device=device, episode_epochs=2)
    target.load_state_dict(source.state_dict())
    context = noisy_sines(3, 32, generator)
    torch.testing.assert_close(target.forecast(context, 10), source.forecast(context, 10))


def test_lora_rejects_a_checkpoint_with_another_patch_size(
    tiny_checkpoint: Path, device: torch.device
):
    config = lora_config(tiny_checkpoint, patch_size=16, horizon_length=16)
    with pytest.raises(ValueError, match="patch_size"):
        TimestralLora(config=config, device=device, episode_epochs=1)


# --- GPU -----------------------------------------------------------------------------------


@pytest.fixture
def cuda() -> Iterator[torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    yield torch.device("cuda")
    if torch.cuda.get_sync_debug_mode():
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_bare_5m_epoch_on_the_gpu(cuda: torch.device):
    config = TimestralBareConfig(architecture=yaml_architecture())
    model = TimestralBare(config=config, device=cuda, episode_epochs=50)
    generator = make_generator(cuda, 0)
    series = noisy_sines(512, 1024, generator)
    torch.cuda.synchronize()
    start = time.perf_counter()
    stats = model.train_epoch(series, generator=generator)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(cuda) / GIB
    sys.stdout.write(
        f"\nTimestral-5M bare, 512 series: {seconds:.2f} s/epoch, peak {peak:.2f} GiB\n"
    )
    assert seconds < 30.0
    assert peak < 4.0
    assert stats.n_steps == 4
    assert torch.isfinite(stats.loss)

    torch.cuda.set_sync_debug_mode("error")  # no host synchronization in a step
    model.train_epoch(series, generator=generator)
    forecast = model.forecast(series[:64, :512], 48)
    torch.cuda.set_sync_debug_mode(0)
    assert forecast.shape == (64, 48, 21)
    assert forecast.device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.network
def test_pretrained_5m_beats_seasonal_naive_on_a_sine(cuda: torch.device):
    config = TimestralLoraConfig(mixed_precision=False)
    model = TimestralLora(config=config, device=cuda, episode_epochs=1)
    period = 17.3
    steps = torch.arange(512 + 96, dtype=torch.float32, device=cuda)
    series = torch.sin(2 * math.pi * steps / period).unsqueeze(0)
    context, target = series[:, :512], series[:, 512:]
    forecast = model.forecast(context, 96)
    levels = model.quantile_levels
    naive = seasonal_naive(context, 96, round(period)).unsqueeze(-1).expand(-1, -1, levels.numel())
    model_loss = pinball_loss(forecast, target, levels).mean()
    naive_loss = pinball_loss(naive, target, levels).mean()
    sys.stdout.write(
        f"\npinball loss: Timestral-5M {model_loss.item():.4f}, "
        f"seasonal naive {naive_loss.item():.4f}\n"
    )
    assert model_loss < naive_loss

    reference = TimestralForecaster(load_model("mistralai/Timestral-5M", device=cuda))
    expected = reference.forecast(
        context, 96, symmetric_averaging=False, clip_nonnegative=False
    ).quantiles[:, 0]
    torch.testing.assert_close(forecast, expected, rtol=1e-3, atol=1e-3)
    del reference


@pytest.mark.gpu
@pytest.mark.network
def test_pretrained_5m_lora_epoch_on_the_gpu(cuda: torch.device):
    model = TimestralLora(config=TimestralLoraConfig(), device=cuda, episode_epochs=10)
    generator = make_generator(cuda, 0)
    series = noisy_sines(256, 1024, generator)
    first = model.train_epoch(series, generator=generator)
    torch.cuda.set_sync_debug_mode("error")  # no host synchronization in a step
    second = model.train_epoch(series, generator=generator)
    forecast = model.forecast(series[:32, :512], 48)
    torch.cuda.set_sync_debug_mode(0)
    assert torch.isfinite(torch.stack([first.loss, second.loss])).all()
    assert torch.isfinite(forecast).all()
    assert torch.cuda.max_memory_allocated(cuda) / GIB < 4.0
