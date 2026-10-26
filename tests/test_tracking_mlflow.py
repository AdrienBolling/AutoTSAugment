import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from mlflow import MlflowClient
from pydantic import ValidationError

from autotsaugment.components.tracking.mlflow import (
    GIT_COMMIT_TAG,
    MAX_PARAM_LENGTH,
    MLflowTracker,
    flatten_params,
    resolve_store,
)

# SQLAlchemy warns about a loader strategy inside MLflow's SQL store. Not our code.
pytestmark = pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy:DeprecationWarning")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # One store per module: the first SQLite store of a process costs about one second.
    return tmp_path_factory.mktemp("mlruns")


def make(store: Path, **config: Any) -> MLflowTracker:
    return MLflowTracker(tracking_uri=str(store), experiment_name="tests", **config)


def history(tracker: MLflowTracker, key: str) -> list[tuple[int, float]]:
    client = MlflowClient(tracking_uri=tracker.store_uri)
    return sorted((m.step, m.value) for m in client.get_metric_history(tracker.run_id, key))


def status(tracker: MLflowTracker) -> str:
    return MlflowClient(tracking_uri=tracker.store_uri).get_run(tracker.run_id).info.status


class Watched(torch.Tensor):
    """A tensor that records every torch function called on it."""

    calls: ClassVar[list[str]] = []

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        cls.calls.append(getattr(func, "__name__", str(func)))
        return super().__torch_function__(func, types, args, kwargs or {})


def test_flatten_params_uses_dotted_keys_and_truncates():
    params = {"env": {"config": {"epochs": 5, "seeds": [0, 1]}, "empty": {}}, "long": "x" * 7000}
    flat = flatten_params(params)
    assert flat == {
        "env.config.epochs": "5",
        "env.config.seeds": "[0, 1]",
        "env.empty": "{}",
        "long": "x" * MAX_PARAM_LENGTH,
    }


def test_relative_directory_resolves_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    store_uri, artifacts = resolve_store("mlruns")
    assert store_uri == f"sqlite:///{tmp_path.resolve() / 'mlruns' / 'mlflow.db'}"
    assert artifacts == (tmp_path.resolve() / "mlruns" / "artifacts").as_uri()
    assert (tmp_path / "mlruns").is_dir()


def test_uri_with_a_scheme_is_unchanged():
    assert resolve_store("http://localhost:5000") == ("http://localhost:5000", None)


def test_unknown_config_key_fails_before_mlflow_starts():
    with pytest.raises(ValidationError):
        MLflowTracker(trackinguri="mlruns")


def test_params_steps_and_episodes_reach_the_store(store: Path):
    tracker = make(store, run_name="round-trip", tags={"study": "tests"})
    tracker.log_params({"seed": 0, "env": {"config": {"epochs": 2}}})
    tracker.log_step(0, {"reward": torch.tensor(0.5), "count": 3})
    tracker.log_step(1, {"reward": torch.tensor(1.5), "nan": torch.tensor(float("nan"))})
    tracker.log_episode(0, {"return": torch.tensor(2.0), "comparison/aggregate": 0.75})
    tracker.log_step(2, {"reward": torch.tensor(2.5)})
    tracker.log_episode(1, {"return": torch.tensor(3.0)})
    tracker.close()

    run = MlflowClient(tracking_uri=tracker.store_uri).get_run(tracker.run_id)
    assert run.info.run_name == "round-trip"
    assert run.data.tags["study"] == "tests"
    assert len(run.data.tags[GIT_COMMIT_TAG]) == 40  # the tests run inside the git checkout
    assert run.data.params == {"seed": "0", "env.config.epochs": "2"}
    assert history(tracker, "reward") == [(0, 0.5), (1, 1.5), (2, 2.5)]
    assert history(tracker, "count") == [(0, 3.0)]
    assert history(tracker, "episode/return") == [(0, 2.0), (1, 3.0)]
    assert history(tracker, "episode/comparison/aggregate") == [(0, 0.75)]
    [(step, value)] = history(tracker, "nan")
    assert step == 1
    assert math.isnan(value)
    assert status(tracker) == "FINISHED"


def test_log_step_leaves_the_tensors_untouched_until_flush(store: Path):
    tracker = make(store)
    value = torch.tensor(1.25).as_subclass(Watched)
    Watched.calls.clear()
    for step in range(5):
        tracker.log_step(step, {"value": value})
    assert Watched.calls == []
    assert all(metrics["value"] is value for _, _, metrics in tracker._buffer)
    tracker.flush()
    assert Watched.calls
    assert tracker._buffer == []
    tracker.close()
    assert history(tracker, "value") == [(step, 1.25) for step in range(5)]


def test_batches_respect_the_mlflow_limits(store: Path, monkeypatch: pytest.MonkeyPatch):
    tracker = make(store)
    calls: list[tuple[int, int]] = []
    log_batch = tracker._client.log_batch

    def counting(run_id: str, metrics: Any = (), params: Any = (), **kwargs: Any) -> Any:
        calls.append((len(metrics), len(params)))
        return log_batch(run_id, metrics=metrics, params=params, **kwargs)

    monkeypatch.setattr(tracker._client, "log_batch", counting)
    tracker.log_params({f"p{i}": i for i in range(250)})
    assert calls == [(0, 100), (0, 100), (0, 50)]
    calls.clear()
    values = torch.arange(25, dtype=torch.float32)
    for step in range(100):
        tracker.log_step(step, {f"m{i}": values[i] for i in range(25)})
    tracker.flush()
    assert calls == [(1000, 0), (1000, 0), (500, 0)]
    calls.clear()
    tracker.flush()
    assert calls == []
    tracker.close()
    assert history(tracker, "m7") == [(step, 7.0) for step in range(100)]


def test_artifacts_hold_files_and_directories(store: Path, tmp_path: Path):
    tracker = make(store)
    note = tmp_path / "note.txt"
    note.write_text("hello")
    checkpoint = tmp_path / "sac"
    checkpoint.mkdir()
    (checkpoint / "agent.pt").write_bytes(b"weights")
    tracker.log_artifact(note)
    tracker.log_artifact(checkpoint)
    tracker.close()
    client = MlflowClient(tracking_uri=tracker.store_uri)
    assert sorted(a.path for a in client.list_artifacts(tracker.run_id)) == ["note.txt", "sac"]
    assert [a.path for a in client.list_artifacts(tracker.run_id, "sac")] == ["sac/agent.pt"]


def test_close_twice_and_failed_status(store: Path):
    tracker = make(store)
    tracker.log_step(0, {"reward": torch.tensor(1.0)})
    tracker.close()
    tracker.close()
    assert status(tracker) == "FINISHED"
    assert history(tracker, "reward") == [(0, 1.0)]
    failed = make(store)
    failed.mark_failed()
    failed.close()
    assert status(failed) == "FAILED"


@pytest.mark.gpu
def test_log_step_does_not_synchronize_on_cuda(store: Path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    tracker = make(store)
    reward = torch.zeros(4, device="cuda")
    try:
        torch.cuda.set_sync_debug_mode("error")
        for step in range(50):
            reward = reward + 1.0
            tracker.log_step(step, {"reward": reward.mean(), "score/a": reward[0], "n": 4})
    finally:
        torch.cuda.set_sync_debug_mode("default")
    tracker.log_episode(0, {"return": reward.sum()})
    tracker.close()
    assert history(tracker, "reward") == [(step, float(step + 1)) for step in range(50)]
    assert history(tracker, "episode/return") == [(0, 200.0)]
