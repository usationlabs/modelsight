import json
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.probes import kv_busy
from test_kv_transfer_probe import UCX_LOG, FakeVllm


@pytest.fixture
def fakes(monkeypatch):
    producer, consumer = FakeVllm(), FakeVllm()
    monkeypatch.setattr(kv_busy, "wait_running", lambda servers: None)
    monkeypatch.setattr(kv_busy, "producer_running", lambda servers: 1.0)
    yield producer, consumer
    producer.server.shutdown()
    consumer.server.shutdown()


def test_probe_runs_every_arm_with_a_fixed_load(fakes, tmp_path):
    producer, consumer = fakes
    (tmp_path / "A.log").write_text("")
    (tmp_path / "B.log").write_text(UCX_LOG)
    (tmp_path / "out").mkdir()

    status = kv_busy.main([
        "--model", "m",
        "--producer", producer.url,
        "--consumer", consumer.url,
        "--producer-log", str(tmp_path / "A.log"),
        "--consumer-log", str(tmp_path / "B.log"),
        "--prefix-lengths", "512,2000",
        "--load-length", "700",
        "--repeats", "2",
        "--output", str(tmp_path / "out" / "results.json"),
    ])  # fmt: skip

    result = json.loads((tmp_path / "out" / "results.json").read_text())
    assert status == 0
    assert len(result["records"]) == 4 * 2 * 2
    assert result["summary"]["fetches_finished_while_busy"] == 4
    loads = [r for r in producer.requests if len(r["prompt"]) == 700]
    assert len(loads) == 8


def test_a_fetch_that_outlasts_the_load_is_not_a_pass():
    row = {"transfer_busy": {"mb": 1}}
    summary = {
        "by_length": {8000: row},
        "errors": [],
        "fetches_busy": 2,
        "fetches_finished_while_busy": 1,
    }

    assert kv_busy.verdict(summary, ["| cuda_ipc |"]).startswith("UNCONFIRMED")


def test_h2_example_is_a_valid_experiment():
    path = Path(__file__).parent.parent / "examples" / "kv-transfer" / "h2.yaml"

    experiment, _ = load_experiment(path)

    assert experiment.timeout == "45m"
    assert "--max-model-len 129024" in experiment.server
    assert "modelsight.probes.kv_busy" in experiment.bench
