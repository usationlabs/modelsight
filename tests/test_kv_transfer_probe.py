import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.probes import kv_transfer

BYTES_PER_TOKEN = 1000
XFER_SECONDS = 0.05
POOL_LOG = "INFO GPU KV cache size: 50,000 tokens, Maximum concurrency: 1.51x\n"
UCX_LOG = (
    "[1] [h:9:0]   | active message by ucp_am_send* from host memory |\n"
    "[1] [h:9:0]   | 0..8176 | short | tcp/eth0 |\n"
    "[1] [h:9:0]   | remote memory read by ucp_get*(multi) into cuda from cuda |\n"
    "[1] [h:9:0]   | Range | Description | Config |\n"
    "[1] [h:9:0]   | 0..inf | zero-copy | cuda_ipc/cuda |\n"
)


class FakeVllm:
    """Enough of a vLLM server: cached prompts answer fast, and fetches add metrics."""

    def __init__(self) -> None:
        self.prompts: set[tuple[int, ...]] = set()
        self.requests: list[dict] = []
        self.transfers = 0
        self.bytes = 0
        self.hits = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _reply(self, body: bytes) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._reply(b"")
                    return
                self._reply(fake.metrics().encode())

            def do_POST(self) -> None:
                size = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(size))
                fake.requests.append(body)
                prompt = tuple(body["prompt"])
                fetched = "do_remote_prefill" in body.get("kv_transfer_params", {}) and (
                    body["kv_transfer_params"]["do_remote_prefill"]
                )
                if prompt in fake.prompts:
                    fake.hits += len(prompt)
                elif not fetched:
                    time.sleep(len(prompt) / 2e6)
                fake.prompts.add(prompt)
                if fetched:
                    fake.transfers += 1
                    fake.bytes += len(prompt) * BYTES_PER_TOKEN
                if body.get("stream"):
                    self._reply(b'data: {"x":1}\n\ndata: [DONE]\n\n')
                else:
                    params = {"do_remote_prefill": True, "remote_engine_id": "e"}
                    self._reply(json.dumps({"kv_transfer_params": params}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def metrics(self) -> str:
        return (
            "# HELP ignored\n"
            f'vllm:nixl_bytes_transferred_count{{m="q"}} {self.transfers}\n'
            f'vllm:nixl_bytes_transferred_sum{{m="q"}} {self.bytes}\n'
            'vllm:nixl_bytes_transferred_bucket{le="1"} 9\n'
            f'vllm:nixl_num_descriptors_sum{{m="q"}} {self.transfers * 28}\n'
            f'vllm:nixl_post_time_seconds_sum{{m="q"}} {self.transfers * 0.001}\n'
            f'vllm:nixl_xfer_time_seconds_sum{{m="q"}} {self.transfers * XFER_SECONDS}\n'
            'vllm:nixl_num_failed_transfers_total{m="q"} 0.0\n'
            f'vllm:prefix_cache_hits_total{{m="q"}} {self.hits}\n'
        )


@pytest.fixture
def fakes():
    producer, consumer = FakeVllm(), FakeVllm()
    yield producer, consumer
    producer.server.shutdown()
    consumer.server.shutdown()


@pytest.fixture
def probe_args(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(kv_transfer, "FILL_PROMPT_LENGTHS", (2000, 8000))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "server-A.log").write_text(POOL_LOG)
    (tmp_path / "server-B.log").write_text(POOL_LOG + UCX_LOG)
    producer, consumer = fakes
    return [
        "--model", "m",
        "--producer", producer.url,
        "--consumer", consumer.url,
        "--producer-log", str(tmp_path / "server-A.log"),
        "--consumer-log", str(tmp_path / "server-B.log"),
        "--prefix-lengths", "512,2000",
        "--repeats", "2",
        "--output", str(tmp_path / "out" / "results.json"),
    ]  # fmt: skip


def test_probe_runs_every_arm_and_writes_results(fakes, probe_args, tmp_path, capsys):
    (tmp_path / "out").mkdir()

    assert kv_transfer.main(probe_args) == 0

    result = json.loads((tmp_path / "out" / "results.json").read_text())
    assert result["verdict"].startswith("PASS")
    measured = [r for r in result["records"] if r["arm"] in kv_transfer.ARMS]
    assert len(measured) == 3 * 2 * 2
    assert result["summary"]["by_length"]["2000"]["warm_hit_fraction"] == 1.0
    assert result["summary"]["by_length"]["2000"]["transfer"]["mb"] == 2.0
    assert (tmp_path / "out" / "server-B.log").exists()
    assert "crossover:" in capsys.readouterr().out


def test_probe_fills_both_pools_past_their_size(fakes, probe_args, tmp_path):
    (tmp_path / "out").mkdir()

    kv_transfer.main(probe_args)

    fill = json.loads((tmp_path / "out" / "results.json").read_text())["fill"]
    assert fill["producer"]["sent_tokens"] >= 50000 * kv_transfer.FILL_FACTOR
    assert fill["consumer"]["sent_tokens"] >= 50000 * kv_transfer.FILL_FACTOR


def test_probe_prompts_are_exact_and_never_reused_by_recompute(fakes, probe_args, tmp_path):
    (tmp_path / "out").mkdir()
    _, consumer = fakes

    kv_transfer.main(probe_args)

    prompts = [tuple(r["prompt"]) for r in consumer.requests]
    assert len(prompts) == len(set(prompts))
    streamed = [r for r in consumer.requests if r.get("stream")]
    assert {len(r["prompt"]) for r in streamed} == {512, 2000}


def test_probe_fails_without_cuda_ipc(probe_args, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "server-B.log").write_text(POOL_LOG + UCX_LOG.replace("cuda_ipc/cuda", "tcp"))

    assert kv_transfer.main(probe_args) == 1


def test_read_path_ignores_control_message_transports():
    assert kv_transfer.read_path(UCX_LOG) == [
        "| remote memory read by ucp_get*(multi) into cuda from cuda |",
        "| 0..inf | zero-copy | cuda_ipc/cuda |",
    ]


def test_verdict_reports_failed_measurements():
    row = {"transfer": {"mb": 1}}
    summary = {"by_length": {512: row}, "errors": [{}], "crossover_prefix_len": 512}

    assert kv_transfer.verdict(summary, ["| cuda_ipc |"]) == "FAIL: 1 measurements failed"


def test_verdict_separates_a_missing_table_from_another_transport():
    summary = {"by_length": {512: {"transfer": {"mb": 1}}}, "errors": []}

    assert kv_transfer.verdict(summary, []).startswith("UNCONFIRMED")
    assert kv_transfer.verdict(summary, ["| tcp/eth0 |"]).startswith("FAIL")


def test_crossover_is_the_first_length_where_fetch_wins():
    def record(arm: str, length: int, ttft: float) -> dict:
        return {"arm": arm, "prefix_len": length, "repeat": 0, "error": None, "ttft_s": ttft}

    records = [
        record("recompute", 512, 0.05),
        record("fetch", 512, 0.06),
        record("recompute", 4000, 0.08),
        record("fetch", 4000, 0.07),
    ]

    assert kv_transfer.summarize(records, (512, 4000))["crossover_prefix_len"] == 4000


def test_h1_example_is_a_valid_experiment():
    path = Path(__file__).parent.parent / "examples" / "kv-transfer" / "h1.yaml"

    experiment, _ = load_experiment(path)

    assert experiment.gpu == "A100-80GB:2"
    assert experiment.secrets == ("HF_TOKEN",)
    assert "modelsight.probes.kv_transfer" in experiment.bench
