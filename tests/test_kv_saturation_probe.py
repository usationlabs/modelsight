import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.probes import kv_saturation

TOKENS_PER_SECOND = 1e5


class FakeVllm:
    """A server whose prefill slows everyone down while it runs.

    Streams a token every `tick` seconds, or every `slow_tick` while a long
    prompt is being computed, so a long prepare raises the load's TPOT.
    """

    def __init__(self, tick: float = 0.01, slow_tick: float = 0.05) -> None:
        self.tick, self.slow_tick = tick, slow_tick
        self.prefilling = 0
        self.running = 0
        self.generated = 0
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()
                if self.path != "/health":
                    self.wfile.write(fake.metrics().encode())

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.end_headers()
                fake.answer(body, self.wfile)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def metrics(self) -> str:
        return (
            f"vllm:num_requests_running {self.running}\n"
            "vllm:num_requests_waiting 0\n"
            "vllm:kv_cache_usage_perc 0.5\n"
            f"vllm:generation_tokens_total {self.generated}\n"
        )

    def answer(self, body: dict, out) -> None:
        prompt = body["prompt"]
        params = body.get("kv_transfer_params") or {}
        long = len(prompt) > 1500 and not params.get("do_remote_prefill")
        with self.lock:
            self.running += 1
            self.prefilling += long
        try:
            if not params.get("do_remote_prefill"):
                time.sleep(len(prompt) / TOKENS_PER_SECOND)
        finally:
            with self.lock:
                self.prefilling -= long
        try:
            if body.get("stream"):
                for _ in range(body.get("max_tokens", 1)):
                    time.sleep(self.slow_tick if self.prefilling else self.tick)
                    out.write(b'data: {"x":1}\n\n')
                    out.flush()
                    self.generated += 1
                out.write(b"data: [DONE]\n\n")
            else:
                reply = {"choices": [{"text": "t"}]}
                if params.get("do_remote_decode"):
                    reply["kv_transfer_params"] = {"do_remote_prefill": True}
                out.write(json.dumps(reply).encode())
        finally:
            with self.lock:
                self.running -= 1


@pytest.fixture
def probe_args(tmp_path):
    producer, consumer = FakeVllm(), FakeVllm()
    (tmp_path / "server-A.log").write_text("")
    (tmp_path / "server-B.log").write_text("")
    (tmp_path / "out").mkdir()
    yield [
        "--model", "m",
        "--producer", producer.url,
        "--consumer", consumer.url,
        "--producer-log", str(tmp_path / "server-A.log"),
        "--consumer-log", str(tmp_path / "server-B.log"),
        "--concurrency", "0,4",
        "--remnants", "1000,8000",
        "--load-prompt-len", "100",
        "--load-output-len", "50",
        "--settle", "0.5",
        "--repeats", "2",
        "--output", str(tmp_path / "out" / "results.json"),
    ]  # fmt: skip
    producer.server.shutdown()
    consumer.server.shutdown()


def test_probe_measures_every_cell_and_finds_the_users_boundary(probe_args, tmp_path, capsys):
    status = kv_saturation.main(probe_args)

    result = json.loads((tmp_path / "out" / "results.json").read_text())
    assert status == 0
    assert len(result["records"]) == 2 * 2 * 2
    table = result["summary"]["by_concurrency"]
    assert set(table) == {"0", "4"}
    loaded = table["4"]
    assert loaded["remnants"]["8000"]["tpot_rise"] > 0.2
    assert loaded["users_hurt_from"] == 8000
    assert table["0"]["users_hurt_from"] is None
    assert result["levels"]["4"]["running"] >= 1
    assert "users hurt from" in capsys.readouterr().out


def test_router_loses_where_the_flow_is_slower_than_recompute():
    def record(length: int, flow: float) -> dict:
        return {
            "concurrency": 8, "remnant": length, "repeat": 0, "error": None,
            "prepare_s": flow, "flow_s": flow,
            "tpot_p95_before_s": 0.01, "tpot_p95_during_s": 0.011,
        }  # fmt: skip

    context = {"recompute_s": {1000: 0.1, 4000: 0.3}}
    summary = kv_saturation.summarize([record(1000, 0.05), record(4000, 0.4)], context, 0.2)

    row = summary["by_concurrency"][8]
    assert row["router_loses_from"] == 4000
    assert row["users_hurt_from"] is None


def test_h3_example_is_a_valid_experiment():
    path = Path(__file__).parent.parent / "examples" / "kv-transfer" / "h3.yaml"

    experiment, _ = load_experiment(path)

    assert experiment.gpu == "A100-80GB:2"
    assert "Qwen/Qwen3-8B" in experiment.server
    assert "modelsight.probes.kv_saturation" in experiment.bench
