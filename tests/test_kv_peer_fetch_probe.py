import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.probes import kv_peer_fetch

BLOCK = 16


class FakeVllm:
    """A vLLM server with a block prefix cache that behaves like stock NIXL.

    A computes the blocks it lacks, which takes time, and hands out blocks for
    its whole prompt. B fails a pull for a longer prompt, and answers a shorter
    one with the wrong output.
    """

    def __init__(self) -> None:
        self.blocks: set[int] = set()
        self.held: dict[int, list[int]] = {}
        self.hits = 0
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _reply(self, body: bytes, status: int = 200) -> None:
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._reply(b"")
                    return
                self._reply(f'vllm:prefix_cache_hits_total{{m="q"}} {fake.hits}\n'.encode())

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                status, reply = fake.answer(body)
                self._reply(reply, status)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, body: dict) -> tuple[int, bytes]:
        prompt = body["prompt"]
        params = body.get("kv_transfer_params") or {}
        keys, parent = [], None
        for i in range(0, len(prompt) - BLOCK + 1, BLOCK):
            parent = hash((parent, tuple(prompt[i : i + BLOCK])))
            keys.append(parent)
        cached = 0
        for key in keys:
            if key not in self.blocks:
                break
            cached += BLOCK
        text = f"t{sum(prompt) % 997}"
        if params.get("do_remote_prefill"):
            pulled = len(params["remote_block_ids"][0]) * BLOCK
            if len(prompt) > pulled:
                return 500, b"assertion failed"
            if len(prompt) < pulled:
                text = "wrong"
            cached = len(prompt)
        else:
            self.hits += cached
            time.sleep((len(prompt) - cached) / 2e5)
        self.blocks.update(keys)
        usage = {"prompt_tokens_details": {"cached_tokens": cached}}
        if body.get("stream"):
            return 200, b'data: {"x":1}\n\ndata: [DONE]\n\n'
        reply = {"choices": [{"text": text}], "usage": usage}
        if params.get("do_remote_decode"):
            reply["kv_transfer_params"] = {
                "do_remote_prefill": True,
                "remote_block_ids": [list(range(len(prompt) // BLOCK))],
            }
        return 200, json.dumps(reply).encode()


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
        "--prefix-lengths", "4000,16000",
        "--repeats", "2",
        "--output", str(tmp_path / "out" / "results.json"),
    ]  # fmt: skip
    producer.server.shutdown()
    consumer.server.shutdown()


def test_probe_confirms_both_problems_on_stock_behaviour(probe_args, tmp_path, capsys):
    status = kv_peer_fetch.main(probe_args)

    result = json.loads((tmp_path / "out" / "results.json").read_text())
    assert result["verdicts"]["problem1"].startswith("CONFIRMED"), result["verdicts"]
    assert result["verdicts"]["problem2"].startswith("CONFIRMED"), result["verdicts"]
    assert status == 0
    row = result["summary"]["by_length"]["16000"]
    assert row["hit"]["a_hit_fraction"] == 1.0
    assert row["partial"]["a_hit_fraction"] == 0.5
    assert row["miss"]["a_hit_fraction"] == 0.0
    assert row["partial"]["a_blocks"] == row["hit"]["a_blocks"] == 1000
    assert row["b_longer"]["error"]
    assert row["b_shorter"]["output_matches"] is False
    assert "problem1: CONFIRMED" in capsys.readouterr().out


def test_flow_arms_run_before_the_subset_arms(probe_args, tmp_path):
    kv_peer_fetch.main(probe_args)

    arms = [r["arm"] for r in json.loads((tmp_path / "out" / "results.json").read_text())["records"]]
    first_subset = min(arms.index(name) for name in kv_peer_fetch.SUBSET_ARMS)
    assert set(arms[:first_subset]) == set(kv_peer_fetch.FLOW_ARMS)
    assert len(arms) == 4 * 2 * 2 + 2 * 2


def test_a_correct_subset_pull_refutes_problem2():
    row = {"b_longer": {"error": None, "output_matches": True}}

    assert kv_peer_fetch.problem2({"by_length": {4000: row}}).startswith("REFUTED")


def test_a_cheap_miss_refutes_problem1():
    hit = {"prepare_s": 0.1, "a_hit_fraction": 1.0, "a_blocks": 250}
    miss = {"prepare_s": 0.15, "a_hit_fraction": 0.0, "a_blocks": 250}
    summary = {"by_length": {4000: {"hit": hit, "partial": hit, "miss": miss}}, "flow_errors": []}

    assert kv_peer_fetch.problem1(summary).startswith("REFUTED")


def test_h3_example_is_a_valid_experiment():
    path = Path(__file__).parent.parent / "examples" / "kv-transfer" / "h3.yaml"

    experiment, _ = load_experiment(path)

    assert experiment.gpu == "A100-80GB:2"
    assert "--enable-prompt-tokens-details" in experiment.server
    assert "modelsight.probes.kv_peer_fetch" in experiment.bench
