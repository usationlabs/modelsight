import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelsight.config import load_experiment
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_transfer

BLOCK = 4
BYTES_PER_TOKEN = 100


def held_prefix(held: set[tuple[int, ...]], prompt: tuple[int, ...]) -> int:
    best = 0
    for p in held:
        n = 0
        for a, b in zip(p, prompt):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


class FakeVllm:
    """Enough of the branch's vLLM: a prefix cache, resets, stock pulls, and a
    peer lookup that pulls what the producer holds and logs it there."""

    def __init__(self, log: Path, producer: "FakeVllm | None" = None) -> None:
        self.log = log
        self.producer = producer
        self.cache: set[tuple[int, ...]] = set()
        self.salts: dict[tuple[int, ...], set[str]] = {}
        self.requests: list[dict] = []
        self.resets = 0
        self.load_file: Path | None = None  # exists while a fake AIPerf runs
        self.served = 0
        self.transfers = 0
        self.bytes = 0
        self.leak = False
        self.serve_peer = False
        self.wrong_tokens = False
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
                size = int(self.headers.get("Content-Length") or 0)
                if self.path.startswith("/reset_prefix_cache"):
                    fake.cache.clear()
                    fake.salts.clear()
                    fake.resets += 1
                    self._reply(json.dumps({"success": not fake.leak}).encode())
                    return
                body = json.loads(self.rfile.read(size))
                request_id = self.headers.get("X-Request-Id")
                fake.requests.append({**body, "_request_id": request_id})
                self._reply(fake.complete(body, request_id))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def holds(self, salt: str) -> set[tuple[int, ...]]:
        return {p for p in self.cache if salt in self.salts.get(p, {""})}

    def complete(self, body: dict, request_id: str | None = None) -> bytes:
        prompt = tuple(body["prompt"])
        salt = body.get("cache_salt") or ""
        params = body.get("kv_transfer_params") or {}
        if request_id and self.log.exists():
            with self.log.open("a") as f:
                f.write(f"DEBUG request_finished(cmpl-{request_id})\n")
        if params.get("peer_lookup"):
            producer = self.producer
            hit = h4.expected_hit(
                list(
                    max(
                        producer.holds(salt),
                        key=lambda p: held_prefix({p}, prompt),
                        default=(),
                    )
                ),
                list(prompt),
                BLOCK,
            )
            with producer.log.open("a") as f:
                f.write(
                    "INFO handled LOOKUP_PIN_MSG\n"
                )
            if producer.serve_peer:
                producer.served += 1
                with producer.log.open("a") as f:
                    f.write(f"DEBUG request_finished(cmpl-{request_id})\n")
            if hit:
                self.transfers += 1
                self.bytes += hit * BYTES_PER_TOKEN
        elif params.get("do_remote_prefill"):
            self.transfers += 1
            self.bytes += (
                h4.expected_hit(list(prompt), list(prompt), BLOCK) * BYTES_PER_TOKEN
            )
        self.cache.add(prompt)
        self.salts.setdefault(prompt, set()).add(salt)
        self.served += 1
        if not body.get("stream"):
            out = {"do_remote_prefill": True, "remote_engine_id": "a",
                   "remote_host": "h", "remote_port": 1, "remote_block_ids": [0]}  # fmt: skip
            return json.dumps({"kv_transfer_params": out}).encode()
        word = "x" if self.wrong_tokens and params else f"t{sum(prompt) % 7}"
        chunk = {
            "choices": [
                {"text": word, "logprobs": {"tokens": [word], "token_logprobs": [-0.1]}}
            ]
        }
        return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

    def metrics(self) -> str:
        running = 0
        if self.load_file is not None and self.load_file.exists():
            running = int(self.load_file.read_text() or 0)
        return (
            f"vllm:num_requests_running {running}\n"
            "vllm:num_requests_waiting 0\n"
            f'vllm:nixl_bytes_transferred_count{{m="q"}} {self.transfers}\n'
            f'vllm:nixl_bytes_transferred_sum{{m="q"}} {self.bytes}\n'
            f'vllm:request_success_total{{m="q",reason="stop"}} {self.served}\n'
        )


SESSION = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "content": "ls"},
    {"role": "user", "content": "a.py b.py"},
    {"role": "assistant", "content": "cat a.py"},
    {"role": "user", "content": "def f(): pass"},
    {"role": "assistant", "content": "done"},
]


ROLES = {"system": 1, "user": 2, "assistant": 3}


def render(messages: list[dict]) -> list[int]:
    """A chat template: a role id, then the content. The generation prompt is
    the assistant's role id, so a later turn extends an earlier one."""
    ids = []
    for m in messages:
        ids += [ROLES[m["role"]]] + [100 + ord(c) for c in m["content"]] + [9]
    return ids + [ROLES["assistant"]]


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_transfer, "METRICS_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(h4, "RESET_ATTEMPTS", 1)
    monkeypatch.setattr(h4, "chat_renderer", lambda model: render)
    producer = FakeVllm(tmp_path / "server-A.log")
    consumer = FakeVllm(tmp_path / "server-B.log", producer)
    producer.log.write_text("")
    consumer.log.write_text("")
    yield producer, consumer
    producer.server.shutdown()
    consumer.server.shutdown()


def args(fakes, tmp_path, mode="peer") -> list[str]:
    producer, consumer = fakes
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"messages": json.dumps(SESSION)}) + "\n")
    return [
        "--model", "m", "--producer", producer.url, "--consumer", consumer.url,
        "--producer-log", str(producer.log), "--consumer-log", str(consumer.log),
        "--mode", mode, "--traces", str(traces), "--sessions", "1",
        "--block-size", str(BLOCK), "--output", str(tmp_path / "out" / "results.json"),
    ]  # fmt: skip


def results(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "results.json").read_text())


def test_peer_mode_passes_every_check(fakes, tmp_path):
    (tmp_path / "out").mkdir()

    assert h4.main(args(fakes, tmp_path)) == 0

    result = results(tmp_path)
    assert {r["case"] for r in result["records"]} == set(h4.CASES)
    assert all(v["ok"] for v in result["checks"].values() if isinstance(v, dict))
    assert result["checks"]["bytes_per_token"] == BYTES_PER_TOKEN
    by_case = {r["case"]: r for r in result["records"]}
    assert by_case["miss"]["expected_hit_tokens"] == 0
    assert (
        0
        < by_case["partial"]["expected_hit_tokens"]
        < by_case["full"]["expected_hit_tokens"]
    )
    assert (tmp_path / "out" / "server-A.log").exists()


def test_peer_request_sends_producer_address_and_no_hashes(fakes, tmp_path):
    (tmp_path / "out").mkdir()
    _, consumer = fakes

    h4.main(args(fakes, tmp_path))

    peer = [r["kv_transfer_params"] for r in consumer.requests
            if (r.get("kv_transfer_params") or {}).get("peer_lookup")]  # fmt: skip
    assert len(peer) == len(h4.CASES)
    assert peer[0]["remote_engine_id"] == "a" and peer[0]["remote_port"] == 1
    assert "remote_block_hashes" not in peer[0]


def test_consumer_pulls_the_address_prefill_first(fakes, tmp_path):
    # Otherwise A holds its blocks until the lease expires and the first
    # resets fail.
    (tmp_path / "out").mkdir()
    _, consumer = fakes

    h4.main(args(fakes, tmp_path))

    first = consumer.requests[0].get("kv_transfer_params") or {}
    assert first.get("do_remote_prefill") and first.get("remote_engine_id") == "a"


def test_stock_mode_skips_the_peer_checks(fakes, tmp_path):
    (tmp_path / "out").mkdir()
    _, consumer = fakes

    assert h4.main(args(fakes, tmp_path, mode="stock")) == 0

    checks = results(tmp_path)["checks"]
    assert checks["served"]["ok"] is None and checks["pulled"]["ok"] is None
    assert not any(
        (r.get("kv_transfer_params") or {}).get("peer_lookup")
        for r in consumer.requests
    )


def test_fails_when_producer_computes_the_peer_request(fakes, tmp_path):
    (tmp_path / "out").mkdir()
    producer, _ = fakes
    producer.serve_peer = True

    assert h4.main(args(fakes, tmp_path)) == 1
    assert results(tmp_path)["checks"]["served"]["ok"] is False


def test_fails_when_a_pin_leaks(fakes, tmp_path):
    (tmp_path / "out").mkdir()
    producer, _ = fakes
    h4.main(args(fakes, tmp_path))  # warm up the fakes, then leak on the last reset
    producer.leak = True

    assert h4.reset_cache(producer.url) is False


def test_fails_when_pulled_kv_changes_the_output(fakes, tmp_path):
    (tmp_path / "out").mkdir()
    _, consumer = fakes
    consumer.wrong_tokens = True

    assert h4.main(args(fakes, tmp_path)) == 1
    assert results(tmp_path)["checks"]["equivalent"]["ok"] is False


def test_expected_hit_counts_shared_full_blocks_short_of_the_last_token():
    prompt = list(range(10))
    assert h4.expected_hit(prompt, prompt, 4) == 8
    assert h4.expected_hit(list(range(8)), prompt, 4) == 8
    assert h4.expected_hit(prompt[:7] + [99], prompt, 4) == 4
    assert h4.expected_hit(list(range(9)), list(range(9)), 4) == 8
    assert h4.expected_hit(list(range(8)), list(range(8)), 4) == 4
    assert h4.expected_hit([], prompt, 4) == 0


def test_messages_accepts_common_trace_shapes():
    sharegpt = {
        "conversations": [
            {"from": "human", "value": "a"},
            {"from": "gpt", "value": "b"},
        ]
    }
    assert h4._messages(sharegpt) == [{"role": "user", "content": "a"},
                                      {"role": "assistant", "content": "b"}]  # fmt: skip
    assert h4._messages({"trajectory": json.dumps(SESSION)}) == SESSION
    assert h4._messages({"text": "no messages"}) is None


def test_pick_turn_takes_the_last_pair_that_fits_and_extends():
    prompts = h4.turn_prompts(render, SESSION)
    assert len(prompts) == 3
    previous, current = h4.pick_turn(prompts, 10_000)
    assert (previous, current) == (prompts[1], prompts[2])
    assert current[: len(previous)] == previous
    assert h4.pick_turn(prompts, len(prompts[1])) == (prompts[0], prompts[1])
    assert h4.pick_turn(prompts, 5) is None


def test_h4_example_is_a_valid_experiment():
    path = Path(__file__).parent.parent / "examples" / "kv-transfer" / "h4.yaml"

    experiment, _ = load_experiment(path)

    assert experiment.dockerfile.endswith("Dockerfile.h4")
    assert "kv_peer_lookup" in experiment.bench
    assert [s.name for s in experiment.sources] == ["VLLM"]


def test_pull_rates_use_vllm_transfer_metrics():
    records = [
        {"peer": {"transfer": {"bytes": 4e9, "xfer_s": 2.0, "descriptors": 36}}},
        {"peer": {"transfer": {"bytes": 2e9, "xfer_s": 2.0, "descriptors": 30}},
         "stock": {"transfer": {"bytes": 1e9, "xfer_s": 0.0, "descriptors": 1}}},
    ]  # fmt: skip

    rates = h4.pull_rates(records)

    assert rates == {
        "peer": {"gbps_median": 1.5, "descriptors_median": 33, "transfers": 2}
    }


def test_stock_ttft_counts_the_prefill_request_to_a(fakes, tmp_path):
    (tmp_path / "out").mkdir()

    assert h4.main(args(fakes, tmp_path)) == 0

    result = results(tmp_path)
    stocks = [r["stock"] for r in result["records"] if "stock" in r]
    assert stocks
    for stock in stocks:
        assert stock["prefill_s"] > 0
        assert stock["ttft_s"] == stock["prefill_s"] + stock["b_ttft_s"]
    for case in ("full", "partial"):
        row = result["latency"][case]
        assert row["stock"] >= row["stock_b"]


def test_latency_reports_stock_end_to_end_and_b_alone():
    records = [
        {"case": "full", "recompute": {"ttft_s": 1.0}, "peer": {"ttft_s": 0.2},
         "stock": {"ttft_s": 0.15, "b_ttft_s": 0.1, "prefill_s": 0.05}},
    ]  # fmt: skip

    table = h4.latency(records)

    assert table["full"] == {
        "recompute": 1.0,
        "stock": 0.15,
        "peer": 0.2,
        "stock_b": 0.1,
    }
