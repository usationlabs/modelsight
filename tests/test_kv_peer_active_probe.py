import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelsight.probes import kv_peer_active as active
from modelsight.probes import kv_peer_load as h5
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_transfer

BLOCK = 4
BYTES_PER_TOKEN = 100
UCX_TABLE = (
    "| ucp_context_0 inter-node cfg#1 | remote memory read by ucp_get* from host memory |\n"
    "|                         0..inf | zero-copy                 cuda_ipc/cuda |\n"
)


def encode(text: str) -> list[int]:
    """A tokenizer: one id per character."""
    return [100 + ord(c) % 500 for c in text]


def stamp() -> str:
    now = time.time()
    return time.strftime("%m-%d %H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"


class FakeEngine:
    """Enough of a branch vLLM for the probe: a prefix cache, resets, the
    peer_lookup record, and a peer lookup that pulls what the peer holds and
    logs the result with the request's id."""

    def __init__(self, name: str, log: Path, port: int) -> None:
        self.name = name
        self.log = log
        self.port = port
        self.peer: FakeEngine | None = None
        self.cache: set[tuple[int, ...]] = set()
        self.served = 0
        self.transfers = 0
        self.bytes = 0
        self.record_enabled = True
        self.lock = threading.Lock()
        log.write_text("INFO GPU KV cache size: 1,000,000 tokens\n" + UCX_TABLE)
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
                elif self.path == "/kv_event_sources":
                    self._reply(json.dumps({"0": {
                        "endpoint": "tcp://*:1", "peer_lookup": fake.record()}}).encode())
                else:
                    self._reply(fake.metrics().encode())

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length") or 0)
                if self.path.startswith("/reset_prefix_cache"):
                    fake.cache.clear()
                    self._reply(json.dumps({"success": True}).encode())
                    return
                body = json.loads(self.rfile.read(size))
                self._reply(fake.complete(body, self.headers.get("X-Request-Id")))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def record(self) -> dict:
        return {"enabled": self.record_enabled, "engine_id": f"engine-{self.name}",
                "side_channel_host": "localhost", "side_channel_port": self.port,
                "tp_size": 1, "dcp_size": 1, "pp_size": 1, "connector": "NixlConnector",
                "connector_version": "3", "layout_fingerprint": "abc"}

    def write(self, line: str) -> None:
        with self.lock, self.log.open("a") as f:
            f.write(line + "\n")

    def complete(self, body: dict, request_id: str | None) -> bytes:
        prompt = tuple(body["prompt"])
        params = body.get("kv_transfer_params") or {}
        vllm_id = f"cmpl-{request_id or 'x'}-0-1a2b3c4d"
        if params.get("peer_lookup"):
            assert self.peer is not None and params["remote_port"] == self.peer.port
            self.peer.write("DEBUG Peer lookup_pin served")
            held = max((h4.expected_hit(list(p), list(prompt), BLOCK)
                        for p in self.peer.cache), default=0)
            if held:
                self.write(f"(EngineCore pid=1) DEBUG {stamp()} [pull_scheduler.py:264] "
                           f"Peer lookup_pin hit for {vllm_id}: {held} tokens under pin p1")
                self.transfers += 1
                self.bytes += held * BYTES_PER_TOKEN
            else:
                self.write(f"(EngineCore pid=1) DEBUG {stamp()} [pull_scheduler.py:166] "
                           f"Peer lookup for {vllm_id} missed: not_cached")
        with self.lock:
            self.cache.add(prompt)
            self.served += 1
        word = f"t{sum(prompt) % 7}"
        chunk = {"choices": [{"text": word, "logprobs": {
            "tokens": [word], "token_logprobs": [-0.1]}}]}
        return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

    def metrics(self) -> str:
        return (
            "vllm:num_requests_running 0\n"
            "vllm:num_requests_waiting 0\n"
            f'vllm:nixl_bytes_transferred_count{{m="q"}} {self.transfers}\n'
            f'vllm:nixl_bytes_transferred_sum{{m="q"}} {self.bytes}\n'
            f'vllm:request_success_total{{m="q"}} {self.served}\n'
            f"vllm:inter_token_latency_seconds_sum {self.served * 0.03}\n"
            f"vllm:inter_token_latency_seconds_count {self.served}\n"
        )


class FakeRouter:
    """Sends every other request to B; in active mode those carry a fetch from
    A, built from A's /kv_event_sources record, and count as sent."""

    def __init__(self, mode: str, a: FakeEngine, b: FakeEngine) -> None:
        self.mode = mode
        self.count = 0
        self.sent = 0
        self.planned = 0
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _reply(self, body: bytes) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._reply(b"" if self.path == "/health" else fake.metrics().encode())

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(size))
                with fake.lock:
                    fake.count += 1
                    to_b = fake.count % 2 == 0
                    if to_b:
                        fake.planned += 1
                if to_b and fake.mode == "active":
                    with urllib.request.urlopen(f"{a.url}/kv_event_sources") as r:
                        record = json.load(r)["0"]["peer_lookup"]
                    body["kv_transfer_params"] = {"do_remote_prefill": True,
                                                  "peer_lookup": True,
                                                  **active.address(record)}
                    with fake.lock:
                        fake.sent += 1
                engine = b if to_b else a
                self._reply(engine.complete(body, self.headers.get("X-Request-Id")))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def metrics(self) -> str:
        return (f'vllm_router_kv_transfer_plans_total{{outcome="planned"}} {self.planned}\n'
                f'vllm_router_kv_transfer_requests_total{{outcome="sent"}} {self.sent}\n')


SESSION = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "content": "ls"},
    {"role": "user", "content": "a.py b.py"},
    {"role": "assistant", "content": "cat a.py"},
]


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_transfer, "METRICS_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(h4, "RESET_ATTEMPTS", 1)
    monkeypatch.setattr(h5, "SAMPLE_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(active, "text_tokenizer", lambda model: encode)
    a = FakeEngine("A", tmp_path / "server-A.log", 5600)
    b = FakeEngine("B", tmp_path / "server-B.log", 5601)
    a.peer, b.peer = b, a
    routers = {"active": FakeRouter("active", a, b), "shadow": FakeRouter("shadow", a, b)}
    yield a, b, routers
    for server in (a.server, b.server, *(r.server for r in routers.values())):
        server.shutdown()


def argv(fakes, tmp_path, *extra: str) -> list[str]:
    a, b, routers = fakes
    traces = tmp_path / "traces.jsonl"
    traces.write_text("".join(json.dumps({"messages": SESSION}) + "\n" for _ in range(4)))
    (tmp_path / "out").mkdir(exist_ok=True)
    return [
        "--model", "m", "--producer", a.url, "--consumer", b.url,
        "--producer-log", str(a.log), "--consumer-log", str(b.log),
        "--router", f"active={routers['active'].url},{routers['active'].url}",
        "--router", f"shadow={routers['shadow'].url},{routers['shadow'].url}",
        "--contexts", "40,80", "--context-count", "4", "--verify-contexts", "2",
        "--rates", "50", "--sessions", "4", "--turns", "2",
        "--max-model-len", "1000", "--block-size", str(BLOCK), "--smoke-burst", "4",
        "--output-tokens", "4", "--traces", str(traces), "--min-sends", "1",
        "--output", str(tmp_path / "out" / "results.json"), *extra,
    ]  # fmt: skip


def results(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "results.json").read_text())


def test_full_run_passes_every_check(fakes, tmp_path):
    assert active.main(argv(fakes, tmp_path)) == 0

    result = results(tmp_path)
    assert all(v["ok"] for v in result["checks"].values())
    assert result["smoke"]["cuda_ipc"]
    assert result["smoke"]["burst"]["requests"]["sent"] > 0
    sweep = result["sweep"]["50.0"]
    assert sweep["active"]["summary"]["fetches"] > 0
    assert sweep["shadow"]["summary"]["fetches"] == 0
    # Both arms sent the same prompts.
    prompts = {arm: sorted((r["session"], r["turn"], r["prompt_tokens"])
                           for r in sweep[arm]["records"]) for arm in active.ARMS}
    assert prompts["active"] == prompts["shadow"]
    comparison = result["comparison"]["50.0"]
    assert comparison["fetches"] == comparison["paired"] > 0


def test_engine_without_peer_record_stops_before_any_load(fakes, tmp_path):
    a, _, routers = fakes
    a.record_enabled = False

    assert active.main(argv(fakes, tmp_path)) == 1

    result = results(tmp_path)
    assert result["checks"] == {"smoke": {"ok": False}}
    assert "A rank 0: not enabled" in result["smoke"]["record_problems"]
    assert routers["active"].count == routers["shadow"].count == 0


def test_burst_without_sends_fails_smoke(fakes, tmp_path):
    routers = fakes[2]
    routers["active"].mode = "shadow"  # plans, but sends nothing

    assert active.main(argv(fakes, tmp_path)) == 1

    assert not results(tmp_path)["smoke"]["burst"]["ok"]
    assert routers["shadow"].count == 0


def test_smoke_only_stops_after_verify(fakes, tmp_path):
    routers = fakes[2]

    assert active.main(argv(fakes, tmp_path, "--smoke-only")) == 0

    result = results(tmp_path)
    # Two sampled contexts spanning the palette's lengths.
    assert sorted(c["context_tokens"] for c in result["verified"].values()) == [40, 80]
    assert set(result["checks"]) == {"smoke", "verified", "no_leaks"}
    assert routers["shadow"].count == 0


def test_contexts_have_exact_lengths_and_share_no_prefix():
    traces = [SESSION, SESSION]
    contexts = active.build_contexts(encode, traces, 4, [40, 80])

    assert {n: len(c) for n, c in contexts.items()} == {0: 40, 1: 80, 2: 40, 3: 80}
    assert contexts[0][:BLOCK] != contexts[1][:BLOCK]
    # Same length, different index: the header keeps their first blocks apart.
    assert contexts[0][:BLOCK] != contexts[2][:BLOCK]


def test_session_prompts_extend_each_other_and_stop_at_the_window():
    context = list(range(10))
    pairs = [([1, 2], [3]), ([4], [5, 6]), ([7] * 50, [8])]

    prompts = active.session_prompts(context, pairs, max_tokens=40, output_tokens=4)

    assert prompts == [context + [1, 2], context + [1, 2, 3, 4]]


def test_arrivals_are_the_same_for_both_arms_and_differ_by_rate():
    assert active.arrivals(2.0, 5, 0) == active.arrivals(2.0, 5, 0)
    assert active.arrivals(2.0, 5, 0)[0] == 0.0
    assert active.arrivals(2.0, 5, 0) != active.arrivals(4.0, 5, 0)


def test_lookups_match_our_ids_and_read_the_log_time():
    year = 2026
    log = (
        "(EngineCore pid=1) DEBUG 11-17 10:00:00.250 [p.py:1] "
        "Peer lookup_pin hit for cmpl-pf0000001x-0-ab12cd34: 4096 tokens under pin p\n"
        "(EngineCore pid=1) DEBUG 11-17 10:00:01.000 [p.py:2] "
        "Peer lookup for cmpl-pf0000002x-0-ffffffff missed: evicted\n"
        "(EngineCore pid=1) DEBUG 11-17 10:00:02.000 [p.py:3] "
        "Peer lookup_pin hit for cmpl-other-0: 64 tokens under pin q\n"
    )

    found = active.lookups(log, year)

    assert set(found) == {"pf0000001x", "pf0000002x"}
    assert found["pf0000001x"]["hit_tokens"] == 4096
    assert found["pf0000002x"]["miss"] == "evicted"
    assert found["pf0000002x"]["at"] - found["pf0000001x"]["at"] == pytest.approx(0.75)


def test_record_problem_matches_the_routers_validation():
    good = {"enabled": True, "engine_id": "e", "side_channel_host": "h",
            "side_channel_port": 5600, "tp_size": 1, "dcp_size": 1, "pp_size": 1,
            "connector": "NixlConnector", "connector_version": "3",
            "layout_fingerprint": "f"}  # fmt: skip

    assert active.record_problem(good) is None
    assert active.record_problem({**good, "enabled": False}) == "not enabled"
    assert active.record_problem({**good, "engine_id": " "}) == "empty engine_id"
    assert active.record_problem({**good, "side_channel_port": 0}) == "bad side_channel_port"
    assert active.record_problem({**good, "tp_size": 0}) == "zero size"


def test_a_fetch_slower_than_the_same_request_recomputed_counts_as_lost():
    shadow = [
        {"id": "pf0000005x", "session": 0, "turn": 0, "context": 0, "ttft_s": 0.5},
        {"id": "pf0000006x", "session": 1, "turn": 0, "context": 0, "ttft_s": 0.5},
    ]
    records = [
        {"id": "pf0000001x", "session": 0, "turn": 0, "context": 0, "ttft_s": 0.2,
         "hit_tokens": 7936, "queue_s": 0.05, "reply_s": 0.02, "pull_s": 0.1},
        {"id": "pf0000002x", "session": 1, "turn": 0, "context": 0, "ttft_s": 0.9,
         "hit_tokens": 7936, "queue_s": 0.6, "reply_s": 0.2, "pull_s": 0.1},
        {"id": "pf0000003x", "session": 2, "turn": 0, "context": 0, "ttft_s": 0.5,
         "miss": "evicted"},
        {"id": "pf0000004x", "session": 3, "turn": 0, "context": 0, "error": "boom"},
    ]  # fmt: skip
    verified = {0: {"context_tokens": 8000, "recompute": {"ttft_s": 0.6}}}
    length_of = {0: 8000}

    summary = active.summarize_arm({"records": records}, length_of)
    comparison = active.compare({"shadow": {"records": shadow},
                                 "active": {"records": records}}, verified, length_of)  # fmt: skip

    assert summary["fetches"] == 2
    assert summary["misses"] == {"evicted": 1}
    assert summary["errors"] == 1
    assert (comparison["paired"], comparison["won"], comparison["lost"]) == (2, 1, 1)
    assert comparison["pull_beats_idle_recompute"] == 2
