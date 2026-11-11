import json
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_kv_peer_lookup_probe import SESSION, FakeVllm, render

from modelsight.probes import kv_peer_load as h5
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_shadow as shadow
from modelsight.probes import kv_transfer

# A stand-in for AIPerf: it records its command, holds the load briefly, writes
# one per-request record for every turn of every session in its input file
# (only the first turn of each when TURNS_PER_SESSION is set to 1), and exits
# by itself, as AIPerf does.
FAKE_AIPERF = textwrap.dedent(
    """
    import json, pathlib, sys, time

    argv = sys.argv[1:]
    flag = lambda name: argv[argv.index(name) + 1]
    artifacts = pathlib.Path(flag("--artifact-dir"))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "command.json").write_text(json.dumps(argv))
    load = pathlib.Path(sys.argv[0]).with_name("load")
    load.write_text(flag("--concurrency"))
    time.sleep(0.3)
    limit = int((pathlib.Path(sys.argv[0]).with_name("turns")).read_text() or 0) \
        if pathlib.Path(sys.argv[0]).with_name("turns").exists() else None
    with (artifacts / "profile_export.jsonl").open("w") as out:
        for line in open(flag("--input-file")):
            session = json.loads(line)
            for turn in range(len(session["turns"])):
                if limit is None or turn < limit:
                    out.write(json.dumps({"metadata": {
                        "conversation_id": session["session_id"],
                        "turn_index": turn}}) + "\\n")
    load.unlink()
    """
)

# What the fake router adds to its metrics between two scrapes: one level.
STEP = """\
vllm_router_kv_transfer_plans_total{outcome="planned"} 3
vllm_router_kv_transfer_plans_total{outcome="cost_rule"} 1
vllm_router_kv_transfer_plans_total{outcome="source_not_busy"} 6
vllm_router_kv_transfer_plans_total{outcome="missing_tokens"} MISSING
vllm_router_kv_transfer_estimated_cost_ms_bucket{path="fetch",le="10"} 1
vllm_router_kv_transfer_estimated_cost_ms_bucket{path="fetch",le="20"} 3
vllm_router_kv_transfer_estimated_cost_ms_bucket{path="fetch",le="50"} 4
vllm_router_kv_transfer_estimated_cost_ms_bucket{path="fetch",le="+Inf"} 4
vllm_router_kv_transfer_estimated_cost_ms_sum{path="fetch"} 60
vllm_router_kv_transfer_estimated_cost_ms_count{path="fetch"} 4
vllm_router_kv_transfer_estimated_saved_ms_bucket{le="100"} 3
vllm_router_kv_transfer_estimated_saved_ms_bucket{le="+Inf"} 3
vllm_router_kv_transfer_estimated_saved_ms_sum 240
vllm_router_kv_transfer_estimated_saved_ms_count 3
vllm_router_prefix_routing_decisions_total{policy="kv_aware",stage="regular",outcome="prefix_cost"} 10
"""


class FakeRouter:
    """Serves /health and a /metrics whose counters grow one STEP per scrape."""

    def __init__(self, engine: FakeVllm, missing: int = 0) -> None:
        self.scrapes = 0
        self.missing = missing
        self.engine = engine
        self.unplanned = 0  # requests the engine serves that the router never plans
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                body = b"" if self.path == "/health" else fake.metrics().encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def metrics(self) -> str:
        times = self.scrapes
        self.scrapes += 1
        if times % 2:  # the probe reads the router before and after each level
            self.engine.served += 10 + self.missing + self.unplanned
        lines = ["# TYPE vllm_router_kv_transfer_plans_total counter"]
        for line in STEP.replace("MISSING", str(self.missing)).splitlines():
            name, value = line.rsplit(" ", 1)
            lines.append(f"{name} {float(value) * times}")
        return "\n".join(lines) + "\n"


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_transfer, "METRICS_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(h4, "RESET_ATTEMPTS", 1)
    monkeypatch.setattr(h4, "chat_renderer", lambda model: render)
    monkeypatch.setattr(h5, "SAMPLE_INTERVAL_SECONDS", 0.05)
    a = FakeVllm(tmp_path / "server-A.log")
    b = FakeVllm(tmp_path / "server-B.log")
    a.load_file = tmp_path / "load"
    router = FakeRouter(a)
    yield a, b, router
    for server in (a, b, router):
        server.server.shutdown()


def args(fakes, tmp_path, levels="2,4") -> list[str]:
    a, b, router = fakes
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"messages": json.dumps(SESSION)}) + "\n")
    script = tmp_path / "aiperf.py"
    script.write_text(FAKE_AIPERF)
    (tmp_path / "out").mkdir(exist_ok=True)
    a.log.write_text("router log stand-in")
    return [
        "--model", "m", "--router", router.url, "--router-metrics", router.url,
        "--worker", a.url, "--worker", b.url, "--log", str(a.log),
        "--traces", str(traces), "--levels", levels, "--sessions", "1",
        "--ramp-seconds", "0", "--aiperf-command", f"{sys.executable} {script}",
        "--output", str(tmp_path / "out" / "results.json"),
    ]  # fmt: skip


def results(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "results.json").read_text())


def test_parse_metrics_keys_series_by_name_and_sorted_labels():
    text = textwrap.dedent(
        """\
        # HELP x a counter
        x_total{b="2",a="1"} 3
        y 4.5
        z{le="+Inf",path="fetch"} 7
        """
    )

    series = shadow.parse_metrics(text)

    assert series == {
        ("x_total", (("a", "1"), ("b", "2"))): 3.0,
        ("y", ()): 4.5,
        ("z", (("le", "+Inf"), ("path", "fetch"))): 7.0,
    }


def test_quantiles_interpolate_inside_a_bucket_like_prometheus():
    buckets = [(10.0, 1.0), (20.0, 3.0), (50.0, 4.0), (float("inf"), 4.0)]

    assert shadow.quantile(buckets, 0.5) == pytest.approx(15.0)  # rank 2 of 4
    assert shadow.quantile(buckets, 0.25) == pytest.approx(10.0)
    assert shadow.quantile(buckets, 1.0) == pytest.approx(50.0)
    assert shadow.quantile([(10.0, 0.0), (float("inf"), 0.0)], 0.5) is None
    # past the last finite bound, the bound is all a histogram knows
    assert shadow.quantile([(10.0, 1.0), (float("inf"), 4.0)], 0.9) == 10.0


def test_a_level_reads_only_what_it_added():
    before = shadow.parse_metrics(
        'p_total{outcome="planned"} 5\nh_bucket{le="10"} 2\nh_bucket{le="+Inf"} 2\n'
        "h_sum 8\nh_count 2\n"
    )
    after = shadow.parse_metrics(
        'p_total{outcome="planned"} 7\np_total{outcome="no_peer"} 1\n'
        'h_bucket{le="10"} 2\nh_bucket{le="+Inf"} 4\nh_sum 108\nh_count 4\n'
    )

    added = shadow.change(before, after)

    assert shadow.by_label(added, "p_total", "outcome") == {"planned": 2, "no_peer": 1}
    summary = shadow.histogram(added, "h")
    assert summary["count"] == 2 and summary["mean"] == 50
    assert summary["p50"] == 10.0  # both new samples are above the last bound


def test_histogram_selects_one_label_value():
    series = shadow.parse_metrics(STEP.replace("MISSING", "0"))

    fetch = shadow.histogram(series, shadow.COST, path="fetch")
    wait = shadow.histogram(series, shadow.COST, path="wait")

    assert fetch["count"] == 4 and fetch["mean"] == 15
    assert wait == {"count": 0.0, "mean": None, "p50": None, "p90": None, "p99": None}


def test_sessions_keep_only_those_with_a_prefix_to_share():
    picked = shadow.session_turns([SESSION, SESSION[:3], SESSION], render, 10_000, 5)

    assert len(picked) == 2  # SESSION[:3] has one user turn: nothing to share
    assert len(picked[0]) == 3
    assert len(shadow.session_turns([SESSION] * 4, render, 10_000, 3)) == 3  # the limit


def test_every_level_writes_the_same_sessions_and_counts_their_turns(tmp_path):
    sessions = [["a", "b", "c"], ["d", "e"]]

    turns = shadow.write_sessions(tmp_path / "s.jsonl", sessions, 8)

    rows = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert turns == 5
    assert [r["session_id"] for r in rows] == ["shadow-8-0", "shadow-8-1"]
    assert [t["text"] for t in rows[1]["turns"]] == ["d", "e"]


def test_coverage_counts_complete_partial_missing_and_repeated_sessions(tmp_path):
    records = tmp_path / "r.jsonl"
    ids = ["shadow-8-0"] * 3 + ["shadow-8-1"] * 1 + ["shadow-8-3"] * 4  # 3: two extra turns
    records.write_text("".join(
        json.dumps({"metadata": {"conversation_id": i}}) + "\n" for i in ids
    ))  # fmt: skip
    sessions = [["a", "b", "c"], ["d", "e"], ["f"], ["g", "h"]]

    coverage = shadow.turn_coverage(records, sessions, 8)

    assert coverage == {"complete": 1, "partial": 1, "missing": 1, "repeated": 2}
    assert shadow.turn_coverage(tmp_path / "none.jsonl", sessions, 8)["missing"] == 4


def test_aiperf_replays_every_session_once_against_the_router(tmp_path):
    argv = shadow.parse_args(
        ["--model", "m", "--router", "http://r", "--router-metrics", "http://r:9",
         "--worker", "http://a", "--worker", "http://b", "--aiperf-command", "ap"]
    )  # fmt: skip

    command = shadow.aiperf_command(argv, 16, tmp_path / "s.jsonl", tmp_path / "art", 1234)

    for flag, value in (("--url", "http://r"), ("--concurrency", "16"),
                        ("--num-sessions", "1234"),
                        ("--dataset-sampling-strategy", "sequential"),
                        ("--custom-dataset-type", "multi_turn")):  # fmt: skip
        assert command[command.index(flag) + 1] == value
    # a level is a fixed amount of work, never a fixed time
    assert "--benchmark-duration" not in command
    assert "--benchmark-grace-period" not in command
    assert "--no-server-metrics" in command  # its polls would be routed as requests
    # a request count lets AIPerf restart sessions and cut long ones short
    assert "--request-count" not in command


def test_two_workers_are_required():
    with pytest.raises(SystemExit):
        shadow.parse_args(["--model", "m", "--router", "r", "--router-metrics", "r",
                       "--worker", "a"])  # fmt: skip


def test_sweep_reads_each_level_from_the_router(fakes, tmp_path):
    a, b, router = fakes

    assert shadow.main(args(fakes, tmp_path)) == 0

    result = results(tmp_path)
    assert result["levels"] == [2, 4]
    assert all(v["ok"] for v in result["checks"].values())
    for level in ("2", "4"):
        level_result = result["results"][level]
        assert level_result["outcomes"] == {"planned": 3, "cost_rule": 1,
                                            "source_not_busy": 6}  # fmt: skip
        assert level_result["outcome_shares"]["planned"] == pytest.approx(0.3)
        assert level_result["fetch_ms"]["p50"] == pytest.approx(15.0)
        assert level_result["saved_ms"]["mean"] == 80
        assert level_result["info"]["aiperf_finished"]
        assert level_result["info"]["turns"] == 3
        assert level_result["info"]["coverage"] == {
            "complete": 1, "partial": 0, "missing": 0, "repeated": 0}  # fmt: skip
        assert level_result["info"]["load"]["A"]["max_running"] == int(level)
    command = json.loads((tmp_path / "out" / "aiperf-4" / "command.json").read_text())
    assert command[command.index("--url") + 1] == router.url
    assert a.resets == b.resets == 3  # once before the sweep, once after each level
    assert (tmp_path / "out" / "server-A.log").exists()


def test_a_level_the_router_could_not_read_fails_inputs(fakes, tmp_path):
    _, _, router = fakes
    router.missing = 20  # 20 of 30 plans had no token ids

    assert shadow.main(args(fakes, tmp_path, levels="2")) == 1

    checks = results(tmp_path)["checks"]
    assert checks["inputs"]["ok"] is False
    assert checks["inputs"]["failure_share"]["2"] == pytest.approx(0.667)
    assert checks["planner_ran"]["ok"] is True


def test_requests_the_router_did_not_plan_once_fail_the_match(fakes, tmp_path):
    _, _, router = fakes
    router.unplanned = 10  # served 20 requests, planned 10

    assert shadow.main(args(fakes, tmp_path, levels="2")) == 1

    check = results(tmp_path)["checks"]["one_plan_per_request"]
    assert check["ok"] is False
    assert check["plans"] == {"2": 10} and check["served"] == {"2": 20}


def test_a_level_that_stopped_before_the_last_turn_fails_every_turn_sent(fakes, tmp_path):
    argv = args(fakes, tmp_path, levels="2")
    (tmp_path / "turns").write_text("1")  # the fake AIPerf sends only turn 0

    assert shadow.main(argv) == 1

    check = results(tmp_path)["checks"]["every_turn_sent"]
    assert check["ok"] is False
    assert check["coverage"]["2"]["partial"] == 1
