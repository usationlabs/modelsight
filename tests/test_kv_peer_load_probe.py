import json
import sys
import textwrap

import pytest
from test_kv_peer_lookup_probe import (
    BLOCK,
    SESSION,
    FakeVllm,
    render,
)

from modelsight.probes import kv_peer_load as h5
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_transfer

# A stand-in for AIPerf: it records the load it was asked for where the fake
# servers can see it, and stops on SIGINT, as AIPerf does.
FAKE_AIPERF = textwrap.dedent(
    """
    import json, pathlib, signal, sys, time

    argv = sys.argv[1:]
    flag = lambda name: argv[argv.index(name) + 1]
    artifacts = pathlib.Path(flag("--artifact-dir"))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "command.json").write_text(json.dumps(argv))
    load = pathlib.Path(sys.argv[0]).with_name("load")

    def stop(*_):
        load.unlink()
        (artifacts / "summary.json").write_text("{}")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    load.write_text(flag("--concurrency"))
    time.sleep(600)
    """
)


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_transfer, "METRICS_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(h4, "RESET_ATTEMPTS", 1)
    monkeypatch.setattr(h4, "chat_renderer", lambda model: render)
    monkeypatch.setattr(h5, "SAMPLE_INTERVAL_SECONDS", 0.05)
    producer = FakeVllm(tmp_path / "server-A.log")
    consumer = FakeVllm(tmp_path / "server-B.log", producer)
    producer.log.write_text("")
    consumer.log.write_text("")
    producer.load_file = tmp_path / "load"
    yield producer, consumer
    producer.server.shutdown()
    consumer.server.shutdown()


def args(fakes, tmp_path, levels="0,2", extra=()) -> list[str]:
    producer, consumer = fakes
    traces = tmp_path / "traces.jsonl"
    # One session is measured; the other loads A.
    traces.write_text(
        "".join(json.dumps({"messages": json.dumps(SESSION)}) + "\n" for _ in range(2))
    )
    script = tmp_path / "aiperf.py"
    script.write_text(FAKE_AIPERF)
    (tmp_path / "out").mkdir(exist_ok=True)
    return [
        "--model", "m", "--producer", producer.url, "--consumer", consumer.url,
        "--producer-log", str(producer.log), "--consumer-log", str(consumer.log),
        "--traces", str(traces), "--sessions", "1", "--levels", levels,
        "--block-size", str(BLOCK), "--aiperf-command", f"{sys.executable} {script}",
        "--output", str(tmp_path / "out" / "results.json"),
        *extra,
    ]  # fmt: skip


def results(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "results.json").read_text())


def test_prompt_cap_shares_the_load_and_stops_at_the_context_window():
    assert h5.prompt_cap(8) == 16_000
    assert h5.prompt_cap(32) == 4_000
    # 128k / 1 would outgrow the engine, so the window decides
    assert h5.prompt_cap(1) == h5.MAX_MODEL_LEN - h5.LOAD_OUTPUT_TOKENS - 64
    assert h5.prompt_cap(2) == h5.prompt_cap(1)


def test_background_turns_stop_before_the_prompt_outgrows_the_cap():
    system, users = SESSION[0], [m for m in SESSION if m["role"] == "user"]
    reply = h5.LOAD_OUTPUT_TOKENS

    def prompt(n: int) -> int:
        # the system prompt and the first n user turns, plus the model's replies
        return len(render([system, *users[:n]])) + reply * (n - 1)

    assert len(h5.background_turns(render, SESSION, 10_000)) == 3
    assert len(h5.background_turns(render, SESSION, prompt(2))) == 2
    assert len(h5.background_turns(render, SESSION, prompt(2) - 1)) == 1
    assert h5.background_turns(render, SESSION, prompt(1) - 1) == []
    turns = h5.background_turns(render, SESSION, 10_000)
    assert turns[0].startswith("you are an agent")  # the system prompt rides along


def test_background_file_holds_multi_turn_sessions_that_fit(tmp_path):
    path = tmp_path / "bg.jsonl"

    written = h5.write_background(path, [SESSION, SESSION[:1]], render, 2)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert written == len(rows) == 1  # the session with no user turn is dropped
    assert rows[0]["session_id"].startswith("bg-2-")
    assert [t["text"] for t in rows[0]["turns"]][1:] == ["a.py b.py", "def f(): pass"]


def test_aiperf_command_runs_one_level_against_a(tmp_path):
    argv = h5.parse_args(
        ["--model", "m", "--producer", "http://a", "--consumer", "http://b",
         "--producer-log", "a", "--consumer-log", "b", "--aiperf-command", "ap"]
    )  # fmt: skip

    command = h5.aiperf_command(argv, 8, tmp_path / "bg.jsonl", tmp_path / "art")

    assert command[:2] == ["ap", "profile"]
    for flag, value in (("--url", "http://a"), ("--concurrency", "8"),
                        ("--endpoint-type", "chat"),
                        ("--input-file", str(tmp_path / "bg.jsonl")),
                        ("--custom-dataset-type", "multi_turn"),
                        ("--osl", str(h5.LOAD_OUTPUT_TOKENS)),
                        ("--artifact-dir", str(tmp_path / "art"))):  # fmt: skip
        assert command[command.index(flag) + 1] == value
    extra = json.loads(command[command.index("--extra-inputs") + 1])
    assert extra == {"ignore_eos": True, "min_tokens": h5.LOAD_OUTPUT_TOKENS}
    assert "--concurrency-ramp-duration" in command  # never a concurrency list


def test_sweep_measures_every_level_under_its_own_load(fakes, tmp_path):
    producer, _ = fakes

    assert h5.main(args(fakes, tmp_path)) == 0

    result = results(tmp_path)
    assert result["levels"] == [0, 2]
    assert {r["level"] for r in result["records"]} == {0, 2}
    assert all(v["ok"] for v in result["checks"].values() if isinstance(v, dict))
    # AIPerf ran for level 2 only, was told 2 sessions, and was stopped (its
    # fake writes a summary when it gets SIGINT).
    artifacts = tmp_path / "out" / "aiperf-2"
    assert json.loads((artifacts / "command.json").read_text())[-1] == "none"
    assert (artifacts / "summary.json").exists()
    assert not (tmp_path / "out" / "aiperf-0").exists()
    assert result["level_info"]["2"]["settle"]["reached"]
    assert result["summaries"]["2"]["a_running"] == 2
    assert result["summaries"]["0"]["a_running"] in (0, None)
    assert not producer.load_file.exists()  # the load is stopped at the end


def test_every_arm_has_its_own_salt_and_nothing_resets_under_load(fakes, tmp_path):
    producer, consumer = fakes

    h5.main(args(fakes, tmp_path, levels="2"))

    salted = [r for r in consumer.requests if r.get("cache_salt")]
    measured = {r["cache_salt"] for r in salted}
    arms = {s.rsplit("-", 1)[1] for s in measured}
    assert arms == {"rc", "st", "pe"}
    assert len(measured) == len(h4.CASES) * 3 - 1  # miss has no stock arm
    # A is reset once the load has stopped, and never while it runs.
    assert producer.resets == 1


def test_peer_request_carries_an_id_a_never_logs(fakes, tmp_path):
    producer, consumer = fakes

    h5.main(args(fakes, tmp_path, levels="2"))

    ids = [r["_request_id"] for r in consumer.requests
           if (r.get("kv_transfer_params") or {}).get("peer_lookup")]  # fmt: skip
    assert len(ids) == len(h4.CASES)
    assert all(i and i.startswith("h5-") for i in ids)
    assert not any(i in producer.log.read_text() for i in ids)


def test_a_computing_a_peer_request_fails_served(fakes, tmp_path):
    producer, _ = fakes
    producer.serve_peer = True

    assert h5.main(args(fakes, tmp_path, levels="2")) == 1

    assert results(tmp_path)["checks"]["served"]["ok"] is False


def test_peer_arm_that_pulled_too_little_under_load_counts_as_evicted():
    def record(level, pulled):
        arm = {"tokens": ["t"], "transfer": {"bytes": pulled}}
        return {"level": level, "case": "full", "expected_hit_tokens": 10,
                "recompute": {"tokens": ["t"]},
                "stock": {**arm, "transfer": {"bytes": 1000}, "pulled_tokens": 10},
                "peer": {**arm, "a_served": 0, "lookups_logged": 1}}  # fmt: skip

    full, short, idle_short = record(2, 1000), record(2, 100), record(0, 100)

    assert h5.mark_evictions([full, short, idle_short]) == 1
    assert short["peer"]["evicted"] and "evicted" not in idle_short["peer"]
    # reported, not failed, under load; an idle A that pulls too little fails
    assert h4.check([full, short], "peer", True)["pulled"]["ok"] is True
    assert h4.check([full, idle_short], "peer", True)["pulled"]["ok"] is False


def test_load_that_is_never_reached_is_reported(fakes, tmp_path):
    producer, _ = fakes

    state = h5.wait_loaded(producer.url, 4, None, timeout=0.2)

    assert state["reached"] is False and state["target"] == pytest.approx(3.6)
