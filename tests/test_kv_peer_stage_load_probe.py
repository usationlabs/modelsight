import json
import sys
import time
import uuid

import pytest
from test_kv_peer_load_probe import FAKE_AIPERF
from test_kv_peer_lookup_probe import (
    BLOCK,
    BYTES_PER_TOKEN,
    SESSION,
    FakeVllm,
    held_prefix,
    render,
)
from test_rdma_gate import H5_CUDA_IPC, RC_MLX5, TCP

from modelsight.control import ControlServer
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_peer_stage_load as h6
from modelsight.probes import kv_transfer


def stamp(offset: float = 0.0) -> str:
    now = time.time() + offset
    return time.strftime("%m-%d %H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def a_line(text: str, offset: float = 0.0) -> str:
    return f"(EngineCore pid=1) DEBUG {stamp(offset)} [peer_pins.py:1] {text}\n"


class StageFake(FakeVllm):
    """FakeVllm with a CPU tier on the producer and peer-stage log lines. A GPU
    reset keeps the CPU tier, as reset_external=false does."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cpu: set[tuple[tuple[int, ...], str]] = set()
        self.pins = 0
        self.ignore_opt_out = False  # an A that stages despite "peer_stage": false

    def complete(self, body: dict, request_id: str | None = None) -> bytes:
        params = body.get("kv_transfer_params") or {}
        prompt, salt = tuple(body["prompt"]), body.get("cache_salt") or ""
        if not params.get("peer_lookup"):
            self.cpu.add((prompt, salt))
            self.store_bytes += len(prompt) * BYTES_PER_TOKEN
            return super().complete(body, request_id)
        producer = self.producer
        hit = lambda held: max(  # noqa: E731
            (h4.expected_hit(list(p), list(prompt), BLOCK) for p in held), default=0)
        gpu = hit(producer.holds(salt))
        total = max(gpu, hit(p for p, s in producer.cpu if s == salt))
        stage = params.get("peer_stage", True) or self.ignore_opt_out
        self.pins += 1
        pin = f"peer-pin-{self.pins}"
        lines = ""
        if stage and total > gpu:
            lines += a_line(f"Peer stage submitted: pin={pin} staged={(total - gpu) // BLOCK} "
                            "in_place=0 deadline_s=1.500")
            lines += a_line(f"Peer stage done: pin={pin} ok=True seconds=0.300", 0.3)
            kept = total
        else:
            if not stage:  # B opted out: reply_budget_s = 0
                lines += a_line(f"Peer stage skipped: pin={pin} reason=reply_budget est_s=0.000")
            kept = gpu
        if kept:
            lines += a_line(f"Peer lookup_pin pinned {kept // BLOCK} of 99 blocks as {pin}", 0.4)
        else:
            lines += a_line("Peer lookup_pin missed: not_cached")
        with producer.log.open("a") as f:
            f.write(lines)
        rid = f"cmpl-{request_id}-0"
        with self.log.open("a") as f:
            f.write(a_line(f"Peer lookup_pin hit for {rid}: {kept} tokens under pin {pin}")
                    if kept else a_line(f"Peer lookup for {rid} missed: not_cached"))
        if kept:
            self.transfers += 1
            self.bytes += kept * BYTES_PER_TOKEN
        return super().complete({**body, "kv_transfer_params": {}}, request_id)


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_transfer, "METRICS_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(h4, "RESET_ATTEMPTS", 1)
    monkeypatch.setattr(h4, "chat_renderer", lambda model: render)
    monkeypatch.setattr(h6, "SAMPLE_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(h6.AEngine.__init__, "__defaults__", (0.05, 30.0))
    producer = StageFake(tmp_path / "server-A.log")
    consumer = StageFake(tmp_path / "server-B.log", producer)
    producer.log.write_text("")
    consumer.log.write_text(RC_MLX5)
    producer.load_file = tmp_path / "load"
    producer.event_sources = {"0": {"peer_lookup": {
        "enabled": True, "engine_id": "a", "side_channel_host": "127.0.0.1",
        "side_channel_port": 5600, "tp_size": 1, "cpu_stage": True}}}  # fmt: skip
    yield producer, consumer
    producer.server.shutdown()
    consumer.server.shutdown()


def args(fakes, tmp_path, levels="0,2", extra=()) -> list[str]:
    producer, consumer = fakes
    traces = tmp_path / "traces.jsonl"
    traces.write_text("".join(json.dumps({"messages": json.dumps(SESSION)}) + "\n" for _ in range(2)))
    script = tmp_path / "aiperf.py"
    script.write_text(FAKE_AIPERF)
    (tmp_path / "out").mkdir(exist_ok=True)
    return [
        "--model", "m", "--producer", producer.url, "--consumer", consumer.url,
        "--producer-log", str(producer.log), "--consumer-log", str(consumer.log),
        "--traces", str(traces), "--sessions", "1", "--levels", levels,
        "--block-size", str(BLOCK), "--bytes-per-token", str(BYTES_PER_TOKEN),
        "--aiperf-command", f"{sys.executable} {script}",
        "--output", str(tmp_path / "out" / "results.json"), *extra,
    ]  # fmt: skip


def results(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "results.json").read_text())


# Fixture records ---------------------------------------------------------------


def arm(ttft, **extra) -> dict:
    return {"ttft_s": ttft, "tokens": ["t"], "logprobs": [-0.1],
            "transfer": {"bytes": 0}, **extra}  # fmt: skip


def stage_record(level, cpu, peer_ttft, stage_ttft, est, step=300.0, outcome="staged",
                 location="cpu", **extra) -> dict:  # fmt: skip
    return {
        "level": level, "case": "cpu", "location_actual": location, "drift": False,
        "prompt_tokens": cpu + 100, "expected_hit_tokens": cpu, "cpu_tokens": cpu,
        "gpu_tokens": 0, "a_step_ms": step,
        "recompute": arm(cpu / 10000),
        "peer": arm(peer_ttft, a_served=0, lookups_logged=1, kept_tokens=0, stage_submitted=False),
        "peer_stage": arm(stage_ttft, a_served=0, lookups_logged=1, kept_tokens=0, outcome=outcome,
                          est_stage_s=est, deadline_s=1.5, stage_s=0.4, fence_wait_s=0.1,
                          reply_upper_s=0.5),
        **extra,
    }  # fmt: skip


# Log parsing ---------------------------------------------------------------------


A_TEXT = (
    "(EngineCore pid=1) DEBUG 11-27 12:00:01.250 [p.py:1] Peer stage submitted: pin=P1 staged=40 in_place=2 deadline_s=1.500\n"
    "DEBUG 11-27 12:00:01.900 [p.py:1] Peer stage done: pin=P1 ok=True seconds=0.650\n"
    "DEBUG 11-27 12:00:02.000 [p.py:1] Peer stage trimmed: pin=P2 reason=deadline kept_blocks=12\n"
    "DEBUG 11-27 12:00:02.500 [p.py:1] Peer stage skipped: pin=P3 reason=estimate est_s=0.912\n"
    "DEBUG 11-27 12:00:03.125 [p.py:1] Peer lookup_pin pinned 42 of 50 blocks as P1\n"
)
B_TEXT = (
    "DEBUG 11-27 12:00:03.500 [s.py:1] Peer lookup_pin hit for cmpl-req1-0: 672 tokens under pin P1\n"
    "DEBUG 11-27 12:00:04.000 [s.py:1] Peer lookup for cmpl-req2-0 missed: not_cached\n"
)


def test_every_log_line_parses():
    events = {e["kind"]: e for e in h6.parse_events(A_TEXT + B_TEXT)}

    assert (events["submitted"]["pin"], events["submitted"]["staged"],
            events["submitted"]["in_place"], events["submitted"]["deadline_s"]) == ("P1", 40, 2, 1.5)  # fmt: skip
    assert (events["done"]["ok"], events["done"]["seconds"]) == (True, 0.65)
    assert (events["trimmed"]["reason"], events["trimmed"]["kept_blocks"]) == ("deadline", 12)
    assert (events["skipped"]["reason"], events["skipped"]["est_s"]) == ("estimate", 0.912)
    assert (events["pinned"]["blocks"], events["pinned"]["of"], events["pinned"]["pin"]) == (42, 50, "P1")
    assert (events["hit"]["request_id"], events["hit"]["tokens"], events["hit"]["pin"]) == ("cmpl-req1-0", 672, "P1")  # fmt: skip
    assert (events["miss"]["request_id"], events["miss"]["reason"]) == ("cmpl-req2-0", "not_cached")
    assert events["pinned"]["ts"] - events["submitted"]["ts"] == pytest.approx(1.875)


def test_arm_events_join_a_and_b_by_pin_and_classify_the_outcome():
    ev = h6.arm_events(A_TEXT, B_TEXT, "req1")

    assert ev["pins"] == ["P1"] and ev["hit_tokens"] == 672
    assert h6.stage_outcome(ev) == "staged"
    # 42 blocks reached the reply, 40 of them staged
    assert h6.split_tokens(ev, "staged", 16) == (672, 32)
    miss = h6.arm_events("", B_TEXT, "req2")
    assert miss["miss_reason"] == "not_cached" and h6.stage_outcome(miss) == "miss"


@pytest.mark.parametrize("line,outcome", [
    ("Peer stage trimmed: pin=P2 reason=deadline kept_blocks=12", "trimmed_deadline"),
    ("Peer stage trimmed: pin=P2 reason=failed kept_blocks=0", "trimmed_failed"),
    ("Peer stage skipped: pin=P2 reason=estimate est_s=0.9", "skipped"),
    ("Peer stage submitted: pin=P2 staged=4 in_place=0 deadline_s=1.5", "unfinished"),
    ("Peer lookup_pin pinned 4 of 4 blocks as P2", "no_cpu"),
])  # fmt: skip
def test_outcomes(line, outcome):
    hit = "DEBUG 11-27 12:00:00.500 Peer lookup_pin hit for cmpl-r-0: 64 tokens under pin P2\n"
    assert h6.stage_outcome(h6.arm_events(f"DEBUG 11-27 12:00:00.000 {line}\n", hit, "r")) == outcome


def test_trimmed_reply_counts_only_the_gpu_run():
    ev = h6.arm_events("DEBUG 11-27 12:00:02.000 Peer stage trimmed: pin=P2 reason=deadline kept_blocks=12\n"
                       "DEBUG 11-27 12:00:02.100 Peer lookup_pin pinned 12 of 50 blocks as P2\n", "", "r")
    assert h6.split_tokens(ev, "trimmed_deadline", 16) == (192, 192)


@pytest.mark.parametrize("gpu,cpu,total,full,location", [
    (0, 0, 0, 100, "miss"), (0, 100, 100, 100, "cpu"), (40, 60, 100, 100, "cpu_tail"),
    (100, 0, 100, 100, "gpu"), (60, 0, 60, 100, "partial"),
])  # fmt: skip
def test_drift_classification_follows_what_a_held(gpu, cpu, total, full, location):
    assert h6.classify_location(gpu, cpu, total, full) == location


def test_drift_counts_per_intended_case():
    records = [
        {"case": "gpu", "location_actual": "gpu", "drift": False},
        {"case": "gpu", "location_actual": "cpu", "drift": True},
        {"case": "cpu_tail", "location_actual": "cpu", "drift": True},
    ]
    assert h6.drift_counts(records) == {
        "gpu": {"n": 2, "drifted": 1, "became": {"cpu": 1}},
        "cpu_tail": {"n": 1, "drifted": 1, "became": {"cpu": 1}}}  # fmt: skip


# LogSource ---------------------------------------------------------------------------


def test_log_source_reads_a_file_and_a_control_endpoint_by_offset(tmp_path):
    rank_dir = tmp_path / "rank-0"
    rank_dir.mkdir()
    log = rank_dir / "server-A.log"
    log.write_text("one\n")
    control = ControlServer(0, rank_dir, port=0, host="127.0.0.1").start()
    try:
        remote = h6.LogSource(f"http://127.0.0.1:{control.port}/logs/server-A.log")
        local = h6.LogSource(log)
        assert remote.name == local.name == "server-A.log"
        assert remote.end() == local.end() == 4
        log.write_text("one\ntwo\n")
        assert remote.read_from(4) == local.read_from(4) == ("two\n", 8)
        assert remote.end() == local.end() == 8
        assert remote.text() == "one\ntwo\n"
        missing = h6.LogSource(f"http://127.0.0.1:{control.port}/logs/none.log")
        assert missing.read_from(0) == ("", 0)  # 404 is an empty log
    finally:
        control.stop()


# Case preparation ----------------------------------------------------------------------


class MockA:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def warm(self, prompt, salt):
        self.calls.append(("warm", len(prompt), salt.rsplit("-", 1)[1]))

    def store_bytes(self):
        self.calls.append(("store_bytes",))
        return 0.0

    def wait_stored(self, before, expected):
        self.calls.append(("wait_stored", expected))

    def reset_gpu(self):
        self.calls.append(("reset_gpu",))


def test_prepare_builds_every_cpu_state_then_resets_once_then_the_gpu_states():
    a = MockA()
    prev, cur = list(range(32)), list(range(64))

    states = h6.prepare_level(a, [(0, prev, cur)], list(h6.CASES), ["peer", "peer_stage"],
                              8, "n", BLOCK, 10)  # fmt: skip

    names = [c[0] for c in a.calls]
    reset = names.index("reset_gpu")
    assert names.count("reset_gpu") == names.count("wait_stored") == 1
    assert names[:2] == ["store_bytes", "warm"] and names[reset - 1] == "wait_stored"
    # before the reset: only the full prompts of cpu and cpu_tail, for both arms
    assert [c[1] for c in a.calls[:reset] if c[0] == "warm"] == [64] * 4
    assert a.calls[reset - 1] == ("wait_stored", 4 * 64 * 10)
    # after it: gpu (64), partial (the previous turn, 32), cpu_tail's head (half, 32)
    after = sorted(c[1] for c in a.calls[reset + 1:])
    assert after == [32, 32, 32, 32, 64, 64]
    assert {k[1] for k in states} == {"gpu", "partial", "cpu", "cpu_tail"}  # miss has none
    assert len(set(states.values())) == len(states)  # a salt per case and arm


def test_prepare_without_cpu_cases_never_resets():
    a = MockA()
    h6.prepare_level(a, [(0, [1] * 8, [1] * 16)], ["gpu", "miss"], ["peer"], 8, "n", BLOCK, 10)
    assert [c[0] for c in a.calls] == ["warm"]


# The smoke run (A100-80GB x2): A's pool held 398,768 tokens, its CPU tier 100 GiB, and
# the two measured sessions had prompts of 26,947 and 14,903 tokens. All four stateful
# arms' states for both sessions, in one batch, were 418k GPU tokens (more than the
# pool, so the first-warmed states were evicted before measurement: gpu became partial
# or cpu) and 670k CPU tokens, which with the load's 86 GB of stores overran the tier
# (cpu became miss at level 32).
SMOKE_POOL = 398_768
SMOKE_CPU = 100 * 2**30 / 147_456
SMOKE_PAIRS = [(0, list(range(26_700)), list(range(26_947))),
               (1, list(range(14_800)), list(range(14_903)))]  # fmt: skip
FOUR_ARMS = ["stock", "peer", "peer_stage", "on_a"]


def unit_cost(unit, arms=FOUR_ARMS, block=16) -> tuple[int, int]:
    pairs, cases = unit
    parts = [h6.state_tokens(p, c, case, block, len(arms)) for _, p, c in pairs for case in cases]
    return sum(g for g, _ in parts), sum(c for _, c in parts)


def test_the_smoke_runs_one_batch_overran_a_and_units_fit_its_budgets():
    whole = unit_cost((SMOKE_PAIRS, list(h6.CASES)))
    assert whole[0] > SMOKE_POOL and whole[1] > 600_000  # the cause of the drift

    gpu_budget = (SMOKE_POOL - 128_000) * h6.GPU_STATE_FRACTION  # a loaded level
    cpu_budget = SMOKE_CPU * h6.CPU_STATE_FRACTION
    units = h6.plan_units(SMOKE_PAIRS, list(h6.CASES), FOUR_ARMS, 16, gpu_budget, cpu_budget)

    assert len(units) > 2
    covered = sorted((n, c) for pairs, cases in units for n, _, _ in pairs for c in cases)
    assert covered == sorted((n, c) for n in (0, 1) for c in h6.CASES)  # each exactly once
    for unit in units:
        gpu, cpu = unit_cost(unit)
        single = len(unit[0]) * len(unit[1]) == 1  # one that cannot be split further
        assert single or (gpu <= gpu_budget and cpu <= cpu_budget)
    # at level 0 the pool is all A's, so fewer and bigger units
    idle = h6.plan_units(SMOKE_PAIRS, list(h6.CASES), FOUR_ARMS, 16, SMOKE_POOL * 0.5, cpu_budget)
    assert len(idle) < len(units)


def test_units_without_budgets_or_with_a_session_cap():
    pairs = [(n, [1] * 32, [1] * 64) for n in range(5)]
    assert len(h6.plan_units(pairs, ["gpu", "cpu"], FOUR_ARMS, 16, None, None)) == 1
    capped = h6.plan_units(pairs, ["gpu", "cpu"], FOUR_ARMS, 16, None, None, max_sessions=2)
    assert [len(p) for p, _ in capped] == [2, 2, 1]
    assert h6.state_tokens([1] * 32, [1] * 64, "cpu", 16, 2) == (0, 128)  # CPU only
    assert h6.state_tokens([1] * 32, [1] * 64, "cpu_tail", 16, 1) == (32, 64)  # head on GPU


def test_budgets_come_from_a_pool_less_the_load_and_the_cpu_tier(fakes, tmp_path):
    producer, _ = fakes
    argv = h6.parse_args(args(fakes, tmp_path))

    idle, cpu = h6.state_budgets(argv, producer.url, 0)
    loaded, _ = h6.state_budgets(argv, producer.url, 8)

    assert idle == 10000 * BLOCK * h6.GPU_STATE_FRACTION  # the fake: 10000 blocks of 4 tokens
    assert loaded == 0.0  # 128k load tokens exceed the fake's tiny pool: one item per unit
    assert cpu == argv.cpu_gb * 2**30 / BYTES_PER_TOKEN * h6.CPU_STATE_FRACTION


def sampler(rows) -> "h6.MetricsSampler":
    s = h6.MetricsSampler("http://unused", 1)
    s.rows = [{"t": t, "steps": n, "running": r, "held": None, "preemptions": 0, "free": None}
              for t, n, r in rows]  # fmt: skip
    return s


def test_step_interval_is_the_gap_between_steps_while_a_is_busy():
    busy = sampler([(i * 0.25, i * 10, 15) for i in range(20)])  # 10 steps per 0.25 s
    assert h6.MetricsSampler.step_ms(busy, 5.0) == pytest.approx(25.0)

    # An idle A takes a step only when our own request arrives. The old
    # (last - first) / steps gave 1199 ms here; with nothing running it is 0:
    # the estimate is the copy alone, as vLLM's own (an idle gap drops its average).
    idle = sampler([(0.0, 100, 0), (1.5, 100, 0), (3.0, 100, 0), (4.5, 101, 0), (6.0, 101, 0)])
    assert h6.MetricsSampler.step_ms(idle, 10.0) == 0.0
    # steps that happen with requests running count; a gap over 2 s is an idle gap
    mixed = sampler([(0.0, 0, 0), (3.0, 5, 1), (3.25, 15, 1), (3.5, 25, 1), (7.0, 25, 0)])
    assert h6.MetricsSampler.step_ms(mixed, 10.0) == pytest.approx(25.0)
    assert h6.MetricsSampler.step_ms(sampler([]), 10.0) is None
    no_series = sampler([(0.0, None, 1), (1.0, None, 1)])
    assert h6.MetricsSampler.step_ms(no_series, 10.0) is None


def test_the_estimate_is_vllms_two_steps_plus_bytes_over_20_gbps():
    staged = {"skip_est_s": None}
    est, source = h6.a_estimate(staged, 26_944, 20.0, 147_456)
    assert source == "model"
    assert est == pytest.approx(2 * 0.020 + 26_944 * 147_456 / 20e9)  # 0.239 s
    idle, _ = h6.a_estimate(staged, 26_944, 0.0, 147_456)  # an idle A: the copy alone
    assert idle == pytest.approx(26_944 * 147_456 / 20e9)
    assert h6.a_estimate({"skip_est_s": 0.912}, 100, 20.0, 147_456) == (0.912, "log")
    assert h6.a_estimate(staged, 0, 20.0, 147_456) == (None, "none")


@pytest.mark.parametrize("est,tokens,rate,deadline,skip", [
    (0.24, 26_944, 10_000, 1.5, False),  # a 2.7 s recompute beats a 0.24 s copy
    (0.24, 1_000, 10_000, 1.5, True),  # but 1,000 tokens recompute in 0.1 s
    (1.6, 26_944, 10_000, 1.5, True),  # over the deadline: skip whatever B would take
    (1.6, 26_944, 0, 1.5, True),
    (0.24, 1_000, 0, 1.5, False),  # rate 0 turns the recompute half off
])  # fmt: skip
def test_skip_rule_matches_vllms(est, tokens, rate, deadline, skip):
    assert h6.would_skip(est, tokens, rate, deadline) is skip


def test_offline_scoring_with_the_smoke_numbers_skips_no_winners():
    # level 0 (idle A) and level 32 (step 20 ms), measured peer vs peer_stage TTFT
    def rec(level, cpu, peer, staged, step):
        est, _ = h6.a_estimate({"skip_est_s": None}, cpu, step, 147_456)
        return stage_record(level, cpu, peer, staged, est=est, step=step)

    records = [rec(0, 26_944, 2.94, 0.68, 0.0), rec(0, 14_896, 1.34, 0.39, 0.0),
               rec(32, 26_944, 2.94, 0.64, 20.0), rec(32, 14_896, 1.36, 0.51, 20.0)]  # fmt: skip

    score = h6.score_offline(records, 10_000)

    assert score["skipped_winners"] == 0 and score["right"] == score["of"] == 4
    # at an absurd rate the rule skips everything: and every skip is a winner lost
    assert h6.score_offline(records, 1e9)["skipped_winners"] == 4


# Tables and verdicts -----------------------------------------------------------------------


def sweep_records() -> list[dict]:
    # cpu tokens 4000 and 8000: staging loses; 12000 and 16000: it wins
    return [
        stage_record(16, 4000, 0.4, 0.9, est=0.9),
        stage_record(16, 8000, 0.7, 1.0, est=1.0),
        stage_record(16, 12000, 1.2, 1.0, est=1.0),
        stage_record(16, 16000, 1.6, 1.1, est=1.1, outcome="trimmed_deadline"),
    ]


def test_table_reply_groups_by_cpu_bucket_and_level():
    rows = h6.table_reply(sweep_records())

    assert [(r["cpu_tokens"], r["n"]) for r in rows] == [
        ("4-8k", 1), ("8-16k", 2), ("16-32k", 1)]  # fmt: skip
    mid = rows[1]
    assert mid["reply"]["p50"] == 0.5 and mid["stage"]["p99"] == 0.4
    assert mid["fence_wait_p50"] == 0.1 and mid["outcomes"] == {"staged": 2}
    assert rows[2]["outcomes"] == {"trimmed_deadline": 1}


def test_table_ttft_win_rates_and_savings():
    row = next(r for r in h6.table_ttft(sweep_records()) if r["size"] == "8-16k")

    assert row["n"] == 2 and row["location"] == "cpu"
    assert row["ttft"]["peer"]["p50"] == pytest.approx(1.2)  # the upper median of two
    assert row["win_vs_peer"] == 0.5  # 8000 loses, 12000 wins
    assert row["saving_vs_peer_s"] == pytest.approx(-0.05)
    # the best other arm: recompute or peer (0.7 s, then 1.2 s); stage beats it once
    assert row["win_vs_best"] == 0.5 and row["saving_vs_best_s"] == pytest.approx(-0.05)


def test_break_even_is_the_size_above_which_staging_always_wins():
    assert h6.break_even(sweep_records()) == [
        "break-even: peer_stage beats peer above 12000 cpu tokens at A step = 300 ms (level 16)"]
    losing = [stage_record(8, 9000, 0.5, 0.9, est=0.9)]
    assert "never beats" in h6.break_even(losing)[0]
    winning = [stage_record(8, 9000, 0.9, 0.5, est=0.5, step=None)]
    assert "every size" in h6.break_even(winning)[0] and "idle A" in h6.break_even(winning)[0]


def test_offline_skip_scoring():
    # 10000 tok/s: a 4000-token tail recomputes in 0.4 s, 8000 in 0.8 s, 12000 in 1.2 s
    score = h6.score_offline(sweep_records(), 10000)

    # est 0.9 > 0.4 skip (loser: right); 1.0 > 0.8 skip (loser: right);
    # 1.0 < 1.2 stage (winner: right); 1.1 > 1.6/1.6? 1.1 < 1.6 stage (winner: right)
    assert score == {"right": 4, "of": 4, "skipped_winners": 0, "staged_losers": 0, "rate": 10000}
    harsh = h6.score_offline(sweep_records(), 100000)  # recompute is so fast it skips all
    assert (harsh["right"], harsh["skipped_winners"]) == (2, 2)
    assert "right on 2 of 4; would skip 2 winners" in h6.verdicts("sweep", sweep_records(), [], 100000)[1]


def test_live_skip_scoring():
    records = [
        stage_record(32, 8000, 0.7, 0.7, est=0.95, outcome="skipped"),  # est > 0.8 s tail: right
        stage_record(32, 12000, 1.2, 0.9, est=0.9),  # staged and won: right
        stage_record(32, 8000, 0.7, 0.7, est=0.5, outcome="skipped"),  # would have won: wrong
    ]
    assert h6.score_live(records) == {"right": 2, "of": 3}
    assert "skip rule (live, level 32): right on 2 of 3" in h6.verdicts("skip", records, [], 10000)


def test_skipped_requests_are_left_out_of_break_even_and_ttft_wins():
    skipped = stage_record(16, 20000, 1.0, 1.0, est=2.0, outcome="skipped")
    records = [*sweep_records(), skipped]

    assert h6.break_even(records) == h6.break_even(sweep_records())
    row = next(r for r in h6.table_ttft(records) if r["size"] == "16-32k")
    assert row["skipped"] == 1 and row["win_vs_peer"] == 1.0  # only the trimmed-deadline row counts


def test_deadline_table_and_verdict():
    records = [
        stage_record(64, 8000, 0.8, 0.9, est=0.5),
        stage_record(64, 8000, 0.8, 2.0, est=0.5, outcome="trimmed_deadline"),
        stage_record(64, 8000, 0.8, 0.8, est=0.5, outcome="skipped"),
    ]
    rows = h6.table_deadline(records)

    assert rows[0]["deadline_s"] == 1.5 and rows[0]["trim_rate"] == pytest.approx(1 / 3)
    assert rows[0]["skipped"] == 1 and rows[0]["skip_rate"] == pytest.approx(1 / 3)
    assert rows[0]["staged_ttft_p50"] == 0.9 and rows[0]["trimmed_vs_peer_s"] == pytest.approx(1.2)
    assert h6.verdicts("deadline", records, [], 1e4, None) == [
        "deadline 1.5 s: trims 33%, skips 33% at level 64; trimmed TTFT +1.20 s vs peer"]


def test_impact_table_and_verdict():
    windows = [
        {"level": 64, "staging": True, "a_ttft_p99": 3.3}, {"level": 64, "staging": False, "a_ttft_p99": 3.0}]
    assert [r["staging"] for r in h6.table_impact(windows)] == [False, True]
    assert h6.verdicts("impact", [], windows, 1e4) == ["impact on A: TTFT p99 +10% with staging at level 64"]


def test_merge_builds_table_4_from_deadline_runs(tmp_path):
    paths = []
    for deadline, ttft, outcome in ((0.5, 2.0, "trimmed_deadline"), (1.5, 0.9, "staged")):
        record = stage_record(64, 8000, 0.8, ttft, est=0.5, outcome=outcome)
        path = tmp_path / f"r{deadline}.json"
        path.write_text(json.dumps({"a_deadline_s": deadline, "records": [record, {"error": "x"}]}))
        paths.append(str(path))
    out = tmp_path / "merged.json"

    assert h6.main(["--merge", *paths, "--output", str(out)]) == 0

    merged = json.loads(out.read_text())
    assert [(r["deadline_s"], r["trim_rate"], r["skip_rate"])
            for r in merged["tables"]["deadline"]] == [(0.5, 1.0, 0.0), (1.5, 0.0, 0.0)]
    assert len(merged["verdicts"]) == 2


# Checks ---------------------------------------------------------------------------------


def checked(records, **infos):
    info = {"level": 16, "leak_free": True, "no_draining_left": {"ok": True, "first_attempt_ok": True}, **infos}
    return h6.check_records(records, [info], {"ok": True})


def test_staged_output_must_equal_recompute_and_the_peer_arm_must_not_stage():
    good = stage_record(16, 8000, 0.8, 0.5, est=0.5)
    assert all(v["ok"] is not False for v in checked([good]).values() if isinstance(v, dict))

    bad = stage_record(16, 8000, 0.8, 0.5, est=0.5)
    bad["peer_stage"]["tokens"], bad["peer_stage"]["logprobs"] = ["x"], [-9.0]
    assert checked([bad])["staged_equivalent"]["ok"] is False
    trimmed = stage_record(16, 8000, 0.8, 0.5, est=0.5, outcome="trimmed_deadline")
    trimmed["peer_stage"]["tokens"], trimmed["peer_stage"]["logprobs"] = ["x"], [-9.0]
    assert checked([trimmed])["staged_equivalent"]["ok"] is None  # only staged records count

    leaky = stage_record(16, 8000, 0.8, 0.5, est=0.5)
    leaky["peer"]["stage_submitted"] = True
    assert checked([leaky])["peer_arm_unstaged"] == {"ok": False, "records": [0]}


def test_b_timeout_after_a_pinned_fails_pulled():
    a = ("DEBUG 11-27 12:00:01.000 [p.py:1] Peer stage submitted: pin=P1 staged=40 in_place=0 deadline_s=1.500\n"
         "DEBUG 11-27 12:00:02.100 [p.py:1] Peer stage done: pin=P1 ok=True seconds=1.100\n"
         "DEBUG 11-27 12:00:02.400 [p.py:1] Peer lookup_pin pinned 40 of 40 blocks as P1\n")
    b = "DEBUG 11-27 12:00:02.050 [s.py:1] Peer lookup for cmpl-r-0 missed: timeout\n"
    ev = h6.arm_events(a, b, "r")
    pinned, _ = h6.split_tokens(ev, h6.stage_outcome(ev), 16)

    assert pinned == 640  # A shipped it, though B had given up
    assert h6.reply_tokens(ev, pinned) == 0  # B was told nothing: the pulled check wants 0 bytes
    assert h6.reply_tokens(h6.arm_events(a, "", "r"), pinned) == 640  # no B line: A's count
    rec = stage_record(64, 8000, 0.8, 0.5, est=0.5)
    rec["peer_stage"].update(kept_tokens=0, pinned_tokens=pinned, transfer={"bytes": 0})
    assert checked([rec])["pulled"]["ok"] is True
    rec["peer_stage"]["transfer"] = {"bytes": 64000}  # bytes moved for a reply B never got
    assert checked([rec])["pulled"]["ok"] is False


def test_pulled_bytes_follow_the_reply_and_leaks_fail():
    r = stage_record(16, 8000, 0.8, 0.5, est=0.5)
    r["stock"] = arm(1.0, pulled_tokens=1000, transfer={"bytes": 100000})  # 100 bytes per token
    r["peer_stage"].update(kept_tokens=800, transfer={"bytes": 80000})
    assert checked([r])["pulled"]["ok"] is True
    r["peer_stage"]["transfer"] = {"bytes": 40000}
    assert checked([r])["pulled"] == {"ok": False, "records": [0]}
    assert checked([r], leak_free=False)["no_leaks"]["ok"] is False
    assert checked([r], no_draining_left={"ok": False, "first_attempt_ok": False})["no_draining_left"]["ok"] is False


# Discovery -------------------------------------------------------------------------------


def info(host, cpu_stage=True):
    return {"side_channel_host": host, "cpu_stage": cpu_stage, "engine_id": "a", "side_channel_port": 5600}


def test_discovery_demands_a_nodes_ip_not_localhost():
    h6.check_discovery(info("10.0.0.2"), "http://10.0.0.2:8100", None)
    with pytest.raises(RuntimeError, match="VLLM_NIXL_SIDE_CHANNEL_HOST"):
        h6.check_discovery(info("localhost"), "http://10.0.0.2:8100", None)
    with pytest.raises(RuntimeError, match="expected 10.0.0.2"):
        h6.check_discovery(info("10.0.0.9"), "http://10.0.0.2:8100", None)
    with pytest.raises(RuntimeError, match="cpu_stage"):
        h6.check_discovery(info("10.0.0.2", cpu_stage=False), "http://10.0.0.2:8100", None)


# Whole probe against fakes ---------------------------------------------------------------


def test_sweep_end_to_end(fakes, tmp_path):
    producer, consumer = fakes

    code = h6.main(args(fakes, tmp_path, levels="0,2", extra=["--cases", "gpu,cpu,cpu_tail,miss"]))

    result = results(tmp_path)
    assert code == 0, result["checks"]
    records = [r for r in result["records"]]
    assert {r["level"] for r in records} == {0, 2}
    by_case = {r["case"]: r for r in records if r["level"] == 0}
    # least-churn cases first, so a copy or a load does not evict states still waiting
    assert [r["case"] for r in records if r["level"] == 0] == ["gpu", "cpu_tail", "cpu", "miss"]
    # A's log, not the intent, decides: cpu really was CPU-only and was staged
    assert by_case["cpu"]["location_actual"] == "cpu" and by_case["cpu"]["cpu_tokens"] > 0
    assert by_case["cpu"]["peer_stage"]["outcome"] == "staged"
    assert by_case["cpu"]["peer_stage"]["kept_tokens"] == by_case["cpu"]["expected_hit_tokens"]
    assert by_case["cpu"]["peer"]["kept_tokens"] == 0 and by_case["cpu"]["peer"]["skip_reason"] == "reply_budget"
    assert by_case["cpu_tail"]["location_actual"] == "cpu_tail"
    assert by_case["gpu"]["location_actual"] == "gpu" and by_case["miss"]["location_actual"] == "miss"
    assert by_case["cpu"]["peer_stage"]["stage_s"] == pytest.approx(0.3)
    assert by_case["cpu"]["peer_stage"]["fence_wait_s"] == pytest.approx(0.1, abs=0.05)
    assert by_case["cpu"]["peer_stage"]["reply_upper_s"] is not None
    # the peer arm asked B not to stage; the peer_stage arm did not
    asked = [(r["kv_transfer_params"].get("peer_stage", True)) for r in consumer.requests
             if (r.get("kv_transfer_params") or {}).get("peer_lookup")]
    assert asked.count(False) == asked.count(True)
    assert {k for k in result["checks"] if isinstance(result["checks"][k], dict)} >= {
        "served", "lookup", "pulled", "equivalent", "no_leaks", "rdma_path",
        "staged_equivalent", "peer_arm_unstaged", "no_draining_left"}  # fmt: skip
    assert all(v["ok"] is not False for v in result["checks"].values() if isinstance(v, dict))
    # records.jsonl was written as the run went, and the tables are in results.json
    lines = (tmp_path / "out" / "records.jsonl").read_text().splitlines()
    assert len(lines) == len(records)
    assert result["tables"]["ttft"] and result["tables"]["reply"]
    # GPU resets: one per level for the CPU states, and none while the load runs
    assert producer.resets >= 2 and not producer.load_file.exists()


def test_a_staged_peer_arm_fails_the_run(fakes, tmp_path):
    producer, consumer = fakes
    consumer.ignore_opt_out = True

    assert h6.main(args(fakes, tmp_path, levels="0", extra=["--cases", "cpu"])) == 1
    assert results(tmp_path)["checks"]["peer_arm_unstaged"]["ok"] is False


def test_cpu_state_lost_from_the_cpu_tier_is_drift_and_a_sweep_without_staging_fails(
        fakes, tmp_path, monkeypatch):  # fmt: skip
    producer, _ = fakes
    real = h6.prepare_level

    def prepare_then_lose_cpu(*a, **k):
        states = real(*a, **k)
        producer.cpu.clear()  # A's CPU tier evicted the prepared states
        return states

    monkeypatch.setattr(h6, "prepare_level", prepare_then_lose_cpu)

    code = h6.main(args(fakes, tmp_path, levels="0", extra=["--cases", "cpu"]))

    result = results(tmp_path)
    rec = result["records"][0]
    assert rec["location_actual"] == "miss" and rec["cpu_tokens"] == 0 and rec["drift"] is True
    assert rec["peer_stage"]["outcome"] == "miss"
    assert result["level_info"]["0"]["drift"]["cpu"]["drifted"] == 1
    assert result["checks"]["staged_equivalent"]["ok"] is False  # CPU cases, nothing staged
    assert code == 1


def test_a_reply_budget_skip_is_not_a_cpu_case():
    skip = {"stage_submitted": False, "staged_blocks": 0, "skip_reason": "reply_budget"}
    assert h6.actual_cpu_tokens(skip, 0, 640, 16) == (0, "log")
    rule = {**skip, "skip_reason": "estimate"}  # A's own rule: it saw a CPU hit
    assert h6.actual_cpu_tokens(rule, 160, 640, 16) == (480, "inferred")
    staged = {"stage_submitted": True, "staged_blocks": 30, "skip_reason": None}
    assert h6.actual_cpu_tokens(staged, 160, 640, 16) == (480, "log")


def test_a_failing_batch_becomes_an_error_record_and_the_run_goes_on(fakes, tmp_path, monkeypatch):
    real = h6.prepare_level
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("A would not reset its GPU prefix cache")
        return real(*a, **k)

    monkeypatch.setattr(h6, "prepare_level", flaky)
    extra = ["--cases", "gpu", "--sessions", "2", "--batch-sessions", "1"]

    code = h6.main(args(fakes, tmp_path, levels="0", extra=extra))

    records = results(tmp_path)["records"]
    assert code == 1 and len(calls) == 2
    assert "batch failed" in records[0]["error"] and "error" not in records[1]


def test_results_are_written_after_every_level(fakes, tmp_path, monkeypatch):
    seen = []
    real = h6.write_results

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(results(tmp_path)["levels_done"])
        return out

    monkeypatch.setattr(h6, "write_results", spy)

    h6.main(args(fakes, tmp_path, levels="0,2", extra=["--cases", "gpu"]))

    assert seen == [[0], [0, 2]]


def test_a_block_size_other_than_the_probes_fails_at_startup(fakes, tmp_path):
    producer, _ = fakes
    producer.block_size = 64

    assert h6.main(args(fakes, tmp_path, levels="0")) == 1
    assert not (tmp_path / "out" / "records.jsonl").exists()
    with pytest.raises(RuntimeError, match="block_size=64.*--block-size 4"):
        h6.check_block_size(producer.url, BLOCK)


@pytest.mark.parametrize("flag", ["--score-rate", "--deadline-s"])
def test_an_empty_rate_or_deadline_fails_at_argument_parsing(flag, capsys):
    base = ["--model", "m", "--producer", "http://a", "--consumer", "http://b",
            "--producer-log", "a", "--consumer-log", "b", "--part", "deadline"]  # fmt: skip
    with pytest.raises(SystemExit):
        h6.parse_args([*base, flag, ""])
    assert "environment variable" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # the deadline part needs its label
        h6.parse_args(base)


def test_tcp_read_table_stops_the_run_at_startup(fakes, tmp_path):
    _, consumer = fakes
    consumer.log.write_text(TCP)

    assert h6.main(args(fakes, tmp_path, levels="0")) == 1
    gate = results(tmp_path)["checks"]["rdma_path"]
    assert gate["ok"] is False and "tcp" in gate["transports"]
    consumer.log.write_text(H5_CUDA_IPC)  # NVLink inside one host is no fabric either
    assert h6.main(args(fakes, tmp_path, levels="0")) == 1


def test_localhost_side_channel_stops_the_run(fakes, tmp_path):
    producer, _ = fakes
    producer.event_sources["0"]["peer_lookup"]["side_channel_host"] = "localhost"

    # the node IP of A, as the yaml passes it
    assert h6.main(args(fakes, tmp_path, levels="0", extra=["--expect-a-host", "10.1.1.1"])) == 1
    assert not (tmp_path / "out" / "records.jsonl").exists()


IMPACT_AIPERF = FAKE_AIPERF.replace(
    '(artifacts / "summary.json").write_text("{}")',
    '(artifacts / "profile_export.jsonl").write_text(json.dumps({"metadata": {"request_start_ns": time.time_ns() - 300_000_000},'
    ' "metrics": {"time_to_first_token": {"value": 250, "unit": "ms"},'
    ' "inter_token_latency": {"value": 20, "unit": "ms"}}}) + "\\n")')


def test_impact_part_runs_two_windows_per_level(fakes, tmp_path, monkeypatch):
    producer, consumer = fakes
    argv = args(fakes, tmp_path, levels="2",
                extra=["--part", "impact", "--impact-seconds", "0.5", "--impact-interval", "0.05"])
    (tmp_path / "aiperf.py").write_text(IMPACT_AIPERF)
    resets = []
    real = h6.reset_gpu
    monkeypatch.setattr(h6, "reset_gpu", lambda url, **k: resets.append((url, k.get("external"))) or real(url, **k))

    code = h6.main(argv)

    result = results(tmp_path)
    assert code == 0, result["checks"]
    assert [w["staging"] for w in result["tables"]["impact"]] == [False, True]
    off, on = result["tables"]["impact"]
    assert off["lookups"] > 0 and on["lookups"] > 0
    # B is reset before each lookup, its CPU tier included; A only ever GPU-only
    assert (consumer.url, True) in resets
    assert all(not external for url, external in resets if url == producer.url)
    assert on["a_ttft_p99"] == pytest.approx(0.25) and on["a_itl_p50"] == pytest.approx(0.02)
    # window 1 opted out of staging, window 2 did not
    flags = [r["kv_transfer_params"].get("peer_stage", True) for r in consumer.requests
             if (r.get("kv_transfer_params") or {}).get("peer_lookup")]
    assert flags[0] is False and flags[-1] is True
    assert any(v.startswith("impact on A") for v in result["verdicts"])
