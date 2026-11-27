"""H6: does staging A's CPU-only blocks pay off under load, over RDMA?

Run as the bench of examples/kv-transfer/h6.yaml, on node 1 (B). A runs on node 0
and B here, one GPU each, so NIXL reads go over the fabric. Both run
MultiConnector(NixlConnector, OffloadingConnector) with peer_lookup on; A also
runs peer_stage. Design: handoff/peer-pin-cpu-stage-benchmark.md.

For each load level the probe builds the case states on A while A is idle, starts
AIPerf against A, waits until A runs about min(N, 16) requests, and measures the
cases while the load keeps running. Cases, by what A holds of the prompt:

  gpu       whole prefix on GPU           partial  GPU holds the previous turn
  cpu       whole prefix on CPU only      miss     nowhere
  cpu_tail  GPU head, CPU-only tail

Arms, each under its own cache_salt:

  recompute   B computes the prompt          stock       A prefills, B pulls (H5)
  peer        lookup, staging opted out      peer_stage  lookup, staging allowed
  on_a        the request goes to A itself

Each record is classified by what A really did (A's log lines for the request's
pin), not by the case it was meant to be: background load can evict a prepared GPU
head. Parts: sweep (all levels, cases, arms), skip (the skip rule is on in A),
deadline (one run per peer_stage_deadline, then --merge) and impact (what staging
costs A's own requests).

Writes records.jsonl as it goes and results.json (tables, verdicts, checks) beside
--output. Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_peer_load as h5
from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_offload_latency import (
    EXPECTED_QWEN3_8B_BYTES_PER_TOKEN,
    MIN_STORE_FRACTION,
    Client,
    counter,
    nudge,
    scrape,
)
from modelsight.probes.kv_peer_stage import (
    RE_B_HIT,
    RE_B_MISS,
    RE_DONE,
    RE_PINNED,
    RE_SKIPPED,
    RE_SUBMITTED,
    RE_TRIMMED,
)
from modelsight.probes.kv_transfer import Servers
from modelsight.probes.rdma_gate import classify_transports

# Fields and sources that only a real run can confirm. Each is the best available
# source today; the run's records show which ones came back empty.
NEEDS_REAL_RUN = [
    "reply_upper_s: vLLM's B logs no lookup-submit line, so it is the B log "
    "timestamp of the hit/miss line minus the request's wall-clock start (an upper "
    "bound that includes B's tokenizing and scheduling); reply_s stays None.",
    "est_stage_s: A logs its estimate only when it skips. Otherwise it is modelled "
    "as 2 * a_step_ms + cpu bytes / 20 GB/s (the rule's own formula).",
    "fence_wait_s: A logs no ship line. 'Peer lookup_pin pinned' is logged at ship, "
    "so it is that line's time minus the later of the pin (submitted/skipped line) "
    "and the stage's done line.",
    "a_step_ms: A logs nothing per step at INFO. It is the mean step gap over the 10 s "
    "before the record, from the iteration counter vllm:iteration_tokens_total "
    "(_count); None when A is idle or the series is missing.",
    "peer_held_peak: no metric of pinned plus draining blocks is known. The sampler "
    "sums series whose name matches peer.*(pinned|draining); null when none exists.",
    "Metric names: vllm:kv_cache_usage_perc (free blocks), vllm:num_preemptions_total, "
    "vllm:kv_offload_store_bytes, and POST /reset_prefix_cache?reset_external=false.",
    "AIPerf export: profile_export.jsonl rows with metadata.request_start_ns and "
    "metrics.time_to_first_token / inter_token_latency {value, unit}.",
    "pull_s is NIXL's xfer time, a lower bound (the clock stops at B's poll); "
    "b_compute_s = ttft - reply_upper_s - pull_s is therefore an upper bound.",
    "Prepared GPU states can outgrow A's pool, and the CPU tier must hold a batch's "
    "states: --batch-sessions (default 2) bounds both, at the cost of one AIPerf "
    "start per batch.",
    "location_actual comes from the peer arm's log (the peer_stage arm's when peer "
    "is not run); the arms' own states can drift differently.",
    "/kv_event_sources is read as {rank: {peer_lookup: {side_channel_host, ...}}}.",
]

PARTS = ("sweep", "skip", "deadline", "impact")
CASES = ("gpu", "partial", "cpu", "cpu_tail", "miss")
ARMS = ("recompute", "stock", "peer", "peer_stage", "on_a")
STATEFUL = ("stock", "peer", "peer_stage", "on_a")  # arms that need a prepared state
CPU_CASES = ("cpu", "cpu_tail")
PART_DEFAULTS = {
    "sweep": {"levels": h5.DEFAULT_LEVELS, "cases": ",".join(CASES),
              "arms": ",".join(ARMS)},
    "skip": {"levels": "32", "cases": "cpu,cpu_tail",
             "arms": "recompute,peer,peer_stage"},
    "deadline": {"levels": "64", "cases": "cpu", "arms": "peer,peer_stage"},
    "impact": {"levels": "16,64", "cases": "cpu", "arms": "peer,peer_stage"},
}  # fmt: skip
SIZE_EDGES = (4000, 8000, 16000, 32000)
STAGE_RATE_BYTES_PER_S = 20e9  # the pin server's copy bandwidth in its estimate
# Share of A's GPU pool, and of its CPU tier, that one batch's prepared states may fill.
# The measurements themselves allocate on A (a staged copy, a load into GPU), and the
# AIPerf load stores to the CPU tier, so the rest is headroom against LRU eviction.
GPU_STATE_FRACTION = 0.5
CPU_STATE_FRACTION = 0.5
MEASURE_ORDER = ("gpu", "partial", "cpu_tail", "cpu", "miss")  # least churn on A first
DEFAULT_DEADLINE_S = 1.5
STAGE_SETTLE_SECONDS = 10.0
STORE_STABLE_SECONDS = 2.0
STORE_TIMEOUT_SECONDS = 300.0
STEP_WINDOW_SECONDS = 10.0
IDLE_GAP_S = 2.0  # vLLM peer_pins._IDLE_GAP_S: a longer step gap means A idled
SAMPLE_INTERVAL_SECONDS = 0.25
END_RESET_ATTEMPTS = 60
LOG_TIMEOUT_SECONDS = 20.0
STEP_METRICS = ("vllm:iteration_tokens_total_count", "vllm:iteration_tokens_total")
HELD_METRIC = re.compile(r"peer.*(pinned|draining)", re.IGNORECASE)
LOG_TIME = re.compile(r"\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\b")
LOOPBACK = ("localhost", "127.", "::1", "0.0.0.0")


# Logs ------------------------------------------------------------------------


class LogSource:
    """A server log: a file, or a rank's control endpoint (http://host:8399/logs/<name>).

    Offsets are byte positions. `end()` is where the log ends now, and `read_from`
    returns the text after an offset and the offset after it, so a caller can mark
    a point, act, and read what the server logged since.
    """

    def __init__(self, target: str | Path) -> None:
        self.target = str(target)
        self.remote = self.target.startswith(("http://", "https://"))
        self._end = 0

    @property
    def name(self) -> str:
        path = urlsplit(self.target).path if self.remote else self.target
        return Path(path).name

    def read_bytes(self, offset: int = 0) -> bytes:
        if not self.remote:
            path = Path(self.target)
            if not path.exists():
                return b""
            with path.open("rb") as f:
                f.seek(offset)
                return f.read()
        try:
            with urllib.request.urlopen(f"{self.target}?offset={offset}",
                                        timeout=LOG_TIMEOUT_SECONDS) as response:  # fmt: skip
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # the server has not created the log yet
                return b""
            raise

    def read_from(self, offset: int) -> tuple[str, int]:
        data = self.read_bytes(offset)
        return data.decode(errors="replace"), offset + len(data)

    def end(self) -> int:
        if not self.remote:
            path = Path(self.target)
            self._end = path.stat().st_size if path.exists() else 0
        else:  # the endpoint reports no size: read on from the last known end
            self._end += len(self.read_bytes(self._end))
        return self._end

    def text(self) -> str:
        return self.read_from(0)[0]


def line_time(line: str) -> float | None:
    """Epoch seconds of a vLLM log line's `MM-DD HH:MM:SS.mmm` stamp (local time)."""
    match = LOG_TIME.search(line)
    if not match:
        return None
    month, day, hour, minute, second, milli = (int(g) for g in match.groups())
    try:
        stamp = datetime(time.localtime().tm_year, month, day, hour, minute, second,
                         milli * 1000)  # fmt: skip
    except ValueError:
        return None
    return stamp.timestamp()


def _event(kind: str, line: str, **fields) -> dict:
    return {"kind": kind, "ts": line_time(line), **fields}


def parse_events(text: str) -> list[dict]:
    """Every peer-stage and peer-lookup line of A's or B's log, in order."""
    events = []
    for line in text.splitlines():
        if m := RE_SUBMITTED.search(line):
            events.append(_event("submitted", line, pin=m[1], staged=int(m[2]),
                                 in_place=int(m[3]), deadline_s=float(m[4])))  # fmt: skip
        elif m := RE_DONE.search(line):
            events.append(_event("done", line, pin=m[1], ok=m[2] == "True",
                                 seconds=float(m[3])))  # fmt: skip
        elif m := RE_TRIMMED.search(line):
            events.append(_event("trimmed", line, pin=m[1], reason=m[2],
                                 kept_blocks=int(m[3])))  # fmt: skip
        elif m := RE_SKIPPED.search(line):
            events.append(_event("skipped", line, pin=m[1], reason=m[2],
                                 est_s=float(m[3])))  # fmt: skip
        elif m := RE_PINNED.search(line):
            events.append(_event("pinned", line, blocks=int(m[1]), of=int(m[2]), pin=m[3]))
        elif m := RE_B_HIT.search(line):
            events.append(_event("hit", line, request_id=m[1], tokens=int(m[2]), pin=m[3]))
        elif m := RE_B_MISS.search(line):
            events.append(_event("miss", line, request_id=m[1], reason=m[2]))
    return events


def arm_events(a_text: str, b_text: str, request_id: str) -> dict:
    """What A and B logged for one lookup, joined by its pin.

    B's `hit` line carries the pin. With no hit, the pin is the one A staged,
    trimmed or skipped in this window (B sends one lookup at a time).
    """
    b_events = [e for e in parse_events(b_text) if e["kind"] in ("hit", "miss")]
    mine = [e for e in b_events if request_id in e["request_id"]] or b_events
    reply = mine[-1] if mine else None
    a_events = parse_events(a_text)
    pins = {reply["pin"]} if reply and reply["kind"] == "hit" else set()
    if not pins:
        pins = {e["pin"] for e in a_events if e["kind"] in ("submitted", "trimmed", "skipped")}

    def first(kind: str) -> dict | None:
        return next((e for e in a_events if e["kind"] == kind and e["pin"] in pins), None)

    return {
        "pins": sorted(pins),
        "hit_tokens": reply["tokens"] if reply and reply["kind"] == "hit" else None,
        "miss_reason": reply["reason"] if reply and reply["kind"] == "miss" else None,
        "reply_seen": reply is not None,
        "b_reply_ts": reply["ts"] if reply else None,
        "submitted": first("submitted"), "done": first("done"),
        "trimmed": first("trimmed"), "skipped": first("skipped"),
        "pinned": first("pinned"),
    }  # fmt: skip


def stage_outcome(ev: dict) -> str:
    """staged, trimmed_*, skipped, unfinished, no_cpu or miss, from A's lines."""
    if ev["skipped"]:
        return "skipped"
    if ev["trimmed"]:
        reason = ev["trimmed"]["reason"]
        return {"deadline": "trimmed_deadline", "failed": "trimmed_failed"}.get(
            reason, f"trimmed_{reason}")  # fmt: skip
    if ev["submitted"]:
        if ev["done"] is None:
            return "unfinished"
        return "staged" if ev["done"]["ok"] else "trimmed_failed"
    return "no_cpu" if (ev["hit_tokens"] or ev["pinned"]) else "miss"


def stage_settled(a_text: str) -> bool:
    """Whether every copy A submitted in this text has reported or been trimmed."""
    events = parse_events(a_text)
    closed = {e["pin"] for e in events if e["kind"] in ("done", "trimmed")}
    return all(e["pin"] in closed for e in events if e["kind"] == "submitted")


def split_tokens(ev: dict, outcome: str, block_size: int) -> tuple[int, int]:
    """(kept tokens in the reply, of which GPU) for one lookup, from A's lines.

    A's `pinned` line is logged at ship, after any staging or trim, so its block
    count is the reply. B's hit line is the fallback.
    """
    if ev["pinned"]:
        kept = ev["pinned"]["blocks"] * block_size
    else:
        kept = ev["hit_tokens"] or 0
    if outcome == "staged" and ev["submitted"]:
        gpu = max(0, kept - ev["submitted"]["staged"] * block_size)
    elif ev["trimmed"]:
        gpu = ev["trimmed"]["kept_blocks"] * block_size
    else:
        gpu = kept
    return kept, gpu


def actual_cpu_tokens(ps: dict | None, gpu_tokens: int, expected_hit: int,
                      block_size: int) -> tuple[int, str]:  # fmt: skip
    """The tokens A held only on CPU, from the peer_stage arm's `Peer stage submitted
    staged=N` line. No line means nothing was staged: a request that found nothing
    on CPU, or a reply_budget skip, is not a CPU case. Only a skip by A's own rule
    (reason other than reply_budget) shows a CPU hit without a submit line, and then
    the CPU part is what A's tiers hold beyond the GPU part. Without a peer_stage
    arm the split is inferred the same way."""
    if ps is None:
        return max(0, expected_hit - gpu_tokens), "inferred"
    if ps["stage_submitted"]:
        return ps["staged_blocks"] * block_size, "log"
    if ps["skip_reason"] not in (None, "reply_budget"):
        return max(0, expected_hit - gpu_tokens), "inferred"
    return 0, "log"


def reply_tokens(ev: dict, pinned: int) -> int:
    """Tokens B was told it could read: its hit line, 0 on a miss (B may have given
    up after A shipped). With no B line at all, A's pinned count."""
    return (ev["hit_tokens"] or 0) if ev["reply_seen"] else pinned


def classify_location(gpu_tokens: int, cpu_tokens: int, total: int, full: int) -> str:
    """Where A held the prefix: miss, cpu, cpu_tail, partial (GPU, short of the
    prompt) or gpu. `total` is what A's tiers hold, `full` the whole prompt's."""
    if total <= 0:
        return "miss"
    if cpu_tokens > 0:
        return "cpu_tail" if gpu_tokens > 0 else "cpu"
    return "gpu" if gpu_tokens >= full else "partial"


# A: cache states ---------------------------------------------------------------


def reset_gpu(url: str, attempts: int = h4.RESET_ATTEMPTS, external: bool = False) -> bool:
    """Prefix cache reset. By default GPU only (the CPU tier survives); `external`
    also clears the connectors' CPU tier. vLLM refuses while any request or pin
    holds a block, so retry for transfers that are finishing."""
    flag = "true" if external else "false"
    request = urllib.request.Request(
        f"{url}/reset_prefix_cache?reset_external={flag}", data=b"", method="POST")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if json.load(response).get("success", True):
                    return True
        except (OSError, ValueError):
            pass
        if attempt + 1 < attempts:
            time.sleep(1)
    return False


class AEngine:
    """The calls that build a case state on A. Tests swap in a recording mock."""

    def __init__(self, servers: Servers, rng: random.Random,
                 stable_s: float = STORE_STABLE_SECONDS,
                 timeout: float = STORE_TIMEOUT_SECONDS) -> None:  # fmt: skip
        self.servers = servers
        self.url = servers.producer
        self.rng = rng
        self.stable_s = stable_s
        self.timeout = timeout
        self.client = Client(self.url, servers.model)

    def warm(self, prompt: list[int], salt: str) -> None:
        """A computes `prompt` under `salt` and caches it (one output token)."""
        body = {"prompt": prompt, "cache_salt": salt, "max_tokens": 1, "stream": False}
        with kv.post(self.servers, self.url, body) as response:
            response.read()

    def store_bytes(self) -> float:
        return counter(scrape(self.url), "vllm:kv_offload_store_bytes")

    def wait_stored(self, before: float, expected: float) -> None:
        """Until the offload store has grown by about `expected` bytes since
        `before` and then stood still for `stable_s`."""
        deadline = time.monotonic() + self.timeout
        last, changed = before, time.monotonic()
        while time.monotonic() < deadline:
            now = self.store_bytes()
            if now != last:
                last, changed = now, time.monotonic()
            grown = now - before
            enough = grown >= MIN_STORE_FRACTION * expected if expected else grown > 0
            if enough and time.monotonic() - changed >= self.stable_s:
                return
            if not enough and time.monotonic() - changed > 1.0:
                nudge(self.client, self.rng)  # an idle engine reports no store
                changed = time.monotonic()
            time.sleep(0.1)
        raise RuntimeError("A's CPU store did not settle")

    def reset_gpu(self) -> None:
        if not reset_gpu(self.url, attempts=20):
            raise RuntimeError("A would not reset its GPU prefix cache")


def prepare_level(a, pairs: list[tuple[int, list[int], list[int]]], cases, arms,
                  level: int, nonce: str, block_size: int,
                  bytes_per_token: float) -> dict[tuple, str]:  # fmt: skip
    """Build every case's state on A, while A is idle; return the salt of each.

    The GPU reset is all-or-nothing, so the order is: warm every CPU state, wait
    for the offload store, reset the GPU once, then warm the GPU states. A salt is
    per (session, case, arm), so arms never share cache state.
    """
    wanted = [arm for arm in STATEFUL if arm in arms]
    states = {
        (n, case, arm): f"{nonce}-{level}-{uuid.uuid4().hex[:8]}-{arm}"
        for n, _, _ in pairs for case in cases if case != "miss" for arm in wanted
    }  # fmt: skip
    cpu = [c for c in cases if c in CPU_CASES]
    if cpu and wanted:
        before = a.store_bytes()
        expected = 0.0
        for n, _, current in pairs:
            for case in cpu:
                for arm in wanted:
                    a.warm(current, states[(n, case, arm)])
                    expected += len(current) // block_size * block_size * bytes_per_token
        a.wait_stored(before, expected)
        a.reset_gpu()
    for n, previous, current in pairs:
        for case in cases:
            for arm in wanted:
                salt = states.get((n, case, arm))
                if case == "gpu":
                    a.warm(current, salt)
                elif case == "partial" and previous:
                    a.warm(previous, salt)
                elif case == "cpu_tail":
                    a.warm(current[: len(current) // 2 // block_size * block_size], salt)
    return states


# A: metrics sampled during a level ----------------------------------------------


class MetricsSampler(h5.Sampler):
    """H5's running/waiting sampler, plus the series H6 needs from A: its step
    rate, free blocks, preemptions and pinned-plus-draining blocks."""

    def __init__(self, url: str, interval: float) -> None:
        super().__init__(url, interval)
        self.rows: list[dict] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                metrics = scrape(self.url)
                now = time.monotonic()
                steps = next((metrics[n] for n in STEP_METRICS if n in metrics), None)
                held = [v for k, v in metrics.items() if HELD_METRIC.search(k)]
                usage = metrics.get("vllm:kv_cache_usage_perc")
                blocks = metrics.get("num_gpu_blocks")
                self.samples.append((now, metrics.get("vllm:num_requests_running", 0.0),
                                     metrics.get("vllm:num_requests_waiting", 0.0)))  # fmt: skip
                self.rows.append({
                    "t": now, "steps": steps, "running": metrics.get("vllm:num_requests_running", 0.0), "held": sum(held) if held else None,
                    "preemptions": counter(metrics, "vllm:num_preemptions"),
                    "free": (1 - usage) * blocks if usage is not None and blocks else None,
                })  # fmt: skip
            except OSError:
                pass
            self._stop.wait(self.interval)

    def _span(self, start: float, end: float) -> list[dict]:
        return [r for r in list(self.rows) if start <= r["t"] <= end]

    def step_ms(self, end: float, span: float = STEP_WINDOW_SECONDS) -> float | None:
        """A's engine step interval over the last `span` seconds, as the pin server's
        own estimate sees it: gaps between consecutive steps, only while A has
        running requests and only gaps within IDLE_GAP_S (a longer one means A
        idled; vLLM drops its average then). Median of the per-sample gaps. An idle
        A returns 0.0, so the estimate is the copy alone; None means no step
        counter is exposed at all."""
        rows = [r for r in self._span(end - span, end) if r["steps"] is not None]
        if not rows:
            return None
        gaps = []
        for before, after in zip(rows, rows[1:]):
            steps = after["steps"] - before["steps"]
            busy = (before["running"] or 0) > 0 or (after["running"] or 0) > 0
            if steps > 0 and busy:
                gap = (after["t"] - before["t"]) / steps
                if gap <= IDLE_GAP_S:
                    gaps.append(gap)
        return statistics.median(gaps) * 1000 if gaps else 0.0

    def free_blocks_min(self, start: float, end: float) -> float | None:
        free = [r["free"] for r in self._span(start, end) if r["free"] is not None]
        return min(free) if free else None

    def preemptions(self, start: float, end: float) -> float | None:
        rows = self._span(start, end)
        return rows[-1]["preemptions"] - rows[0]["preemptions"] if len(rows) > 1 else None

    def held_peak(self, start: float, end: float) -> float | None:
        held = [r["held"] for r in self._span(start, end) if r["held"] is not None]
        return max(held) if held else None


# One case under load -----------------------------------------------------------


@dataclass
class Ctx:
    args: argparse.Namespace
    servers: Servers
    producer: dict  # kv_transfer_params fields that address A
    a_log: LogSource
    b_log: LogSource
    nonce: str
    sampler: MetricsSampler | None = None
    records_file: object = None

    def window(self, started: float) -> dict | None:
        return self.sampler.window(started, time.monotonic()) if self.sampler else None

    def salt(self, level: int, arm: str) -> str:
        return f"{self.nonce}-{level}-{uuid.uuid4().hex[:8]}-{arm}"

    def read_window(self, a_off: int, b_off: int, stage: bool) -> tuple[str, str]:
        """A's and B's log text since the marks, once late lines have landed. With
        staging, until every copy A submitted has reported (or the wait ends)."""
        time.sleep(kv.METRICS_WAIT_SECONDS)
        deadline = time.monotonic() + STAGE_SETTLE_SECONDS
        while True:
            a_text = self.a_log.read_from(a_off)[0]
            b_text = self.b_log.read_from(b_off)[0]
            if not stage or stage_settled(a_text) or time.monotonic() >= deadline:
                return a_text, b_text
            time.sleep(0.25)


def peer_arm(ctx: Ctx, level: int, salt: str, current: list[int], stage: bool) -> dict:
    """One lookup request to B, and what A and B logged for it."""
    args = ctx.args
    request_id = f"h6-{ctx.nonce}-{level}-{uuid.uuid4().hex[:8]}"
    a_off, b_off = ctx.a_log.end(), ctx.b_log.end()
    before = kv.metric_totals(ctx.servers.consumer, kv.NIXL_METRICS)
    params = h4.peer_params(ctx.producer, args.identity)
    if not stage:
        params["peer_stage"] = False  # B sends reply_budget_s=0, A skips staging
    wall, started = time.time(), time.monotonic()
    arm = h4.greedy(ctx.servers, ctx.servers.consumer,
                    {"prompt": current, "cache_salt": salt, "kv_transfer_params": params},
                    headers={"X-Request-Id": request_id})  # fmt: skip
    arm["a_load"] = ctx.window(started)
    arm["transfer"] = kv.metrics_change(ctx.servers.consumer, before)
    a_text, b_text = ctx.read_window(a_off, b_off, stage)
    ev = arm_events(a_text, b_text, request_id)
    outcome = stage_outcome(ev)
    kept, gpu = split_tokens(ev, outcome, args.block_size)
    arm["request_id"] = request_id
    # A computed nothing for this arm if no request with its id ever ran there.
    arm["a_served"] = a_text.count(request_id)
    arm["lookups_logged"] = len(re.findall(args.lookup_pattern, a_text, re.IGNORECASE))
    arm["pin"] = ev["pins"][0] if ev["pins"] else None
    arm["hit_tokens"] = ev["hit_tokens"]
    arm["pinned_tokens"] = kept  # what A pinned and shipped, by its own log
    # What B was told: its hit line, 0 on a miss (B may have timed out after A
    # shipped, and then pulled nothing). With no B line at all, A's count.
    arm["kept_tokens"] = reply_tokens(ev, kept)
    arm["gpu_tokens"] = gpu
    arm["stage_submitted"] = ev["submitted"] is not None
    arm["staged_blocks"] = ev["submitted"]["staged"] if ev["submitted"] else 0
    arm["deadline_s"] = ev["submitted"]["deadline_s"] if ev["submitted"] else None
    arm["stage_s"] = ev["done"]["seconds"] if ev["done"] else None
    arm["skip_reason"] = ev["skipped"]["reason"] if ev["skipped"] else None
    arm["skip_est_s"] = ev["skipped"]["est_s"] if ev["skipped"] else None
    arm["reply_s"] = None  # B logs no submit line; see NEEDS_REAL_RUN
    reply = ev["b_reply_ts"]
    arm["reply_upper_s"] = max(0.0, reply - wall) if reply is not None else None
    pin_ts = next((e["ts"] for e in (ev["submitted"], ev["skipped"]) if e), None)
    ready = [t for t in (pin_ts, ev["done"]["ts"] if ev["done"] else None) if t is not None]
    ship = ev["pinned"]["ts"] if ev["pinned"] else None
    arm["fence_wait_s"] = max(0.0, ship - max(ready)) if ship and ready else None
    arm["pull_s"] = arm["transfer"].get("xfer_s") or None
    arm["b_compute_s"] = max(0.0, arm["ttft_s"] - (arm["reply_upper_s"] or 0.0)
                             - (arm["pull_s"] or 0.0))  # fmt: skip
    if stage:
        arm["outcome"] = outcome
    return arm


def a_estimate(arm: dict, cpu_tokens: int, step_ms: float | None, bytes_per_token: float
               ) -> tuple[float | None, str]:  # fmt: skip
    """A's stage-time estimate: its log when it skipped, else the rule's formula."""
    if arm.get("skip_est_s") is not None:
        return arm["skip_est_s"], "log"
    if cpu_tokens <= 0:
        return None, "none"
    step_s = (step_ms or 0.0) / 1000
    return 2 * step_s + cpu_tokens * bytes_per_token / STAGE_RATE_BYTES_PER_S, "model"


def run_case(ctx: Ctx, level: int, n: int, case: str, previous: list[int],
             current: list[int], states: dict, arms: list[str]) -> dict:  # fmt: skip
    """All arms of one case, each under its own cache_salt, with the load on."""
    args = ctx.args
    held = {"gpu": current, "cpu": current, "cpu_tail": current,
            "partial": previous, "miss": []}[case]  # fmt: skip
    full = h4.expected_hit(current, current, args.block_size)
    record: dict = {
        "level": level, "case": case, "session": n,
        "prompt_tokens": len(current), "held_tokens": len(held),
        "expected_hit_tokens": h4.expected_hit(held, current, args.block_size),
        "full_hit_tokens": full,
    }  # fmt: skip
    step_ms = ctx.sampler.step_ms(time.monotonic()) if ctx.sampler else None
    record["a_step_ms"] = step_ms

    def state(arm: str) -> str:
        return states.get((n, case, arm)) or ctx.salt(level, arm)

    if "recompute" in arms:
        started = time.monotonic()
        record["recompute"] = h4.greedy(
            ctx.servers, ctx.servers.consumer,
            {"prompt": current, "cache_salt": ctx.salt(level, "rc")})  # fmt: skip
        record["recompute"]["a_load"] = ctx.window(started)
    if "stock" in arms and case != "miss":
        record["stock"] = stock_arm(ctx, state("stock"), current)
    for arm in ("peer", "peer_stage"):
        if arm in arms:
            record[arm] = peer_arm(ctx, level, state(arm), current, arm == "peer_stage")
    if "on_a" in arms and case != "miss":
        started = time.monotonic()
        record["on_a"] = h4.greedy(ctx.servers, ctx.servers.producer,
                                   {"prompt": current, "cache_salt": state("on_a")})  # fmt: skip
        record["on_a"]["a_load"] = ctx.window(started)

    source = record.get("peer") or record.get("peer_stage")
    if source is not None:
        gpu = source["gpu_tokens"]
        cpu, cpu_source = actual_cpu_tokens(record.get("peer_stage"), gpu,
                                            record["expected_hit_tokens"], args.block_size)  # fmt: skip
        record["gpu_tokens"], record["cpu_tokens"] = gpu, cpu
        record["cpu_source"] = cpu_source
        record["location_actual"] = classify_location(gpu, cpu, gpu + cpu, full)
        record["drift"] = record["location_actual"] != case
        for arm in ("peer", "peer_stage"):
            if arm in record:
                record[arm]["cpu_tokens"] = cpu
        if "peer_stage" in record:
            ps = record["peer_stage"]
            ps["est_stage_s"], ps["est_source"] = a_estimate(
                ps, ps["cpu_tokens"], step_ms, args.bytes_per_token)
    return record


def stock_arm(ctx: Ctx, salt: str, current: list[int]) -> dict:
    """A prefills (queued behind its load), then B pulls; timed from A's request."""
    started = time.monotonic()
    tick = time.perf_counter()
    params = kv.prefill(ctx.servers, current, cache_salt=salt)
    prefill_s = time.perf_counter() - tick
    before = kv.metric_totals(ctx.servers.consumer, kv.NIXL_METRICS)
    stock = h4.greedy(ctx.servers, ctx.servers.consumer,
                      {"prompt": current, "cache_salt": salt, "kv_transfer_params": params})  # fmt: skip
    stock["b_ttft_s"] = stock["ttft_s"]
    stock["prefill_s"] = prefill_s
    stock["ttft_s"] = prefill_s + stock["b_ttft_s"]
    stock["total_s"] += prefill_s
    stock["transfer"] = kv.metrics_change(ctx.servers.consumer, before)
    stock["pulled_tokens"] = h4.expected_hit(current, current, ctx.args.block_size)
    stock["pull_s"] = stock["transfer"].get("xfer_s") or None
    stock["b_compute_s"] = max(0.0, stock["b_ttft_s"] - (stock["pull_s"] or 0.0))
    stock["a_load"] = ctx.window(started)
    return stock


# Levels ---------------------------------------------------------------------


def state_tokens(previous: list[int], current: list[int], case: str, block_size: int,
                 arms: int) -> tuple[int, int]:  # fmt: skip
    """(GPU tokens, CPU tokens) one session's case state takes on A, over `arms`
    states. Every warm-up is stored to the CPU tier; the GPU keeps what is warmed
    after the reset (a cpu state keeps none there)."""
    full = len(current) // block_size * block_size
    prev = len(previous) // block_size * block_size
    gpu, cpu = {"gpu": (full, full), "partial": (prev, prev), "cpu": (0, full),
                "cpu_tail": (full // 2, full), "miss": (0, 0)}[case]  # fmt: skip
    return gpu * arms, cpu * arms


def plan_units(pairs: list, cases: list[str], arms: list[str], block_size: int,
               gpu_budget: float | None, cpu_budget: float | None,
               max_sessions: int = 0) -> list[tuple[list, list[str]]]:  # fmt: skip
    """Split a level's sessions and cases into units (sessions x cases) whose states
    fit A's GPU pool and CPU tier, so a unit's prepared states are still there when
    it is measured. A session whose cases do not fit together is split by case; a
    single (session, case) that does not fit still gets a unit of its own. No budget
    (None) means no limit; `max_sessions` caps the sessions of a unit."""
    n_arms = len([a for a in STATEFUL if a in arms])

    def foot(pair, case_list) -> tuple[float, float]:
        parts = [state_tokens(pair[1], pair[2], c, block_size, n_arms) for c in case_list]
        return sum(g for g, _ in parts), sum(c for _, c in parts)

    def fits(gpu: float, cpu: float) -> bool:
        return ((gpu_budget is None or gpu <= gpu_budget)
                and (cpu_budget is None or cpu <= cpu_budget))  # fmt: skip

    units: list[list] = []  # [pairs, cases, gpu, cpu]
    for pair in pairs:
        chunks: list[list[str]] = []
        for case in cases:
            if chunks and fits(*foot(pair, chunks[-1] + [case])):
                chunks[-1].append(case)
            else:
                chunks.append([case])
        for chunk in chunks:
            gpu, cpu = foot(pair, chunk)
            last = units[-1] if units else None
            if (last and last[1] == chunk and fits(last[2] + gpu, last[3] + cpu)
                    and (max_sessions <= 0 or len(last[0]) < max_sessions)):  # fmt: skip
                last[0].append(pair)
                last[2] += gpu
                last[3] += cpu
            else:
                units.append([[pair], chunk, gpu, cpu])
    return [(u[0], u[1]) for u in units]


def state_budgets(args: argparse.Namespace, url: str, level: int) -> tuple[float | None, float]:
    """Tokens of GPU pool and CPU tier a unit's states may use. The GPU budget is
    what A's pool holds beyond the AIPerf load's own blocks."""
    metrics = scrape(url)
    blocks, size = metrics.get("num_gpu_blocks"), metrics.get("block_size")
    gpu = None
    if blocks and size:
        gpu = max(0.0, blocks * size - (h5.LOAD_TOKENS if level > 0 else 0)) * GPU_STATE_FRACTION
    cpu = args.cpu_gb * 2**30 / args.bytes_per_token * CPU_STATE_FRACTION
    return gpu, cpu


def end_of_level(ctx: Ctx, info: dict) -> None:
    """A drains, then its GPU prefix cache must reset: no pin or staged block left."""
    servers = ctx.servers
    info["drained"] = h5.wait_idle(servers.producer)
    first = reset_gpu(servers.producer, attempts=1)
    info["no_draining_left"] = {
        "ok": first or reset_gpu(servers.producer, attempts=END_RESET_ATTEMPTS),
        "first_attempt_ok": first}  # fmt: skip
    info["leak_free"] = all(h4.reset_cache(url)
                            for url in (servers.producer, servers.consumer))  # fmt: skip


def emit(ctx: Ctx, record: dict) -> None:
    if ctx.records_file is not None:
        ctx.records_file.write(json.dumps(record) + "\n")
        ctx.records_file.flush()
    shown = ("level", "session", "case", "location_actual", "prompt_tokens",
             "expected_hit_tokens", "error")  # fmt: skip
    print(json.dumps({k: v for k, v in record.items() if k in shown}), flush=True)


def level_load(ctx: Ctx, level: int, dataset: Path | None, tag: str):
    if level == 0 or dataset is None:
        return None
    out_dir = ctx.args.output.parent
    command = h5.aiperf_command(ctx.args, level, dataset, out_dir / f"aiperf-{tag}")
    return h5.Load(command, out_dir / f"aiperf-{tag}.log")


def write_dataset(ctx: Ctx, level: int, background, render, info: dict) -> Path | None:
    if level == 0:
        return None
    dataset = ctx.args.output.parent / f"background-{level}.jsonl"
    info["background_sessions"] = h5.write_background(dataset, background, render, level)
    info["prompt_cap_tokens"] = h5.prompt_cap(level)
    if not info["background_sessions"]:
        raise RuntimeError(f"no background session fits {h5.prompt_cap(level)} tokens")
    if info["background_sessions"] < level:
        print(f"warning: level {level} has only {info['background_sessions']} "
              "background sessions; AIPerf will repeat them", flush=True)  # fmt: skip
    return dataset


def run_level(ctx: Ctx, level: int, measured, background, render, cases, arms) -> dict:
    """Measure every case under `level` background sessions on A.

    The sessions go in batches (--batch-sessions): a batch's states are built while
    A is idle, then AIPerf runs and the batch is measured. One batch is the plan;
    smaller ones keep the prepared GPU states inside A's pool.
    """
    args = ctx.args
    info: dict = {"level": level}
    dataset = write_dataset(ctx, level, background, render, info)
    pairs = [(n, prev, cur) for n, (prev, cur) in enumerate(measured)]
    a = AEngine(ctx.servers, random.Random(args.seed))
    gpu_budget, cpu_budget = state_budgets(args, ctx.servers.producer, level)
    units = plan_units(pairs, cases, arms, args.block_size, gpu_budget, cpu_budget,
                       args.batch_sessions)  # fmt: skip
    info["state_budget_tokens"] = {"gpu": gpu_budget, "cpu": cpu_budget}
    info["units"] = [{"sessions": [n for n, _, _ in u[0]], "cases": u[1]} for u in units]
    records: list[dict] = []
    settles = []
    started_all = time.monotonic()
    with MetricsSampler(ctx.servers.producer, SAMPLE_INTERVAL_SECONDS) as sampler:
        ctx.sampler = sampler
        for b, (batch, unit_cases) in enumerate(units):
            try:
                states = prepare_level(a, batch, unit_cases, arms, level, ctx.nonce,
                                       args.block_size, args.bytes_per_token)  # fmt: skip
                load = level_load(ctx, level, dataset, f"{level}-{b}")
                with (load or contextlib.nullcontext()) as active:
                    if load is not None:
                        settle = h5.wait_loaded(ctx.servers.producer, level, active)
                        settles.append(settle)
                        print(json.dumps({"level": level, "batch": b, **settle}), flush=True)
                    # Least churn first: a measurement allocates on A (a copy, a load)
                    # and would evict the states still waiting behind it.
                    order = sorted(((c, n, prev, cur) for n, prev, cur in batch for c in unit_cases),
                                   key=lambda x: (MEASURE_ORDER.index(x[0]), x[1]))  # fmt: skip
                    for case, n, previous, current in order:
                        try:
                            record = run_case(ctx, level, n, case, previous, current,
                                              states, arms)  # fmt: skip
                        except (OSError, RuntimeError, ValueError, KeyError) as exc:
                            record = {"level": level, "case": case, "session": n,
                                      "error": repr(exc)}  # fmt: skip
                        records.append(record)
                        emit(ctx, record)
            except (OSError, RuntimeError, ValueError) as exc:
                # A batch that cannot be prepared or loaded costs its records, not the run.
                record = {"level": level, "batch": b, "error": f"batch failed: {exc!r}"}
                records.append(record)
                emit(ctx, record)
            if level > 0 and not h5.wait_idle(ctx.servers.producer):
                # The next batch's GPU reset would fail on a busy A: record it.
                info.setdefault("idle_failed_after_batches", []).append(b)
                print(f"warning: A was not idle after batch {b} of level {level}", flush=True)
        info["peer_held_peak"] = sampler.held_peak(started_all, time.monotonic())
        info["a_step_ms_median"] = statistics.median(
            [r["a_step_ms"] for r in records if r.get("a_step_ms")] or [0]) or None
    ctx.sampler = None
    info["settle"] = {"reached": all(s["reached"] for s in settles) if settles else None,
                      "batches": settles}  # fmt: skip
    info["seconds"] = time.monotonic() - started_all
    info["drift"] = drift_counts([r for r in records if "error" not in r])
    end_of_level(ctx, info)
    return {"info": info, "records": records}


def drift_counts(records: list[dict]) -> dict:
    """Per intended case: records measured, and how many A's log put elsewhere."""
    table: dict = {}
    for r in records:
        if r.get("location_actual") is None:
            continue
        row = table.setdefault(r["case"], {"n": 0, "drifted": 0, "became": {}})
        row["n"] += 1
        if r["drift"]:
            row["drifted"] += 1
            row["became"][r["location_actual"]] = row["became"].get(r["location_actual"], 0) + 1
    return table


# Part C: what staging costs A -----------------------------------------------------


def read_aiperf_export(artifacts: Path) -> list[dict]:
    """AIPerf's per-request rows: start_ns, ttft_s and itl_s (None if missing)."""
    path = next(iter(sorted(artifacts.rglob("profile_export.jsonl"))), None)
    if path is None:
        return []

    def seconds(metric) -> float | None:
        if not isinstance(metric, dict) or metric.get("value") is None:
            return None
        scale = {"ms": 1e-3, "s": 1.0, "ns": 1e-9, "us": 1e-6}.get(metric.get("unit", "ms"), 1e-3)
        return float(metric["value"]) * scale

    rows = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        start = (rec.get("metadata") or {}).get("request_start_ns")
        metrics = rec.get("metrics") or {}
        if start is not None:
            rows.append({"start_ns": int(start),
                         "ttft_s": seconds(metrics.get("time_to_first_token")),
                         "itl_s": seconds(metrics.get("inter_token_latency"))})  # fmt: skip
    return rows


def pctl(values: list, fraction: float) -> float | None:
    values = [v for v in values if v is not None]
    return kv.percentile(values, fraction) if values else None


def median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def impact_window(ctx: Ctx, level: int, staging: bool, dataset: Path, states: dict,
                  sessions: list[int], pairs: dict) -> dict:  # fmt: skip
    """One window: AIPerf on A, and B sending cpu-case lookups at a steady rate.

    B is reset before each lookup (GPU and its CPU tier: reset_external=true) so the
    lookup goes to A every time. A's cpu states stay CPU-only, because staged blocks are
    never hashed, so the same states serve every lookup in both windows.
    """
    args = ctx.args
    tag = f"{level}-{'on' if staging else 'off'}"
    load = level_load(ctx, level, dataset, f"impact-{tag}")
    lookups: list[dict] = []
    with load as active, MetricsSampler(ctx.servers.producer, SAMPLE_INTERVAL_SECONDS) as sampler:
        settle = h5.wait_loaded(ctx.servers.producer, level, active)
        a_off = ctx.a_log.end()
        start_ns, start = time.time_ns(), time.monotonic()
        n = 0
        while time.monotonic() - start < args.impact_seconds:
            tick = time.monotonic()
            session = sessions[n % len(sessions)]
            n += 1
            current = pairs[session]
            reset_gpu(ctx.servers.consumer, attempts=3, external=True)
            params = h4.peer_params(ctx.producer, args.identity)
            if not staging:
                params["peer_stage"] = False
            try:
                got = h4.greedy(ctx.servers, ctx.servers.consumer,
                                {"prompt": current,
                                 "cache_salt": states[(session, "cpu", "peer_stage")],
                                 "kv_transfer_params": params})  # fmt: skip
                lookups.append({"session": session, "ttft_s": got["ttft_s"]})
            except (OSError, RuntimeError) as exc:
                lookups.append({"session": session, "error": repr(exc)})
            time.sleep(max(0.0, args.impact_interval - (time.monotonic() - tick)))
        end_ns, end = time.time_ns(), time.monotonic()
        a_text = ctx.a_log.read_from(a_off)[0]
        events = parse_events(a_text)
        window = {"free_blocks_min": sampler.free_blocks_min(start, end),
                  "preemptions": sampler.preemptions(start, end),
                  "held_peak": sampler.held_peak(start, end)}  # fmt: skip
    rows = [r for r in read_aiperf_export(ctx.args.output.parent / f"aiperf-impact-{tag}")
            if start_ns <= r["start_ns"] <= end_ns]  # fmt: skip
    ttft, itl = [r["ttft_s"] for r in rows], [r["itl_s"] for r in rows]
    return {
        "level": level, "staging": staging, "settled": settle["reached"],
        "lookups": len(lookups), "lookup_errors": sum("error" in x for x in lookups),
        "b_ttft_p50": median([x.get("ttft_s") for x in lookups]),
        "a_requests": len(rows),
        "a_ttft_p50": pctl(ttft, 0.5), "a_ttft_p99": pctl(ttft, 0.99),
        "a_itl_p50": pctl(itl, 0.5), "a_itl_p99": pctl(itl, 0.99),
        "stage_events": {k: sum(e["kind"] == k for e in events)
                         for k in ("submitted", "done", "trimmed", "skipped")},  # fmt: skip
        **window,
    }  # fmt: skip


def run_impact_level(ctx: Ctx, level: int, measured, background, render) -> dict:
    """Part C at one level: staging off for one window, then on for another."""
    args = ctx.args
    info: dict = {"level": level}
    dataset = write_dataset(ctx, level, background, render, info)
    assert dataset is not None, "the impact part needs a load level above 0"
    pairs = [(n, prev, cur) for n, (prev, cur) in enumerate(measured)]
    a = AEngine(ctx.servers, random.Random(args.seed))
    states = prepare_level(a, pairs, ["cpu"], ["peer_stage"], level, ctx.nonce,
                           args.block_size, args.bytes_per_token)  # fmt: skip
    prompts = {n: cur for n, _, cur in pairs}
    windows = []
    for staging in (False, True):
        windows.append(impact_window(ctx, level, staging, dataset, states,
                                     [n for n, _, _ in pairs], prompts))  # fmt: skip
        print(json.dumps({k: v for k, v in windows[-1].items()
                          if k in ("level", "staging", "lookups", "a_ttft_p99")}), flush=True)  # fmt: skip
        h5.wait_idle(ctx.servers.producer)
    end_of_level(ctx, info)
    return {"info": info, "records": [], "windows": windows}


# Analysis ---------------------------------------------------------------------


BUCKETS = ["<4k", "4-8k", "8-16k", "16-32k", ">=32k"]  # SIZE_EDGES, in tokens


def bucket(tokens: int | None) -> str:
    if tokens is None:
        return "-"
    return BUCKETS[sum(tokens >= edge for edge in SIZE_EDGES)]


def bucket_order(label: str) -> int:
    return BUCKETS.index(label) if label in BUCKETS else len(BUCKETS)


def stats(values: list) -> dict:
    return {"p50": pctl(values, 0.5), "p99": pctl(values, 0.99), "n": len([v for v in values if v is not None])}


def table_reply(records: list[dict]) -> list[dict]:
    """Table 1: A's reply time, per cpu_tokens bucket and level (peer_stage arm)."""
    groups: dict = {}
    for r in records:
        ps = r.get("peer_stage")
        if ps is not None and r.get("cpu_tokens", 0) > 0:
            groups.setdefault((bucket(r["cpu_tokens"]), r["level"]), []).append(ps)
    rows = []
    for (label, level), arms in sorted(groups.items(), key=lambda g: (g[0][1], bucket_order(g[0][0]))):
        outcomes: dict = {}
        for a in arms:
            outcomes[a["outcome"]] = outcomes.get(a["outcome"], 0) + 1
        rows.append({
            "cpu_tokens": label, "level": level, "n": len(arms),
            "reply": stats([a["reply_upper_s"] for a in arms]),
            "stage": stats([a["stage_s"] for a in arms]),
            "fence_wait_p50": median([a["fence_wait_s"] for a in arms]),
            "outcomes": outcomes,
        })  # fmt: skip
    return rows


def table_ttft(records: list[dict]) -> list[dict]:
    """Table 2, the headline: B's TTFT per location, prefix size and level."""
    groups: dict = {}
    for r in records:
        if r.get("location_actual"):
            key = (r["location_actual"], bucket(r["expected_hit_tokens"]), r["level"])
            groups.setdefault(key, []).append(r)
    rows = []
    for (location, size, level), rs in sorted(groups.items(), key=lambda g: (g[0][2], g[0][0], bucket_order(g[0][1]))):
        row = {"location": location, "size": size, "level": level, "n": len(rs),
               "ttft": {arm: stats([r[arm]["ttft_s"] for r in rs if arm in r])
                        for arm in ARMS if any(arm in r for r in rs)}}  # fmt: skip
        row["skipped"] = sum(r.get("peer_stage", {}).get("outcome") == "skipped" for r in rs)
        rs = [r for r in rs if r.get("peer_stage", {}).get("outcome") != "skipped"]
        both = [r for r in rs if "peer_stage" in r and "peer" in r]
        row["win_vs_peer"] = (sum(r["peer_stage"]["ttft_s"] < r["peer"]["ttft_s"] for r in both) / len(both)
                              if both else None)  # fmt: skip
        row["saving_vs_peer_s"] = median([r["peer"]["ttft_s"] - r["peer_stage"]["ttft_s"] for r in both])
        others = ("recompute", "stock", "peer", "on_a")
        best = [(r, min(r[a]["ttft_s"] for a in others if a in r))
                for r in rs if "peer_stage" in r and any(a in r for a in others)]  # fmt: skip
        row["win_vs_best"] = (sum(r["peer_stage"]["ttft_s"] < b for r, b in best) / len(best)
                              if best else None)  # fmt: skip
        row["saving_vs_best_s"] = median([b - r["peer_stage"]["ttft_s"] for r, b in best])
        rows.append(row)
    return rows


def table_impact(windows: list[dict]) -> list[dict]:
    keys = ("level", "staging", "a_requests", "a_ttft_p50", "a_ttft_p99", "a_itl_p50",
            "a_itl_p99", "free_blocks_min", "preemptions", "lookups", "b_ttft_p50")  # fmt: skip
    return [{k: w.get(k) for k in keys}
            for w in sorted(windows, key=lambda w: (w["level"], w["staging"]))]  # fmt: skip


def record_deadline(r: dict, label: float | None) -> float | None:
    """The deadline A used for a record: the run's label, else A's own log line."""
    return label if label is not None else r["peer_stage"].get("deadline_s")


def table_deadline(records: list[dict], label: float | None = None) -> list[dict]:
    """Table 4: trim rate and B's TTFT, per peer_stage_deadline."""
    groups: dict = {}
    for r in records:
        ps = r.get("peer_stage")
        if ps is None or not r.get("cpu_tokens"):
            continue
        deadline = record_deadline(r, label)
        if deadline is None:
            deadline = next((x["peer_stage"]["deadline_s"] for x in records
                             if x.get("peer_stage", {}).get("deadline_s")), None)  # fmt: skip
        groups.setdefault(deadline, []).append(r)
    rows = []
    for deadline, rs in sorted(groups.items(), key=lambda g: (g[0] is None, g[0])):
        trimmed = [r for r in rs if r["peer_stage"]["outcome"].startswith("trimmed")]
        staged = [r for r in rs if r["peer_stage"]["outcome"] == "staged"]
        skipped = [r for r in rs if r["peer_stage"]["outcome"] == "skipped"]
        both = [r for r in trimmed if "peer" in r]
        rows.append({
            "deadline_s": deadline, "n": len(rs), "trimmed": len(trimmed),
            "trim_rate": len(trimmed) / len(rs),
            "skipped": len(skipped), "skip_rate": len(skipped) / len(rs),
            "staged_ttft_p50": median([r["peer_stage"]["ttft_s"] for r in staged]),
            "trimmed_ttft_p50": median([r["peer_stage"]["ttft_s"] for r in trimmed]),
            "trimmed_vs_peer_s": median([r["peer_stage"]["ttft_s"] - r["peer"]["ttft_s"] for r in both]),
            "levels": sorted({r["level"] for r in rs}),
        })  # fmt: skip
    return rows


def staged_pairs(records: list[dict], locations=CPU_CASES, skipped: bool = False) -> list[dict]:
    """cpu and cpu_tail records with both arms. A skipped request never staged, so
    it says nothing about what staging costs: it is left out unless asked for."""
    return [r for r in records if r.get("location_actual") in locations
            and "peer" in r and "peer_stage" in r
            and (skipped or r["peer_stage"].get("outcome") != "skipped")]  # fmt: skip


def break_even(records: list[dict]) -> list[str]:
    """One line per level: the cpu size above which peer_stage always beat peer."""
    lines = []
    for level in sorted({r["level"] for r in records}):
        rs = staged_pairs([r for r in records if r["level"] == level])
        if not rs:
            continue
        sizes = sorted({r["cpu_tokens"] for r in rs})
        step = median([r["a_step_ms"] for r in rs])
        at = f"at A step = {step:.0f} ms" if step else "at an idle A"
        wins = {s: all(r["peer_stage"]["ttft_s"] < r["peer"]["ttft_s"]
                       for r in rs if r["cpu_tokens"] >= s) for s in sizes}  # fmt: skip
        good = [s for s in sizes if wins[s]]
        if not good:
            lines.append(f"break-even: level {level}: peer_stage never beats peer up to "
                         f"{sizes[-1]} cpu tokens {at}")  # fmt: skip
        elif good[0] == sizes[0]:
            lines.append(f"break-even: level {level}: peer_stage beats peer at every size "
                         f"from {sizes[0]} cpu tokens {at}")  # fmt: skip
        else:
            lines.append(f"break-even: peer_stage beats peer above {good[0]} cpu tokens "
                         f"at A step = {step:.0f} ms (level {level})" if step else
                         f"break-even: peer_stage beats peer above {good[0]} cpu tokens "
                         f"(level {level}, idle A)")  # fmt: skip
    return lines


def would_skip(est_s: float, cpu_tokens: int, rate: float, deadline_s: float) -> bool:
    """A's skip rule: the estimate exceeds the deadline or the tail's recompute time."""
    return est_s > deadline_s or (rate > 0 and est_s > cpu_tokens / rate)


def score_offline(records: list[dict], rate: float) -> dict | None:
    """Score the skip rule against what staging measured: the rule is right when it
    skips a request staging lost, or stages one staging won."""
    rows = [r for r in staged_pairs(records)
            if r["peer_stage"]["outcome"] in ("staged", "trimmed_deadline", "trimmed_failed")
            and r["peer_stage"].get("est_stage_s") is not None]  # fmt: skip
    if not rows:
        return None
    right = skipped_winners = staged_losers = 0
    for r in rows:
        ps = r["peer_stage"]
        win = ps["ttft_s"] < r["peer"]["ttft_s"]
        skip = would_skip(ps["est_stage_s"], r["cpu_tokens"], rate,
                          ps.get("deadline_s") or DEFAULT_DEADLINE_S)  # fmt: skip
        right += skip != win
        skipped_winners += skip and win
        staged_losers += (not skip) and not win
    return {"right": right, "of": len(rows), "skipped_winners": skipped_winners,
            "staged_losers": staged_losers, "rate": rate}  # fmt: skip


def score_live(records: list[dict]) -> dict | None:
    """Score A's live skip decisions. A staged request is judged by its measured
    TTFT against peer. A skipped one, by whether its estimate was above the tail's
    recompute time at B's own rate (from the recompute arm)."""
    right = total = 0
    for r in staged_pairs(records, skipped=True):
        ps, outcome = r["peer_stage"], r["peer_stage"]["outcome"]
        if outcome == "skipped":
            if "recompute" not in r or ps.get("est_stage_s") is None:
                continue
            tail_s = r["cpu_tokens"] * r["recompute"]["ttft_s"] / r["prompt_tokens"]
            right += ps["est_stage_s"] >= tail_s
        elif outcome == "staged" or outcome.startswith("trimmed"):
            right += ps["ttft_s"] < r["peer"]["ttft_s"]
        else:
            continue
        total += 1
    return {"right": right, "of": total} if total else None


def verdicts(part: str, records: list[dict], windows: list[dict], score_rate: float,
             deadline_label: float | None = None) -> list[str]:
    lines: list[str] = []
    if part in ("sweep", "skip"):
        lines += break_even(records)
        off = score_offline(records, score_rate)
        if off:
            lines.append(f"skip rule (offline, {score_rate:.0f} tok/s): right on {off['right']} "
                         f"of {off['of']}; would skip {off['skipped_winners']} winners, "
                         f"stage {off['staged_losers']} losers")  # fmt: skip
    if part == "skip":
        live = score_live(records)
        levels = ",".join(str(x) for x in sorted({r["level"] for r in records}))
        if live:
            lines.append(f"skip rule (live, level {levels}): right on {live['right']} of {live['of']}")
    if part == "deadline":
        for row in table_deadline(records, deadline_label):
            vs = row["trimmed_vs_peer_s"]
            lines.append(f"deadline {row['deadline_s']} s: trims {row['trim_rate'] * 100:.0f}%, "
                         f"skips {row['skip_rate'] * 100:.0f}% "
                         f"at level {','.join(map(str, row['levels']))}; trimmed TTFT "
                         + (f"{vs:+.2f} s vs peer" if vs is not None else "n/a (no trimmed request)"))  # fmt: skip
    if part == "impact":
        by_level: dict = {}
        for w in windows:
            by_level.setdefault(w["level"], {})[w["staging"]] = w
        for level, pair in sorted(by_level.items()):
            off_w, on_w = pair.get(False), pair.get(True)
            if off_w and on_w and off_w["a_ttft_p99"] and on_w["a_ttft_p99"]:
                change = (on_w["a_ttft_p99"] / off_w["a_ttft_p99"] - 1) * 100
                lines.append(f"impact on A: TTFT p99 {change:+.0f}% with staging at level {level}")
    return lines


# Checks ------------------------------------------------------------------------------


def rdma_check(b_log_text: str) -> dict:
    """B's KV reads must take an RDMA transport (UCX_PROTO_INFO in B's log)."""
    gate = classify_transports(b_log_text)
    return {"ok": gate.ok, "transports": gate.transports, "reason": gate.reason}


def check_records(records: list[dict], level_infos: list[dict], rdma: dict | None,
                  require_staged: bool = False) -> dict:  # fmt: skip
    """H5's checks, with the pulled bytes judged against what each reply held, plus
    the H6 checks. A record's index is in the failing list."""
    failures: dict[str, list[int]] = {n: [] for n in ("served", "lookup", "pulled", "peer_arm_unstaged")}
    rate = h4.bytes_per_token(records)
    compared = equal = ties = 0
    staged = staged_bad = 0
    for i, r in enumerate(records):
        rc = r.get("recompute")
        for arm in ("stock", "peer", "peer_stage", "on_a"):
            if rc is not None and arm in r:
                compared += 1
                if h4.matches(r[arm], rc):
                    equal += 1
                elif h4.is_near_tie(r[arm], rc):
                    ties += 1
        if rc is not None and r.get("peer_stage", {}).get("outcome") == "staged":
            staged += 1
            ps = r["peer_stage"]
            staged_bad += not (h4.matches(ps, rc) or h4.is_near_tie(ps, rc))
        for arm in ("peer", "peer_stage"):
            a = r.get(arm)
            if a is None:
                continue
            failures["served"] += [i] if a["a_served"] != 0 else []
            failures["lookup"] += [i] if a["lookups_logged"] == 0 else []
            pulled, kept = a["transfer"]["bytes"], a["kept_tokens"]
            ok = pulled == 0 if kept == 0 else (
                pulled > 0 if rate is None
                else abs(pulled - kept * rate) <= h4.BYTES_TOLERANCE * kept * rate)  # fmt: skip
            failures["pulled"] += [] if ok else [i]
        if r.get("peer", {}).get("stage_submitted"):
            failures["peer_arm_unstaged"].append(i)
    result: dict = {n: {"ok": not idx, "records": idx} for n, idx in failures.items()}
    fraction = (equal + ties) / compared if compared else 1.0
    result["equivalent"] = {"ok": fraction >= h4.MIN_EQUIVALENT_FRACTION, "fraction": fraction,
                            "compared": compared, "exact": equal, "near_ties": ties}  # fmt: skip
    # ok=None (nothing staged) passes only where staging was not expected at all.
    must_stage = require_staged and staged == 0
    result["staged_equivalent"] = {"ok": False if must_stage else (staged_bad == 0) if staged else None,
                                   "staged": staged, "mismatched": staged_bad}  # fmt: skip
    if must_stage:
        result["staged_equivalent"]["reason"] = "CPU cases ran but no peer_stage request was staged"
    result["no_leaks"] = {"ok": all(i.get("leak_free") for i in level_infos)}
    result["no_draining_left"] = {
        "ok": all(i["no_draining_left"]["ok"] for i in level_infos if "no_draining_left" in i),
        "first_attempt_failed_levels": [i["level"] for i in level_infos
                                        if not i.get("no_draining_left", {}).get("first_attempt_ok", True)]}  # fmt: skip
    result["rdma_path"] = rdma if rdma is not None else {"ok": None}
    result["bytes_per_token"] = rate
    return result


# Printing ----------------------------------------------------------------------------


def f(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_tables(tables: dict) -> None:
    if tables.get("reply"):
        print("\nTable 1: A reply time (peer_stage arm)")
        print("cpu_tokens  level   n  reply p50/p99  stage p50/p99  fence p50  outcomes")
        for r in tables["reply"]:
            print(f"{r['cpu_tokens']:<10} {r['level']:>5} {r['n']:>3}  "
                  f"{f(r['reply']['p50']):>5}/{f(r['reply']['p99']):<7}  "
                  f"{f(r['stage']['p50']):>5}/{f(r['stage']['p99']):<7}  "
                  f"{f(r['fence_wait_p50']):>9}  {json.dumps(r['outcomes'])}")  # fmt: skip
    if tables.get("ttft"):
        print("\nTable 2: B TTFT p50/p99 (s)")
        print("location  size  level   n  " + "  ".join(f"{a:<11}" for a in ARMS)
              + "  win/peer  win/best  save/peer")  # fmt: skip
        for r in tables["ttft"]:
            cells = "  ".join(
                f"{f(r['ttft'][a]['p50']):>5}/{f(r['ttft'][a]['p99']):<5}" if a in r["ttft"]
                else f"{'-':<11}" for a in ARMS)  # fmt: skip
            print(f"{r['location']:<8} {r['size']:<6}{r['level']:>4} {r['n']:>3}  {cells}  "
                  f"{f(r['win_vs_peer']):>8}  {f(r['win_vs_best']):>8}  {f(r['saving_vs_peer_s']):>9}")  # fmt: skip
    if tables.get("impact"):
        print("\nTable 3: impact on A")
        print("level  staging  A ttft p50/p99  A itl p50/p99  min free blocks  preempt")
        for r in tables["impact"]:
            print(f"{r['level']:>5}  {'on' if r['staging'] else 'off':<7}  "
                  f"{f(r['a_ttft_p50']):>5}/{f(r['a_ttft_p99']):<7}  "
                  f"{f(r['a_itl_p50'], 4):>6}/{f(r['a_itl_p99'], 4):<7}  "
                  f"{f(r['free_blocks_min'], 0):>15}  {f(r['preemptions'], 0):>7}")  # fmt: skip
    if tables.get("deadline"):
        print("\nTable 4: deadline")
        print("deadline_s  trim rate  skip rate  staged p50  trimmed p50  trimmed vs peer")
        for r in tables["deadline"]:
            print(f"{f(r['deadline_s']):>10}  {r['trim_rate'] * 100:>8.0f}%  "
                  f"{r['skip_rate'] * 100:>8.0f}%  "
                  f"{f(r['staged_ttft_p50']):>10}  {f(r['trimmed_ttft_p50']):>11}  "
                  f"{f(r['trimmed_vs_peer_s']):>15}")  # fmt: skip


def print_checks(checks: dict) -> None:
    for name, result in checks.items():
        if isinstance(result, dict):
            state = {True: "PASS", False: "FAIL", None: "skip"}[result["ok"]]
            print(f"{state}  {name}")


# Main ----------------------------------------------------------------------------


def discover(url: str) -> dict:
    """A's peer_lookup record from /kv_event_sources (where B learns A's address)."""
    with urllib.request.urlopen(f"{url}/kv_event_sources", timeout=10) as response:
        sources = json.load(response)
    for _, source in sorted(sources.items()):
        info = (source or {}).get("peer_lookup")
        if info and info.get("enabled", True):
            return info
    raise RuntimeError("A's /kv_event_sources has no peer_lookup record")


def check_discovery(info: dict, producer_url: str, expect_host: str | None) -> None:
    """A must advertise its node IP, or B on another node would dial localhost."""
    host = str(info.get("side_channel_host", ""))
    wanted = expect_host or urlsplit(producer_url).hostname or ""
    if host.startswith(LOOPBACK) and not wanted.startswith(LOOPBACK):
        raise RuntimeError(f"A advertises its side channel at {host}, not at {wanted}: "
                           "set VLLM_NIXL_SIDE_CHANNEL_HOST=$MS_NODE_IP on A")  # fmt: skip
    if not wanted.startswith(LOOPBACK) and host != wanted:
        raise RuntimeError(f"A advertises its side channel at {host}, expected {wanted} "
                           "(VLLM_NIXL_SIDE_CHANNEL_HOST)")  # fmt: skip
    if not info.get("cpu_stage"):
        raise RuntimeError("A's peer_lookup record has cpu_stage off: peer_stage is not wired "
                           "(NixlConnector before OffloadingConnector, peer_stage true)")  # fmt: skip


def producer_params(info: dict) -> dict:
    """The kv_transfer_params fields that address A, from its discovery record."""
    params = {"remote_engine_id": info["engine_id"], "remote_host": info["side_channel_host"],
              "remote_port": info["side_channel_port"]}  # fmt: skip
    for key in ("tp_size", "dcp_size", "pp_size"):
        if key in info:
            params[key] = info[key]
    return params


def positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"needs a positive number, got {text!r} (an empty value means the environment "
            "variable that fills it, PEER_STAGE_RATE or PEER_STAGE_DEADLINE, is not set)")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model")
    p.add_argument("--no-rdma-gate", action="store_true",
                   help="single node: record B's read transports without requiring RDMA")
    p.add_argument("--producer", help="A's base URL, http://<node0 ip>:8100")
    p.add_argument("--consumer", help="B's base URL")
    p.add_argument("--producer-log", help="a path, or http://<node0 ip>:8399/logs/<name>")
    p.add_argument("--consumer-log", help="a path, or a control endpoint log URL")
    p.add_argument("--part", choices=PARTS, default="sweep")
    p.add_argument("--levels", help="background sessions on A; the default depends on --part")
    p.add_argument("--cases", help=f"of {','.join(CASES)}; the default depends on --part")
    p.add_argument("--arms", help=f"of {','.join(ARMS)}; the default depends on --part")
    p.add_argument("--traces", default=h4.DEFAULT_DATASET)
    p.add_argument("--sessions", type=int, default=10, help="measured sessions per level")
    p.add_argument("--batch-sessions", type=int, default=0,
                   help="most sessions per unit; the units are sized to A's GPU pool and "
                        "CPU tier (0 = no cap)")  # fmt: skip
    p.add_argument("--cpu-gb", type=float, default=64.0,
                   help="A's CPU offload tier in GiB (CPU_GB in h6_server.sh)")  # fmt: skip
    p.add_argument("--max-prompt-tokens", type=int, default=30000)
    p.add_argument("--background-pool", type=int, default=300)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--bytes-per-token", type=float, default=EXPECTED_QWEN3_8B_BYTES_PER_TOKEN)
    p.add_argument("--identity", type=json.loads, default=None)
    p.add_argument("--lookup-pattern", default=h4.LOOKUP_PATTERN)
    p.add_argument("--expect-a-host", help="A's node IP; default: the host of --producer")
    p.add_argument("--aiperf-command", default="aiperf")
    p.add_argument("--ramp-seconds", type=int, default=20)
    p.add_argument("--level-seconds", type=int, default=1800)
    p.add_argument("--impact-interval", type=float, default=0.5)
    p.add_argument("--impact-seconds", type=float, default=180.0)
    p.add_argument("--score-rate", type=positive_float, default=10000.0,
                   help="tokens/s of B's prefill, for the offline skip-rule score")  # fmt: skip
    p.add_argument("--deadline-s", type=positive_float, default=None,
                   help="A's peer_stage_deadline in this run, to label the deadline table")  # fmt: skip
    p.add_argument("--merge", nargs="+", type=Path, metavar="RESULTS",
                   help="merge deadline runs' results.json into Table 4 and exit")  # fmt: skip
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results.json"))
    args = p.parse_args(argv)
    if args.merge is None:
        missing = [n for n in ("model", "producer", "consumer", "producer_log", "consumer_log")
                   if getattr(args, n) is None]  # fmt: skip
        if missing:
            p.error("required: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    if args.merge is None and args.part == "deadline" and args.deadline_s is None:
        p.error("--part deadline needs --deadline-s (A's peer_stage_deadline, for Table 4)")
    defaults = PART_DEFAULTS[args.part]
    for name in ("levels", "cases", "arms"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    return args


def parse_names(text: str, allowed: tuple[str, ...], what: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    bad = [n for n in names if n not in allowed]
    if bad or not names:
        raise ValueError(f"bad {what}: {text!r} (choose from {','.join(allowed)})")
    return names


def merge_main(args: argparse.Namespace) -> int:
    """Table 4 and the deadline verdicts, from several deadline runs' results."""
    records: list[dict] = []
    for path in args.merge:
        result = json.loads(path.read_text())
        label = result.get("a_deadline_s")
        for r in result.get("records", []):
            if "error" in r or "peer_stage" not in r:
                continue
            if label is not None:
                r["peer_stage"]["deadline_s"] = label  # one label per run
            records.append(r)
    tables = {"deadline": table_deadline(records)}
    lines = verdicts("deadline", records, [], args.score_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"part": "deadline", "merged_from": [str(p) for p in args.merge],
         "tables": tables, "verdicts": lines}, indent=2))  # fmt: skip
    print_tables(tables)
    for line in lines:
        print(line)
    return 0 if records else 1


def check_block_size(url: str, block_size: int) -> None:
    """A's KV block size must be the one the probe sizes its prompts in. vLLM picks
    its own on some GPUs (64 on H100), which h6_server.sh pins with --block-size."""
    seen = scrape(url).get("block_size")
    if seen != block_size:
        raise RuntimeError(f"A's vllm:cache_config_info says block_size={seen}, but the probe "
                           f"runs with --block-size {block_size}: start A with --block-size "
                           f"{block_size} (h6_server.sh does)")  # fmt: skip


def write_results(ctx: Ctx, results: dict, windows: list[dict], measured_n: int, info: dict,
                  levels, cases, arms) -> tuple[dict, dict, list[str]]:  # fmt: skip
    """results.json from what has been measured so far; called after every level."""
    args = ctx.args
    records = [r for o in results.values() for r in o["records"]]
    good = [r for r in records if "error" not in r]
    level_infos = [o["info"] for o in results.values()]
    rdma = rdma_check(ctx.b_log.text())  # the read table so far
    if ctx.args.no_rdma_gate:
        rdma = {**rdma, "ok": None, "reason": "not checked (--no-rdma-gate)"}
    expects_staging = (args.part in ("sweep", "deadline") and "peer_stage" in arms
                       and any(c in CPU_CASES for c in cases))  # fmt: skip
    checks = check_records(good, level_infos, rdma, require_staged=expects_staging)
    tables = {"reply": table_reply(good), "ttft": table_ttft(good),
              "impact": table_impact(windows), "deadline": table_deadline(good, args.deadline_s)
              if args.part == "deadline" else []}  # fmt: skip
    lines = verdicts(args.part, good, windows, args.score_rate, args.deadline_s)
    args.output.write_text(json.dumps({
        "part": args.part, "model": args.model, "levels": levels, "cases": cases,
        "arms": arms, "sessions": measured_n, "a_deadline_s": args.deadline_s,
        "levels_done": list(results), "checks": checks, "tables": tables, "verdicts": lines,
        "level_info": {n: o["info"] for n, o in results.items()},
        "discovery": info, "read_path": kv.read_path(ctx.b_log.text()),
        "needs_real_run": NEEDS_REAL_RUN, "records": records}, indent=2))  # fmt: skip
    return checks, tables, lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge is not None:
        return merge_main(args)
    levels = h5.parse_levels(args.levels)
    cases = parse_names(args.cases, CASES, "cases")
    arms = parse_names(args.arms, ARMS, "arms")
    if args.part == "impact" and 0 in levels:
        print("FAIL: the impact part needs load levels above 0", flush=True)
        return 1
    servers = Servers(args.model, args.producer, args.consumer)
    a_log, b_log = LogSource(args.producer_log), LogSource(args.consumer_log)
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    kv.wait_ready(servers.producer)
    kv.wait_ready(servers.consumer)

    try:
        info = discover(servers.producer)
        check_discovery(info, servers.producer, args.expect_a_host)
        check_block_size(servers.producer, args.block_size)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"FAIL: discovery: {exc}", flush=True)
        return 1
    producer = producer_params(info)

    render = h4.chat_renderer(args.model)
    measured, background = h5.pick_sessions(args, render)
    if not measured:
        print("FAIL: no session has a usable turn pair", flush=True)
        return 1
    # One stock prefill that B pulls, before any load: it frees A's blocks now and
    # makes UCX print its read table, so the RDMA gate can stop a TCP run early.
    address_prompt = kv.fresh_prompt(random.Random(args.seed), 64)
    params = kv.prefill(servers, address_prompt)
    kv.timed_stream(servers, servers.consumer,
                    {"prompt": address_prompt, "kv_transfer_params": params})  # fmt: skip
    time.sleep(kv.METRICS_WAIT_SECONDS)
    rdma = rdma_check(b_log.text())
    if args.no_rdma_gate:
        # One node, two GPUs: reads go over NVLink (cuda_ipc), as in H5. Record
        # the transports but do not judge them.
        rdma = {**rdma, "ok": None, "reason": "not checked (--no-rdma-gate)"}
    elif not rdma["ok"]:
        print(f"FAIL: rdma_path: {rdma['reason']} (transports: {rdma['transports']})", flush=True)
        args.output.write_text(json.dumps({"part": args.part, "checks": {"rdma_path": rdma}}, indent=2))
        return 1

    ctx = Ctx(args, servers, producer, a_log, b_log, uuid.uuid4().hex[:6])
    results: dict[int, dict] = {}
    records: list[dict] = []
    windows: list[dict] = []
    with (out_dir / "records.jsonl").open("w") as records_file:
        ctx.records_file = records_file
        for level in levels:
            if args.part == "impact":
                outcome = run_impact_level(ctx, level, measured, background, render)
                windows += outcome["windows"]
            else:
                outcome = run_level(ctx, level, measured, background, render, cases, arms)
            results[level] = outcome
            records += outcome["records"]
            # results.json after every level, so a run that dies keeps what it measured
            checks, tables, lines = write_results(ctx, results, windows, len(measured),
                                                  info, levels, cases, arms)  # fmt: skip

    good = [r for r in records if "error" not in r]
    for log in (a_log, b_log):
        target = out_dir / log.name
        if log.remote or Path(log.target).resolve() != target.resolve():
            target.write_bytes(log.read_bytes(0))
        elif Path(log.target).exists():
            shutil.copy(log.target, target)

    print_tables(tables)
    for line in lines:
        print(line)
    print_checks(checks)
    drift = {n: o["info"]["drift"] for n, o in results.items() if o["info"].get("drift")}
    if drift:
        print(f"drift (intended case -> where A's log put it): {json.dumps(drift)}")
    errors = len(records) - len(good)
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v["ok"] is False]
    unloaded = [n for n, o in results.items() if n and o["info"].get("settle", {}).get("reached") is False]
    if unloaded:
        print(f"warning: A never reached the target load at levels {unloaded}", flush=True)
    if errors or failed or (not good and args.part != "impact"):
        print(f"FAIL: {errors} errored records, failed checks: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
