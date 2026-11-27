"""Peer stage end to end: does A copy CPU-only blocks to GPU before it replies?

SCRATCH, not committed. Run as the bench of examples/kv-transfer/peer_stage_e2e.yaml
(design/vllm/peer-pin-cpu-stage-fallback.md, section 4), against four engines of
Qwen3-0.6B on two GPUs, each running MultiConnector(NixlConnector, OffloadingConnector):

  A      GPU 0, peer_stage on, small GPU pool (num_gpu_blocks_override) so a
         prefix spills to the CPU tier; peer_stage_recompute_tokens_per_s 0
  A_DL   GPU 0, like A with a short peer_stage_deadline, and slow_loads_s armed
         in its control file so a copy really misses it
  A_SKIP GPU 0, like A with peer_stage_recompute_tokens_per_s very high
  B      GPU 1, the consumer; peer_lookup on, no staging

Each scenario sends B a request that carries kv_transfer_params (peer_lookup,
do_remote_prefill, A's address), as the router would, after A was set up with a
fresh random prompt. The state of A's caches is built through its HTTP API, plus
the hook in probes/peer_stage_hook/sitecustomize.py where traffic cannot do it
(see "Faithful vs approximated" below). Scenarios, one lookup each:

  cpu_only         prefix on A's CPU tier only (flood of unique prompts pushes it
                   out of the GPU): full hit, staged == whole prefix, in_place 0
  mixed            GPU / CPU / GPU: first 16 blocks and last 24 stay on GPU, the
                   24 between are CPU only: full hit, staged == 24, in_place == 24
  deadline_trim    A_DL, GPU run 16 blocks then CPU tail, the copy held back past
                   the deadline: partial hit equal to the GPU run, `trimmed
                   reason=deadline`, B's reply says stage_timeout, the copy
                   reports late (ok=True) and the blocks are freed after it
  load_failure     A, GPU run then CPU tail, every load job reported failed:
                   `trimmed reason=failed`, partial hit equal to the GPU run
  older_consumer   B drops reply_budget_s from its lookup. (a) A stages with its
                   own peer_stage_deadline (not the budget-derived one); (b) on a
                   trim (A_DL) the miss reason is pin_refused, not stage_timeout
  engine_idle      A idle before the lookup and untouched after it: the reply
                   still arrives and the pin is released with no request to A
  skip_rule        A_SKIP: `Peer stage skipped`, B gets the GPU run only

Every scenario also checks: B's tokens equal its own recompute (a numeric tie
allowed), NIXL bytes moved on B for a hit (none on a full miss), no traceback in
A's log, and A's prefix cache resets afterwards (a leaked pin or a block still
draining makes vLLM refuse the reset).

Log lines of vLLM this probe depends on (grep'd from the engines' logs, one
scenario's window at a time and then by pin id, so scenarios do not bleed):

  A  Peer stage submitted: pin=%s staged=%d in_place=%d deadline_s=%.3f
  A  Peer stage done: pin=%s ok=%s seconds=%.3f
  A  Peer stage trimmed: pin=%s reason=%s kept_blocks=%d
  A  Peer stage skipped: pin=%s reason=%s est_s=%.3f
  A  Peer lookup_pin pinned %d of %d blocks as %s
  B  Peer lookup_pin hit for %s: %d tokens under pin %s
  B  Peer lookup for %s missed: %s
and the hook's tagged lines ([PSTAGE-HOOK]).

Faithful vs approximated: cpu_only, engine_idle and older_consumer (a) are plain
traffic. mixed, deadline_trim, load_failure, older_consumer (b) and skip_rule drop
the GPU cache entries of chosen blocks through the hook (a GPU / CPU / GPU layout
cannot come from LRU, which displaces tails first; the state is one an LRU cache
could be left in by other requests' blocks). deadline_trim and older_consumer (b)
also hold the CPU->GPU load's result back (slow_loads_s) because the pin server
skips a copy it estimates will miss its deadline (est = bytes / 20 GB/s), and an
idle A copies 90 MB in a few ms; the deadline timer, the in-flight draining and the
late result are real. load_failure flips the worker's TransferResult to failed.

Writes results.json to the current directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from modelsight.probes import kv_peer_lookup as pl
from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_offload_latency import Client, counter, scrape, wait_stored
from modelsight.probes.kv_transfer import Servers

STORE_FRACTION = 0.98
KEY_TOKENS = 8  # the hook keys finished prompts by their first 8 token ids
SETTLE_SECONDS = 20.0
RESET_ATTEMPTS = 60
STAGE_MARGIN_S = 0.2  # vLLM's STAGE_MARGIN_S, in peer_pins.py
FLOOD_PROMPT_TOKENS = 2048

RE_SUBMITTED = re.compile(
    r"Peer stage submitted: pin=(\S+) staged=(\d+) in_place=(\d+) deadline_s=([\d.]+)"
)
RE_DONE = re.compile(r"Peer stage done: pin=(\S+) ok=(\S+) seconds=([\d.]+)")
RE_TRIMMED = re.compile(r"Peer stage trimmed: pin=(\S+) reason=(\S+) kept_blocks=(\d+)")
RE_SKIPPED = re.compile(r"Peer stage skipped: pin=(\S+) reason=(\S+) est_s=([\d.]+)")
RE_PINNED = re.compile(r"Peer lookup_pin pinned (\d+) of (\d+) blocks as (\S+)")
RE_B_HIT = re.compile(r"Peer lookup_pin hit for (\S+): (\d+) tokens under pin (\S+)")
RE_B_MISS = re.compile(r"Peer lookup for (\S+) missed: (\S+)")
RE_SUBMIT_LINE = re.compile(r"\[PSTAGE-HOOK\] lookup submit key=(\S+) reply_budget_s=(\S+)")
RE_REPLY_LINE = re.compile(
    r"\[PSTAGE-HOOK\] lookup reply key=(\S+) hit_tokens=(\S+) miss=(\S+) pin=(\S+) error=(\S+)"
)
RE_EVICTED = re.compile(r"\[PSTAGE-HOOK\] evicted positions (\d+)-(\d+) n=(\d+) missing=(\d+) held=(\d+)")
RE_FAILED_JOB = re.compile(r"\[PSTAGE-HOOK\] failed load job=(\d+)")
RE_HELD_JOB = re.compile(r"\[PSTAGE-HOOK\] released load job=(\d+) after (\d+) ms")
RE_CRASH = re.compile(r"Traceback \(most recent call last\)|Exception in thread|EngineDeadError")
RE_NOT_SHIPPED = re.compile(r"Peer pin \S+ was not shipped within")


# Engines --------------------------------------------------------------------


@dataclass
class Engine:
    """One server: its URL, its log, its hook control file, and its NIXL address."""

    name: str
    url: str
    log: Path
    ctl: Path
    address: dict | None = None
    client: Client | None = None

    def ctl_set(self, **kwargs) -> None:
        """Merge keys into the control file (a key set to None is removed)."""
        try:
            current = json.loads(self.ctl.read_text())
        except (OSError, ValueError):
            current = {}
        for key, value in kwargs.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        tmp = self.ctl.with_suffix(".tmp")
        tmp.write_text(json.dumps(current))
        tmp.replace(self.ctl)

    def ctl_clear(self) -> None:
        self.ctl.write_text("{}")


@dataclass
class Setup:
    model: str
    a: Engine
    a_deadline: Engine
    a_skip: Engine
    b: Engine
    rng: random.Random
    block_size: int
    prompt_blocks: int
    pool_blocks: int
    bytes_per_token: float
    args: argparse.Namespace
    metrics_known: bool = True
    notes: list[str] = field(default_factory=list)

    def servers(self, a: Engine) -> Servers:
        return Servers(self.model, a.url, self.b.url)

    def prompt_tokens(self) -> int:
        # A partial last block, so B always has a token to compute.
        return self.prompt_blocks * self.block_size + self.block_size // 2


def has_series(url: str, prefix: str) -> bool:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        return any(
            line.startswith(prefix) for line in response.read().decode().splitlines()
        )


def reset(url: str, attempts: int = RESET_ATTEMPTS) -> tuple[bool, float]:
    """Reset a server's GPU prefix cache (not the connectors'): success, seconds."""
    started = time.monotonic()
    ok = pl.reset_cache(url, attempts=attempts)
    return ok, time.monotonic() - started


def ensure_address(s: Setup, a: Engine) -> dict:
    """A's NIXL side-channel address, from one stock prefill that B pulls at once
    (so A frees those blocks instead of holding them for the lease)."""
    if a.address is None:
        prompt = kv.fresh_prompt(s.rng, 64)
        servers = s.servers(a)
        addr = kv.prefill(servers, prompt)
        kv.timed_stream(servers, s.b.url, {"prompt": prompt, "kv_transfer_params": addr})
        a.address = addr
    return a.address


def calibrate(s_client: Client, rng: random.Random, block_size: int, override: float | None) -> float:
    """KV bytes per token, from what a prompt adds to the offload store counter."""
    if override:
        return override
    blocks = 128
    before = counter(scrape(s_client.url), "vllm:kv_offload_store_bytes")
    s_client.ttft(kv.fresh_prompt(rng, blocks * block_size + block_size // 2))
    if not wait_stored(s_client, rng, before, None):
        raise RuntimeError("the calibration prompt never reached the CPU store")
    stored = counter(scrape(s_client.url), "vllm:kv_offload_store_bytes") - before
    return stored / (blocks * block_size)


# Building A's cache state ----------------------------------------------------


def warm(s: Setup, a: Engine) -> tuple[list[int], list[int]]:
    """A computes a fresh prompt; return it and its hook key once it is on CPU.

    A also stored it to its CPU tier: the store counter has grown by the prompt's
    KV bytes, so the CPU copy is whole before anything is evicted from the GPU.
    """
    assert a.client is not None
    ensure_address(s, a)  # its prefill is traffic to A: do it before anything is measured
    prompt = kv.fresh_prompt(s.rng, s.prompt_tokens())
    before = counter(scrape(a.url), "vllm:kv_offload_store_bytes")
    a.client.ttft(prompt)
    expected = s.prompt_blocks * s.block_size * s.bytes_per_token
    if not wait_stored(a.client, s.rng, before, expected):
        raise RuntimeError("the prompt never reached A's CPU store")
    return prompt, prompt[:KEY_TOKENS]


def evict_gpu_flood(s: Setup, a: Engine) -> dict:
    """Push everything out of A's GPU with unique prompts, 1.2 x the pool."""
    assert a.client is not None
    target = int(1.2 * s.pool_blocks * s.block_size)
    sent = 0
    while sent < target:
        a.client.ttft(kv.fresh_prompt(s.rng, FLOOD_PROMPT_TOKENS))
        sent += FLOOD_PROMPT_TOKENS
    return {"mode": "flood", "sent_tokens": sent, "pool_tokens": s.pool_blocks * s.block_size}


def evict_gpu_reset(a: Engine) -> dict:
    """GPU-only /reset_prefix_cache (reset_external stays false)."""
    ok, seconds = reset(a.url, attempts=20)
    if not ok:
        raise RuntimeError(f"{a.name} would not reset its GPU prefix cache")
    return {"mode": "reset", "seconds": seconds}


def evict_blocks(s: Setup, a: Engine, key: list[int], lo: int, hi: int) -> dict:
    """Drop the GPU cache entries of prompt blocks lo..hi-1 through the hook."""
    offset = pl.log_size(a.log)
    a.ctl_set(evict={"key": key, "lo": lo, "hi": hi})
    ok = pl.reset_cache(a.url, attempts=3)
    deadline = time.monotonic() + 10
    match = None
    while match is None and time.monotonic() < deadline:
        match = RE_EVICTED.search(pl.log_since(a.log, offset))
        time.sleep(0.2)
    if not ok or match is None:
        raise RuntimeError(f"{a.name}: the hook did not evict blocks {lo}-{hi}")
    done = int(match.group(3))
    if done != hi - lo:
        raise RuntimeError(
            f"{a.name}: hook evicted {done} of {hi - lo} blocks "
            f"(missing {match.group(4)}, held {match.group(5)})"
        )
    return {"mode": "hook", "lo": lo, "hi": hi, "evicted": done}


# One lookup -------------------------------------------------------------------


def parse_events(a_text: str, b_text: str) -> dict:
    """The stage and lookup events of one scenario's log window, by pin."""
    submitted = [m.groups() for m in RE_SUBMITTED.finditer(a_text)]
    trimmed = [m.groups() for m in RE_TRIMMED.finditer(a_text)]
    skipped = [m.groups() for m in RE_SKIPPED.finditer(a_text)]
    done = [m.groups() for m in RE_DONE.finditer(a_text)]
    pinned = [m.groups() for m in RE_PINNED.finditer(a_text)]
    replies = [m.groups() for m in RE_REPLY_LINE.finditer(b_text)]
    submits = [m.groups() for m in RE_SUBMIT_LINE.finditer(b_text)]
    # The scenario's pin: the one B was handed; with no hit, the one A staged,
    # trimmed or skipped. A late line of an earlier scenario carries another id.
    pins = {r[3] for r in replies if r[3] != "None"}
    if not pins:
        pins = {x[0] for x in submitted + trimmed + skipped}
    keep = lambda rows: [r for r in rows if r[0] in pins]  # noqa: E731
    out: dict = {"pins": sorted(pins)}
    sub = keep(submitted)
    out["submitted"] = (
        {"staged": int(sub[0][1]), "in_place": int(sub[0][2]), "deadline_s": float(sub[0][3])}
        if sub
        else None
    )
    tr = keep(trimmed)
    out["trimmed"] = {"reason": tr[0][1], "kept_blocks": int(tr[0][2])} if tr else None
    sk = keep(skipped)
    out["skipped"] = {"reason": sk[0][1], "est_s": float(sk[0][2])} if sk else None
    dn = keep(done)
    out["done"] = {"ok": dn[0][1] == "True", "seconds": float(dn[0][2])} if dn else None
    pn = [p for p in pinned if p[2] in pins]
    out["pinned"] = {"blocks": int(pn[0][0]), "of": int(pn[0][1])} if pn else None
    reply = replies[-1] if replies else None
    out["reply"] = (
        {
            "hit_tokens": None if reply[1] == "None" else int(reply[1]),
            "miss": reply[2],
            "pin": None if reply[3] == "None" else reply[3],
            "error": None if reply[4] == "None" else reply[4],
        }
        if reply
        else None
    )
    out["reply_budget_s"] = (
        None if not submits or submits[-1][1] == "None" else float(submits[-1][1])
    )
    out["b_hit_logged"] = bool(RE_B_HIT.search(b_text))
    miss = RE_B_MISS.search(b_text)
    out["b_miss_logged"] = miss.group(2) if miss else None
    out["a_crash"] = bool(RE_CRASH.search(a_text))
    out["a_not_shipped_warning"] = bool(RE_NOT_SHIPPED.search(a_text))
    out["failed_load_jobs"] = len(RE_FAILED_JOB.findall(a_text))
    out["held_load_jobs"] = [(int(j), int(ms)) for j, ms in RE_HELD_JOB.findall(a_text)]
    return out


def lookup(
    s: Setup,
    a: Engine,
    prompt: list[int],
    *,
    old_consumer: bool = False,
    expect_stage: bool = True,
    settle_for_done: bool = True,
) -> dict:
    """B's recompute, then B's request that asks A by hash; the evidence of both.

    `expect_stage`: wait (up to SETTLE_SECONDS) for A to log `done` for a pin it
    `submitted`. Without it only a short wait for late lines.
    """
    servers = s.servers(a)
    addr = ensure_address(s, a)
    reference = pl.greedy(servers, s.b.url, {"prompt": prompt})
    if not pl.reset_cache(s.b.url):
        raise RuntimeError("B would not reset its prefix cache")
    s.b.ctl_set(drop_reply_budget=True if old_consumer else None)
    a_off, b_off = pl.log_size(a.log), pl.log_size(s.b.log)
    before = kv.metric_totals(s.b.url, kv.NIXL_METRICS)
    got = pl.greedy(
        servers,
        s.b.url,
        {"prompt": prompt, "kv_transfer_params": pl.peer_params(addr, None)},
    )
    moved = kv.metrics_change(s.b.url, before)
    time.sleep(kv.METRICS_WAIT_SECONDS)
    s.b.ctl_set(drop_reply_budget=None)

    deadline = time.monotonic() + SETTLE_SECONDS
    events = parse_events(pl.log_since(a.log, a_off), pl.log_since(s.b.log, b_off))
    while (
        settle_for_done
        and expect_stage
        and events["submitted"] is not None
        and events["done"] is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.5)
        events = parse_events(pl.log_since(a.log, a_off), pl.log_since(s.b.log, b_off))
    reply = events["reply"] or {}
    hit_tokens = reply.get("hit_tokens") or 0
    expected_bytes = hit_tokens * s.bytes_per_token
    return {
        "output_equals_recompute": got["tokens"] == reference["tokens"]
        or pl.is_near_tie(got, reference),
        "nixl_bytes": moved["bytes"],
        "nixl_bytes_over_expected": (moved["bytes"] / expected_bytes) if expected_bytes else None,
        "hit_tokens": hit_tokens,
        "hit_blocks": hit_tokens // s.block_size,
        "prompt_blocks": s.prompt_blocks,
        "events": events,
        "a_log_offset": a_off,
        "b_log_offset": b_off,
    }


def finish(s: Setup, a: Engine, record: dict) -> dict:
    """The scenario's last step: A's GPU prefix cache must reset.

    The first attempt is not retried, so that the record shows whether a copy
    still in flight (draining blocks) or a pin held the cache at that moment.
    """
    first_ok = pl.reset_cache(a.url, attempts=1)
    ok, seconds = (True, 0.0) if first_ok else reset(a.url)
    record["reset_ok"] = ok
    record["reset_first_attempt_ok"] = first_ok
    record["reset_seconds"] = seconds
    return record


# Checks -------------------------------------------------------------------------


def common(s: Setup, r: dict, *, hit: bool = True) -> dict:
    """The checks every scenario shares."""
    ev = r["events"]
    bytes_ok = r["nixl_bytes"] > 0 if hit else r["nixl_bytes"] == 0
    if hit and not s.metrics_known and r["hit_tokens"] > 0:
        bytes_ok = True  # no NIXL metric series on B at all: judged by the hit
    return {
        "output_equals_recompute": {"ok": r["output_equals_recompute"]},
        "nixl_bytes": {
            "ok": bytes_ok,
            "bytes": r["nixl_bytes"],
            "over_expected": r["nixl_bytes_over_expected"],
        },
        "no_crash_in_a_log": {"ok": not ev["a_crash"]},
        "a_prefix_cache_resets": {
            "ok": r["reset_ok"],
            "first_attempt_ok": r["reset_first_attempt_ok"],
            "seconds": r["reset_seconds"],
        },
    }


def check(ok: bool, **info) -> dict:
    return {"ok": bool(ok), **info}


def scenario_cpu_only(s: Setup) -> dict:
    a = s.a
    prompt, _ = warm(s, a)
    r_evict = evict_gpu_flood(s, a) if s.args.cpu_only_evict == "flood" else evict_gpu_reset(a)
    r = lookup(s, a, prompt)
    finish(s, a, r)
    ev, sub = r["events"], r["events"]["submitted"]
    budget = ev["reply_budget_s"]
    checks = common(s, r)
    checks.update(
        {
            "full_hit": check(r["hit_blocks"] == s.prompt_blocks, hit_blocks=r["hit_blocks"]),
            "staged_whole_prefix": check(
                sub is not None and sub["staged"] >= 0.9 * s.prompt_blocks and sub["in_place"] == 0,
                submitted=sub,
            ),
            "done_ok": check(ev["done"] is not None and ev["done"]["ok"], done=ev["done"]),
            "deadline_follows_budget": check(
                sub is not None
                and budget is not None
                and abs(sub["deadline_s"] - min(s.args.stage_deadline, budget - STAGE_MARGIN_S)) < 0.01,
                deadline_s=None if sub is None else sub["deadline_s"],
                reply_budget_s=budget,
            ),
            "not_trimmed": check(ev["trimmed"] is None, trimmed=ev["trimmed"]),
        }
    )
    return {"evict": r_evict, "record": r, "checks": checks}


def scenario_mixed(s: Setup) -> dict:
    a = s.a
    n = s.prompt_blocks
    gpu_head, cpu_mid = 16, 24
    prompt, key = warm(s, a)
    r_evict = evict_blocks(s, a, key, gpu_head, gpu_head + cpu_mid)
    r = lookup(s, a, prompt)
    finish(s, a, r)
    ev, sub = r["events"], r["events"]["submitted"]
    gpu_tail = n - gpu_head - cpu_mid
    chunk = s.args.chunk_blocks
    checks = common(s, r)
    checks.update(
        {
            "full_hit": check(r["hit_blocks"] == n, hit_blocks=r["hit_blocks"]),
            # One block per CPU chunk (the default): no hole inside a chunk, so
            # staged is exactly the CPU part. With larger chunks the first CPU
            # chunk may start before the hole and the GPU blocks of the last
            # one are copied too.
            "staged_equals_cpu_part": check(
                sub is not None and cpu_mid <= sub["staged"] <= cpu_mid + 2 * (chunk - 1),
                staged=None if sub is None else sub["staged"],
                expected=cpu_mid,
            ),
            "later_gpu_part_in_place": check(
                sub is not None and sub["in_place"] > 0 and sub["in_place"] <= gpu_tail,
                in_place=None if sub is None else sub["in_place"],
                expected=gpu_tail,
            ),
            "done_ok": check(ev["done"] is not None and ev["done"]["ok"], done=ev["done"]),
            "not_trimmed": check(ev["trimmed"] is None, trimmed=ev["trimmed"]),
        }
    )
    return {"evict": r_evict, "record": r, "checks": checks}


def partial_setup(s: Setup, a: Engine) -> tuple[list[int], dict, int]:
    """A prompt whose first 16 blocks are on A's GPU and the rest only on CPU."""
    gpu_run = 16
    prompt, key = warm(s, a)
    return prompt, evict_blocks(s, a, key, gpu_run, s.prompt_blocks), gpu_run


def scenario_deadline_trim(s: Setup) -> dict:
    a = s.a_deadline
    prompt, r_evict, gpu_run = partial_setup(s, a)
    a.ctl_set(slow_loads_s=s.args.slow_load_s)
    try:
        r = lookup(s, a, prompt)
        r["reset_while_draining"] = pl.reset_cache(a.url, attempts=1)
    finally:
        a.ctl_set(slow_loads_s=None)
    finish(s, a, r)
    ev, sub, reply = r["events"], r["events"]["submitted"], r["events"]["reply"] or {}
    checks = common(s, r)
    checks.update(
        {
            "staged_then_deadline": check(
                sub is not None and abs(sub["deadline_s"] - s.args.deadline) < 0.01,
                submitted=sub,
            ),
            "trimmed_deadline": check(
                ev["trimmed"] is not None and ev["trimmed"]["reason"] == "deadline",
                trimmed=ev["trimmed"],
            ),
            "partial_hit_is_gpu_run": check(
                r["hit_blocks"] == gpu_run
                and ev["trimmed"] is not None
                and ev["trimmed"]["kept_blocks"] == gpu_run,
                hit_blocks=r["hit_blocks"],
                expected=gpu_run,
            ),
            "b_sees_stage_timeout": check(reply.get("miss") == "stage_timeout", reply=reply),
            "late_result_after_reply": check(
                ev["done"] is not None and ev["done"]["seconds"] >= s.args.slow_load_s * 0.9,
                done=ev["done"],
            ),
        }
    )
    return {"evict": r_evict, "record": r, "checks": checks}


def scenario_load_failure(s: Setup) -> dict:
    a = s.a
    prompt, r_evict, gpu_run = partial_setup(s, a)
    a.ctl_set(fail_loads=True)
    try:
        r = lookup(s, a, prompt)
    finally:
        a.ctl_set(fail_loads=None)
    finish(s, a, r)
    ev, reply = r["events"], r["events"]["reply"] or {}
    checks = common(s, r)
    checks.update(
        {
            "failure_injected": check(ev["failed_load_jobs"] >= 1, jobs=ev["failed_load_jobs"]),
            "done_failed": check(
                ev["done"] is not None and not ev["done"]["ok"], done=ev["done"]
            ),
            "trimmed_failed": check(
                ev["trimmed"] is not None and ev["trimmed"]["reason"] == "failed",
                trimmed=ev["trimmed"],
            ),
            "partial_hit_is_gpu_run": check(
                r["hit_blocks"] == gpu_run, hit_blocks=r["hit_blocks"], expected=gpu_run
            ),
            "b_sees_stage_timeout": check(reply.get("miss") == "stage_timeout", reply=reply),
        }
    )
    return {"evict": r_evict, "record": r, "checks": checks}


def scenario_older_consumer(s: Setup) -> dict:
    out: dict = {}
    # (a) CPU-only prefix on A, B sends no reply_budget_s: A stages with its own
    # peer_stage_deadline. The cpu_only scenario's deadline is budget - margin.
    a = s.a
    prompt, _ = warm(s, a)
    evict_gpu_reset(a)
    ra = lookup(s, a, prompt, old_consumer=True)
    finish(s, a, ra)
    evb, sub = ra["events"], ra["events"]["submitted"]
    checks = {f"a_{k}": v for k, v in common(s, ra).items()}
    checks.update(
        {
            "a_full_hit": check(ra["hit_blocks"] == s.prompt_blocks, hit_blocks=ra["hit_blocks"]),
            "a_b_sent_no_budget": check(evb["reply_budget_s"] is None, budget=evb["reply_budget_s"]),
            "a_staged_with_default_deadline": check(
                sub is not None and sub["staged"] > 0 and abs(sub["deadline_s"] - s.args.stage_deadline) < 0.01,
                submitted=sub,
                expected_deadline_s=s.args.stage_deadline,
            ),
            "a_not_trimmed": check(evb["trimmed"] is None, trimmed=evb["trimmed"]),
        }
    )
    out["a"] = ra
    # (b) A_DL trims on the deadline; the old consumer gets pin_refused.
    ad = s.a_deadline
    prompt, r_evict, gpu_run = partial_setup(s, ad)
    ad.ctl_set(slow_loads_s=s.args.slow_load_s)
    try:
        rb = lookup(s, ad, prompt, old_consumer=True)
    finally:
        ad.ctl_set(slow_loads_s=None)
    finish(s, ad, rb)
    evd, reply = rb["events"], rb["events"]["reply"] or {}
    checks.update({f"b_{k}": v for k, v in common(s, rb).items()})
    checks.update(
        {
            "b_b_sent_no_budget": check(evd["reply_budget_s"] is None, budget=evd["reply_budget_s"]),
            "b_trimmed_deadline": check(
                evd["trimmed"] is not None and evd["trimmed"]["reason"] == "deadline",
                trimmed=evd["trimmed"],
            ),
            "b_partial_hit_is_gpu_run": check(
                rb["hit_blocks"] == gpu_run, hit_blocks=rb["hit_blocks"], expected=gpu_run
            ),
            "b_miss_is_pin_refused": check(reply.get("miss") == "pin_refused", reply=reply),
        }
    )
    out["b"] = rb
    out["evict_b"] = r_evict
    return {"record": out, "checks": checks}


def scenario_engine_idle(s: Setup) -> dict:
    a = s.a
    assert a.client is not None
    prompt, _ = warm(s, a)
    evict_gpu_reset(a)
    # Nothing is sent to A from here on, except the lookup B's request makes it
    # serve and the final reset (an engine call, not a request).
    time.sleep(s.args.idle_seconds)
    before = scrape(a.url)
    r = lookup(s, a, prompt)
    first_ok = pl.reset_cache(a.url, attempts=1)
    ok, seconds = (True, 0.0) if first_ok else reset(a.url, attempts=int(s.args.idle_release_seconds))
    r.update(
        {
            "reset_ok": ok,
            "reset_first_attempt_ok": first_ok,
            "reset_seconds": seconds,
            "a_running_before": before.get("vllm:num_requests_running"),
        }
    )
    ev, sub = r["events"], r["events"]["submitted"]
    checks = common(s, r)
    checks.update(
        {
            "a_idle_before_lookup": check(before.get("vllm:num_requests_running", 0) == 0),
            "reply_arrived_full_hit": check(
                r["hit_blocks"] == s.prompt_blocks and ev["b_hit_logged"], hit_blocks=r["hit_blocks"]
            ),
            "staged": check(sub is not None and sub["staged"] > 0, submitted=sub),
            "no_serve_timeout": check(not ev["a_not_shipped_warning"]),
            "pin_released_without_requests": check(
                ok and r["reset_seconds"] <= s.args.idle_release_seconds, seconds=r["reset_seconds"]
            ),
        }
    )
    return {"record": r, "checks": checks}


def scenario_skip_rule(s: Setup) -> dict:
    a = s.a_skip
    prompt, r_evict, gpu_run = partial_setup(s, a)
    r = lookup(s, a, prompt, expect_stage=False)
    finish(s, a, r)
    ev = r["events"]
    checks = common(s, r)
    checks.update(
        {
            "skipped_for_recompute": check(
                ev["skipped"] is not None and ev["skipped"]["reason"] == "recompute",
                skipped=ev["skipped"],
            ),
            "nothing_submitted": check(ev["submitted"] is None, submitted=ev["submitted"]),
            "gpu_part_only": check(
                r["hit_blocks"] == gpu_run, hit_blocks=r["hit_blocks"], expected=gpu_run
            ),
        }
    )
    return {"evict": r_evict, "record": r, "checks": checks}


SCENARIOS = {
    "cpu_only": scenario_cpu_only,
    "mixed": scenario_mixed,
    "deadline_trim": scenario_deadline_trim,
    "load_failure": scenario_load_failure,
    "older_consumer": scenario_older_consumer,
    "engine_idle": scenario_engine_idle,
    "skip_rule": scenario_skip_rule,
}


# Main -------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    for name in ("a", "a-deadline", "a-skip", "b"):
        p.add_argument(f"--{name}", nargs=3, metavar=("URL", "LOG", "CTL"), required=True)
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--prompt-blocks", type=int, default=64)
    p.add_argument("--stage-deadline", type=float, default=3.0,
                   help="peer_stage_deadline of A (the engine_idle and older-consumer arms)")  # fmt: skip
    p.add_argument("--deadline", type=float, default=0.3,
                   help="peer_stage_deadline of A_DL: above the pin server's own copy estimate "
                   "(2 steps + bytes / 20 GB/s), which skips a copy that cannot make it")  # fmt: skip
    p.add_argument("--slow-load-s", type=float, default=1.2,
                   help="how long A_DL's hook holds a load job's result back")  # fmt: skip
    p.add_argument("--chunk-blocks", type=int, default=1, help="blocks_per_chunk of the offload tier")
    p.add_argument("--cpu-only-evict", choices=("flood", "reset"), default="flood")
    p.add_argument("--idle-seconds", type=float, default=4.0)
    p.add_argument("--idle-release-seconds", type=float, default=30.0)
    p.add_argument("--bytes-per-token", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results.json"))
    return p.parse_args(argv)


def engine(name: str, spec: list[str], model: str) -> Engine:
    url, log, ctl = spec
    e = Engine(name, url, Path(log), Path(ctl))
    e.client = Client(url, model)
    return e


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    a = engine("A", args.a, args.model)
    a_dl = engine("A_DL", args.a_deadline, args.model)
    a_skip = engine("A_SKIP", args.a_skip, args.model)
    b = engine("B", args.b, args.model)
    for e in (a, a_dl, a_skip, b):
        kv.wait_ready(e.url)
        e.ctl_clear()
    rng = random.Random(args.seed)

    metrics = scrape(a.url)
    block_size = int(metrics.get("block_size", 16))
    pool_blocks = int(metrics.get("num_gpu_blocks", 0))
    if not pool_blocks:
        raise RuntimeError("no num_gpu_blocks in A's cache_config_info")
    per_token = calibrate(a.client, rng, block_size, args.bytes_per_token)
    print(f"A: block_size={block_size} pool_blocks={pool_blocks} bytes/token={per_token:.0f}", flush=True)
    s = Setup(
        model=args.model, a=a, a_deadline=a_dl, a_skip=a_skip, b=b, rng=rng,
        block_size=block_size, prompt_blocks=args.prompt_blocks, pool_blocks=pool_blocks,
        bytes_per_token=per_token, args=args,
    )  # fmt: skip
    s.metrics_known = has_series(b.url, "vllm:nixl_bytes_transferred")
    if not s.metrics_known:
        s.notes.append("B exports no vllm:nixl_bytes_transferred series; NIXL bytes judged by the hit")
        print("WARNING:", s.notes[-1], flush=True)

    results: dict = {}
    checks: dict = {}
    errors: dict = {}
    for name in args.scenarios.split(","):
        started = time.monotonic()
        try:
            out = SCENARIOS[name](s)
        except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as exc:
            errors[name] = repr(exc)
            print(f"ERROR {name}: {exc!r}", flush=True)
            for e in (a, a_dl, a_skip):
                e.ctl_clear()
            continue
        results[name] = out["record"] | ({"evict": out["evict"]} if "evict" in out else {})
        for check_name, result in out["checks"].items():
            checks[f"{name}.{check_name}"] = result
        failed = [k for k, v in out["checks"].items() if not v["ok"]]
        print(
            f"{'ok  ' if not failed else 'FAIL'} {name} ({time.monotonic() - started:.0f}s)"
            + (f": {failed}" if failed else ""),
            flush=True,
        )
        print(json.dumps({k: v for k, v in out["checks"].items() if not v["ok"]}, default=str), flush=True)

    args.output.write_text(
        json.dumps(
            {"model": args.model, "block_size": block_size, "pool_blocks": pool_blocks,
             "bytes_per_token": per_token, "notes": s.notes, "errors": errors,
             "checks": checks, "scenarios": results},
            indent=2, default=str,
        )
    )  # fmt: skip
    for e in (a, a_dl, a_skip, b):
        if e.log.exists():
            shutil.copy(e.log, args.output.parent / e.log.name)
    failed = [k for k, v in checks.items() if not v["ok"]]
    for k, v in checks.items():
        print(f"{'ok  ' if v['ok'] else 'FAIL'} {k}")
    if errors or failed:
        print(f"FAIL: errored scenarios {sorted(errors)}, failed checks {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
