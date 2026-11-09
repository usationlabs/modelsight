"""H5: does peer lookup-and-pin still beat stock when A is busy?

Run as the bench of examples/kv-transfer/h5.yaml, against the two vLLM servers of
H4. For each load level N, the probe starts AIPerf against A with N concurrent
agent sessions, waits until A is really running about N requests, and then
measures the same cases and arms as H4 while the load keeps running:

  recompute  B computes the prompt itself
  stock      A prefills (it queues behind its load), then B pulls; timed from the
             prefill request to A until B's first token
  peer       one request to B; B asks A by hash, A pins, B pulls

A cannot reset its prefix cache while it holds the load's blocks, so every arm
uses its own cache_salt: its blocks hash differently from every other arm's and
from the load's. Caches are reset between levels, when A is idle.

Level 0 runs no load and is H4. Writes results.json to the current directory and
exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Self

from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

DEFAULT_LEVELS = "0,8,16,32,64,128"
# A runs at most this many requests at once (h5.yaml passes it as
# --max-num-seqs). Background sessions beyond it wait in A's queue, which is the
# state H5 needs: stock's prefill request has to stand in that queue.
A_MAX_SEQS = 16
LOAD_TOKENS = 128_000
MAX_MODEL_LEN = 32_768
LOAD_OUTPUT_TOKENS = 256
LOAD_METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
}
# A level counts as loaded once A runs this fraction of N requests.
REACHED_FRACTION = 0.9
SETTLE_TIMEOUT_SECONDS = 180.0
DRAIN_TIMEOUT_SECONDS = 180.0
STOP_TIMEOUT_SECONDS = 120.0
SAMPLE_INTERVAL_SECONDS = 0.25
BYTES_TOLERANCE = h4.BYTES_TOLERANCE


# Load ------------------------------------------------------------------------


def parse_levels(text: str) -> list[int]:
    levels = [int(part) for part in text.split(",") if part.strip()]
    if not levels or any(n < 0 for n in levels):
        raise ValueError(f"bad load levels: {text!r}")
    return levels


def prompt_cap(level: int) -> int:
    """Longest prompt a background session may send at this level.

    The requests A runs at once (at most A_MAX_SEQS) share LOAD_TOKENS, and no
    prompt can outgrow the engine's context window. Sessions beyond A_MAX_SEQS
    wait without holding KV, so they do not shrink the cap.
    """
    running = min(level, A_MAX_SEQS)
    return min(LOAD_TOKENS // running, MAX_MODEL_LEN - LOAD_OUTPUT_TOKENS - 64)


def background_turns(render, messages: list[dict], cap: int) -> list[str]:
    """The user turns of a session whose cumulative prompt fits in `cap` tokens.

    AIPerf's multi_turn sessions hold only the user's new text per turn. It
    sends the history with each turn: the earlier user turns and the model's own
    replies, which are LOAD_OUTPUT_TOKENS long here. A turn's prompt is
    estimated as the chat template's token count over the system prompt and the
    user turns so far, plus those replies.
    """
    turns: list[str] = []
    system = ""
    users: list[dict] = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
            continue
        if message["role"] != "user":
            continue
        users.append(message)
        kept = [{"role": "system", "content": system}] if system else []
        kept += users
        if len(render(kept)) + LOAD_OUTPUT_TOKENS * (len(users) - 1) > cap:
            break
        text = message["content"]
        turns.append(f"{system}\n\n{text}" if system and not turns else text)
    return turns


def write_background(
    path: Path, sessions: list[list[dict]], render, level: int
) -> int:
    """JSONL of AIPerf multi_turn sessions for this level; returns how many."""
    cap = prompt_cap(level)
    written = 0
    with path.open("w") as f:
        for n, messages in enumerate(sessions):
            turns = background_turns(render, messages, cap)
            if not turns:
                continue
            f.write(json.dumps({
                "session_id": f"bg-{level}-{n}",
                "turns": [{"text": t} for t in turns],
            }) + "\n")  # fmt: skip
            written += 1
    return written


def aiperf_command(args: argparse.Namespace, level: int, dataset: Path,
                   artifacts: Path) -> list[str]:
    """One AIPerf run against A: `level` concurrent sessions, replayed in turns."""
    return [
        *args.aiperf_command.split(),
        "profile",
        "--model", args.model,
        "--tokenizer", args.model,
        "--url", args.producer,
        "--endpoint-type", "chat",
        "--streaming",
        "--use-server-token-count",
        "--custom-dataset-type", "multi_turn",
        "--input-file", str(dataset),
        "--dataset-sampling-strategy", "shuffle",
        "--concurrency", str(level),
        "--concurrency-ramp-duration", str(args.ramp_seconds),
        "--benchmark-duration", str(args.level_seconds),
        "--benchmark-grace-period", "30",
        "--osl", str(LOAD_OUTPUT_TOKENS),
        "--osl-stddev", "0",
        "--extra-inputs",
        json.dumps({"ignore_eos": True, "min_tokens": LOAD_OUTPUT_TOKENS}),
        "--artifact-dir", str(artifacts),
        "--ui-type", "none",
    ]  # fmt: skip


def load_state(url: str) -> dict[str, float]:
    return kv.metric_totals(url, LOAD_METRICS)


class Sampler:
    """Samples A's running and waiting requests so each arm knows its load."""

    def __init__(self, url: str, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self.url = url
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = load_state(self.url)
                self.samples.append(
                    (time.monotonic(), state["running"], state["waiting"])
                )
            except OSError:
                pass
            self._stop.wait(self.interval)

    def window(self, start: float, end: float) -> dict:
        seen = [s for s in list(self.samples) if s[0] <= end]
        rows = [s for s in seen if s[0] >= start]
        if not rows and seen:
            # An arm can be shorter than the sampling interval; the load was
            # whatever the last sample before it said.
            rows = seen[-1:]
        if not rows:
            return {"running": None, "waiting": None, "samples": 0}
        return {
            "running": statistics.median(r[1] for r in rows),
            "waiting": statistics.median(r[2] for r in rows),
            "samples": len(rows),
        }


class Load:
    """AIPerf against A for one level, started and stopped by the probe."""

    def __init__(self, command: list[str], log: Path) -> None:
        self.command = command
        self.log = log
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("w")
        try:
            self.proc = subprocess.Popen(
                self.command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            handle.close()
        return self

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def __exit__(self, *exc) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        # SIGINT makes AIPerf stop and write its summary.
        for sig, wait in ((signal.SIGINT, STOP_TIMEOUT_SECONDS),
                          (signal.SIGKILL, 30.0)):  # fmt: skip
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue


def wait_loaded(url: str, level: int, load: Load | None,
                timeout: float = SETTLE_TIMEOUT_SECONDS) -> dict:
    """Wait until A runs about `level` requests (at most A_MAX_SEQS)."""
    target = max(1.0, REACHED_FRACTION * min(level, A_MAX_SEQS))
    deadline = time.monotonic() + timeout
    state = load_state(url)
    while state["running"] < target and time.monotonic() < deadline:
        if load is not None and not load.alive():
            break
        time.sleep(0.5)
        state = load_state(url)
    return {"reached": state["running"] >= target, "running": state["running"],
            "waiting": state["waiting"], "target": target}  # fmt: skip


def wait_idle(url: str, timeout: float = DRAIN_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = load_state(url)
        if state["running"] == 0 and state["waiting"] == 0:
            return True
        time.sleep(0.5)
    return False


# One case under load ----------------------------------------------------------


def run_case(
    args: argparse.Namespace,
    servers: Servers,
    sampler: Sampler | None,
    level: int,
    nonce: str,
    case: str,
    previous: list[int],
    current: list[int],
    producer: dict,
) -> dict:
    """All arms of one case, each under its own cache_salt, with the load on."""
    held = {"full": current, "partial": previous, "miss": []}[case]
    record: dict = {
        "level": level,
        "case": case,
        "prompt_tokens": len(current),
        "held_tokens": len(held),
        "expected_hit_tokens": h4.expected_hit(held, current, args.block_size),
    }

    def window(started: float) -> dict | None:
        return sampler.window(started, time.monotonic()) if sampler else None

    def salt(arm: str) -> str:
        return f"{nonce}-{level}-{uuid.uuid4().hex[:8]}-{arm}"

    def warm(cache_salt: str) -> None:
        if held:
            kv.timed_stream(servers, servers.producer,
                            {"prompt": held, "cache_salt": cache_salt})  # fmt: skip

    started = time.monotonic()
    record["recompute"] = h4.greedy(
        servers, servers.consumer, {"prompt": current, "cache_salt": salt("rc")}
    )
    record["recompute"]["a_load"] = window(started)

    if case != "miss":
        cache_salt = salt("st")
        warm(cache_salt)
        started = time.monotonic()
        tick = time.perf_counter()
        params = kv.prefill(servers, current, cache_salt=cache_salt)
        prefill_s = time.perf_counter() - tick
        before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
        stock = h4.greedy(
            servers,
            servers.consumer,
            {"prompt": current, "cache_salt": cache_salt, "kv_transfer_params": params},
        )
        stock["b_ttft_s"] = stock["ttft_s"]
        stock["prefill_s"] = prefill_s
        stock["ttft_s"] = prefill_s + stock["b_ttft_s"]
        stock["total_s"] += prefill_s
        stock["transfer"] = kv.metrics_change(servers.consumer, before)
        stock["pulled_tokens"] = h4.expected_hit(current, current, args.block_size)
        stock["a_load"] = window(started)
        record["stock"] = stock

    cache_salt = salt("pe")
    warm(cache_salt)
    time.sleep(kv.METRICS_WAIT_SECONDS)  # let the warm-up's finish land in A's log
    request_id = f"h5-{nonce}-{level}-{uuid.uuid4().hex[:8]}"
    a_log = h4.log_size(args.producer_log)
    before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
    started = time.monotonic()
    body = {"prompt": current, "cache_salt": cache_salt,
            "kv_transfer_params": h4.peer_params(producer, args.identity)}  # fmt: skip
    peer = h4.greedy(
        servers, servers.consumer, body, headers={"X-Request-Id": request_id}
    )
    peer["a_load"] = window(started)
    peer["transfer"] = kv.metrics_change(servers.consumer, before)
    time.sleep(kv.METRICS_WAIT_SECONDS)
    a_text = h4.log_since(args.producer_log, a_log)
    # A computed nothing for the peer arm if no request with its id ever ran there.
    peer["request_id"] = request_id
    peer["a_served"] = a_text.count(request_id)
    peer["lookups_logged"] = len(
        re.findall(args.lookup_pattern, a_text, re.IGNORECASE)
    )
    record["peer"] = peer
    return record


def mark_evictions(records: list[dict]) -> int:
    """Flag peer arms that pulled less than A should have held, under load.

    A may have evicted the arm's own prefix between the warm-up and the lookup.
    Those arms are counted and reported, not failed: the load is capped so this
    stays rare, and a hit that is too big is still a failure.
    """
    rate = h4.bytes_per_token(records)
    if rate is None:
        return 0
    evicted = 0
    for r in records:
        expected = r["expected_hit_tokens"] * rate
        pulled = r["peer"]["transfer"]["bytes"]
        short = expected > 0 and pulled < expected * (1 - BYTES_TOLERANCE)
        if r["level"] > 0 and short:
            r["peer"]["evicted"] = True
            evicted += 1
    return evicted


# Levels ---------------------------------------------------------------------


def level_summary(records: list[dict]) -> dict:
    """Latency and the load the arms really saw, for one level."""
    loads = [
        r[arm]["a_load"]
        for r in records
        for arm in ("stock", "peer")
        if arm in r and r[arm].get("a_load") and r[arm]["a_load"]["samples"]
    ]
    return {
        "latency": h4.latency(records),
        "a_running": statistics.median(x["running"] for x in loads) if loads else None,
        "a_waiting": statistics.median(x["waiting"] for x in loads) if loads else None,
        "evicted": sum(bool(r["peer"].get("evicted")) for r in records),
    }


def run_level(
    args: argparse.Namespace,
    servers: Servers,
    level: int,
    nonce: str,
    measured: list[tuple[list[int], list[int]]],
    background: list[list[dict]],
    render,
    producer: dict,
    out_dir: Path,
) -> dict:
    """Measure every case under `level` background sessions on A."""
    records: list[dict] = []
    info: dict = {"level": level}
    load: Load | None = None
    command = None
    if level > 0:
        dataset = out_dir / f"background-{level}.jsonl"
        info["background_sessions"] = write_background(dataset, background, render, level)
        info["prompt_cap_tokens"] = prompt_cap(level)
        if not info["background_sessions"]:
            raise RuntimeError(f"no background session fits {prompt_cap(level)} tokens")
        if info["background_sessions"] < level:
            print(f"warning: level {level} has only {info['background_sessions']} "
                  "background sessions; AIPerf will repeat them", flush=True)  # fmt: skip
        command = aiperf_command(args, level, dataset, out_dir / f"aiperf-{level}")
        load = Load(command, out_dir / f"aiperf-{level}.log")
    started_all = time.monotonic()
    with (
        (load if load is not None else _NoLoad()) as active,
        Sampler(servers.producer) as sampler,
    ):
        if load is not None:
            info["settle"] = wait_loaded(servers.producer, level, active)
            print(json.dumps({"level": level, **info["settle"]}), flush=True)
            if level > A_MAX_SEQS and not info["settle"]["waiting"]:
                print(f"warning: level {level} has more sessions than A runs at once "
                      f"({A_MAX_SEQS}), but A has no queue", flush=True)  # fmt: skip
        for n, (previous, current) in enumerate(measured):
            for case in h4.CASES:
                try:
                    record = run_case(args, servers, sampler, level, nonce, case,
                                      previous, current, producer)  # fmt: skip
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    record = {"level": level, "case": case, "error": repr(exc)}
                record["session"] = n
                records.append(record)
                shown = ("level", "session", "case", "prompt_tokens",
                         "expected_hit_tokens", "error")  # fmt: skip
                print(json.dumps({k: v for k, v in record.items() if k in shown}),
                      flush=True)  # fmt: skip
    info["seconds"] = time.monotonic() - started_all
    info["drained"] = wait_idle(servers.producer)
    info["leak_free"] = all(h4.reset_cache(url)
                            for url in (servers.producer, servers.consumer))  # fmt: skip
    return {"info": info, "records": records}


class _NoLoad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> None:
        return None


# Main -----------------------------------------------------------------------


def pick_sessions(args: argparse.Namespace, render):
    """Measured turn pairs, and the other sessions for the background."""
    pool = h4.load_sessions(
        args.traces, max(args.sessions * 12, args.background_pool), args.seed
    )
    measured, background = [], []
    for messages in pool:
        pair = None
        if len(measured) < args.sessions:
            pair = h4.pick_turn(h4.turn_prompts(render, messages), args.max_prompt_tokens)
        if pair is not None:
            measured.append(pair)
        else:
            background.append(messages)
    return measured, background


def print_summary(levels: dict[int, dict], checks: dict) -> None:
    print("level case     recompute_s  stock_s  peer_s  stock_b_s  A run/wait  evicted")
    for level, summary in levels.items():
        running, waiting = summary["a_running"], summary["a_waiting"]
        load = "-" if running is None else f"{running:.0f}/{waiting:.0f}"
        for case, arms in summary["latency"].items():
            cells = [f"{arms[a]:.3f}" if a in arms else "-"
                     for a in ("recompute", "stock", "peer", "stock_b")]  # fmt: skip
            print(f"{level:>5} {case:<8} {cells[0]:>11}  {cells[1]:>7}  {cells[2]:>6}"
                  f"  {cells[3]:>9}  {load:>10}  {summary['evicted']:>7}")
    for name, result in checks.items():
        if isinstance(result, dict):
            state = {True: "PASS", False: "FAIL", None: "skip"}[result["ok"]]
            print(f"{state}  {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="A's base URL")
    parser.add_argument("--consumer", required=True, help="B's base URL")
    parser.add_argument("--producer-log", type=Path, required=True)
    parser.add_argument("--consumer-log", type=Path, required=True)
    parser.add_argument("--levels", default=DEFAULT_LEVELS,
                        help="background sessions on A, one level after another")  # fmt: skip
    parser.add_argument("--traces", default=h4.DEFAULT_DATASET)
    parser.add_argument("--sessions", type=int, default=10,
                        help="measured sessions per level")  # fmt: skip
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--background-pool", type=int, default=300,
                        help="agent sessions to draw the load from; the load "
                             "needs several per concurrent session")  # fmt: skip
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--identity", type=json.loads, default=None)
    parser.add_argument("--lookup-pattern", default=h4.LOOKUP_PATTERN)
    parser.add_argument("--aiperf-command", default="aiperf")
    parser.add_argument("--ramp-seconds", type=int, default=20)
    parser.add_argument("--level-seconds", type=int, default=1800,
                        help="AIPerf's benchmark duration; the probe stops it sooner")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    levels = parse_levels(args.levels)
    servers = Servers(args.model, args.producer, args.consumer)
    kv.wait_ready(servers.producer)
    kv.wait_ready(servers.consumer)

    render = h4.chat_renderer(args.model)
    measured, background = pick_sessions(args, render)
    if not measured:
        print("FAIL: no session has a usable turn pair", flush=True)
        return 1
    # A's side-channel address, from one stock prefill that B then pulls, so A
    # frees those blocks now. This runs before any load.
    address_prompt = kv.fresh_prompt(random.Random(args.seed), 64)
    producer = kv.prefill(servers, address_prompt)
    kv.timed_stream(servers, servers.consumer,
                    {"prompt": address_prompt, "kv_transfer_params": producer})  # fmt: skip

    out_dir = args.output.parent
    nonce = uuid.uuid4().hex[:6]
    results: dict[int, dict] = {}
    records: list[dict] = []
    for level in levels:
        outcome = run_level(args, servers, level, nonce, measured, background,
                            render, producer, out_dir)  # fmt: skip
        results[level] = outcome
        records += outcome["records"]

    good = [r for r in records if "error" not in r]
    evicted = mark_evictions(good)
    leak_free = all(o["info"]["leak_free"] for o in results.values())
    checks = h4.check(good, "peer", leak_free)
    summaries = {
        level: level_summary([r for r in good if r["level"] == level])
        for level in levels
    }
    path = (kv.read_path(args.consumer_log.read_text(errors="replace"))
            if args.consumer_log.exists() else [])  # fmt: skip
    args.output.write_text(json.dumps(
        {"model": args.model, "levels": levels, "sessions": len(measured),
         "background_pool": len(background), "evicted": evicted,
         "checks": checks, "summaries": summaries,
         "level_info": {n: o["info"] for n, o in results.items()},
         "pull_rates": h4.pull_rates(good), "read_path": path,
         "records": records}, indent=2))  # fmt: skip
    for log in (args.producer_log, args.consumer_log):
        if log.exists():
            shutil.copy(log, out_dir / log.name)

    print_summary(summaries, checks)
    print(f"evicted peer arms (reported, not failed): {evicted}")
    errors = len(records) - len(good)
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v["ok"] is False]
    unloaded = [n for n, o in results.items()
                if n and not o["info"].get("settle", {}).get("reached")]  # fmt: skip
    if unloaded:
        print(f"warning: A never reached the target load at levels {unloaded}", flush=True)
    if errors or failed or not good:
        print(f"FAIL: {errors} errored records, failed checks: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
