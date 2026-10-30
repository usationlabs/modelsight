"""H1: when does fetching KV blocks beat recomputing a prefix on another GPU?

Run as the bench of an experiment with a vLLM producer and consumer, each using
NixlConnector. See examples/kv-transfer/h1.yaml. Arms, all with fresh random token
prompts so no cache helps by accident:

  recompute  the consumer computes the whole prefix
  warm       the producer computes it, then serves it again from its cache
  fetch      the producer computes it, then the consumer reads the blocks

Both KV pools are overrun first, because a long-running server's blocks are
scattered, which makes transfers slower. Every fetch records vLLM's NIXL metrics:
bytes, descriptors, post time and transfer time. NIXL stops the transfer clock
when it is polled, so transfer time is an upper bound.

Writes results.json to the current directory, which is the run folder, and exits
with status 1 if the transfers cannot be trusted.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PREFIX_LENGTHS = "512,1000,2000,4000,8000,16000,32000"
READY_TIMEOUT_SECONDS = 15 * 60
REQUEST_TIMEOUT_SECONDS = 5 * 60
METRICS_WAIT_SECONDS = 2.0
OUTPUT_TOKENS = 8
FILL_FACTOR = 1.2
FILL_PROMPT_LENGTHS = (2000, 32000)
FILL_LIMIT_SECONDS = 5 * 60
# Ordinary token ids, clear of the special tokens at the end of a vocabulary.
TOKEN_IDS = range(1000, 100000)
ARMS = ("recompute", "warm", "fetch")

NIXL_METRICS = {
    "vllm:nixl_bytes_transferred_count": "transfers",
    "vllm:nixl_bytes_transferred_sum": "bytes",
    "vllm:nixl_num_descriptors_sum": "descriptors",
    "vllm:nixl_post_time_seconds_sum": "post_s",
    "vllm:nixl_xfer_time_seconds_sum": "xfer_s",
    "vllm:nixl_num_failed_transfers_total": "failed",
}


@dataclass(frozen=True)
class Servers:
    model: str
    producer: str
    consumer: str


def wait_ready(url: str, timeout: float = READY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2):
                return
        except (OSError, urllib.error.URLError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{url} did not become ready: {exc}") from exc
            time.sleep(1)


def post(servers: Servers, url: str, body: dict):
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps({"model": servers.model, "temperature": 0, **body}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)


def timed_stream(servers: Servers, url: str, body: dict) -> dict:
    """Time to first token and to the end of a short streamed completion."""
    started = time.perf_counter()
    first_token = None
    stream = {**body, "stream": True, "max_tokens": OUTPUT_TOKENS}
    with post(servers, url, stream) as response:
        for line in response:
            if first_token is None and line.startswith(b"data:"):
                first_token = time.perf_counter() - started
    if first_token is None:
        raise RuntimeError("the stream ended without a token")
    return {"ttft_s": first_token, "total_s": time.perf_counter() - started}


def fresh_prompt(rng: random.Random, length: int) -> list[int]:
    """Exactly `length` tokens that no server has seen, sent as token ids."""
    return [rng.choice(TOKEN_IDS) for _ in range(length)]


def metric_totals(url: str, names: dict[str, str]) -> dict[str, float]:
    """Sum the named Prometheus series across their labels."""
    totals = dict.fromkeys(names.values(), 0.0)
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        for line in response.read().decode().splitlines():
            name = line.split("{")[0].split(" ")[0]
            if name in names:
                totals[names[name]] += float(line.rsplit(" ", 1)[1])
    return totals


def metrics_change(url: str, before: dict[str, float]) -> dict[str, float]:
    """What one fetch added to the metrics, which reach /metrics a little later."""
    deadline = time.monotonic() + METRICS_WAIT_SECONDS
    while True:
        after = metric_totals(url, NIXL_METRICS)
        if after["transfers"] > before["transfers"] or time.monotonic() > deadline:
            return {name: after[name] - before[name] for name in after}
        time.sleep(0.05)


def recompute(servers: Servers, prompt: list[int]) -> dict:
    return timed_stream(servers, servers.consumer, {"prompt": prompt})


def warm(servers: Servers, prompt: list[int]) -> dict:
    hits = {"vllm:prefix_cache_hits_total": "hits"}
    timed_stream(servers, servers.producer, {"prompt": prompt})
    before = metric_totals(servers.producer, hits)["hits"]
    timing = timed_stream(servers, servers.producer, {"prompt": prompt})
    after = metric_totals(servers.producer, hits)["hits"]
    return {**timing, "hit_fraction": (after - before) / len(prompt)}


def prefill(servers: Servers, prompt: list[int]) -> dict:
    """Have the producer compute a prompt and hold its blocks for a reader."""
    body = {
        "prompt": prompt,
        "stream": False,
        "max_tokens": 1,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        },
    }
    with post(servers, servers.producer, body) as response:
        params = json.load(response).get("kv_transfer_params")
    if not params:
        raise RuntimeError("the producer returned no kv_transfer_params")
    return params


def fetch(servers: Servers, prompt: list[int]) -> dict:
    params = prefill(servers, prompt)
    before = metric_totals(servers.consumer, NIXL_METRICS)
    timing = timed_stream(
        servers, servers.consumer, {"prompt": prompt, "kv_transfer_params": params}
    )
    return {**timing, "transfer": metrics_change(servers.consumer, before)}


def pool_tokens(log: Path) -> int:
    """The KV pool's size, which vLLM logs once at startup."""
    text = log.read_text(errors="replace")
    match = re.search(r"KV cache size: ([\d,]+) tokens", text)
    if not match:
        raise RuntimeError(f"no KV cache size in {log}")
    return int(match.group(1).replace(",", ""))


def fill_pool(
    servers: Servers, url: str, pool: int, rng: random.Random, limit: float = FILL_LIMIT_SECONDS
) -> dict:
    """Overrun a pool with mixed lengths, so that its free blocks end up scattered."""
    target = int(pool * FILL_FACTOR)
    started = time.monotonic()
    sent = 0
    while sent < target and time.monotonic() - started < limit:
        length = rng.randrange(*FILL_PROMPT_LENGTHS)
        body = {"prompt": fresh_prompt(rng, length), "max_tokens": 1, "stream": False}
        with post(servers, url, body) as response:
            response.read()
        sent += length
    return {"pool_tokens": pool, "sent_tokens": sent, "seconds": time.monotonic() - started}


def show(record: dict) -> None:
    moved = record.get("transfer", {})
    detail = record["error"] or f"ttft={record.get('ttft_s', 0):.3f}s"
    if moved:
        detail += f" descriptors={moved['descriptors']:.0f} xfer={moved['xfer_s'] * 1e3:.1f}ms"
    print(f"{record['arm']} {record['prefix_len']} #{record['repeat']}: {detail}", flush=True)


def measure(
    servers: Servers,
    prefix_lengths: tuple[int, ...],
    repeats: int,
    seed: int,
    logs: tuple[Path, Path],
) -> tuple[list[dict], dict]:
    """Warm up, fill both pools, then measure every arm in a shuffled order.

    The shuffle stops length from being mixed up with how full the pools are.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    arms = {"recompute": recompute, "warm": warm, "fetch": fetch}

    def run(name: str, arm, length: int, repeat: int) -> None:
        record = {"arm": name, "prefix_len": length, "repeat": repeat, "error": None}
        try:
            record.update(arm(servers, fresh_prompt(rng, length)))
        except (OSError, RuntimeError, ValueError) as exc:
            record["error"] = repr(exc)
        records.append(record)
        show(record)

    run("warmup_recompute", recompute, prefix_lengths[0], 0)
    run("warmup_warm", warm, prefix_lengths[0], 0)
    for repeat in range(3):
        run("warmup_fetch", fetch, prefix_lengths[0], repeat)

    fill = {}
    for name, url, log in (
        ("producer", servers.producer, logs[0]),
        ("consumer", servers.consumer, logs[1]),
    ):
        fill[name] = fill_pool(servers, url, pool_tokens(log), rng)
        print(f"filled the {name}: {fill[name]}", flush=True)

    order = [
        (name, length, repeat)
        for name in ARMS
        for length in prefix_lengths
        for repeat in range(repeats)
    ]
    rng.shuffle(order)
    for name, length, repeat in order:
        run(name, arms[name], length, repeat)
    return records, fill


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def transfer_summary(records: list[dict]) -> dict:
    moved = [r["transfer"] for r in records if r.get("transfer", {}).get("xfer_s")]
    if not moved:
        return {}
    return {
        "mb": statistics.median(m["bytes"] / 1e6 for m in moved),
        "descriptors": statistics.median(m["descriptors"] for m in moved),
        "descriptors_max": max(m["descriptors"] for m in moved),
        "post_ms": statistics.median(m["post_s"] * 1e3 for m in moved),
        "xfer_ms": statistics.median(m["xfer_s"] * 1e3 for m in moved),
        "gb_per_s": statistics.median(m["bytes"] / 1e9 / m["xfer_s"] for m in moved),
    }


def summarize(records: list[dict], prefix_lengths: tuple[int, ...]) -> dict:
    """p50 and p95 per arm and length, the transfers, and the crossover."""
    by_length: dict[int, dict] = {}
    for length in prefix_lengths:
        at_length = [r for r in records if r["prefix_len"] == length]
        row: dict = {}
        for arm in ARMS:
            ttfts = [r["ttft_s"] for r in at_length if r["arm"] == arm and not r["error"]]
            if ttfts:
                row[arm] = {
                    "p50_s": statistics.median(ttfts),
                    "p95_s": percentile(ttfts, 0.95),
                    "n": len(ttfts),
                }
        row["transfer"] = transfer_summary([r for r in at_length if r["arm"] == "fetch"])
        hits = [r["hit_fraction"] for r in at_length if "hit_fraction" in r]
        row["warm_hit_fraction"] = statistics.median(hits) if hits else None
        by_length[length] = row

    def crossover(stat: str) -> int | None:
        return next(
            (
                length
                for length, row in by_length.items()
                if "fetch" in row
                and "recompute" in row
                and row["fetch"][stat] < row["recompute"][stat]
            ),
            None,
        )

    return {
        "by_length": by_length,
        "crossover_prefix_len": crossover("p50_s"),
        "crossover_prefix_len_p95": crossover("p95_s"),
        "first_transfers": [r.get("transfer") for r in records if r["arm"] == "warmup_fetch"],
        "errors": [r for r in records if r["error"]],
    }


def read_path(log: str) -> list[str]:
    """What UCX selected for remote reads, from its UCX_PROTO_INFO tables.

    The whole log is no use here: control messages take other transports, such
    as sysv, than the data does.
    """
    rows = []
    in_read_table = False
    for line in log.splitlines():
        if "|" not in line:
            continue
        cell = line[line.index("|") :]
        if "ucp_" in line:
            in_read_table = "remote memory read" in line
            if in_read_table:
                rows.append(cell)
        elif in_read_table and ".." in cell:
            rows.append(cell)
    return rows


def verdict(summary: dict, path: list[str]) -> str:
    if not any(row["transfer"] for row in summary["by_length"].values()):
        return "FAIL: no transfer showed up in vLLM's metrics"
    errors = len(summary["errors"])
    if errors:
        return f"FAIL: {errors} measurements failed"
    if not path:
        return "UNCONFIRMED: transfers ran, but UCX's read table is not in the log"
    if not any("cuda_ipc" in row for row in path):
        return "FAIL: UCX did not select cuda_ipc for reads"
    return f"PASS: fetch beats recompute from {summary['crossover_prefix_len']} tokens"


def print_table(summary: dict) -> None:
    print(
        "tokens  recompute  warm    fetch   fetch_p95 hit |    MB  descriptors     max"
        "  post_ms  xfer_ms   GB/s"
    )
    for length, row in summary["by_length"].items():
        arms = [f"{row[arm]['p50_s']:.3f}" if arm in row else "-" for arm in ARMS]
        tail = f"{row['fetch']['p95_s']:.3f}" if "fetch" in row else "-"
        moved = row["transfer"]
        print(
            f"{length:>6}  {arms[0]:>9}  {arms[1]:>6}  {arms[2]:>6}  {tail:>9} "
            f"{row['warm_hit_fraction'] or 0:>3.1f} | "
            f"{moved.get('mb', 0):>5.0f}  {moved.get('descriptors', 0):>11.0f}  "
            f"{moved.get('descriptors_max', 0):>6.0f}  "
            f"{moved.get('post_ms', 0):>7.2f}  {moved.get('xfer_ms', 0):>7.2f}  "
            f"{moved.get('gb_per_s', 0):>5.1f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="producer base URL")
    parser.add_argument("--consumer", required=True, help="consumer base URL")
    parser.add_argument("--producer-log", type=Path, required=True)
    parser.add_argument("--consumer-log", type=Path, required=True)
    parser.add_argument("--prefix-lengths", default=DEFAULT_PREFIX_LENGTHS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    servers = Servers(args.model, args.producer, args.consumer)
    lengths = tuple(int(length) for length in args.prefix_lengths.split(","))
    logs = (args.producer_log, args.consumer_log)

    wait_ready(servers.producer)
    wait_ready(servers.consumer)
    records, fill = measure(servers, lengths, args.repeats, args.seed, logs)

    summary = summarize(records, lengths)
    path = read_path(args.consumer_log.read_text(errors="replace"))
    result = verdict(summary, path)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "fill": fill,
                "read_path": path,
                "summary": summary,
                "records": records,
                "verdict": result,
            },
            indent=2,
        )
    )
    # The saved log keeps the UCX tables that the stream leaves out.
    for log in logs:
        shutil.copy(log, args.output.parent / log.name)

    print_table(summary)
    print(
        f"crossover: p50 {summary['crossover_prefix_len']}, "
        f"p95 {summary['crossover_prefix_len_p95']}"
    )
    print(result, flush=True)
    return 0 if result.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
