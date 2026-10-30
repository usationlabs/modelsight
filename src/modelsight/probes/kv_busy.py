"""H2: can a consumer still fetch KV blocks quickly while the producer is busy?

Run as the bench of examples/kv-transfer/h2.yaml. For each prefix P1, compares:

  recompute   the consumer computes P1
  fetch_idle  the consumer reads P1's blocks from an idle producer
  fetch_busy  the consumer reads them while the producer computes a fixed prompt P2
  wait        P1 goes back to the producer while it computes P2

P2 has one size throughout, so load does not change with prefix length. Writes
results.json to the current directory and exits with status 1 if the transfers
cannot be trusted.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

DEFAULT_PREFIX_LENGTHS = "8000,16000,32000,64000,128000"
ARMS = ("recompute", "fetch_idle", "fetch_busy", "wait")
RUNNING = {"vllm:num_requests_running": "running"}
BUSY_TIMEOUT_SECONDS = 30.0


def producer_running(servers: Servers) -> float:
    return kv.metric_totals(servers.producer, RUNNING)["running"]


def wait_running(servers: Servers) -> None:
    deadline = time.monotonic() + BUSY_TIMEOUT_SECONDS
    while producer_running(servers) < 1:
        if time.monotonic() > deadline:
            raise RuntimeError("the producer never started the load prompt")
        time.sleep(0.02)


def while_busy(servers: Servers, load: list[int], measure: Callable[[], dict]) -> dict:
    """Measure while the producer computes `load`, then wait for it to finish."""
    failure: list[BaseException] = []

    def compute() -> None:
        body = {"prompt": load, "max_tokens": 1, "stream": False}
        try:
            with kv.post(servers, servers.producer, body) as response:
                response.read()
        except OSError as exc:
            failure.append(exc)

    thread = threading.Thread(target=compute)
    thread.start()
    try:
        wait_running(servers)
        started = time.monotonic()
        result = measure()
        finished = time.monotonic() - started
        return {**result, "measure_s": finished, "busy_after": producer_running(servers) >= 1}
    finally:
        thread.join()
        if failure:
            raise RuntimeError(f"the load prompt failed: {failure[0]!r}")


def recompute(servers: Servers, prompt: list[int], load: list[int]) -> dict:
    return kv.recompute(servers, prompt)


def fetch_idle(servers: Servers, prompt: list[int], load: list[int]) -> dict:
    return kv.fetch(servers, prompt)


def fetch_busy(servers: Servers, prompt: list[int], load: list[int]) -> dict:
    params = kv.prefill(servers, prompt)

    def read() -> dict:
        before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
        body = {"prompt": prompt, "kv_transfer_params": params}
        timing = kv.timed_stream(servers, servers.consumer, body)
        return {**timing, "transfer": kv.metrics_change(servers.consumer, before)}

    return while_busy(servers, load, read)


def wait(servers: Servers, prompt: list[int], load: list[int]) -> dict:
    kv.timed_stream(servers, servers.producer, {"prompt": prompt})
    return while_busy(
        servers,
        load,
        lambda: kv.timed_stream(servers, servers.producer, {"prompt": prompt}),
    )


def measure(
    servers: Servers, lengths: tuple[int, ...], repeats: int, seed: int, load_length: int
) -> list[dict]:
    """Every arm, length and repeat in a shuffled order."""
    rng = random.Random(seed)
    arms = {
        "recompute": recompute,
        "fetch_idle": fetch_idle,
        "fetch_busy": fetch_busy,
        "wait": wait,
    }
    order = [(arm, length, repeat) for arm in ARMS for length in lengths for repeat in range(repeats)]
    rng.shuffle(order)
    records = []
    for arm, length, repeat in order:
        record = {"arm": arm, "prefix_len": length, "repeat": repeat, "error": None}
        try:
            prompt = kv.fresh_prompt(rng, length)
            record.update(arms[arm](servers, prompt, kv.fresh_prompt(rng, load_length)))
        except (OSError, RuntimeError, ValueError) as exc:
            record["error"] = repr(exc)
        records.append(record)
        kv.show(record)
    return records


def summarize(records: list[dict], lengths: tuple[int, ...]) -> dict:
    by_length: dict[int, dict] = {}
    for length in lengths:
        at_length = [r for r in records if r["prefix_len"] == length]
        row: dict = {}
        for arm in ARMS:
            ttfts = [r["ttft_s"] for r in at_length if r["arm"] == arm and not r["error"]]
            if ttfts:
                row[arm] = {
                    "p50_s": statistics.median(ttfts),
                    "p95_s": kv.percentile(ttfts, 0.95),
                    "n": len(ttfts),
                }
        row["transfer_busy"] = kv.transfer_summary(
            [r for r in at_length if r["arm"] == "fetch_busy"]
        )
        by_length[length] = row
    busy = [r for r in records if r["arm"] == "fetch_busy" and not r["error"]]
    return {
        "by_length": by_length,
        "fetches_finished_while_busy": sum(1 for r in busy if r["busy_after"]),
        "fetches_busy": len(busy),
        "errors": [r for r in records if r["error"]],
    }


def verdict(summary: dict, path: list[str]) -> str:
    if not any(row["transfer_busy"] for row in summary["by_length"].values()):
        return "FAIL: no busy transfer showed up in vLLM's metrics"
    if summary["errors"]:
        return f"FAIL: {len(summary['errors'])} measurements failed"
    if summary["fetches_finished_while_busy"] < summary["fetches_busy"]:
        return "UNCONFIRMED: some fetches outlasted the load prompt"
    if not path:
        return "UNCONFIRMED: transfers ran, but UCX's read table is not in the log"
    if not any("cuda_ipc" in row for row in path):
        return "FAIL: UCX did not select cuda_ipc for reads"
    return "PASS: every fetch finished while the producer was busy"


def print_table(summary: dict) -> None:
    print("tokens  recompute  fetch_idle  fetch_busy      wait |  busy_xfer_ms   GB/s")
    for length, row in summary["by_length"].items():
        cells = [f"{row[arm]['p50_s']:.3f}" if arm in row else "-" for arm in ARMS]
        moved = row["transfer_busy"]
        print(
            f"{length:>6}  {cells[0]:>9}  {cells[1]:>10}  {cells[2]:>10}  {cells[3]:>8} | "
            f"{moved.get('xfer_ms', 0):>12.2f}  {moved.get('gb_per_s', 0):>5.1f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="producer base URL")
    parser.add_argument("--consumer", required=True, help="consumer base URL")
    parser.add_argument("--producer-log", type=Path, required=True)
    parser.add_argument("--consumer-log", type=Path, required=True)
    parser.add_argument("--prefix-lengths", default=DEFAULT_PREFIX_LENGTHS)
    parser.add_argument("--load-length", type=int, default=64000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    servers = Servers(args.model, args.producer, args.consumer)
    lengths = tuple(int(length) for length in args.prefix_lengths.split(","))

    kv.wait_ready(servers.producer)
    kv.wait_ready(servers.consumer)
    records = measure(servers, lengths, args.repeats, args.seed, args.load_length)

    summary = summarize(records, lengths)
    path = kv.read_path(args.consumer_log.read_text(errors="replace"))
    result = verdict(summary, path)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "load_length": args.load_length,
                "read_path": path,
                "summary": summary,
                "records": records,
                "verdict": result,
            },
            indent=2,
        )
    )
    for log in (args.producer_log, args.consumer_log):
        shutil.copy(log, args.output.parent / log.name)

    print_table(summary)
    print(result, flush=True)
    return 0 if result.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
