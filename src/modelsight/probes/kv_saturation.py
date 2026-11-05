"""H3: how long a remnant can a busy producer recompute before it hurts?

Run as the bench of examples/kv-transfer/h3.yaml. A router that moves a request
off a busy A to B, and asks A to export a prefix A has partly or wholly lost,
makes A compute the remnant while A serves its own users. This sweeps:

  load     concurrency of A's own decode-heavy background traffic
  remnant  uncached tokens A has to compute in the prepare request

and records two costs per cell:

  router   the flow (prepare on A, pull on B, serve on B) against B
           recomputing the remnant itself
  users    p95 time per output token of A's background requests during the
           prepare, against the same load just before it

For each load it reports the shortest remnant where each cost crosses its
threshold. A's saturation at each load is measured, not assumed. Writes
results.json to the current directory.
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
from pathlib import Path

from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

DEFAULT_CONCURRENCY = "0,32,64,128,192"
DEFAULT_REMNANTS = "1000,2000,4000,8000,16000,32000"
SUFFIX_LENGTH = 512
SETTLE_SECONDS = 10.0
MIN_BASELINE_SECONDS = 2.0
SATURATION = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    # Renamed across vLLM versions; only one is exported.
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:gpu_cache_usage_perc": "kv_usage",
    "vllm:generation_tokens_total": "generated",
}


class Load:
    """Keeps `concurrency` streamed requests running on A, and times their tokens."""

    def __init__(
        self, servers: Servers, concurrency: int, prompt_len: int, output_len: int, seed: int
    ) -> None:
        self._servers = servers
        self._prompt_len = prompt_len
        self._output_len = output_len
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # (time the token arrived, gap since the previous token of its request)
        self.gaps: list[tuple[float, float]] = []
        # (time the request was sent, time to its first token)
        self.ttfts: list[tuple[float, float]] = []
        self.errors: list[str] = []
        self._threads = [
            threading.Thread(target=self._loop, args=(random.Random(seed + i),), daemon=True)
            for i in range(concurrency)
        ]

    def __enter__(self) -> Load:
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join()

    def _loop(self, rng: random.Random) -> None:
        while not self._stop.is_set():
            body = {
                "prompt": kv.fresh_prompt(rng, self._prompt_len),
                "max_tokens": self._output_len,
                "ignore_eos": True,
                "stream": True,
            }
            sent = last = time.perf_counter()
            first = True
            try:
                with kv.post(self._servers, self._servers.producer, body) as response:
                    for line in response:
                        if not line.startswith(b"data:") or b"[DONE]" in line:
                            continue
                        now = time.perf_counter()
                        with self._lock:
                            if first:
                                self.ttfts.append((sent, now - sent))
                            else:
                                self.gaps.append((now, now - last))
                        first, last = False, now
            except OSError as exc:
                with self._lock:
                    self.errors.append(repr(exc))
                self._stop.wait(1.0)

    def window(self, start: float, end: float) -> dict:
        """p95 time per output token, and p50 time to first token, in a window."""
        with self._lock:
            gaps = [gap for at, gap in self.gaps if start <= at <= end]
            ttfts = [ttft for sent, ttft in self.ttfts if start <= sent <= end]
        return {
            "tpot_p95_s": kv.percentile(gaps, 0.95) if gaps else None,
            "tokens": len(gaps),
            "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        }


def saturation(servers: Servers, seconds: float) -> dict:
    """A's running and waiting requests, KV usage and decode rate over `seconds`."""
    before = kv.metric_totals(servers.producer, SATURATION)
    time.sleep(seconds)
    after = kv.metric_totals(servers.producer, SATURATION)
    return {
        "running": after["running"],
        "waiting": after["waiting"],
        "kv_usage": after["kv_usage"],
        "tokens_per_s": (after["generated"] - before["generated"]) / seconds,
    }


def recompute(servers: Servers, prompt: list[int]) -> float:
    return kv.timed_stream(servers, servers.consumer, {"prompt": prompt})["ttft_s"]


def flow(servers: Servers, load: Load, remnant: list[int], suffix: list[int]) -> dict:
    """A computes the remnant for B under load, then B pulls it and serves."""
    started = time.perf_counter()
    params = kv.prefill(servers, remnant)
    ended = time.perf_counter()
    prepare = ended - started
    span = max(prepare, MIN_BASELINE_SECONDS)
    during = load.window(started, ended)
    before = load.window(started - span, started)

    pull_started = time.perf_counter()
    body = {"prompt": remnant, "max_tokens": 1, "stream": False, "kv_transfer_params": params}
    with kv.post(servers, servers.consumer, body) as response:
        response.read()
    pull = time.perf_counter() - pull_started
    serve = kv.timed_stream(servers, servers.consumer, {"prompt": remnant + suffix})["ttft_s"]
    return {
        "prepare_s": prepare,
        "pull_s": pull,
        "serve_ttft_s": serve,
        "flow_s": prepare + pull + serve,
        "tpot_p95_before_s": before["tpot_p95_s"],
        "tpot_p95_during_s": during["tpot_p95_s"],
        "tokens_during": during["tokens"],
    }


def measure(args: argparse.Namespace, servers: Servers) -> tuple[list[dict], dict]:
    concurrencies = tuple(int(c) for c in args.concurrency.split(","))
    remnants = tuple(int(r) for r in args.remnants.split(","))
    rng = random.Random(args.seed)
    records: list[dict] = []

    baseline: dict[int, float] = {}
    for length in remnants:
        ttfts = [
            recompute(servers, kv.fresh_prompt(rng, length + SUFFIX_LENGTH))
            for _ in range(args.repeats)
        ]
        baseline[length] = statistics.median(ttfts)
        print(f"recompute {length}: {baseline[length]:.3f}s", flush=True)

    levels: dict[int, dict] = {}
    for concurrency in concurrencies:
        load = Load(servers, concurrency, args.load_prompt_len, args.load_output_len, args.seed)
        with load:
            levels[concurrency] = saturation(servers, args.settle)
            print(f"load {concurrency}: {levels[concurrency]}", flush=True)
            order = [(length, repeat) for length in remnants for repeat in range(args.repeats)]
            rng.shuffle(order)
            for length, repeat in order:
                record = {
                    "concurrency": concurrency, "remnant": length,
                    "repeat": repeat, "error": None,
                }  # fmt: skip
                try:
                    remnant = kv.fresh_prompt(rng, length)
                    record.update(flow(servers, load, remnant, kv.fresh_prompt(rng, SUFFIX_LENGTH)))
                except (OSError, RuntimeError, ValueError) as exc:
                    record["error"] = repr(exc)
                records.append(record)
                outcome = record["error"] or f"flow={record['flow_s']:.3f}s"
                print(f"load {concurrency} remnant {length} #{repeat}: {outcome}", flush=True)
            levels[concurrency]["load_errors"] = len(load.errors)
    return records, {"recompute_s": baseline, "levels": levels}


def inflation(before: float | None, during: float | None) -> float | None:
    if not before or during is None:
        return None
    return during / before - 1


def summarize(records: list[dict], context: dict, tpot_limit: float) -> dict:
    cells: dict[int, dict[int, dict]] = {}
    for record in records:
        if record["error"]:
            continue
        cell = cells.setdefault(record["concurrency"], {}).setdefault(record["remnant"], [])
        cell.append(record)

    table: dict[int, dict] = {}
    for concurrency, by_remnant in cells.items():
        row: dict = {"remnants": {}, "router_loses_from": None, "users_hurt_from": None}
        for length in sorted(by_remnant):
            found = by_remnant[length]
            rises = [
                inflation(r["tpot_p95_before_s"], r["tpot_p95_during_s"]) for r in found
            ]
            rises = [rise for rise in rises if rise is not None]
            cell = {
                "prepare_s": statistics.median(r["prepare_s"] for r in found),
                "flow_s": statistics.median(r["flow_s"] for r in found),
                "recompute_s": context["recompute_s"][length],
                "tpot_rise": statistics.median(rises) if rises else None,
                "n": len(found),
            }
            row["remnants"][length] = cell
            if row["router_loses_from"] is None and cell["flow_s"] > cell["recompute_s"]:
                row["router_loses_from"] = length
            if (
                row["users_hurt_from"] is None
                and cell["tpot_rise"] is not None
                and cell["tpot_rise"] > tpot_limit
            ):
                row["users_hurt_from"] = length
        table[concurrency] = row
    return {
        "by_concurrency": table,
        "tpot_limit": tpot_limit,
        "errors": [r for r in records if r["error"]],
    }


def print_table(summary: dict, levels: dict) -> None:
    print("load  running  kv_use  tok/s | remnant  prepare_s  flow_s  recompute_s  tpot_rise")
    for concurrency, row in summary["by_concurrency"].items():
        level = levels[concurrency]
        head = (
            f"{concurrency:>4}  {level['running']:>7.0f}  {level['kv_usage']:>6.2f}  "
            f"{level['tokens_per_s']:>5.0f}"
        )
        for length, cell in row["remnants"].items():
            rise = cell["tpot_rise"]
            print(
                f"{head} | {length:>7}  {cell['prepare_s']:>9.3f}  {cell['flow_s']:>6.3f}  "
                f"{cell['recompute_s']:>11.3f}  {'-' if rise is None else f'{rise:+.0%}':>9}"
            )
            head = " " * len(head)
        print(
            f"      router loses from {row['router_loses_from'] or '-'} tokens, "
            f"users hurt from {row['users_hurt_from'] or '-'} tokens"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="A's base URL")
    parser.add_argument("--consumer", required=True, help="B's base URL")
    parser.add_argument("--producer-log", type=Path, required=True)
    parser.add_argument("--consumer-log", type=Path, required=True)
    parser.add_argument("--concurrency", default=DEFAULT_CONCURRENCY)
    parser.add_argument("--remnants", default=DEFAULT_REMNANTS)
    parser.add_argument("--load-prompt-len", type=int, default=1000)
    parser.add_argument("--load-output-len", type=int, default=256)
    parser.add_argument("--tpot-limit", type=float, default=0.2, help="p95 TPOT rise, 0.2 = +20%%")
    parser.add_argument("--settle", type=float, default=SETTLE_SECONDS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    servers = Servers(args.model, args.producer, args.consumer)
    kv.wait_ready(servers.producer)
    kv.wait_ready(servers.consumer)

    records, context = measure(args, servers)
    summary = summarize(records, context, args.tpot_limit)
    args.output.write_text(
        json.dumps(
            {"model": args.model, "seed": args.seed, "suffix_len": SUFFIX_LENGTH,
             "load_prompt_len": args.load_prompt_len, "load_output_len": args.load_output_len,
             **context, "summary": summary, "records": records},
            indent=2,
        )  # fmt: skip
    )
    for log in (args.producer_log, args.consumer_log):
        shutil.copy(log, args.output.parent / log.name)

    print_table(summary, context["levels"])
    if summary["errors"]:
        print(f"FAIL: {len(summary['errors'])} measurements failed", flush=True)
        return 1
    print("PASS: every cell measured", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
