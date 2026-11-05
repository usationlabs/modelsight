"""H3: can stock vLLM reuse a peer's cached prefix without recomputing it?

Run as the bench of examples/kv-transfer/h3.yaml. A router that sends a request
to B, but wants the prefix that A has cached, uses three requests with stock
vLLM's NixlConnector:

  prepare  A gets the prefix with do_remote_decode, holds the blocks, returns ids
  pull     B gets the prefix with A's params and do_remote_prefill, and caches it
  serve    B gets the real prompt (prefix + suffix) and hits its own cache

Problem 1: A computes whatever it no longer has, so a stale index costs A a
prefill. The flow arms differ only in what A has cached before prepare:

  hit       A has the whole prefix
  partial   A has the first half
  miss      A has none of it, as after an eviction the router has not seen
  recompute B computes prefix + suffix itself, the baseline

Problem 2: B can only pull blocks for its whole prompt. After a prepare of the
prefix, B is sent one request with A's params:

  b_longer   prefix + suffix: B pulls A's prefix and computes the suffix
  b_shorter  the first half of the prefix: B pulls a subset of A's blocks

Each is checked against the output A gives for the same prompt. These run last,
because they may fail the request or stop B. Writes results.json to the current
directory and exits with status 1 unless both problems are confirmed.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

DEFAULT_PREFIX_LENGTHS = "4000,16000,32000"
SUFFIX_LENGTH = 512
FLOW_ARMS = ("hit", "partial", "miss", "recompute")
SUBSET_ARMS = ("b_longer", "b_shorter")
HITS = {"vllm:prefix_cache_hits_total": "hits"}
# A miss counts as a recompute when A's prepare takes this many times a hit's.
MISS_FACTOR = 2.0
# A cached prefix is "most of it" above this fraction.
HIT_FRACTION = 0.9


def complete(servers: Servers, url: str, body: dict) -> dict:
    """A short greedy completion, with its time, output and cached prompt tokens."""
    started = time.perf_counter()
    with kv.post(servers, url, {"max_tokens": kv.OUTPUT_TOKENS, "stream": False, **body}) as r:
        reply = json.load(r)
    details = (reply.get("usage") or {}).get("prompt_tokens_details") or {}
    return {
        "seconds": time.perf_counter() - started,
        "text": reply["choices"][0]["text"],
        "cached_tokens": details.get("cached_tokens"),
        "kv_transfer_params": reply.get("kv_transfer_params"),
    }


def remote_blocks(params: dict | None) -> int | None:
    """How many blocks A handed out, summed over its KV cache groups."""
    ids = (params or {}).get("remote_block_ids")
    if ids is None:
        return None
    if ids and isinstance(ids[0], list):
        return sum(len(group) for group in ids)
    return len(ids)


def prepare(servers: Servers, prefix: list[int]) -> dict:
    """Ask A to hold the prefix for B, and see how much of it A had to compute."""
    before = kv.metric_totals(servers.producer, HITS)["hits"]
    started = time.perf_counter()
    params = kv.prefill(servers, prefix)
    seconds = time.perf_counter() - started
    hits = kv.metric_totals(servers.producer, HITS)["hits"] - before
    return {
        "params": params,
        "prepare_s": seconds,
        "a_hit_fraction": hits / len(prefix),
        "a_blocks": remote_blocks(params),
    }


def flow(servers: Servers, prefix: list[int], suffix: list[int], cached: int) -> dict:
    """Cache `cached` tokens of the prefix on A, then run the three requests."""
    if cached:
        complete(servers, servers.producer, {"prompt": prefix[:cached], "max_tokens": 1})
    step = prepare(servers, prefix)
    params = step.pop("params")
    started = time.perf_counter()
    complete(
        servers,
        servers.consumer,
        {"prompt": prefix, "max_tokens": 1, "kv_transfer_params": params},
    )
    pull = time.perf_counter() - started
    served = kv.timed_stream(servers, servers.consumer, {"prompt": prefix + suffix})
    return {
        **step,
        "pull_s": pull,
        "ttft_s": served["ttft_s"],
        "flow_s": step["prepare_s"] + pull + served["ttft_s"],
    }


def recompute(servers: Servers, prefix: list[int], suffix: list[int]) -> dict:
    served = kv.timed_stream(servers, servers.consumer, {"prompt": prefix + suffix})
    return {"ttft_s": served["ttft_s"], "flow_s": served["ttft_s"]}


def subset(servers: Servers, prefix: list[int], prompt: list[int]) -> dict:
    """Send B `prompt` with A's params for `prefix`, and check B's output."""
    step = prepare(servers, prefix)
    params = step.pop("params")
    before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
    got = complete(servers, servers.consumer, {"prompt": prompt, "kv_transfer_params": params})
    moved = kv.metrics_change(servers.consumer, before)
    expected = complete(servers, servers.producer, {"prompt": prompt})
    return {
        **step,
        "b_s": got["seconds"],
        "b_cached_tokens": got["cached_tokens"],
        "transfer": moved,
        "output_matches": got["text"] == expected["text"],
        "b_text": got["text"],
        "expected_text": expected["text"],
    }


def run(record: dict, measure) -> dict:
    try:
        record.update(measure())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        record["error"] = repr(exc)
    kv.show(record)
    return record


def measure(
    servers: Servers, lengths: tuple[int, ...], repeats: int, seed: int, save
) -> list[dict]:
    """The flow arms in a shuffled order, then the subset arms."""
    rng = random.Random(seed)
    records: list[dict] = []
    cached = {"hit": 1.0, "partial": 0.5, "miss": 0.0}

    def arm(name: str, length: int):
        prefix = kv.fresh_prompt(rng, length)
        suffix = kv.fresh_prompt(rng, SUFFIX_LENGTH)
        if name == "recompute":
            return lambda: recompute(servers, prefix, suffix)
        if name == "b_longer":
            return lambda: subset(servers, prefix, prefix + suffix)
        if name == "b_shorter":
            return lambda: subset(servers, prefix, prefix[: length // 2])
        return lambda: flow(servers, prefix, suffix, int(length * cached[name]))

    for _ in range(2):
        run({"arm": "warmup", "prefix_len": lengths[0], "repeat": 0, "error": None},
            arm("hit", lengths[0]))  # fmt: skip

    order = [(a, n, r) for a in FLOW_ARMS for n in lengths for r in range(repeats)]
    rng.shuffle(order)
    for name, length, repeat in order:
        record = {"arm": name, "prefix_len": length, "repeat": repeat, "error": None}
        records.append(run(record, arm(name, length)))
    save(records)

    for name in SUBSET_ARMS:
        for length in lengths:
            record = {"arm": name, "prefix_len": length, "repeat": 0, "error": None}
            records.append(run(record, arm(name, length)))
            save(records)
    return records


def p50(records: list[dict], field: str) -> float | None:
    values = [r[field] for r in records if r.get(field) is not None]
    return statistics.median(values) if values else None


def summarize(records: list[dict], lengths: tuple[int, ...]) -> dict:
    by_length: dict[int, dict] = {}
    for length in lengths:
        row: dict = {}
        for name in FLOW_ARMS:
            ok = [
                r for r in records
                if r["arm"] == name and r["prefix_len"] == length and not r["error"]
            ]  # fmt: skip
            if ok:
                row[name] = {
                    field: p50(ok, field)
                    for field in ("prepare_s", "pull_s", "ttft_s", "flow_s",
                                  "a_hit_fraction", "a_blocks")  # fmt: skip
                    if p50(ok, field) is not None
                } | {"n": len(ok)}
        for name in SUBSET_ARMS:
            found = [r for r in records if r["arm"] == name and r["prefix_len"] == length]
            if found:
                row[name] = {
                    key: found[0].get(key)
                    for key in ("error", "output_matches", "b_cached_tokens", "a_blocks")
                } | {"bytes": found[0].get("transfer", {}).get("bytes")}
        by_length[length] = row
    return {
        "by_length": by_length,
        "flow_errors": [r for r in records if r["arm"] in FLOW_ARMS and r["error"]],
    }


def problem1(summary: dict) -> str:
    """A recomputes what it lacks: a miss costs A more, and it hands out every block."""
    rows = list(summary["by_length"].values())
    if summary["flow_errors"] or not rows:
        return "UNCONFIRMED: some flow measurements failed"
    longest = rows[-1]
    if not all(arm in longest for arm in ("hit", "partial", "miss")):
        return "UNCONFIRMED: an arm has no result at the longest prefix"
    hit, partial, miss = longest["hit"], longest["partial"], longest["miss"]
    if hit["a_hit_fraction"] < HIT_FRACTION:
        return "UNCONFIRMED: the hit arm did not hit A's cache"
    if miss["prepare_s"] < MISS_FACTOR * hit["prepare_s"]:
        return "REFUTED: a miss on A costs no more than a hit"
    if not hit["a_blocks"] or partial.get("a_blocks") != hit["a_blocks"]:
        return "REFUTED: A handed out only part of the prefix on a partial hit"
    return (
        f"CONFIRMED: on a miss A computed the prefix "
        f"({miss['prepare_s']:.3f}s against {hit['prepare_s']:.3f}s on a hit), "
        "and on a partial hit it computed the rest instead of returning a subset"
    )


def problem2(summary: dict) -> str:
    """B pulls only for its whole prompt: anything else fails or gives wrong output."""
    cases = [
        row[name] for row in summary["by_length"].values() for name in SUBSET_ARMS if name in row
    ]
    if not cases:
        return "UNCONFIRMED: no subset arm ran"
    if any(not case["error"] and case["output_matches"] for case in cases):
        return "REFUTED: B served a subset pull with correct output"
    return "CONFIRMED: every subset pull failed or gave output that differs from A's"


def print_table(summary: dict) -> None:
    print("tokens  arm        prepare_s  pull_s  ttft_s  flow_s  A_hit  A_blocks")
    for length, row in summary["by_length"].items():
        for name in FLOW_ARMS:
            if name not in row:
                continue
            cells = [row[name].get(f) for f in ("prepare_s", "pull_s", "ttft_s", "flow_s")]
            text = "  ".join(f"{c:>7.3f}" if c is not None else "      -" for c in cells)
            hit = row[name].get("a_hit_fraction")
            hit_cell = "-" if hit is None else f"{hit:.2f}"
            blocks = row[name].get("a_blocks")
            blocks_cell = "-" if blocks is None else f"{blocks:.0f}"
            print(f"{length:>6}  {name:<9}  {text}  {hit_cell:>5}  {blocks_cell:>8}")
        for name in SUBSET_ARMS:
            if name in row:
                case = row[name]
                outcome = case["error"] or f"output_matches={case['output_matches']}"
                print(f"{length:>6}  {name:<9}  {outcome}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="A's base URL")
    parser.add_argument("--consumer", required=True, help="B's base URL")
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

    def save(records: list[dict], verdicts: dict | None = None) -> dict:
        summary = summarize(records, lengths)
        args.output.write_text(
            json.dumps(
                {"model": args.model, "seed": args.seed, "suffix_len": SUFFIX_LENGTH,
                 "summary": summary, "records": records, "verdicts": verdicts},
                indent=2,
            )  # fmt: skip
        )
        return summary

    kv.wait_ready(servers.producer)
    kv.wait_ready(servers.consumer)
    records = measure(servers, lengths, args.repeats, args.seed, save)

    summary = summarize(records, lengths)
    verdicts = {"problem1": problem1(summary), "problem2": problem2(summary)}
    save(records, verdicts)
    for log in (args.producer_log, args.consumer_log):
        shutil.copy(log, args.output.parent / log.name)

    print_table(summary)
    for name, result in verdicts.items():
        print(f"{name}: {result}", flush=True)
    return 0 if all(v.startswith("CONFIRMED") for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
