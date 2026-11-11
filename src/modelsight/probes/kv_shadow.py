"""What would the router's shadow KV transfer planner do on real agent traffic?

Run as the bench of examples/kv-transfer/shadow_kv_transfer_planner.yaml. Two vLLM engines, A and B,
publish KV events; a vllm-router in front of them routes with kv_aware and runs
the transfer planner in shadow mode (it plans, it never moves KV or changes a
pick). Every level replays the same agent sessions, each from its first turn
to its last, with N of them in flight at once. Only N changes between levels,
so every level sends the same prompts. The probe reads what the router counted
over each level:

  outcomes   vllm_router_kv_transfer_plans_total{outcome}: planned, or why not
  estimates  the fetch and wait cost histograms and the saved-time histogram,
             as counts and quantiles of what this level added
  inputs     how many requests had token ids and a verified cache identity
  load       A's and B's running and waiting requests while the level ran

Caches are reset between levels, so every level starts cold. Writes
results.json to the current directory and exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from modelsight.probes import kv_peer_load as h5
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_transfer as kv

DEFAULT_LEVELS = "8,16,32,64"
PLANS = "vllm_router_kv_transfer_plans_total"
COST = "vllm_router_kv_transfer_estimated_cost_ms"
SAVED = "vllm_router_kv_transfer_estimated_saved_ms"
DECISIONS = "vllm_router_prefix_routing_decisions_total"
LOOKUPS = "vllm_router_kv_placement_lookups_total"
# Outcomes that say the router could not see the request well enough to plan.
# If they are most of the traffic, the other outcomes say little (plan D.3.1).
INPUT_FAILURES = ("missing_tokens", "unverified_identity", "unknown_destination_input")
MAX_INPUT_FAILURE_SHARE = 0.5
QUANTILES = (0.5, 0.9, 0.99)
# Plans and served requests can differ by the few requests in flight when a
# level's counters are read.
PLAN_SLACK = 5
PLAN_SLACK_SHARE = 0.02

_SAMPLE = re.compile(r"^([A-Za-z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

Series = dict[tuple[str, tuple[tuple[str, str], ...]], float]


# Router metrics ----------------------------------------------------------------


def parse_metrics(text: str) -> Series:
    """Prometheus text format as {(name, sorted labels): value}."""
    series: Series = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        key = (name, tuple(sorted(_LABEL.findall(labels or ""))))
        series[key] = float(value)
    return series


def scrape(url: str) -> Series:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        return parse_metrics(response.read().decode())


def change(before: Series, after: Series) -> Series:
    """What a level added. A series that first appeared during it started at 0."""
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def by_label(series: Series, name: str, label: str) -> dict[str, float]:
    """One counter's values summed per value of `label`, zeros dropped."""
    totals: dict[str, float] = {}
    for (metric, labels), value in series.items():
        if metric == name and value:
            key = dict(labels).get(label, "")
            totals[key] = totals.get(key, 0.0) + value
    return totals


def histogram(series: Series, name: str, **match: str) -> dict:
    """Count, mean and quantiles of one histogram's change over a level.

    Quantiles interpolate linearly inside a bucket, as Prometheus'
    histogram_quantile does, so they are as coarse as the buckets.
    """
    buckets: dict[float, float] = {}
    total = count = 0.0
    for (metric, labels), value in series.items():
        found = dict(labels)
        if any(found.get(k) != v for k, v in match.items()):
            continue
        if metric == f"{name}_bucket":
            le = found["le"]
            bound = math.inf if le == "+Inf" else float(le)
            buckets[bound] = buckets.get(bound, 0.0) + value
        elif metric == f"{name}_sum":
            total += value
        elif metric == f"{name}_count":
            count += value
    summary: dict = {"count": count, "mean": total / count if count else None}
    ordered = sorted(buckets.items())
    for q in QUANTILES:
        summary[f"p{round(q * 100)}"] = quantile(ordered, q)
    return summary


def quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus histogram_quantile over cumulative (upper bound, count) pairs."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    rank = q * buckets[-1][1]
    lower, below = 0.0, 0.0
    for bound, cumulative in buckets:
        if cumulative >= rank:
            if math.isinf(bound):
                return lower  # past the last finite bucket: its bound is all we know
            if cumulative == below:
                return bound
            return lower + (bound - lower) * (rank - below) / (cumulative - below)
        lower, below = bound, cumulative
    return lower


def shares(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


# Load ------------------------------------------------------------------------


def aiperf_command(args: argparse.Namespace, level: int, dataset: Path,
                   artifacts: Path, turns: int) -> list[str]:  # fmt: skip
    """H5's AIPerf run, sent to the router, replaying every session once.

    A level is a fixed amount of work, not a fixed time: AIPerf walks the
    sessions in file order and stops after `turns` requests, every turn of
    every session. Under a time limit a loaded level gets through fewer turns
    per session, so it would send shorter prompts than a light one.

    AIPerf polls <url>/metrics by default. The router forwards that GET to an
    engine like any other request, so every poll would count as a plan with
    missing_tokens; --no-server-metrics turns the polling off.
    """
    as_h5 = argparse.Namespace(**vars(args), producer=args.router, level_seconds=0)
    command = h5.aiperf_command(as_h5, level, dataset, artifacts)
    for flag in ("--benchmark-duration", "--benchmark-grace-period"):
        at = command.index(flag)
        del command[at : at + 2]
    command[command.index("--dataset-sampling-strategy") + 1] = "sequential"
    return [*command, "--request-count", str(turns), "--no-server-metrics"]


def session_turns(sessions: list[list[dict]], render, cap: int,
                  limit: int) -> list[list[str]]:  # fmt: skip
    """The user turns of the first `limit` sessions with two or more turns
    whose prompts fit in `cap` tokens. One turn leaves no prefix to share."""
    picked = []
    for messages in sessions:
        turns = h5.background_turns(render, messages, cap)
        if len(turns) >= 2:
            picked.append(turns)
        if len(picked) == limit:
            break
    return picked


def write_sessions(path: Path, sessions: list[list[str]], level: int) -> int:
    """JSONL of AIPerf multi_turn sessions; returns how many turns it holds."""
    with path.open("w") as f:
        for n, turns in enumerate(sessions):
            f.write(json.dumps({
                "session_id": f"shadow-{level}-{n}",
                "turns": [{"text": t} for t in turns],
            }) + "\n")  # fmt: skip
    return sum(len(turns) for turns in sessions)


def turn_coverage(records: Path, sessions: list[list[str]], level: int) -> dict:
    """How many sessions AIPerf sent every turn of, from its per-request records.

    AIPerf wraps to the first session when it reaches the end of the file, so
    a session it starts again would take requests from others; `repeated`
    counts the turns it sent more than once.
    """
    sent: dict[str, int] = {}
    if records.exists():
        for line in records.open():
            if line.strip():
                name = json.loads(line)["metadata"].get("conversation_id", "")
                sent[name] = sent.get(name, 0) + 1
    complete = partial = repeated = 0
    for n, turns in enumerate(sessions):
        count = sent.get(f"shadow-{level}-{n}", 0)
        complete += count == len(turns)
        partial += 0 < count < len(turns)
        repeated += max(0, count - len(turns))
    return {"complete": complete, "partial": partial,
            "missing": sum(sent.get(f"shadow-{level}-{n}", 0) == 0
                           for n in range(len(sessions))),
            "repeated": repeated}  # fmt: skip


def engine_load(sampler: h5.Sampler) -> dict:
    rows = list(sampler.samples)
    if not rows:
        return {"running": None, "waiting": None, "max_running": None, "samples": 0}
    return {
        "running": statistics.median(r[1] for r in rows),
        "waiting": statistics.median(r[2] for r in rows),
        "max_running": max(r[1] for r in rows),
        "max_waiting": max(r[2] for r in rows),
        "samples": len(rows),
    }


def wait_exit(load: h5.Load, timeout: float) -> bool:
    """Let AIPerf finish every session; True if it exited by itself."""
    deadline = time.monotonic() + timeout
    while load.alive() and time.monotonic() < deadline:
        time.sleep(1)
    return not load.alive()


# Levels ------------------------------------------------------------------------


def summarize(added: Series) -> dict:
    outcomes = by_label(added, PLANS, "outcome")
    planned = outcomes.get("planned", 0.0)
    return {
        "plans": sum(outcomes.values()),
        "outcomes": outcomes,
        "outcome_shares": shares(outcomes),
        "input_failure_share": sum(shares(outcomes).get(k, 0.0) for k in INPUT_FAILURES),
        "planned": planned,
        "fetch_ms": histogram(added, COST, path="fetch"),
        "wait_ms": histogram(added, COST, path="wait"),
        "saved_ms": histogram(added, SAVED),
        "routing_decisions": by_label(added, DECISIONS, "outcome"),
        "placement_lookups": by_label(added, LOOKUPS, "result"),
    }


def run_level(args: argparse.Namespace, level: int, sessions: list[list[str]],
              out_dir: Path) -> dict:  # fmt: skip
    dataset = out_dir / f"sessions-{level}.jsonl"
    turns = write_sessions(dataset, sessions, level)
    info: dict = {"level": level, "sessions": len(sessions), "turns": turns}
    if len(sessions) < level:
        print(f"warning: level {level} has only {len(sessions)} sessions, so fewer "
              "than that are ever in flight", flush=True)  # fmt: skip
    command = aiperf_command(args, level, dataset, out_dir / f"aiperf-{level}", turns)
    before = scrape(args.router_metrics)
    served = [kv.metric_totals(url, h4.REQUEST_METRICS)["served"] for url in args.workers]
    started = time.monotonic()
    with (
        h5.Load(command, out_dir / f"aiperf-{level}.log") as load,
        h5.Sampler(args.workers[0]) as a,
        h5.Sampler(args.workers[1]) as b,
    ):
        info["aiperf_finished"] = wait_exit(load, args.level_timeout)
    info["seconds"] = time.monotonic() - started
    info["drained"] = all(h5.wait_idle(url) for url in args.workers)
    time.sleep(kv.METRICS_WAIT_SECONDS)
    added = change(before, scrape(args.router_metrics))
    info["served"] = sum(
        kv.metric_totals(url, h4.REQUEST_METRICS)["served"] - start
        for url, start in zip(args.workers, served)
    )
    info["load"] = {"A": engine_load(a), "B": engine_load(b)}
    info["coverage"] = turn_coverage(
        out_dir / f"aiperf-{level}" / "profile_export.jsonl", sessions, level
    )
    info["reset"] = all(h4.reset_cache(url) for url in args.workers)
    return {"info": info, **summarize(added)}


def check(levels: dict[int, dict]) -> dict:
    planned = {n: s["plans"] for n, s in levels.items()}
    # Every request the engines served was planned once, and nothing else was:
    # anything the router forwards that is not a generation (a /metrics poll,
    # say) would otherwise land in the outcomes.
    served = {n: s["info"]["served"] for n, s in levels.items()}
    matched = all(
        abs(planned[n] - served[n]) <= max(PLAN_SLACK, PLAN_SLACK_SHARE * served[n])
        for n in levels
    )
    inputs = {n: round(s["input_failure_share"], 3) for n, s in levels.items()}
    return {
        "planner_ran": {"ok": all(planned.values()), "plans": planned},
        "inputs": {
            "ok": all(v <= MAX_INPUT_FAILURE_SHARE for v in inputs.values()),
            "failure_share": inputs,
        },
        "one_plan_per_request": {"ok": matched, "plans": planned, "served": served},
        # Every level sent every turn of every session, so all levels sent
        # the same prompts.
        "every_turn_sent": {
            "ok": all(s["info"]["coverage"]["complete"] == s["info"]["sessions"]
                      and not s["info"]["coverage"]["repeated"]
                      for s in levels.values()),
            "coverage": {n: s["info"]["coverage"] for n, s in levels.items()},
        },
        "reset": {"ok": all(s["info"]["reset"] for s in levels.values())},
    }


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def print_summary(levels: dict[int, dict], checks: dict) -> None:
    print("level  plans  planned  cost_rule  not_busy  no_peer  holds_longest  "
          "fetch p50/p90  wait p50/p90  saved p50/p90  A run/wait  B run/wait")  # fmt: skip
    for n, s in levels.items():
        o = s["outcomes"]
        loads = []
        for side in ("A", "B"):
            load = s["info"]["load"][side]
            loads.append("-" if load["running"] is None
                         else f"{load['running']:.0f}/{load['waiting']:.0f}")  # fmt: skip
        f, w, v = s["fetch_ms"], s["wait_ms"], s["saved_ms"]
        print(f"{n:>5}  {s['plans']:>5.0f}  {o.get('planned', 0):>7.0f}  "
              f"{o.get('cost_rule', 0):>9.0f}  {o.get('source_not_busy', 0):>8.0f}  "
              f"{o.get('no_peer', 0):>7.0f}  {o.get('picked_holds_longest_prefix', 0):>13.0f}  "
              f"{_ms(f['p50']):>6}/{_ms(f['p90']):<6}  {_ms(w['p50']):>5}/{_ms(w['p90']):<6}  "
              f"{_ms(v['p50']):>6}/{_ms(v['p90']):<6}  {loads[0]:>10}  {loads[1]:>10}")
    for name, result in checks.items():
        print(f"{'PASS' if result['ok'] else 'FAIL'}  {name}")


# Main -----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--router", required=True, help="the router's base URL")
    parser.add_argument("--router-metrics", required=True,
                        help="the router's Prometheus base URL")  # fmt: skip
    parser.add_argument("--worker", dest="workers", action="append", required=True,
                        help="an engine's base URL, A then B")  # fmt: skip
    parser.add_argument("--log", dest="logs", type=Path, action="append", default=[],
                        help="a server log to copy next to results.json")  # fmt: skip
    parser.add_argument("--levels", default=DEFAULT_LEVELS,
                        help="concurrent sessions through the router, one level after another")  # fmt: skip
    parser.add_argument("--traces", default=h4.DEFAULT_DATASET)
    parser.add_argument("--sessions", type=int, default=128,
                        help="agent sessions every level replays in full")  # fmt: skip
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--aiperf-command", default="aiperf")
    parser.add_argument("--ramp-seconds", type=int, default=20)
    parser.add_argument("--level-timeout", type=int, default=3600,
                        help="stop a level's AIPerf if it runs longer than this")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args(argv)
    if len(args.workers) != 2:
        parser.error("pass --worker twice: A, then B")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    levels = [n for n in h5.parse_levels(args.levels) if n > 0]
    for url in args.workers:
        kv.wait_ready(url)
    kv.wait_ready(args.router)
    if not all(h4.reset_cache(url) for url in args.workers):
        print("warning: could not reset the engines' caches before the first level",
              flush=True)  # fmt: skip

    render = h4.chat_renderer(args.model)
    # Some sessions have fewer than two turns under the cap and are dropped,
    # so draw more than needed.
    pool = h4.load_sessions(args.traces, args.sessions * 3, args.seed)
    sessions = session_turns(pool, render, args.max_prompt_tokens, args.sessions)
    if not sessions:
        print(f"FAIL: no session has two turns under {args.max_prompt_tokens} tokens",
              flush=True)  # fmt: skip
        return 1
    out_dir = args.output.parent
    results: dict[int, dict] = {}
    for level in levels:
        results[level] = run_level(args, level, sessions, out_dir)
        shown = {k: results[level][k] for k in ("plans", "outcomes")}
        print(json.dumps({"level": level, **shown}), flush=True)

    checks = check(results)
    args.output.write_text(json.dumps(
        {"model": args.model, "levels": levels, "sessions": len(sessions),
         "checks": checks, "results": results}, indent=2))  # fmt: skip
    for log in args.logs:
        if log.exists():
            shutil.copy(log, out_dir / log.name)

    print_summary(results, checks)
    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed:
        print(f"FAIL: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
