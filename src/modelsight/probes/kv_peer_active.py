"""End to end: the router picks B, plans a fetch from A, and B pulls A's prefix.

Run as the bench of examples/kv-transfer/peer_fetch_active.yaml. Two vLLM
engines, A and B, with peer lookup on, and two vllm-routers in front of them
that differ only in --kv-transfer-mode:

  shadow  plans and counts, sends requests unchanged (the baseline: B
          recomputes the prefix, or the request waits in A's queue)
  active  adds kv_transfer_params to the request it sends to B, so B asks A
          for the prefix by block hash, A pins it, and B pulls it over NIXL

The flow under test, per request: the router picks a destination, the planner
picks a source and checks the cost rule, the router sends the request to B
with A's side-channel address, B looks the prefix up on A, A replies, B pulls
the KV and computes the rest.

Stages, in order. A failing smoke check stops the run before any load.

  smoke     both engines report a valid peer_lookup record on
            /kv_event_sources (that is where the router learns A's address);
            a direct peer request for the smallest context hits, pulls over
            cuda_ipc and matches recompute; a burst through the active router
            ends in at least one `sent`
  verify    for every context length: B's greedy output after a direct peer
            pull matches its own recompute, A serves nothing, B pulls bytes.
            B's recompute TTFT per length is the baseline a fetch must beat.
  sweep     shared-context sessions arrive at a fixed rate (Poisson), each
            session's turns back to back, all through one router; the same
            sessions and arrival times run under shadow, then active

Load model: a few long shared contexts sit in A's cache before each arm. Every
session starts with one of them and adds short turns, so cache affinity keeps
sending sessions to A, A's queue grows with the rate, and B takes the overflow.

Writes results.json to the current directory and exits with status 1 if a
check fails.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from modelsight.probes import kv_peer_load as h5
from modelsight.probes import kv_peer_lookup as h4
from modelsight.probes import kv_shadow as shadow
from modelsight.probes import kv_transfer as kv
from modelsight.probes.kv_transfer import Servers

DEFAULT_CONTEXTS = "8000,16000,24000"
DEFAULT_RATES = "1,2,4,8"
ARMS = ("shadow", "active")
# Most tokens one user or assistant message adds to a session. Kept short so
# 16 running sessions' turns fit in A's cache next to the contexts.
TURN_TOKENS = 500
# Tokens each turn generates. Decode is what keeps a session on A long enough
# for sessions to pile up in A's queue.
OUTPUT_TOKENS = 128
SMOKE_BURST = 24
SMOKE_OUTPUT_TOKENS = 256
# Router counter: one per request the active planner granted a fetch, by what
# the router then did with it (sent, no_peer_endpoint, inflight_cap, ...).
REQUESTS = "vllm_router_kv_transfer_requests_total"
STEP_METRIC = "vllm:inter_token_latency_seconds"
STEP_METRICS = {f"{STEP_METRIC}_sum": "sum", f"{STEP_METRIC}_count": "count"}
# Lookups B logged that the router's `sent` count may differ by: requests in
# flight when the counters are read.
LOOKUP_SLACK = 2
MAX_ERROR_SHARE = 0.01
# Sessions' request ids are "pf<7 digits>x", so they can be found inside the
# ids vLLM builds from the X-Request-Id header.
_OUR_ID = re.compile(r"pf\d{7}x")
_TS = r"(\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})"
_HIT = re.compile(_TS + r".*?Peer lookup_pin hit for (\S+?): (\d+) tokens")
_MISS = re.compile(_TS + r".*?Peer lookup for (\S+?) missed: (\S+)")


# Prompts ----------------------------------------------------------------------


def text_tokenizer(model: str) -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_contexts(
    encode: Callable[[str], list[int]], sessions: list[list[dict]], lengths: list[int]
) -> dict[int, list[int]]:
    """One context per length, exactly that many tokens, from agent traces.

    Each starts with its own header, so no two contexts share a block and a
    shorter one is never a prefix of a longer one.
    """
    texts = [m["content"] for s in sessions for m in s if m["content"].strip()]
    if not texts:
        raise RuntimeError("the traces have no text to build contexts from")
    contexts = {}
    for n, length in enumerate(lengths):
        # The first character differs per context, so their first blocks do.
        ids = encode(f"{chr(ord('A') + n)}: context {length}. Repository notes follow.\n\n")
        at = n * 7
        while len(ids) < length:
            ids += encode(texts[at % len(texts)] + "\n\n")
            at += 1
        contexts[length] = ids[:length]
    return contexts


def session_turns(
    encode: Callable[[str], list[int]], messages: list[dict], turns: int,
    cap: int = TURN_TOKENS,
) -> list[tuple[list[int], list[int]]]:
    """(user, assistant) token pairs from a trace, each message capped."""
    pairs, user = [], None
    for m in messages:
        if m["role"] == "user":
            user = encode(f"\n\nUser: {m['content']}")[:cap]
        elif m["role"] == "assistant" and user is not None:
            reply = encode(f"\n\nAssistant: {m['content']}")[:cap]
            pairs.append((user, reply))
            user = None
        if len(pairs) == turns:
            break
    return pairs


def session_prompts(
    context: list[int], pairs: list[tuple[list[int], list[int]]], max_tokens: int,
    output_tokens: int = OUTPUT_TOKENS,
) -> list[list[int]]:
    """Turn t's prompt: the context, every earlier turn, and user message t.

    The trace's assistant replies stand in for the model's, so every arm sends
    exactly the same prompts. Turns that would not fit are dropped.
    """
    prompts, history = [], list(context)
    for user, reply in pairs:
        prompt = history + user
        if len(prompt) + output_tokens > max_tokens:
            break
        prompts.append(prompt)
        history = prompt + reply
    return prompts


def plan_sessions(args, encode, traces, contexts) -> list[dict]:
    """The sweep's sessions, the same for every rate and arm."""
    lengths = sorted(contexts)
    planned = []
    for n, messages in enumerate(traces):
        context = lengths[len(planned) % len(lengths)]
        prompts = session_prompts(
            contexts[context],
            session_turns(encode, messages, args.turns, args.turn_tokens),
            args.max_model_len, args.output_tokens,
        )  # fmt: skip
        if prompts:
            planned.append({"session": n, "context": context, "prompts": prompts})
        if len(planned) == args.sessions:
            break
    return planned


def arrivals(rate: float, count: int, seed: int) -> list[float]:
    """Poisson session start times in seconds, the same for both arms."""
    rng = random.Random(f"{seed}-{rate}")
    at, times = 0.0, []
    for _ in range(count):
        times.append(at)
        at += rng.expovariate(rate)
    return times


# Engines and logs -------------------------------------------------------------


def peer_records(url: str) -> dict[str, dict]:
    """The peer_lookup record of every DP rank on /kv_event_sources."""
    with urllib.request.urlopen(f"{url}/kv_event_sources", timeout=10) as response:
        sources = json.load(response)
    return {rank: entry.get("peer_lookup") or {} for rank, entry in sources.items()}


def record_problem(record: dict) -> str | None:
    """Why the router would drop this record, as it validates it, or None."""
    if not record.get("enabled"):
        return "not enabled"
    for key in ("engine_id", "side_channel_host", "connector",
                "connector_version", "layout_fingerprint"):  # fmt: skip
        if not str(record.get(key) or "").strip():
            return f"empty {key}"
    port = record.get("side_channel_port") or 0
    if not 0 < port < 65536:
        return "bad side_channel_port"
    if any(not record.get(k) for k in ("tp_size", "dcp_size", "pp_size")):
        return "zero size"
    return None


def address(record: dict) -> dict:
    """The kv_transfer_params fields the router derives from a record."""
    return {
        "remote_engine_id": record["engine_id"],
        "remote_host": record["side_channel_host"],
        "remote_port": record["side_channel_port"],
        "tp_size": record["tp_size"],
        "dcp_size": record["dcp_size"],
        "pp_size": record["pp_size"],
    }


def log_time(stamp: str, year: int) -> float:
    """vLLM's "%m-%d %H:%M:%S.mmm" log time as epoch seconds (local time)."""
    return datetime.strptime(f"{year}-{stamp}", "%Y-%m-%d %H:%M:%S.%f").timestamp()


def lookups(text: str, year: int) -> dict[str, dict]:
    """B's lookup results by our request id: hit tokens or the miss reason."""
    found: dict[str, dict] = {}
    for line in text.splitlines():
        if match := _HIT.search(line):
            stamp, request, tokens = match.groups()
            ours = _OUR_ID.search(request)
            if ours:
                found[ours.group()] = {"hit_tokens": int(tokens),
                                       "at": log_time(stamp, year)}  # fmt: skip
        elif match := _MISS.search(line):
            stamp, request, reason = match.groups()
            ours = _OUR_ID.search(request)
            if ours:
                found[ours.group()] = {"miss": reason, "at": log_time(stamp, year)}
    return found


def step_seconds(url: str, before: dict[str, float]) -> float | None:
    """Mean time between output tokens since `before`: a proxy for the length of
    the engine's steps, which a lookup waits on."""
    after = kv.metric_totals(url, STEP_METRICS)
    count = after["count"] - before["count"]
    return (after["sum"] - before["sum"]) / count if count > 0 else None


def reset_all(urls: list[str]) -> bool:
    return all(h5.wait_idle(url) for url in urls) and all(
        h4.reset_cache(url) for url in urls
    )


# Requests ---------------------------------------------------------------------


class Ids:
    """Request ids, unique across the run."""

    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()

    def take(self) -> str:
        with self._lock:
            self._next += 1
            return f"pf{self._next:07d}x"


def timed(servers: Servers, url: str, prompt: list[int], request_id: str,
          max_tokens: int = OUTPUT_TOKENS) -> dict:  # fmt: skip
    """TTFT of one streamed request, sent with our id."""
    sent_at = time.time()
    started = time.perf_counter()
    first = None
    body = {"prompt": prompt, "stream": True, "max_tokens": max_tokens,
            "ignore_eos": True}  # fmt: skip
    with kv.post(servers, url, body, {"X-Request-Id": request_id}) as response:
        for line in response:
            if first is None and line.startswith(b"data:") and b"[DONE]" not in line:
                first = time.perf_counter() - started
    if first is None:
        raise RuntimeError("the stream ended without a token")
    return {"sent_at": sent_at, "ttft_s": first,
            "total_s": time.perf_counter() - started}  # fmt: skip


def run_session(servers: Servers, url: str, session: dict, ids: Ids,
                out: list[dict], lock: threading.Lock,
                output_tokens: int = OUTPUT_TOKENS) -> None:  # fmt: skip
    for turn, prompt in enumerate(session["prompts"]):
        request_id = ids.take()
        record = {"id": request_id, "session": session["session"],
                  "context": session["context"], "turn": turn,
                  "prompt_tokens": len(prompt)}  # fmt: skip
        try:
            record.update(timed(servers, url, prompt, request_id, output_tokens))
        except (OSError, RuntimeError, ValueError) as exc:
            record["error"] = repr(exc)
        with lock:
            out.append(record)


# Smoke and verify -------------------------------------------------------------


def peer_case(args, servers: Servers, context: list[int], prompt: list[int],
              params: dict, ids: Ids) -> dict:  # fmt: skip
    """B's recompute and B's direct peer pull of one prompt whose context A holds."""
    urls = [servers.producer, servers.consumer]
    if not reset_all(urls):
        raise RuntimeError("an engine refused to reset its prefix cache")
    recompute = h4.greedy(servers, servers.consumer, {"prompt": prompt})
    if not reset_all(urls):
        raise RuntimeError("an engine refused to reset its prefix cache")
    kv.timed_stream(servers, servers.producer, {"prompt": context})
    time.sleep(kv.METRICS_WAIT_SECONDS)
    a_served = h4.served(servers.producer)
    a_log = h4.log_size(args.logs["A"])
    b_log = h4.log_size(args.logs["B"])
    before = kv.metric_totals(servers.consumer, kv.NIXL_METRICS)
    request_id = ids.take()
    body = {"prompt": prompt,
            "kv_transfer_params": {"do_remote_prefill": True, "peer_lookup": True,
                                   **params}}  # fmt: skip
    peer = h4.greedy(servers, servers.consumer, body, {"X-Request-Id": request_id})
    peer["transfer"] = kv.metrics_change(servers.consumer, before)
    time.sleep(kv.METRICS_WAIT_SECONDS)
    peer["a_served"] = h4.served(servers.producer) - a_served
    peer["lookups_logged"] = len(re.findall(
        args.lookup_pattern, h4.log_since(args.logs["A"], a_log), re.IGNORECASE))  # fmt: skip
    result = lookups(h4.log_since(args.logs["B"], b_log), time.localtime().tm_year)
    peer["lookup"] = result.get(request_id, {})
    expected = h4.expected_hit(context, prompt, args.block_size)
    ok = (
        peer["a_served"] == 0
        and peer["lookups_logged"] > 0
        and peer["transfer"]["bytes"] > 0
        and peer["lookup"].get("hit_tokens", 0) > 0
        and (h4.matches(peer, recompute) or h4.is_near_tie(peer, recompute))
    )
    return {"context_tokens": len(context), "prompt_tokens": len(prompt),
            "expected_hit_tokens": expected,
            "hit_tokens": peer["lookup"].get("hit_tokens", 0),
            "recompute": recompute, "peer": peer, "ok": ok}  # fmt: skip


def first_prompt(contexts: dict[int, list[int]], sessions: list[dict],
                 length: int) -> list[int]:  # fmt: skip
    """The first turn of a session on this context, or the context plus a
    short tail if no session uses it."""
    for s in sessions:
        if s["context"] == length:
            return s["prompts"][0]
    return contexts[length] + [11] * 32


def smoke_burst(args, servers: Servers, context: list[int], ids: Ids) -> dict:
    """Concurrent requests on one context through the active router, after A
    holds it: cache affinity sends the first to A, and once A is busy the
    router should pick B for some and send them with a fetch from A."""
    if not reset_all([servers.producer, servers.consumer]):
        raise RuntimeError("an engine refused to reset its prefix cache")
    kv.timed_stream(servers, servers.producer, {"prompt": context})
    time.sleep(kv.METRICS_WAIT_SECONDS)
    router = args.routers["active"]
    before = shadow.scrape(router["metrics"])
    b_log = h4.log_size(args.logs["B"])
    errors: list[str] = []

    def one(n: int) -> None:
        try:
            timed(servers, router["url"], context + [11 + n] * 64, ids.take(),
                  SMOKE_OUTPUT_TOKENS)  # fmt: skip
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=one, args=(n,)) for n in range(args.smoke_burst)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(kv.METRICS_WAIT_SECONDS)
    added = shadow.change(before, shadow.scrape(router["metrics"]))
    found = lookups(h4.log_since(args.logs["B"], b_log), time.localtime().tm_year)
    requests = shadow.by_label(added, REQUESTS, "outcome")
    return {"requests": requests, "plans": shadow.by_label(added, shadow.PLANS, "outcome"),
            "hits": sum("hit_tokens" in v for v in found.values()),
            "errors": errors,
            "ok": requests.get("sent", 0) > 0 and not errors}  # fmt: skip


def smoke(args, servers: Servers, contexts, sessions, ids: Ids) -> dict:
    result: dict = {"records": {}, "pool_tokens": {}}
    problems = []
    for side, url in (("A", servers.producer), ("B", servers.consumer)):
        try:
            records = peer_records(url)
        except (OSError, ValueError) as exc:
            records, problems = {}, [*problems, f"{side}: {exc!r}"]
        result["records"][side] = records
        if not records:
            problems.append(f"{side}: no rank reports peer_lookup")
        for rank, record in records.items():
            if (why := record_problem(record)) is not None:
                problems.append(f"{side} rank {rank}: {why}")
        try:
            result["pool_tokens"][side] = kv.pool_tokens(args.logs[side])
        except (OSError, RuntimeError):
            result["pool_tokens"][side] = None
    result["record_problems"] = problems
    if problems:
        result["ok"] = False
        return result
    pool = result["pool_tokens"].get("A")
    if pool and sum(contexts) > 0.8 * pool:
        print(f"warning: the contexts take {sum(contexts)} tokens of A's {pool}; "
              "sessions may evict them", flush=True)  # fmt: skip

    smallest = min(contexts)
    params = address(result["records"]["A"]["0"])
    case = peer_case(args, servers, contexts[smallest],
                     first_prompt(contexts, sessions, smallest), params, ids)  # fmt: skip
    path = kv.read_path(args.logs["B"].read_text(errors="replace"))
    result["direct_peer"] = case
    result["read_path"] = path
    result["cuda_ipc"] = any("cuda_ipc" in row for row in path)
    result["burst"] = smoke_burst(args, servers, contexts[smallest], ids)
    result["ok"] = case["ok"] and result["cuda_ipc"] and result["burst"]["ok"]
    return result


def verify(args, servers: Servers, contexts, sessions, params, ids: Ids) -> dict:
    return {
        length: peer_case(args, servers, contexts[length],
                          first_prompt(contexts, sessions, length), params, ids)
        for length in sorted(contexts)
    }  # fmt: skip


# Sweep ------------------------------------------------------------------------


def warm(servers: Servers, contexts: dict[int, list[int]]) -> None:
    """Put every context in A's cache, straight to A, not through a router."""
    for length in sorted(contexts):
        kv.timed_stream(servers, servers.producer, {"prompt": contexts[length]})


def run_arm(args, servers: Servers, arm: str, rate: float, sessions: list[dict],
            contexts, ids: Ids) -> dict:  # fmt: skip
    urls = [servers.producer, servers.consumer]
    info: dict = {"arm": arm, "rate": rate, "sessions": len(sessions)}
    info["reset"] = reset_all(urls)
    warm(servers, contexts)
    # Let the KV events of the warm-up reach the router's index.
    time.sleep(kv.METRICS_WAIT_SECONDS)
    router = args.routers[arm]
    before = shadow.scrape(router["metrics"])
    served = [h4.served(url) for url in urls]
    steps = [kv.metric_totals(url, STEP_METRICS) for url in urls]
    b_log = h4.log_size(args.logs["B"])
    records: list[dict] = []
    lock = threading.Lock()
    starts = arrivals(rate, len(sessions), args.seed)
    started = time.monotonic()
    with h5.Sampler(servers.producer) as a, h5.Sampler(servers.consumer) as b:
        threads = []
        for at, session in zip(starts, sessions):
            delay = started + at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            t = threading.Thread(target=run_session, daemon=True,
                                 args=(servers, router["url"], session, ids, records,
                                       lock, args.output_tokens))  # fmt: skip
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=args.arm_timeout)
    info["seconds"] = time.monotonic() - started
    info["drained"] = all(h5.wait_idle(url) for url in urls)
    time.sleep(kv.METRICS_WAIT_SECONDS)
    added = shadow.change(before, shadow.scrape(router["metrics"]))
    info["served"] = {side: h4.served(url) - start
                      for side, url, start in zip("AB", urls, served)}  # fmt: skip
    info["step_s"] = {side: step_seconds(url, before_steps)
                      for side, url, before_steps in zip("AB", urls, steps)}  # fmt: skip
    info["load"] = {"A": shadow.engine_load(a), "B": shadow.engine_load(b)}
    found = lookups(h4.log_since(args.logs["B"], b_log), time.localtime().tm_year)
    for record in records:
        result = found.get(record["id"])
        if result is None:
            continue
        if "hit_tokens" in result:
            record["hit_tokens"] = result["hit_tokens"]
        else:
            record["miss"] = result["miss"]
        if "sent_at" in record:
            # From the client's send to B's lookup result: router hop, B's
            # scheduling, and the wait for A's reply (A's fence and step).
            record["lookup_s"] = result["at"] - record["sent_at"]
    return {"info": info, "records": sorted(records, key=lambda r: r["id"]),
            "requests": shadow.by_label(added, REQUESTS, "outcome"),
            "plans": shadow.by_label(added, shadow.PLANS, "outcome")}  # fmt: skip


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p90": None}
    ordered = sorted(values)
    return {"n": len(ordered), "p50": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]}  # fmt: skip


def summarize_arm(result: dict, baseline: dict[int, float]) -> dict:
    """TTFT by context length, fetches, and fetches that lost to recompute."""
    good = [r for r in result["records"] if "error" not in r]
    by_context = {
        length: _quantiles([r["ttft_s"] for r in good if r["context"] == length])
        for length in sorted({r["context"] for r in good})
    }
    fetched = [r for r in good if "hit_tokens" in r]
    # A fetch lost when B would have answered sooner by recomputing: its TTFT
    # is above B's idle recompute TTFT for the same context's first turn.
    lost = [r for r in fetched
            if r["context"] in baseline and r["ttft_s"] > baseline[r["context"]]]  # fmt: skip
    misses: dict[str, int] = {}
    for r in good:
        if "miss" in r:
            misses[r["miss"]] = misses.get(r["miss"], 0) + 1
    return {
        "ttft_by_context": by_context,
        "fetches": len(fetched),
        "misses": misses,
        "lost_fetches": len(lost),
        "lookup_s": _quantiles([r["lookup_s"] for r in fetched if "lookup_s" in r]),
        "errors": len(result["records"]) - len(good),
        "requests": len(result["records"]),
    }


def check(smoke_result: dict, verified: dict, sweep: dict, leak_free: bool) -> dict:
    arms = [(rate, arm, r) for rate, by_arm in sweep.items() for arm, r in by_arm.items()]
    sent = {f"{rate}": r["requests"].get("sent", 0)
            for rate, arm, r in arms if arm == "active"}  # fmt: skip
    looked = {
        f"{rate}": r["summary"]["fetches"] + sum(r["summary"]["misses"].values())
        for rate, arm, r in arms if arm == "active"
    }  # fmt: skip
    shadow_lookups = sum(
        r["summary"]["fetches"] + sum(r["summary"]["misses"].values())
        for rate, arm, r in arms if arm == "shadow"
    )  # fmt: skip
    requests = sum(r["summary"]["requests"] for _, _, r in arms)
    errors = sum(r["summary"]["errors"] for _, _, r in arms)
    checks = {
        "smoke": {"ok": bool(smoke_result.get("ok"))},
        "verified": {
            "ok": bool(verified) and all(c["ok"] for c in verified.values()),
            "contexts": {n: c["ok"] for n, c in verified.items()},
        },
    }
    if not sweep:
        return {**checks, "no_leaks": {"ok": leak_free}}
    return {
        **checks,
        "active_sent": {"ok": sum(sent.values()) > 0, "sent": sent},
        # Every request the router sent with a fetch did a lookup on B.
        "lookups_match_sends": {
            "ok": all(abs(looked[k] - sent[k]) <= LOOKUP_SLACK for k in sent),
            "sent": sent,
            "lookups": looked,
        },
        "shadow_sends_nothing": {"ok": shadow_lookups == 0, "lookups": shadow_lookups},
        "errors": {
            "ok": requests > 0 and errors <= MAX_ERROR_SHARE * requests,
            "errors": errors,
            "requests": requests,
        },
        "no_leaks": {"ok": leak_free},
    }


def _s(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_summary(verified: dict, sweep: dict, checks: dict) -> None:
    print("context  recompute_s  peer_s  hit_tokens/expected")
    for length, case in verified.items():
        print(f"{length:>7}  {_s(case['recompute']['ttft_s']):>11}  "
              f"{_s(case['peer']['ttft_s']):>6}  "
              f"{case['hit_tokens']}/{case['expected_hit_tokens']}")  # fmt: skip
    print("rate    arm     A run/wait  B run/wait  A/B served  sent  fetches  lost  "
          "lookup p50  ttft p50 by context")  # fmt: skip
    for rate, by_arm in sweep.items():
        for arm, r in by_arm.items():
            info, summary = r["info"], r["summary"]
            loads = []
            for side in "AB":
                load = info["load"][side]
                loads.append("-" if load["running"] is None
                             else f"{load['running']:.0f}/{load['waiting']:.0f}")  # fmt: skip
            ttft = " ".join(f"{n}:{_s(q['p50'])}"
                            for n, q in summary["ttft_by_context"].items())  # fmt: skip
            print(f"{rate:<7} {arm:<7} {loads[0]:>10}  {loads[1]:>10}  "
                  f"{info['served']['A']:.0f}/{info['served']['B']:.0f}  "
                  f"{r['requests'].get('sent', 0):>4.0f}  {summary['fetches']:>7}  "
                  f"{summary['lost_fetches']:>4}  {_s(summary['lookup_s']['p50']):>10}  {ttft}")
    for name, result in checks.items():
        print(f"{'PASS' if result['ok'] else 'FAIL'}  {name}")


# Main -------------------------------------------------------------------------


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--producer", required=True, help="A's base URL")
    parser.add_argument("--consumer", required=True, help="B's base URL")
    parser.add_argument("--producer-log", type=Path, required=True)
    parser.add_argument("--consumer-log", type=Path, required=True)
    parser.add_argument("--router", action="append", default=[], required=True,
                        help="ARM=URL,METRICS_URL for the shadow and active routers")  # fmt: skip
    parser.add_argument("--log", dest="extra_logs", type=Path, action="append",
                        default=[], help="another log to copy next to results.json")  # fmt: skip
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS,
                        help="shared context lengths in tokens")  # fmt: skip
    parser.add_argument("--rates", default=DEFAULT_RATES,
                        help="session arrivals per second, one sweep step each")  # fmt: skip
    parser.add_argument("--sessions", type=int, default=240,
                        help="sessions per arm and rate")  # fmt: skip
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--turn-tokens", type=int, default=TURN_TOKENS,
                        help="most tokens one user or assistant message adds")  # fmt: skip
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS,
                        help="tokens each sweep turn generates")  # fmt: skip
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--traces", default=h4.DEFAULT_DATASET)
    parser.add_argument("--smoke-burst", type=int, default=SMOKE_BURST)
    parser.add_argument("--smoke-only", action="store_true",
                        help="stop after the smoke and verify stages")  # fmt: skip
    parser.add_argument("--arm-timeout", type=int, default=3600)
    parser.add_argument("--lookup-pattern", default=h4.LOOKUP_PATTERN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args(argv)
    args.routers = {}
    for entry in args.router:
        arm, _, urls = entry.partition("=")
        url, _, metrics = urls.partition(",")
        if arm not in ARMS or not url or not metrics:
            parser.error(f"--router {entry!r}: expected ARM=URL,METRICS_URL")
        args.routers[arm] = {"url": url, "metrics": metrics}
    if set(args.routers) != set(ARMS):
        parser.error("pass --router for both shadow and active")
    args.logs = {"A": args.producer_log, "B": args.consumer_log}
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    servers = Servers(args.model, args.producer, args.consumer)
    for url in (servers.producer, servers.consumer,
                *(r["url"] for r in args.routers.values())):  # fmt: skip
        kv.wait_ready(url)

    encode = text_tokenizer(args.model)
    lengths = sorted({int(x) for x in args.contexts.split(",") if x.strip()})
    traces = h4.load_sessions(args.traces, max(args.sessions * 2, 16), args.seed)
    contexts = build_contexts(encode, traces, lengths)
    sessions = plan_sessions(args, encode, traces, contexts)
    if not sessions:
        print("FAIL: no session fits the context window", flush=True)
        return 1
    ids = Ids()
    out_dir = args.output.parent

    def finish(payload: dict, checks: dict) -> None:
        args.output.write_text(json.dumps({"model": args.model, "checks": checks,
                                           **payload}, indent=2))  # fmt: skip
        for log in (*args.logs.values(), *args.extra_logs):
            if log.exists():
                shutil.copy(log, out_dir / log.name)

    smoke_result = smoke(args, servers, contexts, sessions, ids)
    print(json.dumps({"smoke_ok": smoke_result["ok"],
                      "problems": smoke_result.get("record_problems"),
                      "cuda_ipc": smoke_result.get("cuda_ipc"),
                      "burst": smoke_result.get("burst")}), flush=True)  # fmt: skip
    if not smoke_result["ok"]:
        finish({"smoke": smoke_result}, {"smoke": {"ok": False}})
        print("FAIL: smoke check; no load was run", flush=True)
        return 1

    params = address(smoke_result["records"]["A"]["0"])
    verified = verify(args, servers, contexts, sessions, params, ids)
    baseline = {n: c["recompute"]["ttft_s"] for n, c in verified.items()}
    if args.smoke_only or not all(c["ok"] for c in verified.values()):
        checks = check(smoke_result, verified, {}, h4.reset_cache(servers.producer))
        finish({"smoke": smoke_result, "verified": verified}, checks)
        print_summary(verified, {}, checks)
        ok = all(c["ok"] for c in verified.values())
        print("PASS (smoke only)" if ok else "FAIL: verify", flush=True)
        return 0 if ok else 1

    sweep: dict[float, dict[str, dict]] = {}
    for rate in parse_floats(args.rates):
        sweep[rate] = {}
        for arm in ARMS:
            result = run_arm(args, servers, arm, rate, sessions, contexts, ids)
            result["summary"] = summarize_arm(result, baseline)
            sweep[rate][arm] = result
            print(json.dumps({"rate": rate, "arm": arm, "requests": result["requests"],
                              "fetches": result["summary"]["fetches"],
                              "load": result["info"]["load"]}), flush=True)  # fmt: skip

    leak_free = reset_all([servers.producer, servers.consumer])
    checks = check(smoke_result, verified, sweep, leak_free)
    finish({"contexts": lengths, "sessions": len(sessions),
            "smoke": smoke_result, "verified": verified,
            "sweep": {str(k): v for k, v in sweep.items()}}, checks)  # fmt: skip
    print_summary(verified, sweep, checks)
    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed:
        print(f"FAIL: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
