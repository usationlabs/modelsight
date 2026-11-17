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
  verify    for a sample of contexts: B's greedy output after a direct peer
            pull matches its own recompute, A serves nothing, B pulls bytes.
            Idle B's recompute and pull TTFT per length.
  sweep     shared-context sessions arrive at a fixed rate (Poisson), each
            session's turns back to back, all through one router; the same
            sessions and arrival times run under shadow, then active. Each
            fetched request is compared with the same request in the shadow
            arm, recomputed under the same load.

Load model: many distinct shared contexts (one per few sessions, lengths from
a palette) sit in A's cache before each arm. Every session starts with its
context and adds short turns, so cache affinity keeps sending sessions to A,
A's queue grows with the rate, and B takes the overflow. B cannot hold every
context, so a B-picked request's first turn on its context is a fetch the
planner can want; with only a handful of contexts B collects them all at once
and no fetch is ever possible.

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

# Context lengths, cycled across the contexts: 8k to 30k. Three of each of the
# five lengths is about 294k tokens, under A's ~398k-token pool. A 30k context
# leaves room for two turns in the 32k window; the rest keep all four.
DEFAULT_CONTEXTS = "8000,14000,20000,26000,30000"
# One distinct context per this many sessions. With only a few long contexts B
# collects each on its first request and then holds every prefix the planner
# could fetch, so at most a handful of fetches per arm were ever possible (the
# first run sent 0 to 2 of ~960 per rate). Many contexts keep the asymmetry the
# feature needs: A holds them all, B only the few it has already touched, so
# every context's first request on B is a fetch opportunity.
DEFAULT_CONTEXT_COUNT = 15
# At rate 1 A has no queue, so the router never plans a fetch.
DEFAULT_RATES = "1.25,1.5,1.75,2"
DEFAULT_VERIFY_CONTEXTS = 4
# Fewest fetches the active arm must send at each rate for the comparison with
# recompute to mean anything.
MIN_SENDS = 5
ARMS = ("shadow", "active")
# Most tokens one user or assistant message adds to a session. Kept short so
# 16 running sessions' turns fit in A's cache next to the contexts.
TURN_TOKENS = 500
# Tokens each turn generates. Decode is what keeps a session on A long enough
# for sessions to pile up in A's queue.
OUTPUT_TOKENS = 128
SMOKE_BURST = 32
SMOKE_OUTPUT_TOKENS = 256
# New tokens each burst request adds after the shared context. kv_aware prices
# a worker as (uncached prompt tokens + its in-flight prefill tokens) x (its
# active requests + 1), so with a 64-token tail A, which holds the context,
# stays cheapest until it has over 100 requests in flight. With 1024 new
# tokens, B wins once A has about 8.
SMOKE_TAIL_TOKENS = 1024
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
_HIT = re.compile(_TS + r".*?Peer lookup_pin hit for (\S+?): (\d+) tokens"
                  r"(?: under pin (\S+))?")  # fmt: skip
# A's line when it pins blocks for a lookup. B logs the same pin id on its hit.
_PIN = re.compile(_TS + r".*?Peer lookup_pin pinned \d+ of \d+ blocks as (\S+)")
_MISS = re.compile(_TS + r".*?Peer lookup for (\S+?) missed: (\S+)")


# Prompts ----------------------------------------------------------------------


def text_tokenizer(model: str) -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_contexts(
    encode: Callable[[str], list[int]], sessions: list[list[dict]], count: int,
    palette: list[int],
) -> dict[int, list[int]]:  # fmt: skip
    """`count` contexts, lengths cycling through `palette`, from agent traces.

    Keys are indices 0..count-1. Each starts with a header naming its index, so
    no two contexts share a block and a shorter one is never a prefix of a
    longer one, even when two contexts have the same length.
    """
    texts = [m["content"] for s in sessions for m in s if m["content"].strip()]
    if not texts:
        raise RuntimeError("the traces have no text to build contexts from")
    contexts = {}
    for n in range(count):
        length = palette[n % len(palette)]
        ids = encode(f"#{n}: context {length}. Repository notes follow.\n\n")
        at = n * 7
        while len(ids) < length:
            ids += encode(texts[at % len(texts)] + "\n\n")
            at += 1
        contexts[n] = ids[:length]
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
    keys = sorted(contexts)
    planned = []
    for n, messages in enumerate(traces):
        context = keys[len(planned) % len(keys)]
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
            stamp, request, tokens, pin = match.groups()
            ours = _OUR_ID.search(request)
            if ours:
                found[ours.group()] = {"hit_tokens": int(tokens), "pin": pin,
                                       "at": log_time(stamp, year)}  # fmt: skip
        elif match := _MISS.search(line):
            stamp, request, reason = match.groups()
            ours = _OUR_ID.search(request)
            if ours:
                found[ours.group()] = {"miss": reason, "at": log_time(stamp, year)}
    return found


def pins(text: str, year: int) -> dict[str, float]:
    """When A pinned blocks for each lookup, by pin id. B sends the lookup when
    its scheduler first takes the request, and A pins within milliseconds, so
    this is also about when B started on the request."""
    found = {}
    for line in text.splitlines():
        if match := _PIN.search(line):
            stamp, pin = match.groups()
            found[pin] = log_time(stamp, year)
    return found


def step_seconds(url: str, before: dict[str, float]) -> float | None:
    """Mean time between output tokens since `before`: a proxy for the length of
    the engine's steps, which a lookup waits on."""
    after = kv.metric_totals(url, STEP_METRICS)
    count = after["count"] - before["count"]
    return (after["sum"] - before["sum"]) / count if count > 0 else None


def reset_all(servers: Servers) -> bool:
    urls = [servers.producer, servers.consumer]
    if not all(h5.wait_idle(url) for url in urls) or not all(
        h4.reset_cache(url) for url in urls
    ):
        return False
    # vLLM publishes KV events only from a scheduler step, and an idle engine
    # takes none, so a reset's AllBlocksCleared sits unpublished until the next
    # request reaches that engine. One tiny one per engine flushes it: without
    # this, the router's index keeps crediting a cleared engine with blocks it
    # cached before the reset, and no plan ever wants a fetch from it. Eight
    # tokens complete no 64-token block, so the tickle itself stores nothing.
    tickle = {"prompt": kv.fresh_prompt(random.Random("reset-tickle"), 8),
              "max_tokens": 1}  # fmt: skip
    try:
        for url in urls:
            with kv.post(servers, url, tickle):
                pass
    except (OSError, ValueError):
        return False
    return True


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
    """TTFT of one streamed request, sent with our id, and how many of its
    prompt tokens the engine found in its own cache (vLLM reports this with
    --enable-prompt-tokens-details)."""
    sent_at = time.time()
    started = time.perf_counter()
    first, cached = None, None
    body = {"prompt": prompt, "stream": True, "max_tokens": max_tokens,
            "ignore_eos": True, "stream_options": {"include_usage": True}}  # fmt: skip
    with kv.post(servers, url, body, {"X-Request-Id": request_id}) as response:
        for line in response:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            if first is None:
                first = time.perf_counter() - started
            if b'"cached_tokens"' in line:
                usage = json.loads(line[5:]).get("usage") or {}
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if first is None:
        raise RuntimeError("the stream ended without a token")
    result = {"sent_at": sent_at, "ttft_s": first,
              "total_s": time.perf_counter() - started}  # fmt: skip
    if cached is not None:
        result["cached_tokens"] = cached
    return result


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
    if not reset_all(servers):
        raise RuntimeError("an engine refused to reset its prefix cache")
    recompute = h4.greedy(servers, servers.consumer, {"prompt": prompt})
    if not reset_all(servers):
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
                 index: int) -> list[int]:  # fmt: skip
    """The first turn of a session on this context, or the context plus a
    short tail if no session uses it."""
    for s in sessions:
        if s["context"] == index:
            return s["prompts"][0]
    return contexts[index] + [11] * 32


def smoke_burst(args, servers: Servers, context: list[int], ids: Ids) -> dict:
    """Concurrent requests on one context through the active router, after A
    holds it: cache affinity sends the first to A, and once A is busy the
    router should pick B for some and send them with a fetch from A."""
    if not reset_all(servers):
        raise RuntimeError("an engine refused to reset its prefix cache")
    kv.timed_stream(servers, servers.producer, {"prompt": context})
    time.sleep(kv.METRICS_WAIT_SECONDS)
    router = args.routers["active"]
    before = shadow.scrape(router["metrics"])
    b_log = h4.log_size(args.logs["B"])
    errors: list[str] = []

    def one(n: int) -> None:
        try:
            tail = kv.fresh_prompt(random.Random(f"smoke-{n}"), SMOKE_TAIL_TOKENS)
            timed(servers, router["url"], context + tail, ids.take(),
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
            # How the router saw the burst: prefix_cost vs sticky_hit or
            # missing_tokens, and whether its index found the context (hit)
            # or not (miss, index_unavailable).
            "decisions": shadow.by_label(added, shadow.DECISIONS, "outcome"),
            "placement_lookups": shadow.by_label(added, shadow.LOOKUPS, "result"),
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
    total = sum(len(c) for c in contexts.values())
    if pool and total > 0.8 * pool:
        print(f"warning: the contexts take {total} tokens of A's {pool}; "
              "sessions may evict them", flush=True)  # fmt: skip

    smallest = min(contexts, key=lambda n: len(contexts[n]))
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


def sampled_contexts(contexts: dict[int, list[int]], want: int) -> list[int]:
    """Up to `want` context indices, evenly spaced across the length range."""
    ordered = sorted(contexts, key=lambda n: len(contexts[n]))
    if want >= len(ordered):
        return ordered
    step = (len(ordered) - 1) / (want - 1)
    return sorted({ordered[round(j * step)] for j in range(want)})


def verify(args, servers: Servers, contexts, sessions, params, ids: Ids) -> dict:
    return {
        n: peer_case(args, servers, contexts[n],
                     first_prompt(contexts, sessions, n), params, ids)
        for n in sampled_contexts(contexts, args.verify_contexts)
    }  # fmt: skip


# Sweep ------------------------------------------------------------------------


def warm(servers: Servers, contexts: dict[int, list[int]]) -> None:
    """Put every context in A's cache, straight to A, not through a router."""
    for n in sorted(contexts):
        kv.timed_stream(servers, servers.producer, {"prompt": contexts[n]})


def run_arm(args, servers: Servers, arm: str, rate: float, sessions: list[dict],
            contexts, ids: Ids) -> dict:  # fmt: skip
    urls = [servers.producer, servers.consumer]
    info: dict = {"arm": arm, "rate": rate, "sessions": len(sessions)}
    info["reset"] = reset_all(servers)
    warm(servers, contexts)
    # Let the KV events of the warm-up reach the router's index.
    time.sleep(kv.METRICS_WAIT_SECONDS)
    router = args.routers[arm]
    before = shadow.scrape(router["metrics"])
    served = [h4.served(url) for url in urls]
    steps = [kv.metric_totals(url, STEP_METRICS) for url in urls]
    a_log = h4.log_size(args.logs["A"])
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
    year = time.localtime().tm_year
    found = lookups(h4.log_since(args.logs["B"], b_log), year)
    pinned = pins(h4.log_since(args.logs["A"], a_log), year)
    for record in records:
        result = found.get(record["id"])
        if result is None:
            continue
        if "hit_tokens" not in result:
            record["miss"] = result["miss"]
            continue
        record["hit_tokens"] = result["hit_tokens"]
        if "sent_at" in record:
            record.update(fetch_times(record, result, pinned.get(result["pin"])))
    return {"info": info, "records": sorted(records, key=lambda r: r["id"]),
            "requests": shadow.by_label(added, REQUESTS, "outcome"),
            "plans": shadow.by_label(added, shadow.PLANS, "outcome")}  # fmt: skip


def fetch_times(record: dict, hit: dict, pinned_at: float | None) -> dict:
    """Where a fetched request's time to first token went.

    queue_s: send to A's pin. The router hop and B's queue: B sends the lookup
      only when its scheduler first takes the request.
    reply_s: A's pin to B's hit. A's reply and B noticing it on its next step.
    pull_s: B's hit to the first token. The NIXL pull and computing the rest.
    """
    first_token = record["sent_at"] + record["ttft_s"]
    times = {"pull_s": first_token - hit["at"]}
    if pinned_at is not None:
        times["queue_s"] = pinned_at - record["sent_at"]
        times["reply_s"] = hit["at"] - pinned_at
    return times


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p90": None}
    ordered = sorted(values)
    return {"n": len(ordered), "p50": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]}  # fmt: skip


def summarize_arm(result: dict, length_of: dict[int, int]) -> dict:
    """TTFT by context length, fetches and lookup misses."""
    good = [r for r in result["records"] if "error" not in r]
    by_context = {
        length: _quantiles([r["ttft_s"] for r in good
                            if length_of.get(r["context"]) == length])
        for length in sorted({length_of.get(r["context"]) for r in good} - {None})
    }  # fmt: skip
    fetched = [r for r in good if "hit_tokens" in r]
    misses: dict[str, int] = {}
    for r in good:
        if "miss" in r:
            misses[r["miss"]] = misses.get(r["miss"], 0) + 1
    return {
        "ttft_by_context": by_context,
        "fetches": len(fetched),
        "misses": misses,
        "errors": len(result["records"]) - len(good),
        "requests": len(result["records"]),
    }


def compare(by_arm: dict[str, dict], verified: dict, length_of: dict[int, int]) -> dict:
    """Pull against recompute, at one rate.

    Each fetched request in the active arm is paired with the same session's
    same turn in the shadow arm, which arrived at the same time under the same
    load and was recomputed. A fetch lost if its pair got its first token
    sooner. The idle verify numbers show the pull and compute alone.
    """
    twins = {(r["session"], r["turn"]): r
             for r in by_arm["shadow"]["records"] if "error" not in r}  # fmt: skip
    shadow_ttft = {key: r["ttft_s"] for key, r in twins.items()}
    fetched = [r for r in by_arm["active"]["records"]
               if "hit_tokens" in r and "error" not in r]  # fmt: skip
    pairs = [(r, shadow_ttft[(r["session"], r["turn"])]) for r in fetched
             if (r["session"], r["turn"]) in shadow_ttft]  # fmt: skip
    idle = {c["context_tokens"]: c["recompute"]["ttft_s"] for c in verified.values()}
    pulled = [(r, idle[length_of[r["context"]]]) for r in fetched
              if "pull_s" in r and length_of.get(r["context"]) in idle]  # fmt: skip

    def times(key: str) -> dict:
        return _quantiles([r[key] for r in fetched if key in r])

    return {
        "fetches": len(fetched),
        "paired": len(pairs),
        "ttft_pulled": _quantiles([r["ttft_s"] for r, _ in pairs]),
        "ttft_recomputed": _quantiles([s for _, s in pairs]),
        "saved_s": _quantiles([s - r["ttft_s"] for r, s in pairs]),
        "won": sum(r["ttft_s"] < s for r, s in pairs),
        "lost": sum(r["ttft_s"] >= s for r, s in pairs),
        # Losses whose shadow twin found the prefix in its engine's own cache:
        # the shadow run happened to route it to an engine that held the
        # context, so the twin was a cache hit, not a recompute.
        "lost_to_cache_hit": sum(
            r["ttft_s"] >= s
            and twins[(r["session"], r["turn"])].get("cached_tokens", 0) >= r["hit_tokens"]
            for r, s in pairs),
        "queue_s": times("queue_s"),
        "reply_s": times("reply_s"),
        "pull_s": times("pull_s"),
        # Hit to first token against idle B recomputing the same length.
        "pull_beats_idle_recompute": sum(r["pull_s"] < t for r, t in pulled),
        "pulled_with_times": len(pulled),
    }


def check(smoke_result: dict, verified: dict, sweep: dict, leak_free: bool,
          min_sends: int = MIN_SENDS) -> dict:  # fmt: skip
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
        # Each rate's active arm must send at least `min_sends` fetches: a
        # single send proves the plumbing but says nothing about the load regime.
        "active_sent": {"ok": all(v >= min_sends for v in sent.values()),
                        "sent": sent, "min_sends": min_sends},
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


def print_summary(verified: dict, sweep: dict, comparison: dict, checks: dict) -> None:
    print("idle B, first turn of a context: recompute vs peer pull")
    print("context  recompute_s  peer_s  hit_tokens/expected")
    for _, case in sorted(verified.items()):
        print(f"{case['context_tokens']:>7}  {_s(case['recompute']['ttft_s']):>11}  "
              f"{_s(case['peer']['ttft_s']):>6}  "
              f"{case['hit_tokens']}/{case['expected_hit_tokens']}")  # fmt: skip
    print("rate    arm     A run/wait  B run/wait  A/B served  sent  fetches  "
          "ttft p50 by context")  # fmt: skip
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
                  f"{r['requests'].get('sent', 0):>4.0f}  {summary['fetches']:>7}  {ttft}")
    if comparison:
        print("under load: each fetched request vs the same request recomputed in "
              "the shadow arm (p50 seconds)")  # fmt: skip
        print("rate    paired  pulled_ttft  recomputed_ttft  saved  won/lost  "
              "lost_to_hit  B_queue  A_reply  pull+compute  beats_idle_recompute")  # fmt: skip
    for rate, c in comparison.items():
        print(f"{rate:<7} {c['paired']:>6}  {_s(c['ttft_pulled']['p50']):>11}  "
              f"{_s(c['ttft_recomputed']['p50']):>15}  {_s(c['saved_s']['p50']):>5}  "
              f"{c['won']:>4}/{c['lost']:<4} {c['lost_to_cache_hit']:>11}  "
              f"{_s(c['queue_s']['p50']):>7}  "
              f"{_s(c['reply_s']['p50']):>7}  {_s(c['pull_s']['p50']):>12}  "
              f"{c['pull_beats_idle_recompute']}/{c['pulled_with_times']}")  # fmt: skip
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
                        help="context lengths in tokens, cycled across --context-count")  # fmt: skip
    parser.add_argument("--context-count", type=int, default=DEFAULT_CONTEXT_COUNT,
                        help="distinct shared contexts; sessions share each a few at a time")  # fmt: skip
    parser.add_argument("--verify-contexts", type=int, default=DEFAULT_VERIFY_CONTEXTS,
                        help="how many contexts the verify stage checks directly")  # fmt: skip
    parser.add_argument("--min-sends", type=int, default=MIN_SENDS,
                        help="each active arm must send at least this many fetches")  # fmt: skip
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
    palette = sorted({int(x) for x in args.contexts.split(",") if x.strip()})
    traces = h4.load_sessions(args.traces, max(args.sessions * 2, 16), args.seed)
    contexts = build_contexts(encode, traces, args.context_count, palette)
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
    length_of = {n: len(c) for n, c in contexts.items()}
    if args.smoke_only or not all(c["ok"] for c in verified.values()):
        checks = check(smoke_result, verified, {}, h4.reset_cache(servers.producer),
                       args.min_sends)  # fmt: skip
        finish({"smoke": smoke_result, "verified": verified}, checks)
        print_summary(verified, {}, {}, checks)
        ok = all(c["ok"] for c in verified.values())
        print("PASS (smoke only)" if ok else "FAIL: verify", flush=True)
        return 0 if ok else 1

    sweep: dict[float, dict[str, dict]] = {}
    comparison: dict[float, dict] = {}
    for rate in parse_floats(args.rates):
        sweep[rate] = {}
        for arm in ARMS:
            result = run_arm(args, servers, arm, rate, sessions, contexts, ids)
            result["summary"] = summarize_arm(result, length_of)
            sweep[rate][arm] = result
            print(json.dumps({"rate": rate, "arm": arm, "requests": result["requests"],
                              "fetches": result["summary"]["fetches"],
                              "load": result["info"]["load"]}), flush=True)  # fmt: skip
        comparison[rate] = compare(sweep[rate], verified, length_of)

    leak_free = reset_all(servers)
    checks = check(smoke_result, verified, sweep, leak_free, args.min_sends)
    finish({"context_count": len(contexts), "context_palette": palette,
            "sessions": len(sessions),
            "smoke": smoke_result, "verified": verified,
            "sweep": {str(k): v for k, v in sweep.items()},
            "comparison": {str(k): v for k, v in comparison.items()}}, checks)  # fmt: skip
    print_summary(verified, sweep, comparison, checks)
    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed:
        print(f"FAIL: {failed}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
